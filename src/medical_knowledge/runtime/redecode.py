"""Targeted re-decode: the audio decides a MEDIUM proposal, never a full Whisper re-run.

A MEDIUM correction is plausible but not safe. For a few of them the engine asks the speech recogniser to listen AGAIN to just that stretch of audio
(a window of a few seconds around the span), once with the proposed term as a hotword. The proposal is

* CONFIRMED   the second listening produces the term (or something that sounds the same) where the first heard something else -> promoted to HIGH;
* REJECTED    the second listening still produces what was heard first -> the audio says the original; the proposal is dropped;
* UNDECIDED   anything else -> stays MEDIUM (not applied).

The decoder is INJECTED (``decode(start, end, hotwords) -> text``), so this module needs no model, no GPU and no network, and the whole feature is OFF unless a
decoder is passed. Work is bounded: ``max_windows`` windows and ``max_seconds`` of audio per lecture, never the whole file. A hotword can pull a recogniser towards
a term that was not said; ``tools/redecode_probe.py`` measures how often that happens before the feature may be switched on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from ...medical.corrector import SegmentView
from ..normalize.text import compact, normalize
from .engine import V4Correction, V4Result
from .retrieve import sound_similarity

CONFIRMED, REJECTED, UNDECIDED = "confirmed", "rejected", "undecided"
Decoder = Callable[[float, float, Sequence[str]], str]


@dataclass
class RedecodeBudget:
    max_windows: int = 30
    max_seconds: float = 180.0
    pad: float = 1.0                 # seconds of audio on each side of the estimated position of the span
    max_window: float = 8.0          # one window is never longer than this


@dataclass
class Window:
    correction: V4Correction
    start: float
    end: float
    hotwords: list[str]

    @property
    def seconds(self) -> float:
        return self.end - self.start


@dataclass
class Outcome:
    window: Window
    verdict: str
    decoded: str = ""
    why: str = ""

    def to_dict(self) -> dict:
        c = self.window.correction
        return {"original": c.original, "replacement": c.replacement, "confidence": round(c.confidence, 3), "segment": c.segment, "window": [round(self.window.start, 2), round(self.window.end, 2)],
                "hotwords": self.window.hotwords, "verdict": self.verdict, "decoded": self.decoded, "why": self.why}


@dataclass
class RedecodeReport:
    outcomes: list[Outcome] = field(default_factory=list)
    skipped_by_budget: int = 0
    seconds_used: float = 0.0

    def count(self, verdict: str) -> int:
        return sum(1 for o in self.outcomes if o.verdict == verdict)

    def promoted(self) -> list[V4Correction]:
        return [o.window.correction for o in self.outcomes if o.verdict == CONFIRMED]

    def to_dict(self) -> dict:
        return {"windows": len(self.outcomes), "confirmed": self.count(CONFIRMED), "rejected": self.count(REJECTED), "undecided": self.count(UNDECIDED),
                "skipped_by_budget": self.skipped_by_budget, "audio_seconds": round(self.seconds_used, 1), "outcomes": [o.to_dict() for o in self.outcomes]}


def window_for(correction: V4Correction, view: SegmentView, budget: RedecodeBudget) -> Window:
    """The stretch of audio around a span. Word times are not carried by the segment view, so the position is estimated from the character offset
    inside the segment (segments are a few seconds long); the padding covers the estimate's error."""
    length = max(1, len(view.text))
    duration = max(0.0, view.end - view.start)
    first = view.start + duration * correction.start / length
    last = view.start + duration * correction.end / length
    start, end = max(0.0, first - budget.pad), last + budget.pad
    if end - start > budget.max_window:
        mid = (first + last) / 2
        start, end = max(0.0, mid - budget.max_window / 2), mid + budget.max_window / 2
    return Window(correction, start, end, [correction.replacement])


def _grams(text: str, n_max: int = 6) -> list[str]:
    words = normalize(text).split()
    return [" ".join(words[i:i + n]) for n in range(1, n_max + 1) for i in range(0, len(words) - n + 1)]


def judge(correction: V4Correction, decoded: str, sound_threshold: float = 0.93) -> tuple[str, str]:
    """(verdict, reason) for one proposal given what the second listening produced."""
    target, heard = compact(correction.replacement), compact(correction.original)
    if not decoded.strip():
        return UNDECIDED, "the second listening produced no text"
    flat = compact(decoded)
    if target and target in flat and (heard not in flat or heard in target):
        return CONFIRMED, "the second listening wrote the term"
    wanted = normalize(correction.replacement)
    best = max((sound_similarity(g, wanted) for g in _grams(decoded) if abs(len(g.split()) - len(wanted.split())) <= 1), default=0.0)
    if best >= sound_threshold and heard not in flat:
        return CONFIRMED, f"the second listening wrote something that sounds the same ({best:.2f})"
    if heard and heard in flat:
        return REJECTED, "the second listening wrote what was heard before"
    return UNDECIDED, "the second listening wrote something else"


class TargetedRedecoder:
    def __init__(self, decode: Decoder, budget: Optional[RedecodeBudget] = None) -> None:
        self.decode = decode
        self.budget = budget or RedecodeBudget()

    def run(self, result: V4Result, views: Sequence[SegmentView]) -> RedecodeReport:
        report = RedecodeReport()
        for correction in sorted(result.uncertain, key=lambda c: -c.confidence):
            window = window_for(correction, views[correction.segment], self.budget)
            if len(report.outcomes) >= self.budget.max_windows or report.seconds_used + window.seconds > self.budget.max_seconds:
                report.skipped_by_budget += 1
                continue
            try:
                decoded = self.decode(window.start, window.end, window.hotwords) or ""
            except Exception as exc:  # noqa: BLE001 - a failed second listening leaves the proposal MEDIUM
                report.outcomes.append(Outcome(window, UNDECIDED, "", f"decoder failed: {type(exc).__name__}"))
                continue
            report.seconds_used += window.seconds
            verdict, why = judge(correction, decoded)
            report.outcomes.append(Outcome(window, verdict, decoded, why))
        return report


def whisper_window_decoder(transcriber, audio_path, profile: dict, decoding: dict, vad_parameters: dict, language: Optional[str] = "en") -> Decoder:
    """A decoder that listens to a window of ``audio_path`` with the project's own Transcriber (loaded once by the caller). Imports are lazy: nothing here is
    needed unless the feature is switched on."""
    from pathlib import Path

    from ...model_manager import DecodeRequest

    def decode(start: float, end: float, hotwords: Sequence[str]) -> str:
        request = DecodeRequest(audio_path=Path(audio_path), language=language, profile=profile, decoding=decoding, vad_parameters=vad_parameters,
                                hotwords=list(hotwords), start_offset=start, max_seconds=max(0.5, end - start))
        result = transcriber.transcribe(request)
        return re.sub(r"\s+", " ", " ".join(s.text for s in result.segments)).strip()

    return decode
