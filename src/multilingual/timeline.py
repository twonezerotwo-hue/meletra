"""Auto Detect over time: from language identification of windows to a plan of language runs.

Whisper decodes with ONE language per 30 s window. A recording that changes language needs that language to be chosen per stretch of audio, and chosen carefully: the detector is
often wrong on accented speech (it spreads its probability over several languages) and nearly useless on singing over music. So the plan is built like this:

1. every window of the recording gets the detector's probabilities (``WindowEvidence``);
2. a Viterbi smoothing picks ONE language per window. A switch costs evidence, so a single unsure window never splits a lecture, while a window the detector is sure about does;
3. consecutive windows of the same language become a ``LanguageRun`` whose edges are moved to the nearest pause;
4. runs the detector was not sure about are marked ``unsure`` and verified by decoding (``verify.py``), within a fixed budget;
5. stretches that are loud but that the voice-activity filter calls non-speech (singing, speech over music) are marked ``vad_off`` so they are decoded instead of discarded.

Nothing here invents a probability: ``confidence`` is the mean probability the detector itself gave to the run's language, or ``None``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Sequence

SUPPORTED_SPOKEN = ("en", "tr", "de", "hu", "ko", "zh", "hi", "nl", "fr", "es")
EPS = 0.02                       # a language the detector gave (almost) no probability still has this much (log-odds are never infinite)
SWITCH_COST = 2.0                # nats paid for every change of language: a one-window island needs about two strong windows' worth of evidence
STRONG = 0.80                    # a window is strong evidence at this probability
SURE_UNSUPPORTED = 0.90          # an unsupported language is only accepted without checking when the detector is this sure
UNSURE = 0.50                    # a run whose language the detector gave less than this on average is verified by decoding
MIN_RUN_SECONDS = 4.0            # shorter runs are merged into the neighbour that shares their language, or the longer neighbour
CANDIDATE_MIN_P = 0.12
#: a language the detector names that the program does not offer, mapped to the offered language it is almost always mistaken for. Hindi is detected as Urdu (the two are spoken
#: almost identically; the program offers Hindi, and the text is checked by script afterwards: Devanagari = Hindi).
DETECTOR_ALIASES = {"ur": "hi"}


def fold_aliases(probs: dict[str, float]) -> dict[str, float]:
    """The detector's probabilities with aliased languages added to the language the program offers."""
    out: dict[str, float] = {}
    for language, p in probs.items():
        target = DETECTOR_ALIASES.get(language, language)
        out[target] = min(out.get(target, 0.0) + float(p), 1.0)
    return out


@dataclass
class WindowEvidence:
    start: float
    end: float
    probs: dict[str, float] = field(default_factory=dict)      # the detector's own numbers for its top languages
    speech_seconds: float = 0.0                                  # seconds of this window the voice-activity filter keeps
    rms: float = 0.0                                             # loudness of the window (0..1), to tell music from silence
    own_start: Optional[float] = None                            # the stretch this window is responsible for (windows overlap; their owned stretches do not) - set by assign_spans
    own_end: Optional[float] = None

    @property
    def top(self) -> tuple[Optional[str], float]:
        if not self.probs:
            return None, 0.0
        language = max(self.probs, key=self.probs.get)
        return language, float(self.probs[language])

    @property
    def seconds(self) -> float:
        return max(self.end - self.start, 0.0)


@dataclass
class LanguageRun:
    start: float
    end: float
    language: str
    confidence: Optional[float] = None
    status: str = "confident"                                   # confident | smoothed | unsure | verified
    candidates: list[str] = field(default_factory=list)         # plausible alternatives (for an unsure run)
    windows: int = 0
    vad: str = "on"                                              # on = the voice-activity filter may cut silence | off = decode the whole stretch (singing, speech over music)
    source: str = "detector"                                     # detector | smoothing | verified | prior

    @property
    def seconds(self) -> float:
        return max(self.end - self.start, 0.0)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["start"], data["end"] = round(self.start, 2), round(self.end, 2)
        data["confidence"] = None if self.confidence is None else round(self.confidence, 3)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LanguageRun":
        names = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in names})


@dataclass
class LanguagePlan:
    runs: list[LanguageRun] = field(default_factory=list)
    windows: list[dict[str, Any]] = field(default_factory=list)     # the evidence the plan is built from (start, end, top, p, speech seconds) - small, for the record
    hop: float = 0.0
    mode: str = "single"                                              # single | multilingual | music
    languages: list[str] = field(default_factory=list)                # languages of the runs that were decided, most seconds first
    notes: list[str] = field(default_factory=list)

    @property
    def multilingual(self) -> bool:
        return len({r.language for r in self.runs}) > 1

    def language_at(self, t: float) -> Optional[str]:
        for run in self.runs:
            if run.start - 1e-6 <= t < run.end:
                return run.language
        return self.runs[-1].language if self.runs and t >= self.runs[-1].end else (self.runs[0].language if self.runs else None)

    def to_dict(self) -> dict[str, Any]:
        return {"runs": [r.to_dict() for r in self.runs], "windows": self.windows, "hop": self.hop, "mode": self.mode, "languages": self.languages, "notes": self.notes}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LanguagePlan":
        return cls(runs=[LanguageRun.from_dict(r) for r in data.get("runs", [])], windows=list(data.get("windows", [])), hop=float(data.get("hop", 0.0)),
                   mode=str(data.get("mode", "single")), languages=list(data.get("languages", [])), notes=list(data.get("notes", [])))


# ====================================================================================================================== spans
def assign_spans(windows: Sequence[WindowEvidence]) -> None:
    """Windows overlap (a 30 s window every 10 s). Each window is made responsible for the middle of the span around its centre, up to the middle between it and its neighbours,
    so the stretches the windows own do not overlap and a language change is placed between windows, not at the edge of one."""
    ordered = sorted(windows, key=lambda w: w.start)
    for i, w in enumerate(ordered):
        centre = (w.start + w.end) / 2.0
        before = (ordered[i - 1].start + ordered[i - 1].end) / 2.0 if i > 0 else None
        after = (ordered[i + 1].start + ordered[i + 1].end) / 2.0 if i + 1 < len(ordered) else None
        w.own_start = w.start if before is None else max(w.start, (before + centre) / 2.0)
        w.own_end = w.end if after is None else min(w.end, (centre + after) / 2.0)
        if w.own_end <= w.own_start:
            w.own_start, w.own_end = w.start, w.end


# ====================================================================================================================== smoothing
def candidate_languages(windows: Sequence[WindowEvidence], limit: int = 6) -> list[str]:
    """Languages worth considering: those the detector gave real probability somewhere, strongest overall first."""
    mass: dict[str, float] = {}
    for w in windows:
        for language, p in w.probs.items():
            if p >= CANDIDATE_MIN_P:
                mass[language] = mass.get(language, 0.0) + p
    return [language for language, _m in sorted(mass.items(), key=lambda item: -item[1])][:limit]


def smooth_languages(windows: Sequence[WindowEvidence], candidates: Optional[Sequence[str]] = None, switch_cost: float = SWITCH_COST) -> list[Optional[str]]:
    """One language per window (``None`` for a window without any detector output), by Viterbi: each window's log-probability minus the cost of every switch."""
    if not windows:
        return []
    states = list(candidates) if candidates is not None else candidate_languages(windows)
    if not states:
        return [None] * len(windows)

    def emission(w: WindowEvidence, language: str) -> float:
        if not w.probs:
            return 0.0
        top_p = max(w.probs.values())
        weight = min(1.0, max(0.25, (top_p - 0.2) / 0.4))          # a window whose best guess is weak says little: it votes with less weight
        return weight * math.log(max(float(w.probs.get(language, 0.0)), EPS))

    best = {s: emission(windows[0], s) for s in states}
    back: list[dict[str, str]] = []
    for w in windows[1:]:
        new_best: dict[str, float] = {}
        pointers: dict[str, str] = {}
        for s in states:
            stay = best[s]
            switch_from, switch = max(((o, best[o] - switch_cost) for o in states if o != s), key=lambda item: item[1], default=(s, -math.inf))
            if stay >= switch:
                new_best[s], pointers[s] = stay + emission(w, s), s
            else:
                new_best[s], pointers[s] = switch + emission(w, s), switch_from
        best = new_best
        back.append(pointers)
    state = max(best, key=best.get)
    labels = [state]
    for pointers in reversed(back):
        state = pointers[state]
        labels.append(state)
    labels.reverse()
    return [None if not w.probs else label for w, label in zip(windows, labels)]


def runs_from_windows(windows: Sequence[WindowEvidence], labels: Sequence[Optional[str]]) -> list[LanguageRun]:
    """Consecutive windows with the same language become one run (windows without output join the run they sit in)."""
    runs: list[LanguageRun] = []
    group: list[tuple[WindowEvidence, str]] = []
    for w, label in zip(windows, labels):
        if label is None:
            if group:
                group.append((w, group[-1][1]))
            continue
        if group and group[-1][1] != label:
            runs.append(_run(group))
            group = []
        group.append((w, label))
    if group:
        runs.append(_run(group))
    return _merge_short(runs)


def _run(group: Sequence[tuple[WindowEvidence, str]]) -> LanguageRun:
    language = group[0][1]
    own = [w.probs[language] for w, _l in group if language in w.probs]
    top_agrees = sum(1 for w, _l in group if w.top[0] == language)
    confidence = sum(own) / len(own) if own else None
    other = {}
    for w, _l in group:
        for lang, p in w.probs.items():
            if lang != language and p >= CANDIDATE_MIN_P:
                other[lang] = other.get(lang, 0.0) + p
    status = "confident" if confidence is not None and confidence >= UNSURE and top_agrees >= max(1, len(group) // 2) else "unsure"
    if status == "confident" and language not in SUPPORTED_SPOKEN and (confidence or 0.0) < SURE_UNSUPPORTED:
        status = "unsure"                                              # a language the program does not offer, named without conviction: on music this is the detector being lost, so it is checked
    if status == "confident" and top_agrees < len(group):
        status = "smoothed"
    return LanguageRun(start=_own(group[0][0], 0), end=_own(group[-1][0], 1), language=language, confidence=confidence, status=status,
                       candidates=[l for l, _m in sorted(other.items(), key=lambda item: -item[1])][:3], windows=len(group),
                       source="detector" if status == "confident" else ("smoothing" if status == "smoothed" else "detector"))


def _own(w: WindowEvidence, side: int) -> float:
    value = (w.own_start, w.own_end)[side]
    return float((w.start, w.end)[side] if value is None else value)


def _merge_short(runs: list[LanguageRun]) -> list[LanguageRun]:
    """A run shorter than ``MIN_RUN_SECONDS`` is not a passage: it is absorbed by the neighbour with the longer duration (adjacent same-language runs fuse)."""
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for i, run in enumerate(runs):
            if run.seconds < MIN_RUN_SECONDS and len(runs) > 1:
                left, right = runs[i - 1] if i > 0 else None, runs[i + 1] if i + 1 < len(runs) else None
                target = max((r for r in (left, right) if r is not None), key=lambda r: r.seconds)
                target.start, target.end = min(target.start, run.start), max(target.end, run.end)
                target.windows += run.windows
                runs.pop(i)
                changed = True
                break
        for i in range(len(runs) - 1):
            if runs[i].language == runs[i + 1].language:
                a, b = runs[i], runs[i + 1]
                a.end, a.windows = b.end, a.windows + b.windows
                if b.status == "unsure":
                    a.status = "unsure"
                runs.pop(i + 1)
                changed = True
                break
    return runs


def make_contiguous(runs: list[LanguageRun], duration: float) -> None:
    """The runs cover the whole recording: a gap between two runs (silence nobody scanned) is split between them; the first starts at 0, the last ends at the end."""
    if not runs:
        return
    runs[0].start = 0.0
    runs[-1].end = max(runs[-1].end, duration)
    for left, right in zip(runs, runs[1:]):
        edge = (left.end + right.start) / 2.0
        left.end = right.start = edge


def snap_boundaries(runs: list[LanguageRun], speech: Sequence[tuple[float, float]], reach: float = 15.0) -> list[LanguageRun]:
    """Move each boundary between two runs to the longest pause within ``reach`` seconds of it (a language change happens between sentences), else leave it where it is."""
    if len(runs) < 2 or not speech:
        return runs
    pauses = [(a_end, b_start) for (_a0, a_end), (b_start, _b1) in zip(speech, speech[1:]) if b_start - a_end >= 0.3]
    for left, right in zip(runs, runs[1:]):
        edge = left.end
        near = [(b - a, a, b) for a, b in pauses if a - reach <= edge <= b + reach]
        if near:
            _width, a, b = max(near)
            edge = (a + b) / 2.0
            left.end = right.start = min(max(edge, left.start + 0.5), right.end - 0.5)
    return runs


def mark_vocal_stretches(runs: list[LanguageRun], windows: Sequence[WindowEvidence], loud_rms: float = 0.02, max_speech_share: float = 0.2) -> None:
    """A run decoded with the voice-activity filter on would lose singing and speech over music: runs whose windows are loud but mostly 'non-speech' for the filter decode without it."""
    for run in runs:
        mine = [w for w in windows if w.seconds > 0 and run.start - 1e-6 <= (w.start + w.end) / 2.0 < run.end + 1e-6]
        if not mine:
            continue
        loud = [w for w in mine if w.rms >= loud_rms]
        suspicious = [w for w in loud if w.speech_seconds <= max_speech_share * w.seconds]
        if len(suspicious) >= max(1, len(mine) // 2):
            run.vad = "off"


# ====================================================================================================================== window layout
def window_starts(duration: float, speech: Sequence[tuple[float, float]], hop: float, size: float = 30.0, loud: Optional[Sequence[tuple[float, float]]] = None) -> list[float]:
    """Start times of the windows to identify: a window every ``hop`` seconds wherever there is speech (as the voice-activity filter sees it) or loud sound it does not call speech.
    Each window fits inside the file; silence is not scanned."""
    if duration <= 0 or hop <= 0:
        return []
    last = max(duration - size, 0.0)
    regions = sorted([*speech, *(loud or [])])
    if not regions:
        return [round(min(i * hop, last), 2) for i in range(int(duration // hop) + 1)] if duration > 0 else []
    merged: list[tuple[float, float]] = []
    for a, b in regions:
        if merged and a <= merged[-1][1] + 1.0:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    starts: list[float] = []
    for a, b in merged:
        t = a
        while True:
            starts.append(round(min(t, last), 2))
            if t + size >= b:
                break
            t += hop
    return sorted(set(starts))


def adaptive_hop(speech_seconds: float, on_gpu: bool) -> float:
    """How often to identify the language. Every window is one encoder pass (about 0.1 s on a GPU, 8-12 s on a CPU with Large-v3), so the number of windows is capped: dense for short
    recordings, sparser for long ones. The hop is never below 10 s (GPU) / 20 s (CPU) and never above the 30 s window itself."""
    max_windows = 400 if on_gpu else 90
    floor = 10.0 if on_gpu else 20.0
    return float(min(30.0, max(floor, speech_seconds / max_windows)))


def build_plan(windows: Sequence[WindowEvidence], speech: Sequence[tuple[float, float]], hop: float, duration: float) -> LanguagePlan:
    """The language plan of a recording from its window evidence."""
    windows = sorted(windows, key=lambda w: w.start)
    assign_spans(windows)
    labels = smooth_languages(windows)
    runs = runs_from_windows(windows, labels)
    make_contiguous(runs, duration)
    snap_boundaries(runs, speech)
    mark_vocal_stretches(runs, windows)
    seconds: dict[str, float] = {}
    for run in runs:
        seconds[run.language] = seconds.get(run.language, 0.0) + run.seconds
    languages = [l for l, _s in sorted(seconds.items(), key=lambda item: -item[1])]
    mode = "music" if any(r.vad == "off" for r in runs) else ("multilingual" if len(languages) > 1 else "single")
    plan = LanguagePlan(runs=runs, windows=[{"start": round(w.start, 1), "top": w.top[0], "p": round(w.top[1], 3), "speech": round(w.speech_seconds, 1)} for w in windows], hop=hop, mode=mode,
                        languages=languages)
    if not runs:
        plan.notes.append("no language evidence")
    return plan
