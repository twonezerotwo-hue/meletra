"""English output for a lecture spoken in another language: Whisper's own translate task over the same 16 kHz audio (no extra model)."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional, Sequence

from ..model_manager import DecodeRequest
from ..schema import Segment

log = logging.getLogger(__name__)


def translate_to_english(transcriber: Any, wav: Any, language: Optional[str], profile: dict[str, Any], cfg: dict[str, Any], duration: float,
                         progress: Optional[Callable[[float], None]] = None, should_stop: Optional[Callable[[], bool]] = None) -> list[Segment]:
    """Decode ``wav`` with ``task="translate"``. ``language`` is the spoken language (None = let Whisper decide per window).
    No hotwords (they are terms of the SPOKEN language) and no conditioning on earlier text. ``should_stop`` is polled after every segment and raises
    the pipeline's pause exception (the caller passes the function that does)."""
    run_profile = {**profile, "use_hotwords": False, "condition_on_previous_text": False, "word_timestamps": False}

    def on_segment(segment: Segment) -> None:
        if progress is not None and duration > 0:
            progress(min(max(segment.end / duration, 0.0), 1.0))
        if should_stop is not None:
            should_stop()

    request = DecodeRequest(audio_path=wav, language=language, profile=run_profile, decoding=cfg["decoding"], vad_parameters=cfg["vad"], hotwords=[],
                            multilingual=language is None, total_duration=duration, on_segment=on_segment, task="translate")
    started = time.perf_counter()
    result = transcriber.transcribe(request)
    log.info("Whisper translated %d segments to English in %.0fs", len(result.segments), time.perf_counter() - started)
    return list(result.segments)


PAD = 0.3                       # seconds of audio kept around a stretch so a word at its edge is not cut
JOIN_GAP = 2.0                  # consecutive segments of one language closer than this are translated together


def foreign_stretches(segments: Sequence[Segment], target: str = "en") -> list[tuple[float, float, Optional[str], bool]]:
    """``(start, end, language, no_voice_filter)`` for each stretch of consecutive segments NOT already in ``target``. ``language`` is None when the segment's language is not known."""
    stretches: list[list[Any]] = []
    open_stretch = False                    # the previous segment belonged to the last stretch (an English segment in between closes it: it is never decoded again)
    for seg in segments:
        language = getattr(seg, "language", None) or None
        language = None if language in ("und", "") else language
        if not seg.text.strip():
            continue
        if language == target:
            open_stretch = False
            continue
        unfiltered = (getattr(seg, "language_meta", None) or {}).get("vad") == "off"
        if open_stretch and stretches[-1][2] == language and stretches[-1][3] == unfiltered and seg.start - stretches[-1][1] <= JOIN_GAP:
            stretches[-1][1] = max(stretches[-1][1], seg.end)
        else:
            stretches.append([seg.start, seg.end, language, unfiltered])
        open_stretch = True
    return [(a, b, l, u) for a, b, l, u in stretches]


def translate_runs_to_english(transcriber: Any, wav: Any, segments: Sequence[Segment], profile: dict[str, Any], cfg: dict[str, Any], duration: float,
                              progress: Optional[Callable[[float], None]] = None, should_stop: Optional[Callable[[], bool]] = None) -> list[Segment]:
    """English for a recording spoken in several languages, without the offline translation model: each stretch that is NOT English goes through Whisper's translate task in ITS OWN
    language (so a Korean section is not decoded as Hindi); English segments are copied unchanged. The stretches are the ones the transcript already holds - nothing is searched for."""
    run_profile = {**profile, "use_hotwords": False, "condition_on_previous_text": False, "word_timestamps": False}
    stretches = foreign_stretches(segments, "en")
    done = 0
    translated: list[Segment] = []
    for start, end, language, unfiltered in stretches:
        if should_stop is not None:
            should_stop()
        a, b = max(0.0, start - PAD), min(duration or end + PAD, end + PAD)
        request = DecodeRequest(audio_path=wav, language=language, profile=run_profile, decoding=cfg["decoding"], vad_parameters=cfg["vad"], hotwords=[], multilingual=False,
                                total_duration=duration, start_offset=a, max_seconds=max(b - a, 0.5), task="translate", vad_override=False if unfiltered else None)
        result = transcriber.transcribe(request)
        for seg in result.segments:
            seg.language = "en"
            seg.language_meta = {"source": "translate", "from": language or "und"}
            translated.append(seg)
        done += 1
        if progress is not None:
            progress(done / max(len(stretches), 1))
    english = [s for s in segments if (getattr(s, "language", None) or None) == "en" and s.text.strip()]
    for seg in english:
        seg.language_meta = {**(getattr(seg, "language_meta", None) or {}), "source_of_text": "spoken"}
    copies = [Segment(s.start, s.end, s.text, "en", s.model, s.backend, language_meta={"source": "copied"}) for s in english]
    merged = sorted([*copies, *translated], key=lambda s: (s.start, s.end))
    log.info("Whisper translated %d non-English stretch(es) to English (%d English segments kept as spoken)", len(stretches), len(english))
    return merged
