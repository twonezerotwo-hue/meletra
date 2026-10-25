"""Segment language check: after decoding, the language of each segment is checked against its OWN audio.

The window scan finds language changes of a few tens of seconds. A short passage - a quotation, a greeting, one sentence in another language - fills only part of a 30-second window and
is invisible to it; decoded in the lecture's language the recogniser writes something plausible, or repeats earlier text (measured on authorized test recordings: the English decode of
a Korean sentence was an echo of sentences spoken a minute before). A segment is the natural unit: its time span comes from the audio itself.

For each (suspicious) segment: the recogniser's language detector listens to just that span; when it names another offered language with some conviction, the model's own likelihood of
the old and the new language on that span is compared; only when the new language is clearly better is the span decoded again in it. Bounded: every GPU run checks all long-enough
segments, a CPU run only the most suspicious few (``CPU_BUDGET``). Nothing is invented: a segment that cannot be improved stays exactly as it was.
"""

from __future__ import annotations

import difflib
import logging
import re
from typing import Any, Callable, Optional, Sequence

from .decode import label_segments, reject_reason
from .plan import language_score
from .scripts import compatible
from .timeline import SUPPORTED_SPOKEN, fold_aliases

log = logging.getLogger(__name__)

MIN_SEGMENT_SECONDS = 2.5
MIN_SPAN_SECONDS = 5.0              # the detector gets at least this much audio around a short segment
DETECT_P = 0.55                     # the detector must name another offered language this surely
SCORE_GAIN = 0.05                   # and the model's own log-probability of it must beat the old language by this much (two independent signals must agree)
CPU_BUDGET = 12
GPU_BUDGET = 400
NEAR_DEVIATION_SECONDS = 15.0
_WORDS = re.compile(r"\w+", re.UNICODE)


def _norm(text: str) -> str:
    return " ".join(_WORDS.findall(text.casefold()))


def suspicion(segments: Sequence[Any], deviating: Sequence[tuple[float, str, float]] = ()) -> list[float]:
    """A number per segment: higher = more worth checking. Echoes of earlier text, odd decoder statistics and nearness to a window where the first sampling saw another language."""
    seen: list[str] = []
    out: list[float] = []
    for seg in segments:
        text = _norm(seg.text or "")
        score = 0.0
        if text and any(text == old or (len(text) > 20 and difflib.SequenceMatcher(None, text, old).ratio() >= 0.9) for old in seen[-60:]):
            score += 3.0
        if seg.compression_ratio is not None and seg.compression_ratio >= 2.0:
            score += 1.5
        if seg.avg_logprob is not None:
            score += max(0.0, min(-seg.avg_logprob, 1.5))
        middle = (seg.start + seg.end) / 2.0
        if any(abs(middle - float(t) - 15.0) <= NEAR_DEVIATION_SECONDS for t, _l, _p in deviating):    # sampled windows start at t and are 30 s long
            score += 3.0
        score += min(seg.end - seg.start, 12.0) / 12.0                                                     # longer segments say more when wrong
        out.append(score)
        if text:
            seen.append(text)
    return out


def recheck_languages(*, transcriber: Any, wav: Any, segments: list, duration: float, profile: dict[str, Any], cfg: dict[str, Any], on_gpu: bool,
                      deviating: Sequence[tuple[float, str, float]] = (), skip_sources: Sequence[str] = ("verified", "boundary", "script", "recheck"),
                      emit: Optional[Callable[..., None]] = None, should_stop: Optional[Callable[[], bool]] = None) -> tuple[list, dict[str, Any]]:
    """Returns ``(segments, report)``; ``segments`` is the input with the improved ones replaced, in time order."""
    from ..model_manager import DecodeRequest

    report: dict[str, Any] = {"checked": 0, "candidates": 0, "changed": 0, "changes": [], "budget": GPU_BUDGET if on_gpu else CPU_BUDGET}
    detector = getattr(getattr(transcriber, "adapter", None), "detect_language_probs", None)
    if detector is None or not segments:
        report["skipped"] = "this speech model cannot identify languages"
        return segments, report
    ordered = sorted(segments, key=lambda s: (s.start, s.end))
    scores = suspicion(ordered, deviating)
    eligible = [i for i, seg in enumerate(ordered) if seg.text.strip() and seg.end - seg.start >= MIN_SEGMENT_SECONDS
                and (getattr(seg, "language_meta", None) or {}).get("source") not in skip_sources]
    report["candidates"] = len(eligible)
    chosen = sorted(sorted(eligible, key=lambda i: -scores[i])[:report["budget"]])
    run_profile = {**profile, "use_hotwords": False, "condition_on_previous_text": False, "word_timestamps": False}
    replaced: dict[int, list] = {}
    taken_until = -1.0
    for number, i in enumerate(chosen):
        if should_stop is not None and should_stop():
            break
        seg = ordered[i]
        if seg.start < taken_until:                                               # inside a span that was just decoded again
            continue
        if emit is not None:
            emit("stage", stage="language_check", percent=round(100.0 * number / max(len(chosen), 1), 1), done=number, total=len(chosen))
        centre = (seg.start + seg.end) / 2.0
        length = min(max(seg.end - seg.start, MIN_SPAN_SECONDS), 30.0)
        start = max(0.0, min(centre - length / 2.0, max(duration - length, 0.0)))
        try:
            found = transcriber.detect_language_windows(wav, [start], length)
        except Exception as exc:  # noqa: BLE001 - an aid on top of a finished transcript
            log.warning("language check failed at %.0f s: %s", seg.start, exc)
            continue
        report["checked"] += 1
        probs = fold_aliases(found[0][1]) if found else {}
        current = seg.language or ""
        options = sorted(((p, l) for l, p in probs.items() if l != current and l in SUPPORTED_SPOKEN and p >= DETECT_P), reverse=True)
        if not options:
            continue
        candidate = options[0][1]
        try:
            evidence = transcriber.score_languages(wav, start, length, [l for l in (current, candidate) if l], 24)
        except Exception as exc:  # noqa: BLE001
            log.warning("language scoring failed at %.0f s: %s", seg.start, exc)
            evidence = {}
        if evidence and candidate in evidence:
            better = language_score(candidate, evidence[candidate]) - (language_score(current, evidence[current]) if current in evidence else 0.0)
            if better < SCORE_GAIN:
                continue
        elif probs[candidate] < 0.85:
            continue
        a, b = max(seg.start - 0.3, 0.0), min(seg.end + 0.3, duration)
        request = DecodeRequest(audio_path=wav, language=candidate, profile=run_profile, decoding=cfg["decoding"], vad_parameters=cfg["vad"], hotwords=[], multilingual=False,
                                total_duration=duration, start_offset=a, max_seconds=max(b - a, 0.8))
        try:
            fresh = [s for s in transcriber.transcribe(request).segments if s.text.strip() and reject_reason(s) is None and compatible(s.text, candidate)]
        except Exception as exc:  # noqa: BLE001
            log.warning("re-reading %.0f s in %s failed: %s", seg.start, candidate, exc)
            continue
        if not fresh:
            continue
        label_segments(fresh, candidate, {"source": "recheck", "candidates": [current] if current else [], "vad": "on"})
        replaced[i] = fresh
        taken_until = max(taken_until, max(s.end for s in fresh))
        report["changed"] += 1
        report["changes"].append({"start": round(seg.start, 1), "end": round(seg.end, 1), "from": current or "und", "to": candidate, "detector": round(probs[candidate], 2)})
    if not replaced:
        return segments, report
    out: list = []
    covered_until = -1.0
    for i, seg in enumerate(ordered):
        if i in replaced:
            out.extend(replaced[i])
            covered_until = max(covered_until, max(s.end for s in replaced[i]))
        elif seg.start >= covered_until - 0.2 or seg.end > covered_until + 0.5:
            out.append(seg)
    out.sort(key=lambda s: (s.start, s.end))
    return out, report
