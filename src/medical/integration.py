"""Glue between ``src/pipeline.py`` and the medical expert layer (kept here so the pipeline itself only gains a few call lines).

Nothing here changes how audio is decoded: the soft language lock only changes the DECISION about which language to force, and the
re-decode helper reuses the same ``Transcriber`` / ``DecodeRequest`` the pipeline already has.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from .engine import MedicalOptions, MedicalResult, run_medical_pass
from .language import LanguageVerdict, apply_to_summary, stabilize
from .lexicon import Lexicon

log = logging.getLogger(__name__)

WINDOW_SECONDS = 30.0


def clean_terms(terms: Optional[Sequence[str]], limit: int = 5000) -> list[str]:
    """User / slide terms as a tidy list: stripped, de-duplicated, no empty lines, no comment lines."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in terms or ():
        term = str(raw).strip()
        key = term.casefold()
        if not term or term.startswith("#") or key in seen:
            continue
        seen.add(key)
        out.append(term)
        if len(out) >= limit:
            break
    return out


def _mean_probability(result: Any) -> Optional[float]:
    probs = [w.probability for seg in result.segments for w in (seg.words or []) if w.probability is not None]
    if probs:
        return sum(probs) / len(probs)
    logs = [seg.avg_logprob for seg in result.segments if seg.avg_logprob is not None]
    return math.exp(sum(logs) / len(logs)) if logs else None


def make_decoders(transcriber: Any, wav: Path, profile: dict[str, Any], cfg: dict[str, Any], duration: float,
                  decode_request: Callable[..., Any]) -> tuple[Callable[[str, float], tuple[str, Optional[float]]],
                                                              Callable[[float, float, str], tuple[str, Optional[float]]]]:
    """``(decode_window, redecode)`` built on the pipeline's own transcriber. Both decode with the language FORCED, hotwords off, word
    probabilities on, and without conditioning on earlier text (a short isolated decode must not be steered by anything else)."""
    # a short isolated decode is a check, not the transcript: greedy for the language arbitration (several times faster), a small beam for a repair
    base = {**profile, "word_timestamps": False, "use_hotwords": False, "condition_on_previous_text": False}
    check_profile, repair_profile = {**base, "beam_size": 1}, {**base, "beam_size": min(int(profile.get("beam_size") or 3), 3)}

    def decode(language: str, start: float, seconds: float, short_profile: dict[str, Any]) -> tuple[str, Optional[float]]:
        request = decode_request(audio_path=wav, language=language, profile=short_profile, decoding=cfg["decoding"], vad_parameters=cfg["vad"],
                                 start_offset=max(0.0, start), max_seconds=max(0.5, seconds), total_duration=duration)
        result = transcriber.transcribe(request)
        return " ".join(seg.text.strip() for seg in result.segments).strip(), _mean_probability(result)

    def decode_window(language: str, start: float) -> tuple[str, Optional[float]]:
        return decode(language, start, WINDOW_SECONDS, check_profile)

    def redecode(start: float, end: float, language: str) -> tuple[str, Optional[float]]:
        return decode(language, start, end - start, repair_profile)

    return decode_window, redecode


def soft_lock(sampled: dict[str, Any], decode_window: Optional[Callable[[str, float], tuple[str, Optional[float]]]], *, on_cpu: bool,
              should_stop: Optional[Callable[[], bool]] = None, vocabulary: Optional[Lexicon] = None,
              probe: Optional[tuple[str, float]] = None) -> LanguageVerdict:
    """Decide the primary language and write the decision into the language-sampling summary (see ``language.stabilize``)."""
    medical = None
    if vocabulary is not None:
        index = vocabulary.subset(["abbreviations", "eponyms", "latin"])
        medical = lambda word: bool(index.exact(word))                      # noqa: E731
    verdict = stabilize(sampled, decode_window, on_cpu=on_cpu, medical=medical, should_stop=should_stop, probe=probe)
    apply_to_summary(sampled, verdict)
    log.info("LANGUAGE SOFT LOCK: %s (confidence %.2f)%s | %s", verdict.primary, verdict.confidence, " [multilingual]" if verdict.multilingual else "",
             verdict.reason)
    return verdict


def run_for_pipeline(segments: Sequence[Any], *, plan: Any, options: Any, duration: float, asr_seconds: Optional[float],
                     redecode: Optional[Callable[[float, float, str], tuple[str, Optional[float]]]] = None) -> Optional[MedicalResult]:
    """The medical pass for one finished recording. ``plan`` is the pipeline's RunPlan, ``options`` its RunOptions."""
    sampling = plan.language_samples or {}
    verdict = sampling.get("soft_lock") if isinstance(sampling, dict) else None
    language = plan.language or plan.language_dominant
    confidence = (verdict or {}).get("confidence") if verdict else plan.language_probability
    return run_medical_pass(segments, MedicalOptions(
        language=language, language_confidence=confidence, subjects=list(plan.subjects or ()),
        custom_terms=clean_terms(getattr(options, "custom_medical_terms", None)),
        lecture_terms=clean_terms(plan.pdf_terms), duration=duration, redecode=redecode, asr_seconds=asr_seconds,
        language_verdict=verdict, v4_mode=getattr(options, "medical_intelligence", None) or None))
