"""The pipeline's translation step: decides, runs it, and never lets a failure touch the finished spoken-language transcript."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional

from . import METHOD_FAILED, METHOD_M2M100, METHOD_MISSING, METHOD_NONE, METHOD_WHISPER
from .m2m100 import TranslationUnavailable, find_model
from .outputs import TranslationResult
from .plan import decide
from .whisper_pass import translate_runs_to_english, translate_to_english

log = logging.getLogger(__name__)


def spoken_languages(segments: list) -> list[str]:
    """The languages the recogniser actually labelled the segments with (most frequent first); a segment without a language adds nothing."""
    counts: dict[str, int] = {}
    for seg in segments:
        language = getattr(seg, "language", None)
        if language and language != "und" and seg.text.strip():
            counts[language] = counts.get(language, 0) + 1
    return [l for l, _n in sorted(counts.items(), key=lambda item: -item[1])]


def run_for_pipeline(*, output_language: Optional[str], source_language: Optional[str], mixed: bool, segments: list, transcriber: Any, wav: Any,
                     profile: dict[str, Any], cfg: dict[str, Any], duration: float, model_parent_dirs: list, emit: Callable[..., None],
                     stop_point: Callable[[], None]) -> TranslationResult:
    """The audio-side translation step. With the offline model installed the translation is made later from the saved transcript (``deferred``: no audio needed, nothing to do here).
    Without it English is made by Whisper's translate task from the audio - the whole recording for one spoken language, stretch by stretch for several.
    ``stop_point`` raises the pipeline's pause exception when the student pauses (the translation is then simply redone on resume).
    Any other exception is caught: the spoken transcript is complete whatever happens here."""
    spoken = spoken_languages(segments)
    source = None if (mixed or len(spoken) > 1) else ((spoken[0] if spoken else None) or source_language or None)
    model_dir = find_model(*model_parent_dirs)
    plan = decide(source, output_language, m2m_ready=model_dir is not None, spoken=spoken)
    result = TranslationResult(plan.method, plan.source, plan.target, plan.reason, source_languages=spoken)
    if plan.method in (METHOD_NONE, METHOD_MISSING):
        return result
    if plan.method == METHOD_M2M100:
        result.deferred = True
        return result
    started = time.perf_counter()
    spec = getattr(getattr(transcriber, "adapter", None), "spec", None)
    emit("translation_started", method=plan.method, source=plan.source, target=plan.target)

    def progress(fraction: float) -> None:
        emit("translation_progress", percent=round(100 * fraction, 1), target=plan.target)

    try:
        if spec is not None and getattr(spec, "engine", "faster_whisper") not in ("faster_whisper", "mlx_whisper"):
            raise TranslationUnavailable("this speech model cannot translate")
        unfiltered = any((getattr(s, "language_meta", None) or {}).get("vad") == "off" for s in segments)
        if source is None or unfiltered:
            result.segments = translate_runs_to_english(transcriber, wav, segments, profile, cfg, duration, progress=progress, should_stop=stop_point)
        else:
            result.segments = translate_to_english(transcriber, wav, plan.source, profile, cfg, duration, progress=progress, should_stop=stop_point)
    except Exception as exc:  # noqa: BLE001
        if type(exc).__name__ == "PauseRequested":
            raise
        log.warning("Translation to %s failed: %s", plan.target, exc, exc_info=True)
        result.method, result.error, result.segments = METHOD_FAILED, f"{type(exc).__name__}: {str(exc)[:300]}", []
    result.seconds = time.perf_counter() - started
    emit("translation_finished", method=result.method, target=plan.target, segments=len(result.segments), seconds=round(result.seconds, 1))
    return result
