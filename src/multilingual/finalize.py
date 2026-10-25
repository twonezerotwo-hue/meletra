"""After the decode: the canonical source transcript (what was heard, language by language) and the text translation made from it - no audio, no recogniser."""

from __future__ import annotations

import logging
import time
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from ..translation import METHOD_FAILED, METHOD_M2M100
from ..translation.outputs import TranslationResult
from ..translation.service import STATUS_SAME, STATUS_TRANSLATED, TranslatedSegment, TranslationDoc, TranslationService, write_files
from . import source as source_mod
from .source import SourceTranscript

log = logging.getLogger(__name__)


def build_source(lecture_id: str, segments: Sequence[Any], medical: Any = None, *, input_language: str = "auto", output_language: str = "", engine: Optional[dict[str, Any]] = None,
                 coverage: Optional[dict[str, Any]] = None, notes: Optional[list[str]] = None) -> SourceTranscript:
    """The transcript as the student sees it: the medical-corrected text when that pass ran (each corrected segment keeps the language of the recogniser's segment it came from),
    else the recogniser's own text. The recogniser's text stays in ``raw_text`` whenever it differs."""
    corrected = list(getattr(medical, "segments", None) or [])
    if not corrected:
        return source_mod.from_segments(lecture_id, segments, input_language=input_language, requested_output_language=output_language, engine=engine, coverage=coverage, notes=notes)
    proxies, texts = [], []
    for item in corrected:
        raw = segments[item.raw_index] if 0 <= item.raw_index < len(segments) else None
        proxies.append(SimpleNamespace(start=item.start, end=item.end, text=item.raw_text or item.text, language=getattr(raw, "language", None),
                                       language_meta=getattr(raw, "language_meta", None), flags=getattr(raw, "flags", []) or []))
        texts.append(item.text)
    return source_mod.from_segments(lecture_id, proxies, input_language=input_language, requested_output_language=output_language, engine=engine, coverage=coverage,
                                    corrected_texts=texts, notes=notes)


def store_source(folder: Path, source: SourceTranscript) -> Path:
    return source_mod.write(Path(folder), source)


def run_text_translation(folder: Path, result: TranslationResult, *, models_dirs: Sequence[Optional[Path]], device: str, emit: Callable[..., None],
                         stop_point: Callable[[], None], release: Optional[Callable[[], None]] = None) -> TranslationResult:
    """Translate the saved transcript of ``folder`` into ``result.target`` (the service writes ``translations/<code>.json`` and the ``translated_<code>_transcript.*`` copies).
    A failure leaves the spoken transcript as it is and is recorded, never raised; a pause is passed on. ``release`` frees the speech model first (on a GPU the speech model and the
    translation model are never in memory together: the recognition is finished, the translation starts)."""
    target = result.target or ""
    started = time.perf_counter()
    emit("translation_started", method=METHOD_M2M100, source=result.source, target=target)
    if release is not None:
        try:
            release()
        except Exception:  # noqa: BLE001 - freeing memory is an aid; the translation goes on
            log.warning("could not free the speech model before translating", exc_info=True)
    service = TranslationService(models_dirs, device=device)

    def progress(fraction: float) -> None:
        emit("translation_progress", percent=round(100 * fraction, 1), target=target)

    def cancel() -> bool:
        stop_point()                                              # raises the pipeline's pause exception when the student paused
        return False

    try:
        doc = service.translate(Path(folder), target, progress=progress, should_cancel=cancel)
        result.method, result.stored_by_service, result.counts = METHOD_M2M100, True, doc.counts
        result.segments = []
        result.source_languages = sorted({s.source_language for s in doc.segments if s.source_language and s.source_language != "und"})
        result.reason = result.reason or "offline translation model"
        result.deferred = False
    except Exception as exc:  # noqa: BLE001
        if type(exc).__name__ == "PauseRequested":
            raise
        log.warning("Translation to %s failed: %s", target, exc, exc_info=True)
        result.method, result.error, result.segments = METHOD_FAILED, f"{type(exc).__name__}: {str(exc)[:300]}", []
        result.deferred = False
    finally:
        service.close()
    result.seconds = time.perf_counter() - started
    emit("translation_finished", method=result.method, target=target, segments=sum(result.counts.values()) or len(result.segments), seconds=round(result.seconds, 1))
    return result


def store_whisper_translation(folder: Path, result: TranslationResult, source: SourceTranscript, engine_id: str) -> None:
    """Keep an English translation that Whisper made from the audio in the same place and shape as a text translation (``translations/en.json``), so it is found, shown and reused the same way.
    Its segments are Whisper's own (they are not aligned one-to-one with the transcript's), and the document says so."""
    rows = []
    for index, seg in enumerate(result.segments, start=1):
        meta = getattr(seg, "language_meta", None) or {}
        copied = meta.get("source") == "copied"
        origin = meta.get("from") or (getattr(seg, "language", None) or "und")
        rows.append(TranslatedSegment(index, float(seg.start), float(seg.end), "en" if copied else str(origin), "", seg.text.strip(), STATUS_SAME if copied else STATUS_TRANSLATED,
                                      "copied" if copied else "whisper"))
    doc = TranslationDoc(lecture_id=source.lecture_id, target=result.target or "en", source_revision=source.revision, engine_id=engine_id, segments=rows,
                         created_at=time.strftime("%Y-%m-%dT%H:%M:%S"), seconds=result.seconds, source_origin=source.origin,
                         warnings=["Made by the speech model's own translate task from the audio; its segments are not aligned one-to-one with the transcript."])
    write_files(Path(folder), doc)
    source_mod.record_translation(Path(folder), doc.target, engine_id, {})
    result.stored_by_service = True
    result.counts = doc.counts


def lecture_record(source: SourceTranscript, *, plan_report: Optional[dict[str, Any]] = None, translation: Optional[dict[str, Any]] = None,
                   audio: Optional[dict[str, Any]] = None, language_check: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """The lecture-level multilingual facts for ``metadata.json`` (only languages that were actually recognised are listed)."""
    return {
        "input_language_mode": source.input_mode, "input_language": source.input_language, "detected_languages": source.detected_languages, "language_counts": source.language_counts,
        "unknown_language_segments": sum(1 for s in source.segments if not s.language_known and s.source_text.strip()),
        "output_languages_available": list(source.output_languages_available), "source_transcript_revision": source.revision, "source_transcript_file": source_mod.SOURCE_FILE,
        "translation_engine_versions": dict(source.translation_engine_versions), "translation_status": (translation or {}).get("segment_status") or {},
        "coverage": source.coverage, "plan": plan_report or {}, "engine": source.engine, "audio_leveling": audio or {}, "language_check": language_check or {},
    }


def translated_ok(counts: dict[str, int]) -> bool:
    return counts.get(STATUS_TRANSLATED, 0) > 0
