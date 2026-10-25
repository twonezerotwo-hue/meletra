"""Working with a lecture by its id, from the files it already has: no recording, no speech recogniser, no new job.

``translate_saved_lecture(lecture_id, target_language)`` is the one way to translate a finished lecture. It reads the SAVED transcript (``source_transcript.json``, or the transcript files of
an earlier version), translates it as text with the offline translation model, and stores the result next to the transcript (``translations/<language>.json`` and the readable files).
The result is kept per lecture, per version of the saved transcript, per target language and per translation engine: asking again opens the stored translation, changing the transcript
makes the old translation stale, and several languages live side by side. The original recording may have been moved or deleted - it is never opened.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Iterable, Optional

from ..archive.resolver import lecture_folder
from ..translation.service import TranslationDoc, TranslationService
from .job_service import JobStore

log = logging.getLogger(__name__)


class LectureNotFound(LookupError):
    """There is no finished lecture with this id, or its folder cannot be found."""


class LectureService:
    def __init__(self, jobs: JobStore, search_dirs: Callable[[], Iterable[Optional[Path]]] = lambda: ()) -> None:
        self.jobs = jobs
        self._search_dirs = search_dirs

    def folder(self, lecture_id: str) -> Path:
        record = self.jobs.get(lecture_id)
        folder = lecture_folder(record) if record is not None else None
        if folder is None:
            raise LectureNotFound(f"No saved lecture with the id {lecture_id!r} was found.")
        return folder

    def saved_translations(self, lecture_id: str) -> dict[str, TranslationDoc]:
        """The translations stored for the lecture (any engine): language code -> document."""
        service = TranslationService(self._search_dirs())
        try:
            return service.stored_targets(self.folder(lecture_id), any_engine=True)
        finally:
            service.close()

    def translate_saved_lecture(self, lecture_id: str, target_language: str, *, translator: Optional[TranslationService] = None, progress: Optional[Callable[[float], None]] = None,
                                should_cancel: Optional[Callable[[], bool]] = None, force: bool = False) -> TranslationDoc:
        """Translate the saved transcript of the lecture into ``target_language`` (an Output Language code). Returns the stored translation at once when it is current.
        Raises ``LectureNotFound`` (no such lecture), ``TranslationUnavailableError`` (no saved transcript, unsupported language, translation model not installed) or
        ``TranslationCancelled``. Nothing is read from the recording and no job is created."""
        folder = self.folder(lecture_id)
        own = translator is None
        service = translator or TranslationService(self._search_dirs())
        try:
            doc = service.translate(folder, target_language, progress=progress, should_cancel=should_cancel, force=force)
            codes = sorted(service.stored_targets(folder, any_engine=True))
        finally:
            if own:
                service.close()
        try:
            self.jobs.update(lecture_id, translations=codes)                 # the cards in Recent lectures and Files show which languages the lecture has
        except Exception:  # noqa: BLE001 - the list is a convenience; the translation itself is saved
            log.debug("could not record the translations of %s", lecture_id, exc_info=True)
        return doc
