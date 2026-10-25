"""Runs a ``BatchQueue`` with the existing machinery: audio/video through the 0.1 ``JobController`` (the real worker process), documents
through a ``TaskThread``. One job at a time; the queue decides which item is next, this class only starts jobs and reports back."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Signal

from ..ingest import queue as q
from ..ingest.documents import DocumentCancelled, read_document, write_document
from ..ingest.router import Detected
from ..pipeline import job_key
from ..services.paths import default_output_dir
from ..services.transcription_service import JobRequest
from .context import AppServices
from .job_controller import JobController
from .tasks import TaskThread


class QueueRunner(QObject):
    changed = Signal()                    # any item or the queue changed state
    progress = Signal(int, dict)          # index of the item, its progress event (chunk_progress / document_progress)
    finished = Signal(str)                # "done" | "cancelled"

    def __init__(self, services: AppServices, controller: JobController, items: list[Detected], template: JobRequest,
                 language: str = "auto", parent=None) -> None:
        super().__init__(parent)
        self.services, self.controller, self.template, self.language = services, controller, template, language
        self.queue = q.BatchQueue(items, self._start, self._pause, self._cancel, self._changed)
        first = items[0].path
        base = Path(services.settings.settings.output_dir) if services.settings.settings.output_dir else default_output_dir(first)
        self.root = q.batch_output_root(base) if len(items) > 1 else Path(template.output_dir or base)
        self._thread: Optional[TaskThread] = None
        self._announced = False
        self._outcome = None                                       # a job report waiting for the worker process to end
        controller.completed.connect(self._job_completed)
        controller.failed.connect(self._job_failed)
        controller.paused.connect(self._job_paused)
        controller.progress.connect(self._job_progress)
        controller.running_changed.connect(self._worker_state)

    # ------------------------------------------------------------------ control (what the screens call)
    def start(self) -> None:
        self.queue.start()

    def pause(self) -> None:
        self.queue.pause()

    def resume(self) -> None:
        self.queue.resume()

    def cancel(self) -> None:
        self.queue.cancel()

    @property
    def items(self) -> list[q.QueueItem]:
        return self.queue.items

    # ------------------------------------------------------------------ callbacks of the queue
    def _index(self) -> int:
        item = self.queue.current
        return self.queue.items.index(item) if item else -1

    def _changed(self) -> None:
        self.changed.emit()
        if self.queue.state in (q.DONE, q.CANCELLED) and not self._announced:
            self._announced = True
            for signal, slot in ((self.controller.completed, self._job_completed), (self.controller.failed, self._job_failed),
                                 (self.controller.paused, self._job_paused), (self.controller.progress, self._job_progress),
                                 (self.controller.running_changed, self._worker_state)):
                signal.disconnect(slot)
            self.finished.emit("done" if self.queue.state == q.DONE else "cancelled")

    def _start(self, item: q.QueueItem) -> None:
        detected = item.detected
        if detected.is_media and detected.needs_wav:
            self._prepare_then_start(item)                       # a long conversion: never on the window's thread
        elif detected.is_media:
            self._launch(item, detected.path)
        else:
            self._start_document(item)

    def _launch(self, item: q.QueueItem, source: Path) -> None:
        single = len(self.queue.items) == 1
        self.controller.start(replace(self.template, source=str(source), output_dir=str(self.root),
                                      pdf_path=self.template.pdf_path if single else "", resume=True))

    def _prepare_then_start(self, item: q.QueueItem) -> None:
        """Media the pipeline would refuse by its extension is first decoded to a WAV (audio only, with real progress, cancellable)."""
        detected = item.detected
        duration = float(detected.duration_s or 0.0)

        def work(progress, cancelled):
            def extracted(seconds: float) -> None:
                progress({"type": "stage", "stage": "extracting", "percent": min(100.0, 100.0 * seconds / duration) if duration else None})
            progress({"type": "stage", "stage": "extracting", "percent": 0.0 if duration else None})
            return str(q.prepare_media(detected, self.services.paths.temp / "v3_inputs", extracted, cancelled))

        thread = TaskThread(work, self)
        thread.progressed.connect(self._document_progress)
        thread.succeeded.connect(lambda path: self._prepared(item, Path(path)))
        thread.failed.connect(self._document_failed)
        thread.done.connect(self._document_finished)
        self._thread = thread
        thread.start()

    def _prepared(self, item: q.QueueItem, path: Path) -> None:
        if item.state == q.PROCESSING:                           # not cancelled / paused in the meantime
            self._launch(item, path)

    def _pause(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            self._thread.cancel()                                  # the page cache keeps the finished pages
        else:
            self.controller.request_pause()

    def _cancel(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            self._thread.cancel()
        else:
            self.controller.kill()

    # ------------------------------------------------------------------ audio / video
    def _active_is_document(self) -> bool:
        item = next((i for i in self.queue.items if i.state == q.PROCESSING), None)
        return item is not None and not item.detected.is_media

    def _deliver(self, report, *args) -> None:
        """The controller reports the end of a job while its worker process is still closing; the next job can only start after."""
        if self.controller.is_running:
            self._outcome = (report, args)
        else:
            report(*args)

    def _worker_state(self, running: bool) -> None:
        if not running and self._outcome is not None:
            (report, args), self._outcome = self._outcome, None
            report(*args)

    def _job_completed(self, event: dict) -> None:
        item = next((i for i in self.queue.items if i.state == q.PROCESSING), None)
        if item is None or not item.detected.is_media:
            return
        folder = str(event.get("output_dir") or "")
        files = q.export_named(Path(folder), item.detected.path) if folder else {}
        self._deliver(self.queue.item_completed, folder, files)

    def _job_failed(self, message: str) -> None:
        if not self._active_is_document():
            self._deliver(self.queue.item_failed, message)

    def _job_paused(self, _event: dict) -> None:
        if not self._active_is_document():
            self._deliver(self.queue.item_paused)

    def _job_progress(self, event: dict) -> None:
        if not self._active_is_document():
            self.progress.emit(self._index(), event)

    # ------------------------------------------------------------------ documents
    def _start_document(self, item: q.QueueItem) -> None:
        detected = item.detected
        cache = self.services.paths.temp / "v3_docs" / f"{job_key(detected.path)}.pages.jsonl"
        root = self.root if len(self.queue.items) > 1 else (Path(self.template.output_dir) if self.template.output_dir
                                                              else default_output_dir(detected.path))

        def work(progress, cancelled):
            document = read_document(detected, language=self.language, progress=progress, should_stop=cancelled, cache_path=cache)
            files = write_document(document, root)
            self.services.study.attach_pages(files["text"].parent, document.page_texts, detected.path.name)
            cache.unlink(missing_ok=True)
            return {"folder": str(files["text"].parent), "files": {k: str(v) for k, v in files.items()},
                    "warnings": document.warnings}

        thread = TaskThread(work, self)
        thread.progressed.connect(self._document_progress)
        thread.succeeded.connect(self._document_done)
        thread.failed.connect(self._document_failed)
        thread.done.connect(self._document_finished)
        self._thread = thread
        thread.start()

    def _document_progress(self, event: dict) -> None:
        self.progress.emit(self._index(), event)

    def _document_done(self, result: dict) -> None:
        self.queue.item_completed(result["folder"], result["files"])

    def _document_finished(self) -> None:
        if self.sender() is self._thread:                         # not the thread of the NEXT document
            self._thread = None

    def _document_failed(self, message: str) -> None:
        if message == "cancelled" or DocumentCancelled.__name__ in message:
            if self.queue.state == q.RUNNING:                      # a pause request: wait for resume()
                self.queue.item_paused()
            return
        self.queue.item_failed(message)
