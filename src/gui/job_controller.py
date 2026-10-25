"""Runs ONE transcription in a separate worker process and turns its event file into Qt signals.

The GUI never does speech recognition. ``JobController.start`` writes a request file, launches the worker
(the packaged .exe itself with ``--worker``, or ``python -m src.services.worker`` from a source checkout) and then, four times
a second, reads the worker's event file. If the worker dies without saying why, the controller says so.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional, Protocol

from PySide6.QtCore import QObject, QProcess, QTimer, Signal

from ..pipeline import job_key
from ..services.transcription_service import (
    EXIT_COMPLETED, EXIT_PAUSED, EventTail, JobRequest, events_path, request_path, request_stop, run_worker, stop_path,
    worker_command,
)
from . import friendly
from .context import AppServices

log = logging.getLogger(__name__)


class ProcessHandle(Protocol):
    def running(self) -> bool: ...
    def exit_code(self) -> Optional[int]: ...
    def terminate(self) -> None: ...


class QProcessHandle:
    def __init__(self, program: str, args: list[str]) -> None:
        self.process = QProcess()
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.setStandardOutputFile(QProcess.nullDevice())
        if sys.platform == "win32":
            try:
                self.process.setCreateProcessArgumentsModifier(lambda a: setattr(a, "flags", a.flags | 0x08000000))  # CREATE_NO_WINDOW
            except Exception:  # noqa: BLE001 - cosmetic only
                pass
        self.process.start(program, args)

    def running(self) -> bool:
        return self.process.state() != QProcess.NotRunning

    def exit_code(self) -> Optional[int]:
        if self.running():
            return None
        if self.process.error() == QProcess.FailedToStart:
            return -1
        return self.process.exitCode()

    def terminate(self) -> None:
        if self.running():
            self.process.kill()


class ThreadHandle:
    """Runs the worker code in a thread of THIS process (tests and `RUN_GUI_DEV` without a second interpreter)."""

    def __init__(self, request_file: Path) -> None:
        self._code: Optional[int] = None
        self._thread = threading.Thread(target=self._run, args=(request_file,), daemon=True)
        self._thread.start()

    def _run(self, request_file: Path) -> None:
        try:
            self._code = run_worker(request_file)
        except BaseException:  # noqa: BLE001
            self._code = 1

    def running(self) -> bool:
        return self._thread.is_alive()

    def exit_code(self) -> Optional[int]:
        return None if self._thread.is_alive() else self._code

    def terminate(self) -> None:  # cannot kill a thread; the stop file is the only way
        pass


class Launcher(Protocol):
    def launch(self, request_file: Path) -> ProcessHandle: ...


class QProcessLauncher:
    def launch(self, request_file: Path) -> ProcessHandle:
        program, args = worker_command(request_file)
        return QProcessHandle(program, args)


class ThreadLauncher:
    def launch(self, request_file: Path) -> ProcessHandle:
        return ThreadHandle(request_file)


STAGE_TEXT = {
    "job_started": "Starting ...", "source_validated": "Source file checked", "audio_preparing": "Preparing audio ...",
    "audio_ready": "Audio ready", "language_detecting": "Detecting language ...", "gap_scan_started": "Looking for missed speech ...",
    "quality_audit_started": "Checking transcript quality ...", "pause_requested": "Pausing safely - saving the current segment ...",
}


class JobController(QObject):
    status = Signal(str)                  # one line for the live status label
    progress = Signal(dict)               # chunk_progress payload
    log_line = Signal(str, str)           # level, text (live log panel)
    warning = Signal(str)
    job_started = Signal(str)             # job id
    completed = Signal(dict)
    paused = Signal(dict)
    failed = Signal(str)
    info = Signal(dict)                   # language / subject / model / device facts for the active-job screen
    stage = Signal(dict)                  # {stage, percent (None = not measurable), done, total}: the work before the first segment
    running_changed = Signal(bool)

    def __init__(self, services: AppServices, launcher: Optional[Launcher] = None, poll_ms: int = 200, parent=None) -> None:
        super().__init__(parent)
        self.services = services
        self.launcher: Launcher = launcher or QProcessLauncher()
        self.job_id = ""
        self.handle: Optional[ProcessHandle] = None
        self._tail: Optional[EventTail] = None
        self._terminal: Optional[str] = None
        self._pause_requested = False
        self._finalizing = False
        self._progress_seen = False
        self._started = 0.0
        self.facts: dict[str, Any] = {}
        self._timer = QTimer(self)
        self._timer.setInterval(poll_ms)
        self._timer.timeout.connect(self.poll)

    @property
    def is_running(self) -> bool:
        return self.handle is not None

    # ------------------------------------------------------------------ control
    def start(self, request: JobRequest) -> str:
        if self.is_running:
            raise RuntimeError("A transcription is already running.")
        paths = self.services.paths
        source = Path(request.source)
        job_id = job_key(source)
        request.job_id, request.data_root, request.parent_pid = job_id, str(paths.root), os.getpid()
        request.extra_model_dirs = list(self.services.settings.settings.extra_model_dirs)
        self.services.jobs.create_or_get(
            job_id, source, output_dir=request.output_dir, checkpoint_dir=str(paths.checkpoints / job_id),
            model_key=request.model_key, language=request.language, subject=request.subject, mode=request.mode,
            pdf=request.pdf_path, output_language=request.output_language, state="queued", last_error="")
        file = request_path(paths, job_id)
        request.save(file)
        events_path(paths, job_id).write_text("", encoding="utf-8")
        stop_path(paths, job_id).unlink(missing_ok=True)
        self.job_id, self._tail, self._terminal, self._pause_requested = job_id, EventTail(events_path(paths, job_id)), None, False
        self._finalizing = False
        self._progress_seen = False
        self.facts = {"model": request.model_key, "mode": request.mode, "language": request.language, "subject": request.subject}
        self._started = time.monotonic()
        self.handle = self.launcher.launch(file)
        self._timer.start()
        self.running_changed.emit(True)
        self.job_started.emit(job_id)
        self.status.emit("Starting ...")
        return job_id

    def request_pause(self) -> None:
        """PAUSE SAFELY: the worker saves the segment it is writing, closes the checkpoint and stops."""
        if self.is_running and not self._pause_requested:
            self._pause_requested = True
            request_stop(self.services.paths, self.job_id)
            self.status.emit(STAGE_TEXT["pause_requested"])

    def kill(self) -> None:
        """Last resort (application is closing): the checkpoint of the last saved segment is still valid."""
        if self.handle is not None:
            self.handle.terminate()

    # ------------------------------------------------------------------ polling
    def poll(self) -> None:
        if self._tail is not None:
            for event in self._tail.poll():
                self._dispatch(event)
        handle = self.handle
        if handle is not None and not handle.running():
            for event in (self._tail.poll() if self._tail else []):       # whatever it wrote just before exiting
                self._dispatch(event)
            self._worker_finished(handle.exit_code())

    def _worker_finished(self, code: Optional[int]) -> None:
        self._timer.stop()
        self.handle = None
        if self._terminal is None:
            if code == EXIT_PAUSED:
                self._terminal = "paused"
                self.paused.emit({})
            elif code == EXIT_COMPLETED:
                self._terminal = "failed"
                self.failed.emit("The transcription process ended without a result. Open the log for details.")
            else:
                self._terminal = "failed"
                self.failed.emit(f"The transcription process ended unexpectedly (exit code {code}). "
                                 f"Your progress is saved - choose the same file again to resume. Open the log for details.")
        self.running_changed.emit(False)

    def _dispatch(self, event: dict[str, Any]) -> None:
        kind = event.get("type", "")
        if kind == "chunk_progress":
            self._progress_seen = True
            self.progress.emit(event)
        elif kind == "log":
            self.log_line.emit(str(event.get("level", "INFO")), str(event.get("text", "")))
        elif kind == "warning":
            self.warning.emit(str(event.get("message", "")))
            self.log_line.emit("WARNING", str(event.get("message", "")))
        elif kind == "language_detected":
            self.facts["language"] = event.get("language") or self.facts.get("language")
            self.info.emit(dict(self.facts))
            self._say(f"Language: {event.get('language') or 'auto'}")
        elif kind == "subject_detected":
            areas = event.get("areas") or []
            self.facts["subject"] = ", ".join(areas) if areas else "general medical"
            self.info.emit(dict(self.facts))
            self._say(f"Subject: {self.facts['subject']}")
        elif kind == "job_started":
            self.facts["device"] = event.get("device", "")
            self.info.emit(dict(self.facts))
            self._say(STAGE_TEXT["job_started"])
        elif kind == "model_loading":
            self.stage.emit({"stage": "model_loading", "percent": None})
            self.facts["device"] = event.get("device", self.facts.get("device", ""))
            self.facts["compute_type"] = event.get("compute_type", "")
            self.info.emit(dict(self.facts))
            self._say(f"Loading {Path(str(event.get('model', 'model'))).name or 'model'} on {str(event.get('device', '')).upper()} ...")
        elif kind == "model_loaded":
            self._say(f"Model loaded in {event.get('seconds', 0):.0f} s ({event.get('compute_type', '')})")
        elif kind == "resume_started":
            self._say(f"Resuming: {event.get('completed', 0)} of {event.get('chunks', 0)} chunks already finished")
        elif kind == "chunk_started":
            if not self._progress_seen:
                self.stage.emit({"stage": "transcribing", "percent": None})
            self._say(f"Chunk {event.get('index')} / {event.get('chunks')}")
        elif kind == "chunk_finished":
            self._say(f"Chunk {event.get('index')} {'already done' if event.get('skipped') else 'complete'}")
        elif kind == "checkpoint_saved":
            self._say("Checkpoint saved")
        elif kind == "gap_found":
            self._say(f"Found {event.get('seconds', 0):.0f} s of speech without text (gap {event.get('index')}/{event.get('gaps')})")
        elif kind == "gap_repair_finished":
            self._say("Recovered the missing speech" if event.get("repaired") else "Missing stretch could not be recovered safely")
        elif kind == "loop_repaired":
            self._say("Repeated-text loop repaired" if event.get("repaired") else "A repeated-text loop could not be repaired")
        elif kind == "translation_started":
            self.stage.emit({"stage": "translating", "percent": None})
            self._say("Translating the transcript ...")
        elif kind == "translation_progress":
            self.stage.emit({"stage": "translating", "percent": float(event.get("percent", 0.0))})
            self.status.emit(f"Translating ... {float(event.get('percent', 0.0)):.0f}%")
        elif kind == "translation_finished":
            self._say("Translation finished" if event.get("segments") else "No translation was made (the spoken-language transcript is complete)")
        elif kind == "stage":
            self.stage.emit(dict(event))
            self.status.emit(friendly.stage_texts(str(event.get("stage")))[1])
        elif kind in ("gap_scan_started", "quality_audit_started"):
            if not self._finalizing:                                   # the simple screen: one "Finalizing..." for both
                self._finalizing = True
                self.stage.emit({"stage": "finalizing", "percent": None})
            self._say(STAGE_TEXT[kind])
        elif kind in STAGE_TEXT:
            self._say(STAGE_TEXT[kind])
        elif kind == "completed":
            self._terminal = "completed"
            self.status.emit("Completed")
            self.completed.emit(event)
        elif kind == "paused":
            self._terminal = "paused"
            self.status.emit("Paused safely.")
            self.paused.emit(event)
        elif kind == "failed":
            self._terminal = "failed"
            self.status.emit("Failed")
            self.failed.emit(str(event.get("error", "The transcription failed.")))

    def _say(self, text: str) -> None:
        self.status.emit(text)
        self.log_line.emit("INFO", text)
