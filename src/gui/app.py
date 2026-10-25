"""GUI entry: QApplication, theme, icon, single-instance lock, crash log, first-run wizard, main window."""

from __future__ import annotations

import json
import time
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Optional

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")

from PySide6.QtCore import QEvent, QLockFile, QObject, QTimer
from PySide6.QtWidgets import QApplication, QMessageBox

from .. import APP_ID, APP_NAME, COMPANY, __version__
from . import diagnostics_probe as probe_tools
from . import theme
from .context import AppServices, configure_logging
from .first_run import FirstRunDialog, needs_first_run
from .sharing_notice import PrivacyChoiceDialog
from .job_controller import Launcher
from .main_window import MainWindow
from .tasks import stop_threads

log = logging.getLogger(__name__)


def set_windows_app_id() -> None:
    """Own taskbar identity (Windows); nothing to do on other systems."""
    from ..platform import current

    current().prepare_process(f"{COMPANY.replace(' ', '')}.{APP_ID}.{__version__}")


def install_excepthook(services: AppServices) -> None:
    """A bug must not close the program silently: write the traceback to the log and tell the user where it is."""
    def hook(kind, value, tb) -> None:
        text = "".join(traceback.format_exception(kind, value, tb))
        log.critical("Unhandled error:\n%s", text)
        try:
            QMessageBox.critical(None, APP_NAME, f"An unexpected error occurred:\n{value}\n\nDetails are in:\n{services.paths.app_log}")
        except Exception:  # noqa: BLE001
            pass

    sys.excepthook = hook


class FileOpenFilter(QObject):
    """macOS sends "open this file" (double click in Finder, drop on the Dock icon) as a QFileOpenEvent, possibly before the window exists:
    those files wait here until the window is ready. Windows and Linux never send the event."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.pending: list[str] = []
        self.handler = None

    def attach(self, handler) -> None:
        self.handler = handler
        for path in self.pending:
            handler(path)
        self.pending.clear()

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 - Qt API
        if event.type() == QEvent.FileOpen:
            path = event.file()
            if path:
                if self.handler is None:
                    self.pending.append(path)
                else:
                    self.handler(path)
            return True
        return False


def create_application(argv: list[str]) -> QApplication:
    app = QApplication.instance() or QApplication(argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(COMPANY)
    app.setApplicationVersion(__version__)
    app.setWindowIcon(theme.app_icon())
    return app


def unattended_device_choice(_parent, check) -> str:
    """The GPU-memory question of an unattended run (--autorun): the recommended answer, written to the log, never a dialog that waits for a click."""
    from .home_page import recommended_device

    choice = recommended_device(check)
    log.warning("GPU memory low (%s of %s MiB needed free): unattended run continues on the %s", getattr(check, "free_mib", "?"), getattr(check, "required_mib", "?"),
                choice.upper())
    return choice


class AutoRun:
    """Diagnostic hook (``--autorun FILE``): drive the real window - choose the file, press START, wait, write a result file, quit.

    ``pause_at`` (percent) additionally presses PAUSE SAFELY once the job passes that progress, waits for the paused state and
    presses RESUME - the same clicks a user would make. Used to test the packaged .exe end-to-end without a human.
    """

    def __init__(self, window: MainWindow, source: Path, result_file: Path, timeout_s: float, language: str = "auto",
                 pause_at: float = 0.0) -> None:
        self.window, self.source, self.result_file = window, source, result_file
        self.deadline_ms = int(timeout_s * 1000)
        self.language = language
        self.pause_at = pause_at
        self.waited = 0
        self.started = False
        self.finished = False
        self.pause_clicked = False
        self.resume_clicked = False
        self.paused_percent = 0.0
        window.controller.completed.connect(self._completed)
        window.controller.failed.connect(lambda msg: self._finish(1, {"state": "failed", "error": msg}))
        window.controller.paused.connect(self._paused)
        window.controller.progress.connect(self._progress)
        window.home.choose_device = unattended_device_choice                                 # nobody is there to answer a dialog: take the recommended device
        self.timer = QTimer(window)
        self.timer.setInterval(250)
        self.timer.timeout.connect(self._tick)
        window.select_file(str(source))
        idx = window.home.language_combo.findData(language)
        if idx >= 0:
            window.home.language_combo.setCurrentIndex(idx)
        self.probe = probe_tools.UiProbe(window) if probe_tools.enabled() else None          # MLT_UI_TRACE=1: report what the window shows
        self.timer.start()

    def _progress(self, event: dict) -> None:
        if self.pause_at and not self.pause_clicked and float(event.get("percent", 0.0)) >= self.pause_at:
            self.pause_clicked = True
            self.paused_percent = float(event.get("percent", 0.0))
            self.window.active.pause_button.click()

    def _paused(self, _event: dict) -> None:
        if not (self.pause_at and self.pause_clicked and not self.resume_clicked):
            self._finish(10, {"state": "paused"})              # a pause nobody asked for ends the run

    def _tick(self) -> None:
        self.waited += 250
        home = self.window.home
        if not self.started and home.start_button.isEnabled():
            self.started = True
            home.start_button.click()
        elif self.pause_clicked and not self.resume_clicked and self.window.active.resume_button.isEnabled()                 and not self.window.active.resume_button.isHidden():
            self.resume_clicked = True
            self.window.active.resume_button.click()
        elif self.waited >= self.deadline_ms:
            self.window.controller.kill()
            self._finish(124, {"state": "timeout", "error": f"no result within {self.deadline_ms // 1000} s"})
        elif not self.started and self.waited >= 60000:
            self._finish(2, {"state": "not_ready", "error": "the START button never became available (model missing?)"})

    def _completed(self, event: dict) -> None:
        extra = {"pause_requested": self.pause_clicked, "paused_at_percent": round(self.paused_percent, 1),
                 "resumed": self.resume_clicked} if self.pause_at else {}
        self._finish(0, {"state": "completed", **extra, **{k: v for k, v in event.items() if k in (
            "output_dir", "words", "segments", "audio_seconds", "language", "device", "compute_type", "timing", "recovered_gaps")}})

    def _finish(self, code: int, payload: dict) -> None:
        if self.finished:
            return
        self.finished = True
        self.timer.stop()
        try:
            self.result_file.parent.mkdir(parents=True, exist_ok=True)
            extra = probe_tools.payload(getattr(self, "probe", None), self.window, getattr(self, "study_probe", False))
            self.result_file.write_text(json.dumps({"exit_code": code, **payload, **extra}, indent=2, default=str), encoding="utf-8")
        except OSError:
            log.exception("Could not write the autorun result")
        QTimer.singleShot(300, lambda: QApplication.exit(code))


class AutoRunQueue:
    """Diagnostic hook (``--autorun-queue A|B|C``): choose several files / documents on the start screen, press TRANSCRIBE, wait until
    the queue is finished, write a result file, quit. Used to test the packaged build end-to-end (batch, documents, OCR) without a human."""

    def __init__(self, window: MainWindow, paths: list[str], result_file: Path, timeout_s: float, language: str = "auto") -> None:
        self.window, self.paths, self.result_file = window, paths, result_file
        self.deadline_ms = int(timeout_s * 1000)
        self.waited = 0
        self.started = False
        self.finished = False
        idx = window.home.language_combo.findData(language)
        if idx >= 0:
            window.home.language_combo.setCurrentIndex(idx)
        window.simple_home.set_files(paths)
        self.probe = probe_tools.UiProbe(window) if probe_tools.enabled() else None
        self.timer = QTimer(window)
        self.timer.setInterval(250)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

    def _tick(self) -> None:
        self.waited += 250
        home = self.window.simple_home
        runner = self.window.runner
        if not self.started:
            if home.transcribe_button.isEnabled() and home.selection:
                self.started = True
                home.transcribe_button.click()
            elif self.waited >= 60000:
                self._finish(2, {"state": "not_ready", "error": home.problem.text() or "TRANSCRIBE never became available"})
            return
        if runner is not None and runner.queue.state in ("done", "cancelled"):
            items = [{"name": i.name, "state": i.state, "error": i.error, "output_dir": i.output_dir, "files": i.files} for i in runner.items]
            ok = all(i["state"] == "completed" for i in items)
            self._finish(0 if ok else 1, {"state": "completed" if ok else "failed", "root": str(runner.root), "items": items})
        elif self.waited >= self.deadline_ms:
            self.window.controller.kill()
            self._finish(124, {"state": "timeout", "error": f"no result within {self.deadline_ms // 1000} s"})

    def _finish(self, code: int, payload: dict) -> None:
        if self.finished:
            return
        self.finished = True
        self.timer.stop()
        try:
            self.result_file.parent.mkdir(parents=True, exist_ok=True)
            extra = probe_tools.payload(getattr(self, "probe", None), self.window, getattr(self, "study_probe", False))
            self.result_file.write_text(json.dumps({"exit_code": code, **payload, **extra}, indent=2, default=str), encoding="utf-8")
        except OSError:
            log.exception("Could not write the autorun result")
        QTimer.singleShot(300, lambda: QApplication.exit(code))


class AutoRunStudy:
    """Diagnostic hook (``--autorun-study FOLDER``): open a finished lecture's folder in the Study workspace, wait for the material, report the
    workspace (tabs, header, stat cards, buttons of every tab, a revealed flashcard) with ``MLT_UI_TRACE=1``, quit."""

    study_probe = True

    def __init__(self, window: MainWindow, folder: Path, result_file: Path, timeout_s: float) -> None:
        self.window, self.folder, self.result_file = window, folder, result_file
        self.deadline_ms = int(timeout_s * 1000)
        self.waited = 0
        self.finished = False
        self.probe = probe_tools.UiProbe(window) if probe_tools.enabled() else None
        window.show_finished(folder)
        window.open_study("overview")
        self.timer = QTimer(window)
        self.timer.setInterval(250)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

    def _tick(self) -> None:
        self.waited += 250
        if probe_tools.study_ready(self.window):
            self._finish(0, {"state": "completed", "study": "tabs shown"})
        elif self.window.study.stack.currentIndex() == probe_tools.message_page():
            self._finish(1, {"state": "failed", "error": self.window.study.message_text.text()})
        elif self.waited >= self.deadline_ms:
            self._finish(124, {"state": "timeout", "error": f"no study material within {self.deadline_ms // 1000} s"})

    def _finish(self, code: int, payload: dict) -> None:
        if self.finished:
            return
        self.finished = True
        self.timer.stop()
        try:
            self.result_file.parent.mkdir(parents=True, exist_ok=True)
            extra = probe_tools.payload(self.probe, self.window, True)
            self.result_file.write_text(json.dumps({"exit_code": code, **payload, **extra}, indent=2, default=str), encoding="utf-8")
        except OSError:
            log.exception("Could not write the autorun result")
        QTimer.singleShot(300, lambda: QApplication.exit(code))


class AutoRunTranslate:
    """Diagnostic hook (``--autorun-translate FOLDER --to CODE --result OUT``): the real result screen of a finished lecture, the real Translate window and its real Translate button, then
    the same request once more (it must open the SAVED translation at once). Reports what happened: how long each took, the tabs, whether a job was created or the speech model
    was used. No file is chosen, nothing is transcribed."""

    def __init__(self, window: MainWindow, folder: Path, target: str, result_file: Path, timeout_s: float) -> None:
        self.window, self.folder, self.target, self.result_file = window, folder, target, result_file
        self.deadline = time.monotonic() + timeout_s
        self.finished = False
        self.report: dict = {"target": target, "folder": folder.name}
        self.jobs_before = len(window.services.jobs.list())
        window.show_finished(folder)
        button = window.simple_result.translate_button
        self.report["button"] = {"text": button.text(), "visible": button.isVisibleTo(window.simple_result), "enabled": button.isEnabled()}
        self.stage = 1
        self.dialog = None
        self.done = False
        self.started = time.monotonic()
        self._open_dialog()
        self.timer = QTimer(window)
        self.timer.setInterval(200)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

    def _open_dialog(self) -> None:
        self.done = False
        self.dialog = self.window.make_translate_dialog(self.folder)
        self.dialog.translated.connect(lambda code: setattr(self, "done", True))
        self.dialog.combo.setCurrentIndex(max(self.dialog.combo.findData(self.target), 0))
        self.report[f"stage{self.stage}_dialog"] = {"target": self.dialog.target, "button": self.dialog.go_button.text(), "status": self.dialog.status.text()}
        self.dialog.show()
        self.started = time.monotonic()
        if self.dialog.go_button.isEnabled():
            self.dialog.go_button.click()

    def _tick(self) -> None:
        if self.finished:
            return
        if time.monotonic() > self.deadline:
            self._finish(124, {"state": "timeout", "error": f"the translation did not finish (stage {self.stage})", "dialog_status": self.dialog.status.text() if self.dialog else ""})
            return
        if not self.done:
            if self.dialog is not None and not self.dialog._busy and not self.dialog.go_button.isEnabled() and time.monotonic() - self.started > 3:
                self._finish(1, {"state": "failed", "error": self.dialog.status.text()})
            return
        seconds = round(time.monotonic() - self.started, 2)
        doc = self.dialog.result_doc
        self.report[f"stage{self.stage}"] = {"seconds": seconds, "counts": doc.counts if doc else None, "engine": doc.engine_id if doc else None, "created_at": doc.created_at if doc else None}
        self.dialog.close()
        if self.stage == 1:
            self.window.show_translation(self.target)
            view = self.window.transcript_view
            self.report["tabs"] = list(view.mode.buttons)
            self.report["transcript_language"] = view.language
            self.report["transcript_head"] = view.text.toPlainText()[:240]
            self.stage = 2
            self._open_dialog()
            return
        first, second = self.report.get("stage1", {}), self.report.get("stage2", {})
        self.report["reopened_saved_translation"] = bool(first.get("created_at")) and first.get("created_at") == second.get("created_at") and "Open translation" in self.report["stage2_dialog"]["button"]
        self.report["jobs_created"] = len(self.window.services.jobs.list()) - self.jobs_before
        self.report["job_running"] = self.window.controller.is_running
        ok = self.report["reopened_saved_translation"] and self.report["jobs_created"] == 0 and not self.report["job_running"]
        self._finish(0 if ok else 1, {"state": "completed" if ok else "failed"})

    def _finish(self, code: int, payload: dict) -> None:
        if self.finished:
            return
        self.finished = True
        self.timer.stop()
        try:
            self.result_file.parent.mkdir(parents=True, exist_ok=True)
            self.result_file.write_text(json.dumps({"exit_code": code, **self.report, **payload}, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
        except OSError:
            log.exception("Could not write the autorun result")
        QTimer.singleShot(300, lambda: QApplication.exit(code))


def show_startup_dialogs(services, window) -> list[str]:
    """The welcome / model setup (first run only), then - once - the separate question about sharing transcript text with MELETRA. Until it is answered, and whatever the answer was
    except an explicit "share", nothing is shared. Returns what was shown. Unattended runs (--autorun ...) never reach here."""
    shown: list[str] = []
    if needs_first_run(services):
        shown.append("welcome")
        if FirstRunDialog(services, window).exec():
            window.home.refresh_models()
    if services.sharing.needs_choice():
        shown.append("sharing")
        PrivacyChoiceDialog(services, window).exec()
        window.on_settings_changed()                              # the header chip and the Privacy & Data page show the answer at once
    return shown


def run(argv: Optional[list[str]] = None, services: Optional[AppServices] = None, launcher: Optional[Launcher] = None,
        autorun: Optional[dict] = None, autorun_queue: Optional[dict] = None, autorun_study: Optional[dict] = None, ui_check: Optional[dict] = None,
        autorun_translate: Optional[dict] = None, autorun_archive: Optional[dict] = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    set_windows_app_id()
    app = create_application(argv)
    file_open = FileOpenFilter(app)
    app.installEventFilter(file_open)
    services = services or AppServices.create()
    configure_logging(services.paths)
    install_excepthook(services)
    log.info("%s %s starting (data folder %s)", APP_NAME, __version__, services.paths.root)
    lock = QLockFile(str(services.paths.temp / "app.lock"))
    lock.setStaleLockTime(15000)
    if os.environ.get("MLT_ALLOW_MULTIPLE") != "1" and autorun is None and autorun_queue is None and autorun_study is None and autorun_translate is None and autorun_archive is None and ui_check is None and not lock.tryLock(200):
        QMessageBox.information(None, APP_NAME, f"{APP_NAME} is already running.")
        return 0
    theme.apply_theme(app, services.settings.settings.theme)
    window = MainWindow(services, launcher)
    window.show()
    file_open.attach(window.open_external_file)
    runner = None
    if autorun is not None:
        runner = AutoRun(window, Path(autorun["source"]), Path(autorun["result"]), float(autorun.get("timeout", 900)),
                         autorun.get("language", "auto"), float(autorun.get("pause_at", 0.0)))
    elif autorun_queue is not None:
        runner = AutoRunQueue(window, list(autorun_queue["paths"]), Path(autorun_queue["result"]), float(autorun_queue.get("timeout", 900)),
                              autorun_queue.get("language", "auto"))
    elif autorun_study is not None:
        runner = AutoRunStudy(window, Path(autorun_study["folder"]), Path(autorun_study["result"]), float(autorun_study.get("timeout", 300)))
    elif autorun_archive is not None:
        from .autorun_archive import AutoRunArchive

        runner = AutoRunArchive(window, autorun_archive["recording"], Path(autorun_archive["result"]), float(autorun_archive.get("timeout", 300)), bool(autorun_archive.get("study")),
                                bool(autorun_archive.get("exercise")))
    elif autorun_translate is not None:
        runner = AutoRunTranslate(window, Path(autorun_translate["folder"]), autorun_translate["target"], Path(autorun_translate["result"]), float(autorun_translate.get("timeout", 900)))
    elif ui_check is not None:
        from .ui_check import UiCheck

        runner = UiCheck(window, Path(ui_check["out"]), Path(ui_check["lecture"]) if ui_check.get("lecture") else None,
                         Path(ui_check["translated"]) if ui_check.get("translated") else None, ui_check.get("size", ""), ui_check.get("pages", ""))
    else:
        QTimer.singleShot(150, lambda: show_startup_dialogs(services, window))
    services.sharing.start()                                  # the background sender (does nothing while sharing is off, unconfirmed or without an address)
    code = app.exec()
    try:
        stop_threads(window)                                  # however the event loop ended: a Study generation or a download is cancelled and awaited (a QThread destroyed while it runs ends the program with a crash code)
        services.study.release_models(wait=True)              # and the study model this program loaded leaves the graphics card
    except Exception:  # noqa: BLE001 - nothing may keep the program from exiting
        log.debug("clean-up at exit failed", exc_info=True)
    services.sharing.stop()
    del runner
    lock.unlock()
    log.info("%s exits with code %s", APP_NAME, code)
    return code
