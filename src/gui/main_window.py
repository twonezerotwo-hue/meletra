"""The application window.

Version 0.2 shows a *simple* flow: start -> transcribing -> "Your lecture is ready" -> transcript / study workspace. The pages of
version 0.1.1 (start, running, results) are still built and still driven by every event, but they sit behind the simple pages: they
do the real work (file checks, request building, live technical numbers) and are shown as "Technical details" or as the whole window
when the window style "Classic" is chosen in Settings -> Advanced.
"""

from __future__ import annotations

import base64
import os
import sys
import logging
from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import QByteArray, QTimer, Qt
from PySide6.QtGui import QAction, QCloseEvent, QDragEnterEvent, QDropEvent, QKeySequence
from PySide6.QtWidgets import (QApplication, QComboBox, QFileDialog, QLineEdit, QMainWindow, QPlainTextEdit, QSizePolicy, QSpinBox, QStackedWidget,
                               QWidget)

from .. import APP_NAME, __version__
from ..ingest.queue import export_named
from ..services.job_service import lecture_title
from ..ingest.router import VIDEO_EXT
from ..services.paths import default_output_dir
from ..services.transcription_service import JobRequest
from . import accessibility, power, theme
from .about_dialog import AboutDialog
from .active_page import ActivePage
from .components import ask
from .context import AppServices
from .drop_overlay import DropOverlay, Toast, split_supported, unsupported_message
from .files_page import FilesPage
from .home_page import HomePage
from .job_controller import JobController, Launcher
from .queue_runner import QueueRunner
from .results_page import ResultsPage
from .settings_dialog import SettingsDialog
from .settings_simple import SimpleSettingsDialog
from .shell import RAIL_WIDTH, AppShell
from .simple_active import SimpleActive
from .simple_batch import SimpleBatch
from .simple_home import SimpleHome
from .simple_result import SimpleResult
from .study_view import StudyWorkspace
from .tasks import TaskThread, stop_threads
from .technical import TechnicalDetailsDialog
from ..language_catalogue import language_name
from ..translation.service import TranslationService
from .transcript_view import TranscriptView
from .translate_dialog import TranslateDialog
from .widgets import any_local_file, first_media_path, media_filter

log = logging.getLogger(__name__)

PAGE_HOME, PAGE_ACTIVE, PAGE_RESULTS = 0, 1, 2                 # the classic pages (0.1.1)
SIMPLE_HOME, SIMPLE_ACTIVE, SIMPLE_RESULT, SIMPLE_TRANSCRIPT, SIMPLE_STUDY, SIMPLE_BATCH, SIMPLE_FILES = range(7)
ROOT_SIMPLE, ROOT_CLASSIC = 0, 1
PAGE_MARGINS = (32, 28, 32, 24)                                 # left, top, right, bottom of every page of the new window
CANCEL_GRACE_MS = 15000                     # how long CANCEL waits for a safe stop before the worker is ended
CLOSE_GRACE_MS = 20000                       # how long the window waits for a safe pause before it ends the worker


class MainWindow(QMainWindow):
    def __init__(self, services: AppServices, launcher: Optional[Launcher] = None) -> None:
        super().__init__()
        accessibility.install()                                # the lecture cards are buttons for assistive technology
        self.services = services
        self.setWindowTitle(f"{APP_NAME}")
        self.setWindowIcon(theme.app_icon())
        self.setAcceptDrops(True)
        self.setMinimumSize(980, 640)                          # explicit: a hidden page's wide labels must never make the window wider than a laptop screen
        self.resize(1180, 760)
        self.last_request: Optional[JobRequest] = None
        self.source_name = ""
        self.current_output: Optional[Path] = None              # the lecture folder shown on the result / transcript / study screens
        self.current_pdf: Optional[Path] = None                 # the slides attached on the start screen: the study material uses them too
        self.choose_folder: Callable[[str], str] = self._choose_folder_dialog      # replaced in tests
        self.choose_pdf_file: Callable[[], str] = self._choose_pdf_dialog
        self._closing = False
        self._threads: list[TaskThread] = []
        self.runner: Optional[QueueRunner] = None                 # the running batch / document conversion, if any
        self._cancel_after_pause = False
        self._tts = None                                          # created the first time something is read aloud
        self._listening = False
        self.confirm_close: Callable[[], bool] = self._ask_close

        self.interrupted = services.jobs.reconcile()           # a worker that vanished (crash, power loss) = paused job
        self.archive_report = self.repair_archive()            # lecture folders on disk and archive records agree (silent; changes nothing but the records)

        self.controller = JobController(services, launcher, parent=self)
        self.stack = QStackedWidget()
        self.home = HomePage(services)
        self.active = ActivePage(services)
        self.results = ResultsPage(services)
        for page in (self.home, self.active, self.results):
            page.layout().setContentsMargins(24, 16, 24, 16)
            self.stack.addWidget(page)
        self.simple_stack = QStackedWidget()
        self.simple_home = SimpleHome(services, self.home)
        self.simple_active = SimpleActive()
        self.simple_result = SimpleResult(services)
        self.transcript_view = TranscriptView(services)
        self.study = StudyWorkspace(services)
        services.study.is_busy = lambda: self.controller.is_running or self.queue_active          # a study model never shares the graphics card with a transcription
        self.simple_batch = SimpleBatch()
        self.files_page = FilesPage(services)
        for page in (self.simple_home, self.simple_active, self.simple_result, self.transcript_view, self.study, self.simple_batch, self.files_page):
            page.layout().setContentsMargins(*PAGE_MARGINS)
            self.simple_stack.addWidget(page)
        self.simple_shell = AppShell(self.simple_stack)
        self.simple_shell.navigate.connect(self._navigate)
        self.root_stack = QStackedWidget()
        self.root_stack.addWidget(self.simple_shell)
        self.root_stack.addWidget(self.stack)
        self.setCentralWidget(self.root_stack)
        self.drop_overlay = DropOverlay(self.root_stack)             # whole-window drop: shown while a supported file is dragged over any part of the window
        self.toast = Toast(self.root_stack)                          # a small notice (a file that cannot be used, a resumable lecture)
        self.toast.inset_left = RAIL_WIDTH
        self.drop_overlay.inset_left = RAIL_WIDTH
        self._build_menu()
        self._wire()
        self.apply_ui_mode()
        self._restore_geometry()
        # a 1 px widget that holds the focus while nothing is being edited: no gold ring after a click or a page change. It is the LAST widget of the tab chain, so Tab goes
        # straight to the first control of the page ("skip to content") and Shift+Tab to the sidebar.
        self.focus_anchor = QWidget(self.root_stack)
        self.focus_anchor.setFixedSize(1, 1)
        self.focus_anchor.setFocusPolicy(Qt.ClickFocus)
        self.focus_anchor.setAccessibleName("Start of the page")
        self.focus_anchor.move(0, 0)
        self.focus_anchor.show()
        QWidget.setTabOrder(self.simple_shell.buttons["settings"], self.focus_anchor)      # sidebar, anchor, then the page: Tab from the anchor enters the page, after the page comes the sidebar

        unfinished = services.jobs.unfinished()
        if unfinished and self.interrupted:                      # the cards say Paused / Processing; only a crash deserves a notice
            n = len(unfinished)
            self.notify(f"{n} transcription{'s' if n != 1 else ''} stopped unexpectedly. Your progress is saved: open {'them' if n != 1 else 'it'} from Recent lectures to continue.", 12000)
        self.detect_gpu()

    # ------------------------------------------------------------------ construction
    def _build_menu(self) -> None:
        bar = self.menuBar()
        file_menu = bar.addMenu("&File")
        self.open_action = QAction("&Open audio or video ...", self)
        self.open_action.setShortcut(QKeySequence.Open)
        self.open_action.triggered.connect(self.open_file_dialog)
        exit_action = QAction("E&xit", self)
        exit_action.setMenuRole(QAction.QuitRole)              # macOS: moves to the application menu as "Quit" (Cmd+Q); ignored elsewhere
        exit_action.setShortcut(QKeySequence.Quit)             # Ctrl+Q / Cmd+Q; empty on Windows (no shortcut there, as before)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(self.open_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)
        tools = bar.addMenu("&Settings")
        for attr, text, tab in (("settings_action", "&Settings ...", "General"), ("models_action", "&Models ...", "Models"),
                                ("diagnostics_action", "Advanced &Diagnostics ...", "Diagnostics")):
            action = QAction(text, self)
            action.triggered.connect(lambda _c=False, t=tab: self.open_settings(t))
            setattr(self, attr, action)
            if attr != "diagnostics_action":
                tools.addAction(action)
        self.settings_action.setShortcut(QKeySequence.Preferences)
        self.settings_action.setMenuRole(QAction.PreferencesRole)      # macOS: "Settings..." in the application menu (Cmd+,)
        help_menu = bar.addMenu("&Help")
        logs = QAction("Open &logs folder", self)
        logs.triggered.connect(lambda: self.services.open_path(self.services.paths.logs))
        about = QAction("&About", self)
        about.setMenuRole(QAction.AboutRole)                           # macOS: "About MELETRA" in the application menu
        about.triggered.connect(self.show_about)
        help_menu.addAction(self.diagnostics_action)
        help_menu.addAction(logs)
        help_menu.addSeparator()
        help_menu.addAction(about)
        for action in (self.open_action, self.settings_action):
            self.addAction(action)                                         # Ctrl+O and Ctrl+, also work without a visible menu bar
        self.open_lecture_action = QAction("Add a finished &lecture folder ...", self)
        self.open_lecture_action.triggered.connect(self.open_finished_lecture)
        self.study_pdf_action = QAction("Study from a &PDF only ...", self)
        self.study_pdf_action.triggered.connect(self.study_from_pdf)
        file_menu.insertAction(exit_action, self.open_lecture_action)
        file_menu.insertAction(exit_action, self.study_pdf_action)

    def _wire(self) -> None:
        c, a = self.controller, self.active
        self.home.start_requested.connect(self.start_job)
        self.home.settings_requested.connect(self.open_settings)
        self.home.models_changed.connect(lambda: None)
        c.status.connect(a.set_status)
        c.progress.connect(a.set_progress)
        c.progress.connect(self._show_progress_in_title)
        c.log_line.connect(a.log.add)
        c.warning.connect(a.show_warning)
        c.info.connect(a.update_facts)
        c.completed.connect(self.on_completed)
        c.paused.connect(self.on_paused)
        c.failed.connect(self.on_failed)
        c.running_changed.connect(self._running_changed)
        a.pause_requested.connect(c.request_pause)
        a.resume_requested.connect(self.resume_job)
        a.open_log_requested.connect(self.open_log)
        a.back_requested.connect(self.show_home)
        self.results.new_requested.connect(self.new_transcription)
        self.results.view_requested.connect(self.view_lecture_file)
        sa, sr = self.simple_active, self.simple_result
        c.progress.connect(sa.set_progress)
        c.stage.connect(sa.set_stage)
        c.stage.connect(self.simple_batch_stage)
        c.info.connect(sa.set_facts)
        c.warning.connect(sa.show_warning)
        sa.pause_requested.connect(c.request_pause)
        sa.resume_requested.connect(self.resume_job)
        sa.back_requested.connect(self.show_home)
        sa.cancel_requested.connect(self.cancel_job)
        sa.details_requested.connect(self.show_technical_details)
        self.simple_home.convert_requested.connect(self.start_queue)
        self.simple_home.lecture_requested.connect(self.open_lecture_record)
        self.simple_home.view_all_requested.connect(lambda: self._navigate("files"))
        self.simple_home.privacy_requested.connect(lambda: self.open_settings("Privacy & Data"))
        self.files_page.open_requested.connect(self.open_lecture_record)
        self.files_page.add_folder_requested.connect(self.open_finished_lecture)
        self.files_page.pdf_study_requested.connect(self.study_from_pdf)
        self.files_page.repair_requested.connect(self.repair_archive_with_report)
        sb = self.simple_batch
        sb.pause_requested.connect(lambda: self.runner and self.runner.pause())
        sb.resume_requested.connect(lambda: self.runner and self.runner.resume())
        sb.cancel_requested.connect(lambda: self.runner and self.runner.cancel())
        sb.open_item_requested.connect(self.open_queue_item)
        sb.open_folder_requested.connect(lambda: self.runner and self.services.open_path(self.runner.root))
        sb.new_requested.connect(self.new_transcription)
        sr.read_requested.connect(self.open_transcript)
        sr.study_requested.connect(self.open_study)
        sr.listen_requested.connect(self.toggle_listen)
        sr.details_requested.connect(self.show_technical_details)
        sr.new_requested.connect(self.new_transcription)
        sr.translate_requested.connect(self.open_translate_dialog)
        sr.reprocess_requested.connect(self.reprocess_original)
        self.transcript_view.back_requested.connect(lambda: self._goto("result"))
        self.study.back_requested.connect(self._study_back)
        self.study.translate_requested.connect(self.open_translate_dialog)

    # ------------------------------------------------------------------ GPU
    def detect_gpu(self) -> None:
        thread = TaskThread(lambda progress, cancelled: self.services.gpu.detect(), self)
        thread.succeeded.connect(self.home.set_gpu_status)
        thread.failed.connect(lambda msg: self.home.device_banner.setText(f"Could not check the GPU ({msg}). The CPU will be used."))
        self._threads.append(thread)
        self.gpu_thread = thread
        thread.start()

    # ------------------------------------------------------------------ job flow
    def start_job(self, request: JobRequest) -> None:
        self.last_request = request
        self.current_pdf = Path(request.pdf_path) if request.pdf_path else None
        self.source_name = Path(request.source).name
        info = self.services.models.info(request.model_key)
        status = self.home.gpu_status
        facts = {"model": info.title, "language": request.language, "subject": request.subject}
        self.active.begin(self.source_name, facts, status.name if status and status.cuda_available else "",
                          status.total_mib if status and status.cuda_available else 0)
        self.simple_active.begin(self.source_name, request.language, request.subject,
                                 "video" if Path(request.source).suffix.lower() in VIDEO_EXT else "audio")
        self._goto("active")
        try:
            self.controller.start(request)
        except Exception as exc:  # noqa: BLE001 - shown instead of crashing the window
            log.exception("Could not start the job")
            self.active.finished_failed(f"The transcription could not be started: {exc}")
            self.active.resume_button.setEnabled(True)
            self.simple_active.finished_failed(f"The transcription could not be started: {exc}")
            self.simple_active.resume_button.setEnabled(True)

    def resume_job(self) -> None:
        if self.last_request is None or self.controller.is_running:
            return
        request = self.last_request
        request.resume = True
        self.start_job(request)

    def on_completed(self, event: dict) -> None:
        output = event.get("output_dir")
        if not output:
            message = "The transcription finished but reported no output folder. Open the log for details."
            self.active.finished_failed(message)
            self.simple_active.finished_failed(message)
            return
        self.current_output = Path(output)
        if self.last_request is not None:
            try:
                export_named(Path(output), Path(self.last_request.source))       # Lecture_01_raw.txt, Lecture_01.srt ...
            except OSError:
                log.warning("Could not write the named copies of the transcript", exc_info=True)
        self.results.show_results(Path(output), self.source_name, event)
        self.simple_result.show_results(Path(output), self.source_name, event)
        self._share_transcript(Path(output), event)
        if not self.queue_active:                              # a batch keeps its own screen: the next file must stay visible while it converts
            self._goto("result")
        self._attention(f"Finished: {self.source_name}")
        self.home.refresh_unfinished()
        if self.services.settings.settings.open_folder_after:
            try:
                self.services.open_path(Path(output))
            except OSError:
                log.warning("Could not open the output folder", exc_info=True)

    def _share_transcript(self, folder: Path, event: dict) -> None:
        """Settings -> Privacy & Data: queue the finished transcript for sharing. The transcript is already saved here; this only queues a copy of its text and
        returns at once (the upload runs in the background). Nothing it does can fail the lecture."""
        try:
            record = self.services.jobs.get(self.controller.job_id) if self.controller.job_id else None
            name = lecture_title(record) if record is not None else self.source_name
            request = self.last_request
            self.services.sharing.transcript_saved(folder, name, spoken=str(event.get("language") or ""), requested=request.language if request else "",
                                                   output=request.output_language if request else "")
        except Exception:  # noqa: BLE001
            log.debug("sharing: not queued", exc_info=True)

    def on_paused(self, _event: dict) -> None:
        self.active.finished_paused()
        self.simple_active.finished_paused()
        self.home.refresh_unfinished()
        if self._cancel_after_pause:                           # CANCEL: the job was paused safely (progress kept), then we go back
            self._cancel_after_pause = False
            self.new_transcription()

    def cancel_job(self) -> None:
        if self.controller.is_running:
            self._cancel_after_pause = True
            self.controller.request_pause()
            QTimer.singleShot(CANCEL_GRACE_MS, self._force_cancel)

    def _force_cancel(self) -> None:
        """The job did not stop within the grace time (it is inside something that cannot be interrupted): end that worker only.
        Nothing is lost that a safe pause would have kept - before the first segment there is nothing to keep."""
        if self._cancel_after_pause and self.controller.is_running:
            log.warning("The worker did not stop in time after CANCEL; ending it")
            self.controller.kill()
            QTimer.singleShot(500, self.controller.poll)

    def on_failed(self, message: str) -> None:
        if self._cancel_after_pause:                           # the worker was ended on purpose after CANCEL: that is not a failure
            self._cancel_after_pause = False
            self.home.refresh_unfinished()
            self.new_transcription()
            return
        self.active.finished_failed(message)
        self.simple_active.finished_failed(message)
        self._attention(f"Stopped: {self.source_name}")
        self.home.refresh_unfinished()

    def _show_progress_in_title(self, event: dict) -> None:
        self.setWindowTitle(f"{float(event.get('percent', 0.0)):.0f}%  -  {self.source_name}  -  {APP_NAME}")

    def notify(self, text: str, milliseconds: int = 8000) -> None:
        """A short message: the status bar of the classic window, a small notice at the bottom of the new one."""
        self.statusBar().showMessage(text, milliseconds)
        if self.services.settings.settings.ui_mode != "classic":
            self.toast.show_message(text, milliseconds, kind="info")

    def _attention(self, text: str) -> None:
        """Long jobs run while the user does something else: flash the taskbar button when the job needs them."""
        self.notify(text, 30000)
        if not self.isActiveWindow():
            QApplication.alert(self, 0)

    def _running_changed(self, running: bool) -> None:
        if running:
            self.study._stop_running()                       # study material being made by a model is abandoned: the graphics card belongs to the transcription (it is made again on request)
            self.services.study.release_models()             # ... and the model this program loaded for it leaves the graphics card
        power.keep_awake(running)
        if not running:
            self.setWindowTitle(APP_NAME)
        self.home.setEnabled(not running)
        self.simple_home.set_busy("Transcribing..." if running else "")            # the recent lectures stay clickable while one is running
        self.open_action.setEnabled(not running)
        self.active.resume_button.setEnabled(not running)
        self.simple_active.resume_button.setEnabled(not running)
        if not running and self._closing:
            QTimer.singleShot(0, self.close)

    def open_log(self) -> None:
        log_file = self.services.paths.job_log(self.controller.job_id) if self.controller.job_id else self.services.paths.app_log
        self.services.open_path(log_file if Path(log_file).exists() else self.services.paths.logs)

    def show_home(self) -> None:
        self.home.refresh_unfinished()
        self.simple_home.refresh()
        self._goto("home")

    def new_transcription(self) -> None:
        self.home.reset_file()
        self.simple_home.clear_selection()
        self.simple_home.refresh()
        self._goto("home")

    @property
    def queue_active(self) -> bool:
        """A batch / document conversion is running or paused: new files wait until it is finished."""
        return self.runner is not None and self.runner.queue.state in ("running", "paused")

    # ------------------------------------------------------------------ simple flow
    def _goto(self, name: str) -> None:
        """Show a screen. The classic stack follows the three classic screens, so both views always agree about where the job is."""
        classic = {"home": PAGE_HOME, "active": PAGE_ACTIVE, "result": PAGE_RESULTS}
        simple = {"home": SIMPLE_HOME, "active": SIMPLE_ACTIVE, "result": SIMPLE_RESULT, "transcript": SIMPLE_TRANSCRIPT,
                  "study": SIMPLE_STUDY, "batch": SIMPLE_BATCH, "files": SIMPLE_FILES}
        if name in classic:
            self.stack.setCurrentIndex(classic[name])
        self.simple_stack.setCurrentIndex(simple[name])
        self.simple_shell.set_current(name)
        self.rest_focus()

    def rest_focus(self) -> None:
        """Nothing is highlighted until the keyboard is used: the focus rests on the anchor after a page change (text fields keep it while someone types)."""
        anchor = getattr(self, "focus_anchor", None)
        if anchor is None or self.services.settings.settings.ui_mode == "classic":
            return
        current = QApplication.focusWidget()
        if current is None or not isinstance(current, (QLineEdit, QComboBox, QPlainTextEdit, QSpinBox)):
            anchor.setFocus(Qt.OtherFocusReason)

    def _navigate(self, name: str) -> None:
        if name == "home":
            self.show_home()
        elif name == "transcription":
            if self.controller.is_running:
                self._goto("active")
            elif self.current_output is not None:
                self._goto("result")
            else:
                self.show_home()
        elif name == "study":
            if self.current_output is not None:
                self.open_study("overview")
            else:
                self._navigate("files")                       # no lecture is open: pick one from the archive
        elif name == "files":
            self.files_page.refresh()
            self._goto("files")
        elif name == "settings":
            self.open_settings("General")

    def apply_ui_mode(self) -> None:
        """"simple" (default): the new screens. "classic": the technical window of 0.1.1 (Settings -> Advanced -> Window style)."""
        classic = self.services.settings.settings.ui_mode == "classic"
        self.root_stack.setCurrentIndex(ROOT_CLASSIC if classic else ROOT_SIMPLE)
        ignored = QSizePolicy.Ignored
        self.stack.setSizePolicy(QSizePolicy.Expanding if classic else ignored, QSizePolicy.Expanding if classic else ignored)    # a hidden window style adds nothing to the minimum size
        self.open_lecture_action.setEnabled(not classic)
        self.study_pdf_action.setEnabled(not classic)
        # the new window has ONE navigation (the sidebar) and ONE Settings entry: no menu bar and no status bar (messages appear as a small notice); macOS keeps its native menu bar
        bars_visible = classic or sys.platform == "darwin"
        self.menuBar().setVisible(bars_visible)
        self.statusBar().setVisible(classic)

    def open_transcript(self) -> None:
        if self.current_output is None:
            return
        text = self.simple_result.text_file                               # a converted document (PDF, picture, text file)
        if text is not None:
            self.transcript_view.show_document(self.current_output, text, self.source_name)
        else:                                                              # a lecture: Original | Translated when the Output Language was made (translated first)
            self.transcript_view.show_transcript(self.current_output, self.source_name, prefer_translated=True)
        self._goto("transcript")

    def view_lecture_file(self, path: str) -> None:
        """Technical details -> a transcript or report file: shown in MELETRA's own reader (search, copy), never in an outside text editor."""
        file = Path(path)
        folder = self.results.output_dir or file.parent
        self.transcript_view.show_document(folder, file, f"{self.source_name or folder.name} - {file.name}")
        self._goto("transcript")

    def make_translate_dialog(self, folder: Path) -> TranslateDialog:
        return TranslateDialog(self.services, folder, self)

    def open_translate_dialog(self) -> None:
        """Translate the saved transcript of the lecture on screen into another language: no file is asked for, no job is started, the speech recogniser is not used."""
        if self.current_output is None:
            return
        dialog = self.make_translate_dialog(self.current_output)
        dialog.translated.connect(self.show_translation)
        dialog.exec()

    def reprocess_original(self) -> None:
        """A lecture made by an earlier version: read the original recording again. This is a separate action from translating - it needs the recording, and the finished results of the
        earlier run are kept in the lecture folder (previous_versions) when the new run starts."""
        folder = self.current_output
        if folder is None:
            return
        wanted = os.path.normcase(os.path.abspath(str(folder)))
        record = next((r for r in self.services.jobs.list() if r.output_dir and os.path.normcase(os.path.abspath(r.output_dir)) == wanted), None)
        source = Path(record.source) if record is not None and record.source else None
        if source is None or not source.is_file():
            self.toast.show_message("The original recording was not found, so it cannot be read again. Your transcript and its translations are not affected.")
            return
        self.show_home()
        self.simple_home.set_files([str(source)])
        self.toast.show_message("Press Transcribe to read the recording again with the new language detection. Your current transcript is kept.")

    def show_translation(self, language: str) -> None:
        """A translation was just made (or opened): the result screen learns about it and the transcript opens on its tab."""
        if self.current_output is None:
            return
        try:
            stored = TranslationService().stored_targets(self.current_output, any_engine=True)
            self.services.jobs.set_translations(self.current_output, list(stored))
        except Exception:  # noqa: BLE001 - the archive line is a convenience; the translation itself is saved
            stored = {}
        doc = stored.get(language)
        self.simple_result.show_results(self.current_output, self.source_name, None)
        self.transcript_view.show_transcript(self.current_output, self.source_name, language=language)
        self.files_page.refresh()
        self.simple_home.refresh()
        self._goto("transcript")
        if doc is not None and not doc.complete:
            missing = sum(doc.counts.get(k, 0) for k in ("failed", "untranslated"))
            retry = " Translate to another language > the same language tries only the failed parts again." if doc.failed else ""
            self.toast.show_message(f"{language_name(language)} translation saved, but {missing} part(s) could not be translated and are shown in their original language.{retry}")
        elif doc is not None:
            self.toast.show_message(f"{language_name(language)} translation saved. Study material can be made in {language_name(language)} too: Create Study Material > Study language.")

    # ------------------------------------------------------------------ several files, documents, pictures
    def start_queue(self, items: list) -> None:
        """TRANSCRIBE for several files, or for documents and pictures: one after another, the audio through the same worker."""
        if self.controller.is_running or (self.runner is not None and self.runner.queue.state in ("running", "paused")):
            return
        template = self.home.request_template()
        try:
            runner = QueueRunner(self.services, self.controller, items, template, language=template.language, parent=self)
        except ValueError as exc:
            self.simple_home.engine.show_problem(str(exc))
            return
        self.runner = runner
        self.last_request = None
        self.simple_batch.begin(runner.queue)
        runner.changed.connect(self.simple_batch.refresh)
        runner.changed.connect(self._queue_changed)
        runner.progress.connect(self.simple_batch.set_progress)
        runner.finished.connect(self._queue_finished)
        self._goto("batch")
        runner.start()

    def simple_batch_stage(self, event: dict) -> None:
        """The same stage reports drive the queue screen while a recording of a batch is being prepared."""
        if self.runner is not None and self.queue_active and self.stack.currentIndex() != PAGE_ACTIVE:
            self.simple_batch.set_stage(event)

    def _queue_changed(self) -> None:
        runner = self.runner
        if runner is not None and runner.queue.current is not None:
            self.source_name = runner.queue.current.name

    def _queue_finished(self, outcome: str) -> None:
        runner = self.runner
        if runner is None or outcome != "done":
            return
        done = [i for i in runner.items if i.state == "completed"]
        self._attention(f"Finished: {len(done)} of {len(runner.items)} file(s)")
        if len(runner.items) == 1 and done:
            self.open_queue_item(0)

    def open_queue_item(self, index: int) -> None:
        """The result screen of one finished file of the queue."""
        if self.runner is None or not 0 <= index < len(self.runner.items):
            return
        item = self.runner.items[index]
        if item.state != "completed" or not item.output_dir:
            return
        folder = Path(item.output_dir)
        self.current_output, self.source_name, self.current_pdf = folder, item.detected.path.name, None
        if item.detected.is_media:
            self.results.show_results(folder, self.source_name, None)
            self.simple_result.show_results(folder, self.source_name, None)
        else:
            self.simple_result.show_results(folder, self.source_name, None, text_file=Path(item.files["text"]))
        self._goto("result")

    # ------------------------------------------------------------------ read aloud
    def toggle_listen(self) -> None:
        """LISTEN TO TEXT: reads the transcript / document text aloud; pressing it again stops. Never touches the transcription."""
        if self._listening and self._tts is not None:
            self._tts.stop()
            return
        file = self.simple_result._transcript_file()
        if file is None:
            return
        try:
            from ..tts import TtsService
            self._tts = self._tts or TtsService()
            if file.suffix.lower() == ".json" or "timestamped" in file.name.lower():        # the clock times are not spoken
                from ..archive.transcript import load_transcript

                found = load_transcript(file.parent)
                text = found.text if found is not None else file.read_text(encoding="utf-8")
            else:
                text = file.read_text(encoding="utf-8-sig")
        except (OSError, ImportError) as exc:
            self.simple_result.message.setText(f"The text could not be read aloud ({exc.__class__.__name__}).")
            return
        tts = self._tts
        self._listening = True
        self.simple_result.set_listening(True)
        self.simple_result.message.setText("Reading aloud ...")
        language = self.services.settings.settings.language
        if self.simple_result.translated_file is not None:                   # the translated text is read aloud in its own language
            language = self.simple_result.translated_file.stem.split("_")[1]
        thread = TaskThread(lambda progress, cancelled: tts.speak(text, language=language if language in ("en", "tr", "de", "hu") else "auto",
                                                                  should_stop=cancelled), self)
        thread.succeeded.connect(self._listen_finished)
        thread.failed.connect(self._listen_failed)
        self._threads.append(thread)
        self._listen_thread = thread
        thread.start()

    def _listen_finished(self, _result=None) -> None:
        self._listening = False
        self.simple_result.set_listening(False)
        self.simple_result.message.setText("")

    def _listen_failed(self, message: str) -> None:
        self._listen_finished()
        self.simple_result.message.setText(message)

    def open_study(self, tab: str = "overview", pdf: Optional[Path] = None) -> None:
        if self.current_output is None:
            return
        slides = self.current_pdf
        if pdf is None and slides is not None and slides.is_file() and not self.services.study.pdf_name(self.current_output):
            pdf = slides                                         # first Study Lecture after a transcription that had slides: use them
        self.study.open_lecture(self.current_output, self.source_name, tab=tab, pdf=pdf)
        self._goto("study")

    def _study_back(self) -> None:
        self.simple_result.refresh_study()
        self._goto("result")

    def repair_archive(self, extra_roots=()):
        """Make the lecture folders on disk and the archive records agree (``src/archive/reconcile.py``). Never raises, never touches a transcript."""
        from ..archive.reconcile import reconcile_archive

        try:
            report = reconcile_archive(self.services.jobs, self.services.settings.settings.output_dir, extra_roots)
        except Exception:                                         # noqa: BLE001 - the archive must open even if a folder is unreadable
            log.warning("archive repair failed", exc_info=True)
            return None
        if report.changed:
            log.info("Archive repair: %d record(s) repaired, %d restored", report.repaired, report.recovered)
        return report

    def repair_archive_with_report(self) -> None:
        """The Files page action: repair, refresh the lists and say exactly what was found."""
        report = self.repair_archive()
        self.files_page.refresh()
        self.simple_home.refresh()
        if report is None:
            self.toast.show_message("The lecture archive could not be checked. Nothing was changed.")
            return
        self.archive_report = report
        text = "\n".join(report.summary_lines())
        if getattr(self, "show_report", None) is not None:
            self.show_report("Repair lecture archive", text)
        else:
            from .components import ConfirmDialog

            ConfirmDialog(self, "Repair lecture archive", text, accept="OK", reject="Close").exec()

    def open_lecture_record(self, record) -> None:
        """Click on a lecture card: completed -> its transcript / study screen; running -> the progress; queued -> the queue; paused or failed -> the start screen
        with the file chosen (and the error, if there was one) so TRANSCRIBE continues or retries."""
        state = record.state
        if state == "completed":
            from ..archive.resolver import lecture_folder

            folder = lecture_folder(record)
            if folder is None or not folder.is_dir():
                self.toast.show_message(f"The files of '{lecture_title(record)}' were not found at {record.output_dir or 'its folder'}. They may have been moved or deleted. "
                                        "Files -> Repair lecture archive looks for them again.")
                return
            if record.output_dir != str(folder):                  # an old record that named the chosen folder: from now on it names the lecture's own
                self.services.jobs.repair(record.job_id, output_dir=str(folder), output_root=record.output_root or record.output_dir)
            self.show_finished(folder, lecture_title(record), dict(record.summary or {}))
        elif state == "running":
            if self.controller.is_running:
                self._goto("active")
            elif self.queue_active:
                self._goto("batch")
            else:
                self.show_home()
        elif state == "queued" and self.queue_active:
            self._goto("batch")
        else:
            self.show_home()
            source = Path(record.source) if record.source else None
            if source is not None and source.is_file():
                self.simple_home.set_files([str(source)])
                if state == "failed" and record.last_error:
                    self.simple_home.engine.show_problem(f"Last time this stopped: {record.last_error}")
            else:
                self.toast.show_message(f"The recording '{record.source_name}' was not found, so this lecture cannot be continued.")

    def show_finished(self, folder: Path, name: str = "", summary: Optional[dict] = None) -> None:
        """The result screen for a lecture that was transcribed earlier (or in an earlier session). ``summary`` is what the lecture's own run reported (its job record keeps it), so the
        numbers and the status here are the ones the cards show."""
        folder = Path(folder)
        self.current_output = folder
        self.current_pdf = None                                  # an older lecture: its slides, if any, were saved with its study files
        self.source_name = name or folder.name
        self.results.show_results(folder, self.source_name, summary or None)
        self.simple_result.show_results(folder, self.source_name, summary or None)
        self._goto("result")

    def open_finished_lecture(self) -> None:
        start = self.services.settings.settings.output_dir or self.services.settings.settings.last_input_dir
        folder = self.choose_folder(start)
        if not folder:
            return
        path = Path(folder)
        if not ((path / "local_clean_transcript.txt").is_file() or (path / "local_raw_transcript.txt").is_file()
                or (path / "local_transcript.json").is_file() or (path / "study").is_dir()):
            self.simple_home.engine.show_problem(f"'{path.name}' does not look like a finished lecture. Choose the folder that contains "
                                                 "the transcript files.")
            return
        try:
            self.services.jobs.adopt_folder(path)             # it is in the archive from now on (Files, Recent lectures)
        except OSError:
            log.debug("could not add the lecture to the archive", exc_info=True)
        self.files_page.refresh()
        self.show_finished(path)

    def study_from_pdf(self) -> None:
        """Study a lecture that has slides but no recording: the study files go to a folder named after the PDF."""
        chosen = self.choose_pdf_file()
        if not chosen:
            return
        pdf = Path(chosen)
        s = self.services.settings.settings
        folder = (Path(s.output_dir) if s.output_dir else default_output_dir(pdf)) / pdf.stem
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self.simple_home.engine.show_problem(f"The folder for the study material could not be created ({exc}).")
            return
        self.current_output = folder
        self.source_name = pdf.stem
        self.simple_result.show_results(folder, self.source_name, None)
        self.open_study("overview", pdf=pdf)

    def show_technical_details(self) -> None:
        """The classic pages in a window: the live numbers of the running job, or the reports of a finished one."""
        if self.stack.currentIndex() == PAGE_RESULTS and self.current_output is not None:
            page = ResultsPage(self.services)
            page.show_results(self.current_output, self.source_name)
            TechnicalDetailsDialog("Technical details", page, self).exec()
            return
        index = self.stack.indexOf(self.active)
        if index < 0:
            return
        self.stack.removeWidget(self.active)

        def restore(widget) -> None:
            self.stack.insertWidget(PAGE_ACTIVE, widget)
            self._goto("active") if self.controller.is_running else None

        TechnicalDetailsDialog("Technical details", self.active, self, restore).exec()

    def _choose_folder_dialog(self, start: str) -> str:
        return QFileDialog.getExistingDirectory(self, "Choose the folder of a finished lecture", start)

    def _choose_pdf_dialog(self) -> str:
        path, _ = QFileDialog.getOpenFileName(self, "Select the lecture slides (PDF)", self.services.settings.settings.last_input_dir,
                                              "PDF files (*.pdf)")
        return path

    # ------------------------------------------------------------------ files
    def open_file_dialog(self) -> None:
        if self.queue_active:
            self.notify("Files are being converted. Choose new files when it is finished.")
            return
        start = self.services.settings.settings.last_input_dir
        path, _ = QFileDialog.getOpenFileName(self, "Select an audio or video file", start, media_filter())
        if path:
            self.select_file(path)

    def select_file(self, path: str) -> None:
        self.show_home()
        self.home.set_file(path)

    def open_external_file(self, path: str) -> None:
        """A file handed over by the operating system (Finder "Open With", the Dock icon): audio / video is selected, PDFs are attached."""
        if self.controller.is_running or self.queue_active:
            self.notify("A lecture is being transcribed. Open the new file when it is finished.")
            return
        target = Path(path)
        if self.services.settings.settings.ui_mode != "classic":
            self.show_home()
            if target.suffix.lower() == ".pdf" and self.home.source is not None and not self.simple_home.selection:
                self.home.set_pdf(target)                       # a recording is chosen already: the PDF is its slides
            else:
                self.simple_home.set_files([str(target)])
        elif target.suffix.lower() == ".pdf":
            self.show_home()
            self.home.set_pdf(target)
        elif target.suffix.lower() in (".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".mp4", ".mov", ".mkv", ".webm"):
            self.select_file(str(target))

    @staticmethod
    def _dragged_paths(mime) -> list[str]:
        return [url.toLocalFile() for url in mime.urls() if url.isLocalFile()] if mime.hasUrls() else []

    def _drop_blocked(self) -> str:
        if self.controller.is_running or self.queue_active:
            return "A lecture is being transcribed. Drop new files when it is finished."
        return ""

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        """Anywhere in the window: a supported file shows the "Drop lecture files" overlay; files that cannot be used show a small error and no overlay."""
        paths = self._dragged_paths(event.mimeData())
        if not paths:
            event.ignore()
            return
        if self.services.settings.settings.ui_mode == "classic":                       # the technical window keeps its old, media-only drop
            if self.stack.currentIndex() == PAGE_HOME and not self.controller.is_running and not self.queue_active and any_local_file(event.mimeData()) is not None:
                event.acceptProposedAction()
            else:
                event.ignore()
            return
        good, bad = split_supported(paths)
        busy = self._drop_blocked()
        if busy:
            self.toast.show_message(busy)
            event.ignore()
        elif not good:
            self.toast.show_message(unsupported_message(bad))
            event.ignore()
        else:
            self.drop_overlay.show_over()
            self.simple_home.set_drag_active(self.simple_stack.currentWidget() is self.simple_home)
            event.acceptProposedAction()

    def dragMoveEvent(self, event) -> None:  # noqa: N802
        if self.drop_overlay.isVisible() or self.services.settings.settings.ui_mode == "classic":
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:  # noqa: N802
        self.drop_overlay.hide()
        self.simple_home.set_drag_active(False)
        event.accept()

    def add_dropped_files(self, paths: list[str]) -> None:
        """What a drop does: the supported files go to the start screen (one recording, a recording with its slides, or several files as a queue); the rest is named."""
        paths = list(dict.fromkeys(str(Path(p)) for p in paths))                           # the same file twice is one file
        good, bad = split_supported(paths)
        busy = self._drop_blocked()
        if busy:
            self.toast.show_message(busy)
            return
        if not good:
            self.toast.show_message(unsupported_message(bad))
            return
        self.show_home()
        if len(good) == 1 and good[0].lower().endswith(".pdf") and self.home.source is not None and not self.simple_home.selection:
            self.home.set_pdf(Path(good[0]))                                              # a recording is chosen already: the PDF is its slides
        else:
            self.simple_home.set_files(good)
        if bad:
            self.toast.show_message(unsupported_message(bad) + " The other files were added.")

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        self.drop_overlay.hide()
        self.simple_home.set_drag_active(False)
        if self.services.settings.settings.ui_mode != "classic" and event.mimeData().hasUrls():
            paths = self._dragged_paths(event.mimeData())
            if paths:
                event.acceptProposedAction()
                self.add_dropped_files(paths)
                return
        path = first_media_path(event.mimeData())
        if path is None:
            other = any_local_file(event.mimeData())
            self.home.show_problem(f"'{other.name}' is not a supported audio or video file." if other else "Nothing usable was dropped.")
            event.ignore()
            return
        event.acceptProposedAction()
        self.select_file(str(path))

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        QTimer.singleShot(0, self.rest_focus)
        if self.toast.isVisible():
            self.toast.reposition()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self.drop_overlay.isVisible():
            self.drop_overlay.setGeometry(self.root_stack.rect())
        if self.toast.isVisible():
            self.toast.reposition()

    # ------------------------------------------------------------------ dialogs
    def make_settings_dialog(self, tab: str = "General") -> SettingsDialog:
        classic = self.services.settings.settings.ui_mode == "classic"
        dialog = (SettingsDialog if classic else SimpleSettingsDialog)(self.services, self, tab)
        dialog.settings_changed.connect(self.on_settings_changed)
        dialog.models_changed.connect(self.home.refresh_models)
        return dialog

    def open_settings(self, tab: str = "General") -> None:
        dialog = self.make_settings_dialog(tab)
        dialog.exec()
        self.home.refresh_models()

    def on_settings_changed(self) -> None:
        app = QApplication.instance()
        if app is not None:
            theme.apply_theme(app, self.services.settings.settings.theme)
        self.simple_shell.retheme()
        self.simple_home.retheme()
        self.home.load_settings()
        self.simple_home.refresh()
        self.files_page.refresh()
        self.apply_ui_mode()

    def show_about(self) -> None:
        AboutDialog(self).exec()

    # ------------------------------------------------------------------ window state
    def _restore_geometry(self) -> None:
        text = self.services.settings.settings.window_geometry
        if text:
            try:
                self.restoreGeometry(QByteArray(base64.b64decode(text)))
            except (ValueError, TypeError):
                pass

    def _ask_close(self) -> bool:
        return ask(self, "Transcription is running", "Pause it safely and close the program? Your progress is saved and you can resume later.",
                   accept="Pause and close", reject="Keep working", default_accept=False)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if self.controller.is_running:
            if not self._closing:
                if not self.confirm_close():
                    event.ignore()
                    return
                self._closing = True
                self.controller.request_pause()
                self.statusBar().showMessage("Saving your progress ... the window closes by itself.")
                QTimer.singleShot(CLOSE_GRACE_MS, self._force_close)
            event.ignore()                                  # closes itself when the worker has stopped
            return
        try:
            self.services.update_setting(window_geometry=base64.b64encode(bytes(self.saveGeometry())).decode("ascii"))
        except Exception:  # noqa: BLE001 - never block closing
            log.debug("Could not save the window geometry", exc_info=True)
        stop_threads(self)                                  # a model download is cancelled (resumable); checks finish by themselves
        self.services.sharing.stop()
        event.accept()

    def _force_close(self) -> None:
        if self.controller.is_running:
            log.warning("The worker did not pause in time; ending it (the last saved checkpoint stays valid)")
            self.controller.kill()
            QTimer.singleShot(500, self.controller.poll)

