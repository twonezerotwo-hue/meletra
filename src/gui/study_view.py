"""The study workspace: Overview, Summary, Key Concepts, Flashcards, Questions, Study Mode - for one finished lecture.

Opening a lecture shows what was made before; if nothing exists yet, the material is made on a background thread (index, then the
chosen study method) with a progress bar. Transcription is never involved: the study side only reads the transcript files.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFrame, QHBoxLayout, QLabel, QPlainTextEdit, QProgressBar, QPushButton, QStackedWidget,
    QTabWidget, QVBoxLayout, QWidget,
)

from ..language_catalogue import language_name
from ..services.study_service import StudyState
from ..study.automatic import ENGINE_NAME
from ..study.models import SourceRef
from ..study.sources import SourceError
from . import friendly, tokens
from .components import back_button
from .context import AppServices
from .study_tabs import (ConceptsTab, FlashcardsTab, OverviewTab, QuestionsTab, StudyModeTab, SummaryTab)
from .tasks import TaskThread

log = logging.getLogger(__name__)

TAB_NAMES = ["Overview", "Summary", "Key Concepts", "Flashcards", "Questions", "Study Mode"]
TAB_KEYS = {"overview": 0, "summary": 1, "concepts": 2, "flashcards": 3, "questions": 4, "study": 5}
PAGE_WORKING, PAGE_TABS, PAGE_MESSAGE = 0, 1, 2


class SourceDialog(QDialog):
    """Shows the passage an item was made from."""

    def __init__(self, title: str, text: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(560, 360)
        layout = QVBoxLayout(self)
        self.view = QPlainTextEdit(text)
        self.view.setReadOnly(True)
        layout.addWidget(self.view)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class StudyWorkspace(QWidget):
    back_requested = Signal()
    translate_requested = Signal()            # a lecture spoken in several languages is studied in a translation: the translate window is opened by the main window

    def __init__(self, services: AppServices, parent=None) -> None:
        super().__init__(parent)
        self.services = services
        self.state: Optional[StudyState] = None
        self._threads: list[TaskThread] = []
        self._thread: Optional[TaskThread] = None
        self._pending_tab = "overview"
        self._pending_pdf: Optional[Path] = None
        self.choose_pdf: Callable[[], str] = self._choose_pdf_dialog                   # replaced in tests
        self.show_source_dialog: Callable[[str, str], None] = self._source_dialog       # replaced in tests

        root = QVBoxLayout(self)
        root.setSpacing(tokens.SPACE["lg"])
        self.header = QFrame()
        self.header.setObjectName("studyHeader")
        header_layout = QVBoxLayout(self.header)
        head = QHBoxLayout()
        self.back_button = back_button()
        self.back_button.clicked.connect(self.back_requested.emit)
        title_box = QVBoxLayout()
        self.title = QLabel("Study")
        self.title.setObjectName("studyTitle")
        self.lecture_meta = QLabel("Lecture study workspace")
        self.lecture_meta.setObjectName("muted")
        self.provider_label = QLabel("")
        self.provider_label.setObjectName("muted")
        head.addWidget(self.back_button)
        title_box.addWidget(self.title)
        title_box.addWidget(self.lecture_meta)
        head.addLayout(title_box, 1)
        self.header_study_button = QPushButton("Start Study Mode")
        self.header_study_button.setObjectName("primary")
        self.header_study_button.clicked.connect(lambda: self.select_tab("study"))
        self.header_pdf_button = QPushButton("Add PDF")
        self.header_pdf_button.clicked.connect(self.add_pdf)
        self.header_regen_button = QPushButton("Regenerate")
        self.header_regen_button.clicked.connect(lambda: self.regenerate())
        self.header_export_button = QPushButton("Export")
        self.header_export_button.clicked.connect(self.export_guide)
        for button in (self.header_pdf_button, self.header_regen_button, self.header_export_button, self.header_study_button):
            head.addWidget(button)
        header_layout.addLayout(head)
        header_layout.addWidget(self.provider_label)
        self.language_row = QWidget()
        language_layout = QHBoxLayout(self.language_row)
        language_layout.setContentsMargins(0, 0, 0, 0)
        language_caption = QLabel("Study language")
        language_caption.setObjectName("muted")
        self.language_combo = QComboBox()
        self.language_combo.setAccessibleName("Study language")
        language_caption.setBuddy(self.language_combo)
        self.language_combo.activated.connect(self._language_chosen)
        language_layout.addWidget(language_caption)
        language_layout.addWidget(self.language_combo)
        language_layout.addStretch(1)
        self.language_row.hide()
        header_layout.addWidget(self.language_row)
        root.addWidget(self.header)

        self.stats_widget = QWidget()
        stats = QHBoxLayout(self.stats_widget)
        stats.setContentsMargins(0, 0, 0, 0)
        stats.setSpacing(tokens.SPACE["md"])
        self.stat_values: dict[str, QLabel] = {}
        for key, label in (("concepts", "Key concepts"), ("flashcards", "Flashcards"), ("questions", "Questions"),
                           ("mastery", "Mastery")):
            frame = QFrame()
            frame.setObjectName("statCard")
            box = QHBoxLayout(frame)                                  # one line: the name on the left, the number on the right (the content below gets the room)
            box.setContentsMargins(tokens.SPACE["md"], tokens.SPACE["sm"], tokens.SPACE["md"], tokens.SPACE["sm"])
            caption = QLabel(label)
            caption.setObjectName("muted")
            value = QLabel("—")
            value.setObjectName("studyStat")
            value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            box.addWidget(caption)
            box.addStretch(1)
            box.addWidget(value)
            self.stat_values[key] = value
            stats.addWidget(frame, 1)
        root.addWidget(self.stats_widget)

        self.stack = QStackedWidget()
        root.addWidget(self.stack, 1)

        working = QWidget()
        wl = QVBoxLayout(working)
        wl.addStretch(1)
        self.working_title = QLabel("Preparing your study material ...")
        self.working_title.setObjectName("title")
        self.working_text = QLabel("")
        self.working_text.setObjectName("muted")
        self.working_bar = QProgressBar()
        self.working_bar.setRange(0, 1000)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel)
        for w in (self.working_title, self.working_text, self.working_bar):
            wl.addWidget(w)
        wl.addWidget(self.cancel_button, 0, Qt.AlignLeft)
        wl.addStretch(2)
        self.stack.addWidget(working)

        self.tabs = QTabWidget()
        self.tabs.setObjectName("studyTabs")
        self.tabs.setAccessibleName("Study sections")
        self.tabs.tabBar().setAccessibleName("Study sections")
        self.overview = OverviewTab()
        self.summary = SummaryTab()
        self.concepts = ConceptsTab()
        self.flashcards = FlashcardsTab()
        self.questions = QuestionsTab()
        self.study_mode = StudyModeTab()
        for widget, name in zip((self.overview, self.summary, self.concepts, self.flashcards, self.questions, self.study_mode), TAB_NAMES):
            self.tabs.addTab(widget, name)
        self.stack.addWidget(self.tabs)

        message = QWidget()
        ml = QVBoxLayout(message)
        ml.addStretch(1)
        self.message_title = QLabel("")
        self.message_title.setObjectName("hero")
        self.message_text = QLabel("")
        self.message_text.setWordWrap(True)
        self.message_text.setObjectName("muted")
        ml.addWidget(self.message_title)
        ml.addWidget(self.message_text)
        row = QHBoxLayout()
        self.retry_button = QPushButton("Try again")
        self.retry_button.setObjectName("primary")
        self.retry_button.clicked.connect(lambda: self.regenerate())
        self.message_pdf_button = QPushButton("Add lecture PDF ...")
        self.message_pdf_button.clicked.connect(self.add_pdf)
        self.message_translate_button = QPushButton("Translate to another language")
        self.message_translate_button.setObjectName("primary")
        self.message_translate_button.clicked.connect(self.translate_requested.emit)
        self.message_translate_button.hide()
        row.addWidget(self.retry_button)
        row.addWidget(self.message_translate_button)
        row.addWidget(self.message_pdf_button)
        row.addStretch(1)
        ml.addLayout(row)
        ml.addStretch(2)
        self.stack.addWidget(message)

        self.overview.start_requested.connect(self.select_tab)
        self.overview.regenerate_requested.connect(lambda: self.regenerate())
        self.overview.add_pdf_requested.connect(self.add_pdf)
        self.overview.ask_requested.connect(self.ask)
        self.overview.export_requested.connect(self.export_guide)
        self.overview.source_clicked.connect(self.show_source)
        for tab in (self.summary, self.concepts, self.flashcards, self.questions, self.study_mode):
            tab.source_clicked.connect(self.show_source)
        for tab in (self.flashcards, self.questions, self.study_mode):
            tab.changed.connect(self._refresh_overview)
        self.study_mode.focus_changed.connect(lambda on: self.stats_widget.setVisible(not on))     # Study Mode: one question, little else

    # ------------------------------------------------------------------------------------------------------------------ opening
    def open_lecture(self, folder: Path, title: str = "", tab: str = "overview", pdf: Optional[Path] = None) -> None:
        """Show the study material of the lecture in ``folder``; make it first if it does not exist yet."""
        self._stop_running()                         # a lecture opened while another one is still being made: that work is abandoned
        self._pending_tab = tab
        self._pending_pdf = pdf
        self.state = self.services.study.open(Path(folder), title)
        self.title.setText(self.state.title)
        self.lecture_meta.setText(Path(folder).name)
        self.provider_label.setText(self._engine_text())
        self._fill_languages(Path(folder))
        if pdf is None and self.services.study.needs_translation(Path(folder)):
            source = self.services.study.spoken_languages(Path(folder))
            self._show_message("Choose a language to study in",
                               "This lecture is spoken in " + " · ".join(language_name(c) for c in source) + ". Study material is made in one language, so translate the lecture first - "
                               "the translation is made from the saved transcript, no recording is needed.", translate=True)
            return
        if self.state.ready and pdf is None:
            self._show_tabs()
            return
        if not self.services.study.has_transcript(self.state.folder) and pdf is None and not self.services.study.pdf_name(self.state.folder):
            self._show_message("Nothing to study yet", "This lecture has no transcript. Add the lecture PDF to study from its slides.", pdf=True)
            return
        self.regenerate(pdf=pdf)

    def regenerate(self, only: Optional[set[str]] = None, pdf: Optional[Path] = None) -> None:
        """Make the study material (again). ``only`` limits it to some parts; a new PDF rebuilds everything."""
        if self.state is None or (self._thread is not None and self._thread.isRunning()):
            return
        state, service = self.state, self.services.study
        pdf = pdf or self._pending_pdf
        self._pending_pdf = None
        self.stack.setCurrentIndex(PAGE_WORKING)
        self.working_title.setText("Preparing your study material ...")
        self._work_message = "Reading your lecture"
        self.working_text.setText("Reading your lecture")
        self.working_bar.setValue(0)
        self.cancel_button.setEnabled(True)

        def work(progress, cancelled):
            fresh = state
            if pdf is not None or state.index is None or only is None:
                progress({"percent": 2.0, "message": "Reading your lecture"})
                fresh = service.prepare(state.folder, pdf_path=pdf, title=state.title)
                fresh.progress = state.progress
            service.generate(fresh, progress=lambda f, m: progress({"percent": 5.0 + 95.0 * f, "message": m}), cancelled=cancelled, only=only)
            return fresh

        thread = TaskThread(work, self)
        thread.progressed.connect(self._work_progress)
        thread.succeeded.connect(self._work_done)
        thread.failed.connect(self._work_failed)
        self._threads.append(thread)
        self._thread = thread
        thread.start()

    def _work_progress(self, event: dict) -> None:
        percent = float(event.get("percent", 0.0))
        self.working_bar.setValue(int(percent * 10))
        if event.get("message"):
            self._work_message = str(event["message"])
        self.working_text.setText(f"{getattr(self, '_work_message', '')}  ·  {percent:.0f}%".strip(" ·"))

    def _work_done(self, state: StudyState) -> None:
        self.state = state
        self._show_tabs()

    def _work_failed(self, message: str) -> None:
        if message == "StudyCancelled":
            if self.state is not None and self.state.ready:
                self._show_tabs()
            else:
                self._show_message("Cancelled", "The study material was not made. You can try again whenever you like.")
            return
        log.warning("study material failed: %s", message)
        self._show_message("The study material could not be made", message)

    def _stop_running(self) -> None:
        thread = self._thread
        if thread is not None and thread.isRunning():
            for signal in (thread.succeeded, thread.failed, thread.progressed):
                try:
                    signal.disconnect()
                except (RuntimeError, TypeError):
                    pass
            thread.cancel()
            thread.wait(8000)

    def cancel(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            self._thread.cancel()
            self.cancel_button.setEnabled(False)
            self.working_text.setText("Stopping ...")

    # ------------------------------------------------------------------------------------------------------------------ display
    def _show_message(self, title: str, text: str, pdf: bool = False, translate: bool = False) -> None:
        self.message_title.setText(title)
        self.message_text.setText(text)
        self.retry_button.setVisible(not pdf and not translate)
        self.message_translate_button.setVisible(translate)
        self.message_pdf_button.setVisible(True)
        self.stack.setCurrentIndex(PAGE_MESSAGE)

    def _fill_languages(self, folder: Path) -> None:
        """The languages this lecture can be studied in (its spoken language and every saved translation); the picker is hidden when there is only one."""
        languages = self.services.study.study_languages(folder)
        current = self.services.study.study_language(folder)
        self.language_combo.blockSignals(True)
        self.language_combo.clear()
        for code in languages:
            self.language_combo.addItem(language_name(code), code)
        index = self.language_combo.findData(current)
        if index >= 0:
            self.language_combo.setCurrentIndex(index)
        self.language_combo.blockSignals(False)
        self.language_row.setVisible(len(languages) > 1)

    def _language_chosen(self, _index: int) -> None:
        state, code = self.state, str(self.language_combo.currentData() or "")
        if state is None or not code:
            return
        self.services.study.set_language(state.folder, code)
        self.open_lecture(state.folder, state.title, tab=self._pending_tab)

    def _engine_text(self, pack=None) -> str:
        """The student sees the name of the engine; which model it used and whether it fell back to the built-in method are Advanced diagnostics."""
        study = self.services.study
        if not self.services.settings.settings.show_advanced:
            return ENGINE_NAME
        return study.describe_pack(pack) if pack is not None else study.provider_summary()

    def _show_tabs(self) -> None:
        state = self.state
        assert state is not None and state.pack is not None and state.index is not None
        pack, progress = state.pack, state.progress
        evaluator = self.services.study.evaluator(state)
        provider_line = self._engine_text(pack)
        self.provider_label.setText(provider_line)
        self.overview.show_pack(pack, progress, provider_line, self.services.study.pdf_name(state.folder), notes=state.warnings)
        self.summary.show_pack(pack)
        self.concepts.show_pack(pack)
        self.flashcards.show_pack(pack, progress)
        self.questions.show_pack(pack, progress, evaluator)
        self.study_mode.show_pack(pack, progress, evaluator)
        self._refresh_stats(pack, progress)
        mode = {"pdf+transcript": "Transcript + slides", "transcript": "Transcript", "pdf": "Slides"}.get(pack.source_mode, pack.source_mode)
        self.lecture_meta.setText(f"{mode}  ·  {len(pack.topics)} topic{'s' if len(pack.topics) != 1 else ''}")
        self.stack.setCurrentIndex(PAGE_TABS)
        self.select_tab(self._pending_tab)

    def select_tab(self, key: str) -> None:
        if self.stack.currentIndex() == PAGE_TABS:
            self.tabs.setCurrentIndex(TAB_KEYS.get(key, 0))

    def _refresh_overview(self) -> None:
        state = self.state
        if state is not None and state.pack is not None:
            self.overview.show_pack(state.pack, state.progress, self.provider_label.text(), self.services.study.pdf_name(state.folder))
            self._refresh_stats(state.pack, state.progress)

    def _refresh_stats(self, pack, progress) -> None:
        concepts = len(pack.summaries.get("key_concepts", [])) + len(pack.summaries.get("definitions", []))
        counts = progress.counts(pack)
        mastered = counts["cards"]["mastered"] + counts["questions"]["mastered"]
        total = len(pack.flashcards) + len(pack.questions)
        self.stat_values["concepts"].setText(str(concepts))
        self.stat_values["flashcards"].setText(str(len(pack.flashcards)))
        self.stat_values["questions"].setText(str(len(pack.questions)))
        self.stat_values["mastery"].setText(f"{round(100 * mastered / total) if total else 0}%")

    # ------------------------------------------------------------------------------------------------------------------ ask / export
    def ask(self, question: str) -> None:
        """"Ask something about this lecture": answered from the lecture's passages only (the provider writes it, grounding checks it)."""
        state = self.state
        if state is None or state.index is None:
            return
        thread = TaskThread(lambda progress, cancelled: self.services.study.ask(state, question), self)
        thread.succeeded.connect(lambda item: self.overview.show_answer(item.text, item.sources))
        thread.failed.connect(lambda msg: self.overview.show_answer(f"The question could not be answered ({msg}).", []))
        self._threads.append(thread)
        thread.start()

    def export_guide(self) -> None:
        """A Markdown study guide and an Anki file, written next to the study files."""
        from ..study.export import write_exports

        state = self.state
        if state is None or state.pack is None:
            return
        try:
            written = write_exports(state.pack, state.folder / "study")
        except OSError as exc:
            self.overview.export_note.setText(f"The files could not be written ({exc.__class__.__name__}).")
            return
        self.overview.export_note.setText("Saved " + " and ".join(p.name for p in written) + f" in {written[0].parent}")
        self.services.open_path(written[0].parent)

    # ------------------------------------------------------------------------------------------------------------------ sources
    def show_source(self, chunk_id: str) -> None:
        """Open the passage with this id (a transcript passage with its times, or a slide with its page)."""
        state = self.state
        chunk = state.index.get(chunk_id) if state and state.index else None
        if chunk is None:
            return
        header = chunk.label() + (f" - {chunk.title}" if chunk.title else "") + (f" - {chunk.topic}" if chunk.topic else "")
        self.show_source_dialog(header, chunk.text)

    def _source_dialog(self, title: str, text: str) -> None:
        SourceDialog(title, text, self).exec()

    # ------------------------------------------------------------------------------------------------------------------ PDF
    def add_pdf(self) -> None:
        if self.state is None:
            return
        path = self.choose_pdf()
        if path:
            self.regenerate(pdf=Path(path))

    def _choose_pdf_dialog(self) -> str:
        start = str(self.state.folder) if self.state else ""
        path, _ = QFileDialog.getOpenFileName(self, "Select the lecture slides (PDF)", start, "PDF files (*.pdf)")
        return path
