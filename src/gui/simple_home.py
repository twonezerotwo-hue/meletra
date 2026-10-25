"""The start screen: one workflow on the left - drop a lecture, (optionally) add its slides, choose the languages, TRANSCRIBE - and the recent lectures on the right.

All the real work (reading the file, checking the model and disk space, building the request) is done by the classic ``HomePage``, which
stays alive but is never shown in this mode; this page mirrors its state and forwards the student's clicks to it.
Dropping works anywhere in the window (see ``MainWindow`` and ``drop_overlay``); the drop area here is one more way in, and a click or Enter on it opens the file dialog.
Below about 980 px the recent lectures move under the workflow instead of squeezing it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (QBoxLayout, QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QProgressBar, QPushButton, QScrollArea, QSizePolicy,
                               QStackedWidget, QVBoxLayout, QWidget)

from .. import APP_NAME, APP_TAGLINE
from ..ingest.router import VIDEO_EXT, Detected, classify
from ..services.model_service import human_bytes
from . import friendly, icons, tokens
from .choices import INPUT_LANGUAGE_CHOICES, LANGUAGE_CHOICES, OUTPUT_LANGUAGE_CHOICES, index_of
from .components import EmptyState, icon_button
from .context import AppServices
from .drop_area import DROP_SUB, DROP_TITLE, DropArea, SelectedFileCard
from .home_page import HomePage
from .lecture_cards import CARD_HEIGHT_HINT, ElidedLabel, LectureActions, LectureCard
from .tasks import TaskThread

DROP_TEXT = DROP_TITLE
DROP_HINT = DROP_SUB
PDF_TIP = "Slides improve terminology recognition and study generation."
RECENT_MIN, RECENT_MAX = 5, 7               # how many recent lectures Home shows, by the height of the window
RECENT_COUNT = RECENT_MAX
DROP_MIN_HEIGHT = 300                       # about twice the 0.2 drop area: the dominant element of the left side
DROP_MAX_HEIGHT = 600                       # ... but never a huge empty field on a big screen
CONTENT_MAX_WIDTH = 1500
STACK_BELOW = 980                           # narrower than this the recent lectures go under the workflow
SIDE_WIDTH = (340, 420)
FileDropZone = DropArea                      # the name the tests and the 3.x code know


class SimpleHome(QWidget):
    privacy_requested = Signal()
    view_all_requested = Signal()
    convert_requested = Signal(list)            # [Detected] - several files, or documents / pictures (one recording takes the 0.2 path)
    lecture_requested = Signal(object)          # a JobRecord of the recent list was clicked (the window opens it according to its state)

    def __init__(self, services: AppServices, engine: HomePage, parent=None) -> None:
        super().__init__(parent)
        self.services = services
        self.engine = engine
        self._busy_text = ""
        self.recent_shown = RECENT_MIN
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setFocusPolicy(Qt.NoFocus)                      # the controls inside take the focus, not the scroll area
        outer.addWidget(self.scroll, 1)
        page = QWidget()
        page.setMaximumWidth(CONTENT_MAX_WIDTH)
        page.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        holder = QWidget()
        hl = QHBoxLayout(holder)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.addWidget(page, 100)                                              # left-aligned like every other page: the titles line up from page to page
        hl.addStretch(1)
        self.scroll.setWidget(holder)
        root = QVBoxLayout(page)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(tokens.SPACE["xl"])

        # ---------------------------------------------------------------------------------------------- header: name, subtitle, one small status chip
        top = QHBoxLayout()
        title_box = QVBoxLayout()
        title_box.setSpacing(tokens.SPACE["xs"])
        self.title = QLabel(APP_NAME)
        self.title.setObjectName("hero")
        self.subtitle = QLabel(APP_TAGLINE)
        self.subtitle.setObjectName("muted")
        title_box.addWidget(self.title)
        title_box.addWidget(self.subtitle)
        top.addLayout(title_box, 1)
        self.privacy = QPushButton("Local processing")
        self.privacy.setObjectName("privacyChip")
        self.privacy.setCursor(Qt.PointingHandCursor)
        self.privacy.setIconSize(QSize(10, 10))
        self.privacy.clicked.connect(self.privacy_requested.emit)
        top.addWidget(self.privacy, 0, Qt.AlignTop)
        self.header = QWidget()
        self.header.setLayout(top)
        top.setContentsMargins(0, 0, 0, 0)
        root.addWidget(self.header)

        self.content = QBoxLayout(QBoxLayout.LeftToRight)
        self.content.setSpacing(tokens.SPACE["xl"])
        main = QVBoxLayout()
        main.setSpacing(tokens.SPACE["lg"])
        main.setContentsMargins(0, 0, 0, 0)
        self.main_column = QWidget()
        self.main_column.setLayout(main)
        self.main_column.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.side = QWidget()
        side = QVBoxLayout(self.side)
        side.setContentsMargins(0, 0, 0, 0)
        side.setSpacing(tokens.SPACE["md"])
        self.side.setMinimumWidth(SIDE_WIDTH[0])
        self.side.setMaximumWidth(SIDE_WIDTH[1])
        self.side.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.content.addWidget(self.main_column, 1)
        self.content.addWidget(self.side, 0)
        root.addLayout(self.content, 1)

        # ---------------------------------------------------------------------------------------------- the speech model (only while it is missing)
        self.model_box = QFrame()
        self.model_box.setObjectName("card")
        mb = QVBoxLayout(self.model_box)
        mb.setContentsMargins(16, 16, 16, 16)
        self.model_text = QLabel("")
        self.model_text.setWordWrap(True)
        mb.addWidget(self.model_text)
        brow = QHBoxLayout()
        self.download_button = QPushButton("Download the speech model")
        self.download_button.setObjectName("primary")
        self.download_button.clicked.connect(self.engine.start_model_download)
        self.cancel_download_button = QPushButton("Cancel")
        self.cancel_download_button.clicked.connect(self.engine.cancel_model_download)
        self.cancel_download_button.hide()
        brow.addWidget(self.download_button)
        brow.addWidget(self.cancel_download_button)
        brow.addStretch(1)
        mb.addLayout(brow)
        self.download_bar = QProgressBar()
        self.download_bar.setRange(0, 1000)
        self.download_bar.hide()
        mb.addWidget(self.download_bar)
        self.download_text = QLabel("")
        self.download_text.setObjectName("muted")
        self.download_text.hide()
        mb.addWidget(self.download_text)
        main.addWidget(self.model_box)

        # ---------------------------------------------------------------------------------------------- 1. the file: the big drop area, or the card of the chosen file
        self.selection: list[Detected] = []                           # several files, or documents / pictures (not one plain recording)
        self._classifier: Optional[TaskThread] = None
        self._reading = False
        self._pending: Optional[list] = None
        self.drop_zone = DropArea()
        self.drop_zone.files_chosen.connect(self.set_files)
        self.drop_zone.file_chosen.connect(lambda path: self.set_files([path]))     # the single-file signal of the 0.2 drop zone
        self.drop_zone.rejected.connect(self.engine.show_problem)
        self.drop_hint = self.drop_zone.sub_label
        self.browse_button = self.drop_zone.browse_button
        self.file_card = SelectedFileCard()
        self.file_card.replace_requested.connect(self.drop_zone.choose)
        self.file_card.remove_requested.connect(self._new_transcription)
        self.file_name, self.file_info = self.file_card.name, self.file_card.info
        self.input_stack = QStackedWidget()
        self.input_stack.addWidget(self.drop_zone)
        self.input_stack.addWidget(self.file_card)
        self.input_stack.setMinimumHeight(DROP_MIN_HEIGHT)
        self.input_stack.setMaximumHeight(DROP_MAX_HEIGHT)
        self.input_stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main.addWidget(self.input_stack, 1)

        # ---------------------------------------------------------------------------------------------- 2. the slides (optional, visually secondary)
        pdf_row = QHBoxLayout()
        pdf_row.setSpacing(tokens.SPACE["sm"])
        pdf_row.setContentsMargins(4, 0, 0, 0)
        self.pdf_icon = QLabel()
        self.pdf_icon.setPixmap(icons.pixmap("pdf", tokens.color("text2"), 18))
        self.pdf_title = QLabel("Lecture material")
        self.pdf_title.setToolTip(PDF_TIP)
        self.pdf_optional = QLabel("optional")
        self.pdf_optional.setObjectName("meta")
        self.pdf_optional.setToolTip(PDF_TIP)
        self.pdf_button = QPushButton("+ Add PDF")
        self.pdf_button.setObjectName("smallButton")
        self.pdf_button.setCursor(Qt.PointingHandCursor)
        self.pdf_button.setToolTip(PDF_TIP)
        self.pdf_button.setAccessibleName("Add lecture PDF")
        self.pdf_button.setAccessibleDescription(PDF_TIP)
        self.pdf_button.clicked.connect(self.engine.add_pdf)
        self.pdf_label = ElidedLabel("")
        self.pdf_label.setObjectName("fileType")
        self.pdf_remove = icon_button("close", "Remove the attached PDF", "Remove the attached PDF", size=16)
        self.pdf_remove.clicked.connect(lambda: self.engine.set_pdf(None))
        pdf_row.addWidget(self.pdf_icon)
        pdf_row.addWidget(self.pdf_title)
        pdf_row.addWidget(self.pdf_optional)
        pdf_row.addSpacing(tokens.SPACE["sm"])
        pdf_row.addWidget(self.pdf_label, 1)
        pdf_row.addStretch(1)
        pdf_row.addWidget(self.pdf_remove)
        pdf_row.addWidget(self.pdf_button)
        self.pdf_row_widget = QWidget()
        self.pdf_row_widget.setLayout(pdf_row)
        main.addWidget(self.pdf_row_widget)

        # ---------------------------------------------------------------------------------------------- 3. the languages (per transcription, never hidden in Settings)
        grid = QGridLayout()
        grid.setHorizontalSpacing(tokens.SPACE["lg"])
        grid.setVerticalSpacing(tokens.SPACE["sm"])
        self.input_label = QLabel("Input Language")
        self.output_label = QLabel("Output Language")
        for label in (self.input_label, self.output_label):
            label.setObjectName("fieldLabel")
        self.input_combo = QComboBox()
        self.output_combo = QComboBox()
        for combo in (self.input_combo, self.output_combo):
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(12)
            combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        for label, code in INPUT_LANGUAGE_CHOICES:
            self.input_combo.addItem(label, code)
        for label, code in OUTPUT_LANGUAGE_CHOICES:
            self.output_combo.addItem(label, code)
        self.input_combo.setAccessibleName("Input language")                 # the screen reader adds the chosen value ("Auto Detect")
        self.input_combo.setAccessibleDescription("The language spoken in the recording")
        self.output_combo.setAccessibleName("Output language")
        self.output_combo.setAccessibleDescription("The language of the final transcript and the study material")
        self.input_combo.setToolTip("The language spoken in the recording. Auto Detect lets MELETRA decide.")
        self.output_combo.setToolTip("The language of the final transcript and the study material. The spoken-language transcript is always kept.")
        self.input_label.setBuddy(self.input_combo)
        self.output_label.setBuddy(self.output_combo)
        grid.addWidget(self.input_label, 0, 0)
        grid.addWidget(self.output_label, 0, 1)
        grid.addWidget(self.input_combo, 1, 0)
        grid.addWidget(self.output_combo, 1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        main.addLayout(grid)
        self.input_combo.currentIndexChanged.connect(self._input_language_changed)
        self.output_combo.currentIndexChanged.connect(self._output_language_changed)

        self.problem = QLabel("")
        self.problem.setObjectName("bannerBad")
        self.problem.setWordWrap(True)
        main.addWidget(self.problem)

        self.resume_box = QFrame()
        self.resume_box.setObjectName("card")
        rb = QVBoxLayout(self.resume_box)
        rb.setContentsMargins(16, 16, 16, 16)
        self.resume_text = QLabel("")
        self.resume_text.setWordWrap(True)
        rb.addWidget(self.resume_text)
        rrow = QHBoxLayout()
        self.resume_button = QPushButton("Continue where it stopped")
        self.resume_button.setObjectName("primary")
        self.resume_button.clicked.connect(lambda: self.engine.start(resume=True))
        self.start_over_button = QPushButton("Start over")
        self.start_over_button.clicked.connect(self.engine.start_over)
        rrow.addWidget(self.resume_button)
        rrow.addWidget(self.start_over_button)
        rrow.addStretch(1)
        rb.addLayout(rrow)
        main.addWidget(self.resume_box)

        # ---------------------------------------------------------------------------------------------- 4. the action: one strong button
        self.transcribe_button = QPushButton("TRANSCRIBE")
        self.transcribe_button.setObjectName("big")
        self.transcribe_button.setCursor(Qt.PointingHandCursor)
        self.transcribe_button.setToolTip("Transcribe the lecture on this computer")
        self.transcribe_button.clicked.connect(self._convert_clicked)
        main.addWidget(self.transcribe_button)
        main.addStretch(0)                                           # anything left over (a very tall window) stays below the button, never between the rows

        # ---------------------------------------------------------------------------------------------- recent lectures (the right side, secondary)
        self.recent_card = QWidget()
        recent = QVBoxLayout(self.recent_card)
        recent.setContentsMargins(0, 0, 0, 0)
        recent.setSpacing(tokens.SPACE["md"])
        head = QHBoxLayout()
        self.recent_title = QLabel("Recent lectures")
        self.recent_title.setObjectName("sectionTitle")
        self.view_all_button = QPushButton("View all")
        self.view_all_button.setObjectName("link")
        self.view_all_button.setLayoutDirection(Qt.RightToLeft)                  # the arrow comes after the words
        self.view_all_button.setIconSize(QSize(16, 16))
        icons.bind(self.view_all_button, "arrow_right", normal="teal", hover="teal_hover", selected="teal", size=16)
        self.view_all_button.setCursor(Qt.PointingHandCursor)
        self.view_all_button.setToolTip("Open the Files page: every lecture, searchable")
        self.view_all_button.clicked.connect(self.view_all_requested.emit)
        head.addWidget(self.recent_title, 1)
        head.addWidget(self.view_all_button)
        recent.addLayout(head)
        self.recent_holder = QVBoxLayout()
        self.recent_holder.setSpacing(tokens.SPACE["sm"])
        recent.addLayout(self.recent_holder)
        side.addWidget(self.recent_card)
        side.addStretch(1)
        self.actions = LectureActions(services, lambda record: self.lecture_requested.emit(record), self.refresh, self)

        engine.state_changed.connect(self.refresh)
        engine.download_progress.connect(self._download_progress)
        engine.download_done.connect(self._download_done)
        self.refresh()

    # ------------------------------------------------------------------------------------------------------------------ layout
    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        wide = self.width() >= STACK_BELOW
        direction = QBoxLayout.LeftToRight if wide else QBoxLayout.TopToBottom
        if self.content.direction() != direction:
            self.content.setDirection(direction)
            self.side.setMaximumWidth(SIDE_WIDTH[1] if wide else 16777215)
            self.side.setMinimumWidth(SIDE_WIDTH[0] if wide else 0)
            self.side.setSizePolicy(QSizePolicy.Preferred if wide else QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._fit_recent()

    def stacked(self) -> bool:
        return self.content.direction() == QBoxLayout.TopToBottom

    def _fit_recent(self) -> None:
        """5 to 7 recent lectures, by the height the window leaves for them (the cards are never squeezed)."""
        if self.stacked():
            wanted = RECENT_MIN
        else:
            free = self.height() - 52 - self.header.sizeHint().height() - tokens.SPACE["xl"] - 40          # window height - page margins - header - gap - the "Recent lectures" row
            wanted = max(RECENT_MIN, min(RECENT_MAX, int(free // (CARD_HEIGHT_HINT + tokens.SPACE["sm"]))))
        if wanted != self.recent_shown:
            self.recent_shown = wanted
            self._refresh_recent()

    def set_drag_active(self, on: bool) -> None:
        self.drop_zone.set_drag_active(on and self.input_stack.currentIndex() == 0)

    def retheme(self) -> None:
        self.drop_zone.retheme()
        self.file_card.retheme()
        self.pdf_icon.setPixmap(icons.pixmap("pdf", tokens.color("text2"), 18))
        self._apply_chip()

    def set_busy(self, text: str) -> None:
        """While a lecture is being transcribed the start button says so instead of inviting a second start ("" = idle)."""
        self._busy_text = text
        self.refresh()

    # ------------------------------------------------------------------------------------------------------------------ languages
    def _sync_languages(self) -> None:
        s = self.services.settings.settings
        spoken = self.engine.language_combo.currentData() or s.language
        if spoken == "mixed" and self.input_combo.findData("mixed") < 0:
            self.input_combo.addItem("Mixed languages (each sentence)", "mixed")
        for combo, code in ((self.input_combo, spoken), (self.output_combo, s.output_language)):
            combo.blockSignals(True)
            combo.setCurrentIndex(max(combo.findData(code), 0))
            combo.blockSignals(False)

    def _input_language_changed(self, _index: int) -> None:
        code = self.input_combo.currentData()
        self.engine.language_combo.setCurrentIndex(index_of(LANGUAGE_CHOICES, code))
        self.services.update_setting(language=code)

    def _output_language_changed(self, _index: int) -> None:
        self.services.update_setting(output_language=self.output_combo.currentData())

    # ------------------------------------------------------------------------------------------------------------------ the status chip
    def _apply_chip(self) -> None:
        s = self.services.settings.settings
        sharing = self.services.sharing
        if s.cloud_enabled:
            chip, role = "Cloud comparison on", "warning"
            tip = "Audio is sent to the cloud only when you start a cloud comparison. Click for Settings -> Privacy & Data."
        elif sharing.active():                                  # the same answer the uploader acts on: a stored wish that was never agreed to does not show as sharing
            chip, role = ("Sharing enabled", "gold") if sharing.configured() else ("Sharing on - not connected", "gold")
            tip = ("Transcription runs on this computer. Completed transcript text is shared with MELETRA because you switched that on. "
                   "Click to see what is shared and to turn it off." if sharing.configured() else
                   "Sharing is on, but this version has no upload address, so nothing is sent. Click for Settings -> Privacy & Data.")
        else:
            chip, role = "Local processing", "success"
            tip = "Transcription runs on this computer and nothing is shared. Click for Settings -> Privacy & Data."
        self.privacy.setText(chip)
        self.privacy.setToolTip(tip)
        self.privacy.setAccessibleName("Privacy status: " + chip)
        self.privacy.setAccessibleDescription("Opens the privacy settings")
        icons.bind(self.privacy, "dot", normal=role, hover=role, selected=role, size=10)

    # ------------------------------------------------------------------------------------------------------------------ mirroring
    def refresh(self) -> None:
        e = self.engine
        self._apply_chip()
        self._sync_languages()

        chosen = self.selection
        has_file = e.source is not None and not chosen
        self.input_stack.setCurrentIndex(1 if (has_file or chosen) else 0)
        if chosen:
            self.file_name.setText(chosen[0].path.name if len(chosen) == 1 else f"{len(chosen)} files")
            self.file_info.setText(chosen[0].label.replace(" • ", " · ") if len(chosen) == 1 else "\n".join(f"{d.path.name}  -  {d.label}" for d in chosen))
            self.file_card.set_kind(_kind_icon(chosen[0].kind) if len(chosen) == 1 else "files")
        elif has_file:
            self.file_name.setText(e.source.name)
            video = e.source.suffix.lower() in VIDEO_EXT
            kind = "Video" if video else "Audio"
            parts = [f"{kind} · {friendly.duration(e.duration)}" if e.duration else f"{kind} · reading ..."]
            try:
                parts.append(friendly.size(e.source.stat().st_size))
            except OSError:
                pass
            self.file_info.setText(" · ".join(parts))
            self.file_card.set_kind("video" if video else "audio")
        attached = e.pdf_path is not None and not chosen
        self.pdf_row_widget.setVisible(not chosen)
        self.pdf_title.setVisible(not attached)
        self.pdf_optional.setVisible(not attached)
        self.pdf_label.setVisible(attached)
        self.pdf_remove.setVisible(attached)
        self.pdf_button.setText("Change PDF" if e.pdf_path is not None else "+ Add PDF")
        self.pdf_label.setText(Path(e.pdf_path).name if e.pdf_path is not None else "")

        message = e.problem_label.text()
        self.problem.setText(message)
        self.problem.setVisible(bool(message) and not e.problem_label.isHidden())

        self._refresh_model()
        self.resume_box.setVisible(not e.resume_box.isHidden() and not chosen)
        if not e.resume_box.isHidden() and e.unfinished is not None:
            job = e.unfinished
            self.resume_text.setText(f"You already started this lecture ({job.progress:.0f}% done). Your progress was saved.")
        if self._busy_text:
            self.transcribe_button.setText(self._busy_text)
            self.transcribe_button.setEnabled(False)
        else:
            self.transcribe_button.setText("Preparing..." if self._reading else "TRANSCRIBE")
            self.transcribe_button.setEnabled(False if self._reading else (self._selection_ready() if chosen else e.start_button.isEnabled()))
        self._refresh_recent()

    def _new_transcription(self) -> None:
        """Remove the chosen file(s): back to the empty drop area."""
        self.clear_selection()
        self.engine.reset_file()
        self.refresh()
        self.drop_zone.setFocus()

    def _refresh_recent(self) -> None:
        while self.recent_holder.count():
            item = self.recent_holder.takeAt(0)
            old = item.widget()
            if old is not None:
                old.hide()
                old.setParent(None)                                                        # gone at once (deleteLater alone leaves it a child until the next event)
                old.deleteLater()
        records = self.services.jobs.archive(limit=self.recent_shown)
        total = len(self.services.jobs.archive())
        self.view_all_button.setVisible(total > len(records))
        if not records:
            self.recent_holder.addWidget(EmptyState("files", "No lectures yet", "Your transcribed lectures will appear here.", min_height=260))
            return
        previous = self.view_all_button if self.view_all_button.isVisibleTo(self) else self.transcribe_button
        for record in records:
            card = LectureCard(record, self.actions)
            card.clicked.connect(self.lecture_requested.emit)
            self.recent_holder.addWidget(card)
            QWidget.setTabOrder(previous, card)                      # Tab order: the workflow, then the recent lectures (card, its menu, next card ...), then the sidebar
            QWidget.setTabOrder(card, card.menu_button)
            previous = card.menu_button
        QWidget.setTabOrder(previous, self.privacy)                  # the small status chip comes last on the page

    # ------------------------------------------------------------------------------------------------------------------ choosing files
    @property
    def classifying(self) -> bool:
        """True from the moment files are handed in until the result has been applied to the screen (not just until the thread ends)."""
        return self._reading

    def set_files(self, paths: list) -> None:
        """Whatever was dropped or chosen. The files are read in a background thread (a slow drive or a huge video must never freeze
        the window); then ``_classified`` decides: one recording (optionally with its slides) takes the 0.2 path, the rest becomes a queue."""
        if self.classifying:
            self._pending = list(paths)                                 # the newest choice wins once the current one is read
            return
        self._reading = True
        thread = TaskThread(lambda progress, cancelled: classify(paths), self)
        thread.succeeded.connect(self._classified)
        thread.failed.connect(lambda message: self.engine.show_problem(f"The files could not be read ({message})."))
        thread.done.connect(self._classification_finished)
        self._classifier = thread
        self.transcribe_button.setEnabled(False)
        thread.start()

    def _classification_finished(self) -> None:
        if self.sender() is not self._classifier:
            return
        self._reading = False                                    # "done" is delivered after "succeeded": the result is on the screen
        pending, self._pending = self._pending, None
        if pending:
            self.set_files(pending)
        else:
            self.refresh()

    def _classified(self, selection) -> None:
        if not selection.items or selection.too_many:
            self.engine.show_problem(" ".join(selection.problems) or "Nothing usable was selected.")
            return
        media = [d for d in selection.items if d.is_media]
        pdfs = [d for d in selection.items if d.kind == "pdf"]
        if len(selection.items) == 1 and media:
            self.clear_selection()
            self.engine.set_file(str(media[0].path))
        elif len(selection.items) == 2 and len(media) == 1 and len(pdfs) == 1:        # a lecture and its slides
            self.clear_selection()
            self.engine.set_file(str(media[0].path))
            self.engine.set_pdf(pdfs[0].path)
        else:
            self.engine.reset_file()
            self.selection = selection.items
        if selection.problems:
            self.engine.show_problem(" ".join(selection.problems))
        self.refresh()

    def clear_selection(self) -> None:
        self.selection = []

    def _selection_ready(self) -> bool:
        if not any(d.is_media for d in self.selection):
            return True                                       # documents and pictures need no speech model
        key = self.engine.model_combo.currentData()
        return bool(key) and self.services.models.is_installed(key) and self.services.models.state(key).runtime_ok

    def _convert_clicked(self) -> None:
        if self.selection:
            self.convert_requested.emit(list(self.selection))
        else:
            self.engine.start()

    def _refresh_model(self) -> None:
        e = self.engine
        key = e.model_combo.currentData()
        if not key:
            self.model_box.hide()
            return
        state = self.services.models.state(key)
        info = self.services.models.info(key)
        busy = not self.cancel_download_button.isHidden()
        if state.installed and state.runtime_ok:
            self.model_box.setVisible(busy)
            return
        self.model_box.show()
        if state.installed:
            self.model_text.setText("The speech model is installed but cannot run on this computer. Open Settings -> Models for details.")
            self.download_button.hide()
        else:
            gb = info.approx_bytes / 1024 ** 3
            partial = f" ({human_bytes(state.partial_bytes)} already downloaded, it will continue)" if state.partial_bytes else ""
            self.model_text.setText(f"One-time setup: the speech model (about {gb:.1f} GB) has to be downloaded once. "
                                    f"After that the program works without internet.{partial}")
            self.download_button.setText("Continue download" if state.partial_bytes else "Download the speech model")
            self.download_button.setEnabled(not busy)
            self.download_button.show()

    def _download_progress(self, event: dict) -> None:
        self.cancel_download_button.show()
        self.download_bar.show()
        self.download_text.show()
        self.download_button.setEnabled(False)
        if event.get("message"):
            self.download_text.setText(event["message"])
            return
        percent = float(event.get("percent", 0.0))
        self.download_bar.setValue(int(percent * 10))
        eta = event.get("eta_seconds")
        self.download_text.setText(f"Downloading ... {percent:.0f}%" + (f"  -  about {max(int(eta // 60), 1)} min left" if eta else ""))

    def _download_done(self, error: str) -> None:
        self.cancel_download_button.hide()
        self.download_bar.hide()
        self.download_text.hide()
        self.download_button.setEnabled(True)
        self.refresh()

    def set_pdf_label_hint(self, text: str) -> None:
        self.pdf_label.setText(text)


def _kind_icon(kind: str) -> str:
    return {"video": "video", "audio": "audio", "pdf": "pdf", "image": "image"}.get(kind, "document")
