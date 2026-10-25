"""While a lecture is being transcribed (simple view): a title, a progress bar, the percent, the time left, PAUSE.

Chunks, speed factor, GPU memory, devices and the live log are still collected by the classic active page (kept alive behind this one)
and are available under "Technical details".
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout, QWidget

from . import friendly, icons, tokens
from .lecture_cards import ElidedLabel


class SimpleActive(QWidget):
    pause_requested = Signal()
    resume_requested = Signal()
    cancel_requested = Signal()
    back_requested = Signal()
    details_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.addStretch(1)
        column = QWidget()
        column.setMaximumWidth(560)                                # one compact block, centred: nothing to look for around it
        column_layout = QVBoxLayout(column)
        column_layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(column, 100)
        row.addStretch(1)
        outer.addLayout(row)
        outer.addStretch(1)

        panel = QFrame()
        panel.setObjectName("card")
        column_layout.addWidget(panel)
        root = QVBoxLayout(panel)
        root.setContentsMargins(tokens.SPACE["xl"], tokens.SPACE["xl"], tokens.SPACE["xl"], tokens.SPACE["lg"])
        root.setSpacing(tokens.SPACE["sm"])

        self.title = QLabel("Transcribing...")                     # the stage: "Preparing video...", "Transcribing...", "Finalizing..."
        self.title.setObjectName("title")
        root.addWidget(self.title)

        info = QHBoxLayout()                                       # which lecture
        info.setSpacing(tokens.SPACE["sm"])
        self.file_icon = QLabel()
        self.file_icon.setPixmap(icons.pixmap("video", tokens.color("text3"), 16))
        self.file_label = ElidedLabel("")
        self.file_label.setObjectName("fileName")
        info.addWidget(self.file_icon)
        info.addWidget(self.file_label, 1)
        root.addLayout(info)
        root.addSpacing(tokens.SPACE["sm"])

        bar = QHBoxLayout()                                        # progress bar and its percent on one line
        bar.setSpacing(tokens.SPACE["md"])
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setTextVisible(False)
        self.percent_label = QLabel("0%")
        self.percent_label.setObjectName("percent")
        self.percent_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        bar.addWidget(self.progress_bar, 1)
        bar.addWidget(self.percent_label)
        root.addLayout(bar)

        state = QHBoxLayout()                                      # what is happening now, and the time left
        state.setSpacing(tokens.SPACE["md"])
        self.stage_label = QLabel("")                              # also before the first words exist
        self.stage_label.setObjectName("statusLine")
        self.stage_label.setWordWrap(True)
        self.eta_label = QLabel("")
        self.eta_label.setObjectName("meta")
        self.eta_label.setAlignment(Qt.AlignRight | Qt.AlignTop)
        state.addWidget(self.stage_label, 1)
        state.addWidget(self.eta_label)
        root.addLayout(state)
        self.kind = "audio"

        self.note = QLabel("")
        self.note.setWordWrap(True)
        self.note.setObjectName("bannerWarn")
        self.note.hide()
        root.addWidget(self.note)

        buttons = QHBoxLayout()
        buttons.setSpacing(tokens.SPACE["sm"])
        self.pause_button = QPushButton("Pause")
        self.pause_button.setFocusPolicy(Qt.ClickFocus)           # a stray Space / Enter in another program must never pause the job
        self.pause_button.clicked.connect(self._pause_clicked)
        self.resume_button = QPushButton("Resume")
        self.resume_button.setObjectName("primary")
        self.resume_button.clicked.connect(self.resume_requested.emit)
        self.back_button = QPushButton("Back to start")
        self.back_button.clicked.connect(self.back_requested.emit)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setFocusPolicy(Qt.ClickFocus)
        self.cancel_button.clicked.connect(self._cancel_clicked)
        buttons.addWidget(self.pause_button)
        buttons.addWidget(self.resume_button)
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.back_button)
        buttons.addStretch(1)
        root.addSpacing(tokens.SPACE["sm"])
        root.addLayout(buttons)

        foot = QHBoxLayout()
        self.facts_label = QLabel("")
        self.facts_label.setObjectName("meta")
        self.details_button = QPushButton("Technical details")
        self.details_button.setObjectName("link")
        self.details_button.clicked.connect(self.details_requested.emit)
        foot.addWidget(self.facts_label, 1)
        foot.addWidget(self.details_button)
        root.addLayout(foot)
        self.language = "-"
        self.subject = "-"

    # ------------------------------------------------------------------------------------------------------------------ lifecycle
    def begin(self, file_name: str, language: str = "auto", subject: str = "auto", kind: str = "audio") -> None:
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()                                           # keyboard focus rests on the page, not on a button
        self.kind = kind
        title, line = friendly.stage_texts("probing", kind)       # never a bare "0%": the stage is named from the first second on
        self.title.setText(title)
        self.stage_label.setText(line)
        self.file_label.setText(file_name)
        self.file_icon.setPixmap(icons.pixmap("video" if kind == "video" else "audio", tokens.color("text3"), 16))
        self._busy()
        self.eta_label.setText("")                                # no time estimate until real speed data exists
        self.note.hide()
        self.language, self.subject = friendly.language_name(language), friendly.subject_name(subject)
        self._show_facts()
        self.pause_button.setEnabled(True)
        self.pause_button.setText("Pause")
        self.pause_button.show()
        self.cancel_button.show()
        self.cancel_button.setEnabled(True)
        self.resume_button.hide()
        self.resume_button.setEnabled(True)
        self.back_button.hide()

    def set_facts(self, facts: dict[str, Any]) -> None:
        """The worker reports what it detected (language, subject) once it knows."""
        if facts.get("language"):
            self.language = friendly.language_name(facts["language"])
        if facts.get("subject"):
            self.subject = friendly.subject_name(facts["subject"])
        self._show_facts()

    def _show_facts(self) -> None:
        self.facts_label.setText(f"Language: {self.language}     Subject: {self.subject}")

    def _busy(self) -> None:
        """Work that cannot be measured yet: an animated bar and no percent, instead of a 0% that looks frozen."""
        self.progress_bar.setRange(0, 0)
        self.percent_label.setText("")

    def set_stage(self, event: dict[str, Any]) -> None:
        """The job reports what it is doing before the first words exist (reading, extracting audio, detecting the language ...)."""
        title, line = friendly.stage_texts(str(event.get("stage", "")), self.kind)
        self.title.setText(title)
        percent = event.get("percent")
        if event.get("stage") == "language" and event.get("total"):
            line = f"{line} ({int(event.get('done', 0))} of {int(event['total'])})"
        self.stage_label.setText(line)
        if percent is None:
            self._busy()
        else:
            self.progress_bar.setRange(0, 1000)
            percent = min(max(float(percent), 0.0), 100.0)
            self.progress_bar.setValue(int(percent * 10))
            self.percent_label.setText(f"{percent:.0f}%")
        self.eta_label.setText("")

    def set_progress(self, event: dict[str, Any]) -> None:
        percent = min(max(float(event.get("percent", 0.0)), 0.0), 100.0)
        if self.progress_bar.maximum() == 0:                      # the first real transcript progress ends the busy animation
            self.progress_bar.setRange(0, 1000)
        self.title.setText("Transcribing...")
        self.stage_label.setText("")
        self.progress_bar.setValue(int(percent * 10))
        self.percent_label.setText(f"{percent:.0f}%")
        rtf = float(event.get("rtf") or 0.0)
        eta = event.get("eta")
        self.eta_label.setText(friendly.time_left(eta if rtf > 0 and eta is not None else None))

    def show_warning(self, text: str) -> None:
        """A non-fatal warning of the job, in plain words (technical ones are only in Technical details)."""
        self.show_note(friendly.simple_warning(text))

    def show_note(self, text: str) -> None:
        if text:
            self.note.setText(text)
            self.note.show()

    def _pause_clicked(self) -> None:
        self.pause_button.setEnabled(False)
        self.pause_button.setText("Pausing ...")
        self.pause_requested.emit()

    def _cancel_clicked(self) -> None:
        self.cancel_button.setEnabled(False)
        self.pause_button.setEnabled(False)
        self.cancel_requested.emit()

    def finished_paused(self) -> None:
        self.title.setText("Paused")
        self.stage_label.setText("")
        self.progress_bar.setRange(0, 1000)
        self.pause_button.hide()
        self.cancel_button.hide()
        self.resume_button.show()
        self.resume_button.setEnabled(False)               # the window enables it once the worker process has really ended
        self.back_button.show()
        self.eta_label.setText("Your progress is saved. Press Resume to continue.")

    def finished_failed(self, message: str) -> None:
        self.title.setText("The transcription stopped")
        self.stage_label.setText("")
        self.progress_bar.setRange(0, 1000)
        self.pause_button.hide()
        self.cancel_button.hide()
        self.resume_button.show()
        self.resume_button.setEnabled(False)
        self.resume_button.setText("Try again")
        self.back_button.show()
        self.eta_label.setText("Your progress is saved, nothing was lost.")
        self.show_note(friendly.plain_failure(message))
