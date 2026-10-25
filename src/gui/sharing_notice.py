"""The question asked when a student switches on the sharing of completed transcripts (Settings -> Privacy & Data). Plain words, no tricks: sharing is OFF until the student
says yes here; "Not now" (or closing the window) leaves it off."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QVBoxLayout

from . import brand, tokens

NOTICE_TITLE = "Share completed transcripts?"
NOTICE_HEADLINE = ("Share the text of your completed transcripts with MELETRA to help improve transcription quality? "
                   "Nothing is shared unless you say yes. You can turn it off again in Settings -> Privacy & Data.")
NOTICE_DETAILS = ("Your audio, video, PDF files and pictures are never sent. What would be shared, only after a transcript has been saved on your computer: the transcript text "
                  "(and the medical corrected transcript), the lecture name, the languages, the program version, a random installation number that identifies no person, "
                  "and the time. If the connection fails nothing is lost and nothing is shown: the program tries again later, and transcribing is never delayed. "
                  "Turning sharing off stops future uploads and deletes everything still waiting to be sent.")


class SharingNoticeDialog(QDialog):
    """``exec()`` returns accepted only when the student chose "Share transcripts"; the choice is then recorded (sharing on, notice agreed to)."""

    def __init__(self, services, parent=None) -> None:
        super().__init__(parent)
        self.services = services
        self.setWindowTitle(NOTICE_TITLE)
        self.setModal(True)
        self.setFixedWidth(560)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(tokens.SPACE["xl"], tokens.SPACE["xl"], tokens.SPACE["xl"], tokens.SPACE["lg"])
        layout.setSpacing(12)
        logo = QLabel()
        logo.setPixmap(brand.logo_pixmap(small=True, height=64))
        logo.setAlignment(Qt.AlignCenter)
        layout.addWidget(logo)
        self.headline = QLabel(NOTICE_HEADLINE)
        self.headline.setWordWrap(True)
        self.headline.setObjectName("sectionTitle")
        self.details = QLabel(NOTICE_DETAILS)
        self.details.setWordWrap(True)
        self.details.setObjectName("muted")
        layout.addWidget(self.headline)
        layout.addWidget(self.details)
        row = QHBoxLayout()
        row.addStretch(1)
        self.off_button = QPushButton("Not now")
        self.ok_button = QPushButton("Share transcripts")
        self.ok_button.setObjectName("primary")
        self.off_button.setDefault(True)                              # Enter answers "Not now": the safe answer
        row.addWidget(self.off_button)
        row.addWidget(self.ok_button)
        layout.addLayout(row)
        self.ok_button.clicked.connect(self._agree)
        self.off_button.clicked.connect(self.reject)

    def showEvent(self, event) -> None:  # noqa: N802
        """As tall as the wrapped text needs at this width: nothing is cut (a dialog does not compute height-for-width by itself)."""
        super().showEvent(event)
        self.setFixedHeight(self.layout().heightForWidth(self.width()))

    def _agree(self) -> None:
        self.services.sharing.set_enabled(True)
        self.services.sharing.notice_shown()
        self.accept()


CHOICE_TITLE = "Your transcripts and your privacy"
CHOICE_HEADLINE = ("MELETRA transcribes on this computer. Would you also like to share the TEXT of your completed transcripts with MELETRA, to help improve transcription quality?")
CHOICE_SHARE = "Share completed transcript text with MELETRA"
CHOICE_PRIVATE = "Keep transcripts private"


class PrivacyChoiceDialog(QDialog):
    """The question asked once, on the first launch. Two answers of equal weight; nothing is shared unless the student presses the sharing button. Enter, Esc and the X button all
    mean "Keep transcripts private" - the answer is recorded either way, so the question is not asked again (Settings -> Privacy & Data changes it at any time)."""

    def __init__(self, services, parent=None) -> None:
        super().__init__(parent)
        self.services = services
        self.answered = False
        self.setWindowTitle(CHOICE_TITLE)
        self.setModal(True)
        self.setFixedWidth(600)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(tokens.SPACE["xl"], tokens.SPACE["xl"], tokens.SPACE["xl"], tokens.SPACE["lg"])
        layout.setSpacing(12)
        logo = QLabel()
        logo.setPixmap(brand.logo_pixmap(small=True, height=64))
        logo.setAlignment(Qt.AlignCenter)
        layout.addWidget(logo)
        self.headline = QLabel(CHOICE_HEADLINE)
        self.headline.setWordWrap(True)
        self.headline.setObjectName("sectionTitle")
        self.details = QLabel(NOTICE_DETAILS + " Your audio and video are never uploaded by this choice, and you can look at exactly what would be shared before you decide: "
                              "Settings -> Privacy & Data -> Show what would be shared.")
        self.details.setWordWrap(True)
        self.details.setObjectName("muted")
        layout.addWidget(self.headline)
        layout.addWidget(self.details)
        row = QHBoxLayout()
        self.private_button = QPushButton(CHOICE_PRIVATE)
        self.share_button = QPushButton(CHOICE_SHARE)
        self.private_button.setDefault(True)
        for button in (self.private_button, self.share_button):
            button.setMinimumHeight(40)                                  # the same size and weight: neither answer is pushed
            row.addWidget(button, 1)
        layout.addLayout(row)
        self.private_button.clicked.connect(self._keep_private)
        self.share_button.clicked.connect(self._share)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self.setFixedHeight(self.layout().heightForWidth(self.width()))

    def _keep_private(self) -> None:
        self.answered = True
        self.services.sharing.choose(False)
        self.accept()

    def _share(self) -> None:
        self.answered = True
        self.services.sharing.choose(True)
        self.accept()

    def reject(self) -> None:                                           # Esc / the X button: no consent was given
        if not self.answered:
            self.answered = True
            self.services.sharing.choose(False)
        super().reject()


class SharingPreviewDialog(QDialog):
    """What would be shared, before anything is: the fields and their meaning, a sample of the text, and what is never sent. Looking shares nothing."""

    def __init__(self, preview: dict, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("What would be shared")
        self.setModal(True)
        self.resize(640, 560)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(tokens.SPACE["xl"], tokens.SPACE["lg"], tokens.SPACE["xl"], tokens.SPACE["lg"])
        layout.setSpacing(tokens.SPACE["md"])
        self.fields_label = QLabel("Only these fields are ever sent, after a transcript has been saved on your computer:\n" + "\n".join(f"  -  {key}: {meaning}" for key, meaning in preview["fields"]))
        self.fields_label.setWordWrap(True)
        layout.addWidget(self.fields_label)
        self.never_label = QLabel("Never sent: " + ", ".join(preview["never"]) + ".")
        self.never_label.setWordWrap(True)
        layout.addWidget(self.never_label)
        self.sample_label = QLabel(f"Your latest lecture would send {preview['bytes']:,} bytes. Start of its text:" if preview["bytes"] else
                                   "No finished lecture yet - there is nothing to show. Nothing has been shared.")
        self.sample_label.setObjectName("sectionTitle")
        layout.addWidget(self.sample_label)
        self.excerpt = QPlainTextEdit()
        self.excerpt.setReadOnly(True)
        self.excerpt.setPlainText(preview["excerpt"])
        layout.addWidget(self.excerpt, 1)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        layout.addWidget(close, 0, Qt.AlignRight)
