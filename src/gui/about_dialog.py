"""About box: version, owner, privacy statement, licences of what is bundled."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QVBoxLayout

from .. import APP_NAME, APP_TAGLINE, COMPANY, __version__, build_info
from . import brand, tokens

SECTIONS = (
    ("Privacy",
     "Transcription, medical term correction, PDF reading and text recognition run locally, on this computer. Your audio, video, PDF files and pictures never "
     "leave it unless you switch on the optional Deepgram comparison, which uploads the audio to that service. Nothing else is shared: sending completed "
     "transcript text to MELETRA to improve transcription quality is OFF unless you switch it on in Settings -> Privacy & Data. The study material (summaries, "
     "flashcards, questions) is made on this computer; only if you choose a cloud AI in Settings -> Study is the lecture TEXT (never the audio) sent to that provider."),
    ("Built with",
     "OpenAI Whisper large-v3 (MIT licence) through faster-whisper and CTranslate2, PyAV / FFmpeg (LGPL), Silero VAD, Qt for Python (LGPL)."),
    ("Please note", "The transcript is a draft. Check drug names, doses and numbers against the source before relying on them."),
)
ABOUT_TEXT = APP_TAGLINE + ". Transcribes lecture recordings on this computer.\n\n" + "\n\n".join(f"{title}\n{text}" for title, text in SECTIONS)


class AboutDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"About {APP_NAME}")
        self.setFixedWidth(560)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(tokens.SPACE["xl"], tokens.SPACE["xl"], tokens.SPACE["xl"], tokens.SPACE["lg"])
        layout.setSpacing(tokens.SPACE["md"])
        head = QHBoxLayout()
        head.setSpacing(tokens.SPACE["lg"])
        icon = QLabel()
        icon.setPixmap(brand.logo_pixmap(small=True, height=84))
        head.addWidget(icon, 0, Qt.AlignTop)
        titles = QVBoxLayout()
        titles.setSpacing(tokens.SPACE["xs"])
        title = QLabel(APP_NAME)
        title.setObjectName("title")
        version = QLabel(f"Version {__version__}")
        version.setObjectName("statusLine")
        build = QLabel(f"{COMPANY}  ·  build {build_info.commit()}")
        build.setObjectName("meta")
        for widget in (title, version, build):
            titles.addWidget(widget)
        titles.addStretch(1)
        head.addLayout(titles, 1)
        layout.addLayout(head)
        intro = QLabel(APP_TAGLINE + ". Transcribes lecture recordings on this computer.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.sections = []
        for heading, text in SECTIONS:
            caption = QLabel(heading)
            caption.setObjectName("sectionTitle")
            body = QLabel(text)
            body.setObjectName("muted")
            body.setWordWrap(True)
            body.setTextInteractionFlags(Qt.TextSelectableByMouse)
            layout.addWidget(caption)
            layout.addWidget(body)
            self.sections.append((caption, body))
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def showEvent(self, event) -> None:  # noqa: N802
        """As tall as the wrapped text needs at this width: nothing is cut (a dialog does not compute height-for-width by itself)."""
        super().showEvent(event)
        self.setFixedHeight(self.layout().heightForWidth(self.width()))
