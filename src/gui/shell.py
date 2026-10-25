"""Persistent navigation shell: the MELETRA symbol and name on top, five pages, nothing else."""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import QButtonGroup, QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from . import brand, icons, tokens

RAIL_WIDTH = 208


class AppShell(QWidget):
    """A narrow navigation rail around the page stack. The privacy status lives in the Home header; Settings has exactly one entry: this rail."""

    navigate = Signal(str)

    ITEMS = (
        ("home", "Home"),
        ("transcription", "Transcription"),
        ("study", "Study"),
        ("files", "Files"),
        ("settings", "Settings"),
    )

    def __init__(self, content: QWidget, parent=None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.rail = QFrame()
        self.rail.setObjectName("sideRail")
        self.rail.setFixedWidth(RAIL_WIDTH)
        rail = QVBoxLayout(self.rail)
        rail.setContentsMargins(16, 24, 16, 24)
        rail.setSpacing(4)

        head = QHBoxLayout()
        head.setContentsMargins(4, 0, 0, 0)
        head.setSpacing(12)
        self.logo = QLabel()
        self.logo.setPixmap(brand.logo_pixmap(small=True, height=36))
        self.logo.setAccessibleName("MELETRA logo")
        self.brand = QLabel(self.wordmark_html())
        self.brand.setObjectName("brandName")
        self.brand.setTextFormat(Qt.RichText)
        self.brand.setAccessibleName("MELETRA")
        head.addWidget(self.logo)
        head.addWidget(self.brand, 1)
        rail.addLayout(head)
        rail.addSpacing(32)

        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.buttons: dict[str, QPushButton] = {}
        for key, text in self.ITEMS:
            button = QPushButton(text)
            button.setObjectName("navButton")
            button.setCheckable(True)
            button.setCursor(Qt.PointingHandCursor)
            button.setIconSize(QSize(20, 20))
            icons.bind(button, key)
            button.setAccessibleName(text)
            button.clicked.connect(lambda _checked=False, name=key: self.navigate.emit(name))
            self.group.addButton(button)
            self.buttons[key] = button
            rail.addWidget(button)
        rail.addStretch(1)

        layout.addWidget(self.rail)
        layout.addWidget(content, 1)
        self.set_current("home")

    def set_current(self, key: str) -> None:
        mapped = {"active": "transcription", "result": "transcription", "transcript": "transcription", "batch": "transcription"}
        button = self.buttons.get(mapped.get(key, key))
        if button is not None:
            button.setChecked(True)

    @staticmethod
    def wordmark_html() -> str:
        """The name in the brand's gold, letter-spaced (it follows the theme: the gold of the light theme is darker)."""
        return f'<span style="color:{tokens.color("gold")}; letter-spacing:3px">MELETRA</span>'

    def retheme(self) -> None:
        """The wordmark colours follow the theme (the icons are redrawn by ``icons.refresh``)."""
        self.brand.setText(self.wordmark_html())
