"""The whole-window drop: a dimming overlay with "Drop lecture files / Release to add to transcription" while a supported file is dragged over ANY part of the window,
and a small error toast for files that are not supported."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QLabel, QWidget

from ..ingest.router import ALL_EXT
from . import brand, tokens


def supported_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() in ALL_EXT


def split_supported(paths: Iterable[str]) -> tuple[list[str], list[str]]:
    """(supported, unsupported) by extension; a folder or a file without extension is unsupported. The content is checked later by the file router."""
    good, bad = [], []
    for path in paths:
        (good if supported_path(path) else bad).append(path)
    return good, bad


def unsupported_message(bad: list[str]) -> str:
    names = ", ".join(f"'{Path(p).name}'" for p in bad[:2]) + (f" and {len(bad) - 2} more" if len(bad) > 2 else "")
    return f"{names} {'is' if len(bad) == 1 else 'are'} not supported. Use audio, video, PDF, picture or text files."


class DropOverlay(QWidget):
    """Covers the window, lets every mouse / drag event through to the window underneath (it only paints): the UI is dimmed and a panel, centred on the page, shows the
    symbol, what to do ("Drop lecture files") and what happens ("Release to add to transcription")."""

    TEXT = "Drop lecture files"
    SUB = "Release to add to transcription"

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.inset_left = 0                                                          # the navigation rail: the panel is centred over the page next to it
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.hide()

    def show_over(self) -> None:
        self.setGeometry(self.parentWidget().rect())
        self.raise_()
        self.show()

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt API
        c = tokens.current()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        scrim = QColor(c.bg)
        scrim.setAlpha(175)
        painter.fillRect(self.rect(), scrim)                                         # dims the window, so the panel is what the eye finds
        area = QRectF(self.inset_left, 0, max(self.width() - self.inset_left, 1), self.height())
        box_w, box_h = min(area.width() * 0.5, 480), 224
        box = QRectF(area.center().x() - box_w / 2, area.center().y() - box_h / 2, box_w, box_h)
        painter.setBrush(QColor(c.surface))
        painter.setPen(QPen(QColor(c.teal), 2))
        painter.drawRoundedRect(box, tokens.RADIUS["drop"], tokens.RADIUS["drop"])
        symbol = brand.logo_pixmap(small=True, height=88)
        if not symbol.isNull():
            painter.drawPixmap(int(box.center().x() - symbol.width() / 2), int(box.top() + tokens.SPACE["xl"]), symbol)
        title_font = QFont(self.font())
        title_font.setPixelSize(tokens.FONT_PX["lead"])
        title_font.setWeight(QFont.DemiBold)
        painter.setFont(title_font)
        painter.setPen(QColor(c.text))
        painter.drawText(QRectF(box.left(), box.top() + 124, box.width(), 32), Qt.AlignCenter, self.TEXT)
        sub_font = QFont(self.font())
        sub_font.setPixelSize(tokens.FONT_PX["body"])
        painter.setFont(sub_font)
        painter.setPen(QColor(c.text2))
        painter.drawText(QRectF(box.left(), box.top() + 160, box.width(), 24), Qt.AlignCenter, self.SUB)
        painter.end()


class Toast(QLabel):
    """A small notice at the bottom of the window that goes away by itself: red edge for a problem, teal for information."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("toast")
        self.setWordWrap(True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.hide()
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.hide)
        self.inset_left = 0                                      # the width of the navigation rail: notices are centred over the page, at the top, clear of the main button

    def show_message(self, text: str, milliseconds: int = 6000, kind: str = "error") -> None:
        self.setText(text)
        self.setProperty("kind", kind)
        self.style().unpolish(self)
        self.style().polish(self)
        self.reposition()
        self.raise_()
        self.show()
        self._timer.start(milliseconds)

    def reposition(self) -> None:
        """Top centre of the page (called again when the window is resized or first shown)."""
        parent = self.parentWidget()
        area = max(parent.width() - self.inset_left, 320)
        width = min(max(area - 80, 300), 560)
        self.setFixedWidth(width)
        self.adjustSize()
        self.move(self.inset_left + (area - self.width()) // 2, 18)
