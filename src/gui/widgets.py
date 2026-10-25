"""Small reusable widgets."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import QMimeData, Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDragLeaveEvent, QDropEvent, QFont
from PySide6.QtWidgets import (
    QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

AUDIO_EXTENSIONS = (".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus")
VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".webm")
MEDIA_EXTENSIONS = AUDIO_EXTENSIONS + VIDEO_EXTENSIONS


def media_filter() -> str:
    patterns = " ".join(f"*{ext}" for ext in MEDIA_EXTENSIONS)
    return f"Audio and video ({patterns});;All files (*.*)"


def is_supported_media(path: Path) -> bool:
    return Path(path).suffix.lower() in MEDIA_EXTENSIONS


def first_media_path(mime: QMimeData) -> Optional[Path]:
    """The first dropped local file that is a supported audio/video type."""
    if not mime.hasUrls():
        return None
    for url in mime.urls():
        if url.isLocalFile() and is_supported_media(Path(url.toLocalFile())):
            return Path(url.toLocalFile())
    return None


def any_local_file(mime: QMimeData) -> Optional[Path]:
    if mime.hasUrls():
        for url in mime.urls():
            if url.isLocalFile():
                return Path(url.toLocalFile())
    return None


def caps(widget):
    """Show a label or button in capital letters without changing its text (``text()`` stays as written)."""
    font = QFont(widget.font())
    font.setCapitalization(QFont.AllUppercase)
    widget.setFont(font)
    return widget


def card() -> QFrame:
    frame = QFrame()
    frame.setObjectName("card")
    return frame


def format_clock(seconds: Optional[float]) -> str:
    if seconds is None or seconds < 0:
        return "--:--"
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def format_hms(seconds: Optional[float]) -> str:
    """Always HH:MM:SS (the duration of a recording)."""
    if seconds is None or seconds < 0:
        return "--:--:--"
    seconds = int(round(seconds))
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def format_size(nbytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024 or unit == "GB":
            return f"{nbytes:.0f} {unit}" if unit == "B" else f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} GB"


class DropZone(QLabel):
    """Drop an audio/video file here, or click to choose one. Unsupported files are refused with a clear message."""

    file_chosen = Signal(str)
    rejected = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("dropZone")
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setText("Drop audio or video here\n\nor click to select a file\n\n"
                     "mp3  wav  m4a  aac  flac  ogg  opus  -  mp4  mov  mkv  webm")
        self.setMinimumHeight(150)
        self.setCursor(Qt.PointingHandCursor)
        self.setProperty("hover", False)
        self.chooser: Callable[[], str] = self._default_chooser

    def _default_chooser(self) -> str:
        path, _ = QFileDialog.getOpenFileName(self, "Select an audio or video file", "", media_filter())
        return path

    def _set_hover(self, on: bool) -> None:
        self.setProperty("hover", on)
        self.style().unpolish(self)
        self.style().polish(self)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.LeftButton:
            path = self.chooser()
            if path:
                self._accept(Path(path))

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        if any_local_file(event.mimeData()) is not None:
            event.acceptProposedAction()
            self._set_hover(True)
        else:
            event.ignore()

    def dragLeaveEvent(self, event: QDragLeaveEvent) -> None:  # noqa: N802
        self._set_hover(False)

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        self._set_hover(False)
        path = first_media_path(event.mimeData())
        if path is None:
            other = any_local_file(event.mimeData())
            self.rejected.emit(f"'{other.name}' is not a supported audio or video file." if other else "Nothing usable was dropped.")
            event.ignore()
            return
        event.acceptProposedAction()
        self._accept(path)

    def _accept(self, path: Path) -> None:
        if not is_supported_media(path):
            self.rejected.emit(f"'{path.name}' is not a supported audio or video file.")
            return
        if not path.is_file():
            self.rejected.emit(f"'{path.name}' was not found.")
            return
        self.file_chosen.emit(str(path))


class StatCell(QWidget):
    """A small label over a value, for the active-job screen."""

    def __init__(self, caption: str, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.caption = QLabel(caption)
        self.caption.setObjectName("muted")
        self.value = QLabel("-")
        self.value.setObjectName("bigNumber")
        layout.addWidget(self.caption)
        layout.addWidget(self.value)

    def set(self, text: str) -> None:
        self.value.setText(text)


class StatGrid(QWidget):
    def __init__(self, names: list[str], columns: int = 4, parent=None) -> None:
        super().__init__(parent)
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(24)
        grid.setVerticalSpacing(12)
        self.cells: dict[str, StatCell] = {}
        for i, name in enumerate(names):
            cell = StatCell(name)
            self.cells[name] = cell
            grid.addWidget(cell, i // columns, i % columns)

    def set(self, name: str, text: str) -> None:
        self.cells[name].set(text)


class LogPanel(QPlainTextEdit):
    def __init__(self, parent=None, max_lines: int = 4000) -> None:
        super().__init__(parent)
        self.setObjectName("log")
        self.setReadOnly(True)
        self.setMaximumBlockCount(max_lines)
        self.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.setMinimumHeight(150)

    def add(self, level: str, text: str) -> None:
        prefix = "! " if level in ("WARNING", "ERROR", "CRITICAL") else "  "
        self.appendPlainText(f"{prefix}{text}")
        bar = self.verticalScrollBar()
        bar.setValue(bar.maximum())


def button_row(*buttons: QPushButton) -> QWidget:
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    for button in buttons:
        layout.addWidget(button)
    layout.addStretch(1)
    return row


def expanding(widget: QWidget) -> QWidget:
    widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    return widget
