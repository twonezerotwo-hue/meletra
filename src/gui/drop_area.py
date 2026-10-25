"""The big drop area of the start screen and the card that replaces it once a file is chosen.

The area paints itself (dashed soft border; teal border, a slightly raised surface and a faint glow while a file is dragged over; a gold ring for the keyboard focus) and fades
between these states in about 150 ms. The whole window accepts drops (see ``MainWindow``); a click, Enter or Space here, or the small "Browse files" button, opens the file dialog.
"""

from __future__ import annotations

from PySide6.QtCore import QEasingCurve, QRectF, QSize, Qt, QVariantAnimation, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QFileDialog, QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from ..ingest.router import file_dialog_filter
from . import icons, tokens
from .components import icon_button
from .tokens import RADIUS, TRANSITION_MS

DROP_TITLE = "Drop lecture files here"
DROP_SUB = "Audio · Video · PDF · Image"


def mix(a: str, b: str, t: float) -> QColor:
    """The colour ``t`` of the way from ``a`` to ``b`` (both ``#rrggbb``)."""
    t = min(max(t, 0.0), 1.0)
    ca, cb = QColor(a), QColor(b)
    return QColor(round(ca.red() + (cb.red() - ca.red()) * t), round(ca.green() + (cb.green() - ca.green()) * t),
                  round(ca.blue() + (cb.blue() - ca.blue()) * t))


class DropArea(QFrame):
    """Drop any number of supported files, or click / press Enter to choose them."""

    files_chosen = Signal(list)
    file_chosen = Signal(str)                       # the single-file signal of the 0.2 drop zone
    rejected = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("dropArea")
        self.choose_many = self._default_choose
        self.setFocusPolicy(Qt.NoFocus)                 # the keyboard goes to the Browse button inside it (one stop, a real button); the area itself takes the click and the drop
        self.setCursor(Qt.PointingHandCursor)
        self.setAcceptDrops(False)                  # the window takes every drop (one code path, one overlay)
        self.setAttribute(Qt.WA_Hover, True)
        self.setAccessibleName("Lecture files drop area")
        self.setAccessibleDescription("Drop audio, video, PDF or image files here or anywhere in the window")
        self.setToolTip("Click to select files, or drop them anywhere in this window")
        self._drag = 0.0
        self._hover = 0.0
        self._animation = QVariantAnimation(self)
        self._animation.setDuration(TRANSITION_MS)
        self._animation.setEasingCurve(QEasingCurve.OutCubic)
        self._animation.valueChanged.connect(self._tick)
        self._target = (0.0, 0.0)

        column = QVBoxLayout(self)
        column.setContentsMargins(24, 24, 24, 24)
        column.setSpacing(tokens.SPACE["sm"])
        column.addStretch(1)
        self.icon_label = QLabel()
        self.icon_label.setAlignment(Qt.AlignCenter)
        self.icon_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.title_label = QLabel(DROP_TITLE)
        self.title_label.setObjectName("dropTitle")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.sub_label = QLabel(DROP_SUB)
        self.sub_label.setObjectName("dropSub")
        self.sub_label.setAlignment(Qt.AlignCenter)
        self.sub_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.browse_button = QPushButton("Browse files")
        self.browse_button.setObjectName("dropBrowse")
        self.browse_button.setCursor(Qt.PointingHandCursor)
        self.browse_button.clicked.connect(self.choose)
        self.browse_button.setToolTip("Choose audio, video, PDF or image files on this computer")
        self.browse_button.setAccessibleName("Browse lecture files")
        self.browse_button.setAccessibleDescription("Choose audio, video, PDF or image files on this computer")
        column.addWidget(self.icon_label, 0, Qt.AlignHCenter)
        column.addSpacing(tokens.SPACE["sm"])
        column.addWidget(self.title_label)
        column.addWidget(self.sub_label)
        column.addSpacing(12)
        column.addWidget(self.browse_button, 0, Qt.AlignHCenter)
        column.addStretch(1)
        self.retheme()

    # ------------------------------------------------------------------------------------------------------------------------- text (compatible with the old drop zone)
    def text(self) -> str:
        return self.title_label.text()

    def setText(self, text: str) -> None:  # noqa: N802 - Qt-style name kept for the callers of the old drop zone
        self.title_label.setText(text)

    def retheme(self) -> None:
        self.icon_label.setPixmap(icons.badge_pixmap("upload", 64, 28))

    # ------------------------------------------------------------------------------------------------------------------------- choosing
    def _default_choose(self) -> list[str]:
        paths, _ = QFileDialog.getOpenFileNames(self, "Select files", "", file_dialog_filter())
        return list(paths)

    def choose(self) -> None:
        paths = self.choose_many()
        if paths:
            self.files_chosen.emit(list(paths))

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.LeftButton:
            self.choose()

    # ------------------------------------------------------------------------------------------------------------------------- states
    def set_drag_active(self, on: bool) -> None:
        """A file is being dragged over the window: teal border, raised surface, faint glow (fades in about 150 ms)."""
        self._go(1.0 if on else 0.0, self._hover)

    def drag_active(self) -> bool:
        return self._target[0] >= 0.5

    def enterEvent(self, event) -> None:  # noqa: N802
        self._go(self._target[0], 1.0)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._go(self._target[0], 0.0)
        super().leaveEvent(event)

    def _go(self, drag: float, hover: float) -> None:
        self._target = (drag, hover)
        self._animation.stop()
        start = (self._drag, self._hover)
        end = (drag, hover)
        if start == end:
            return
        self._animation.setStartValue(0.0)
        self._animation.setEndValue(1.0)
        self._from = start
        self._animation.start()

    def _tick(self, value) -> None:
        t = float(value)
        d0, h0 = getattr(self, "_from", (self._drag, self._hover))
        d1, h1 = self._target
        self._drag = d0 + (d1 - d0) * t
        self._hover = h0 + (h1 - h0) * t
        self.update()

    def focusInEvent(self, event) -> None:  # noqa: N802
        super().focusInEvent(event)
        self.update()

    def focusOutEvent(self, event) -> None:  # noqa: N802
        super().focusOutEvent(event)
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt API
        c = tokens.current()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        lift = max(self._drag, self._hover * 0.55)
        background = mix(c.surface, c.surface_hover, lift)
        if self._drag > 0:
            background = mix(background.name(), c.teal_soft, self._drag * 0.8)
        rect = QRectF(self.rect()).adjusted(1.5, 1.5, -1.5, -1.5)
        border = mix(c.border_strong, c.teal, max(self._drag, self._hover * 0.8))
        pen = QPen(border, 1.5 + 0.6 * self._drag)
        if self._drag < 0.5:
            pen.setStyle(Qt.CustomDashLine)
            pen.setDashPattern([5.0, 4.0])
        p.setPen(pen)
        p.setBrush(background)
        p.drawRoundedRect(rect, RADIUS["drop"], RADIUS["drop"])
        if self._drag > 0.02:                                                              # a faint glow just inside the border: rings that fade inwards
            for i in range(1, 6):
                ring = QColor(c.teal)
                ring.setAlphaF(0.22 * self._drag * (1 - i / 6))
                p.setPen(QPen(ring, 2))
                p.setBrush(Qt.NoBrush)
                p.drawRoundedRect(rect.adjusted(i * 2, i * 2, -i * 2, -i * 2), max(RADIUS["drop"] - i * 2, 4), max(RADIUS["drop"] - i * 2, 4))
        if self.hasFocus():                                                                  # keyboard focus: a solid gold ring, not only a colour change
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(QColor(c.gold), 2))
            p.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), RADIUS["drop"] + 3, RADIUS["drop"] + 3)
        p.end()


class SelectedFileCard(QFrame):
    """The chosen file(s): a type icon, the name, "Video · 45 min · 279 MB", and compact Replace / Remove actions."""

    replace_requested = Signal()
    remove_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("fileCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(tokens.SPACE["sm"])
        layout.addStretch(1)
        self.icon_label = QLabel()
        self.icon_label.setAlignment(Qt.AlignCenter)
        self.name = QLabel("")
        self.name.setObjectName("fileName")
        self.name.setWordWrap(True)
        self.name.setAlignment(Qt.AlignCenter)
        self.info = QLabel("")
        self.info.setObjectName("fileType")
        self.info.setWordWrap(True)
        self.info.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.icon_label, 0, Qt.AlignHCenter)
        layout.addSpacing(tokens.SPACE["sm"])
        layout.addWidget(self.name)
        layout.addWidget(self.info)
        layout.addSpacing(tokens.SPACE["lg"])
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addStretch(1)
        self.replace_button = QPushButton("Replace")
        self.remove_button = QPushButton("Remove")
        for button, icon_name in ((self.replace_button, "replace"), (self.remove_button, "close")):
            button.setObjectName("smallButton")
            button.setIconSize(QSize(16, 16))
            icons.bind(button, icon_name, size=16)
            button.setCursor(Qt.PointingHandCursor)
        self.replace_button.clicked.connect(self.replace_requested.emit)
        self.remove_button.clicked.connect(self.remove_requested.emit)
        row.addWidget(self.replace_button)
        row.addWidget(self.remove_button)
        row.addStretch(1)
        layout.addLayout(row)
        layout.addStretch(1)
        self.set_kind("video")

    def set_kind(self, icon_name: str) -> None:
        self._kind = icon_name
        self.icon_label.setPixmap(icons.badge_pixmap(icon_name, 56, 28))

    def retheme(self) -> None:
        self.set_kind(getattr(self, "_kind", "video"))
