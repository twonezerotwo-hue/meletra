"""Shared building blocks of the screens, so Home, Files, Transcription, Study and Settings look and behave alike: page header, status chip, segmented control,
empty state, styled context menu."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import QButtonGroup, QDialog, QFrame, QHBoxLayout, QLabel, QMenu, QPushButton, QVBoxLayout, QWidget

from . import icons, tokens


def repolish(widget: QWidget) -> None:
    widget.style().unpolish(widget)
    widget.style().polish(widget)


def make_chip(text: str, state: str) -> QLabel:
    """A small status chip: completed (green), running / processing (teal), queued (neutral), paused (amber), failed (red)."""
    chip = QLabel(text)
    chip.setObjectName("statusChip")
    chip.setProperty("state", state)
    chip.setAlignment(Qt.AlignCenter)
    return chip


def set_chip(chip: QLabel, text: str, state: str) -> None:
    chip.setText(text)
    chip.setProperty("state", state)
    repolish(chip)


def make_badge(text: str, kind: str) -> QLabel:
    """A small badge: ``stable`` (teal), ``experimental`` (amber), ``on`` (green), ``off`` (neutral)."""
    badge = QLabel(text)
    badge.setObjectName("badge")
    badge.setProperty("kind", kind)
    badge.setAlignment(Qt.AlignCenter)
    return badge


class PageHeader(QWidget):
    """The top of a page: title (26-30 px, semibold), a muted subtitle, and an optional widget on the right."""

    def __init__(self, title: str, subtitle: str = "", right: Optional[QWidget] = None, parent=None) -> None:
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(16)
        box = QVBoxLayout()
        box.setSpacing(tokens.SPACE["xs"])
        self.title = QLabel(title)
        self.title.setObjectName("hero")
        self.subtitle = QLabel(subtitle)
        self.subtitle.setObjectName("muted")
        self.subtitle.setWordWrap(True)
        self.subtitle.setVisible(bool(subtitle))
        box.addWidget(self.title)
        box.addWidget(self.subtitle)
        row.addLayout(box, 1)
        if right is not None:
            row.addWidget(right, 0, Qt.AlignTop)


class SegmentedControl(QFrame):
    """Two to five mutually exclusive choices in one rounded bar (Original | Translated, All | Completed | Processing | Failed). Not two giant buttons."""

    changed = Signal(str)

    def __init__(self, items: list[tuple[str, str]], parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("segmented")
        self._row = QHBoxLayout(self)
        self._row.setContentsMargins(tokens.SPACE["xs"], tokens.SPACE["xs"], tokens.SPACE["xs"], tokens.SPACE["xs"])
        self._row.setSpacing(tokens.SPACE["xs"])
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.buttons: dict[str, QPushButton] = {}
        self.set_items(items)

    def set_items(self, items: list[tuple[str, str]], current: str = "") -> None:
        """Replace the choices (the transcript view gets one tab per saved translation); keeps ``current`` selected when it is still there, else the first."""
        for button in self.buttons.values():
            self.group.removeButton(button)
            self._row.removeWidget(button)
            button.deleteLater()
        self.buttons = {}
        for key, label in items:
            button = QPushButton(label)
            button.setObjectName("segment")
            button.setCheckable(True)
            button.setCursor(Qt.PointingHandCursor)
            button.clicked.connect(lambda _c=False, k=key: self.changed.emit(k))
            self.group.addButton(button)
            self.buttons[key] = button
            self._row.addWidget(button)
        if items:
            self.buttons[current if current in self.buttons else items[0][0]].setChecked(True)

    def current(self) -> str:
        for key, button in self.buttons.items():
            if button.isChecked():
                return key
        return ""

    def set_current(self, key: str, emit: bool = False) -> None:
        button = self.buttons.get(key)
        if button is not None:
            button.setChecked(True)
            if emit:
                self.changed.emit(key)

    def set_labels(self, labels: dict[str, str]) -> None:
        for key, text in labels.items():
            if key in self.buttons:
                self.buttons[key].setText(text)

    def set_visible_keys(self, keys: list[str]) -> None:
        for key, button in self.buttons.items():
            button.setVisible(key in keys)


class EmptyState(QFrame):
    """A calm empty state: an icon, a plain sentence, an optional hint. Never an error text, never a stack trace."""

    def __init__(self, icon_name: str, title: str, hint: str = "", parent=None, min_height: int = 200) -> None:
        super().__init__(parent)
        self.setObjectName("emptyState")
        column = QVBoxLayout(self)
        column.setContentsMargins(tokens.SPACE["xl"], tokens.SPACE["xl"], tokens.SPACE["xl"], tokens.SPACE["xl"])
        column.setSpacing(8)
        column.addStretch(1)
        self.icon = QLabel()
        self.icon.setAlignment(Qt.AlignCenter)
        self.icon.setPixmap(icons.pixmap(icon_name, _muted(), 36))
        self.title = QLabel(title)
        self.title.setObjectName("sectionTitle")
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setWordWrap(True)
        self.hint = QLabel(hint)
        self.hint.setObjectName("muted")
        self.hint.setAlignment(Qt.AlignCenter)
        self.hint.setWordWrap(True)
        self.hint.setVisible(bool(hint))
        column.addWidget(self.icon)
        column.addWidget(self.title)
        column.addWidget(self.hint)
        column.addStretch(1)
        self.setMinimumHeight(min_height)
        self._icon_name = icon_name

    def set_text(self, title: str, hint: str = "") -> None:
        self.title.setText(title)
        self.hint.setText(hint)
        self.hint.setVisible(bool(hint))


def _muted() -> str:
    from . import tokens

    return tokens.color("text3")


class StyledMenu(QMenu):
    """The one context-menu style of the program: rounded, on the surface colour, icons on the left."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowFlag(Qt.NoDropShadowWindowHint, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

    def add(self, text: str, icon_name: str, callback=None, enabled: bool = True):
        action = self.addAction(icons.icon(icon_name, normal="text2", hover="text", size=18), text)
        action.setEnabled(enabled)
        if callback is not None:
            action.triggered.connect(callback)
        return action


def icon_button(icon_name: str, tooltip: str, accessible: str = "", size: int = 18) -> QPushButton:
    """A small borderless icon button (remove, replace ...)."""
    button = QPushButton()
    button.setObjectName("iconButton")
    button.setCursor(Qt.PointingHandCursor)
    button.setIconSize(QSize(size, size))
    icons.bind(button, icon_name, size=size)
    button.setToolTip(tooltip)
    button.setAccessibleName(accessible or tooltip)
    return button


class Tile(QPushButton):
    """A clickable card with an icon, a title and one status line. ``setText("Title\nStatus line")`` keeps working; ``text()`` returns it unchanged."""

    def __init__(self, icon_name: str, text: str = "", parent=None) -> None:
        super().__init__("", parent)
        self.setObjectName("tile")
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(72)
        row = QHBoxLayout(self)
        row.setContentsMargins(tokens.SPACE["lg"], tokens.SPACE["md"], tokens.SPACE["lg"], tokens.SPACE["md"])
        row.setSpacing(12)
        self.icon_label = QLabel()
        self.icon_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._icon_name = icon_name
        column = QVBoxLayout()
        column.setSpacing(tokens.SPACE["xs"])
        self.title_label = QLabel()
        self.title_label.setObjectName("tileTitle")
        self.sub_label = QLabel()
        self.sub_label.setObjectName("tileSub")
        self.sub_label.setWordWrap(True)
        for label in (self.title_label, self.sub_label):
            label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            column.addWidget(label)
        row.addWidget(self.icon_label)
        row.addLayout(column, 1)
        self._full = ""
        self.setText(text)
        self.retheme()

    def retheme(self) -> None:
        self.icon_label.setPixmap(icons.badge_pixmap(self._icon_name, 40, 20, background="teal_soft", foreground="teal"))

    def setText(self, text: str) -> None:  # noqa: N802 - Qt API
        self._full = text
        title, _, rest = text.partition("\n")
        self.title_label.setText(title)
        self.sub_label.setText(rest)
        self.sub_label.setVisible(bool(rest))
        self.setAccessibleName(text.replace("\n", ". "))

    def text(self) -> str:
        return self._full


def back_button(text: str = "Back") -> QPushButton:
    """The same small "< Back" button on every page that has one."""
    button = QPushButton(text)
    button.setObjectName("smallButton")
    button.setIconSize(QSize(16, 16))
    icons.bind(button, "back", size=16)
    button.setCursor(Qt.PointingHandCursor)
    return button


def name_clear_button(edit, name: str = "Clear") -> None:
    """The little x of a search field is a button without text: give it a name a screen reader can say."""
    from PySide6.QtWidgets import QToolButton

    for button in edit.findChildren(QToolButton):
        button.setAccessibleName(name)
        button.setToolTip(name)


READING_WIDTH = 570                  # the text column: 70-90 characters of body text per line at the reading size
CARD_WIDTH = 760


def centered(widget: QWidget, max_width: int = READING_WIDTH) -> QWidget:
    """``widget`` in a column of at most ``max_width`` px, centred in the space it is given (long lines are hard to read; short ones are never stretched)."""
    holder = QWidget()
    row = QHBoxLayout(holder)
    row.setContentsMargins(0, 0, 0, 0)
    row.addStretch(1)
    widget.setMaximumWidth(max_width)
    row.addWidget(widget, 100)
    row.addStretch(1)
    return holder


class ConfirmDialog(QDialog):
    """A small question, the same everywhere: a title, one or two sentences, Cancel on the left and the action (named by what it does) on the right."""

    WIDTH = 440

    def __init__(self, parent: Optional[QWidget], title: str, text: str, accept: str = "OK", reject: str = "Cancel", default_accept: bool = False) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setFixedWidth(self.WIDTH)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(tokens.SPACE["xl"], tokens.SPACE["xl"], tokens.SPACE["xl"], tokens.SPACE["lg"])
        layout.setSpacing(tokens.SPACE["md"])
        heading = QLabel(title)
        heading.setObjectName("sectionTitle")
        body = QLabel(text)
        body.setWordWrap(True)
        body.setObjectName("muted")
        layout.addWidget(heading)
        layout.addWidget(body)
        layout.addSpacing(tokens.SPACE["sm"])
        row = QHBoxLayout()
        row.setSpacing(tokens.SPACE["sm"])
        self.reject_button = QPushButton(reject)
        self.accept_button = QPushButton(accept)
        self.accept_button.setObjectName("primary")
        self.reject_button.clicked.connect(self.reject)
        self.accept_button.clicked.connect(self.accept)
        row.addStretch(1)
        row.addWidget(self.reject_button)
        row.addWidget(self.accept_button)
        layout.addLayout(row)
        (self.accept_button if default_accept else self.reject_button).setDefault(True)      # the safe answer is the one Enter gives, unless the question is harmless


def ask(parent: Optional[QWidget], title: str, text: str, accept: str = "OK", reject: str = "Cancel", default_accept: bool = False) -> bool:
    return ConfirmDialog(parent, title, text, accept, reject, default_accept).exec() == QDialog.Accepted
