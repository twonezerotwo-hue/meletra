"""The lecture card (Recent lectures on Home, the list on Files) and what its menu does: open, rename (in place), show files, remove from the archive.

A card reads like an archive entry: the name, the original file name only when the lecture was renamed, "Auto (English) -> Turkish", "45 min · Oct 16", a status chip and a ... menu.
The whole card is clickable (Enter / Space too). Renaming never touches the recording: only the name MELETRA shows.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QFontMetrics, QPainter
from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QLineEdit, QSizePolicy, QToolButton, QVBoxLayout, QWidget

from ..archive.resolver import quick_status, status_kind
from ..language_catalogue import language_name
from ..services.job_service import JobRecord, lecture_title
from . import friendly, icons, tokens
from .components import StyledMenu, ask, make_chip, repolish, set_chip

STATUS_TEXT = {"completed": "Completed", "running": "Processing", "queued": "Queued", "paused": "Paused", "failed": "Failed"}
CARD_HEIGHT_HINT = 98                                           # a card with the language line and the date line; one with a file-name line is about 16 px taller


class ElidedLabel(QLabel):
    """A one-line label that shows "..." instead of making its card wider; the full text is the tooltip."""

    def __init__(self, text: str = "", parent=None) -> None:
        super().__init__(text, parent)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.setMinimumWidth(24)

    def setText(self, text: str) -> None:  # noqa: N802 - Qt API
        super().setText(text)
        self.setToolTip(text)

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return QSize(24, super().sizeHint().height())

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(160, super().sizeHint().height())

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setFont(self.font())
        painter.setPen(self.palette().color(self.foregroundRole()))
        text = QFontMetrics(self.font()).elidedText(self.text(), Qt.ElideRight, self.width())
        painter.drawText(self.rect(), int(self.alignment()) | Qt.AlignVCenter, text)
        painter.end()


def language_path(record: JobRecord) -> str:
    """"Auto (English) -> Turkish": what was spoken (what the student chose, and what Whisper heard when it was Auto) and, when it differs, the Output Language."""
    spoken = record.detected_language or (record.language if record.language not in ("", "auto", "mixed") else "")
    heard = list(record.detected_languages or [])
    spoken_name = " · ".join(language_name(c) for c in heard) if len(heard) > 1 else (language_name(spoken) if spoken else "")
    if record.language in ("", "auto"):
        source = f"Auto ({spoken_name})" if spoken_name else "Auto"
    elif record.language == "mixed":
        source = "Mixed"
    else:
        source = language_name(record.language)
    made = [c for c in (record.translations or []) if len(heard) > 1 or c != spoken]            # a mixed lecture can be translated into any of its languages; a one-language lecture is not 'translated' into itself
    if made:                                                   # translations that exist as saved files (4.1.1): all of them are named
        return f"{source} → {', '.join(language_name(c) for c in made)}"
    out = record.output_language
    if out and spoken and out != spoken and record.translation not in ("model_missing", "failed", ""):
        return f"{source} → {language_name(out)}"
    if out and spoken and out != spoken:                       # asked for, not made (the translation model is not installed, or it failed): say so
        return f"{source} → {language_name(out)} (not translated)"
    return source


def short_date(stamp: str) -> str:
    """"Oct 16" (with the year when it is not this year)."""
    try:
        when = datetime.strptime(stamp[:16], "%Y-%m-%dT%H:%M")
    except ValueError:
        return ""
    text = f"{when:%b} {when.day}"
    return text if when.year == datetime.now().year else f"{text}, {when.year}"


def when_line(record: JobRecord) -> str:
    parts = []
    if record.duration_seconds:
        parts.append(friendly.duration(record.duration_seconds))
    parts.append(short_date(record.updated or record.created))
    return " · ".join(p for p in parts if p)


def meta_line(record: JobRecord) -> str:
    return " · ".join(p for p in (language_path(record), when_line(record)) if p)


def status_text(record: JobRecord) -> str:
    """The status the lecture really has: a completed job whose transcript is missing says so, and one whose numbers had to be recovered from its files says that."""
    if record.state == "completed":
        return quick_status(record)[1]
    if record.state == "paused" and "interrupted" in (record.last_error or "").lower():
        return quick_status(record)[1]
    text = STATUS_TEXT.get(record.state, record.state.title())
    return f"{text} {record.progress:.0f}%" if record.state in ("running", "paused") and record.progress else text


def status_state(record: JobRecord) -> str:
    """The chip colour for ``status_text``."""
    if record.state == "completed":
        return status_kind(quick_status(record)[0])
    return record.state


class MenuButton(QToolButton):
    """The "..." button: Enter, Space or the Down arrow opens its menu (a mouse click does the same); Esc closes it again."""

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space, Qt.Key_Down):
            self.showMenu()
            return
        super().keyPressEvent(event)


class LectureActions:
    """What the card menu does, in one place for Home and Files. ``confirm`` is replaced in tests (no modal dialog)."""

    def __init__(self, services, open_lecture: Callable[[JobRecord], None], changed: Callable[[], None], parent: Optional[QWidget] = None) -> None:
        self.services, self.open_lecture, self.changed, self.parent = services, open_lecture, changed, parent
        self.confirm: Callable[[str, str], bool] = self._confirm

    def _confirm(self, title: str, text: str) -> bool:
        return ask(self.parent, title, text, accept=title)                  # "Remove from archive" / "Remove from archive": the button says what it does

    def apply_rename(self, record: JobRecord, name: str) -> None:
        """Save the name shown for the lecture (the original file name stays what it was; an empty name or the file name itself clears the custom name)."""
        name = name.strip()
        self.services.jobs.rename(record.job_id, name if name and name != (record.source_name or "") else "")
        self.changed()

    def show_files(self, record: JobRecord) -> None:
        folder = Path(record.output_dir) if record.output_dir else None
        if folder is not None and folder.is_dir():
            self.services.open_path(folder)

    def remove(self, record: JobRecord) -> None:
        if not self.confirm("Remove from archive",
                            f"Remove '{lecture_title(record)}' from the lecture list?\n\nIts transcript, study material and recording stay on your computer."):
            return
        self.services.jobs.hide(record.job_id)
        self.changed()


class LectureCard(QFrame):
    """A clickable archive card. On a wide page (Files) the same card becomes a row: name | language | duration | date | status | menu."""

    clicked = Signal(object)
    WIDE_FROM = 760                                                      # a card at least this wide is shown as a row

    def __init__(self, record: JobRecord, actions: LectureActions, parent=None, allow_wide: bool = False) -> None:
        super().__init__(parent)
        self.record = record
        self.actions = actions
        self.allow_wide = allow_wide
        self._wide: Optional[bool] = None
        self.setObjectName("lectureCard")
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.TabFocus)
        self.setProperty("lecture", record.job_id)
        self.setAccessibleName(f"Lecture {lecture_title(record)}, {status_text(record).lower()}")
        self.setAccessibleDescription("Press Enter to open it. The button after it has more actions.")
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(tokens.SPACE["lg"], tokens.SPACE["md"], tokens.SPACE["sm"], tokens.SPACE["md"])
        self.grid.setHorizontalSpacing(8)
        self.grid.setVerticalSpacing(tokens.SPACE["xs"])

        self.name_box = QWidget()
        name_column = QVBoxLayout(self.name_box)
        name_column.setContentsMargins(0, 0, 0, 0)
        name_column.setSpacing(tokens.SPACE["xs"])
        self.title = ElidedLabel(lecture_title(record))
        self.title.setObjectName("lectureTitle")
        self.rename_edit = QLineEdit()
        self.rename_edit.setObjectName("renameEdit")
        self.rename_edit.setAccessibleName("Lecture name: Enter saves, Esc cancels")
        self.rename_edit.hide()
        self.rename_edit.installEventFilter(self)
        self.rename_hint = QLabel("Enter to save · Esc to cancel")
        self.rename_hint.setObjectName("lectureMeta")
        self.rename_hint.hide()
        self.sub = ElidedLabel(record.source_name if record.display_name and record.source_name else "")
        self.sub.setObjectName("lectureSub")
        self.sub.setVisible(bool(self.sub.text()))
        self.path = ElidedLabel(language_path(record))
        self.path.setObjectName("lectureMeta")
        self.when = ElidedLabel(when_line(record))
        self.when.setObjectName("lectureMeta")
        self.duration_label = ElidedLabel(friendly.duration(record.duration_seconds) if record.duration_seconds else "")
        self.duration_label.setObjectName("lectureMeta")
        self.date_label = ElidedLabel(short_date(record.updated or record.created))
        self.date_label.setObjectName("lectureMeta")
        for widget in (self.title, self.rename_edit, self.sub, self.rename_hint):
            name_column.addWidget(widget)
        self.status = make_chip(status_text(record), status_state(record))
        self.meta = self.path                                                    # the line the tests and the screen reader read first

        self.menu_button = MenuButton()
        self.menu_button.setObjectName("menuButton")
        self.menu_button.setIconSize(QSize(18, 18))
        icons.bind(self.menu_button, "more", size=18)
        self.menu_button.setToolTip("Open, rename, show files, remove from archive")
        self.menu_button.setAccessibleName(f"Actions for {lecture_title(record)}")
        self.menu_button.setPopupMode(QToolButton.InstantPopup)
        self.menu_button.setCursor(Qt.PointingHandCursor)
        menu = StyledMenu(self.menu_button)
        self.open_action = menu.add("Open", "open", lambda: self.clicked.emit(record))
        self.rename_action = menu.add("Rename", "rename", self.begin_rename)
        self.files_action = menu.add("Show Files", "folder", lambda: actions.show_files(record),
                                     enabled=bool(record.output_dir) and Path(record.output_dir).is_dir())
        menu.addSeparator()
        self.remove_action = menu.add("Remove from Archive", "close", lambda: actions.remove(record))
        self.menu_button.setMenu(menu)
        self._arrange(False)

    # ------------------------------------------------------------------------------------------------------------------------- layout
    def _arrange(self, wide: bool) -> None:
        if wide == self._wide:
            return
        self._wide = wide
        for widget in (self.name_box, self.menu_button, self.path, self.when, self.duration_label, self.date_label, self.status):
            self.grid.removeWidget(widget)
        for column in range(8):
            self.grid.setColumnStretch(column, 0)
            self.grid.setColumnMinimumWidth(column, 0)
        g = self.grid
        if wide:
            g.setContentsMargins(tokens.SPACE["lg"], tokens.SPACE["sm"], tokens.SPACE["sm"], tokens.SPACE["sm"])
            g.setHorizontalSpacing(16)
            g.setColumnStretch(0, 1)
            for column, width in ((1, 200), (2, 80), (3, 70), (4, 116)):
                g.setColumnMinimumWidth(column, width)
            g.addWidget(self.name_box, 0, 0, Qt.AlignVCenter)
            g.addWidget(self.path, 0, 1, Qt.AlignVCenter)
            g.addWidget(self.duration_label, 0, 2, Qt.AlignVCenter)
            g.addWidget(self.date_label, 0, 3, Qt.AlignVCenter)
            g.addWidget(self.status, 0, 4, Qt.AlignVCenter | Qt.AlignLeft)
            g.addWidget(self.menu_button, 0, 5, Qt.AlignVCenter | Qt.AlignRight)
            self.when.hide()
            self.duration_label.show()
            self.date_label.show()
        else:
            g.setContentsMargins(tokens.SPACE["lg"], tokens.SPACE["md"], tokens.SPACE["sm"], tokens.SPACE["md"])
            g.setHorizontalSpacing(8)
            g.setColumnStretch(0, 1)
            g.addWidget(self.name_box, 0, 0)
            g.addWidget(self.menu_button, 0, 1, Qt.AlignTop | Qt.AlignRight)
            g.addWidget(self.path, 1, 0, 1, 2)
            g.addWidget(self.when, 2, 0)
            g.addWidget(self.status, 2, 1, Qt.AlignRight | Qt.AlignVCenter)
            self.when.show()
            self.duration_label.hide()
            self.date_label.hide()

    def is_wide(self) -> bool:
        return bool(self._wide)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self.allow_wide:
            self._arrange(self.width() >= self.WIDE_FROM)

    # ------------------------------------------------------------------------------------------------------------------------- opening
    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.LeftButton and self.rect().contains(event.position().toPoint()) and not self.rename_edit.isVisible():
            self.clicked.emit(self.record)
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            self.clicked.emit(self.record)
            return
        if event.key() == Qt.Key_Menu or (event.key() == Qt.Key_F10 and event.modifiers() & Qt.ShiftModifier):       # the Menu key / Shift+F10: the same menu as the "..." button
            self.menu_button.showMenu()
            return
        super().keyPressEvent(event)

    def contextMenuEvent(self, event) -> None:  # noqa: N802
        """A right click on the card opens the menu where the mouse is."""
        self.menu_button.menu().exec(event.globalPos())

    # ------------------------------------------------------------------------------------------------------------------------- renaming in place
    def begin_rename(self) -> None:
        """Replace the name by a text field: Enter saves, Esc cancels. Only the displayed name changes, never the recording."""
        self.title.hide()
        self.sub.hide()
        self.rename_edit.setText(lecture_title(self.record))
        self.rename_edit.show()
        self.rename_hint.show()
        self.rename_edit.setFocus()
        self.rename_edit.selectAll()

    def commit_rename(self) -> None:
        name = self.rename_edit.text()
        self._end_rename()
        self.actions.apply_rename(self.record, name)

    def cancel_rename(self) -> None:
        self._end_rename()

    def renaming(self) -> bool:
        return self.rename_edit.isVisibleTo(self)

    def _end_rename(self) -> None:
        self.rename_edit.hide()
        self.rename_hint.hide()
        self.title.show()
        self.sub.setVisible(bool(self.sub.text()))
        self.setFocus()

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if obj is self.rename_edit:
            if event.type() == event.Type.KeyPress:
                if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                    self.commit_rename()
                    return True
                if event.key() == Qt.Key_Escape:
                    self.cancel_rename()
                    return True
            elif event.type() == event.Type.FocusOut and self.rename_edit.isVisible():
                self.cancel_rename()                                      # leaving the field without Enter changes nothing
        return super().eventFilter(obj, event)

    def set_record(self, record: JobRecord) -> None:
        self.record = record
        set_chip(self.status, status_text(record), status_state(record))
        repolish(self)
