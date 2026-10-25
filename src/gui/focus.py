"""Keyboard focus ring, like the web's :focus-visible: buttons, check boxes and cards take the focus from Tab / Shift+Tab (and shortcuts) but not from a mouse click, so the teal
2 px ring of the style sheet appears only for people who use the keyboard. Enter and Space both press a focused button. Text fields and combo boxes keep the normal behaviour (they show the ring whenever they have the focus).

This is done with a style hint (``SH_Button_FocusPolicy``, the same hint macOS uses): no event filter, no cost while the program runs.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QDialogButtonBox, QLineEdit, QPlainTextEdit, QProxyStyle, QPushButton, QStyle

_installed = False


class WhisperStyle(QProxyStyle):
    """Qt's Fusion style, with buttons that are reached by the keyboard only."""

    def polish(self, target):
        super().polish(target)
        if isinstance(target, QPushButton):
            target.setAutoDefault(True)                  # Qt presses an "auto default" button on Enter as well as on Space (outside dialogs it is off by default)
        elif isinstance(target, (QLineEdit, QPlainTextEdit)) and not target.accessibleName() and target.placeholderText():
            target.setAccessibleName(target.placeholderText().rstrip(". "))      # a field is named by its hint ("Search in the transcript"): a screen reader does not read placeholders

    def styleHint(self, hint, option=None, widget=None, returnData=None):  # noqa: N802 - Qt API
        if hint == QStyle.SH_Button_FocusPolicy:
            return int(Qt.TabFocus.value)
        if hint == QStyle.SH_DialogButtonLayout:
            return int(QDialogButtonBox.MacLayout.value)      # every dialog: Cancel / Close on the left, the main action on the right (Windows would put it first)
        return super().styleHint(hint, option, widget, returnData)


def install(app: QApplication) -> None:
    """Use the Whisper style (once; widgets created afterwards get the keyboard-only buttons)."""
    global _installed
    if not _installed:
        app.setStyle(WhisperStyle("Fusion"))
        _installed = True
