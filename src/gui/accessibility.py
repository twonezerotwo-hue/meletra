"""What assistive technology is told about the window, and a readable dump of it.

Qt builds the accessibility tree by itself: every widget gets a role (button, combo box, edit ...), a name (``accessibleName``, else its text), a description and a state
(focusable, checked ...). Two things are added here:

* ``install()`` gives the widgets that are *pressed like buttons* but are not buttons (the lecture cards) the role "button", so an assistive technology says "button"
  and offers to press it;
* ``snapshot()`` / ``tree()`` read the tree back the way an assistive technology does (the tests and the packaged ``--ui-check`` run use them).

This is the tree Qt hands to Windows UI Automation / macOS accessibility. It is checked programmatically; it is not a test with a screen reader.
"""

from __future__ import annotations

from typing import Any, Optional

import shiboken6
from PySide6.QtCore import QObject
from PySide6.QtGui import QAccessible
from PySide6.QtWidgets import QAccessibleWidget, QComboBox, QWidget

#: classes (Qt class names) that are pressed like a button although they are frames
BUTTON_LIKE = ("LectureCard",)
_installed = False
_alive: list[Any] = []                                     # the Python side of the interfaces Qt owns (dropped again once Qt has deleted them)


class _ButtonRole(QAccessibleWidget):
    def __init__(self, widget: QWidget) -> None:
        super().__init__(widget, QAccessible.Button)


def _factory(class_name: str, obj: Optional[QObject]):
    if class_name not in BUTTON_LIKE or obj is None or not obj.isWidgetType():
        return None
    _alive[:] = [item for item in _alive if shiboken6.isValid(item)]
    interface = _ButtonRole(obj)
    _alive.append(interface)
    return interface


def install() -> None:
    """Register the roles (once per process; must run before the first lookup of an interface)."""
    global _installed
    if not _installed:
        QAccessible.installFactory(_factory)
        _installed = True


# ---------------------------------------------------------------------------------------------------------------------------------- reading the tree back
def snapshot(widget: QObject) -> Optional[dict]:
    """``{role, name, description, value, focusable, checked, enabled}`` of one object, as an assistive technology sees it (None: no interface)."""
    interface = QAccessible.queryAccessibleInterface(widget)
    if interface is None:
        return None
    state = interface.state()
    value = interface.text(QAccessible.Value)
    if not value and isinstance(widget, QComboBox):
        value = widget.currentText()                 # Qt's offscreen macOS backend may omit QAccessible.Value for combo boxes
    return {"role": interface.role().name, "name": interface.text(QAccessible.Name), "description": interface.text(QAccessible.Description),
            "value": value, "focusable": bool(state.focusable), "checked": bool(state.checked),
            "enabled": not bool(state.disabled), "class": type(widget).__name__}


def tree(root: QObject, max_depth: int = 12) -> list[dict]:
    """The accessibility tree below ``root`` (visible objects only), depth first: a flat list of ``snapshot()`` dicts with ``depth``."""
    out: list[dict] = []
    combos = ([root] if isinstance(root, QComboBox) else []) + root.findChildren(QComboBox)
    combo_values = {
        combo.accessibleName(): combo.currentText()
        for combo in combos
        if combo.accessibleName() and combo.currentText()
    }

    def walk(interface, depth: int) -> None:
        if interface is None or depth > max_depth:
            return
        state = interface.state()
        if state.invisible:
            return
        obj = interface.object()
        entry = snapshot(obj) if obj is not None else {"role": interface.role().name, "name": interface.text(QAccessible.Name), "description": "",
                                                       "value": interface.text(QAccessible.Value), "focusable": bool(state.focusable),
                                                       "checked": bool(state.checked), "enabled": not bool(state.disabled), "class": ""}
        if entry is not None:
            if entry["role"] == "ComboBox" and not entry["value"]:
                # The macOS accessibility bridge can expose a combo box as a virtual
                # object and omit QAccessible.Value.  Keep the diagnostic dump useful
                # by reading the same visible widget's current selection.
                entry["value"] = combo_values.get(entry["name"], "")
            entry["depth"] = depth
            entry["object"] = obj.objectName() if obj is not None else ""
            out.append(entry)
        for index in range(interface.childCount()):
            walk(interface.child(index), depth + 1)

    walk(QAccessible.queryAccessibleInterface(root), 0)
    return out


def find(entries: list[dict], name: str, role: str = "") -> Optional[dict]:
    """The first entry whose name starts with ``name`` (and has ``role`` when given)."""
    return next((e for e in entries if e["name"].startswith(name) and (not role or e["role"] == role)), None)


def unnamed_focusable(entries: list[dict]) -> list[dict]:
    """Focusable objects an assistive technology could not name (no name at all): the list must stay empty."""
    return [e for e in entries if e["focusable"] and e["enabled"] and not e["name"].strip() and e["role"] != "StaticText"]
