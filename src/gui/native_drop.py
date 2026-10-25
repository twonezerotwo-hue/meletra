"""Drives the window's REAL Windows (OLE) drop target with a REAL shell data object, without the mouse.

When a file is dragged from Explorer, Windows finds the drop target of the window under the cursor (``IDropTarget``, registered by Qt with ``RegisterDragDrop``) and calls
``DragEnter`` / ``DragOver`` / ``DragLeave`` / ``Drop`` with an ``IDataObject`` that holds the files (``CF_HDROP``). This module makes exactly those calls, from the inside:

* the data object is the Explorer one (``IShellFolder::GetUIObjectOf`` of the real file), not a Qt ``QMimeData``;
* the target is the one Qt registered for the window, so Qt's own OLE code converts the data, creates the ``QDragEnterEvent`` / ``QDropEvent`` and calls the application.

What it does NOT do: the mouse and the operating system's drag loop (``DoDragDrop`` hit-testing the window under the cursor, the cursor feedback, ``Esc``). Windows ignores
synthetic mouse input for that loop, so a physical drag stays a manual acceptance test (see docs/v4_1/MANUAL_DRAG_TEST.md).

Windows only. Everything runs on the GUI thread of the process that owns the window (the ``OleDropTargetInterface`` window property is process-local).
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import POINTER, WINFUNCTYPE, Structure, byref, c_long, c_ulong, c_void_p, c_wchar_p
from pathlib import Path
from typing import Optional, Sequence

DROPEFFECT_NONE, DROPEFFECT_COPY, DROPEFFECT_MOVE, DROPEFFECT_LINK = 0, 1, 2, 4
MK_LBUTTON = 1


class NativeDropUnavailable(RuntimeError):
    """Not Windows, no drop target registered for the window, or the shell would not make a data object."""


class POINTL(Structure):
    _fields_ = [("x", c_long), ("y", c_long)]


class GUID(Structure):
    _fields_ = [("a", ctypes.c_ulong), ("b", ctypes.c_ushort), ("c", ctypes.c_ushort), ("d", ctypes.c_ubyte * 8)]


def _guid(text: str) -> GUID:
    guid = GUID()
    if ctypes.windll.ole32.IIDFromString(text, byref(guid)) != 0:
        raise NativeDropUnavailable(f"bad IID {text}")
    return guid


def _method(com_pointer: int, index: int, *argtypes):
    """The method ``index`` of the vtable of a COM object, as a callable (the first argument is the object itself)."""
    table = ctypes.cast(ctypes.cast(com_pointer, POINTER(c_void_p))[0], POINTER(c_void_p))
    return WINFUNCTYPE(ctypes.HRESULT, c_void_p, *argtypes)(table[index])


def _release(com_pointer: Optional[int]) -> None:
    if com_pointer:
        _method(com_pointer, 2)(com_pointer)


class ShellDataObject:
    """The data object Explorer makes for dragged files (``CF_HDROP`` and the shell ID list)."""

    def __init__(self, paths: Sequence[str | Path]) -> None:
        if sys.platform != "win32":
            raise NativeDropUnavailable("Windows only")
        if not paths:
            raise NativeDropUnavailable("no file")
        shell32, ole32 = ctypes.windll.shell32, ctypes.windll.ole32
        ole32.OleInitialize(None)                                    # Qt did it already (S_FALSE); harmless
        parent_folder = None
        pidls: list[int] = []
        children = (c_void_p * len(paths))()
        try:
            for index, path in enumerate(paths):
                pidl = c_void_p()
                hr = shell32.SHParseDisplayName(c_wchar_p(str(Path(path).resolve())), None, byref(pidl), c_ulong(0), None)
                if hr != 0 or not pidl.value:
                    raise NativeDropUnavailable(f"the shell does not know {path} (0x{hr & 0xFFFFFFFF:08X})")
                pidls.append(pidl.value)
                folder, child = c_void_p(), c_void_p()
                hr = shell32.SHBindToParent(c_void_p(pidl.value), byref(_guid("{000214E6-0000-0000-C000-000000000046}")), byref(folder), byref(child))   # IShellFolder
                if hr != 0:
                    raise NativeDropUnavailable(f"SHBindToParent failed (0x{hr & 0xFFFFFFFF:08X})")
                if parent_folder is None:
                    parent_folder = folder.value
                else:
                    _release(folder.value)
                children[index] = child.value
            data = c_void_p()
            # IShellFolder::GetUIObjectOf(hwndOwner, cidl, apidl, riid, rgfReserved, ppv)
            get_ui_object = _method(parent_folder, 10, c_void_p, ctypes.c_uint, POINTER(c_void_p), POINTER(GUID), POINTER(ctypes.c_uint), POINTER(c_void_p))
            iid = _guid("{0000010E-0000-0000-C000-000000000046}")                                                                              # IDataObject
            get_ui_object(parent_folder, None, len(paths), children, byref(iid), None, byref(data))
            if not data.value:
                raise NativeDropUnavailable("no data object")
            self.pointer = data.value
        finally:
            _release(parent_folder)
            for pidl in pidls:
                ole32.CoTaskMemFree(c_void_p(pidl))

    def formats(self) -> list[int]:
        """The clipboard formats the object offers (15 = CF_HDROP)."""
        enumerate_formats = _method(self.pointer, 8, ctypes.c_ulong, POINTER(c_void_p))               # IDataObject::EnumFormatEtc(DATADIR_GET = 1)
        enumerator = c_void_p()
        enumerate_formats(self.pointer, 1, byref(enumerator))
        found: list[int] = []

        class FORMATETC(Structure):
            _fields_ = [("cfFormat", ctypes.c_ushort), ("ptd", c_void_p), ("dwAspect", c_ulong), ("lindex", c_long), ("tymed", c_ulong)]

        next_format = _method(enumerator.value, 3, c_ulong, POINTER(FORMATETC), POINTER(c_ulong))      # IEnumFORMATETC::Next
        while True:
            item, fetched = FORMATETC(), c_ulong(0)
            if next_format(enumerator.value, 1, byref(item), byref(fetched)) != 0 or fetched.value == 0:
                break
            found.append(int(item.cfFormat))
        _release(enumerator.value)
        return found

    def release(self) -> None:
        _release(getattr(self, "pointer", None))
        self.pointer = None


def drop_target_of(window) -> int:
    """The ``IDropTarget`` Qt registered for the top-level window (OLE keeps it in the window property ``OleDropTargetInterface``)."""
    if sys.platform != "win32":
        raise NativeDropUnavailable("Windows only")
    user32 = ctypes.windll.user32
    user32.GetPropW.restype = c_void_p
    user32.GetPropW.argtypes = [c_void_p, c_wchar_p]
    hwnd = int(window.winId())
    target = user32.GetPropW(hwnd, "OleDropTargetInterface")
    if not target:
        raise NativeDropUnavailable("no drop target is registered for the window (is the platform plugin 'windows'? is the window created and accepting drops?)")
    return int(target)


class NativeDrag:
    """One drag session against the window's drop target: ``enter`` -> ``over`` ... -> ``drop`` (or ``leave``). Screen coordinates, like Windows passes them."""

    def __init__(self, window, paths: Sequence[str | Path]) -> None:
        self.window = window
        self.target = drop_target_of(window)
        self.data = ShellDataObject(paths)
        self.effect = DROPEFFECT_NONE

    def _point(self, x: int, y: int) -> POINTL:
        return POINTL(int(x), int(y))

    def _call(self, index: int, x: int, y: int, with_data: bool) -> int:
        effect = c_ulong(DROPEFFECT_COPY | DROPEFFECT_MOVE | DROPEFFECT_LINK)
        if with_data:
            method = _method(self.target, index, c_void_p, c_ulong, POINTL, POINTER(c_ulong))
            method(self.target, self.data.pointer, MK_LBUTTON, self._point(x, y), byref(effect))
        else:
            method = _method(self.target, index, c_ulong, POINTL, POINTER(c_ulong))
            method(self.target, MK_LBUTTON, self._point(x, y), byref(effect))
        self.effect = int(effect.value)
        return self.effect

    def enter(self, x: int, y: int) -> int:
        return self._call(3, x, y, True)                         # IDropTarget::DragEnter

    def over(self, x: int, y: int) -> int:
        return self._call(4, x, y, False)                        # IDropTarget::DragOver

    def leave(self) -> None:
        _method(self.target, 5)(self.target)                     # IDropTarget::DragLeave

    def drop(self, x: int, y: int) -> int:
        return self._call(6, x, y, True)                         # IDropTarget::Drop

    def close(self) -> None:
        self.data.release()


def screen_point_in(window, x: int, y: int) -> tuple[int, int]:
    """The screen position of the window-relative point (x, y)."""
    point = window.mapToGlobal(window.rect().topLeft())
    scale = window.devicePixelRatioF()
    return int(point.x() * scale + x * scale), int(point.y() * scale + y * scale)
