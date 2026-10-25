"""One icon set for the whole program: simple line icons drawn with QPainter on a 24 x 24 grid (stroke 1.8, round caps), tinted with the theme colours.

No emoji, no Unicode symbols, no second style. ``bind`` gives a button an icon that follows the theme (normal / hover / selected) and is redrawn when the theme changes.
"""

from __future__ import annotations

import math
import weakref
from typing import Callable

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QGuiApplication, QIcon, QPainter, QPainterPath, QPen, QPixmap

from . import tokens

GRID = 24.0
STROKE = 1.8
_SCALE = 2                                         # drawn at twice the size: crisp on high-DPI screens, smooth on normal ones


def _path(*commands) -> QPainterPath:
    path = QPainterPath()
    for command in commands:
        kind, *args = command
        if kind == "M":
            path.moveTo(*args)
        elif kind == "L":
            path.lineTo(*args)
        elif kind == "C":
            path.cubicTo(*args)
        elif kind == "Z":
            path.closeSubpath()
    return path


def _stroke(p: QPainter, *commands) -> None:
    p.drawPath(_path(*commands))


def _home(p, c):
    _stroke(p, ("M", 3.5, 11), ("L", 12, 3.8), ("L", 20.5, 11))
    _stroke(p, ("M", 5.5, 9.5), ("L", 5.5, 20), ("L", 18.5, 20), ("L", 18.5, 9.5))
    _stroke(p, ("M", 10, 20), ("L", 10, 14), ("L", 14, 14), ("L", 14, 20))


def _waveform(p, c):
    for x, h in ((4, 2.5), (8, 6), (12, 9), (16, 5), (20, 2.5)):
        p.drawLine(QPointF(x, 12 - h), QPointF(x, 12 + h))


def _book(p, c):
    _stroke(p, ("M", 12, 6.5), ("C", 9.5, 4.6, 6, 4.2, 3, 5), ("L", 3, 18.2), ("C", 6, 17.4, 9.5, 17.8, 12, 19.6),
            ("C", 14.5, 17.8, 18, 17.4, 21, 18.2), ("L", 21, 5), ("C", 18, 4.2, 14.5, 4.6, 12, 6.5), ("Z",))
    _stroke(p, ("M", 12, 6.5), ("L", 12, 19.6))


def _folder(p, c):
    _stroke(p, ("M", 3, 7.5), ("C", 3, 6.4, 3.9, 5.5, 5, 5.5), ("L", 9.2, 5.5), ("L", 11.2, 7.7), ("L", 19, 7.7), ("C", 20.1, 7.7, 21, 8.6, 21, 9.7),
            ("L", 21, 17.5), ("C", 21, 18.6, 20.1, 19.5, 19, 19.5), ("L", 5, 19.5), ("C", 3.9, 19.5, 3, 18.6, 3, 17.5), ("Z",))


def _gear(p, c):
    centre = QPointF(12, 12)
    p.drawEllipse(centre, 3.1, 3.1)
    p.drawEllipse(centre, 6.6, 6.6)
    for i in range(8):
        angle = math.radians(i * 45)
        a = QPointF(12 + 6.6 * math.cos(angle), 12 + 6.6 * math.sin(angle))
        b = QPointF(12 + 9.4 * math.cos(angle), 12 + 9.4 * math.sin(angle))
        p.drawLine(a, b)


def _upload(p, c):
    _stroke(p, ("M", 12, 15.5), ("L", 12, 4.5))
    _stroke(p, ("M", 7.5, 9), ("L", 12, 4.5), ("L", 16.5, 9))
    _stroke(p, ("M", 4, 15.5), ("L", 4, 18.5), ("C", 4, 19.3, 4.7, 20, 5.5, 20), ("L", 18.5, 20), ("C", 19.3, 20, 20, 19.3, 20, 18.5), ("L", 20, 15.5))


def _video(p, c):
    p.drawRoundedRect(QRectF(3, 5.5, 18, 13), 3, 3)
    _stroke(p, ("M", 10.3, 9.4), ("L", 10.3, 14.6), ("L", 14.8, 12), ("Z",))


def _pdf(p, c):
    _stroke(p, ("M", 6.5, 3.5), ("L", 13.5, 3.5), ("L", 19, 9), ("L", 19, 19.5), ("C", 19, 20.1, 18.6, 20.5, 18, 20.5), ("L", 6.5, 20.5),
            ("C", 5.9, 20.5, 5.5, 20.1, 5.5, 19.5), ("L", 5.5, 4.5), ("C", 5.5, 3.9, 5.9, 3.5, 6.5, 3.5), ("Z",))
    _stroke(p, ("M", 13.5, 3.5), ("L", 13.5, 9), ("L", 19, 9))
    _stroke(p, ("M", 8.5, 13.5), ("L", 16, 13.5))
    _stroke(p, ("M", 8.5, 17), ("L", 13.5, 17))


def _image(p, c):
    p.drawRoundedRect(QRectF(3.5, 4.5, 17, 15), 2.5, 2.5)
    p.drawEllipse(QPointF(9, 10), 1.6, 1.6)
    _stroke(p, ("M", 4, 17.5), ("L", 9, 12.8), ("L", 12.6, 16), ("L", 15.2, 13.6), ("L", 20, 18))


def _pencil(p, c):
    _stroke(p, ("M", 4, 20), ("L", 5.2, 15.2), ("L", 15.6, 4.8), ("C", 16.4, 4, 17.7, 4, 18.5, 4.8), ("L", 19.2, 5.5), ("C", 20, 6.3, 20, 7.6, 19.2, 8.4),
            ("L", 8.8, 18.8), ("Z",))
    _stroke(p, ("M", 14, 6.5), ("L", 17.5, 10))


def _more(p, c):
    p.setPen(Qt.NoPen)
    p.setBrush(c)
    for x in (5.5, 12, 18.5):
        p.drawEllipse(QPointF(x, 12), 1.7, 1.7)


def _close(p, c):
    _stroke(p, ("M", 6.5, 6.5), ("L", 17.5, 17.5))
    _stroke(p, ("M", 17.5, 6.5), ("L", 6.5, 17.5))


def _replace(p, c):
    _stroke(p, ("M", 4, 8.5), ("L", 18.5, 8.5))
    _stroke(p, ("M", 15, 5), ("L", 18.5, 8.5), ("L", 15, 12))
    _stroke(p, ("M", 20, 15.5), ("L", 5.5, 15.5))
    _stroke(p, ("M", 9, 12), ("L", 5.5, 15.5), ("L", 9, 19))


def _search(p, c):
    p.drawEllipse(QPointF(10.5, 10.5), 6, 6)
    _stroke(p, ("M", 15, 15), ("L", 20.5, 20.5))


def _check(p, c):
    _stroke(p, ("M", 5, 12.5), ("L", 9.8, 17.3), ("L", 19, 7.5))


def _chevron(p, c):
    _stroke(p, ("M", 6, 9.5), ("L", 12, 15.5), ("L", 18, 9.5))


def _chevron_up(p, c):
    _stroke(p, ("M", 6, 14.5), ("L", 12, 8.5), ("L", 18, 14.5))


def _plus(p, c):
    _stroke(p, ("M", 12, 5), ("L", 12, 19))
    _stroke(p, ("M", 5, 12), ("L", 19, 12))


def _dot(p, c):
    p.setPen(Qt.NoPen)
    p.setBrush(c)
    p.drawEllipse(QPointF(12, 12), 7.5, 7.5)


def _open(p, c):
    _stroke(p, ("M", 8, 16), ("L", 16.5, 7.5))
    _stroke(p, ("M", 9, 7.5), ("L", 16.5, 7.5), ("L", 16.5, 15))


def _arrow_left(p, c):
    _stroke(p, ("M", 19, 12), ("L", 5.5, 12))
    _stroke(p, ("M", 11, 6.5), ("L", 5.5, 12), ("L", 11, 17.5))


def _clock(p, c):
    p.drawEllipse(QPointF(12, 12), 8.5, 8.5)
    _stroke(p, ("M", 12, 7), ("L", 12, 12), ("L", 15.5, 14))


def _alert(p, c):
    _stroke(p, ("M", 12, 4), ("L", 21, 19.5), ("L", 3, 19.5), ("Z",))
    _stroke(p, ("M", 12, 10), ("L", 12, 14.2))
    p.drawPoint(QPointF(12, 16.9))


def _cards(p, c):
    p.drawRoundedRect(QRectF(3.5, 7.5, 14, 11), 2.2, 2.2)
    _stroke(p, ("M", 7, 4.5), ("L", 18.5, 4.5), ("C", 19.9, 4.5, 20.5, 5.1, 20.5, 6.5), ("L", 20.5, 14.5))


def _arrow_right(p, c):
    _stroke(p, ("M", 5, 12), ("L", 18.5, 12))
    _stroke(p, ("M", 13, 6.5), ("L", 18.5, 12), ("L", 13, 17.5))


def _lock(p, c):
    p.drawRoundedRect(QRectF(5, 10.5, 14, 9.5), 2.2, 2.2)
    _stroke(p, ("M", 8.2, 10.5), ("L", 8.2, 8), ("C", 8.2, 5.5, 9.8, 4, 12, 4), ("C", 14.2, 4, 15.8, 5.5, 15.8, 8), ("L", 15.8, 10.5))
    p.drawPoint(QPointF(12, 15.2))


def _chip(p, c):
    p.drawRoundedRect(QRectF(7, 7, 10, 10), 2, 2)
    for x in (9.5, 12, 14.5):
        p.drawLine(QPointF(x, 3.5), QPointF(x, 7))
        p.drawLine(QPointF(x, 17), QPointF(x, 20.5))
    for y in (9.5, 12, 14.5):
        p.drawLine(QPointF(3.5, y), QPointF(7, y))
        p.drawLine(QPointF(17, y), QPointF(20.5, y))


def _help(p, c):
    p.drawEllipse(QPointF(12, 12), 8.5, 8.5)
    _stroke(p, ("M", 9.6, 9.8), ("C", 9.8, 8.4, 10.8, 7.6, 12.1, 7.6), ("C", 13.5, 7.6, 14.5, 8.5, 14.5, 9.7),
            ("C", 14.5, 11.6, 12, 11.5, 12, 13.6))
    p.drawPoint(QPointF(12, 16.4))


DRAWERS: dict[str, Callable] = {
    "home": _home, "transcription": _waveform, "audio": _waveform, "study": _book, "files": _folder, "folder": _folder, "settings": _gear,
    "upload": _upload, "video": _video, "pdf": _pdf, "document": _pdf, "image": _image, "rename": _pencil, "more": _more, "close": _close,
    "replace": _replace, "search": _search, "check": _check, "chevron": _chevron, "chevron_up": _chevron_up, "plus": _plus, "dot": _dot, "open": _open, "back": _arrow_left,
    "clock": _clock, "alert": _alert, "cards": _cards, "help": _help, "arrow_right": _arrow_right, "lock": _lock, "chip": _chip,
}


def pixmap(name: str, color: str, size: int = 20) -> QPixmap:
    """One icon, tinted ``color`` (``#rrggbb``), ``size`` logical pixels square."""
    draw = DRAWERS[name]
    pm = QPixmap(size * _SCALE, size * _SCALE)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
    p.scale(size * _SCALE / GRID, size * _SCALE / GRID)
    pen = QPen(QColor(color), STROKE, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    draw(p, QColor(color))
    p.end()
    pm.setDevicePixelRatio(_SCALE)
    return pm


def icon(name: str, normal: str = "text2", hover: str = "text", selected: str = "teal", disabled: str = "text3", size: int = 20) -> QIcon:
    """An icon whose colours are palette ROLES: normal, hover (QIcon.Active), checked / selected (QIcon.On) and disabled."""
    result = QIcon()
    result.addPixmap(pixmap(name, tokens.color(normal), size), QIcon.Normal, QIcon.Off)
    result.addPixmap(pixmap(name, tokens.color(hover), size), QIcon.Active, QIcon.Off)
    result.addPixmap(pixmap(name, tokens.color(selected), size), QIcon.Normal, QIcon.On)
    result.addPixmap(pixmap(name, tokens.color(selected), size), QIcon.Active, QIcon.On)
    result.addPixmap(pixmap(name, tokens.color(disabled), size), QIcon.Disabled, QIcon.Off)
    return result


def badge_pixmap(name: str, size: int = 64, glyph: int = 28, background: str = "teal_soft", foreground: str = "teal", ring: str = "") -> QPixmap:
    """The icon in a round, softly tinted badge (the drop area, the selected file, empty states)."""
    pm = QPixmap(size * _SCALE, size * _SCALE)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
    p.setPen(QPen(QColor(tokens.color(ring)), 1.5 * _SCALE) if ring else Qt.NoPen)
    p.setBrush(QColor(tokens.color(background)))
    margin = 1.0 * _SCALE
    p.drawEllipse(QRectF(margin, margin, size * _SCALE - 2 * margin, size * _SCALE - 2 * margin))
    inner = pixmap(name, tokens.color(foreground), glyph)
    offset = (size - glyph) / 2 * _SCALE
    p.drawPixmap(QRectF(offset, offset, glyph * _SCALE, glyph * _SCALE), inner, QRectF(inner.rect()))
    p.end()
    pm.setDevicePixelRatio(_SCALE)
    return pm


_bound: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def bind(widget, name: str, normal: str = "text2", hover: str = "text", selected: str = "teal", size: int = 20):
    """Give a button / action an icon that follows the theme; ``refresh()`` redraws every bound icon after a theme change."""
    _bound[widget] = (name, normal, hover, selected, size)
    widget.setIcon(icon(name, normal, hover, selected, size=size))
    return widget


def refresh() -> None:
    for widget, (name, normal, hover, selected, size) in list(_bound.items()):
        try:
            widget.setIcon(icon(name, normal, hover, selected, size=size))
        except RuntimeError:                                    # the widget is gone (deleted C++ object)
            pass


def png_file(name: str, color: str, size: int, folder) -> str:
    """The icon as a PNG file (for the few places a style sheet needs a file: the combo-box arrow and the check mark). Returns a forward-slash path."""
    from pathlib import Path

    target = Path(folder) / f"{name}_{color.lstrip('#')}_{size}.png"
    if not target.is_file() and QGuiApplication.instance() is not None:                 # (a QPixmap needs the application; without it the style sheet text is still produced)
        target.parent.mkdir(parents=True, exist_ok=True)
        pixmap(name, color, size).save(str(target), "PNG")
    return target.as_posix()
