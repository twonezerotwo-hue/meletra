"""MELETRA brand assets at run time: the logo and the application icon, loaded from assets/brand (in the build: the bundled copy)."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QIcon, QImage, QPixmap

from .. import APP_NAME, APP_TAGLINE, RESOURCE_ROOT

ICON_SIZES = (16, 24, 32, 48, 64, 128, 256)
NAME = APP_NAME
TAGLINE = APP_TAGLINE


def brand_dir() -> Path:
    return RESOURCE_ROOT / "assets" / "brand"


def icon_image(size: int) -> QImage:
    """The application icon at ``size`` px (rounded square with the symbol). A missing asset gives an empty image, never an error."""
    folder = brand_dir()
    candidates = [n for n in (16, 24, 32, 48, 64, 128, 256, 512, 1024) if n >= size] or [1024]
    image = QImage(str(folder / f"meletra_icon_{candidates[0]}.png"))
    if image.isNull():
        image = QImage(size, size, QImage.Format_ARGB32)
        image.fill(0)
        return image
    from PySide6.QtCore import Qt

    return image if image.width() == size else image.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)


def app_icon() -> QIcon:
    icon = QIcon()
    for size in ICON_SIZES:
        icon.addPixmap(QPixmap.fromImage(icon_image(size)))
    return icon


def logo_pixmap(small: bool = True, height: int = 0) -> QPixmap:
    """The symbol (``small``) or the whole lock-up on a transparent background, optionally scaled to ``height`` px."""
    pixmap = QPixmap(str(brand_dir() / ("meletra_symbol.png" if small else "meletra_logo.png")))
    if height and not pixmap.isNull():
        from PySide6.QtCore import Qt

        pixmap = pixmap.scaledToHeight(height, Qt.SmoothTransformation)
    return pixmap
