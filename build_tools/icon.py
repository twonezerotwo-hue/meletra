"""Application icon: drawn in code (src/gui/theme.py) and written as a multi-size .ico (PNG-compressed entries)."""

from __future__ import annotations

import struct
from pathlib import Path

ICON_SIZES = (16, 24, 32, 48, 64, 128, 256)


def ico_bytes(pngs: dict[int, bytes]) -> bytes:
    """A Windows .ico holding one PNG per size (supported since Windows Vista)."""
    sizes = sorted(pngs)
    header = struct.pack("<HHH", 0, 1, len(sizes))
    offset = 6 + 16 * len(sizes)
    entries, blobs = b"", b""
    for size in sizes:
        data = pngs[size]
        dim = 0 if size >= 256 else size
        entries += struct.pack("<BBBBHHII", dim, dim, 0, 0, 1, 32, len(data), offset)
        blobs += data
        offset += len(data)
    return header + entries + blobs


def render_pngs(sizes: tuple[int, ...] = ICON_SIZES) -> dict[int, bytes]:
    from PySide6.QtCore import QBuffer, QByteArray, QIODevice

    from src.gui.theme import draw_icon

    result: dict[int, bytes] = {}
    for size in sizes:
        data = QByteArray()
        buffer = QBuffer(data)
        buffer.open(QIODevice.WriteOnly)
        draw_icon(size).save(buffer, "PNG")
        result[size] = bytes(data)
    return result


def write_icon(destination: Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pngs = render_pngs()
    destination.write_bytes(ico_bytes(pngs))
    destination.with_suffix(".png").write_bytes(pngs[256])
    return destination


def read_ico_sizes(data: bytes) -> list[int]:
    """Sizes stored in an .ico (used by tests and by the build's self-check)."""
    reserved, kind, count = struct.unpack_from("<HHH", data, 0)
    if reserved != 0 or kind != 1:
        raise ValueError("not an .ico file")
    sizes = []
    for i in range(count):
        width = data[6 + 16 * i]
        sizes.append(256 if width == 0 else width)
    return sizes
