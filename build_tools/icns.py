"""The macOS application icon (.icns) written without any macOS tool: PNG entries in Apple's container format.

``iconutil`` exists only on a Mac; the format itself is a four-byte tag, a big-endian length, and entries of ``type + length + PNG``.
Modern macOS reads PNG-compressed entries for every size from 16 to 1024 pixels, so this file works in the Dock, Finder and the DMG.
"""

from __future__ import annotations

import struct
from pathlib import Path

#: edge length in pixels -> icns OSType of the PNG-compressed entry
ICNS_TYPES = {16: b"icp4", 32: b"icp5", 64: b"icp6", 128: b"ic07", 256: b"ic08", 512: b"ic09", 1024: b"ic10"}
ICNS_SIZES = tuple(sorted(ICNS_TYPES))
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def icns_bytes(pngs: dict[int, bytes]) -> bytes:
    """An .icns file holding one PNG per size (sizes Apple does not define are rejected)."""
    entries = b""
    for size in sorted(pngs):
        if size not in ICNS_TYPES:
            raise ValueError(f"{size}px is not an icns size (use {', '.join(map(str, ICNS_SIZES))})")
        data = pngs[size]
        if not data.startswith(PNG_SIGNATURE):
            raise ValueError(f"the {size}px image is not a PNG")
        entries += ICNS_TYPES[size] + struct.pack(">I", len(data) + 8) + data
    return b"icns" + struct.pack(">I", len(entries) + 8) + entries


def read_icns_sizes(data: bytes) -> list[int]:
    """The sizes stored in an .icns (tests and the build's self-check)."""
    if data[:4] != b"icns":
        raise ValueError("not an .icns file")
    total = struct.unpack_from(">I", data, 4)[0]
    by_type = {v: k for k, v in ICNS_TYPES.items()}
    sizes, offset = [], 8
    while offset + 8 <= min(total, len(data)):
        kind, length = data[offset:offset + 4], struct.unpack_from(">I", data, offset + 4)[0]
        if kind in by_type:
            sizes.append(by_type[kind])
        offset += max(length, 8)
    return sizes


def write_icns(destination: Path) -> Path:
    """Draw the app icon (the same drawing as the Windows icon) at every icns size and write the .icns."""
    from .icon import render_pngs

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(icns_bytes(render_pngs(ICNS_SIZES)))
    return destination
