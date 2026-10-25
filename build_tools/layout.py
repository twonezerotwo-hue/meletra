"""Where a build of THIS version puts its files, so building 0.2.x can never overwrite the production 0.1.1 artefacts.

    0.1.x (legacy, the production baseline)   dist/MedicalLectureTranscriber        release/<files>   (the product was called Medical Lecture Transcriber)
    0.2.x - 4.0.x                              dist/v4.0.0/MedicalLectureTranscriber  release/v4.0.0/Windows/<files>
    4.1 and later (MELETRA)                    dist/v4.1.0/MELETRA                    release/v4.1.0/Windows/<files>
                                                                                      release/v0.2.0/macOS/<files>
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP_ID = "MedicalLectureTranscriber"            # the identity of the data folders and of the releases before 4.1 (never changes)
PRODUCT_ID = "MELETRA"                           # the file identity of 4.1 and later: MELETRA.exe, MELETRA-Setup-Windows-x64.exe, MELETRA.app
_VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:\.(\d+))?(?:[-+.]?([A-Za-z][\w.]*))?$")


def split_version(version: str) -> tuple[tuple[int, int, int, int], str]:
    """``"0.2.0-dev"`` -> ``((0, 2, 0, 0), "dev")``; ``"0.1.1"`` -> ``((0, 1, 1, 0), "")``. Raises ValueError for junk."""
    match = _VERSION_RE.match(version.strip())
    if not match:
        raise ValueError(f"not a version: {version!r}")
    numbers = tuple(int(match.group(i) or 0) for i in range(1, 5))
    return numbers, (match.group(5) or "")           # type: ignore[return-value]


def base_version(version: str) -> str:
    """The numeric part only (``0.2.0-rc1`` -> ``0.2.0``): what installers and the exe version resource can hold."""
    (major, minor, patch, _), _suffix = split_version(version)
    return f"{major}.{minor}.{patch}"


def is_legacy(version: str) -> bool:
    return split_version(version)[0][:2] == (0, 1)


def file_id(version: str) -> str:
    """The file identity of a version: 4.1 and later are MELETRA, older ones keep the names they were released with."""
    return PRODUCT_ID if split_version(version)[0][:2] >= (4, 1) else APP_ID


def dist_dir(version: str, root: Path = ROOT) -> Path:
    if is_legacy(version):
        return root / "dist" / APP_ID
    return root / "dist" / f"v{base_version(version)}" / file_id(version)


def dist_parent(version: str, root: Path = ROOT) -> Path:
    """The ``--distpath`` PyInstaller must be given (it creates ``<distpath>/MELETRA``, before 4.1 ``<distpath>/MedicalLectureTranscriber``)."""
    return dist_dir(version, root).parent


def release_dir(version: str, platform_name: str = "Windows", root: Path = ROOT) -> Path:
    if is_legacy(version):
        return root / "release"
    return root / "release" / f"v{base_version(version)}" / platform_name
