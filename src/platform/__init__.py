"""Operating-system specifics in ONE place: folders, opening files, keeping the computer awake, system facts, credential storage.

    from src.platform import current
    current().user_data_dir()          # LOCALAPPDATA/MedicalLectureTranscriber (Windows)  |  ~/Library/Application Support/MedicalLectureTranscriber

The rest of the program never asks "is this Windows?"; it asks the platform object. Tests build a platform object with a fake home folder,
environment and command runner, so macOS behaviour is checked on a Windows machine too.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Iterator, Optional

from .common import AcceleratorInfo, Platform, PosixPlatform
from .macos import MacPlatform
from .windows import WindowsPlatform

__all__ = ["AcceleratorInfo", "MacPlatform", "Platform", "PosixPlatform", "WindowsPlatform", "current", "detect", "use"]

_current: Optional[Platform] = None


def detect() -> Platform:
    """The platform object for the running operating system."""
    if sys.platform == "win32":
        return WindowsPlatform()
    if sys.platform == "darwin":
        return MacPlatform()
    return PosixPlatform()


def current() -> Platform:
    global _current
    if _current is None:
        _current = detect()
    return _current


@contextmanager
def use(platform: Platform) -> Iterator[Platform]:
    """Temporarily pretend to run on ``platform`` (tests)."""
    global _current
    previous = _current
    _current = platform
    try:
        yield platform
    finally:
        _current = previous
