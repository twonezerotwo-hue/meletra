"""Keep Windows awake while a transcription runs (a two-hour lecture must not be interrupted by the sleep timer)."""

from __future__ import annotations

import logging
import sys
from typing import Callable, Optional

log = logging.getLogger(__name__)

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001          # the PC stays awake; the screen may still turn off


def _windows_setter(flags: int) -> int:
    import ctypes

    return int(ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint(flags)))


def keep_awake(on: bool, setter: Optional[Callable[[int], int]] = None) -> bool:
    """Ask Windows not to sleep (``on``) or release the request. The request belongs to the calling thread: call both times
    from the GUI thread. Returns True when Windows accepted it; never raises."""
    if setter is None:
        from ..platform import current

        return bool(current().keep_awake(on))
    try:
        previous = setter(ES_CONTINUOUS | ES_SYSTEM_REQUIRED if on else ES_CONTINUOUS)
    except Exception:  # noqa: BLE001 - cosmetic feature, never a reason to fail
        log.debug("SetThreadExecutionState failed", exc_info=True)
        return False
    return bool(previous)
