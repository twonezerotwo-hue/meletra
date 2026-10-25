"""Background threads for everything that can take a moment (system check, file probe, model download ...).

The main thread only ever shows results; it never waits for them, so the window can not freeze.
"""

from __future__ import annotations

import logging
import traceback
from typing import Any, Callable

from PySide6.QtCore import QThread, Signal

log = logging.getLogger(__name__)


class TaskThread(QThread):
    """Run ``fn(progress, is_cancelled)`` off the GUI thread.

    ``progress(dict)`` may be called from ``fn`` at any time; ``cancel()`` makes ``is_cancelled()`` return True.
    Emits ``succeeded(result)`` or ``failed(message)``, then ``done()`` in both cases.
    """

    progressed = Signal(dict)
    succeeded = Signal(object)
    failed = Signal(str)
    done = Signal()

    def __init__(self, fn: Callable[..., Any], parent=None) -> None:
        super().__init__(parent)
        self._fn = fn
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def is_cancelled(self) -> bool:
        return self._cancel

    def run(self) -> None:
        try:
            result = self._fn(self.progressed.emit, self.is_cancelled)
        except Exception as exc:  # noqa: BLE001 - shown to the user, full trace in the log
            log.error("Background task failed: %s\n%s", exc, traceback.format_exc())
            self.failed.emit(str(exc) or type(exc).__name__)
        else:
            self.succeeded.emit(result)
        finally:
            self.done.emit()


def stop_threads(widget, timeout_ms: int = 6000) -> int:
    """Cancel and wait for every background task owned by ``widget`` (and its children).

    A QThread that is destroyed while it still runs aborts the whole program, so every window or dialog that can start a download or a
    system check calls this before it goes away. Returns how many threads were still running when the timeout ended (normally 0).
    """
    still = 0
    for thread in widget.findChildren(TaskThread):
        if thread.isRunning():
            thread.cancel()
            if not thread.wait(timeout_ms):
                still += 1
    return still
