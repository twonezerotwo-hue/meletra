"""Sends the queued transcripts to the developer's backend, in a background thread. Never raises, never blocks the program, never shows an error."""

from __future__ import annotations

import json
import logging
import threading
import urllib.request
from typing import Callable, Optional

from .queue import UploadQueue

log = logging.getLogger(__name__)

TIMEOUT_SECONDS = 20
INTERVAL_SECONDS = 60


def http_post(url: str, body: bytes, timeout: float = TIMEOUT_SECONDS) -> int:
    """POST the JSON ``body``; returns the HTTP status. Raises on a network problem (the caller queues it again)."""
    request = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": "application/json; charset=utf-8", "User-Agent": "MELETRA"})
    with urllib.request.urlopen(request, timeout=timeout) as response:     # noqa: S310 - https (or loopback in tests) only: checked by config.valid_endpoint
        response.read(1024)
        return int(response.status)


class Uploader:
    def __init__(self, queue: UploadQueue, endpoint: Callable[[], str], enabled: Callable[[], bool], post: Callable[[str, bytes], int] = http_post) -> None:
        self.queue, self._endpoint, self._enabled, self._post = queue, endpoint, enabled, post
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def run_once(self) -> int:
        """One pass over the items that are due. Returns how many were sent. Does nothing while sharing is off or no address is configured."""
        sent = 0
        try:
            if not self._enabled():
                return 0
            url = self._endpoint()
            if not url:
                return 0
            with self._lock:
                self.queue.drop_old()
                for item in self.queue.due():
                    if self._stop.is_set() or not self._enabled():          # switched off while a pass runs: stop at once
                        break
                    try:
                        status = self._post(url, json.dumps(item.payload, ensure_ascii=False).encode("utf-8"))
                    except Exception as exc:  # noqa: BLE001 - network down, timeout, refused: silent, retried later
                        self.queue.failed(item, f"{type(exc).__name__}: {exc}")
                        log.debug("sharing: upload failed: %s", exc)
                        continue
                    if 200 <= status < 300:
                        if self._enabled():                                 # a switch-off during the request has already cleared the queue
                            self.queue.remove(item.item_id)
                            self.queue.record(success=True)
                            sent += 1
                    else:
                        self.queue.failed(item, f"HTTP {status}")
        except Exception:  # noqa: BLE001 - sharing must never break the program
            log.debug("sharing: pass failed", exc_info=True)
        return sent

    # ------------------------------------------------------------------ background thread
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="meletra-sharing", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.run_once()
            self._wake.wait(INTERVAL_SECONDS)
            self._wake.clear()

    def nudge(self) -> None:
        """Try soon (a transcript was just queued)."""
        self._wake.set()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
