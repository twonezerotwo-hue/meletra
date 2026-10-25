"""The local upload queue: one small JSON file per waiting transcript, in the program's data folder. Survives restarts; deleted entirely when sharing is switched off."""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

#: pause before retry number n (seconds); the last value repeats. A queued item older than MAX_AGE_DAYS is dropped (nothing is kept forever).
BACKOFF_SECONDS = (300, 1800, 7200, 21600, 86400)
MAX_AGE_DAYS = 14


@dataclass
class QueueItem:
    item_id: str
    created: float
    attempts: int = 0
    next_try: float = 0.0
    last_error: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


class UploadQueue:
    def __init__(self, folder: Path) -> None:
        self.folder = Path(folder)
        self.items_dir = self.folder / "queue"
        self.status_file = self.folder / "status.json"
        self.log_file = self.folder / "upload.log"

    # ------------------------------------------------------------------ items
    def _path(self, item_id: str) -> Path:
        return self.items_dir / f"{item_id}.json"

    def enqueue(self, payload: dict[str, Any], now: Optional[float] = None) -> QueueItem:
        self.items_dir.mkdir(parents=True, exist_ok=True)
        item = QueueItem(item_id=uuid.uuid4().hex, created=now if now is not None else time.time(), payload=payload)
        self._write(item)
        return item

    def _write(self, item: QueueItem) -> None:
        tmp = self._path(item.item_id).with_suffix(".tmp")
        tmp.write_text(json.dumps(item.__dict__, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self._path(item.item_id))

    def all(self) -> Iterator[QueueItem]:
        if not self.items_dir.is_dir():
            return
        for file in sorted(self.items_dir.glob("*.json")):
            try:
                data = json.loads(file.read_text(encoding="utf-8"))
                yield QueueItem(**{k: data[k] for k in QueueItem.__dataclass_fields__ if k in data})
            except (OSError, ValueError, TypeError):
                continue

    def due(self, now: Optional[float] = None) -> list[QueueItem]:
        now = now if now is not None else time.time()
        return [i for i in self.all() if i.next_try <= now]

    def count(self) -> int:
        return sum(1 for _ in self.all())

    def remove(self, item_id: str) -> None:
        self._path(item_id).unlink(missing_ok=True)

    def failed(self, item: QueueItem, error: str, now: Optional[float] = None) -> None:
        """The attempt failed: keep the item, try again later (growing pauses). Nothing is shown to the student."""
        now = now if now is not None else time.time()
        item.attempts += 1
        item.last_error = error[:300]
        item.next_try = now + BACKOFF_SECONDS[min(item.attempts - 1, len(BACKOFF_SECONDS) - 1)]
        self._write(item)
        self.record(failure=error, now=now)

    def drop_old(self, now: Optional[float] = None) -> int:
        now = now if now is not None else time.time()
        old = [i for i in self.all() if now - i.created > MAX_AGE_DAYS * 86400]
        for item in old:
            self.remove(item.item_id)
        return len(old)

    def clear(self) -> int:
        """Delete every waiting item (sharing was switched off). Returns how many were removed."""
        removed = 0
        if self.items_dir.is_dir():
            for file in list(self.items_dir.glob("*")):
                try:
                    file.unlink()
                    removed += 1
                except OSError:
                    pass
        return removed

    # ------------------------------------------------------------------ status (Settings -> Privacy & Data -> Upload status)
    def status(self) -> dict[str, Any]:
        try:
            data = json.loads(self.status_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        data = {"sent": int(data.get("sent", 0)), "failed_attempts": int(data.get("failed_attempts", 0)), "last_attempt": data.get("last_attempt"),
                "last_success": data.get("last_success"), "last_error": data.get("last_error", "")}
        data["queued"] = self.count()
        return data

    def record(self, success: bool = False, failure: str = "", now: Optional[float] = None) -> None:
        now = now if now is not None else time.time()
        data = self.status()
        data.pop("queued", None)
        data["last_attempt"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
        if success:
            data["sent"] += 1
            data["last_success"] = data["last_attempt"]
            data["last_error"] = ""
        else:
            data["failed_attempts"] += 1
            data["last_error"] = failure[:300]
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            self.status_file.write_text(json.dumps(data), encoding="utf-8")
            with self.log_file.open("a", encoding="utf-8") as handle:
                handle.write(f"{data['last_attempt']}  {'sent' if success else 'failed: ' + failure[:200]}\n")
            if self.log_file.stat().st_size > 200_000:
                lines = self.log_file.read_text(encoding="utf-8").splitlines()[-200:]
                self.log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except OSError:
            pass

    def reset_status(self) -> None:
        for file in (self.status_file, self.log_file):
            file.unlink(missing_ok=True)
