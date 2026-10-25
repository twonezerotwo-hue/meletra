"""The batch queue: up to 10 audio/video files (plus any documents) one after another.

The queue itself does no transcription. For every item it calls the ``start`` function it was given - the GUI starts the proven
0.1/0.2 worker through the existing ``JobController`` - and is told how that job ended. One job runs at a time. Pause uses the
worker's own safe-pause (checkpoint); finished items never run again; a failed item does not stop the others.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from .router import MAX_QUEUE, Detected

QUEUED, PROCESSING, COMPLETED, FAILED, PAUSED, CANCELLED = "queued", "processing", "completed", "failed", "paused", "cancelled"
IDLE, RUNNING, DONE = "idle", "running", "done"                       # queue states (PAUSED / CANCELLED are shared with the items)
SAME_ERROR_LIMIT = 2                                                  # this many identical failures in a row: the problem is not the file


class QueueError(ValueError):
    pass


@dataclass
class QueueItem:
    detected: Detected
    state: str = QUEUED
    error: str = ""
    output_dir: str = ""
    files: dict[str, str] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.detected.path.name


class BatchQueue:
    def __init__(self, items: list[Detected], start: Callable[[QueueItem], None], pause: Callable[[], None],
                 cancel: Callable[[], None], changed: Optional[Callable[[], None]] = None) -> None:
        if sum(1 for d in items if d.is_media) > MAX_QUEUE:
            raise QueueError(f"You selected {sum(1 for d in items if d.is_media)} audio/video files. The limit is {MAX_QUEUE} at a time.")
        self.items = [QueueItem(d) for d in items]
        self._start, self._pause, self._cancel = start, pause, cancel
        self._changed = changed or (lambda: None)
        self.state = IDLE
        self._pause_requested = False
        self._recent_errors: list[str] = []

    # ------------------------------------------------------------------ views
    @property
    def current(self) -> Optional[QueueItem]:
        return next((i for i in self.items if i.state in (PROCESSING, PAUSED)), None)

    @property
    def position(self) -> tuple[int, int]:
        """(number of the item being worked on, total)."""
        item = self.current
        done = sum(1 for i in self.items if i.state in (COMPLETED, FAILED, CANCELLED))
        return (self.items.index(item) + 1 if item else min(done + 1, len(self.items)), len(self.items))

    def count(self, state: str) -> int:
        return sum(1 for i in self.items if i.state == state)

    # ------------------------------------------------------------------ control
    def start(self) -> None:
        if self.state in (IDLE, DONE):
            self.state = RUNNING
            self._advance()

    def pause(self) -> None:
        """Ask the running job to stop at a safe point; the queue waits (``item_paused``) and does not start the next item."""
        if self.state == RUNNING and self.current is not None and not self._pause_requested:
            self._pause_requested = True
            self._pause()

    def resume(self) -> None:
        if self.state != PAUSED:
            return
        item = self.current
        self.state, self._pause_requested = RUNNING, False
        if item is not None:
            item.state = PROCESSING
            self._run(item)
        else:
            self._advance()

    def cancel(self) -> None:
        """Stop the running job (its progress stays saved) and everything still waiting. Finished items are kept."""
        if self.state in (CANCELLED, DONE):
            return
        running = any(i.state == PROCESSING for i in self.items)
        for item in self.items:
            if item.state in (QUEUED, PROCESSING, PAUSED):
                item.state = CANCELLED
        self.state = CANCELLED
        if running:
            self._cancel()
        self._changed()

    # ------------------------------------------------------------------ how the running job ended
    def item_completed(self, output_dir: str = "", files: Optional[dict[str, str]] = None) -> None:
        item = self._active()
        if item is None:
            return
        item.state, item.output_dir, item.files = COMPLETED, output_dir, files or {}
        self._recent_errors.clear()
        self._next()

    def item_failed(self, error: str) -> None:
        item = self._active()
        if item is None:
            return
        item.state, item.error = FAILED, error
        self._recent_errors = (self._recent_errors + [error])[-SAME_ERROR_LIMIT:]
        if len(self._recent_errors) == SAME_ERROR_LIMIT and len(set(self._recent_errors)) == 1:
            for rest in self.items:
                if rest.state == QUEUED:
                    rest.state, rest.error = CANCELLED, "Stopped: the same problem happened twice in a row."
        self._next()

    def item_paused(self) -> None:
        item = self._active()
        if item is None:
            return
        item.state, self.state = PAUSED, PAUSED
        self._changed()

    # ------------------------------------------------------------------ internals
    def _active(self) -> Optional[QueueItem]:
        """The item a job report belongs to; a report that arrives after cancel() is ignored."""
        return next((i for i in self.items if i.state == PROCESSING), None) if self.state in (RUNNING, PAUSED) else None

    def _next(self) -> None:
        if self.state == RUNNING and self._pause_requested and any(i.state == QUEUED for i in self.items):
            self.state = PAUSED                                        # paused between two items: the next one waits for resume()
        elif self.state == RUNNING:
            self._pause_requested = False
            self._advance()
            return
        self._changed()

    def _advance(self) -> None:
        item = next((i for i in self.items if i.state == QUEUED), None)
        if item is None:
            self.state = DONE
            self._changed()
            return
        item.state = PROCESSING
        self._run(item)

    def _run(self, item: QueueItem) -> None:
        self._changed()
        try:
            self._start(item)
        except Exception as exc:                                       # noqa: BLE001 - a start problem is that item's failure
            self.item_failed(str(exc) or exc.__class__.__name__)


# ------------------------------------------------------------------------------------------------ files
NAMED = {"raw": ("local_raw_transcript.txt", "_raw.txt"), "clean": ("local_clean_transcript.txt", "_clean.txt"),
         "timestamped": ("local_transcript_with_timestamps.txt", "_timestamped.txt"), "srt": ("local_transcript.srt", ".srt"),
         "vtt": ("local_transcript.vtt", ".vtt"), "json": ("local_transcript.json", ".json")}
WINDOWS_SAFE_PATH = 240


def _named_stem(output_dir: Path, stem: str) -> str:
    """Keep convenience-copy paths below the legacy Windows limit; canonical transcript names are already short."""
    reserve = max(len(suffix) for _, suffix in NAMED.values()) + len(" (9999)")
    available = WINDOWS_SAFE_PATH - len(str(Path(output_dir).resolve())) - 1 - reserve
    if len(stem) <= available:
        return stem
    digest = hashlib.sha256(stem.encode("utf-8")).hexdigest()[:8]
    return f"{stem[:max(available - len(digest) - 1, 1)].rstrip(' .')}-{digest}"


def _free_name(path: Path) -> Path:
    candidate, n = path, 1
    while candidate.exists():
        n += 1
        candidate = path.with_name(f"{path.stem} ({n}){path.suffix}")
    return candidate


def export_named(output_dir: Path, source: Path) -> dict[str, str]:
    """Copies named after the source (``Lecture_01_raw.txt``, ``Lecture_01.srt`` ...) next to the pipeline's own files, which stay
    untouched (the study screens read them). A name that is already taken by different content is never overwritten."""
    from ..exporters import safe_name

    stem, made = _named_stem(Path(output_dir), safe_name(Path(source).stem)), {}
    for key, (original, suffix) in NAMED.items():
        src = Path(output_dir) / original
        if not src.is_file():
            continue
        target = Path(output_dir) / f"{stem}{suffix}"
        if target.exists() and target.read_bytes() == src.read_bytes():
            made[key] = str(target)
            continue
        target = _free_name(target)
        shutil.copyfile(src, target)
        made[key] = str(target)
    return made


def batch_output_root(base: Path, now: Optional[datetime] = None) -> Path:
    """``<base>/Batch_<timestamp>`` (a second batch in the same second gets its own folder)."""
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    candidate, n = Path(base) / f"Batch_{stamp}", 1
    while candidate.exists():
        n += 1
        candidate = Path(base) / f"Batch_{stamp}_{n}"
    return candidate


def prepare_media(detected: Detected, work_dir: Path, progress: Optional[Callable[[float], None]] = None,
                  should_stop: Optional[Callable[[], bool]] = None) -> Path:
    """The path the transcription pipeline should read. Normally the file itself; media whose extension the pipeline refuses (or that
    has none) is first decoded to a 16 kHz WAV with the same PyAV routine the pipeline uses. The original is never modified."""
    if not detected.needs_wav:
        return detected.path
    from ..audio import extract_audio_pyav
    from ..config import load_transcription_config
    from ..exporters import safe_name

    work_dir.mkdir(parents=True, exist_ok=True)
    target = work_dir / f"{safe_name(detected.path.stem)}.wav"
    if not target.is_file():
        extract_audio_pyav(detected.path, target, load_transcription_config()["audio"], progress, should_stop)
    return target
