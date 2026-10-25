"""Crash-safe progress for long recordings: every finished segment is appended to disk.

If a run dies (CUDA error, power loss, Ctrl+C), the same command resumes after the last saved
segment instead of starting over. Timestamps stay absolute: the resumed decode runs on the audio
*after* ``resume_from`` and the pipeline adds that offset back (faster-whisper ignores ``vad_filter``
when ``clip_timestamps`` is used, so cutting the audio keeps VAD active).

Layout (``<output>/checkpoint/``):
    checkpoint.json        fingerprint of source+settings, the run plan (language, subjects, hotwords), status
    partial_segments.jsonl one JSON segment per line, flushed as it is produced
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Optional

from .schema import Segment

log = logging.getLogger(__name__)

META_NAME = "checkpoint.json"
SEGMENTS_NAME = "partial_segments.jsonl"
FSYNC_INTERVAL_S = 10.0


def file_sha256(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def make_fingerprint(source_sha256: str, **settings: Any) -> dict[str, Any]:
    """What must be identical for a checkpoint to be reusable (changed settings => fresh start)."""
    return {"source_sha256": source_sha256, **settings}


class SegmentCheckpoint:
    """Append-only segment log plus a small metadata file."""

    def __init__(self, directory: Path, fingerprint: dict[str, Any]) -> None:
        self.directory = Path(directory)
        self.fingerprint = fingerprint
        self.meta_path = self.directory / META_NAME
        self.segments_path = self.directory / SEGMENTS_NAME
        self._handle: Optional[Any] = None
        self._count = 0
        self._base_count = 0
        self._last_fsync = 0.0

    # ------------------------------------------------------------------ loading
    def load(self) -> tuple[list[Segment], Optional[dict[str, Any]]]:
        """Return ``(saved_segments, plan)`` if a matching checkpoint exists, else ``([], None)``.

        A torn last line (crash while writing) is dropped. A different fingerprint means the source
        or the settings changed, so the old checkpoint is ignored.
        """
        if not self.meta_path.is_file():
            return [], None
        try:
            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            log.warning("Checkpoint metadata unreadable; starting fresh")
            return [], None
        if meta.get("fingerprint") != self.fingerprint:
            log.warning("Existing checkpoint belongs to different settings/source; starting fresh")
            return [], None
        segments: list[Segment] = []
        if self.segments_path.is_file():
            with self.segments_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        segments.append(Segment.from_checkpoint(json.loads(line)))
                    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                        break  # torn / corrupt tail: keep everything before it
        return segments, meta.get("plan")

    def read_status(self) -> Optional[str]:
        """'planned' | 'running' | 'completed' (None if there is no readable checkpoint for these settings)."""
        if not self.meta_path.is_file():
            return None
        try:
            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return meta.get("status") if meta.get("fingerprint") == self.fingerprint else None

    # ------------------------------------------------------------------ writing
    def begin(self, plan: dict[str, Any], resumed: list[Segment]) -> None:
        """Start (or continue) logging. ``resumed`` segments are rewritten so the file is always consistent."""
        self.directory.mkdir(parents=True, exist_ok=True)
        self._write_meta(plan, status="running")
        with self.segments_path.open("w", encoding="utf-8", newline="\n") as handle:
            for seg in resumed:
                handle.write(json.dumps(seg.to_checkpoint(), ensure_ascii=False) + "\n")
        self._handle = self.segments_path.open("a", encoding="utf-8", newline="\n")
        self._count = self._base_count = len(resumed)
        self._last_fsync = time.monotonic()

    def save_plan(self, plan: dict[str, Any]) -> None:
        """Persist only the plan (used by the ``probe`` command: decisions are made, no segments yet)."""
        self.directory.mkdir(parents=True, exist_ok=True)
        self._write_meta(plan, status="planned")

    def _write_meta(self, plan: dict[str, Any], status: str, stats: Optional[dict[str, Any]] = None) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = {"fingerprint": self.fingerprint, "plan": plan, "status": status,
                   "updated": time.strftime("%Y-%m-%dT%H:%M:%S")}
        if stats is None:                       # keep the statistics written by an earlier mark()
            stats = self._read_stats()
        if stats:
            payload["stats"] = stats
        tmp = self.meta_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.meta_path)

    def add(self, segment: Segment) -> None:
        if self._handle is None:
            raise RuntimeError("checkpoint not started; call begin() first")
        self._handle.write(json.dumps(segment.to_checkpoint(), ensure_ascii=False) + "\n")
        self._handle.flush()
        self._count += 1
        now = time.monotonic()
        if now - self._last_fsync >= FSYNC_INTERVAL_S:
            os.fsync(self._handle.fileno())
            self._last_fsync = now

    def rollback(self) -> None:
        """Drop segments added since ``begin`` (a decode attempt is being repeated, e.g. after a GPU fallback)."""
        if self._handle is None:
            return
        self._handle.close()
        kept: list[str] = []
        with self.segments_path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index >= self._base_count:
                    break
                kept.append(line)
        self.segments_path.write_text("".join(kept), encoding="utf-8", newline="\n")
        self._handle = self.segments_path.open("a", encoding="utf-8", newline="\n")
        self._count = self._base_count

    def close(self) -> None:
        if self._handle is not None:
            try:
                self._handle.flush()
                os.fsync(self._handle.fileno())
            except OSError:
                pass
            self._handle.close()
            self._handle = None

    def mark(self, plan: dict[str, Any], status: str, stats: Optional[dict[str, Any]] = None) -> None:
        """Record the status. ``stats`` (e.g. decode seconds / audio seconds of a finished chunk) is kept across later marks."""
        self._write_meta(plan, status, stats)

    def _read_stats(self) -> Optional[dict[str, Any]]:
        try:
            return json.loads(self.meta_path.read_text(encoding="utf-8")).get("stats")
        except (OSError, json.JSONDecodeError, AttributeError):
            return None

    def read_stats(self) -> dict[str, Any]:
        """Statistics stored with the checkpoint (empty dict if none), for cumulative reporting after a resume."""
        return self._read_stats() or {}

    def discard(self) -> None:
        """Remove the checkpoint after a fully successful run."""
        self.close()
        shutil.rmtree(self.directory, ignore_errors=True)

    @property
    def saved_count(self) -> int:
        return self._count
