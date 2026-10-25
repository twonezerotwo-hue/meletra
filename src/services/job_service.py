"""Jobs: one small JSON file per transcription, so the app can offer "unfinished jobs" after a crash or a reboot.

States: ``queued`` -> ``running`` -> ``paused`` | ``completed`` | ``failed``.
The job id is the pipeline's ``job_key`` (hash of source path + size), which is also the name of the job's checkpoint folder:
choosing the same file again finds the same job and its checkpoint, whatever happened to the app in between.
"""

from __future__ import annotations

import ctypes
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable, Optional

STATES = ("queued", "running", "paused", "completed", "failed")
UNFINISHED = ("queued", "running", "paused", "failed")


REPLACE_RETRIES = 12
READ_RETRIES = 6
REPLACE_PAUSE_SECONDS = 0.02


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


@dataclass
class JobRecord:
    job_id: str
    source: str
    source_name: str = ""
    source_size: int = 0
    source_mtime: float = 0.0
    output_dir: str = ""
    checkpoint_dir: str = ""
    model_key: str = ""
    language: str = "auto"
    subject: str = "auto"
    mode: str = "quality"
    pdf: str = ""
    state: str = "queued"
    progress: float = 0.0                 # percent 0-100
    position_seconds: float = 0.0
    duration_seconds: float = 0.0
    chunk: int = 0
    chunks: int = 0
    created: str = field(default_factory=now)
    updated: str = field(default_factory=now)
    last_error: str = ""
    summary: dict[str, Any] = field(default_factory=dict)
    pid: int = 0
    # --- the archive (4.1). A record written by an older version lacks these and simply gets the defaults: nothing has to be migrated.
    display_name: str = ""                # what the student named the lecture; "" = the file name. The recording itself is never renamed.
    output_language: str = ""             # the language asked for (Output Language); "" = not recorded (4.0 lectures: the spoken language)
    detected_language: str = ""           # what Whisper heard (set when the job completes)
    translation: str = ""                 # "" | "none" | "whisper" | "m2m100" | "model_missing"
    hidden: bool = False                  # "Remove from Archive": the entry leaves the lists, the lecture's files stay on disk
    detected_languages: list[str] = field(default_factory=list)   # 4.1.1: every language recognised in the lecture (a recording can change language)
    translations: list[str] = field(default_factory=list)         # 4.1.1: the Output Languages that exist as saved translations of this lecture
    output_root: str = ""                 # 4.1.2: the folder the student chose for the results; once the job is finished ``output_dir`` is the LECTURE's own folder inside it

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobRecord":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def lecture_title(record: JobRecord) -> str:
    """The name shown for a lecture: the student's name for it, else the file name, else the folder name."""
    return record.display_name or record.source_name or (Path(record.output_dir).name if record.output_dir else "") or "Untitled lecture"


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform != "win32":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)        # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        ok = ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        return bool(ok) and code.value == 259                                # STILL_ACTIVE
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


class JobStore:
    def __init__(self, directory: Path, alive: Any = pid_alive) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._alive = alive

    def _path(self, job_id: str) -> Path:
        return self.directory / f"{job_id}.json"

    def save(self, record: JobRecord, touch: bool = True) -> None:
        """Write a record. ``touch=False`` keeps its ``updated`` time (an archive repair must not make old lectures look new)."""
        self._write(record, touch)

    def repair(self, job_id: str, **changes: Any) -> Optional[JobRecord]:
        """Change fields of a record without touching its ``updated`` time."""
        record = self.get(job_id)
        if record is None:
            return None
        for key, value in changes.items():
            setattr(record, key, value)
        self._write(record, touch=False)
        return record

    def _write(self, record: JobRecord, touch: bool = True) -> None:
        if touch:
            record.updated = now()
        tmp = self._path(record.job_id).with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(record.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
        for attempt in range(REPLACE_RETRIES):
            try:
                os.replace(tmp, self._path(record.job_id))
                return
            except PermissionError:
                # Windows refuses to replace a file another process has open for reading (the GUI polls these files).
                # A silently lost "completed"/"paused" update would leave a finished job looking interrupted.
                if attempt == REPLACE_RETRIES - 1:
                    tmp.unlink(missing_ok=True)
                    raise
                time.sleep(REPLACE_PAUSE_SECONDS * (attempt + 1))

    # ------------------------------------------------------------------ create / read
    def create_or_get(self, job_id: str, source: Path, **fields_: Any) -> JobRecord:
        """The existing job for this source (any state) or a new queued one. Request fields refresh the record."""
        record = self.get(job_id)
        source = Path(source)
        stat = source.stat() if source.exists() else None
        if record is None:
            record = JobRecord(job_id=job_id, source=str(source), source_name=source.name)
        record.source, record.source_name = str(source), source.name
        if stat is not None:
            record.source_size, record.source_mtime = stat.st_size, stat.st_mtime
        for key, value in fields_.items():
            setattr(record, key, value)
        self._write(record)
        return record

    def get(self, job_id: str) -> Optional[JobRecord]:
        if "." in job_id or "/" in job_id or "\\" in job_id:
            return None
        for attempt in range(READ_RETRIES):
            try:
                return JobRecord.from_dict(json.loads(self._path(job_id).read_text(encoding="utf-8")))
            except PermissionError:                      # being replaced right now: try again instead of "no such job"
                time.sleep(REPLACE_PAUSE_SECONDS * (attempt + 1))
            except (OSError, ValueError, TypeError):
                return None
        return None

    def list(self, states: Optional[Iterable[str]] = None) -> list[JobRecord]:
        wanted = set(states) if states is not None else None
        records = []
        for path in self.directory.glob("*.json"):
            if "." in path.stem:                                  # <id>.request.json is a job's request, not a second job (it used to be listed as a "Queued" lecture named after the output folder)
                continue
            record = self.get(path.stem)
            if record is not None and (wanted is None or record.state in wanted):
                records.append(record)
        return sorted(records, key=lambda r: r.updated, reverse=True)

    def unfinished(self) -> list[JobRecord]:
        """Jobs the user could resume. A job still marked ``running`` whose worker is gone was interrupted: it is reported
        as paused (its checkpoint is intact), with the reason."""
        self.reconcile()
        return self.list(UNFINISHED)

    def find_for_source(self, source: Path) -> Optional[JobRecord]:
        """The unfinished job for exactly this file (same path and size), if any."""
        source = Path(source)
        try:
            size = source.stat().st_size
        except OSError:
            return None
        resolved = str(source.resolve()).lower()
        for record in self.unfinished():
            if str(Path(record.source).resolve()).lower() == resolved and record.source_size == size:
                return record
        return None

    def reconcile(self) -> int:
        """Mark orphaned ``running`` jobs as paused. Returns how many were fixed."""
        fixed = 0
        for record in self.list(["running"]):
            if not self._alive(record.pid):
                record.state = "paused"
                record.last_error = record.last_error or "Interrupted: the application was closed or crashed."
                self._write(record)
                fixed += 1
        return fixed

    # ------------------------------------------------------------------ transitions
    def update(self, job_id: str, **changes: Any) -> Optional[JobRecord]:
        record = self.get(job_id)
        if record is None:
            return None
        if "state" in changes and changes["state"] not in STATES:
            raise ValueError(f"unknown job state {changes['state']!r}")
        for key, value in changes.items():
            setattr(record, key, value)
        self._write(record)
        return record

    def mark_running(self, job_id: str, pid: int = 0) -> Optional[JobRecord]:
        return self.update(job_id, state="running", pid=pid or os.getpid(), last_error="")

    def mark_paused(self, job_id: str, **extra: Any) -> Optional[JobRecord]:
        return self.update(job_id, state="paused", pid=0, **extra)

    def mark_completed(self, job_id: str, summary: Optional[dict[str, Any]] = None) -> Optional[JobRecord]:
        return self.update(job_id, state="completed", pid=0, progress=100.0, summary=summary or {}, last_error="")

    def mark_failed(self, job_id: str, error: str) -> Optional[JobRecord]:
        return self.update(job_id, state="failed", pid=0, last_error=error[:800])

    def apply_event(self, job_id: str, event: dict[str, Any]) -> None:
        """Keep the record in step with pipeline events (progress, chunk numbers)."""
        kind = event.get("type")
        if kind == "chunk_progress":
            self.update(job_id, progress=round(float(event.get("percent", 0.0)), 1),
                        position_seconds=float(event.get("processed", 0.0)), duration_seconds=float(event.get("total", 0.0)),
                        chunk=int(event.get("chunk", 0)), chunks=int(event.get("chunks", 0)))
        elif kind == "paused":
            self.mark_paused(job_id, position_seconds=float(event.get("position_seconds", 0.0)))
        elif kind == "completed":
            translation = event.get("translation") or {}
            current = self.get(job_id)
            lecture_dir = str(event.get("output_dir") or "")
            extra: dict[str, Any] = {}
            if lecture_dir:                                    # 4.1.2: from now on the record names the LECTURE's folder; the folder the student chose is kept as output_root
                extra = {"output_dir": lecture_dir, "output_root": (current.output_root or current.output_dir) if current is not None else ""}
            self.update(job_id, detected_language=str(event.get("language") or ""), translation=str(translation.get("method") or ""),
                        duration_seconds=float(event.get("audio_seconds") or 0.0), detected_languages=[str(c) for c in (event.get("languages") or [])],
                        translations=[str(c) for c in (event.get("translations") or [])], **extra)
            self.mark_completed(job_id, {k: v for k, v in event.items() if k != "type"})
        elif kind == "failed":
            self.mark_failed(job_id, str(event.get("error", "failed")))

    # ------------------------------------------------------------------ the archive (Recent lectures, Files)
    def rename(self, job_id: str, name: str) -> Optional[JobRecord]:
        """Change ONLY the name shown for the lecture. The recording and the transcript folder keep their names."""
        return self.update(job_id, display_name=" ".join((name or "").split())[:160])

    def set_translations(self, output_dir: Path, codes: Iterable[str]) -> int:
        """Remember which saved translations the lecture in ``output_dir`` has (the cards in Recent lectures and Files show them). Returns the number of records updated."""
        wanted = os.path.normcase(os.path.abspath(str(output_dir)))
        changed = 0
        for record in self.list():
            if record.output_dir and os.path.normcase(os.path.abspath(record.output_dir)) == wanted:
                self.update(record.job_id, translations=list(dict.fromkeys(codes)))
                changed += 1
        return changed

    def hide(self, job_id: str) -> Optional[JobRecord]:
        """Remove a lecture from the archive lists. Nothing on disk is touched: its transcript, study pack and recording stay where they are."""
        return self.update(job_id, hidden=True)

    def archive(self, query: str = "", sort: str = "newest", states: Optional[Iterable[str]] = None, limit: int = 0,
                include_hidden: bool = False) -> list[JobRecord]:
        """Every lecture the app knows (completed ones stay available even when the recording was deleted), filtered by a search in the shown name
        AND the original file name, sorted ``newest`` | ``oldest`` | ``name`` | ``duration``."""
        records = [r for r in self.list(states) if include_hidden or not r.hidden]
        needle = " ".join((query or "").casefold().split())
        if needle:
            records = [r for r in records if needle in lecture_title(r).casefold() or needle in (r.source_name or "").casefold()]
        if sort == "oldest":
            records.sort(key=lambda r: r.created)
        elif sort == "name":
            records.sort(key=lambda r: lecture_title(r).casefold())
        elif sort == "duration":
            records.sort(key=lambda r: r.duration_seconds, reverse=True)
        else:
            records.sort(key=lambda r: r.updated, reverse=True)
        return records[:limit] if limit else records

    def adopt_folder(self, folder: Path) -> JobRecord:
        """Put a finished lecture folder (made by this program, now on another disk or from an older install) into the archive. Idempotent."""
        import hashlib

        folder = Path(folder)
        job_id = "adopted-" + hashlib.sha1(str(folder.resolve()).lower().encode("utf-8")).hexdigest()[:12]
        record = self.get(job_id)
        if record is not None:
            if record.hidden:
                self.update(job_id, hidden=False)
            return self.get(job_id) or record
        meta: dict[str, Any] = {}
        try:
            meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        source = (meta.get("source") or {}).get("file") or folder.name
        language = meta.get("language") or {}
        record = JobRecord(job_id=job_id, source=str(folder / source), source_name=str(source), output_dir=str(folder), state="completed", progress=100.0,
                           duration_seconds=float(((meta.get("source") or {}).get("probe") or {}).get("duration_s") or meta.get("duration_seconds") or 0.0),
                           language=str(language.get("requested") or "auto"), detected_language=str(language.get("detected") or ""))
        self._write(record)
        return record

    def delete(self, job_id: str) -> None:
        self._path(job_id).unlink(missing_ok=True)

    def prune(self, keep_completed: int = 20) -> int:
        """Forget old completed jobs (their transcripts stay where the user put them)."""
        done = self.list(["completed"])
        removed = 0
        for record in done[keep_completed:]:
            self.delete(record.job_id)
            removed += 1
        return removed
