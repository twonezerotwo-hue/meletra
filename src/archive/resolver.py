"""record -> lecture folder, and what is known about a lecture. The one place every screen asks."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .artifacts import Artifacts, find_artifacts, read_job_id
from .transcript import TranscriptText, load_transcript

COMPLETED, RECOVERED, MISSING = "completed", "recovered", "transcript_missing"
STATUS_LABELS = {
    "completed": "Completed",
    "recovered": "Completed - metadata recovered",
    "transcript_missing": "Transcript not found",
    "queued": "Queued",
    "running": "Processing",
    "paused": "Paused",
    "interrupted": "Interrupted",
    "failed": "Failed",
    "study_only": "Study material only",
}

_root_cache: dict[str, tuple[int, dict[str, Path], dict[str, Path]]] = {}


def _is_lecture(folder: Path) -> bool:
    try:
        return folder.is_dir() and find_artifacts(folder).is_lecture
    except OSError:
        return False


def _index_root(root: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    """``({job_id: folder}, {source file name: folder})`` of the lecture folders directly inside ``root`` (and inside its ``Batch_*`` folders). Cached until the folder changes."""
    try:
        stamp = root.stat().st_mtime_ns
    except OSError:
        return {}, {}
    key = os.path.normcase(str(root))
    cached = _root_cache.get(key)
    if cached and cached[0] == stamp:
        return cached[1], cached[2]
    by_id: dict[str, Path] = {}
    by_source: dict[str, Path] = {}

    def look(folder: Path, depth: int) -> None:
        try:
            children = [c for c in folder.iterdir() if c.is_dir()]
        except OSError:
            return
        for child in children:
            if _is_lecture(child):
                job_id = read_job_id(child)
                if job_id:
                    by_id.setdefault(job_id, child)
                source = source_file_of(child)
                if source:
                    by_source.setdefault(source.casefold(), child)
            elif depth == 0 and child.name.casefold().startswith("batch_"):
                look(child, 1)

    look(root, 0)
    _root_cache[key] = (stamp, by_id, by_source)
    return by_id, by_source


def source_file_of(folder: Path) -> str:
    """The recording's file name as ``metadata.json`` of the lecture recorded it."""
    try:
        meta = json.loads((Path(folder) / "metadata.json").read_text(encoding="utf-8-sig"))
        return str((meta.get("source") or {}).get("file") or "")
    except (OSError, ValueError, AttributeError):
        return ""


def lecture_folder(record) -> Optional[Path]:
    """The folder that holds this lecture's files, or ``None``.

    A finished job's ``output_dir`` is the folder the student chose (``...\\Medical Transcription``), the lecture's own folder is a sub-folder of it. In order: the folder the job reported
    when it finished, ``output_dir`` itself when it is the lecture, the sub-folder whose ``.job_id`` is this job, the sub-folder named like the recording, the sub-folder whose
    ``metadata.json`` names this recording."""
    summary = getattr(record, "summary", None) or {}
    reported = str(summary.get("output_dir") or "")
    output_dir = str(getattr(record, "output_dir", "") or "")
    for text in (reported, output_dir):
        if text and _is_lecture(Path(text)):
            return Path(text)
    root = Path(output_dir) if output_dir else (Path(reported).parent if reported else None)
    if root is None or not root.is_dir():
        return None
    by_id, by_source = _index_root(root)
    job_id = str(getattr(record, "job_id", "") or "")
    if job_id and job_id in by_id:
        return by_id[job_id]
    name = str(getattr(record, "source_name", "") or "")
    if name:
        from ..exporters import safe_name

        stem = safe_name(Path(name).stem)
        guess = root / stem
        if _is_lecture(guess):
            return guess
        if name.casefold() in by_source:
            return by_source[name.casefold()]
    return None


def _read_json(path: Optional[Path]) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


@dataclass
class LectureFacts:
    folder: Optional[Path]
    transcript_available: bool
    status: str
    status_label: str
    words: Optional[int] = None
    duration_seconds: Optional[float] = None
    duration_source: str = ""              # recorded | metadata | timestamps | "" (unknown)
    language: str = ""
    languages: list[str] = field(default_factory=list)
    processing_seconds: Optional[float] = None
    transcript_kind: str = ""
    translations: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _has_job_numbers(record, art: Artifacts) -> bool:
    """The lecture's own run recorded its numbers: the job's summary has the word count, or the lecture folder's metadata has it."""
    if not art.metadata:
        return False
    if isinstance((getattr(record, "summary", None) or {}).get("words"), int):
        return True
    words = (_read_json(art.metadata).get("quality") or {}).get("words")
    return isinstance(words, int) and words > 0


def quick_status(record, folder: Optional[Path] = None, transcript_ok: Optional[bool] = None) -> tuple[str, str]:
    """``(status, label)`` of a lecture - the ONE answer every screen shows (cards, Files, the result screen, Technical details, Study). It reads no transcript text: the job state,
    and for a completed job whether its transcript is really there and whether its numbers were recorded. ``transcript_ok`` lets a caller that already read the transcript say so."""
    state = str(getattr(record, "state", "") or "")
    if state != "completed":
        if state == "paused" and "interrupted" in str(getattr(record, "last_error", "") or "").lower():
            return "interrupted", STATUS_LABELS["interrupted"]
        return state, STATUS_LABELS.get(state, state.title())
    folder = folder if folder is not None else lecture_folder(record)
    if folder is None:
        return MISSING, STATUS_LABELS[MISSING]
    art = find_artifacts(folder)
    if transcript_ok is None:
        transcript_ok = art.has_text_transcript or (art.has_transcript and load_transcript(art) is not None)        # only a lecture with JSON files alone is read to be sure it is not damaged
    if transcript_ok:
        status = COMPLETED if _has_job_numbers(record, art) else RECOVERED
        return status, STATUS_LABELS[status]
    if art.has_study:
        return "study_only", STATUS_LABELS["study_only"]
    return MISSING, STATUS_LABELS[MISSING]


def status_kind(status: str) -> str:
    """The colour family of a status chip: a recovered lecture is as good as a completed one, a lecture without its transcript needs attention."""
    return {COMPLETED: "completed", RECOVERED: "completed", MISSING: "paused", "study_only": "paused", "interrupted": "paused"}.get(status, status)


def lecture_facts(record, folder: Optional[Path] = None) -> LectureFacts:
    """Everything known about a lecture, recovered from its files where the job record is silent. Nothing is invented: what is not recorded is ``None`` / ``""``."""
    folder = folder if folder is not None else lecture_folder(record)
    state = str(getattr(record, "state", "") or "")
    summary = getattr(record, "summary", None) or {}
    if folder is None:
        status, label = quick_status(record, None)
        return LectureFacts(None, False, status, label, notes=["The lecture's folder was not found; its files may have been moved or deleted."] if state == "completed" else [])
    art = find_artifacts(folder)
    meta = _read_json(art.metadata)
    transcript: Optional[TranscriptText] = load_transcript(art)
    notes: list[str] = []
    words = summary.get("words") if isinstance(summary.get("words"), int) and summary.get("words") else None
    if words is None and transcript is not None:
        words = transcript.words
        notes.append("The word count was counted from the saved transcript.")
    duration = float(getattr(record, "duration_seconds", 0.0) or 0.0) or None
    duration_source = "recorded" if duration else ""
    if duration is None:
        probed = ((meta.get("source") or {}).get("probe") or {}).get("duration_s") or (meta.get("timing") or {}).get("audio_duration_total")
        if isinstance(probed, (int, float)) and probed > 0:
            duration, duration_source = float(probed), "metadata"
    if duration is None and transcript is not None and transcript.last_time:
        duration, duration_source = float(transcript.last_time), "timestamps"
        notes.append("The duration is the time of the last line of the transcript.")
    lang_meta = meta.get("language") or {}
    ml = meta.get("multilingual") or {}
    languages = [str(c) for c in (getattr(record, "detected_languages", None) or ml.get("detected_languages") or []) if c]
    language = str(getattr(record, "detected_language", "") or lang_meta.get("detected") or "") or (languages[0] if languages else "")
    timing = meta.get("timing") or {}
    processing = timing.get("wall_time_cumulative") or timing.get("wall_time_this_run")
    processing = float(processing) if isinstance(processing, (int, float)) and processing > 0 else None
    status, label = quick_status(record, folder, transcript_ok=transcript is not None if state == "completed" else None)
    return LectureFacts(folder, transcript is not None, status, label, words, duration, duration_source, language, languages, processing,
                        transcript.kind if transcript else "", sorted(art.translations), notes)


def format_clock(seconds: Optional[float]) -> str:
    """``HH:MM:SS`` or ``Unknown`` - never ``00:00:00`` for a value that is not known."""
    if seconds is None or seconds <= 0:
        return "Unknown"
    total = int(round(seconds))
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def safe_stem(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip()
