"""Synthetic lecture folders shaped like the ones a real MELETRA 4.1.x run leaves behind (no real transcript text is used anywhere)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from src.services.job_service import JobRecord, JobStore

LINES = [
    "Hello everyone and welcome to the lecture on cholinergic regulation of the brain.",
    "The ascending reticular activating system keeps the cortex awake.",
    "Acetylcholine released in the thalamus increases the responsiveness of thalamic neurons.",
    "Noradrenaline from the locus coeruleus raises attention and the heart rate.",
    "A lesion of the pedunculopontine nucleus reduces cortical arousal and REM sleep.",
    "In summary the monoaminergic and cholinergic systems together regulate the sleep wake cycle.",
]


def stamp(seconds: int) -> str:
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def write_lecture(root: Path, folder_name: str, *, source: str = "IMG_0001.MP4", job_id: str = "job0000000000001", duration: float = 2548.0, files: str = "full",
                  study: bool = False, language: str = "en") -> Path:
    """A lecture folder inside ``root``. ``files``: ``full`` (every file the pipeline writes), ``legacy`` (only the timestamped text and ``<name>_clean.txt`` copies), ``timestamped`` (only the
    timestamped text), ``empty`` (no transcript at all)."""
    folder = root / folder_name
    folder.mkdir(parents=True, exist_ok=True)
    if job_id:
        (folder / ".job_id").write_text(job_id, encoding="utf-8")
    (folder / "metadata.json").write_text(json.dumps({"tool": "medical-lecture-transcriber", "created_at": "2026-10-11T08:00:00+00:00",
                                                       "source": {"file": source, "probe": {"duration_s": duration}}, "language": {"requested": "auto", "detected": language},
                                                       "timing": {"wall_time_cumulative": 406.9}, "multilingual": {"detected_languages": [language]}}), encoding="utf-8")
    segments = [{"start": 1.0 + 10 * i, "end": 9.0 + 10 * i, "text": line} for i, line in enumerate(LINES)]
    timestamped = "\n".join(f"[{stamp(int(s['start']))} - {stamp(int(s['end']))}] {s['text']}" for s in segments) + "\n"
    plain = " ".join(LINES)
    if files == "full":
        (folder / "local_transcript.json").write_text(json.dumps({"segments": segments, "language": language}), encoding="utf-8")
        (folder / "local_raw_transcript.txt").write_text(plain, encoding="utf-8")
        (folder / "local_clean_transcript.txt").write_text(plain, encoding="utf-8")
        (folder / "local_transcript_with_timestamps.txt").write_text(timestamped, encoding="utf-8")
    elif files == "legacy":
        stem = Path(source).stem
        (folder / f"{stem}_clean.txt").write_text(plain, encoding="utf-8")
        (folder / f"{stem}_timestamped.txt").write_text(timestamped, encoding="utf-8")
    elif files == "timestamped":
        (folder / "local_transcript_with_timestamps.txt").write_text(timestamped, encoding="utf-8")
    if study:
        (folder / "study").mkdir(exist_ok=True)
        (folder / "study" / "study_pack.json").write_text(json.dumps({"flashcards": [], "questions": []}), encoding="utf-8")
    return folder


def finished_record(store: JobStore, root: Path, folder: Path, *, source: str = "IMG_0001.MP4", job_id: str = "job0000000000001", with_summary: bool = True, **fields) -> JobRecord:
    """A completed job the way the 4.1.1 worker left it: ``output_dir`` is the folder the student CHOSE, the lecture folder is only in the summary."""
    record = JobRecord(job_id=job_id, source=str(root.parent / source), source_name=source, output_dir=str(root), state="completed", progress=100.0, duration_seconds=2548.0,
                       summary={"output_dir": str(folder), "segments": 6, "words": 55} if with_summary else {}, **fields)
    store.save(record)
    return record


def tree_hashes(folder: Path) -> dict[str, str]:
    import hashlib

    return {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.rglob("*")) if p.is_file()}
