"""Make the lecture folders on disk and the archive records agree - safely, and the same way every time.

What it does (and the only things it does):
* a record that names the folder the student chose instead of the lecture's own folder gets the lecture's folder (the chosen folder is kept as ``output_root``);
* a lecture folder (``.job_id``, ``metadata.json``, a transcript or a study pack) that no record points to gets a record built from what the folder itself says;
* two folders claiming one lecture, or a folder that already has a record, never make a second entry;
* a lecture the student removed from the lists (``hidden``) stays removed.

It never deletes, moves or rewrites a transcript, a study pack or a recording, never starts a transcription, and leaves the ``updated`` time of a record alone. What it cannot resolve
(a finished lecture whose folder is gone, a folder with only part of its files) is reported, not removed.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from ..services.job_service import JobRecord, JobStore
from .artifacts import find_artifacts, read_job_id
from .resolver import lecture_folder
from .transcript import load_transcript

MAX_ROOT_CHILDREN = 5000


@dataclass
class ReconcileReport:
    roots: list[str] = field(default_factory=list)
    found: int = 0                       # lecture folders with recognisable files
    already_indexed: int = 0             # ... that had a correct record
    repaired: int = 0                    # records whose folder link was corrected
    recovered: int = 0                   # records created for folders that had none
    duplicates_prevented: int = 0        # folders that would have made a second entry
    hidden_by_user: int = 0              # lectures the student removed from the lists (kept hidden)
    unresolved: list[str] = field(default_factory=list)
    details: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.repaired or self.recovered)

    def summary_lines(self) -> list[str]:
        lines = [f"Lecture folders found: {self.found}", f"Already in the archive: {self.already_indexed}", f"Records repaired: {self.repaired}",
                 f"Lectures restored to the archive: {self.recovered}", f"Duplicates prevented: {self.duplicates_prevented}"]
        if self.hidden_by_user:
            lines.append(f"Kept hidden (you removed them from the lists): {self.hidden_by_user}")
        lines.append(f"Could not be resolved: {len(self.unresolved)}")
        lines += [f"  - {item}" for item in self.unresolved[:12]]
        return lines


def _key(path: Path | str) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


def _children(folder: Path) -> list[Path]:
    try:
        return [c for c in folder.iterdir() if c.is_dir()][:MAX_ROOT_CHILDREN]
    except OSError:
        return []


def discover_roots(records: Iterable[JobRecord], configured_output_dir: str = "", extra: Iterable[Path] = ()) -> list[Path]:
    """The folders that hold lecture folders: where the known lectures are (their parent, or the folder the student chose), the output folder of the settings, anything handed in.
    Nothing else on the computer is looked at."""
    roots: dict[str, Path] = {}

    def add(path: Optional[Path]) -> None:
        if path is not None and str(path) not in ("", ".") and path.is_dir():
            roots.setdefault(_key(path), path)

    for record in records:
        summary = record.summary or {}
        reported = str(summary.get("output_dir") or "")
        if reported:
            add(Path(reported).parent)
        for text in (record.output_root, record.output_dir):
            if text:
                path = Path(text)
                art = find_artifacts(path)
                add(path.parent if art.is_lecture else path)
    if configured_output_dir:
        add(Path(configured_output_dir))
    for path in extra:
        add(Path(path))
    return list(roots.values())


def _candidate_folders(roots: Iterable[Path]) -> list[Path]:
    seen: dict[str, Path] = {}
    for root in roots:
        for child in _children(root):
            if child.name.casefold().startswith("batch_"):
                for grand in _children(child):
                    seen.setdefault(_key(grand), grand)
            seen.setdefault(_key(child), child)
    return list(seen.values())


def _recover_record(folder: Path, job_id: str) -> JobRecord:
    """A record for a lecture folder, from what the folder says. Times come from the files (trustworthy), numbers only where the folder recorded them."""
    try:
        meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        meta = {}
    source_info = meta.get("source") or {}
    name = str(source_info.get("file") or folder.name)
    probe = source_info.get("probe") or {}
    duration = probe.get("duration_s") if isinstance(probe.get("duration_s"), (int, float)) else 0.0
    language = meta.get("language") or {}
    ml = meta.get("multilingual") or {}
    art = find_artifacts(folder)
    stamp = str(meta.get("created_at") or "")[:19]
    if not stamp:
        newest = max((p.stat().st_mtime for p in (art.metadata, art.clean, art.raw, art.timestamped, art.transcript_json) if p is not None), default=0.0)
        import time

        stamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(newest)) if newest else ""
    transcript = load_transcript(art)
    if not duration and transcript is not None and transcript.last_time:
        duration = float(transcript.last_time)
    record = JobRecord(job_id=job_id, source=str(source_info.get("path") or (folder / name)), source_name=name, output_dir=str(folder), output_root=str(folder.parent),
                       state="completed", progress=100.0, duration_seconds=float(duration or 0.0), language=str(language.get("requested") or "auto"),
                       detected_language=str(language.get("detected") or ""), detected_languages=[str(c) for c in (ml.get("detected_languages") or []) if c],
                       translations=sorted(art.translations))
    if stamp:
        record.created = record.updated = stamp
    return record


def reconcile_archive(store: JobStore, configured_output_dir: str = "", extra_roots: Iterable[Path] = ()) -> ReconcileReport:
    report = ReconcileReport()
    records = {r.job_id: r for r in store.list()}
    roots = discover_roots(records.values(), configured_output_dir, extra_roots)
    report.roots = [str(r) for r in roots]
    by_folder: dict[str, JobRecord] = {}
    for record in records.values():
        folder = lecture_folder(record)
        if folder is not None:
            by_folder.setdefault(_key(folder), record)
    claimed_ids = set()
    for folder in _candidate_folders(roots):
        art = find_artifacts(folder)
        if not art.is_lecture:
            continue
        report.found += 1
        job_id = read_job_id(folder)
        record = by_folder.get(_key(folder)) or (records.get(job_id) if job_id else None)
        if record is not None:
            if record.job_id in claimed_ids:
                report.duplicates_prevented += 1
                report.details.append(f"{folder.name}: not added again (the lecture {record.source_name or record.job_id} already has a folder)")
                continue
            claimed_ids.add(record.job_id)
            if record.hidden:
                report.hidden_by_user += 1
                report.details.append(f"{record.source_name or folder.name}: kept hidden")
            if _key(record.output_dir or "") != _key(folder):
                store.repair(record.job_id, output_dir=str(folder), output_root=record.output_root or record.output_dir)
                report.repaired += 1
                report.details.append(f"{record.source_name or folder.name}: now points to its own folder")
            else:
                report.already_indexed += 1
            continue
        new_id = job_id or "adopted-" + hashlib.sha1(_key(folder).encode("utf-8")).hexdigest()[:12]
        if new_id in records or new_id in claimed_ids:
            report.duplicates_prevented += 1                  # the same lecture already has an entry (a copied folder)
            report.details.append(f"{folder.name}: not added again (a copy of a lecture that is already in the archive)")
            continue
        recovered = _recover_record(folder, new_id)
        store.save(recovered, touch=False)
        records[new_id] = recovered
        claimed_ids.add(new_id)
        by_folder[_key(folder)] = recovered
        report.recovered += 1
        report.details.append(f"{recovered.source_name or folder.name}: restored to the archive")
    for record in records.values():
        if record.state == "completed" and not record.hidden and lecture_folder(record) is None:
            report.unresolved.append(f"{record.display_name or record.source_name or record.job_id}: its folder was not found ({record.output_dir or 'no folder recorded'})")
    return report
