"""Which files of a lecture folder are what. Current names first, then the names older versions (and the convenience copies) used."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

CLEAN = "local_clean_transcript.txt"
RAW = "local_raw_transcript.txt"
TIMESTAMPED = "local_transcript_with_timestamps.txt"
TRANSCRIPT_JSON = "local_transcript.json"
SOURCE_JSON = "source_transcript.json"
CORRECTED_TEXT = "medical_corrected_transcript.txt"
CORRECTED_TIMESTAMPED = "medical_corrected_transcript_timestamped.txt"
CORRECTED_SEGMENTS = "medical_corrected_segments.json"
METADATA = "metadata.json"
JOB_ID_FILE = ".job_id"
MIN_TRANSCRIPT_CHARS = 20                       # a file shorter than this holds no lecture
_TRANSLATED = re.compile(r"^translated_([A-Za-z\-]{2,8})_transcript\.txt$")


@dataclass
class Artifacts:
    folder: Path
    clean: Optional[Path] = None
    raw: Optional[Path] = None
    timestamped: Optional[Path] = None
    transcript_json: Optional[Path] = None
    source_json: Optional[Path] = None
    corrected_text: Optional[Path] = None
    corrected_timestamped: Optional[Path] = None
    corrected_segments: Optional[Path] = None
    srt: Optional[Path] = None
    vtt: Optional[Path] = None
    metadata: Optional[Path] = None
    job_id_file: Optional[Path] = None
    study_dir: Optional[Path] = None
    translations: dict[str, Path] = field(default_factory=dict)       # language code -> translated_<code>_transcript.txt
    legacy_names: bool = False                                        # the transcript was found under a <name>_clean.txt / _raw.txt / _timestamped.txt copy only

    @property
    def text_files(self) -> list[Path]:
        """Every plain-text transcript, best first (the medical-corrected text, then clean, raw, the timestamped ones)."""
        files = [self.corrected_text, self.clean, self.raw, self.corrected_timestamped, self.timestamped]
        return [p for p in files if p is not None]

    @property
    def has_transcript(self) -> bool:
        for path in (*self.text_files, self.transcript_json, self.source_json, self.corrected_segments):
            if path is not None and _size(path) >= MIN_TRANSCRIPT_CHARS:
                return True
        return False

    @property
    def has_text_transcript(self) -> bool:
        """A readable text copy of the transcript exists (cheap to know: a size, no parsing). A lecture with only JSON files has to be read to be sure."""
        return any(_size(path) >= MIN_TRANSCRIPT_CHARS for path in self.text_files)

    @property
    def has_study(self) -> bool:
        return self.study_dir is not None and (self.study_dir / "study_pack.json").is_file()

    @property
    def is_lecture(self) -> bool:
        return self.has_transcript or self.has_study


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _first(folder: Path, canonical: str, suffix: str) -> tuple[Optional[Path], bool]:
    """``folder/canonical`` if it exists, else the only ``*<suffix>`` convenience copy (``FIXTURE_0005_clean.txt``)."""
    path = folder / canonical
    if path.is_file():
        return path, False
    copies = sorted(p for p in folder.glob(f"*{suffix}") if p.is_file() and not p.name.startswith(("local_", "medical_", "translated_")))
    return (copies[0], True) if len(copies) >= 1 else (None, False)


def find_artifacts(folder: Path) -> Artifacts:
    """What a folder holds. Never reads a file's content; never fails on a missing or unreadable folder."""
    folder = Path(folder)
    found = Artifacts(folder=folder)
    try:
        if not folder.is_dir():
            return found
        found.clean, a = _first(folder, CLEAN, "_clean.txt")
        found.raw, b = _first(folder, RAW, "_raw.txt")
        found.timestamped, c = _first(folder, TIMESTAMPED, "_timestamped.txt")
        found.legacy_names = any((a, b, c))
        found.transcript_json = folder / TRANSCRIPT_JSON if (folder / TRANSCRIPT_JSON).is_file() else None
        found.source_json = folder / SOURCE_JSON if (folder / SOURCE_JSON).is_file() else None
        found.corrected_text = folder / CORRECTED_TEXT if (folder / CORRECTED_TEXT).is_file() else None
        found.corrected_timestamped = folder / CORRECTED_TIMESTAMPED if (folder / CORRECTED_TIMESTAMPED).is_file() else None
        found.corrected_segments = folder / CORRECTED_SEGMENTS if (folder / CORRECTED_SEGMENTS).is_file() else None
        found.srt = folder / "local_transcript.srt" if (folder / "local_transcript.srt").is_file() else None
        found.vtt = folder / "local_transcript.vtt" if (folder / "local_transcript.vtt").is_file() else None
        found.metadata = folder / METADATA if (folder / METADATA).is_file() else None
        found.job_id_file = folder / JOB_ID_FILE if (folder / JOB_ID_FILE).is_file() else None
        found.study_dir = folder / "study" if (folder / "study").is_dir() else None
        for path in folder.iterdir():
            match = _TRANSLATED.match(path.name)
            if match and path.is_file():
                found.translations[match.group(1).lower()] = path
        tdir = folder / "translations"
        if tdir.is_dir():
            for path in tdir.glob("*.json"):
                found.translations.setdefault(path.stem.lower(), path)
    except OSError:
        pass
    return found


def read_job_id(folder: Path) -> str:
    """The job id the pipeline wrote into the folder (``.job_id``), or ``""``."""
    try:
        return (Path(folder) / JOB_ID_FILE).read_text(encoding="utf-8").strip()
    except OSError:
        return ""
