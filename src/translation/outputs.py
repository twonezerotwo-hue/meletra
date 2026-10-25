"""The files of a translation and the record of it (``translation`` in metadata.json). The spoken-language transcript is never touched."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..exporters import raw_transcript_text, segments_to_srt, write_json, write_text
from ..schema import Segment


def file_stem(target: str) -> str:
    return f"translated_{target}_transcript"


@dataclass
class TranslationResult:
    method: str
    source: Optional[str]
    target: Optional[str]
    reason: str = ""
    segments: list[Segment] = field(default_factory=list)
    seconds: float = 0.0
    error: str = ""
    deferred: bool = False                          # the text translation runs after the transcript is final (it needs no audio): nothing to do inside the audio step
    stored_by_service: bool = False                 # the translation service already wrote the files (translations/<code>.json and the translated_<code>_transcript.* copies)
    counts: dict[str, int] = field(default_factory=dict)       # segments by translation status (translated / same_language / untranslated / failed)
    source_languages: list[str] = field(default_factory=list)  # the spoken languages that were translated

    @property
    def written(self) -> bool:
        return bool(self.target) and (bool(self.segments) or self.stored_by_service)

    def record(self) -> dict[str, Any]:
        """What metadata.json, the job record and the lecture card say about the translation."""
        record = {"method": self.method, "source_language": self.source, "output_language": self.target, "reason": self.reason,
                  "segments": len(self.segments), "seconds": round(self.seconds, 1), "error": self.error or None,
                  "files": [f"{file_stem(self.target)}{ext}" for ext in (".txt", ".srt", ".json")] if self.written else []}
        if self.counts:
            record["segment_status"] = dict(self.counts)
        if self.source_languages:
            record["source_languages"] = list(self.source_languages)
        return record

    def write(self, folder: Path) -> list[Path]:
        if not self.written or self.stored_by_service:
            return []
        folder = Path(folder)
        stem = file_stem(self.target)
        text = write_text(folder / f"{stem}.txt", raw_transcript_text(self.segments))
        srt = write_text(folder / f"{stem}.srt", segments_to_srt(self.segments))
        data = write_json(folder / f"{stem}.json", {"language": self.target, "source_language": self.source, "method": self.method,
                                                    "note": "Translation of the spoken-language transcript (local_raw_transcript.txt, which is unchanged).",
                                                    "segments": [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()} for s in self.segments]})
        return [text, srt, data]


def read_record(folder: Path) -> Optional[dict[str, Any]]:
    """The ``translation`` record of a finished lecture (None for a lecture made before translation existed)."""
    try:
        return json.loads((Path(folder) / "metadata.json").read_text(encoding="utf-8")).get("translation")
    except (OSError, ValueError):
        return None


def translated_text_file(folder: Path, target: Optional[str]) -> Optional[Path]:
    if not target:
        return None
    path = Path(folder) / f"{file_stem(target)}.txt"
    return path if path.is_file() else None
