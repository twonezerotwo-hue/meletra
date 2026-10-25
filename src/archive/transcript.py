"""The text of a lecture, read from whichever transcript file exists - with the times the file really has and none that it does not."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .artifacts import Artifacts, find_artifacts

# [00:00:01 - 00:00:08] text   |   [00:01:05] text   |   [01:05] text   |   00:01:05 text
_RANGE = re.compile(r"^\s*\[\s*(\d{1,2}):(\d{2}):(\d{2})(?:[.,]\d+)?\s*[-–—]+\s*(\d{1,2}):(\d{2}):(\d{2})(?:[.,]\d+)?\s*\]\s*(.*)$")
_POINT = re.compile(r"^\s*\[?\s*(?:(\d{1,2}):)?(\d{1,2}):(\d{2})(?:[.,]\d+)?\s*\]?\s*[-–—:]?\s+(.*)$")
_WORD = re.compile(r"\S+")


@dataclass
class Segment:
    start: Optional[float]
    end: Optional[float]
    text: str
    language: str = ""


@dataclass
class TranscriptText:
    text: str
    segments: list[Segment] = field(default_factory=list)
    kind: str = ""                     # corrected_segments | transcript_json | source_json | corrected_text | clean | raw | timestamped
    path: Optional[Path] = None
    has_timestamps: bool = False

    @property
    def words(self) -> int:
        return word_count(self.text)

    @property
    def last_time(self) -> Optional[float]:
        times = [s.end if s.end is not None else s.start for s in self.segments if (s.end is not None or s.start is not None)]
        return max(times) if times else None


def word_count(text: str) -> int:
    return len(_WORD.findall(text or ""))


def _seconds(h: str, m: str, s: str) -> float:
    return int(h or 0) * 3600 + int(m) * 60 + int(s)


def parse_timestamped(text: str) -> list[Segment]:
    """The pipeline's timestamped text (``[00:00:01 - 00:00:08] ...``) and plain ``[MM:SS] ...`` lines. A line without a time continues the previous segment; text without any times
    becomes paragraphs with NO times (none are invented)."""
    lines = [ln.rstrip() for ln in (text or "").splitlines()]
    out: list[Segment] = []
    timed = 0
    for line in lines:
        if not line.strip():
            continue
        match = _RANGE.match(line)
        if match:
            h1, m1, s1, h2, m2, s2, body = match.groups()
            out.append(Segment(_seconds(h1, m1, s1), _seconds(h2, m2, s2), body.strip()))
            timed += 1
            continue
        match = _POINT.match(line)
        if match:
            h, m, s, body = match.groups()
            start = _seconds(h or "0", m, s)
            if out and out[-1].end is None and out[-1].start is not None:
                out[-1].end = start
            out.append(Segment(start, None, body.strip()))
            timed += 1
            continue
        if out and timed:
            out[-1].text += " " + line.strip()
        else:
            out.append(Segment(None, None, line.strip()))
    if timed == 0:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]
        return [Segment(None, None, " ".join(p.split())) for p in paragraphs] or [s for s in out if s.text]
    for index, seg in enumerate(out):
        if seg.end is None:
            seg.end = out[index + 1].start if index + 1 < len(out) and out[index + 1].start is not None else seg.start
    return [s for s in out if s.text]


def _from_json_segments(raw: Any, text_key: str = "text") -> list[Segment]:
    items = raw.get("segments") if isinstance(raw, dict) else raw
    out: list[Segment] = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        body = str(item.get(text_key) or item.get("text") or item.get("source_text") or "").strip()
        if not body:
            continue
        start, end = item.get("start"), item.get("end")
        out.append(Segment(float(start) if isinstance(start, (int, float)) else None, float(end) if isinstance(end, (int, float)) else None, body,
                           str(item.get("language") or item.get("detected_language") or "")))
    return out


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return ""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None


def load_transcript(folder: Path | Artifacts) -> Optional[TranscriptText]:
    """The best transcript a lecture folder holds, or ``None``. Order: the medical-corrected segments (timed, terms checked), the transcript JSON, the saved source transcript, then the
    plain texts. The files are only read, never changed."""
    art = folder if isinstance(folder, Artifacts) else find_artifacts(Path(folder))
    candidates: list[tuple[str, Optional[Path]]] = [("corrected_segments", art.corrected_segments), ("transcript_json", art.transcript_json), ("source_json", art.source_json)]
    for kind, path in candidates:
        if path is None:
            continue
        segments = _from_json_segments(_read_json(path), "source_text" if kind == "source_json" else "text")
        if segments:
            return TranscriptText("\n".join(s.text for s in segments), segments, kind, path, any(s.start is not None for s in segments))
    for kind, path in (("corrected_text", art.corrected_text), ("clean", art.clean), ("raw", art.raw)):
        if path is None:
            continue
        text = _read(path).strip()
        if len(text) >= 20:
            return TranscriptText(text, [Segment(None, None, p.strip()) for p in re.split(r"\n\s*\n|\n", text) if p.strip()], kind, path, False)
    for kind, path in (("timestamped", art.timestamped), ("timestamped", art.corrected_timestamped)):
        if path is None:
            continue
        raw = _read(path)
        segments = parse_timestamped(raw)
        if segments:
            return TranscriptText("\n".join(s.text for s in segments), segments, kind, path, any(s.start is not None for s in segments))
    return None
