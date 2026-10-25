"""Reading the sources of a lecture: the transcript JSON written by the transcriber (or a plain text file) and the lecture PDF."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from .index import DocumentIndex, build_index
from .models import MODE_BOTH, MODE_PDF, MODE_TRANSCRIPT

log = logging.getLogger(__name__)

TRANSCRIPT_JSON = "local_transcript.json"
CLEAN_TEXT = "local_clean_transcript.txt"
RAW_TEXT = "local_raw_transcript.txt"
MEDICAL_SEGMENTS = "medical_corrected_segments.json"          # the medical expert pass: timed, terms checked in context
MEDICAL_TEXT = "medical_corrected_transcript.txt"
CORRECTION_REPORT = "correction_report.json"
QUALITY_REPORT = "medical_quality_report.json"


class SourceError(Exception):
    """A source could not be read; the message is meant for the student."""


@dataclass
class LectureSources:
    segments: list[dict[str, Any]] = field(default_factory=list)
    pdf_pages: list[str] = field(default_factory=list)
    language: str = ""
    title: str = ""
    warnings: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)       # heard words the medical pass doubted: nothing may be built on them
    note: str = ""                                            # which transcript the material comes from
    meta: dict[str, Any] = field(default_factory=dict)        # subject, duration (from the medical quality report / metadata)

    @property
    def mode(self) -> str:
        if self.segments and self.pdf_pages:
            return MODE_BOTH
        return MODE_TRANSCRIPT if self.segments else MODE_PDF

    @property
    def empty(self) -> bool:
        return not (self.segments or self.pdf_pages)


def segments_from_json(path: Path) -> tuple[list[dict[str, Any]], str]:
    """Segments (start, end, text) and the language code from the transcript JSON of a finished transcription."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SourceError(f"The transcript file could not be read ({exc}).") from exc
    raw = data.get("segments") if isinstance(data, dict) else data
    if not isinstance(raw, list):
        raise SourceError("The transcript file has no segments.")
    segments = []
    for s in raw:
        if not (isinstance(s, dict) and str(s.get("text", "")).strip()):
            continue
        item = {"start": float(s.get("start", 0.0)), "end": float(s.get("end", 0.0)), "text": str(s.get("text", ""))}
        item.update(_confidence_of(s))
        segments.append(item)
    language = str(data.get("language") or "") if isinstance(data, dict) else ""
    return segments, language


LOW_WORD_PROBABILITY = 0.35           # a content word the recogniser itself doubted this much is not a safe basis for study material


def _confidence_of(segment: dict[str, Any]) -> dict[str, Any]:
    """``confidence`` (mean of the recogniser's word probabilities) and ``low_words`` (content words it doubted) of one transcript segment, when it records them."""
    out: dict[str, Any] = {}
    words = segment.get("words") or []
    probs = [float(w["probability"]) for w in words if isinstance(w, dict) and w.get("probability") is not None]
    if probs:
        out["confidence"] = round(sum(probs) / len(probs), 3)
        low = []
        for w in words:
            token = re.sub(r"[^A-Za-z]", "", str(w.get("word", ""))).lower()
            if len(token) >= 4 and w.get("probability") is not None and float(w["probability"]) < LOW_WORD_PROBABILITY:
                low.append(token)
        out["low_words"] = sorted(set(low))
    elif segment.get("confidence") is not None:
        out["confidence"] = float(segment["confidence"])
    return out


def enrich_with_recogniser_confidence(transcript: Path, segments: list[dict[str, Any]]) -> None:
    """The medical-corrected segments carry no word probabilities: take them from the transcript JSON of the same lecture folder (``raw_index`` points at its segment)."""
    folder = Path(transcript).parent
    raw_file = folder / TRANSCRIPT_JSON
    if Path(transcript).name != MEDICAL_SEGMENTS or not raw_file.is_file():
        return
    try:
        raw = json.loads(raw_file.read_text(encoding="utf-8")).get("segments") or []
        mapping = json.loads(Path(transcript).read_text(encoding="utf-8"))
    except (OSError, ValueError, AttributeError):
        return
    by_text = [m for m in mapping if isinstance(m, dict) and str(m.get("text", "")).strip()]
    if len(by_text) != len(segments):
        return
    for seg, meta in zip(segments, by_text):
        i = meta.get("raw_index")
        if isinstance(i, int) and 0 <= i < len(raw) and isinstance(raw[i], dict):
            seg.update(_confidence_of(raw[i]))


def segments_from_text(text: str) -> list[dict[str, Any]]:
    """A plain text transcript. Lines that start with a time ("12:30 ...", "[01:02:03] ...", "[00:01:05 - 00:01:09] ...") keep their time; without times the
    text is cut into paragraphs and each gets a position in lecture order only (no timestamps are invented). The parsing is the one the transcript reader uses."""
    from ..archive.transcript import parse_timestamped

    return [{"start": seg.start, "end": seg.end if seg.end is not None else seg.start, "text": seg.text} for seg in parse_timestamped(text) if seg.text.strip()]


def pdf_pages(path: Path) -> list[str]:
    """Text of every page of a PDF. Raises ``SourceError`` for a missing file, a damaged or password-protected PDF, or a PDF
    that holds only scanned images (no text layer)."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:                            # pragma: no cover - pypdf is a pinned requirement
        raise SourceError("Reading PDFs needs the 'pypdf' package.") from exc
    try:
        reader = PdfReader(str(path))
        if getattr(reader, "is_encrypted", False):
            try:
                reader.decrypt("")
            except Exception:                              # noqa: BLE001 - pypdf raises several kinds
                raise SourceError("This PDF is password-protected.")
        pages = [(page.extract_text() or "") for page in reader.pages]
    except SourceError:
        raise
    except Exception as exc:                               # noqa: BLE001 - damaged files raise very different errors
        raise SourceError(f"The PDF could not be read ({exc.__class__.__name__}).") from exc
    if not any(p.strip() for p in pages):
        raise SourceError("The PDF has no text (it may be scanned pictures of slides). Text-based PDFs work best.")
    return pages


def load_sources(*, transcript: Optional[Path] = None, pdf: Optional[Path] = None, title: str = "") -> LectureSources:
    """Read whatever is available. A broken PDF does not block the transcript (a warning is kept); with nothing readable it raises."""
    sources = LectureSources(title=title)
    if transcript:
        transcript = Path(transcript)
        try:
            if transcript.suffix.lower() == ".json":
                sources.segments, sources.language = segments_from_json(transcript)
            else:
                sources.segments = segments_from_text(transcript.read_text(encoding="utf-8", errors="replace"))
            sources.unresolved, sources.note, sources.meta = medical_context(transcript)
            if transcript.suffix.lower() == ".json":
                enrich_with_recogniser_confidence(transcript, sources.segments)
        except (SourceError, OSError) as exc:
            if not pdf:
                raise SourceError(str(exc)) from exc
            sources.warnings.append(f"Transcript not used: {exc}")
    if pdf:
        try:
            sources.pdf_pages = pdf_pages(Path(pdf))
        except SourceError as exc:
            if not sources.segments:
                raise
            sources.warnings.append(f"Slides not used: {exc}")
    if sources.empty:
        raise SourceError("There is nothing to study yet: add a transcript or a lecture PDF.")
    return sources


def find_transcript(folder: Path) -> Optional[Path]:
    """The transcript study material is made from: the medical-corrected transcript first (it has the times), then the transcript JSON (times),
    then the plain texts. The raw ASR text is only the last resort, and the study side says so."""
    folder = Path(folder)
    for name in (MEDICAL_SEGMENTS, TRANSCRIPT_JSON, MEDICAL_TEXT, CLEAN_TEXT, RAW_TEXT):
        if (folder / name).is_file():
            return folder / name
    from ..archive.artifacts import find_artifacts          # a lecture of an earlier version: <name>_clean.txt, <name>_raw.txt, <name>_timestamped.txt

    art = find_artifacts(folder)
    for path in (art.clean, art.raw, art.timestamped, art.corrected_timestamped):
        if path is not None and path.is_file() and path.stat().st_size >= 20:
            return path
    return None


def _read_json_file(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def medical_context(transcript: Path) -> tuple[list[str], str, dict[str, Any]]:
    """``(unresolved words, source note, meta)`` for a transcript in a finished lecture folder: what the medical pass doubted, whether the
    text is the medical-corrected one, and the subject / duration it found. Missing files give empty answers."""
    folder = Path(transcript).parent
    unresolved: list[str] = []
    meta: dict[str, Any] = {}
    corrected = Path(transcript).name in (MEDICAL_SEGMENTS, MEDICAL_TEXT)
    report = _read_json_file(folder / CORRECTION_REPORT) if (folder / CORRECTION_REPORT).is_file() else None
    if isinstance(report, dict):
        unresolved += [str(c.get("original", "")) for c in report.get("uncertain", []) if c.get("original")]
        if not corrected:                                                     # the raw text still holds what the medical pass changed
            unresolved += [str(c.get("original", "")) for c in report.get("corrections", []) if c.get("original")]
    quality = _read_json_file(folder / QUALITY_REPORT) if (folder / QUALITY_REPORT).is_file() else None
    if isinstance(quality, dict):
        meta["subject"] = quality.get("detected_subject")
        meta["primary_language"] = quality.get("primary_language")
    info = _read_json_file(folder / "metadata.json") if (folder / "metadata.json").is_file() else None
    if isinstance(info, dict):
        meta.setdefault("subject", ((info.get("subject") or {}).get("subjects") or [None])[0])
        meta["duration"] = (info.get("source") or {}).get("probe", {}).get("duration_s")
    note = ("Made from the medical-corrected transcript." if corrected else
            "Made from the transcript text without the medical-term check (older lecture or the check was off).")
    return [u for u in dict.fromkeys(unresolved) if u.strip()], note, meta


def build_lecture_index(sources: LectureSources, glossary: Sequence[str] = ()) -> DocumentIndex:
    index = build_index(sources.segments, sources.pdf_pages, glossary)
    index.unresolved = list(sources.unresolved)
    index.meta = {**sources.meta, "source_note": sources.note, "title": sources.title, "language": sources.language}
    return index
