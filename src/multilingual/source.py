"""The canonical source transcript of a lecture: ``source_transcript.json``.

What the speech recogniser heard is stored here ONCE, segment by segment, each in the language it was spoken in. Translations (``translations/<language>.json``) are made from this
file and never replace it; the revision below changes when (and only when) the heard text, the times or the language labels change, so a translation can tell whether it is stale.

Lectures made before this file existed (4.1.0 and older) are read through the same API (``origin == "legacy"``): their segments come from ``local_transcript.json`` and carry the
single language the old pipeline recorded; nothing about other languages is invented for them.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from . import ENGINE_VERSION, SOURCE_FILE

SCHEMA = 1
ORIGIN_PIPELINE, ORIGIN_LEGACY = "pipeline", "legacy"
RECOGNITION_STATUSES = ("ok", "uncertain", "low_confidence", "repaired", "unintelligible")
LANGUAGE_SOURCES = ("detector", "verified", "smoothing", "script", "boundary", "forced", "model", "translate", "legacy", "unknown")
UNKNOWN = "und"                                   # ISO 639-3 "undetermined": the segment's language is not known (never replaced by a guess)


@dataclass
class SourceSegment:
    segment_id: int
    start: float
    end: float
    source_text: str
    detected_language: str = UNKNOWN              # the language this segment is written in ("und" when not known)
    language_confidence: Optional[float] = None   # the recogniser's own probability when it gave one; never invented
    language_source: str = "unknown"              # detector | script | forced | model | legacy | unknown
    language_candidates: list[str] = field(default_factory=list)      # other languages that were plausible for an ambiguous segment
    raw_text: str = ""                            # the recogniser's text before any correction (kept for comparison)
    recognition_status: str = "ok"
    flags: list[str] = field(default_factory=list)
    translation_status: dict[str, str] = field(default_factory=dict)  # target language -> translated | same_language | untranslated | failed

    @property
    def language_known(self) -> bool:
        return self.detected_language not in ("", UNKNOWN)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["start"], data["end"] = round(float(self.start), 3), round(float(self.end), 3)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SourceSegment":
        names = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in names})


@dataclass
class SourceTranscript:
    lecture_id: str
    segments: list[SourceSegment]
    origin: str = ORIGIN_PIPELINE
    input_mode: str = "auto"                      # auto | manual
    input_language: str = "auto"                  # auto, or the language the student chose
    requested_output_language: str = ""
    created_at: str = ""
    engine: dict[str, Any] = field(default_factory=dict)
    coverage: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    output_languages_available: list[str] = field(default_factory=list)
    translation_engine_versions: dict[str, str] = field(default_factory=dict)
    schema: int = SCHEMA

    # ------------------------------------------------------------------------------------------------------------------ derived values
    @property
    def revision(self) -> str:
        return revision_of(self.segments)

    @property
    def language_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for segment in self.segments:
            if segment.language_known:
                counts[segment.detected_language] = counts.get(segment.detected_language, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: -item[1]))

    @property
    def detected_languages(self) -> list[str]:
        """Only languages that were actually recognised in this lecture, most frequent first."""
        return list(self.language_counts)

    @property
    def multilingual(self) -> bool:
        return len(self.detected_languages) > 1

    @property
    def has_unknown_language(self) -> bool:
        return any(not s.language_known and s.source_text.strip() for s in self.segments)

    def text(self) -> str:
        return "\n".join(s.source_text.strip() for s in self.segments if s.source_text.strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "meletra-source-transcript", "schema": self.schema, "lecture_id": self.lecture_id, "revision": self.revision, "origin": self.origin,
            "created_at": self.created_at, "engine": self.engine, "input_mode": self.input_mode, "input_language": self.input_language,
            "requested_output_language": self.requested_output_language, "detected_languages": self.detected_languages, "language_counts": self.language_counts,
            "output_languages_available": self.output_languages_available, "translation_engine_versions": self.translation_engine_versions,
            "coverage": self.coverage, "notes": self.notes, "segments": [s.to_dict() for s in self.segments],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SourceTranscript":
        return cls(lecture_id=str(data.get("lecture_id") or ""), segments=[SourceSegment.from_dict(s) for s in data.get("segments", [])], origin=data.get("origin", ORIGIN_PIPELINE),
                   input_mode=data.get("input_mode", "auto"), input_language=data.get("input_language", "auto"), requested_output_language=data.get("requested_output_language", ""),
                   created_at=data.get("created_at", ""), engine=data.get("engine", {}), coverage=data.get("coverage", {}), notes=list(data.get("notes", [])),
                   output_languages_available=list(data.get("output_languages_available", [])), translation_engine_versions=dict(data.get("translation_engine_versions", {})),
                   schema=int(data.get("schema", SCHEMA)))


def revision_of(segments: Iterable[SourceSegment]) -> str:
    """A short fingerprint of what was heard: the text, the times and the language of every segment (not statuses or translation bookkeeping)."""
    payload = json.dumps([[round(float(s.start), 2), round(float(s.end), 2), s.detected_language, s.source_text] for s in segments], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def source_path(folder: Path) -> Path:
    return Path(folder) / SOURCE_FILE


def write(folder: Path, source: SourceTranscript) -> Path:
    """Write ``source_transcript.json`` atomically (UTF-8, no ASCII escaping: Hangul and Devanagari stay readable)."""
    path = source_path(folder)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(source.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_file(folder: Path) -> Optional[SourceTranscript]:
    path = source_path(folder)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if data.get("kind") != "meletra-source-transcript":
        return None
    return SourceTranscript.from_dict(data)


def load(folder: Path) -> Optional[SourceTranscript]:
    """The source transcript of a lecture folder: the stored file, or (for a lecture made by an older version) one read from ``local_transcript.json``. ``None`` when the folder
    holds no transcript at all."""
    folder = Path(folder)
    stored = read_file(folder)
    if stored is not None:
        return stored
    return legacy(folder)


def legacy(folder: Path) -> Optional[SourceTranscript]:
    """A source transcript for a lecture made before ``source_transcript.json`` existed. It says what the old files say (one language per lecture) and nothing more."""
    folder = Path(folder)
    transcript = folder / "local_transcript.json"
    if not transcript.is_file():
        return _from_text_files(folder)
    try:
        data = json.loads(transcript.read_text(encoding="utf-8"))
        meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8")) if (folder / "metadata.json").is_file() else {}
    except (OSError, ValueError):
        return _from_text_files(folder)
    language = str(data.get("language") or (meta.get("language") or {}).get("detected") or "") or UNKNOWN
    corrected = _corrected_by_raw_index(folder)
    segments: list[SourceSegment] = []
    for index, raw in enumerate(data.get("segments", [])):
        text = str(raw.get("text") or "").strip()
        if not text:
            continue
        segment_language = str(raw.get("language") or language or UNKNOWN)
        shown = corrected.get(index, text)
        segments.append(SourceSegment(segment_id=len(segments) + 1, start=float(raw.get("start", 0.0)), end=float(raw.get("end", 0.0)), source_text=shown,
                                      raw_text=text if shown != text else "", detected_language=segment_language, language_confidence=None, language_source="legacy",
                                      recognition_status="uncertain" if raw.get("uncertain") else "ok"))
    if not segments:
        return None
    requested = str((meta.get("language") or {}).get("requested") or "auto")
    record = meta.get("translation") or {}
    return SourceTranscript(lecture_id=folder.name, segments=segments, origin=ORIGIN_LEGACY, input_mode="auto" if requested in ("auto", "") else "manual", input_language=requested or "auto",
                            requested_output_language=str(record.get("output_language") or ""), created_at=str(meta.get("created_at") or ""),
                            engine={"asr": str(data.get("model") or ""), "multilingual": "none (made before MELETRA 4.1.1: one language per lecture)"},
                            notes=["Made by an earlier version: the program recorded a single language for the whole lecture. Other spoken languages are not in this transcript."])


def _from_text_files(folder: Path) -> Optional[SourceTranscript]:
    """A lecture whose only transcript is text (the files of an earlier version: ``<name>_clean.txt``, ``<name>_timestamped.txt`` ...): its lines are the segments, with the times the file
    has (none are invented: a plain text has all zeros)."""
    from ..archive.transcript import load_transcript

    found = load_transcript(folder)
    if found is None or not found.segments:
        return None
    try:
        meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8")) if (folder / "metadata.json").is_file() else {}
    except (OSError, ValueError):
        meta = {}
    language = str((meta.get("language") or {}).get("detected") or "") or UNKNOWN
    segments = [SourceSegment(segment_id=i + 1, start=float(seg.start or 0.0), end=float(seg.end if seg.end is not None else (seg.start or 0.0)), source_text=seg.text, raw_text="",
                              detected_language=language, language_confidence=None, language_source="legacy", recognition_status="ok") for i, seg in enumerate(found.segments)]
    return SourceTranscript(lecture_id=folder.name, segments=segments, origin=ORIGIN_LEGACY, input_mode="auto", input_language="auto", requested_output_language="",
                            created_at=str(meta.get("created_at") or ""), engine={"asr": "", "multilingual": "none (made before MELETRA 4.1.1: one language per lecture)"},
                            notes=["Made by an earlier version; only its text files are left, so this transcript is read from them." + ("" if found.has_timestamps else " It has no times.")])


def _corrected_by_raw_index(folder: Path) -> dict[int, str]:
    """Texts of the medical-correction pass, keyed by the index of the raw segment they came from (when that file is there and readable)."""
    path = folder / "medical_corrected_segments.json"
    if not path.is_file():
        return {}
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out: dict[int, str] = {}
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict) and "raw_index" in row and str(row.get("text") or "").strip():
                out[int(row["raw_index"])] = str(row["text"]).strip()
    return out


def from_segments(lecture_id: str, segments: Sequence[Any], *, input_language: str = "auto", requested_output_language: str = "", engine: Optional[dict[str, Any]] = None,
                  coverage: Optional[dict[str, Any]] = None, corrected_texts: Optional[Sequence[str]] = None, notes: Optional[list[str]] = None) -> SourceTranscript:
    """Build the canonical transcript from the pipeline's finished segments (``src.schema.Segment`` objects: ``language``, ``confidence`` ... may be set by the language-aware decoder).

    ``corrected_texts``: the text of each segment after language-aware correction, same order; the recogniser's own text is kept as ``raw_text`` when it differs."""
    out: list[SourceSegment] = []
    for index, seg in enumerate(segments):
        raw = str(getattr(seg, "text", "") or "").strip()
        shown = (str(corrected_texts[index]).strip() if corrected_texts is not None and index < len(corrected_texts) else raw) or raw
        if not shown:
            continue
        language = str(getattr(seg, "language", "") or UNKNOWN)
        flags = [str(getattr(f, "code", f)) for f in getattr(seg, "flags", []) or []]
        meta = getattr(seg, "language_meta", None) or {}
        status = "uncertain" if flags else "ok"
        if meta.get("status") in RECOGNITION_STATUSES:
            status = meta["status"]
        out.append(SourceSegment(segment_id=len(out) + 1, start=float(seg.start), end=float(seg.end), source_text=shown, raw_text=raw if raw != shown else "", detected_language=language,
                                 language_confidence=meta.get("confidence"), language_source=str(meta.get("source") or ("detector" if language != UNKNOWN else "unknown")),
                                 language_candidates=list(meta.get("candidates") or []), recognition_status=status, flags=flags))
    manual = input_language not in ("", "auto", None)
    return SourceTranscript(lecture_id=lecture_id, segments=out, origin=ORIGIN_PIPELINE, input_mode="manual" if manual else "auto", input_language=input_language or "auto",
                            requested_output_language=requested_output_language or "", created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                            engine={"multilingual": ENGINE_VERSION, **(engine or {})}, coverage=coverage or {}, notes=list(notes or []))


def record_translation(folder: Path, target: str, engine_id: str, statuses: dict[int, str]) -> None:
    """Remember in the lecture-level metadata (and per segment) that ``target`` exists. Does not change the revision: the heard text is untouched."""
    source = read_file(folder)
    if source is None:
        return
    if target not in source.output_languages_available:
        source.output_languages_available.append(target)
    source.translation_engine_versions[target] = engine_id
    for segment in source.segments:
        if segment.segment_id in statuses:
            segment.translation_status[target] = statuses[segment.segment_id]
    write(folder, source)


def ensure_stored(folder: Path) -> Optional[SourceTranscript]:
    """The source transcript of a lecture, written to ``source_transcript.json`` when it was only readable from the old files (the translation bookkeeping needs a place to live).
    The file is derived from files that stay exactly as they were; nothing is added that the old files did not say."""
    source = load(folder)
    if source is not None and read_file(folder) is None:
        write(folder, source)
    return source
