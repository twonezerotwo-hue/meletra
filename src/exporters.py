"""Writers for TXT / SRT / JSON outputs. All files are UTF-8."""

from __future__ import annotations

import json
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from .schema import Segment

SRT_LINE_WIDTH = 42
_INVALID_NAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


@dataclass(frozen=True)
class OutputPaths:
    """All files written for one media file."""

    directory: Path
    checkpoint_override: Optional[Path] = None      # the desktop app keeps checkpoints in its own data folder

    # --- the local Whisper transcript (the Deepgram one, when requested, gets a deepgram_ prefix)
    @property
    def raw(self) -> Path:
        return self.directory / "local_raw_transcript.txt"

    @property
    def clean(self) -> Path:
        return self.directory / "local_clean_transcript.txt"

    @property
    def srt(self) -> Path:
        return self.directory / "local_transcript.srt"

    @property
    def vtt(self) -> Path:
        return self.directory / "local_transcript.vtt"

    @property
    def json(self) -> Path:
        return self.directory / "local_transcript.json"

    # --- medical analysis
    @property
    def entities(self) -> Path:
        return self.directory / "medical_entities.json"

    @property
    def acronyms(self) -> Path:
        return self.directory / "medical_acronyms.txt"

    @property
    def context(self) -> Path:
        return self.directory / "medical_context.json"

    @property
    def names(self) -> Path:
        return self.directory / "drug_gene_protein_report.txt"

    # --- optional Deepgram / compare outputs
    @property
    def deepgram_raw(self) -> Path:
        return self.directory / "deepgram_raw_transcript.txt"

    @property
    def deepgram_json(self) -> Path:
        return self.directory / "deepgram_transcript.json"

    @property
    def deepgram_response(self) -> Path:
        return self.directory / "deepgram_raw_response.json"

    @property
    def disagreement(self) -> Path:
        return self.directory / "medical_disagreement_report.txt"

    @property
    def comparison(self) -> Path:
        return self.directory / "comparison.json"

    @property
    def consensus(self) -> Path:
        return self.directory / "consensus_medical_terms.txt"

    @property
    def uncertain(self) -> Path:
        return self.directory / "uncertain_segments.json"

    @property
    def metadata(self) -> Path:
        return self.directory / "metadata.json"

    @property
    def log(self) -> Path:
        return self.directory / "processing.log"

    @property
    def timestamped(self) -> Path:
        return self.directory / "local_transcript_with_timestamps.txt"

    @property
    def terms(self) -> Path:
        return self.directory / "medical_terms_detected.txt"

    @property
    def possible_errors(self) -> Path:
        return self.directory / "possible_medical_errors.txt"

    @property
    def quality(self) -> Path:
        return self.directory / "quality_report.json"

    @property
    def partial(self) -> Path:
        return self.directory / "partial_transcript.txt"

    @property
    def checkpoint_dir(self) -> Path:
        return self.checkpoint_override or self.directory / "checkpoint"


def safe_name(stem: str) -> str:
    """Make a media stem safe as a Windows folder name (keeps Turkish characters)."""
    cleaned = _INVALID_NAME_CHARS.sub("_", stem).strip(" .")
    return cleaned or "untitled"


def make_output_paths(outputs_root: Path, media_stem: str, create: bool = True,
                      checkpoint_dir: Optional[Path] = None) -> OutputPaths:
    paths = OutputPaths(Path(outputs_root) / safe_name(media_stem), checkpoint_dir)
    if create:
        paths.directory.mkdir(parents=True, exist_ok=True)
    return paths


def format_timestamp(seconds: float, decimal_marker: str = ",") -> str:
    """``HH:MM:SS,mmm`` (SRT) - rounds to whole milliseconds without drift."""
    total_ms = max(0, int(round(seconds * 1000)))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{decimal_marker}{millis:03d}"


def format_hms(seconds: float) -> str:
    """``HH:MM:SS`` (whole seconds)."""
    total = max(0, int(seconds))
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def timestamped_text(segments: Sequence[Segment]) -> str:
    """``[HH:MM:SS - HH:MM:SS] text`` per segment (the raw ASR text, with its time range). A recording in several languages names the language of each line: ``[00:01:05 - 00:01:12] [KO] text``."""
    languages = {seg.language for seg in segments if seg.language and seg.text.strip()}
    tag = len(languages) > 1
    lines = []
    for seg in segments:
        if not seg.text.strip():
            continue
        label = f"[{str(seg.language).upper()}] " if tag and seg.language else ""
        for start, end, text in cue_parts(seg):
            lines.append(f"[{format_hms(start)} - {format_hms(end if end > start else start + 0.5)}] {label}{text}")
    return "\n".join(lines) + ("\n" if lines else "")


def terms_report_text(terms: Sequence[Any], candidates: Sequence[Any], language_note: str = "") -> str:
    """Human-readable list of dictionary terms heard and medical-looking words outside the dictionary."""
    out = ["MEDICAL TERMS DETECTED (dictionary matches)",
           f"{len(terms)} distinct terms, {sum(t.count for t in terms)} occurrences", ""]
    if language_note:
        out += [language_note, ""]
    out.append(f"{'count':>5}  {'first seen':<10}  term")
    out += [f"{t.count:>5}  {format_hms(t.first_start):<10}  {t.term}" for t in terms] or ["  (none)"]
    out += ["", "MEDICAL-LOOKING WORDS NOT IN THE DICTIONARY (candidates; add real ones to medical_terms/*.txt)",
            f"{len(candidates)} words seen at least twice", "", f"{'count':>5}  {'first seen':<10}  word"]
    out += [f"{c.count:>5}  {format_hms(c.first_start):<10}  {c.term}" for c in candidates] or ["  (none)"]
    return "\n".join(out) + "\n"


def possible_errors_text(errors: Sequence[Any]) -> str:
    """Suspected mis-heard medical terms. The transcript itself is NOT changed."""
    out = ["POSSIBLE MEDICAL ASR ERRORS",
           "These are SUGGESTIONS only. raw_transcript.txt keeps exactly what the model wrote; a human should",
           "listen at the timestamp before accepting any of them.", f"{len(errors)} item(s)", ""]
    for number, err in enumerate(errors, start=1):
        out += [f"[{number}]",
                f"Timestamp: {format_hms(err.start)}",
                f"Original phrase: {err.original}",
                f"Possible medical term: {err.suggestion}",
                f"Similarity: {err.confidence:.2f}",
                f"Subject: {getattr(err, 'subject', '') or '-'}",
                f"PDF evidence: {getattr(err, 'pdf_evidence', '') or '-'}",
                f"Reason: {err.reason}",
                f"Confidence: {(getattr(err, 'confidence_label', '') or f'{err.confidence:.2f}').upper()}"
                f" (evidence score {getattr(err, 'score', 0.0):.2f})",
                f"Context: {getattr(err, 'context', '') or '-'}", ""]
    return "\n".join(out) + "\n"


def raw_transcript_text(segments: Sequence[Segment]) -> str:
    """Segment texts exactly as the ASR returned them, one per line (only the leading
    space Whisper prepends is stripped)."""
    return "\n".join(seg.text.strip() for seg in segments) + "\n"


CUE_SPLIT_GAP_S = 2.5          # a silence this long INSIDE one segment (word timestamps) ends the subtitle cue


CUE_MAX_SECONDS = 7.0           # a subtitle cue stays on screen at most this long ...
CUE_MAX_CHARS = SRT_LINE_WIDTH * 2 - 12   # ... and fits two wrapped lines (word wrapping wastes a few characters)


def split_cue_by_words(words: Sequence[Any], max_seconds: float = CUE_MAX_SECONDS, max_chars: int = CUE_MAX_CHARS
                       ) -> list[list[Any]]:
    """Cut a run of timed words into subtitle-sized groups: at most ``max_seconds`` and ``max_chars`` each, preferring to
    break after a sentence end, then after a comma, then anywhere. Whisper returns segments of up to 30 s (measured: cues of 33 s
    with 8 lines); the raw transcript and the JSON are never touched, only the subtitle files use this."""
    groups: list[list[Any]] = []
    current: list[Any] = []

    def length(ws: Sequence[Any]) -> int:
        return len("".join(w.word for w in ws).strip())

    def close_at(index: int) -> None:
        nonlocal current
        groups.append(current[:index])
        current = current[index:]

    for word in words:
        current.append(word)
        too_long = length(current) > max_chars or (current[-1].end - current[0].start) > max_seconds
        if not too_long or len(current) < 2:
            continue
        # the best break inside what is already collected (without the word that overflowed)
        body = current[:-1]
        half = max(len(body) // 2, 1)
        choice = None
        for ending in ((".", "!", "?", "…"), (";", ":"), (",",)):
            candidates = [i for i in range(len(body)) if i >= half - 1 and body[i].word.rstrip().endswith(ending)]
            if candidates:
                choice = candidates[-1] + 1
                break
        close_at(choice if choice else len(body))
    if current:
        groups.append(current)
    return [g for g in groups if g]


def cue_parts(seg: Segment, gap: float = CUE_SPLIT_GAP_S) -> list[tuple[float, float, str]]:
    """``(start, end, text)`` pieces for the subtitle files. A segment normally is one cue, but if the speaker paused for
    ``gap`` seconds or more between two words (word timestamps) the segment is split THERE, so a cue never stays on screen
    across a long silence (measured: one 342-second cue on the real lecture). Only the subtitle files do this; the raw
    transcript and the JSON keep the segment exactly as the model returned it."""
    words = [w for w in seg.words if w.word.strip()]
    if len(words) < 2:
        return [(seg.start, seg.end, seg.text.strip())]
    groups: list[list[Any]] = [[words[0]]]
    for previous, word in zip(words, words[1:]):
        if word.start - previous.end >= gap:
            groups.append([])
        groups[-1].append(word)
    groups = [piece for group in groups for piece in split_cue_by_words(group)]
    if len(groups) == 1 and groups[0][0] is words[0] and groups[0][-1] is words[-1]:
        return [(seg.start, seg.end, seg.text.strip())]
    parts = []
    for group in groups:
        text = "".join(w.word for w in group).strip()
        if text:
            parts.append((group[0].start, max(group[-1].end, group[0].start + 0.5), text))
    return parts or [(seg.start, seg.end, seg.text.strip())]


def subtitle_cues(segments: Sequence[Segment]) -> list[tuple[float, float, str]]:
    cues: list[tuple[float, float, str]] = []
    for seg in segments:
        if not seg.text.strip():
            continue
        for start, end, text in cue_parts(seg):
            cues.append((start, end if end > start else start + 0.5, text))
    return cues


_NL = chr(10)


def segments_to_srt(segments: Sequence[Segment]) -> str:
    blocks: list[str] = []
    for start, end, text in subtitle_cues(segments):
        wrapped = textwrap.wrap(text, width=SRT_LINE_WIDTH, break_long_words=False) or [text]
        head = str(len(blocks) + 1) + _NL + format_timestamp(start) + ' --> ' + format_timestamp(end)
        blocks.append(head + _NL + _NL.join(wrapped))
    return (_NL + _NL).join(blocks) + (_NL if blocks else '')


def segments_to_vtt(segments: Sequence[Segment]) -> str:
    '''WebVTT (browsers / video players): like SRT but with a header and a dot as the millisecond marker.'''
    blocks: list[str] = []
    for start, end, text in subtitle_cues(segments):
        wrapped = textwrap.wrap(text, width=SRT_LINE_WIDTH, break_long_words=False) or [text]
        head = format_timestamp(start, '.') + ' --> ' + format_timestamp(end, '.')
        blocks.append(head + _NL + _NL.join(wrapped))
    return 'WEBVTT' + _NL + _NL + (_NL + _NL).join(blocks) + (_NL if blocks else '')


def transcript_json(
    segments: Sequence[Segment], *, language: str | None, language_probability: float | None,
    model: str, backend: str, duration: float, source: str,
) -> dict[str, Any]:
    return {
        "source": source,
        "language": language,
        "language_probability": None if language_probability is None else round(language_probability, 4),
        "model": model,
        "backend": backend,
        "duration": round(duration, 2),
        "segments": [seg.to_dict() for seg in segments],
    }


def uncertain_json(segments: Sequence[Segment]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for index, seg in enumerate(segments):
        if not seg.uncertain:
            continue
        entries.append({
            "index": index,
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip(),
            "confidence": None if seg.confidence is None else round(seg.confidence, 3),
            "reasons": [flag.to_dict() for flag in seg.flags],
        })
    return entries


def write_text(path: Path, content: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")
    return path


def write_json(path: Path, data: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return path


def export_all(
    paths: OutputPaths,
    segments: Sequence[Segment],
    *,
    clean_text: str,
    metadata: dict[str, Any],
    language: str | None,
    language_probability: float | None,
    model: str,
    backend: str,
    duration: float,
    source: str,
    terms_text: Optional[str] = None,
    errors_text: Optional[str] = None,
    quality: Optional[dict[str, Any]] = None,
    entities: Optional[Any] = None,
    acronyms_text: Optional[str] = None,
    context: Optional[dict[str, Any]] = None,
    names_text: Optional[str] = None,
) -> dict[str, Path]:
    """Write every output file except ``processing.log`` (written by the logger)."""
    written = {
        "raw": write_text(paths.raw, raw_transcript_text(segments)),
        "clean": write_text(paths.clean, clean_text),
        "timestamped": write_text(paths.timestamped, timestamped_text(segments)),
        "srt": write_text(paths.srt, segments_to_srt(segments)),
        "vtt": write_text(paths.vtt, segments_to_vtt(segments)),
        "json": write_json(paths.json, transcript_json(
            segments, language=language, language_probability=language_probability,
            model=model, backend=backend, duration=duration, source=source)),
        "uncertain": write_json(paths.uncertain, uncertain_json(segments)),
        "metadata": write_json(paths.metadata, metadata),
    }
    if terms_text is not None:
        written["terms"] = write_text(paths.terms, terms_text)
    if errors_text is not None:
        written["possible_errors"] = write_text(paths.possible_errors, errors_text)
    if quality is not None:
        written["quality"] = write_json(paths.quality, quality)
    if entities is not None:
        written["entities"] = write_json(paths.entities, entities)
    if acronyms_text is not None:
        written["acronyms"] = write_text(paths.acronyms, acronyms_text)
    if context is not None:
        written["context"] = write_json(paths.context, context)
    if names_text is not None:
        written["names"] = write_text(paths.names, names_text)
    return written


def write_deepgram_outputs(paths: OutputPaths, result: Any) -> dict[str, Path]:
    """``deepgram_*`` files for a Deepgram run (kept apart from the local transcript, never merged into it)."""
    segments = result.segments
    return {
        "deepgram_raw": write_text(paths.deepgram_raw, raw_transcript_text(segments)),
        "deepgram_json": write_json(paths.deepgram_json, transcript_json(
            segments, language=result.language, language_probability=None, model=result.meta.get("model", "nova-3-medical"),
            backend="deepgram", duration=float(result.duration or 0.0), source="deepgram")),
        "deepgram_response": write_json(paths.deepgram_response, result.raw),
    }
