"""The medical expert second pass: RAW segments in, a separate MEDICAL CORRECTED transcript + reports out.

    RAW ASR segments (never modified)
      -> language stabilisation   isolated drifted segments re-decoded in the primary language (language.repair_drift)
      -> repetition repair        stutters collapsed, decoder-artefact duplicates removed, chunk-edge overlaps trimmed (repair)
      -> timestamp repair         monotonic, no negative / reversed durations, overlaps resolved (repair)
      -> context-aware correction MedicalContextCorrector: sound + cues + neighbours + topic + slides + lecture terms (corrector)
      -> medical grammar cleanup  spacing, sentence capitals, unit symbols; nothing is paraphrased
      -> MEDICAL CORRECTED TRANSCRIPT + correction_report.json + medical_quality_report.json

Everything is offline and deterministic. The pass never raises into the caller's pipeline: ``run_medical_pass`` returns ``None`` after logging
if something unexpected happens (the raw / clean transcripts are already written by then).
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from . import language as lang_mod
from . import repair as repair_mod
from .corrector import Correction, MedicalContextCorrector, Options, SegmentView
from .lexicon import Lexicon, norm, shared_lexicon

log = logging.getLogger(__name__)

PASS_VERSION = "1.0.0"
PARAGRAPH_GAP = 2.0

_UNIT_CANON = {"mg": "mg", "ml": "mL", "µg": "µg", "ug": "µg", "mcg": "µg", "ng": "ng", "pg": "pg", "kg": "kg", "dl": "dL", "mmhg": "mmHg",
               "mmol/l": "mmol/L", "meq/l": "mEq/L", "mmol": "mmol", "hz": "Hz", "khz": "kHz", "mv": "mV", "µm": "µm", "nm": "nm", "mm": "mm",
               "cm": "cm", "iu": "IU"}
_UNIT_RE = re.compile(r"(?<![\w.])(\d+(?:[.,]\d+)?)\s*(mg|ml|µg|ug|mcg|ng|pg|kg|dl|mmhg|mmol/l|meq/l|mmol|hz|khz|mv|µm|nm|mm|cm|iu)(?![\w/])", re.I)
_CAP_SKIP = {"mg", "ml", "ng", "pg", "kg", "dl", "mmhg", "hz", "khz", "mv", "nm", "mm", "cm", "iu", "mmol", "meq"}
_SENTENCE_END = re.compile(r"[.!?]['\")\]]*\s*$")


@dataclass
class MedicalOptions:
    language: Optional[str] = None                  # primary language of the recording (forced or soft-locked); None = unknown
    language_confidence: Optional[float] = None
    subjects: Sequence[str] = ()                    # subject areas the pipeline detected / the user chose
    custom_terms: Sequence[str] = ()                # Settings > Medical Vocabulary
    lecture_terms: Sequence[str] = ()               # terms of the attached slides / PDF
    duration: Optional[float] = None
    redecode: Optional[lang_mod.Redecode] = None    # re-decode one time range in a language (needs the audio)
    apply_medium: bool = True
    asr_seconds: Optional[float] = None             # how long the recognition took (for the overhead figure)
    vocabulary_dir: Optional[Path] = None
    language_verdict: Optional[dict] = None         # what the soft lock decided (copied into the report)
    repair_language: bool = True
    remove_duplicates: bool = True
    repair_time: bool = True
    shadow_v4: Optional[bool] = None                # legacy flag: v4 in SHADOW mode (None = follow MLT_V4_SHADOW)
    v4_mode: Optional[str] = None                   # v4 Medical Intelligence: off | shadow | on (None = MLT_V4_MODE, then the legacy flag, then off - docs/v4/HELDOUT_V3.md)
    knowledge_db: Optional[Path] = None             # explicit knowledge database for the v4 layer


@dataclass
class CorrectedSegment:
    index: int                    # position in the medical transcript
    raw_index: int                # position of the raw segment it comes from
    start: float
    end: float
    text: str
    raw_text: str
    confidence: Optional[float] = None
    corrections: int = 0
    uncertain: int = 0

    def to_dict(self) -> dict:
        return {"index": self.index, "raw_index": self.raw_index, "start": round(self.start, 3), "end": round(self.end, 3), "text": self.text,
                "raw_text": self.raw_text, "corrections": self.corrections, "uncertain_terms": self.uncertain}


@dataclass
class MedicalResult:
    segments: list[CorrectedSegment]
    text: str
    timestamped: str
    correction_report: dict
    quality_report: dict
    seconds: float
    corrections: list[Correction] = field(default_factory=list)
    shadow: Optional[dict] = None                   # v4 shadow report (only when the feature flag is on); never used to change the transcript

    @property
    def files(self) -> dict[str, str]:
        return {"medical_corrected_transcript.txt": "text", "medical_corrected_transcript_timestamped.txt": "timestamped",
                "medical_corrected_segments.json": "segments", "correction_report.json": "correction_report",
                "medical_quality_report.json": "quality_report"}

    def write(self, directory: Path) -> dict[str, Path]:
        """Write the medical outputs next to the raw / clean transcripts (those are never touched)."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}
        for name, attr in self.files.items():
            path = directory / name
            if attr in ("text", "timestamped"):
                path.write_text(getattr(self, attr), encoding="utf-8", newline="\n")
            elif attr == "segments":
                path.write_text(json.dumps([s.to_dict() for s in self.segments], ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
            else:
                path.write_text(json.dumps(getattr(self, attr), ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
            written[name] = path
        if self.shadow:
            from ..medical_knowledge.runtime.shadow import REPORT_NAME

            path = directory / REPORT_NAME
            path.write_text(json.dumps(self.shadow, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
            written[REPORT_NAME] = path
        return written


# ====================================================================== grammar cleanup (medical, minimal)
def tidy(text: str) -> str:
    """Whitespace, space before punctuation, ``so -called`` -> ``so-called``, unit symbols after numbers. Never rewrites words."""
    t = re.sub(r"\s+", " ", text).strip()
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    t = re.sub(r"(?<=[A-Za-zÀ-ɏ]) -(?=[A-Za-zÀ-ɏ])", "-", t)
    t = _UNIT_RE.sub(lambda m: f"{m.group(1)} {_UNIT_CANON[m.group(2).lower()]}", t)
    return t


def capitalise_sentence_start(text: str, previous_text: Optional[str]) -> str:
    if previous_text is not None and not _SENTENCE_END.search(previous_text):
        return text
    m = re.match(r"^(\W*)([a-zà-ÿ][\w'’-]*)", text)
    if not m:
        return text
    first = m.group(2)
    if first != first.lower() or first.lower() in _CAP_SKIP or len(first) < 2:
        return text
    return m.group(1) + first[0].upper() + first[1:] + text[m.end():]


# ====================================================================== the pass
def run_medical_pass(segments: Sequence[Any], options: Optional[MedicalOptions] = None, lexicon: Optional[Lexicon] = None) -> Optional[MedicalResult]:
    """Never raises: returns ``None`` (and logs) when the pass cannot run, so the caller keeps its raw / clean transcripts."""
    try:
        return _run(segments, options or MedicalOptions(), lexicon)
    except Exception:                                                              # pragma: no cover - defensive
        log.exception("medical expert pass failed; the raw and clean transcripts are unaffected")
        return None


def _run(segments: Sequence[Any], opt: MedicalOptions, lexicon: Optional[Lexicon]) -> MedicalResult:
    started = time.perf_counter()
    if lexicon is None:
        lexicon = shared_lexicon(opt.vocabulary_dir)
    raw_texts = [s.text.strip() for s in segments]
    full_index = lexicon.subset(None)

    def is_medical_word(word: str) -> bool:
        return bool(full_index.exact(word)) or word in full_index.known

    # --- 1. language stability
    primary = opt.language
    if primary is None:
        primary = "en" if lang_mod.english_share(" ".join(raw_texts[: min(len(raw_texts), 200)]), is_medical_word) >= 0.5 else None
    drift_repairs: list[lang_mod.DriftRepair] = []
    drifted: list[int] = []
    texts = list(raw_texts)
    if opt.repair_language and primary:
        texts, drift_repairs, drifted = lang_mod.repair_drift(segments, primary, opt.redecode, is_medical_word)

    # --- 2. repetition, 3. timestamps
    items = [repair_mod.Item(i, float(s.start), float(s.end), texts[i], getattr(s, "confidence", None), getattr(s, "language", None), i)
             for i, s in enumerate(segments) if texts[i].strip()]
    stutters = 0
    for it in items:
        it.text, n = repair_mod.collapse_repeats(it.text)
        stutters += n
    removed: list[repair_mod.DuplicateRemoval] = []
    if opt.remove_duplicates:
        items, removed = repair_mod.remove_duplicates(items)
    items = [it for it in items if it.text.strip()]
    ts_repairs: list[repair_mod.TimestampRepair] = []
    overlaps_before = repair_mod.count_overlaps([(float(s.start), float(s.end)) for s in segments])
    reversed_before = repair_mod.count_reversed([(float(s.start), float(s.end)) for s in segments])
    if opt.repair_time:
        items, ts_repairs = repair_mod.repair_timestamps(items, opt.duration)

    # --- 4. context-aware medical correction (on the repaired texts; word probabilities only where the text is still the raw one)
    views = []
    for k, it in enumerate(items):
        src = segments[it.source]
        probs = [(w.word.strip(), w.probability) for w in getattr(src, "words", [])]                  # matched to the words by the corrector
        views.append(SegmentView(k, it.start, it.end, it.text, it.language, probs))
    corrector = MedicalContextCorrector(lexicon, Options(language=primary or "en", subjects=tuple(opt.subjects), custom_terms=tuple(opt.custom_terms),
                                                         lecture_terms=tuple(opt.lecture_terms), apply_medium=opt.apply_medium))
    result = corrector.correct(views)
    foreign = {k for k, it in enumerate(items) if lang_mod.other_language(segments[it.source], primary)}
    if foreign:                                                                  # English-only correction rules are never applied to a passage the plan labelled as another language
        for k in foreign:
            result.texts[k] = items[k].text
        result.corrections = [c for c in result.corrections if c.segment not in foreign]
        result.uncertain = [c for c in result.uncertain if c.segment not in foreign]
    shadow, v4, v4_texts = _intelligence(views, result, opt)
    v4_applied = list(v4.get("applied", []))
    if v4_texts is not None:
        result.texts = v4_texts                                                    # the combined text: the Medical Expert's, plus v4 HIGH where it changed nothing

    # --- 5. grammar cleanup, output segments
    out: list[CorrectedSegment] = []
    previous: Optional[str] = None
    for k, it in enumerate(items):
        text = tidy(result.texts[k])
        text = capitalise_sentence_start(text, previous)
        previous = text
        out.append(CorrectedSegment(k, it.source, it.start, it.end, text, raw_texts[it.source], it.confidence,
                                    sum(1 for c in result.corrections if c.segment == k and c.tier in ("high", "medium") and c.category == "medical")
                                    + sum(1 for a in v4_applied if a.get("segment") == k),
                                    sum(1 for c in result.uncertain if c.segment == k)))
    text = _paragraphs(out)
    timestamped = "\n".join(f"[{_clock(s.start)} --> {_clock(s.end)}] {s.text}" for s in out) + "\n"

    seconds = time.perf_counter() - started
    applied = [c for c in result.corrections if c.category == "medical" and c.tier in ("high", "medium")]
    case_fixes = [c for c in result.corrections if c.category == "case"]
    high = [c for c in applied if c.tier == "high"]
    medium = [c for c in applied if c.tier == "medium"]
    hits_total = sum(result.term_hits.values())
    topic = result.topic
    after_pairs = [(s.start, s.end) for s in out]
    low_conf = sorted({s.raw_index for s in out if s.uncertain or (s.confidence is not None and s.confidence < 0.5)})

    correction_report = {
        "version": PASS_VERSION, "raw_transcript_modified": False,
        "note": "original = what the recogniser wrote (local_raw_transcript.txt, never changed); tier high/medium are applied in "
                "medical_corrected_transcript.txt, low is only reported",
        "vocabulary": lexicon.version(), "domains_used": result.domains_used, "topic": topic,
        "corrections": [c.to_dict() for c in sorted(result.corrections, key=lambda c: (c.timestamp, c.span)) if c.category == "medical" and c.tier != "low"],
        "case_and_spelling": [c.to_dict() for c in sorted(case_fixes, key=lambda c: (c.timestamp, c.span))],
        "uncertain": [c.to_dict() for c in result.uncertain],
        "language_drift": [r.to_dict() for r in drift_repairs],
        "duplicates_removed": [r.to_dict() for r in removed],
        "repeated_phrases_collapsed": stutters,
        "timestamp_repairs": [r.to_dict() for r in ts_repairs],
        "medical_intelligence_v4": v4,
    }
    quality_report = {
        "version": PASS_VERSION,
        "primary_language": primary,
        "language_confidence": None if opt.language_confidence is None else round(float(opt.language_confidence), 3),
        "language_decision": opt.language_verdict,
        "detected_subject": _subject(topic, opt.subjects),
        "total_segments": len(segments),
        "medical_segments": len(out),
        "medical_terms_detected": hits_total,
        "distinct_medical_terms": len(result.term_hits),
        "medical_corrections": len(applied),
        "high_confidence_corrections": len(high),
        "medium_confidence_corrections": len(medium),
        "uncertain_terms": len(result.uncertain),
        "case_and_spelling_fixes": len(case_fixes),
        "language_drift_segments": len(drifted),
        "redecoded_segments": sum(1 for r in drift_repairs if r.redecoded),
        "duplicate_segments_removed": len(removed),
        "repeated_phrases_collapsed": stutters,
        "timestamp_repairs": len(ts_repairs),
        "timestamp_overlaps_before": overlaps_before,
        "timestamp_overlaps_after": repair_mod.count_overlaps(after_pairs),
        "reversed_durations_before": reversed_before,
        "reversed_durations_after": repair_mod.count_reversed(after_pairs),
        "remaining_low_confidence_segments": len(low_conf),
        "seconds": round(seconds, 2),
        "overhead_percent": None if not opt.asr_seconds else round(100.0 * seconds / opt.asr_seconds, 1),
        "raw_transcript_modified": False,
        "v4_mode": v4.get("mode"), "v4_status": v4.get("status"), "v4_corrections_applied": len(v4_applied), "v4_suggestions": len(v4.get("suggested", [])),
    }
    log.info("MEDICAL PASS: %d terms heard, %d corrected (%d high, %d medium), %d uncertain, %d drifted, %d duplicates, %d timestamp repairs in %.1fs",
             hits_total, len(applied), len(high), len(medium), len(result.uncertain), len(drifted), len(removed), len(ts_repairs), seconds)
    return MedicalResult(out, text, timestamped, correction_report, quality_report, seconds, result.corrections, shadow)


def _intelligence(views: list, result: Any, opt: MedicalOptions) -> tuple[Optional[dict], dict, Optional[list]]:
    """v4 Medical Intelligence as a layer: ``(shadow report, the v4 block of the correction report, combined texts or None)``. Off by default; never raises, never stops the pass."""
    try:
        from ..medical_knowledge.runtime.combine import resolve_mode, run_intelligence

        mode = resolve_mode(opt.v4_mode, opt.shadow_v4)
        return run_intelligence(views, result, mode=mode, subjects=tuple(opt.subjects), lecture_terms=tuple(opt.lecture_terms), custom_terms=tuple(opt.custom_terms),
                                knowledge_path=opt.knowledge_db)
    except Exception as exc:                                                       # pragma: no cover - defensive: the layer must never affect production
        log.exception("v4 Medical Intelligence could not start; the v3 Medical Expert is used")
        return None, {"mode": "off", "status": f"unavailable: {type(exc).__name__}: {exc}", "applied": [], "suggested": [], "database": None}, None


def _shadow(views: list, result: Any, opt: MedicalOptions) -> Optional[dict]:
    """v4 Medical Intelligence, SHADOW ONLY: compared with ``result`` and written to its own file; the transcript is never touched. Off unless flagged."""
    try:
        from ..medical_knowledge.runtime import shadow

        if not shadow.enabled(opt.shadow_v4):
            return None
        return shadow.run_shadow(views, result, subjects=tuple(opt.subjects), lecture_terms=tuple(opt.lecture_terms), custom_terms=tuple(opt.custom_terms),
                                 knowledge_path=opt.knowledge_db)
    except Exception:                                                              # pragma: no cover - defensive: shadow mode must never affect production
        log.exception("v4 shadow mode failed; the medical transcript is unaffected")
        return None


def _subject(topic: dict[str, float], subjects: Sequence[str]) -> Optional[str]:
    if topic:
        return max(topic, key=topic.get)
    return list(subjects)[0] if subjects else None


def _clock(seconds: float) -> str:
    h, rest = divmod(int(max(seconds, 0)), 3600)
    m, s = divmod(rest, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _paragraphs(segments: Sequence[CorrectedSegment]) -> str:
    paragraphs: list[list[str]] = []
    previous_end: Optional[float] = None
    for seg in segments:
        if not seg.text.strip():
            continue
        if previous_end is not None and seg.start - previous_end > PARAGRAPH_GAP and paragraphs:
            paragraphs.append([])
        if not paragraphs:
            paragraphs.append([])
        paragraphs[-1].append(seg.text)
        previous_end = seg.end
    return "\n\n".join(" ".join(p) for p in paragraphs if p) + "\n"
