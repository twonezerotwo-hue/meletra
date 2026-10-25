"""Decoding by language runs: the recording is decoded stretch by stretch, each stretch in the language the plan gives it.

``transcribe_runs`` is called by ``Transcriber.transcribe`` when a request carries a ``language_plan``. Each run becomes an ordinary one-language decode request (so the compute-type
fallback, checkpoints, pause and progress all work exactly as before); the segments are labelled with the run's language and joined in time order. Runs that must be decoded without
the voice-activity filter (singing, speech over music) get a guard that rejects what Whisper is known to write on instrumental passages instead of keeping it as transcript.
"""

from __future__ import annotations

import dataclasses
import logging
import re
from typing import Any, Callable, Optional, Sequence

from ..schema import Segment, TranscriptionResult
from .artifacts import is_artifact
from .scripts import LATIN_LANGUAGES, compatible, dominant_script, looks_english, romanised_indic, script_language

log = logging.getLogger(__name__)

MIN_RUN = 0.4                          # a stretch shorter than this is not decoded on its own
NO_SPEECH_REJECT = 0.90                # Whisper itself says "no speech" this surely
LOOP_UNIQUE_SHARE = 0.25               # fewer distinct words than this share (over 8+ words) is a loop, not lyrics
LOW_CONFIDENCE_LOGPROB = -1.0
MAX_SAME_LINE = 4                      # in one stretch decoded without the voice filter the same line is kept at most this often (a chorus repeats a few times; a decoder that locks onto a line repeats it dozens of times)
BOUNDARY_REACH = 8.0                   # seconds on each side of a change of language where a doubtful segment is tried in the neighbouring language
BOUNDARY_DOUBT = -0.8                  # a segment decoded with a lower average log-probability than this next to a boundary is doubtful
BOUNDARY_GAIN = 0.25                   # the other language must score this much better (average log-probability) to take the segment over
_MUSIC_TAG = re.compile(r"^[\s\W_]*(?:[\[(]?(?:music|musik|müzik|音楽|음악|संगीत)[\])]?)?[\s\W_♪♫]*$", re.IGNORECASE)


def run_segments(plan: Sequence[tuple[float, float, str, str]], lo: float, hi: float) -> list[tuple[float, float, str, str]]:
    """The runs of the plan clipped to the audio ``[lo, hi]`` that this request covers."""
    out = []
    for start, end, language, vad in plan:
        s, e = max(float(start), lo), min(float(end), hi)
        if e - s >= MIN_RUN:
            out.append((s, e, language, vad))
    return out


def reject_reason(segment: Segment) -> Optional[str]:
    """Why a segment decoded WITHOUT the voice-activity filter is not kept as speech (``None`` = keep). Conservative: only what is certainly not a transcript is dropped."""
    from .. import quality as quality_mod

    text = segment.text.strip()
    if not text or _MUSIC_TAG.match(text):
        return "no text / a music tag"
    if quality_mod.is_stock_phrase(text) or is_artifact(text):
        return "a phrase Whisper writes on silence or music"
    if segment.no_speech_prob is not None and segment.no_speech_prob >= NO_SPEECH_REJECT:
        return f"Whisper's own no-speech probability {segment.no_speech_prob:.2f}"
    words = re.findall(r"\w+", text.casefold())
    if len(words) >= 8 and len(set(words)) / len(words) < LOOP_UNIQUE_SHARE:
        return "a repeated loop of the same few words"
    if segment.compression_ratio is not None and segment.compression_ratio >= 2.6 and (segment.avg_logprob or 0.0) < -0.6:
        return f"repetitive and unsure (compression ratio {segment.compression_ratio:.1f})"
    return None


def line_key(text: str) -> str:
    """The text of a line reduced to its words, so that the same line is the same line whatever its punctuation."""
    return " ".join(re.findall(r"\w+", (text or "").casefold()))


def label_segments(segments: Sequence[Segment], language: str, run: dict[str, Any]) -> None:
    """Write the run's language and how it was decided on every segment; a segment whose own writing system contradicts the run is relabelled by its script, and Latin letters that
    spell Hindi sounds in a run labelled with a Latin-script language (Whisper forced into English writes sung Hindi that way) are not called by that language."""
    for segment in segments:
        meta = {"confidence": run.get("confidence"), "source": run.get("source", "detector"), "candidates": list(run.get("candidates") or []), "run_status": run.get("status", ""),
                "vad": run.get("vad", "on")}
        label = language
        if segment.text.strip() and not compatible(segment.text, language):
            proven = script_language(segment.text)
            if proven:
                label, meta = proven, {"confidence": None, "source": "script", "candidates": [language], "run_status": run.get("status", ""), "vad": run.get("vad", "on")}
            elif dominant_script(segment.text) == "latin":              # Latin letters in a run of another writing system: English the recogniser kept as English, or romanised speech
                english = looks_english(segment.text)
                label, meta = ("en" if english else "und"), {"confidence": None, "source": "script", "candidates": [language], "run_status": run.get("status", ""), "vad": run.get("vad", "on"),
                                                              **({} if english else {"status": "uncertain"})}
        elif segment.text.strip() and language in LATIN_LANGUAGES and romanised_indic(segment.text):
            label, meta = "und", {"confidence": None, "source": "script", "candidates": [language], "run_status": run.get("status", ""), "vad": run.get("vad", "on"), "status": "uncertain"}
        segment.language = label
        if segment.avg_logprob is not None and segment.avg_logprob < LOW_CONFIDENCE_LOGPROB and "status" not in meta:
            meta["status"] = "low_confidence"
        segment.language_meta = meta


def transcribe_runs(inner: Callable[[Any], TranscriptionResult], request: Any, runs_meta: Optional[Sequence[dict[str, Any]]] = None) -> TranscriptionResult:
    """Decode ``request`` run by run with ``inner`` (the one-language decode) and join the results."""
    lo = float(request.start_offset or 0.0)
    hi = lo + float(request.max_seconds) if request.max_seconds else float(request.total_duration or 1e12)
    pieces = run_segments(request.language_plan or [], lo, hi)
    per_piece: list[list[Segment]] = []
    rejected: list[dict[str, Any]] = []
    notes: list[str] = []
    seconds_by_language: dict[str, float] = {}
    last: Optional[TranscriptionResult] = None
    load_seconds = 0.0
    fallback = False
    hotwords: list[str] = []
    for start, end, language, vad in pieces:
        profile = {**request.profile, "condition_on_previous_text": False} if vad == "off" else request.profile          # without the voice filter the previous text is what a stuck decoder repeats
        sub = dataclasses.replace(request, language=language, multilingual=False, language_plan=None, start_offset=start, max_seconds=end - start,
                                  vad_override=False if vad == "off" else request.vad_override, profile=profile)
        result = inner(sub)
        last = result
        load_seconds += result.load_seconds
        fallback = fallback or result.fallback_used
        notes.extend(n for n in result.notes if n not in notes)
        hotwords = result.hotwords_applied or hotwords
        meta = next((m for m in (runs_meta or []) if abs(float(m.get("start", -1)) - start) < 1.0 and m.get("language") == language), {})
        kept = list(result.segments)
        if vad == "off":
            kept = []
            seen: dict[str, int] = {}
            for seg in result.segments:
                reason = reject_reason(seg)
                key = line_key(seg.text)
                if not reason and key:
                    seen[key] = seen.get(key, 0) + 1
                    if seen[key] > MAX_SAME_LINE:
                        reason = f"the same line for the {seen[key]}th time in one stretch (a decoder stuck on a line, not a chorus)"
                if reason:
                    rejected.append({"start": round(seg.start, 2), "end": round(seg.end, 2), "reason": reason, "language": language})
                else:
                    kept.append(seg)
        label_segments(kept, language, {**meta, "vad": vad})
        per_piece.append(kept)
        seconds_by_language[language] = seconds_by_language.get(language, 0.0) + (end - start)
    refined = refine_boundaries(inner, request, pieces, per_piece)
    if refined:
        notes.append(f"{refined} segment(s) next to a change of language were read again in the neighbouring language (the change is only placed to within a few seconds)")
    segments = [seg for kept in per_piece for seg in kept]
    if last is None:
        return TranscriptionResult(segments=[], language=None, language_probability=None, duration=lo, model="", backend="", notes=["the language plan covered no audio"])
    dominant = max(seconds_by_language, key=seconds_by_language.get) if seconds_by_language else None
    segments.sort(key=lambda s: (s.start, s.end))
    if rejected:
        notes.append(f"{len(rejected)} decoded stretch(es) were not kept as speech (instrumental or looping output)")
    return TranscriptionResult(segments=segments, language=dominant, language_probability=None, duration=last.duration, model=last.model, backend=last.backend,
                               compute_type=last.compute_type, fallback_used=fallback, load_seconds=load_seconds,
                               capabilities_used={**last.capabilities_used, "language_runs": len(pieces)}, notes=notes, hotwords_applied=hotwords,
                               hotword_tokens=last.hotword_tokens, rejected=rejected)


def refine_boundaries(inner: Callable[[Any], TranscriptionResult], request: Any, pieces: Sequence[tuple[float, float, str, str]], per_piece: list[list[Segment]]) -> int:
    """A change of language is placed within a few seconds, not exactly. A segment next to a change that was decoded with a poor score is therefore decoded again in the NEIGHBOURING
    language; the other reading replaces it only when it scores clearly better (and is not instrumental filler). Returns the number of segments replaced (``per_piece`` is edited in place)."""
    changed = 0
    for index in range(len(pieces) - 1):
        (a0, a1, lang_a, vad_a), (b0, b1, lang_b, vad_b) = pieces[index], pieces[index + 1]
        if lang_a == lang_b or abs(a1 - b0) > 1.0:
            continue
        boundary = (a1 + b0) / 2.0
        for side, other_language, other_vad in ((index, lang_b, vad_b), (index + 1, lang_a, vad_a)):
            for position in reversed(range(len(per_piece[side]))):                      # from the end: a replacement does not shift the positions still to be visited
                seg = per_piece[side][position]
                near = seg.start >= boundary - BOUNDARY_REACH and seg.end <= boundary + BOUNDARY_REACH
                if not near or seg.avg_logprob is None or seg.avg_logprob >= BOUNDARY_DOUBT:
                    continue
                sub = dataclasses.replace(request, language=other_language, multilingual=False, language_plan=None, start_offset=seg.start,
                                          max_seconds=max(seg.end - seg.start, 0.8), vad_override=False if other_vad == "off" else request.vad_override)
                try:
                    alternative = [x for x in inner(sub).segments if x.text.strip() and (other_vad != "off" or reject_reason(x) is None)]
                except Exception as exc:  # noqa: BLE001 - the original reading simply stays
                    log.debug("boundary re-read failed at %.1f s: %s", seg.start, exc)
                    continue
                scores = [x.avg_logprob for x in alternative if x.avg_logprob is not None]
                if not alternative or not scores or sum(scores) / len(scores) < seg.avg_logprob + BOUNDARY_GAIN:
                    continue
                label_segments(alternative, other_language, {"source": "boundary", "vad": other_vad})
                per_piece[side][position:position + 1] = alternative
                changed += 1
    return changed
