"""Quality audit of a finished transcript against an independent Silero VAD pass.

It never changes the transcript. It reports:
  * timestamp anomalies (reversed, zero-length, past the end, overlapping, non-monotonic, words outside their segment)
  * hallucination signals (stock phrases Whisper invents, text sitting in a region the VAD calls non-speech)
  * repetition signals (the uncertainty flags that mean loops / duplicates)
  * missing speech (VAD says someone is talking, but no segment covers it)
  * per-window statistics for the first / middle / last N minutes
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence

from .multilingual.artifacts import is_credit
from .schema import Segment, UncertaintyFlag
from .vocabulary import normalize_key, tokenize

log = logging.getLogger(__name__)

REPETITION_CODES = ("repetition", "repeated_segment", "duplicate_nearby", "hotword_echo")

DEFAULTS: dict[str, Any] = {
    "hallucination_phrases": [
        "thanks for watching", "thank you for watching", "please subscribe", "like and subscribe",
        "subtitles by", "subtitle by", "amara org", "transcribed by", "abone ol", "altyazı", "izlediğiniz için teşekkürler",
        "bir sonraki videoda görüşmek üzere", "www ", "http",
        "köszönöm hogy megnézted", "köszönöm hogy megnéztétek", "iratkozz fel", "vielen dank fürs zuschauen",
        "untertitel der amara", "merci d avoir regardé", "sous titres", "gracias por ver", "suscríbete",
        "спасибо за просмотр", "продолжение следует",
    ],
    "non_speech_overlap_max": 0.15,    # a segment with less VAD-speech overlap than this sits in non-speech
    "non_speech_min_words": 3,
    "non_speech_min_seconds": 1.5,
    "overlap_tolerance_s": 0.3,
    "zero_length_s": 0.05,
    "max_segment_s": 35.0,
    "word_tolerance_s": 0.5,
    "missing_speech_min_gap_s": 8.0,
    "window_minutes": 10.0,
}


def is_stock_phrase(text: str, extra_words: int = 3) -> bool:
    """True when ``text`` is (essentially only) a phrase Whisper invents on silence, e.g. "Köszönöm, hogy megnézted!".

    A long segment that merely contains such a phrase is real speech; only short texts count, so a recovered passage is never
    thrown away for one stray "thank you" at its end.
    """
    key = normalize_key(text)
    words = len(tokenize(text))
    for phrase in DEFAULTS["hallucination_phrases"]:
        p = normalize_key(phrase)
        if p and p in key and words <= len(p.split()) + extra_words:
            return True
    return False


def compute_speech_ranges(wav_path: Path, vad_parameters: dict[str, Any]) -> list[tuple[float, float]]:
    """Speech regions (seconds) from the same Silero VAD that faster-whisper uses."""
    from faster_whisper.audio import decode_audio
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    rate = 16000
    audio = decode_audio(str(wav_path), sampling_rate=rate)
    fields = {k: v for k, v in vad_parameters.items() if k in VadOptions.__dataclass_fields__}
    chunks = get_speech_timestamps(audio, VadOptions(**fields), sampling_rate=rate)
    return [(c["start"] / rate, c["end"] / rate) for c in chunks]


def find_speech_onset(
    speech: Sequence[tuple[float, float]], min_speech_s: float = 20.0, lookahead_s: float = 60.0,
) -> float:
    """Start (seconds) of the first *sustained* speech: at least ``min_speech_s`` of VAD speech within
    ``lookahead_s``. Skips leading silence, room noise and short blips (the real recording starts with ~5
    minutes of digital silence, so "the first 4 minutes" would otherwise be analysed as nothing)."""
    for start, _end in speech:
        if _overlap(start, start + lookahead_s, speech) >= min_speech_s:
            return max(0.0, start - 1.0)
    return 0.0


def _overlap(a_start: float, a_end: float, ranges: Sequence[tuple[float, float]]) -> float:
    return sum(max(0.0, min(a_end, e) - max(a_start, s)) for s, e in ranges if s < a_end and e > a_start)


def _union(intervals: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for s, e in sorted(intervals):
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _entry(index: int, seg: Segment, code: str, detail: str) -> dict[str, Any]:
    return {"segment": index, "start": round(seg.start, 2), "end": round(seg.end, 2), "code": code,
            "detail": detail, "text": seg.text.strip()[:120]}


def timestamp_issues(segments: Sequence[Segment], duration: float, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    previous: Optional[Segment] = None
    for i, seg in enumerate(segments):
        words = len(tokenize(seg.text))
        if seg.start < 0 or seg.end < seg.start:
            issues.append(_entry(i, seg, "reversed_or_negative", f"start {seg.start:.2f} end {seg.end:.2f}"))
        elif seg.end - seg.start < cfg["zero_length_s"] and words >= 2:
            issues.append(_entry(i, seg, "zero_length", f"{words} words in {seg.end - seg.start:.2f}s"))
        if seg.end > duration + 0.5:
            issues.append(_entry(i, seg, "beyond_audio_end", f"end {seg.end:.2f} > duration {duration:.2f}"))
        if seg.end - seg.start > cfg["max_segment_s"]:
            issues.append(_entry(i, seg, "overlong_segment", f"{seg.end - seg.start:.1f}s"))
        if previous is not None:
            if seg.start < previous.start:
                issues.append(_entry(i, seg, "non_monotonic", f"starts at {seg.start:.2f} before previous start {previous.start:.2f}"))
            elif seg.start < previous.end - cfg["overlap_tolerance_s"]:
                issues.append(_entry(i, seg, "overlap_with_previous", f"{previous.end - seg.start:.2f}s overlap"))
        tol = cfg["word_tolerance_s"]
        bad = [w for w in seg.words if w.start < seg.start - tol or w.end > seg.end + tol]
        if bad:
            issues.append(_entry(i, seg, "word_outside_segment", f"{len(bad)} word(s) outside [{seg.start:.1f},{seg.end:.1f}]"))
        previous = seg
    return issues


def hallucination_flags(
    segments: Sequence[Segment], speech: Sequence[tuple[float, float]], cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    phrases = [normalize_key(p) for p in cfg["hallucination_phrases"]]
    for i, seg in enumerate(segments):
        key = normalize_key(seg.text)
        hit = next((p for p in phrases if p and p in key), None)
        if hit:
            flags.append(_entry(i, seg, "stock_phrase", "contains a phrase Whisper commonly invents"))
        elif is_credit(seg.text):                                   # Hangul, Han and Devanagari have no letters in the ASCII key above: the same credits in the other offered languages
            flags.append(_entry(i, seg, "stock_phrase", "a subtitle credit or sign-off Whisper commonly invents"))
        duration = seg.end - seg.start
        words = len(tokenize(seg.text))
        if speech and duration >= cfg["non_speech_min_seconds"] and words >= cfg["non_speech_min_words"]:
            share = _overlap(seg.start, seg.end, speech) / duration if duration else 0.0
            if share < cfg["non_speech_overlap_max"]:
                flags.append(_entry(i, seg, "text_in_non_speech", f"only {share:.0%} of the segment overlaps VAD speech"))
    return flags


def repetition_flags(segments: Sequence[Segment]) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    for i, seg in enumerate(segments):
        for flag in seg.flags:
            if flag.code in REPETITION_CODES:
                flags.append(_entry(i, seg, flag.code, flag.detail))
    return flags


#: The VAD cuts a continuous stretch of speech into several ranges at every short pause; uncovered pieces that touch across such
#: a cut (or are separated by less than this) are ONE missing stretch. Measured: a 9.5 s sentence went missing as two uncovered
#: pieces of 4.3 s and 4.6 s, each under the 5 s threshold, and was never reported or repaired.
GAP_BRIDGE_SECONDS = 0.6


def uncovered_pieces(
    speech: Sequence[tuple[float, float]], covered: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Every piece of VAD speech that the (merged, sorted) ``covered`` intervals leave open, before any size filter."""
    pieces: list[tuple[float, float]] = []
    for s, e in speech:
        cursor = s
        for cs, ce in covered:
            if ce <= cursor:
                continue
            if cs >= e:
                break
            if cs > cursor:
                pieces.append((cursor, min(cs, e)))
            cursor = max(cursor, ce)
            if cursor >= e:
                break
        if cursor < e:
            pieces.append((cursor, e))
    return pieces


def bridge_pieces(
    pieces: Sequence[tuple[float, float]], covered: Sequence[tuple[float, float]], bridge: float = GAP_BRIDGE_SECONDS,
) -> list[tuple[float, float]]:
    """Join uncovered pieces that are separated only by a VAD cut: closer than ``bridge`` and with no covered text in between."""
    joined: list[list[float]] = []
    for a, b in sorted(pieces):
        if joined and a - joined[-1][1] <= bridge and not any(cs < a and ce > joined[-1][1] for cs, ce in covered):
            joined[-1][1] = max(joined[-1][1], b)
        else:
            joined.append([a, b])
    return [(a, b) for a, b in joined]


def missing_speech(
    segments: Sequence[Segment], speech: Sequence[tuple[float, float]], min_gap_s: float,
) -> tuple[list[dict[str, float]], float]:
    """Speech (per VAD) that no segment covers. Returns ``(gaps >= min_gap_s, coverage 0-1)``."""
    covered = _union([(s.start, s.end) for s in segments])
    total = sum(e - s for s, e in speech)
    pieces = uncovered_pieces(speech, covered)
    uncovered_total = sum(b - a for a, b in pieces)
    gaps = [{"start": round(a, 2), "end": round(b, 2), "seconds": round(b - a, 2)}
            for a, b in bridge_pieces(pieces, covered) if b - a >= min_gap_s]
    coverage = 1.0 - uncovered_total / total if total else 1.0
    return gaps, coverage


def window_stats(
    segments: Sequence[Segment], duration: float, window_minutes: float, speech: Sequence[tuple[float, float]],
) -> dict[str, dict[str, Any]]:
    """Stats for the first, middle and last ``window_minutes`` of the recording."""
    span = window_minutes * 60.0
    middle_start = max(0.0, duration / 2 - span / 2)
    windows = {
        "beginning": (0.0, min(span, duration)),
        "middle": (middle_start, min(middle_start + span, duration)),
        "end": (max(0.0, duration - span), duration),
    }
    result: dict[str, dict[str, Any]] = {}
    for name, (lo, hi) in windows.items():
        inside = [s for s in segments if lo <= s.start < hi]
        words = sum(len(tokenize(s.text)) for s in inside)
        minutes = max((hi - lo) / 60.0, 1e-9)
        speech_s = _overlap(lo, hi, speech)
        gaps, coverage = missing_speech([s for s in inside], [(max(a, lo), min(b, hi)) for a, b in speech if b > lo and a < hi], 8.0)
        logprobs = [s.avg_logprob for s in inside if s.avg_logprob is not None]
        codes = Counter(f.code for s in inside for f in s.flags)
        result[name] = {
            "from": round(lo, 1), "to": round(hi, 1), "segments": len(inside), "words": words,
            "words_per_minute": round(words / minutes, 1), "vad_speech_seconds": round(speech_s, 1),
            "speech_coverage": round(coverage, 3), "uncovered_gaps": len(gaps),
            "uncertain_segments": sum(1 for s in inside if s.uncertain),
            "flag_codes": dict(codes),
            "mean_avg_logprob": round(sum(logprobs) / len(logprobs), 3) if logprobs else None,
            "min_avg_logprob": round(min(logprobs), 3) if logprobs else None,
        }
    return result


def add_engine_flags(
    segments: Sequence[Segment], duration: float, speech: Sequence[tuple[float, float]], config: Optional[dict[str, Any]] = None,
) -> int:
    """Fold timestamp anomalies and hallucination signals into the segments' own ``flags``.

    The decode-time signals (avg_logprob, no_speech_prob, compression_ratio, repetition, duplicate_nearby,
    low_speech_density) are flagged by :mod:`src.uncertainty`; this adds the two that need the whole timeline
    (timestamp anomalies, stock phrases / text over non-speech) so ``uncertain_segments.json`` is ONE engine's output.
    Idempotent. Returns the number of flags added.
    """
    cfg = {**DEFAULTS, **(config or {})}
    added = 0
    present = {(id(seg), f.code, f.detail) for seg in segments for f in seg.flags}
    for issue in timestamp_issues(segments, duration, cfg):
        seg = segments[issue["segment"]]
        flag = UncertaintyFlag("timestamp_anomaly", f"{issue['code']}: {issue['detail']}")
        if (id(seg), flag.code, flag.detail) not in present:
            seg.flags.append(flag)
            added += 1
    for issue in hallucination_flags(segments, speech, cfg):
        seg = segments[issue["segment"]]
        code = "hallucination_stock_phrase" if issue["code"] == "stock_phrase" else "hallucination_text_in_non_speech"
        flag = UncertaintyFlag(code, issue["detail"])
        if (id(seg), flag.code, flag.detail) not in present:
            seg.flags.append(flag)
            added += 1
    return added


def analyze_quality(
    segments: Sequence[Segment], duration: float, speech: Sequence[tuple[float, float]],
    config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    cfg = {**DEFAULTS, **(config or {})}
    ts = timestamp_issues(segments, duration, cfg)
    hall = hallucination_flags(segments, speech, cfg)
    rep = repetition_flags(segments)
    gaps, coverage = missing_speech(segments, speech, cfg["missing_speech_min_gap_s"])
    total_words = sum(len(tokenize(s.text)) for s in segments)
    return {
        "summary": {
            "segments": len(segments), "words": total_words, "audio_seconds": round(duration, 1),
            "vad_speech_seconds": round(sum(e - s for s, e in speech), 1),
            "speech_coverage": round(coverage, 4),
            "timestamp_issues": len(ts), "hallucination_flags": len(hall), "repetition_flags": len(rep),
            "missing_speech_gaps": len(gaps),
            "missing_speech_seconds": round(sum(g["seconds"] for g in gaps), 1),
        },
        "timestamp_issues": ts, "hallucination_flags": hall, "repetition_flags": rep, "missing_speech": gaps,
        "windows": window_stats(segments, duration, cfg["window_minutes"], speech),
    }
