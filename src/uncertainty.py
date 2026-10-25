"""Heuristic confidence and uncertain-segment classification.

All thresholds come from ``config/transcription.yaml`` (``uncertainty`` and ``confidence``).
The confidence value is a *heuristic*, not a calibrated model probability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from .schema import SEGMENT_SCOPE, TERM_SCOPE, Segment, UncertaintyFlag
from .vocabulary import VocabularyIndex, normalize_key, tokenize

CONFIDENCE_METHOD = (
    "heuristic = clip(exp(avg_logprob), 0, 1) * (1 - no_speech_prob) * compression_penalty "
    "(penalty applies only when compression_ratio exceeds the threshold). This is NOT a real model "
    "confidence; it is derived from Whisper's decoding statistics. Backends that expose no "
    "statistics (Transformers pipeline) report confidence = null."
)


@dataclass(frozen=True)
class UncertaintyConfig:
    avg_logprob_threshold: float
    no_speech_threshold: float
    compression_ratio_threshold: float
    min_words: int = 2
    min_chars: int = 6
    repetition_min_repeats: int = 4
    repetition_max_ngram: int = 4
    repeated_segment_count: int = 3
    hotword_echo_min_words: int = 4
    duplicate_lookback: int = 6
    duplicate_min_words: int = 4
    low_density_min_seconds: float = 6.0
    min_words_per_second: float = 0.6
    near_miss_enabled: bool = True
    near_miss_ratio: float = 0.88
    near_miss_min_length: int = 7
    compression_penalty: float = 0.5

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "UncertaintyConfig":
        """Build from the full transcription config dict."""
        unc = cfg["uncertainty"]
        known = {k: unc[k] for k in cls.__dataclass_fields__ if k in unc}
        penalty = cfg.get("confidence", {}).get("compression_penalty", 0.5)
        return cls(compression_penalty=penalty, **known)


def compute_confidence(segment: Segment, cfg: UncertaintyConfig) -> Optional[float]:
    """Heuristic confidence in [0, 1], or ``None`` when the backend gave no statistics."""
    if segment.avg_logprob is None:
        return None
    base = min(1.0, max(0.0, math.exp(segment.avg_logprob)))
    base *= 1.0 - min(1.0, max(0.0, segment.no_speech_prob or 0.0))
    if segment.compression_ratio is not None and segment.compression_ratio > cfg.compression_ratio_threshold:
        base *= cfg.compression_penalty
    return round(base, 4)


def find_repetition(text: str, min_repeats: int, max_ngram: int) -> Optional[str]:
    """Return the repeated phrase if a word/phrase loops ``min_repeats``+ times in a row."""
    words = [normalize_key(w) for w in tokenize(text)]
    words = [w for w in words if w]
    total = len(words)
    for size in range(1, max_ngram + 1):
        for start in range(0, total - size * min_repeats + 1):
            gram = words[start:start + size]
            repeats = 1
            while words[start + repeats * size:start + (repeats + 1) * size] == gram:
                repeats += 1
            if repeats >= min_repeats:
                return " ".join(tokenize(text)[start:start + size])
    return None


def _is_hotword_echo(text: str, hotwords: Sequence[str], min_words: int) -> bool:
    """True if the segment is a verbatim contiguous slice of the hotword list (prompt echo)."""
    if not hotwords:
        return False
    words = [normalize_key(w) for w in tokenize(text)]
    words = [w for w in words if w]
    if len(words) < min_words:
        return False
    hot = [normalize_key(w) for term in hotwords for w in tokenize(term)]
    hot = [w for w in hot if w]
    span = len(words)
    return any(hot[i:i + span] == words for i in range(len(hot) - span + 1))


def _acoustic_flags(segment: Segment, cfg: UncertaintyConfig) -> list[UncertaintyFlag]:
    flags: list[UncertaintyFlag] = []
    if segment.avg_logprob is not None and segment.avg_logprob < cfg.avg_logprob_threshold:
        flags.append(UncertaintyFlag(
            "low_avg_logprob", f"avg_logprob {segment.avg_logprob:.2f} < {cfg.avg_logprob_threshold}"))
    if segment.no_speech_prob is not None and segment.no_speech_prob > cfg.no_speech_threshold:
        flags.append(UncertaintyFlag(
            "high_no_speech_prob", f"no_speech_prob {segment.no_speech_prob:.2f} > {cfg.no_speech_threshold}"))
    if segment.compression_ratio is not None and segment.compression_ratio > cfg.compression_ratio_threshold:
        flags.append(UncertaintyFlag(
            "high_compression_ratio",
            f"compression_ratio {segment.compression_ratio:.2f} > {cfg.compression_ratio_threshold}"))
    return flags


def _text_flags(
    segment: Segment, cfg: UncertaintyConfig, hotwords: Sequence[str]
) -> list[UncertaintyFlag]:
    flags: list[UncertaintyFlag] = []
    words = tokenize(segment.text)
    letters = sum(ch.isalpha() for ch in segment.text)
    if len(words) < cfg.min_words and letters < cfg.min_chars:
        flags.append(UncertaintyFlag(
            "too_short", f"{len(words)} word(s), {letters} letters: too short to be reliable"))
    duration = segment.end - segment.start
    if duration >= cfg.low_density_min_seconds and len(words) / duration < cfg.min_words_per_second:
        flags.append(UncertaintyFlag(
            "low_speech_density",
            f"{len(words)} words in {duration:.1f}s ({len(words) / duration:.2f}/s < {cfg.min_words_per_second}/s): "
            "text may have been cut off"))
    loop = find_repetition(segment.text, cfg.repetition_min_repeats, cfg.repetition_max_ngram)
    if loop:
        flags.append(UncertaintyFlag(
            "repetition", f"'{loop}' repeats {cfg.repetition_min_repeats}+ times (possible hallucination loop)"))
    if _is_hotword_echo(segment.text, hotwords, cfg.hotword_echo_min_words):
        flags.append(UncertaintyFlag(
            "hotword_echo", "text is a verbatim slice of the hotword list (prompt echo / hallucination)"))
    return flags


def _near_miss_flags(
    segment: Segment, cfg: UncertaintyConfig, vocab: Optional[VocabularyIndex]
) -> list[UncertaintyFlag]:
    if vocab is None or not cfg.near_miss_enabled:
        return []
    flags: list[UncertaintyFlag] = []
    seen: set[str] = set()
    for word in tokenize(segment.text):
        key = normalize_key(word)
        if key in seen:
            continue
        seen.add(key)
        suggestion = vocab.near_miss(word, cfg.near_miss_ratio, cfg.near_miss_min_length)
        if suggestion:
            flags.append(UncertaintyFlag(
                "near_miss_term", f"'{word}' resembles dictionary term '{suggestion}' but is not one",
                scope=TERM_SCOPE, token=word, suggestion=suggestion))
    return flags


def _flag_repeated_segments(segments: Sequence[Segment], cfg: UncertaintyConfig) -> None:
    """Identical consecutive segments (classic Whisper loop) are all flagged."""
    run_start = 0
    keys = [normalize_key(s.text) for s in segments]
    for index in range(1, len(segments) + 1):
        if index < len(segments) and keys[index] == keys[run_start] and keys[run_start]:
            continue
        run_length = index - run_start
        if run_length >= cfg.repeated_segment_count:
            for member in segments[run_start:index]:
                if not any(f.code == "repeated_segment" for f in member.flags):
                    member.flags.append(UncertaintyFlag(
                        "repeated_segment", f"same text repeated in {run_length} consecutive segments"))
        run_start = index


def _flag_nearby_duplicates(segments: Sequence[Segment], cfg: UncertaintyConfig) -> None:
    """The same sentence coming back a few segments later (loops that are not strictly consecutive)."""
    keys = [normalize_key(s.text) for s in segments]
    word_counts = [len(tokenize(s.text)) for s in segments]
    for index, key in enumerate(keys):
        if not key or word_counts[index] < cfg.duplicate_min_words:
            continue
        earlier = range(max(0, index - cfg.duplicate_lookback), index)
        match = next((j for j in earlier if keys[j] == key), None)
        # skip strictly adjacent runs: _flag_repeated_segments handles those (needs 3+)
        if match is not None and not any(f.code == "duplicate_nearby" for f in segments[index].flags):
            segments[index].flags.append(UncertaintyFlag(
                "duplicate_nearby", f"same text as segment {match} (possible repetition loop)"))


def annotate_segments(
    segments: Sequence[Segment],
    cfg: UncertaintyConfig,
    vocab: Optional[VocabularyIndex] = None,
    hotwords: Sequence[str] = (),
) -> int:
    """Set ``confidence`` and ``flags`` on every segment in place. Returns the uncertain count."""
    for segment in segments:
        segment.confidence = compute_confidence(segment, cfg)
        segment.flags = [
            *_acoustic_flags(segment, cfg),
            *_text_flags(segment, cfg, hotwords),
            *_near_miss_flags(segment, cfg, vocab),
        ]
    _flag_repeated_segments(segments, cfg)
    _flag_nearby_duplicates(segments, cfg)
    return sum(1 for s in segments if s.uncertain)


def segment_scope_flags(segment: Segment) -> list[UncertaintyFlag]:
    return [f for f in segment.flags if f.scope == SEGMENT_SCOPE]


def term_scope_flags(segment: Segment) -> list[UncertaintyFlag]:
    return [f for f in segment.flags if f.scope == TERM_SCOPE]
