"""Shared helpers of the medical expert tests."""

from __future__ import annotations

from typing import Optional, Sequence

from src.medical.corrector import MedicalContextCorrector, Options, SegmentView
from src.medical.lexicon import Lexicon, shared_lexicon
from src.schema import Segment, Word


def views(*texts: str, probs: Optional[dict[int, dict[str, float]]] = None, step: float = 4.0) -> list[SegmentView]:
    """Segments of ``step`` seconds each. ``probs[i]`` maps a word to the ASR probability it had in segment ``i``."""
    out = []
    for i, text in enumerate(texts):
        word_probs = []
        for word in text.split():
            key = word.strip(".,;:?!").lower()
            word_probs.append((word, (probs or {}).get(i, {}).get(key, 0.97)))
        out.append(SegmentView(i, i * step, (i + 1) * step - 0.5, text, "en", word_probs))
    return out


def correct(*texts: str, subjects: Sequence[str] = (), custom: Sequence[str] = (), lecture: Sequence[str] = (),
            probs: Optional[dict[int, dict[str, float]]] = None, lexicon: Optional[Lexicon] = None):
    corrector = MedicalContextCorrector(lexicon or shared_lexicon(), Options(subjects=tuple(subjects), custom_terms=tuple(custom), lecture_terms=tuple(lecture)))
    return corrector.correct(views(*texts, probs=probs))


def applied(result) -> list[tuple[str, str]]:
    return [(c.original, c.corrected) for c in result.corrections if c.category == "medical" and c.tier in ("high", "medium")]


def seg(text: str, start: float, end: float, confidence: Optional[float] = 0.9, words: bool = False,
        doubt: Optional[dict[str, float]] = None) -> Segment:
    """A raw segment. ``doubt`` maps a word to the (low) probability the recogniser gave it; every other word has 0.95."""
    ws = []
    if words:
        pieces = text.split()
        step = (end - start) / max(len(pieces), 1)
        ws = [Word(start + i * step, start + (i + 1) * step, " " + w, (doubt or {}).get(w.strip(".,;:?!").lower(), 0.95)) for i, w in enumerate(pieces)]
    return Segment(start, end, text, "en", "m", "fake", confidence=confidence, words=ws)
