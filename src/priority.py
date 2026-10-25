"""Medical term priority engine: score every candidate term instead of sending whole word lists.

score = w_subject*subject_match + w_pdf*pdf_presence + w_early*early_transcript_presence
        + w_freq*frequency + w_rare*rarity + w_spec*medical_specificity           (weights sum to 1.0)
then + a bonus for multi-word terms, and x a penalty for generic single words.

The highest-scoring terms become the hotword candidates; the backend's *token* budget (real tokenizer)
decides how many of them fit.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional

from .knowledge_entities import (
    ANATOMICAL_HEADS, COMMON_ENGLISH, GENERIC_TERMS, LATINATE_PREFIXES, LATINATE_SUFFIXES,
)
from .vocabulary import normalize_key, tokenize

DEFAULT_WEIGHTS: dict[str, float] = {
    "subject_match": 0.30, "pdf_presence": 0.25, "early_transcript_presence": 0.15,
    "frequency": 0.10, "rarity": 0.10, "medical_specificity": 0.10,
}
FREQUENCY_CAP = 6               # counts above this add nothing
GENERIC_PENALTY = 0.4


@dataclass
class TermScore:
    term: str
    score: float
    components: dict[str, float] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)       # vocabulary files / "pdf" / "heard"
    words: int = 1

    def to_dict(self) -> dict:
        return {"term": self.term, "score": round(self.score, 4), "words": self.words, "sources": self.sources,
                "components": {k: round(v, 3) for k, v in self.components.items()}}


def rarity(term: str) -> float:
    """0 (everyday word) .. 1 (rare): share of non-common words plus a long-word signal."""
    words = [w.casefold() for w in tokenize(term)]
    if not words:
        return 0.0
    uncommon = sum(1 for w in words if w not in COMMON_ENGLISH) / len(words)
    average_length = sum(len(w) for w in words) / len(words)
    return max(0.0, min(1.0, 0.5 * uncommon + 0.5 * min(1.0, (average_length - 4) / 8)))


def medical_specificity(term: str) -> float:
    """How clearly medical a term is: Latin/Greek morphology, anatomical head noun, eponym, acronym."""
    words = tokenize(term)
    if not words:
        return 0.0
    lowered = [w.casefold() for w in words]
    score = 0.25
    if any(w.endswith(LATINATE_SUFFIXES) and len(w) > 5 for w in lowered):
        score += 0.30
    if any(w.startswith(LATINATE_PREFIXES) and len(w) > 6 for w in lowered):
        score += 0.20
    if lowered[-1] in ANATOMICAL_HEADS:
        score += 0.25
    if any(w[:1].isupper() for w in words[1:]) or (len(words) > 1 and words[0][:1].isupper() and words[0].lower() not in COMMON_ENGLISH):
        score += 0.10                                       # eponym ("Broca area", "Michaelis-Menten")
    if any(w.isupper() and 2 <= len(w) <= 6 for w in words):
        score += 0.15                                       # acronym
    if len(words) == 1 and lowered[0] in GENERIC_TERMS:
        score = min(score, 0.1)
    return max(0.0, min(1.0, score))


def log_frequency(count: int) -> float:
    return min(1.0, math.log1p(count) / math.log1p(FREQUENCY_CAP))


class TermPriorityEngine:
    """Ranks candidate terms for one lecture."""

    def __init__(self, weights: Optional[Mapping[str, float]] = None, multiword_bonus: Optional[Mapping[str, float]] = None) -> None:
        merged = {**DEFAULT_WEIGHTS, **(weights or {})}
        total = sum(merged.values()) or 1.0
        self.weights = {k: v / total for k, v in merged.items()}           # always sums to 1
        self.multiword_bonus = {2: 0.15, 3: 0.20, **{int(k): v for k, v in (multiword_bonus or {}).items()}}

    def bonus(self, words: int) -> float:
        if words < 2:
            return 0.0
        return self.multiword_bonus.get(words, self.multiword_bonus.get(3, 0.2))

    def rank(
        self,
        candidates: Mapping[str, Iterable[str]],              # term -> sources (vocabulary files, "pdf", "heard")
        file_scores: Mapping[str, float],                     # vocabulary file -> 0-1 relevance
        pdf_terms: Iterable[str] = (),
        pdf_text: str = "",
        early_counts: Optional[Counter] = None,
        pdf_counts: Optional[Counter] = None,
    ) -> list[TermScore]:
        pdf_keys = {normalize_key(t) for t in pdf_terms}
        pdf_blob = normalize_key(pdf_text)
        early = Counter({normalize_key(k): v for k, v in (early_counts or {}).items()})
        pdfc = Counter({normalize_key(k): v for k, v in (pdf_counts or {}).items()})
        ranked: list[TermScore] = []
        for term, sources in candidates.items():
            key = normalize_key(term)
            if not key:
                continue
            sources = list(sources)
            words = len(tokenize(term))
            subject = max([file_scores.get(s, 0.0) for s in sources] + [0.0])
            in_pdf = key in pdf_keys or (len(key) >= 7 and key in pdf_blob)
            comps = {
                "subject_match": subject,
                "pdf_presence": 1.0 if in_pdf else 0.0,
                "early_transcript_presence": 1.0 if early.get(key, 0) > 0 else 0.0,
                "frequency": log_frequency(early.get(key, 0) + pdfc.get(key, 0)),
                "rarity": rarity(term),
                "medical_specificity": medical_specificity(term),
            }
            score = sum(self.weights[name] * value for name, value in comps.items())
            score += self.bonus(words)
            if words == 1 and term.casefold() in GENERIC_TERMS:
                score *= GENERIC_PENALTY
            comps["multiword_bonus"] = self.bonus(words)
            ranked.append(TermScore(term=term, score=min(score, 1.5), components=comps, sources=sorted(set(sources)), words=words))
        ranked.sort(key=lambda r: (-r.score, -r.words, -len(r.term), r.term.casefold()))
        return ranked
