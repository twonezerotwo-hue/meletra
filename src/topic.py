"""Guess the lecture subject from the first minutes, so only relevant vocabulary is prioritised.

Scoring is deliberately simple and explainable (no LLM): every word of a subject's term list is
weighted by 1 / (number of subjects containing it), so words shared by many subjects ("nerve",
"tract") count little and subject-specific words ("lemniscus", "glycolysis") count a lot. Exact
multi-word dictionary terms heard in the text add extra weight.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .vocabulary import (
    DEFAULT_TERMS_DIR, VocabularyIndex, available_subjects, build_index, load_terms, normalize_key, tokenize,
)

log = logging.getLogger(__name__)

GENERIC_SUBJECT = "general_medical"
MIN_WORD_LENGTH = 4
PER_WORD_CAP = 5            # one repeated word must not decide the subject alone
PHRASE_WEIGHT = 3.0         # an exact multi-word term is stronger evidence than a shared word


@dataclass
class TopicResult:
    scores: dict[str, float] = field(default_factory=dict)
    selected: list[str] = field(default_factory=list)
    matched_terms: list[str] = field(default_factory=list)   # dictionary terms literally heard, most frequent first
    words_analyzed: int = 0
    confident: bool = False

    def to_dict(self) -> dict:
        return {
            "scores": {k: round(v, 2) for k, v in sorted(self.scores.items(), key=lambda kv: -kv[1])},
            "selected": self.selected, "matched_terms": self.matched_terms,
            "words_analyzed": self.words_analyzed, "confident": self.confident,
        }


def detect_topic(
    text: str,
    language: Optional[str] = None,
    terms_dir: Path = DEFAULT_TERMS_DIR,
    max_subjects: int = 2,
    relative_cutoff: float = 0.5,
    min_score: float = 4.0,
) -> TopicResult:
    """Rank subjects for ``text``; ``selected`` holds at most ``max_subjects`` with a real signal.

    When nothing scores at least ``min_score`` the result is *not confident* and ``selected`` falls
    back to the generic list, so a wrongly guessed subject can never bias the decoder.
    """
    subjects = [s for s in available_subjects(terms_dir) if s != GENERIC_SUBJECT]
    word_sets: dict[str, set[str]] = {}
    for subject in subjects:
        words = {normalize_key(w) for term in load_terms(subject, language, terms_dir) for w in tokenize(term)}
        word_sets[subject] = {w for w in words if len(w) >= MIN_WORD_LENGTH}
    document_frequency = Counter(w for words in word_sets.values() for w in words)

    counts = Counter(normalize_key(w) for w in tokenize(text))
    result = TopicResult(words_analyzed=sum(counts.values()))
    for subject in subjects:
        weighted = sum(
            min(counts[w], PER_WORD_CAP) / document_frequency[w] for w in word_sets[subject] if w in counts
        )
        index = build_index(subject, language, terms_dir)
        phrases = [t for t in index.find_terms(text) if len(tokenize(t)) > 1]
        result.scores[subject] = weighted + PHRASE_WEIGHT * min(len(phrases), 10)

    ranked = sorted(result.scores.items(), key=lambda kv: -kv[1])
    top = ranked[0][1] if ranked else 0.0
    if top >= min_score:
        result.selected = [s for s, v in ranked if v >= relative_cutoff * top and v >= min_score][:max_subjects]
        result.confident = True
    else:
        result.selected = [GENERIC_SUBJECT]

    everything = build_index("all", language, terms_dir)
    heard = Counter(everything.find_terms(text))
    result.matched_terms = [term for term, _ in heard.most_common()]
    log.info("Topic guess from %d words: %s (scores %s)", result.words_analyzed, result.selected,
             {k: round(v, 1) for k, v in ranked[:4]})
    return result
