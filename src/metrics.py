"""Evaluation metrics: WER and medical-term recall / precision."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any, Optional

from .vocabulary import VocabularyIndex

_PUNCT = re.compile(r"[^\w\s]|_", re.UNICODE)


def normalize_for_wer(text: str, language: Optional[str] = None) -> str:
    """Lower-case, strip punctuation, collapse spaces. Turkish-aware dotted/dotless I."""
    text = unicodedata.normalize("NFC", text)
    if language == "tr":
        text = text.replace("İ", "i").replace("I", "ı")
    text = text.casefold()
    text = _PUNCT.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def _edit_distance(ref: list[str], hyp: list[str]) -> int:
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        current = [i]
        for j, h in enumerate(hyp, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (r != h)))
        previous = current
    return previous[-1]


def word_error_rate(reference: str, hypothesis: str, language: Optional[str] = None) -> float:
    """WER after normalisation. Uses ``jiwer`` when available, otherwise an equivalent DP."""
    ref = normalize_for_wer(reference, language)
    hyp = normalize_for_wer(hypothesis, language)
    if not ref:
        raise ValueError("reference transcript is empty")
    try:
        import jiwer

        return float(jiwer.wer(ref, hyp if hyp else ""))
    except ImportError:
        return _edit_distance(ref.split(), hyp.split()) / len(ref.split())


def medical_term_scores(reference: str, hypothesis: str, vocab: VocabularyIndex) -> dict[str, Any]:
    """Medical Term Recall / Precision over dictionary terms (multiset matching).

    recall    = matched terms / dictionary terms in the REFERENCE
    precision = matched terms / dictionary terms in the HYPOTHESIS
    Only terms in the supplied vocabulary are counted, so keep the vocabulary broad for fair scores.
    """
    ref_terms = Counter(t.casefold() for t in vocab.find_terms(reference))
    hyp_terms = Counter(t.casefold() for t in vocab.find_terms(hypothesis))
    matched = sum((ref_terms & hyp_terms).values())
    ref_total, hyp_total = sum(ref_terms.values()), sum(hyp_terms.values())
    recall = matched / ref_total if ref_total else None
    precision = matched / hyp_total if hyp_total else None
    f1 = None
    if recall is not None and precision is not None and (recall + precision) > 0:
        f1 = 2 * recall * precision / (recall + precision)
    return {
        "reference_terms": ref_total,
        "hypothesis_terms": hyp_total,
        "matched_terms": matched,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "missed": sorted((ref_terms - hyp_terms).elements()),
        "extra": sorted((hyp_terms - ref_terms).elements()),
    }
