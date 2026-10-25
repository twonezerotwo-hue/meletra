"""Source grounding: an item may only enter the study pack when the lecture supports it.

Providers (especially language models) can write something fluent that the lecture never said. Every item therefore has to

* cite at least one passage that exists in the index (unknown ids are dropped; an item with no valid citation gets the best
  matching passage from the local BM25 search, or is dropped), and
* be *supported* by the cited passages: most of its content words occur there, and every number in it occurs there too
  (a wrong dose or a wrong date is the typical hallucination).

In SOURCE-BOUND mode (the default) an item that fails is dropped and counted; it is never shown to the student.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from . import text as T
from .i18n import canonical_true_false
from .index import DocumentIndex
from .models import Chunk, Flashcard, Question, SourceRef, SummaryItem

MIN_SUPPORT = 0.55
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
_NUMBER_WORDS = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"}


@dataclass
class GroundingReport:
    kept: int = 0
    dropped: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    def drop(self, reason: str) -> None:
        self.dropped += 1
        self.reasons[reason] = self.reasons.get(reason, 0) + 1

    def merge(self, other: "GroundingReport") -> None:
        self.kept += other.kept
        self.dropped += other.dropped
        for k, v in other.reasons.items():
            self.reasons[k] = self.reasons.get(k, 0) + v

    def to_dict(self) -> dict:
        return {"kept": self.kept, "dropped": self.dropped, "reasons": dict(self.reasons)}


def _stems(text: str) -> set[str]:
    return {T.stem(w) for w in T.content_words(text)}


def _numbers(text: str) -> set[str]:
    return {n.replace(",", ".") for n in _NUMBER.findall(text)}


def support_score(claim: str, chunks: Sequence[Chunk]) -> float:
    """Share (0..1) of the content words of ``claim`` that occur in the cited passages; 0 if a number of the claim is missing there."""
    stems = _stems(claim)
    if not stems:
        return 0.0
    source = " ".join(c.text for c in chunks)
    if not _numbers(claim) <= _numbers(source):
        return 0.0
    return T.overlap(stems, _stems(source))


def resolve_sources(ids: Iterable[str], index: DocumentIndex) -> list[SourceRef]:
    out: list[SourceRef] = []
    for chunk_id in ids:
        chunk = index.get(str(chunk_id).strip())
        if chunk and all(r.chunk_id != chunk.id for r in out):
            out.append(SourceRef.of(chunk))
    return out


def _chunks_of(refs: Sequence[SourceRef], index: DocumentIndex) -> list[Chunk]:
    return [c for c in (index.get(r.chunk_id) for r in refs) if c]


def _claim_of(item) -> str:
    if isinstance(item, SummaryItem):
        return item.text
    if isinstance(item, Flashcard):
        return item.back
    answer = "" if canonical_true_false(item.answer) else item.answer
    return f"{answer} {item.explanation}".strip()


def _attach_sources(item, claim: str, index: DocumentIndex, min_support: float) -> Optional[str]:
    """Make sure the item cites real passages; return a drop reason, or None when it is fine."""
    chunks = _chunks_of(item.sources, index)
    if not chunks:
        for chunk, _score in index.search(claim, k=2):
            if support_score(claim, [chunk]) >= min_support:
                item.sources = [SourceRef.of(chunk)]
                chunks = [chunk]
                break
    if not chunks:
        return "no-source"
    if support_score(claim, chunks) < min_support:
        # a language model may cite the wrong neighbour: look for any passage of the lecture that supports the claim
        for chunk, _score in index.search(claim, k=3):
            if support_score(claim, [chunk]) >= min_support:
                item.sources = [SourceRef.of(chunk)]
                return None
        return "unsupported"
    return None


def _check_question(q: Question, option_counts: tuple[int, ...] = (5,)) -> Optional[str]:
    if not q.question.strip() or not q.answer.strip():
        return "malformed"
    if q.type == "mcq":
        options = [o.strip() for o in q.options if o.strip()]
        if len(options) not in option_counts or len({o.casefold() for o in options}) != len(options) or q.answer not in options:
            return "malformed"
    elif q.type == "true_false":
        canonical = canonical_true_false(q.answer)
        if canonical is None:
            return "malformed"
        q.answer, pair = canonical[0], canonical[1]
        q.options = list(pair)
    elif q.type in ("clinical_vignette", "mechanism") and q.options:
        if len(q.options) < 3 or q.answer not in q.options or len({o.casefold() for o in q.options}) != len(q.options):
            return "malformed"
    return None


def _check_card(card: Flashcard) -> Optional[str]:
    if not card.front.strip() or not card.back.strip() or card.front.strip().casefold() == card.back.strip().casefold():
        return "malformed"
    return None


def ground_items(items: Sequence, index: DocumentIndex, *, strict: bool = True, min_support: float = MIN_SUPPORT,
                 check_text: bool = True, option_counts: tuple[int, ...] = (5,)) -> tuple[list, GroundingReport]:
    """Keep the items the lecture supports. ``check_text=False`` is for the built-in provider whose items are the lecture's own
    sentences (only the citations are verified)."""
    report = GroundingReport()
    kept = []
    for item in items:
        if getattr(item, "external", False) and strict:
            report.drop("external-knowledge")
            continue
        problem: Optional[str] = None
        if isinstance(item, Question):
            problem = _check_question(item, option_counts)
        elif isinstance(item, Flashcard):
            problem = _check_card(item)
        elif not item.text.strip():
            problem = "malformed"
        if problem is None and (strict or check_text):
            claim = _claim_of(item)
            if check_text:
                problem = _attach_sources(item, claim, index, min_support)
            elif not _chunks_of(item.sources, index):
                problem = "no-source"
        if problem is not None:
            report.drop(problem)
            continue
        if not item.topic and item.sources:
            first = index.get(item.sources[0].chunk_id)
            item.topic = first.topic if first else ""
        kept.append(item)
        report.kept += 1
    return kept, report


def confidence(item, index: DocumentIndex) -> float:
    """How strongly the lecture supports an item (shown in technical details; 1.0 for quoted sentences)."""
    return round(support_score(_claim_of(item), _chunks_of(item.sources, index)), 2)
