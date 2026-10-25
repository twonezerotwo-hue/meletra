"""What the engine knows about ONE lecture before it looks at any doubtful word: which known terms occur where, which fields of study the lecture is about,
which concepts the slides / the user's vocabulary name, and which concepts the lecturer keeps using.

All lookups are batched indexed queries on the knowledge database (distinct n-grams of the transcript, a few hundred per query); nothing of the
vocabulary is loaded into memory except the concepts this lecture actually touches.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from ...common_words import is_common_word
from ...medical.corrector import _FUNCTION, _STOP, _TOKEN, SegmentView, _align
from ...medical.corrector import _SUBJECT_DOMAINS
from ..domains import SUPPORT_DOMAINS, expand
from ..normalize.text import compact, normalize
from .knowledge import MedicalKnowledge

MAX_SPAN = 4
CHUNK = 800


@dataclass
class Token:
    text: str
    start: int
    end: int
    prob: Optional[float] = None

    @property
    def low(self) -> str:
        return self.text.lower()


@dataclass
class ExactHit:
    first: int                      # first token index
    last: int                       # one past the last token
    concepts: tuple[str, ...]
    common: bool                    # a single ordinary English word ("cell", "body"): not evidence of anything on its own


@dataclass
class SegmentInfo:
    index: int
    view: SegmentView
    tokens: list[Token]
    hits: list[ExactHit] = field(default_factory=list)


class LectureContext:
    def __init__(self, kb: MedicalKnowledge, views: Sequence[SegmentView], subjects: Sequence[str] = (), lecture_terms: Iterable[str] = (),
                 custom_terms: Iterable[str] = (), window: int = 4) -> None:
        self.kb = kb
        self.window = window
        self.segments: list[SegmentInfo] = []
        for i, v in enumerate(views):
            toks = [Token(m.group(0), m.start(), m.end()) for m in _TOKEN.finditer(v.text)]
            for tok, (_, p) in zip(toks, _align([(t.text, t.start, t.end) for t in toks], v.probs)):
                tok.prob = p
            self.segments.append(SegmentInfo(i, v, toks))
        self.subjects = tuple(s.lower() for s in subjects)
        self.lecture_terms = [t for t in lecture_terms if t and t.strip()]
        self.custom_terms = [t for t in custom_terms if t and t.strip()]
        self.counts: Counter = Counter()                    # concept -> how many times the lecture says it
        self.topic: dict[str, float] = {}
        self.focus: list[str] = []
        self.lecture_concepts: dict[str, str] = {}          # concept -> "lecture" | "custom"  (slides / own vocabulary)
        self.unmatched_lecture_terms: list[tuple[str, str]] = []   # (term, kind) for slide / own terms the knowledge base does not know
        self._build()

    # ------------------------------------------------------------------ build
    def _ngrams(self) -> set[str]:
        grams: set[str] = set()
        for seg in self.segments:
            toks = seg.tokens
            for n in range(1, MAX_SPAN + 3):
                for s in range(len(toks) - n + 1):
                    if n == 1 and toks[s].low in _STOP:
                        continue
                    key = compact(" ".join(t.text for t in toks[s:s + n]))
                    if len(key) >= 2:
                        grams.add(key)
        return grams

    def _lookup_many(self, keys: Iterable[str]) -> dict[str, list[str]]:
        found: dict[str, list[str]] = defaultdict(list)
        keys = list(keys)
        for start in range(0, len(keys), CHUNK):
            chunk = keys[start:start + CHUNK]
            marks = ",".join("?" * len(chunk))
            for key, cid in self.kb.db.execute(f"SELECT compact_text, concept_id FROM term WHERE compact_text IN ({marks}) AND language IN ('en','la')", chunk):
                if cid not in found[key]:
                    found[key].append(cid)
        return found

    def _build(self) -> None:
        known = self._lookup_many(self._ngrams())
        for seg in self.segments:
            toks, taken = seg.tokens, set()
            for n in range(MAX_SPAN + 2, 0, -1):                                  # longest first, no overlap
                for s in range(len(toks) - n + 1):
                    if any(j in taken for j in range(s, s + n)) or (n == 1 and toks[s].low in _STOP):
                        continue
                    cids = known.get(compact(" ".join(t.text for t in toks[s:s + n])))
                    if cids:
                        common = n == 1 and (is_common_word(toks[s].low) or len(toks[s].low) < 4)
                        seg.hits.append(ExactHit(s, s + n, tuple(cids), common))
                        taken.update(range(s, s + n))
            seg.hits.sort(key=lambda h: h.first)
        every = {c for seg in self.segments for h in seg.hits for c in h.concepts}
        fields = self.kb.concept_domains_many(every)
        weight: Counter = Counter()
        for seg in self.segments:
            for h in seg.hits:
                if not h.common:
                    n = h.last - h.first
                    for cid in h.concepts[:1]:
                        self.counts[cid] += 1
                        for d in fields.get(cid, ()):
                            if d not in SUPPORT_DOMAINS:
                                weight[d] += 1.0 + 0.5 * (n - 1)
        for subject in self.subjects:
            for d in _SUBJECT_DOMAINS.get(subject, (subject,)):
                weight[d] += 2.0
        total = sum(weight.values())
        self.topic = {d: round(w / total, 4) for d, w in weight.most_common(8)} if total else {}
        self.focus = [d for d, w in self.topic.items() if w >= 0.06]
        # slides / own vocabulary -> concepts (terms the knowledge base does not know stay as lecture-specific virtual terms)
        for kind, terms in (("lecture", self.lecture_terms), ("custom", self.custom_terms)):
            hits = self._lookup_many({compact(t) for t in terms if len(compact(t)) >= 2})
            for t in terms:
                cids = hits.get(compact(t))
                if cids:
                    for cid in cids:
                        self.lecture_concepts.setdefault(cid, kind)
                else:
                    self.unmatched_lecture_terms.append((t, kind))

    # ------------------------------------------------------------------ queries
    @property
    def domains(self) -> list[str]:
        """Fields of study to focus retrieval on: the lecture's own plus their direct relatives."""
        return expand(self.focus, 1) if self.focus else []

    @property
    def domain_confidence(self) -> float:
        """0..1: how far the topic classification can be trusted. Needs BOTH many exact terms (evidence) and one field holding most of the mass (dominance); a lecture with a
        handful of terms, or one that is spread over many fields, gets a low value, and then the field of study must not steer any correction."""
        if not self.topic:
            return 0.0
        terms = sum(1 for seg in self.segments for h in seg.hits if not h.common)
        evidence = 1.0 - math.exp(-terms / 6.0)
        dominance = min(1.0, max(self.topic.values()) / 0.40)
        return round(evidence * dominance, 3)

    def window_concepts(self, index: int, exclude: Optional[tuple[int, int]] = None) -> set[str]:
        lo, hi = max(0, index - self.window), min(len(self.segments), index + self.window + 1)
        out: set[str] = set()
        for j in range(lo, hi):
            for h in self.segments[j].hits:
                if h.common:
                    continue
                if j == index and exclude and h.first < exclude[1] and exclude[0] < h.last:
                    continue
                out.update(h.concepts)
        return out

    def window_words(self, index: int) -> set[str]:
        lo, hi = max(0, index - self.window), min(len(self.segments), index + self.window + 1)
        words: set[str] = set()
        for j in range(lo, hi):
            words.update(normalize(t.text) for t in self.segments[j].tokens)
        return words

    def topic_weight(self, domains: Sequence[str]) -> float:
        """0..1: how much of the lecture's topic mass sits in these fields (the strongest field counts as 1)."""
        if not self.topic or not domains:
            return 0.4
        top = max(self.topic.values())
        best = max((self.topic.get(d, 0.0) for d in domains), default=0.0)
        if best == 0.0 and any(d in SUPPORT_DOMAINS for d in domains):
            return 0.5
        near = [d for d in expand(list(self.topic), 1)]
        if best == 0.0 and any(d in near for d in domains):
            return 0.3
        return min(1.0, best / top)
