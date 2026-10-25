"""Suspicious-span detection: which stretches of the transcript are even allowed to be looked at for a correction.

A medical vocabulary is NOT run against every word. A span (1-4 words) is only examined when one of these holds:

    unknown       it contains a word that is neither an ordinary English word nor a known medical term ("nisal", "tanny", "exodendritic")
    confusion     the knowledge base lists the span as a known mishearing
    collision     (only in a medical context, spans of two or more words) the span SOUNDS exactly like a term of the lecture's fields although every
                  word is a real word ("sell body" ~ "cell body", "lumber puncture" ~ "lumbar puncture", "mitral glial" ~ "microglial")

Ordinary correct English around a medical term is never examined, so "spinal", "between", "signals", "formation", "central" cannot be "corrected" just because
a similar medical term exists. Spans inside an exact known term are protected as well (unless they are a recorded mishearing).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterator, Optional, Sequence

from rapidfuzz.distance import Levenshtein

from ...common_words import is_common_word
from ...medical.corrector import _CONTRACTION, _FUNCTION, _GREEK, _STOP, _UNIT, _boundary, _english_text, _plain
from ...medical.phonetics import skeleton
from ..indexes.phonetic import dm_key
from ..normalize.text import compact, normalize
from .context import MAX_SPAN, LectureContext, SegmentInfo, Token
from .identifiers import identifier_reason
from .knowledge import MedicalKnowledge
from .policy import DOMAIN_CONFIDENCE_MIN
from .validity import WordValidity

#: a collision is only believed in a medical context: at least this many known (non-common) terms in the window
COLLISION_MIN_TERMS = 2
MIN_COLLISION_KEY = 4
COMPOUND_SIMILARITY = 0.82
_IDENTIFIER = re.compile(r"[a-z][A-Z]|^[A-Z]{2,}[a-z]")                      # camelCase / acronym+word: an identifier or a product name
FOCUS_KEY_LIMIT = 250000


@dataclass
class Span:
    segment: int
    first: int                    # first token
    last: int                     # one past the last token
    start: int                    # character offsets in the segment text
    end: int
    text: str
    tokens: list[Token]
    reasons: list[str] = field(default_factory=list)
    common_only: bool = False
    unknown_tokens: int = 0
    min_prob: Optional[float] = None
    mean_prob: Optional[float] = None
    unknown_mask: Optional[list[bool]] = None     # per token: not a word of English, nor of the vocabulary (a mishearing candidate)
    words: Optional[list[str]] = None             # the span's words (hyphenated tokens split), lower case
    word_unknown: Optional[list[bool]] = None     # per word: the same test
    word_validity: Optional[list[float]] = None   # per word: graded realness 0..1 (validity.py): 0 = not a word, ~0.4 derivative / compound, ~0.95 ordinary English
    expected: Optional[dict] = None               # terms the lecture makes likely that sound like the span (term id -> channels): the 'expected' reason
    word_rare: Optional[list[bool]] = None        # per word: unknown at the common level, but a word of the large English dictionary (SCOWL 80): "glycerins", "fibrocytes"

    @property
    def n(self) -> int:
        return self.last - self.first


GARBLED_KNOWN_SHARE = 0.65
RARE_LEVEL = 80
ENGLISH_LEVEL = 70                                  # SCOWL level: about 110,000 ordinary words (docs/v4/RETRIEVAL.md)


def _protected(word: str) -> bool:
    low = word.lower()
    return low in _GREEK or bool(_UNIT.match(low)) or len(low) < 3 or bool(_CONTRACTION.match(low))


class FocusKeys:
    """Sound keys (skeleton, Double Metaphone) of the terms of the lecture's fields: lets ordinary-word spans be tested for a sound collision without a search."""

    def __init__(self, kb: MedicalKnowledge, domains: Sequence[str]) -> None:
        self.sk: set[int] = set()                      # hashes, not strings: a third of the memory (a false match is a 1 in 2^61 event and only elevates a span)
        self.dm: set[int] = set()
        if not domains:
            return
        marks = ",".join("?" * len(domains))
        rows = kb.db.execute("SELECT t.sk, t.dm FROM term t WHERE t.token_count <= 3 AND t.language IN ('en','la') AND length(t.sk) >= ? AND t.concept_id IN "
                             f"(SELECT concept_id FROM concept_domain WHERE domain IN ({marks})) LIMIT ?", (MIN_COLLISION_KEY, *domains, FOCUS_KEY_LIMIT))
        for sk, dm in rows:
            self.sk.add(hash(sk))
            self.dm.add(hash(dm))

    def collides(self, text: str) -> bool:
        sk = skeleton(text)
        return len(sk) >= MIN_COLLISION_KEY and (hash(sk) in self.sk or hash(dm_key(text)) in self.dm)


_FOCUS_CACHE: dict[tuple, FocusKeys] = {}                                   # a few entries at most: one per (database, set of fields)


def focus_keys(kb: MedicalKnowledge, domains: Sequence[str]) -> FocusKeys:
    key = (str(kb.path), kb.meta.get("build_date", ""), tuple(sorted(domains)))
    if key not in _FOCUS_CACHE:
        if len(_FOCUS_CACHE) >= 1:
            _FOCUS_CACHE.pop(next(iter(_FOCUS_CACHE)))
        _FOCUS_CACHE[key] = FocusKeys(kb, domains)
    return _FOCUS_CACHE[key]


def word_is_unknown(kb: MedicalKnowledge, word: str) -> bool:
    """A word that is neither ordinary English (SCOWL, Whisper's common words) nor part of the vocabulary: a candidate mishearing."""
    low = word.lower()
    return not (len(low) < 4 or is_common_word(low) or kb.is_english_word(low, ENGLISH_LEVEL) or kb.token_frequency(low) > 0)


def word_is_rare(kb: MedicalKnowledge, word: str) -> bool:
    """A rare but real dictionary word: not a mishearing candidate on sound alone."""
    return len(word) >= 4 and kb.is_english_word(word.lower(), RARE_LEVEL)


class SpanDetector:
    def __init__(self, kb: MedicalKnowledge, ctx: LectureContext, expected=None) -> None:
        self.kb, self.ctx = kb, ctx
        self.expected = expected                                           # runtime.expected.ExpectedTerms or None
        self.validity = WordValidity(kb)
        # a weak topic classification must not create "collisions": the field only steers detection when it is confident
        self.focus = focus_keys(kb, ctx.domains if ctx.domain_confidence >= DOMAIN_CONFIDENCE_MIN else [])
        self._confused: set[str] = set()
        self._checked: set[str] = set()

    # ------------------------------------------------------------------ known mishearings (batched per lecture)
    def _prime_confusions(self) -> None:
        keys: set[str] = set()
        for seg in self.ctx.segments:
            toks = seg.tokens
            for n in range(1, MAX_SPAN + 1):
                for s in range(len(toks) - n + 1):
                    keys.add(compact(" ".join(t.text for t in toks[s:s + n])))
        keys = {k for k in keys if len(k) >= 3} - self._checked
        keys_l = list(keys)
        for start in range(0, len(keys_l), 800):
            chunk = keys_l[start:start + 800]
            marks = ",".join("?" * len(chunk))
            self._confused.update(r[0] for r in self.kb.db.execute(f"SELECT DISTINCT norm_heard FROM confusion WHERE norm_heard IN ({marks})", chunk))
        self._checked |= keys

    def is_confusion(self, text: str) -> bool:
        return compact(text) in self._confused

    def _garbled(self, toks: list[Token]) -> bool:
        """A segment in another language or full of noise: fewer than ~65 % of its words are English or known medical words."""
        if len(toks) < 5:
            return False
        known = sum(1 for t in toks if t.low in _FUNCTION or is_common_word(t.low) or self.kb.is_english_word(t.low, ENGLISH_LEVEL) or self.kb.token_frequency(t.low) > 0)
        return known / len(toks) < GARBLED_KNOWN_SHARE

    # ------------------------------------------------------------------ contextual reasons
    def _expected_hits(self, seg: SegmentInfo, s: int, n: int, joined: str, inside: list[bool]) -> dict:
        """Terms the lecture expects that sound like this span. A span that IS a known term which the lecture supports is left alone; an unsupported known term is not."""
        if self.expected is None:
            return {}
        letters = sum(len(t.text) for t in seg.tokens[s:s + n])
        if letters < 4 or (n == 1 and letters < 5 and is_common_word(seg.tokens[s].low)):
            return {}
        found = {}
        if all(inside):
            hit = next((h for h in seg.hits if h.first == s and h.last == s + n), None)
            if hit is None or self.expected.supported_elsewhere(hit.concepts):
                return {}
            if self.expected.disagrees_with_field(hit.concepts) and not hit.common:
                found = self.expected.domain_matches(compact(joined))            # a known term that fits no field of this lecture, next to one that does
            found.update(self.expected.matches(joined, own=hit.concepts))        # a term said once with nothing around it that fits: a valid-word mishearing ("ilium" for "ileum")
            return found
        elif any(inside):
            return {}
        found.update(self.expected.matches(joined))
        if n == 1 and not is_common_word(seg.tokens[s].low) and (n > 1 or letters >= 5):
            found.update(self.expected.domain_matches(compact(joined)))          # a rare word that is no term here, next to one of the lecture's field
        return found

    def _compound_hit(self, joined: str) -> bool:
        """Several words written apart that read like ONE rare vocabulary word ("neural pore" ~ "neuropore", "ventral lateral" ~ "ventrolateral")."""
        comp = compact(joined)
        if len(comp) < 8 or not self.kb.has_token_index:
            return False
        pool = self.kb.long_tokens().get(comp[:3])
        if not pool:
            return False
        best = max((Levenshtein.normalized_similarity(comp, tok) for tok in pool if abs(len(tok) - len(comp)) <= 3 and tok != comp), default=0.0)
        return best >= COMPOUND_SIMILARITY

    # ------------------------------------------------------------------ detection
    def detect(self) -> Iterator[Span]:
        self._prime_confusions()
        for seg in self.ctx.segments:
            yield from self.detect_segment(seg)

    def detect_segment(self, seg: SegmentInfo) -> Iterator[Span]:
        toks, text = seg.tokens, seg.view.text
        language = (seg.view.language or "en").lower()
        if not toks or not language.startswith("en"):
            return                                                      # the recogniser tagged another language (the production helper `_english_text` also rejected lists of terms)
        if self._garbled(toks):
            return                                                      # a segment of foreign or garbled words: nothing in it can be judged
        covered = {j for h in seg.hits for j in range(h.first, h.last)}
        medical_window = sum(1 for j in range(max(0, seg.index - self.ctx.window), min(len(self.ctx.segments), seg.index + self.ctx.window + 1))
                             for h in self.ctx.segments[j].hits if not h.common) >= COLLISION_MIN_TERMS
        for n in range(MAX_SPAN, 0, -1):
            for s in range(len(toks) - n + 1):
                span = toks[s:s + n]
                if toks[s].low in _STOP or span[-1].low in _STOP or all(_protected(t.text) for t in span):
                    continue
                raw = [(t.text, t.start, t.end) for t in span]
                if _boundary(text, raw) or text[span[-1].end:span[-1].end + 1] == "-" or text[max(0, span[0].start - 1):span[0].start] == "-":
                    continue
                joined = text[span[0].start:span[-1].end]
                if any(identifier_reason(text, t.start, t.end) for t in span):
                    continue                                                  # BRCA1, CTranslate2, PyTorch, CNS: written on purpose, not a recogniser's spelling
                if re.search(r"[,;:()\[\]]", joined):
                    continue                                                  # a correction must not swallow punctuation ("GFAP, gliofibrillary ...")
                reasons: list[str] = []
                confusion = self.is_confusion(joined)
                if confusion:
                    reasons.append("confusion")
                inside = [(s + k) in covered for k in range(n)]
                expected_hits = self._expected_hits(seg, s, n, joined, inside)
                if all(inside) and not confusion and not expected_hits:
                    continue
                wl = [w for t in span for w in normalize(t.text).split()]
                wu = [word_is_unknown(self.kb, w) for w in wl]
                wr = [u and word_is_rare(self.kb, w) for u, w in zip(wu, wl)]
                wv = [self.validity.of(w).value for w in wl]
                mask = [(not inside[k] and any(word_is_unknown(self.kb, w) for w in normalize(t.text).split())) for k, t in enumerate(span)]
                unknown = sum(mask)
                if unknown:
                    reasons.append("unknown")
                common_only = all(is_common_word(t.low) or _plain(t.text) for t in span)
                if not reasons and medical_window and n >= 2 and not any(inside) and self.focus.collides(joined):
                    reasons.append("collision")
                if expected_hits:
                    reasons.append("expected")
                single_hits = all(h.last - h.first == 1 for h in seg.hits if h.first < s + n and s < h.last)       # the words are known only one by one: no whole term covers them
                if n >= 2 and single_hits and self._compound_hit(joined):
                    reasons.append("compound")
                if not reasons:
                    continue
                probs = [t.prob for t in span if t.prob is not None]
                yield Span(seg.index, s, s + n, span[0].start, span[-1].end, joined, span, reasons, common_only and not unknown, unknown,
                           min(probs) if probs else None, sum(probs) / len(probs) if probs else None, mask, wl, wu, wv, expected_hits or None, wr)
