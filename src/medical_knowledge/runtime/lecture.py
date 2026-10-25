"""Lecture vocabulary: what THIS lecture is about, mapped onto knowledge-base concepts.

Inputs: slide / PDF terms, the user's own vocabulary (Settings -> Medical Vocabulary), the detected subject, repeated terms of the transcript itself.
They do three things:

    boost       a concept the slides name is a far better candidate for an odd-sounding span than any of the millions of others (scorer signal ``lecture``)
    virtual     a slide / own term the knowledge base does not know is still a candidate, with trust LECTURE_SPECIFIC / USER_DEFINED (never removed, never enough
                on its own to correct an impossible-sounding word: sound and spelling gates still apply)
    hotwords    a SMALL ranked list (a few dozen terms - the decoder prompt holds ~223 tokens) of the lecture's rare, relevant terms for a later targeted
                re-decode; never the knowledge base
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

from ...common_words import is_common_word
from ...medical.phonetics import skeleton
from ..indexes.phonetic import dm_key
from ..normalize.text import compact, letters, normalize
from .context import LectureContext
from .retrieve import Candidate, _sim, sound_similarity
from ..indexes.phonetic import mp_key


@dataclass(frozen=True)
class VirtualTerm:
    text: str
    kind: str                      # "lecture" (slides / PDF) or "custom" (the user's vocabulary)

    @property
    def trust(self) -> str:
        return "USER_DEFINED" if self.kind == "custom" else "LECTURE_SPECIFIC"

    @property
    def concept_id(self) -> str:
        return "LECT:" + compact(self.text)


class LectureVocabulary:
    def __init__(self, ctx: LectureContext) -> None:
        self.ctx = ctx
        self.virtual = [VirtualTerm(t, k) for t, k in ctx.unmatched_lecture_terms if len(compact(t)) >= 4]
        self._keys = [(v, skeleton(v.text), dm_key(v.text), mp_key(v.text), letters(v.text)) for v in self.virtual]

    def candidates(self, span: str, domains: Sequence[str] = (), minimum: float = 0.55) -> list[Candidate]:
        """Slide / own terms that sound like the span (brute force: there are a few hundred of them at most)."""
        if not self._keys:
            return []
        sk, dm, mp, lt = skeleton(span), dm_key(span), mp_key(span), letters(span)
        out: list[Candidate] = []
        for v, vsk, vdm, vmp, vlt in self._keys:
            sound = 0.45 * _sim(sk, vsk) + 0.15 * _sim(mp, vmp) + 0.40 * _sim(dm, vdm)
            ortho = _sim(lt, vlt)
            if sound < minimum or compact(span) == compact(v.text):
                continue
            tokens = len(normalize(v.text).split())
            out.append(Candidate(v.concept_id, -1, v.text, v.text, "en", "lecture", v.trust, tuple(domains[:1]), tokens, ("lecture",), "term", sound, ortho, False, False,
                                 1.0, 0.55 * sound + 0.25 * ortho + 0.2))
        out.sort(key=lambda c: -c.prior)
        return out[:5]

    # ------------------------------------------------------------------ hotwords
    def hotwords(self, limit: int = 60, token_count: Optional[Callable[[str], int]] = None, token_budget: int = 200) -> list[str]:
        """The lecture's rare, relevant terms, most useful first; cut at ``limit`` terms and (with a tokenizer) at ``token_budget`` tokens.

        Order: the user's own terms, slide / PDF terms, terms the lecturer keeps saying, then ontology neighbours of those. Common English words are never
        hotwords (they would only distort the decoder); multi-word terms and rarer words come first within a group (they gain the most)."""
        ctx = self.ctx
        kb = ctx.kb
        ranked: list[tuple[float, str]] = []
        seen: set[str] = set()

        def add(text: str, weight: float) -> None:
            key = compact(text)
            if not key or key in seen or (len(normalize(text).split()) == 1 and (is_common_word(text.lower()) or len(key) < 4)):
                return
            seen.add(key)
            rarity = 1.0 / (1.0 + math.log1p(min(kb.token_frequency(normalize(text).split()[0]), 2000)) / 4.0)
            ranked.append((weight + 0.15 * rarity + 0.05 * min(2, len(normalize(text).split()) - 1), text))

        for v in self.virtual:
            add(v.text, 3.0 if v.kind == "custom" else 2.5)
        for cid, kind in ctx.lecture_concepts.items():
            concept = kb.get_concept(cid)
            if concept:
                add(concept.preferred_name if len(concept.preferred_name) >= 4 else concept.terms[0].text, 3.0 if kind == "custom" else 2.5)
        for cid, n in ctx.counts.most_common(80):
            if n >= 2:
                concept = kb.get_concept(cid)
                if concept:
                    add(concept.preferred_name, 1.0 + min(0.9, 0.15 * n))
        top = [cid for cid, _ in ctx.counts.most_common(15)] + list(ctx.lecture_concepts)
        for cid in top:
            for r in kb.related(cid, limit=6):
                add(r.preferred_name, 0.5 * r.weight)
        ranked.sort(key=lambda x: -x[0])
        out: list[str] = []
        used = 0
        for _, text in ranked:
            cost = token_count(text) + 1 if token_count else 0
            if len(out) >= limit or (token_count and used + cost > token_budget):
                break
            out.append(text)
            used += cost
        return out
