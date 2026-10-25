"""Expected terms: what THIS lecture makes likely, and which heard spans sound like one of it.

A recogniser often writes another real word for a medical term ("leptin" for "lectin", "meiosis" for "miosis", "interior olive" for "inferior olive"). Such spans cannot be found by asking
"is this a word?" - it is one. They are found by asking "does the lecture expect something that SOUNDS like this, and does the span itself have no support here?". The expected set is
built only from what the lecture and the knowledge base already say:

    the concepts the lecture mentions (exact term hits, the slides, the user's own vocabulary)
    + the concepts those are linked to by the ontology
    + the concepts whose cue words occur in the lecture (rare cue words weigh more)

A span is *elevated* (examined by the candidate scorer and the gate - never corrected by this stage) when its sound key equals, or is one edit from, the key of an expected term of another
spelling and the span is not itself expected. Ordinary valid words are NOT fuzzy-corrected: elevation only puts the expected candidate on the table; the keep-original candidate, the
margins and the independent-evidence rules still decide, and a real word needs far more evidence than a non-word.
"""

from __future__ import annotations

import math
from functools import lru_cache

from rapidfuzz.distance import Levenshtein

from ...medical.phonetics import skeleton
from ..indexes.phonetic import deletions, dm_key
from ..normalize.text import compact, normalize
from .context import LectureContext
from .knowledge import MedicalKnowledge
from ..domains import SUPPORT_DOMAINS
from .policy import DOMAIN_CONFIDENCE_MIN

SEEDS = 150
RELATED_PER_SEED = 25
CUE_CONCEPTS = 400
TERMS_PER_CONCEPT = 6
MIN_SPELLING = 0.45                       # the span and the expected term must also be spelled recognisably alike
MAX_PER_SPAN = 6


@lru_cache(maxsize=65536)
def _keys(comp: str) -> tuple[str, str]:
    return skeleton(comp), dm_key(comp)


class ExpectedTerms:
    def __init__(self, kb: MedicalKnowledge, ctx: LectureContext) -> None:
        self.kb, self.ctx = kb, ctx
        self.weights: dict[str, float] = {}                                  # concept -> how strongly the lecture expects it
        self.info: dict[int, tuple[str, str, str]] = {}                      # term id -> (concept, text, compact)
        self.by_sk: dict[str, list[int]] = {}
        self.by_dm: dict[str, list[int]] = {}
        self.by_dm_longer: dict[str, list[int]] = {}                         # key with one letter deleted -> terms whose key is one letter longer
        self.compacts: set[str] = set()
        self.external: set[str] = set()                                      # concepts the lecture makes likely by more than one mention of their own (see supported_elsewhere)
        # the lecture's field of study, when the topic classification can be trusted: its terms are "expected" too (but only against a span that DISAGREES with the field)
        self.core = set(ctx.focus) if ctx.domain_confidence >= DOMAIN_CONFIDENCE_MIN else set()
        self.broad = set(ctx.domains) if self.core else set()
        self._domain_cache: dict[str, dict[int, set[str]]] = {}
        self._build()

    # ------------------------------------------------------------------ the expected concepts
    def _lecture_ngrams(self) -> set[str]:
        grams: set[str] = set()
        for seg in self.ctx.segments:
            words = [normalize(t.text) for t in seg.tokens]
            for i, w in enumerate(words):
                if len(w) >= 4:
                    grams.add(w)
                if i + 1 < len(words) and len(w) + len(words[i + 1]) >= 7:
                    grams.add(w + " " + words[i + 1])
        return grams

    def _cue_concepts(self) -> dict[str, float]:
        db = self.kb.db
        grams = sorted(self._lecture_ngrams())
        n_concepts = max(1, self.kb.counts().get("concepts", 1))
        score: dict[str, float] = {}
        seen: dict[str, set[str]] = {}
        for start in range(0, len(grams), 700):
            chunk = grams[start:start + 700]
            marks = ",".join("?" * len(chunk))
            rows = db.execute(f"SELECT cue, concept_id FROM context_cue WHERE cue IN ({marks})", chunk).fetchall()
            for cue, cid in rows:
                if cid in seen and cue in seen[cid]:
                    continue
                seen.setdefault(cid, set()).add(cue)
        if not seen:
            return {}
        cues = sorted({c for cs in seen.values() for c in cs})
        df: dict[str, int] = {}
        for start in range(0, len(cues), 700):
            chunk = cues[start:start + 700]
            marks = ",".join("?" * len(chunk))
            df.update(dict(db.execute(f"SELECT cue, COUNT(*) FROM context_cue WHERE cue IN ({marks}) GROUP BY cue", chunk).fetchall()))
        for cid, cs in seen.items():
            score[cid] = sum(min(1.0, math.log((n_concepts + 1) / (df.get(c, 1) + 1)) / 6.0) for c in cs)
        return dict(sorted(score.items(), key=lambda kv: -kv[1])[:CUE_CONCEPTS])

    def _build(self) -> None:
        kb, ctx = self.kb, self.ctx
        w = self.weights
        for cid, n in ctx.counts.most_common(SEEDS):
            w[cid] = max(w.get(cid, 0.0), min(1.0, 0.4 + 0.15 * n))
        for cid in ctx.lecture_concepts:
            w[cid] = 1.0
        self.external.update(cid for cid, n in ctx.counts.items() if n >= 2)
        self.external.update(ctx.lecture_concepts)
        for cid in list(w)[:SEEDS]:
            for r in kb.related(cid, limit=RELATED_PER_SEED):
                if r.weight >= 0.3:
                    w[r.concept_id] = max(w.get(r.concept_id, 0.0), 0.4 * w[cid])
                    self.external.add(r.concept_id)
        for cid, s in self._cue_concepts().items():
            if s >= 0.5:
                w[cid] = max(w.get(cid, 0.0), min(1.0, 0.3 + 0.2 * s))
                self.external.add(cid)
        ids = list(w)
        for start in range(0, len(ids), 500):
            chunk = ids[start:start + 500]
            marks = ",".join("?" * len(chunk))
            rows = kb.db.execute(f"SELECT term_id, concept_id, text, compact_text, sk, dm FROM term WHERE concept_id IN ({marks}) AND language IN ('en', 'la') "
                                 "AND token_count <= 4 ORDER BY is_preferred DESC, token_count", chunk).fetchall()
            per: dict[str, int] = {}
            for tid, cid, text, comp, sk, dm in rows:
                per[cid] = per.get(cid, 0) + 1
                if per[cid] > TERMS_PER_CONCEPT or len(comp) < 4:
                    continue
                self.info[tid] = (cid, text, comp)
                self.compacts.add(comp)
                if len(sk) >= 3:
                    self.by_sk.setdefault(sk, []).append(tid)
                if len(dm) >= 3:
                    self.by_dm.setdefault(dm, []).append(tid)
                    if len(dm) >= 4:
                        for k in deletions(dm):
                            if len(k) == len(dm) - 1:
                                self.by_dm_longer.setdefault(k, []).append(tid)

    # ------------------------------------------------------------------ matching
    def is_expected(self, concept_ids) -> bool:
        return any(c in self.weights for c in concept_ids)

    def supported_elsewhere(self, concept_ids) -> bool:
        """The lecture makes these concepts likely by something other than the one mention under test: said again, from the slides / the user's vocabulary, linked to another mentioned
        concept, or cued by its words. A term said once with nothing around it that fits is exactly the valid-word mishearing this stage looks for ("ilium" in a lecture on the gut)."""
        return any(c in self.external for c in concept_ids)

    def disagrees_with_field(self, concept_ids) -> bool:
        """The concepts belong to NO field the lecture is about (and the lecture does not expect them): a known term that does not fit here."""
        if not self.core or self.is_expected(concept_ids):
            return False
        fields = self.kb.concept_domains_many(list(concept_ids))
        return not any(set(fields.get(c, ())) & self.broad for c in concept_ids)

    def _fits(self, fields) -> bool:
        """A concept fits the lecture's field when it belongs to a related field, or to none in particular (eponyms, Latin forms, abbreviations)."""
        fields = set(fields)
        return bool(fields & self.broad) or (bool(fields) and fields <= SUPPORT_DOMAINS)

    def domain_matches(self, comp: str) -> dict[int, set[str]]:
        """Terms of the lecture's FIELD that sound like ``comp`` (same skeleton or Double Metaphone key, or one edit away) and are spelled recognisably alike."""
        if not self.core or len(comp) < 5:
            return {}
        cached = self._domain_cache.get(comp)
        if cached is not None:
            return cached
        sk, dm = _keys(comp)
        db = self.kb.db
        tids: set[int] = set()
        if len(sk) >= 4:
            tids.update(r[0] for r in db.execute("SELECT term_id FROM term WHERE sk = ? AND token_count <= 3 AND language IN ('en', 'la') LIMIT 60", (sk,)))
        if len(dm) >= 4:
            keys = sorted(deletions(dm))
            marks = ",".join("?" * len(keys))
            tids.update(r[0] for r in db.execute(f"SELECT DISTINCT term_id FROM phonetic_del WHERE key IN ({marks}) LIMIT 300", keys))
        out: dict[int, set[str]] = {}
        if tids:
            ids = sorted(tids)
            marks = ",".join("?" * len(ids))
            rows = db.execute(f"SELECT term_id, concept_id, compact_text FROM term WHERE term_id IN ({marks}) AND token_count <= 3 AND language IN ('en', 'la')", ids).fetchall()
            scored = [(Levenshtein.normalized_similarity(comp, c), tid, cid) for tid, cid, c in rows if c != comp and len(c) >= 4]
            scored = [s for s in scored if s[0] >= MIN_SPELLING]
            fields = self.kb.concept_domains_many({cid for _s, _t, cid in scored})
            keep = [(s, tid) for s, tid, cid in scored if self._fits(fields.get(cid, ()))]
            keep.sort(reverse=True)
            out = {tid: {"expected"} for _s, tid in keep[:MAX_PER_SPAN]}
        if len(self._domain_cache) < 5000:
            self._domain_cache[comp] = out
        return out

    def matches(self, text: str, own=()) -> dict[int, set[str]]:
        """Expected terms (term id -> {"expected"}) that sound like ``text`` without being spelled the same; empty when ``text`` is itself an expected term.
        ``own``: the concepts of ``text`` when it is a known term whose only support is this very mention - they are not candidates for themselves."""
        comp = compact(text)
        if len(comp) < 4 or (comp in self.compacts and not own):
            return {}
        sk, dm = _keys(comp)
        found: set[int] = set()
        if len(sk) >= 3:
            found.update(self.by_sk.get(sk, ()))
        if len(dm) >= 3:
            found.update(self.by_dm.get(dm, ()))
            found.update(self.by_dm_longer.get(dm, ()))
            if len(dm) >= 4:
                for k in deletions(dm):
                    if len(k) == len(dm) - 1:
                        found.update(self.by_dm.get(k, ()))
                        found.update(self.by_dm_longer.get(k, ()))                  # one letter different (substitution): both keys share a deletion
        out = []
        for tid in found:
            cid, term_text, term_comp = self.info[tid]
            if term_comp == comp or cid in own:
                continue
            sim = Levenshtein.normalized_similarity(comp, term_comp)
            if sim >= MIN_SPELLING:
                out.append((sim * (0.5 + 0.5 * self.weights.get(cid, 0.3)), tid))
        out.sort(reverse=True)
        return {tid: {"expected"} for _, tid in out[:MAX_PER_SPAN]}
