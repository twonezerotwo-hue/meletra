"""Transparent candidate scoring.

Every candidate gets one number from separate, named signals - nothing is hidden inside a model - and the report keeps the breakdown:

    phonetic     how alike the span and the candidate SOUND (skeleton, phoneme string, Double Metaphone; a recorded mishearing counts as a sound match)
    lexical      how alike they are SPELLED
    context      neighbouring terms and cue words: cue words of the candidate near the span, ontology neighbours among the known terms around it,
                 fields shared with the terms around it
    domain       how much of the lecture's topic sits in the candidate's fields
    lecture      the slides / the user's vocabulary name the candidate, or the lecturer already said it elsewhere
    oddness      how unlike ordinary English the heard span is (an unknown word is a better suspect than a plain phrase)
    asr          how doubtful the recogniser was about the span (weak evidence on its own: ASR is often confidently wrong)
    repetition   the same odd span recurs (a systematic mishearing, not a slip)
    language     the candidate's language fits the text

The weights below are the first calibration (milestone 5); they are re-fitted only against a documented benchmark, never to a single lecture.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from ...medical.lexicon import stem
from ..normalize.text import normalize
from . import grammar
from .cache import LRU
from .context import LectureContext
from .knowledge import MedicalKnowledge
from .retrieve import Candidate
from .spans import Span

#: BASE evidence every candidate has (sums to 1) and BONUS evidence that only some have (lecture vocabulary, a doubtful ASR word, a recurring mishearing)
BASE = {"phonetic": 0.30, "lexical": 0.10, "context": 0.36, "domain": 0.10, "oddness": 0.14}
BONUS = {"lecture": 0.10, "asr": 0.04, "repetition": 0.04}
WEIGHTS = {**BASE, **BONUS}
RELATION_FACTOR = {"is_a": 1.0, "part_of": 1.0, "related_to": 0.8, "broader": 0.6}
TRUST_ADJUST = {"CURATED": 0.0, "ONTOLOGY_VERIFIED": 0.0, "LITERATURE_VERIFIED": 0.03, "USER_DEFINED": -0.05, "LECTURE_SPECIFIC": -0.05, "UNVERIFIED_CANDIDATE": 0.10}


@dataclass
class Scored:
    candidate: Candidate
    score: float
    signals: dict[str, float]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"candidate": self.candidate.preferred_name, "surface": self.candidate.surface, "concept_id": self.candidate.concept_id, "score": round(self.score, 3),
                "signals": {k: round(v, 3) for k, v in self.signals.items()}, "channels": list(self.candidate.channels), "domains": list(self.candidate.domains),
                "trust": self.candidate.trust, "notes": self.notes}


class CandidateScorer:
    def __init__(self, kb: MedicalKnowledge, ctx: LectureContext) -> None:
        self.kb, self.ctx = kb, ctx
        self._cues: LRU = LRU(4096)
        self._neigh: LRU = LRU(4096)
        self._stems: LRU = LRU(512)
        self._cue_df: LRU = LRU(8192)
        self._n_concepts = max(1, kb.counts().get("concepts", 1))
        self._repeat: dict[str, int] = {}
        self._domain_conf = ctx.domain_confidence
        self._span_features: dict[tuple[int, int, int], dict[str, float]] = {}

    # ------------------------------------------------------------------ helpers
    def prime_repetition(self, spans: list[Span]) -> None:
        from ..normalize.text import compact

        counts: dict[str, int] = {}
        for s in spans:
            counts[compact(s.text)] = counts.get(compact(s.text), 0) + 1
        self._repeat = counts

    def _window_stems(self, segment: int) -> set[str]:
        return self._stems.get(segment, lambda: {stem(w) for w in self.ctx.window_words(segment)})

    def _concept_cues(self, concept_id: str) -> list[tuple[str, float]]:
        def load() -> list[tuple[str, float]]:
            out = []
            for (cue,) in self.kb.db.execute("SELECT cue FROM context_cue WHERE concept_id = ? LIMIT 24", (concept_id,)):
                df = self._cue_df.get(cue, lambda cue=cue: self.kb.db.execute("SELECT COUNT(*) FROM context_cue WHERE cue = ?", (cue,)).fetchone()[0])
                out.append((cue, min(1.0, math.log((self._n_concepts + 1) / (df + 1)) / 6.0)))
            return out
        return self._cues.get(concept_id, load)

    def _neighbours(self, concept_id: str) -> dict[str, float]:
        def load() -> dict[str, float]:
            return {r.concept_id: r.weight * RELATION_FACTOR.get(r.relation, 0.6) for r in self.kb.related(concept_id, limit=200)}
        return self._neigh.get(concept_id, load)

    def span_features(self, span: Span) -> dict[str, float]:
        """Candidate-independent facts about the heard span (computed once per span): how sure the recogniser was, how frequent the heard words are in the vocabulary,
        whether it is a capitalised name in the middle of a sentence."""
        key = (span.segment, span.start, span.end)
        found = self._span_features.get(key)
        if found is None:
            text = self.ctx.segments[span.segment].view.text
            sentence_start = span.start == 0 or text[:span.start].rstrip().endswith((".", "!", "?"))
            freq = max((self.kb.token_frequency(w) for w in (span.words or [])), default=0)
            found = {"asr_sure": 0.4 if span.mean_prob is None else max(0.0, min(1.0, (span.mean_prob - 0.55) / 0.4)),
                     "orig_freq": min(1.0, math.log1p(freq) / math.log1p(2000)),
                     "capitalised": 1.0 if (span.text[:1].isupper() and not sentence_start and not span.text.isupper()) else 0.0}
            self._span_features[key] = found
        return found

    # ------------------------------------------------------------------ signals
    def cue_signal(self, cand: Candidate, span: Span) -> float:
        stems = self._window_stems(span.segment)
        own = {stem(w) for w in normalize(span.text).split()}
        matched = 0.0
        for cue, weight in self._concept_cues(cand.concept_id):
            cue_stems = {stem(w) for w in cue.split()}
            if cue_stems and cue_stems <= stems and not cue_stems <= own:
                matched += weight
        return 1.0 - math.exp(-1.1 * matched)

    def ontology_signal(self, cand: Candidate, span: Span) -> tuple[float, float, int]:
        """(strength of the strongest ontology link to a known term around the span, share of those terms that share a field with the candidate,
        number of DISTINCT known terms around the span that the candidate is linked to)."""
        around = self.ctx.window_concepts(span.segment, (span.first, span.last))
        around.discard(cand.concept_id)
        if not around:
            return 0.0, 0.0, 0
        neigh = self._neighbours(cand.concept_id)
        link = max((neigh.get(c, 0.0) for c in around), default=0.0)
        linked = sum(1 for c in around if neigh.get(c, 0.0) >= 0.3)
        fields = self.kb.concept_domains_many(list(around)[:60])
        mine = {d for d in cand.domains if d not in ("abbreviations", "eponyms", "latin")}
        shared = sum(1 for c in list(around)[:60] if mine.intersection(fields.get(c, ())))
        return link, shared / max(1, min(len(around), 60)), linked

    def score(self, span: Span, cand: Candidate) -> Scored:
        ctx = self.ctx
        phon = max(cand.sound, 0.88 if cand.known_confusion else 0.0)
        cue = self.cue_signal(cand, span)
        link, coherence, linked = self.ontology_signal(cand, span)
        context = 1.0 - (1.0 - 0.85 * cue) * (1.0 - 0.9 * link) * (1.0 - 0.45 * coherence)
        raw_domain = ctx.topic_weight([d for d in cand.domains]) if cand.domains else 0.4
        domain = 0.4 + (raw_domain - 0.4) * self._domain_conf                                  # a weak topic classification must not steer corrections
        lecture = 0.0
        if cand.concept_id in ctx.lecture_concepts:
            lecture = 1.0 if ctx.lecture_concepts[cand.concept_id] == "custom" else 0.95
        else:
            lecture = min(0.8, 0.25 * math.log2(1 + ctx.counts.get(cand.concept_id, 0)))
        words = max(1, span.n)
        oddness = min(1.0, span.unknown_tokens / words + (0.15 if "confusion" in span.reasons else 0.0) + (0.1 if "unknown" in span.reasons and span.n > 1 else 0.0))
        asr = 0.5 if span.mean_prob is None else max(0.0, min(1.0, (0.92 - span.mean_prob) / 0.6))
        from ..normalize.text import compact

        rep = self._repeat.get(compact(span.text), 1)
        repetition = min(1.0, (rep - 1) / 3.0) if oddness >= 0.5 else 0.0
        language = 1.0 if cand.language in ("en", "la") else 0.2
        signals = {"phonetic": phon, "lexical": cand.ortho, "context": context, "domain": domain, "lecture": lecture, "oddness": oddness, "asr": asr,
                   "repetition": repetition, "language": language, "cue": cue, "ontology": link, "link_count": float(linked), "coherence": coherence,
                   "domain_confidence": self._domain_conf, **self.span_features(span)}
        text = ctx.segments[span.segment].view.text
        g = grammar.assess(text[:span.start], span.text, text[span.end:], cand.surface, cand.kind, self.kb.semantic_type(cand.concept_id))
        signals["pos"], signals["type_fit"] = g["pos"], g["type_fit"]
        score = (sum(BASE[k] * signals[k] for k in BASE) + sum(BONUS[k] * signals[k] for k in BONUS)) * (0.5 + 0.5 * language)
        seg_hits = ctx.segments[span.segment].hits
        siblings = any(cand.concept_id in h.concepts and not (h.first < span.last and span.first < h.last) for h in seg_hits if not h.common)
        signals["sibling"] = 1.0 if siblings else 0.0
        if siblings and not cand.known_confusion:
            score *= 0.85                                                          # listed alongside: "telencephalon and [diencephalon]"
        notes = []
        if cand.known_confusion:
            notes.append("recorded mishearing of this concept")
        if link > 0:
            notes.append("ontology link to a known term nearby")
        return Scored(cand, max(0.0, min(1.0, score)), signals, notes)
