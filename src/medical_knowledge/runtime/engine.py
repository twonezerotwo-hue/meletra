"""The v4 engine: suspicious spans -> candidate retrieval -> scoring -> confidence gate -> corrections (not applied to anything by itself).

    engine = V4Engine(kb)
    result = engine.run(segment_views, subjects=[...], lecture_terms=[...], custom_terms=[...])
    result.corrections       list of V4Correction (HIGH tier) with the explainable breakdown
    result.uncertain         MEDIUM tier (advisory, never applied)
    result.texts             the segment texts WITH the HIGH corrections applied (hypothetical v4 transcript; production never uses it in shadow mode)

The raw transcript is immutable: the engine reads segment views and returns new strings.

``run`` = ``prepare`` (spans, retrieval, scoring - the expensive part) + ``finish`` (decisions with a gate configuration). Benchmarks call ``prepare`` once and
``finish`` for every candidate gate configuration, so settings can be compared without repeating the retrieval.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
import re
from typing import Callable, Iterable, Optional, Sequence

from ...medical.corrector import SegmentView
from ..normalize.forms import adapt_number, match_case, strip_qualifier, trim_neighbours
from ..normalize.text import compact
from .abbreviations import AbbreviationResolver
from .context import LectureContext
from .expected import ExpectedTerms
from .gate import HIGH, LOW, MEDIUM, Decision, GateConfig, alignment_problem, decide
from .knowledge import MedicalKnowledge
from .lecture import LectureVocabulary
from .policy import DOMAIN_CONFIDENCE_MIN
from .retrieve import CandidateRetriever
from .score import CandidateScorer, Scored
from .spans import Span, SpanDetector
from .validity import WordValidity

#: Latin / Greek adjective endings of anatomical and medical terms ("transversalis", "pterygoideus"): such a word that the vocabulary does not list is more often a real, rare term the
#: recogniser got right than a garble - replacing it needs the lecture's own evidence
LATIN_ADJECTIVE = re.compile(r"(?:alis|aris|ilis|arius|orius|ensis|icus|ivus|osus|eus|ideus|oideus|aceus|inus|anus)$")
LATIN_ADJECTIVE_MIN = 9
#: a word next to a HIGH correction that is still no word of any kind: the garble is probably larger than the corrected span
UNEXPLAINED_VALIDITY = 0.6
#: Derivational endings of medical nouns and adjectives (longest match wins). A recogniser garbles the middle of a word far more often than its ending, so a "correction" that turns a heard
#: noun ("pseudohypertrophy") into an adjective ("pseudohypertrophic") or the other way round writes a word the speaker probably did not say
NOUN_ENDINGS = ("trophy", "plasia", "pathy", "itis", "osis", "emia", "aemia", "uria", "algia", "ectomy", "ostomy", "otomy", "oma", "ism", "ity", "ion", "ia", "ium", "sis", "ty", "y")
ADJECTIVE_ENDINGS = ("ical", "ic", "ous", "ary", "ive", "oid", "al", "ar", "ac")
RETRIEVE_LIMIT = 50          # candidates retrieved per span (candidate recall@10 / @25 / @50 is reported)
SCORE_LIMIT = 25             # the best of them (by the pre-ranking) that are scored in full


@dataclass
class V4Correction:
    segment: int
    timestamp: float
    original: str
    replacement: str
    confidence: float
    tier: str
    concept_id: str
    domain: str
    start: int
    end: int
    signals: dict[str, float]
    alternatives: list[dict]
    reasons: list[str]
    sources: list[str]
    context: str = ""
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        h, rest = divmod(int(self.timestamp), 3600)
        m, s = divmod(rest, 60)
        return {"timestamp": f"{h:02d}:{m:02d}:{s:02d}", "segment": self.segment, "original": self.original, "replacement": self.replacement,
                "confidence": round(self.confidence, 3), "tier": self.tier, "concept_id": self.concept_id, "domain": self.domain, "sources": self.sources,
                "signals": {k: round(v, 3) for k, v in self.signals.items()}, "alternatives": self.alternatives, "reasons": self.reasons, "context": self.context,
                **self.details}


@dataclass
class V4Result:
    corrections: list[V4Correction]
    uncertain: list[V4Correction]
    texts: list[str]
    topic: dict[str, float]
    stats: dict = field(default_factory=dict)
    decisions: list[Decision] = field(default_factory=list)
    abbreviations: list[dict] = field(default_factory=list)
    hotwords: list[str] = field(default_factory=list)


@dataclass
class Prepared:
    """Everything up to the decisions: reusable with any gate configuration."""
    views: Sequence[SegmentView]
    ctx: LectureContext
    items: list[tuple[Span, list[Scored], list]]       # (span, scored candidates, pool of every retrieved candidate)
    vocab: LectureVocabulary
    started: float
    retrieval_seconds: float
    spans: int


def word_class(word: str) -> Optional[str]:
    """"noun" / "adjective" by the longest derivational ending the word has, None when it has none of the listed ones."""
    low = word.casefold()
    best: tuple[int, str] = (0, "")
    for kind, endings in (("noun", NOUN_ENDINGS), ("adjective", ADJECTIVE_ENDINGS)):
        for ending in endings:
            if low.endswith(ending) and len(low) - len(ending) >= 4 and len(ending) > best[0]:
                best = (len(ending), kind)
    return best[1] or None


def dropped_fragment(heard: str, shown: str) -> str:
    """The 1-2 letters at the BEGINNING of a heard word that the replacement leaves out ("omegaloblastic" -> "megaloblastic": a glued word, here the article "a"). "" when nothing is
    dropped - a glued article that the replacement writes out ("a normocytic") drops nothing. A lost final letter is an ordinary recogniser error and is not looked at."""
    h, w = compact(heard).casefold(), compact(shown).casefold()
    if w and 0 < len(h) - len(w) <= 2 and h.endswith(w):
        return h[:len(h) - len(w)]
    return ""


def apply_case(original: str, surface: str, text: str, start: int) -> str:
    """The replacement keeps the sentence-start capital of the span it replaces (terms like 'mTOR' / 'Nissl' keep their own spelling)."""
    if original[:1].isupper() and surface[:1].islower() and (start == 0 or text[:start].rstrip().endswith((".", "!", "?"))):
        return surface[:1].upper() + surface[1:]
    return surface


class V4Engine:
    def __init__(self, kb: MedicalKnowledge, gate: Optional[GateConfig] = None, language_ok: Optional[Callable[[SegmentView], bool]] = None) -> None:
        self.kb = kb
        self.gate = gate or GateConfig()
        self.retriever = CandidateRetriever(kb)
        self._validity: Optional[WordValidity] = None
        self._eponyms: Optional[set] = None

    def render(self, span: Span, surface: str, text: str, sentence_start: bool) -> tuple[str, str]:
        """(the words written for the span, number status). No dictionary qualifiers, the neighbours' words not repeated, the number and the case of what was heard;
        a number form that no source verifies is NOT invented (status "unverified")."""
        eponyms = self.eponyms()
        out = trim_neighbours(strip_qualifier(surface), text[:span.start], text[span.end:])
        out, status = adapt_number(span.text, out, eponyms, self.kb.has_form)
        heard = span.text.strip()
        if span.n == 1 and " " not in out:
            low = heard.casefold()
            for article in ("an", "a"):                                    # the recogniser glued the article to the term: "anormocytic" is "a normocytic", not "normocytic"
                if low.startswith(article) and len(low) - len(article) >= 6 and compact(low[len(article):]) == compact(out) and not self.kb.is_english_word(low, 80):
                    out = heard[:len(article)] + " " + out
                    break
        return match_case(span.text, out, lambda w: self.kb.is_english_word(w, 80), sentence_start, eponyms), status

    def eponyms(self) -> set:
        """Capitalised words of the vocabulary's eponyms (Golgi, Nissl, Schwann ...): they keep their capital letter in a replacement."""
        if self._eponyms is None:
            self._eponyms = {w for (text,) in self.kb.db.execute("SELECT DISTINCT t.text FROM term t JOIN concept c ON c.concept_id = t.concept_id WHERE c.kind = 'eponym'")
                             for w in text.replace("'s", "").split() if w[:1].isupper()}
        return self._eponyms

    # ------------------------------------------------------------------ the two phases
    def prepare(self, views: Sequence[SegmentView], subjects: Sequence[str] = (), lecture_terms: Iterable[str] = (), custom_terms: Iterable[str] = ()) -> Prepared:
        started = time.perf_counter()
        ctx = LectureContext(self.kb, views, subjects, lecture_terms, custom_terms)
        expected = ExpectedTerms(self.kb, ctx)
        detector = SpanDetector(self.kb, ctx, expected)
        spans = list(detector.detect())
        boost = set(ctx.lecture_concepts)                                  # the slides / the user's vocabulary: retrieved earlier, never replaced by that alone
        vocab = LectureVocabulary(ctx)
        scorer = CandidateScorer(self.kb, ctx)
        scorer.prime_repetition(spans)
        # a weak topic classification must not steer retrieval either
        domains = ctx.domains if ctx.domain_confidence >= DOMAIN_CONFIDENCE_MIN else []
        items: list[tuple[Span, list[Scored], list]] = []
        retrieval_seconds = 0.0
        for span in spans:
            t = time.perf_counter()
            cands = self.retriever.retrieve(span.text, domains, RETRIEVE_LIMIT, extra=span.expected, boost=boost or None)
            pool = [(c.surface, c.concept_id, round(c.prior, 4), c.channels) for c in cands]
            lecture = vocab.candidates(span.text, domains)                     # slide / own terms the knowledge base does not know
            retrieval_seconds += time.perf_counter() - t
            scored = [scorer.score(span, c) for c in cands[:SCORE_LIMIT] + lecture]
            items.append((span, scored, pool))
        return Prepared(views, ctx, items, vocab, started, retrieval_seconds, len(spans))

    def finish(self, prepared: Prepared, gate: Optional[GateConfig] = None, extras: bool = True) -> V4Result:
        """Decide every span with ``gate``. ``extras`` = also resolve the abbreviations and build the hotword list (skipped by parameter sweeps)."""
        gate = gate or self.gate
        ctx, views = prepared.ctx, prepared.views
        decisions: list[Decision] = []
        for span, scored, pool in prepared.items:
            d = decide(span, list(scored), gate)
            d.pool = pool
            decisions.append(d)
        holder: list = []

        def resolver() -> AbbreviationResolver:                           # built only when a candidate is an ambiguous abbreviation
            if not holder:
                holder.append(AbbreviationResolver(ctx))
            return holder[0]

        for d in decisions:
            if d.best is not None and d.tier in (HIGH, MEDIUM):
                self._post_checks(d, ctx, resolver)
        chosen = self._resolve(decisions, sum(len(s.tokens) for s in ctx.segments), gate)
        for d in chosen:
            if d.tier == HIGH:
                problem = self._unexplained_neighbour(d, ctx, chosen)
                if problem:
                    d.tier = MEDIUM
                    d.blocked_by.append(problem)
                    d.reason = (d.reason + "; " if d.reason and not d.reason.startswith("passed") else "") + problem
        corrections, uncertain = [], []
        for d in chosen:
            (corrections if d.tier == HIGH else uncertain).append(self._make(d, ctx))
        texts = self._apply([v.text for v in views], corrections)
        stats = {"segments": len(views), "words": sum(len(s.tokens) for s in ctx.segments), "suspicious_spans": prepared.spans, "high": len(corrections), "medium": len(uncertain),
                 "retrieval_seconds": round(prepared.retrieval_seconds, 3), "seconds": round(time.perf_counter() - prepared.started, 3),
                 "known_term_hits": sum(1 for s in ctx.segments for h in s.hits if not h.common), "domain_confidence": ctx.domain_confidence, "cache": self.kb.cache_stats()}
        return V4Result(corrections, uncertain, texts, ctx.topic, stats, decisions, self._abbreviations(ctx) if extras else [], prepared.vocab.hotwords(60) if extras else [])

    def run(self, views: Sequence[SegmentView], subjects: Sequence[str] = (), lecture_terms: Iterable[str] = (), custom_terms: Iterable[str] = ()) -> V4Result:
        return self.finish(self.prepare(views, subjects, lecture_terms, custom_terms))

    # ------------------------------------------------------------------ checks that need the sentence, the sources or the lecture
    def _post_checks(self, d: Decision, ctx: LectureContext, resolver: Callable[[], AbbreviationResolver]) -> None:
        def demote(reason: str, to: str = MEDIUM) -> None:
            if d.tier == HIGH or to == LOW:
                d.tier = to
            d.blocked_by.append(reason)
            d.reason = (d.reason + "; " if d.reason and not d.reason.startswith("passed") else "") + reason

        text = ctx.segments[d.span.segment].view.text
        shown, status = self.render(d.span, d.best.candidate.surface, text, False)
        if compact(shown) == compact(d.span.text) and len(shown.split()) == len(d.span.text.split()):                            # a "correction" that writes the very words that were heard is no correction
            d.tier = LOW
            d.blocked_by.append("same words as heard")
            d.reason = "same words as heard"
            return
        problem = alignment_problem(d.span, shown)                           # the guards judge the words that WOULD BE WRITTEN, not only the dictionary form
        if problem:
            d.tier = LOW
            d.blocked_by.append(problem + " (checked on the rendered replacement)")
            d.reason = problem
            return
        if status == "unverified":
            demote("the singular / plural (or Latin) form is not verified in the sources: the form is uncertain, so the original stays")
        if "(" in shown or ")" in shown or compact(shown) != compact(shown.strip()) or any(a.lower() == b.lower() for a, b in zip(shown.split(), shown.split()[1:])):
            demote("the resulting phrase is malformed", LOW)
        problem = self._abbreviation_problem(d, resolver)
        if problem:
            demote(problem)
        word = d.span.text.strip()
        if (d.tier == HIGH and d.span.n == 1 and len(word) >= LATIN_ADJECTIVE_MIN and LATIN_ADJECTIVE.search(word.casefold()) and not d.best.candidate.known_confusion
                and "lecture" not in d.assessment.families):
            demote(f"'{word}' has the shape of a Latin medical adjective that the vocabulary may simply lack, and the lecture does not name the replacement: the original stays")
        heard_words, shown_words = d.span.text.split(), shown.split()
        if d.tier == HIGH and len(heard_words) == len(shown_words):
            for heard_word, shown_word in zip(heard_words, shown_words):
                fragment = dropped_fragment(heard_word, shown_word)
                if fragment:
                    demote(f"the replacement drops '{fragment}' of the heard '{heard_word}' without explaining it (a glued word or a different word): the original stays")
                    break
        if d.tier == HIGH and len(heard_words) == len(shown_words):
            was, now = word_class(heard_words[-1]), word_class(shown_words[-1])
            if was and now and was != now and not d.best.candidate.known_confusion:
                demote(f"the heard '{d.span.text}' is {'an' if was[0] == 'a' else 'a'} {was} and the replacement '{shown}' is {'an' if now[0] == 'a' else 'a'} {now}: "
                       "the ending is rarely what the recogniser gets wrong, so the original stays")

    def validity(self) -> WordValidity:
        if self._validity is None:
            self._validity = WordValidity(self.kb)
        return self._validity

    def _unexplained_neighbour(self, d: Decision, ctx: LectureContext, chosen: list[Decision]) -> Optional[str]:
        """A HIGH correction of ONE word of a garble ("pontum dullari" -> "ponte dullari") leaves its neighbour unexplained: the replacement is then probably wrong or partial."""
        seg = ctx.segments[d.span.segment]
        toks, text = seg.tokens, seg.view.text
        covered = [(c.span.first, c.span.last) for c in chosen if c is not d and c.span.segment == d.span.segment]
        for idx, left in ((d.span.first - 1, True), (d.span.last, False)):
            if not 0 <= idx < len(toks) or any(a <= idx < b for a, b in covered):
                continue
            tok = toks[idx]
            a, b = (tok.end, toks[d.span.first].start) if left else (toks[d.span.last - 1].end, tok.start)
            if text[a:b].strip():                                             # punctuation between: a different phrase
                continue
            word = tok.low
            if len(word) < 4 or not word.isalpha() or self.kb.is_english_word(word, 80) or self.validity().of(word).value >= UNEXPLAINED_VALIDITY:
                continue
            return f"the neighbouring word '{tok.text}' is still no word and nothing explains it: the garble is probably larger than the corrected '{d.span.text}'"
        return None

    def _abbreviation_problem(self, d: Decision, resolver: Callable[[], AbbreviationResolver]) -> Optional[str]:
        c = d.best.candidate
        short = c.surface
        if not (c.term_type == "abbreviation" or (short.isupper() and 2 <= len(short) <= 6)):
            return None
        concepts = {r[0] for r in self.kb.db.execute("SELECT DISTINCT concept_id FROM abbreviation WHERE norm_short = ?", (compact(short),)) if r[0]}
        if len(concepts) < 2:
            return None
        res = resolver().resolve(short, d.span.segment)
        if res is None or res.ambiguous or res.chosen is None or (res.chosen.concept_id and res.chosen.concept_id != c.concept_id):
            return f"'{short}' is an ambiguous abbreviation ({len(concepts)} expansions) and the lecture does not settle which one"
        return None

    @staticmethod
    def _abbreviations(ctx: LectureContext) -> list[dict]:
        """Informational: every abbreviation of the lecture resolved in its context (a defined / ambiguous / unresolved verdict, never a replacement)."""
        seen: set[tuple[str, Optional[str]]] = set()
        out = []
        for r in AbbreviationResolver(ctx).resolve_all():
            key = (r.short, r.chosen.long_form if r.chosen else None)
            if key not in seen:
                seen.add(key)
                out.append(r.to_dict())
        return out

    # ------------------------------------------------------------------ overlaps, limits
    def _resolve(self, decisions: list[Decision], words: int, gate: GateConfig) -> list[Decision]:
        live = [d for d in decisions if d.tier in (HIGH, MEDIUM) and d.best is not None]
        live.sort(key=lambda d: (-(d.tier == HIGH), -d.assessment.adjusted, d.span.n))
        taken: dict[int, list[tuple[int, int]]] = {}
        chosen: list[Decision] = []
        per_segment: dict[int, int] = {}
        for d in live:
            spans = taken.setdefault(d.span.segment, [])
            if any(d.span.start < e and s < d.span.end for s, e in spans) or per_segment.get(d.span.segment, 0) >= gate.max_per_segment:
                continue
            spans.append((d.span.start, d.span.end))
            per_segment[d.span.segment] = per_segment.get(d.span.segment, 0) + 1
            chosen.append(d)
        high = [d for d in chosen if d.tier == HIGH]
        limit = max(3, int(gate.max_share * words))                      # a lecture where "everything" needs correcting is a miscalibrated engine
        if len(high) > limit:
            keep = {id(d) for d in sorted(high, key=lambda d: -d.assessment.adjusted)[:limit]}
            for d in high:
                if id(d) not in keep:
                    d.tier, d.blocked_by = MEDIUM, d.blocked_by + ["share of corrections above the safety limit"]
        return chosen

    def _make(self, d: Decision, ctx: LectureContext) -> V4Correction:
        best = d.best
        view = ctx.segments[d.span.segment].view
        sentence_start = d.span.start == 0 or view.text[:d.span.start].rstrip().endswith((".", "!", "?"))
        surface, form_status = self.render(d.span, best.candidate.surface, view.text, sentence_start)
        sources = [r[0] for r in self.kb.db.execute("SELECT DISTINCT ts.source_id FROM term_source ts WHERE ts.term_id = ?", (best.candidate.term_id,))] or ["lecture vocabulary"]
        lo, hi = max(0, d.span.start - 50), min(len(view.text), d.span.end + 50)
        a = d.assessment
        second = d.runner_up
        sig = best.signals
        details = {"keep_original_score": round(a.keep, 3), "correction_cost": round(a.cost, 3), "adjusted_score": round(a.adjusted, 3), "best_score": round(best.score, 3),
                   "second_best": second.candidate.surface if second else "", "second_best_score": round(second.score, 3) if second else None,
                   "score_margin": round(d.candidate_margin, 3), "margin_over_original": round(a.keep_margin, 3), "retrieval_source": list(best.candidate.channels),
                   "evidence_families": list(a.families), "decision_reason": d.reason, "number_form": form_status,
                   "phonetic_score": round(sig["phonetic"], 3), "lexical_score": round(sig["lexical"], 3),
                   "context_evidence": {"cue_words": round(sig.get("cue", 0.0), 3), "ontology_link": round(sig.get("ontology", 0.0), 3), "linked_terms": int(sig.get("link_count", 0)),
                                        "context": round(sig["context"], 3)},
                   "pdf_evidence": {"lecture_vocabulary": round(sig["lecture"], 3), "named_in_slides_or_own_terms": best.candidate.concept_id in ctx.lecture_concepts},
                   "domain_evidence": {"domain": round(sig["domain"], 3), "domain_confidence": round(sig.get("domain_confidence", 0.0), 3)},
                   "changed_word_validity": round(a.changed_validity, 3), "near_high": bool(d.near_high)}
        return V4Correction(d.span.segment, view.start, d.span.text, surface, a.adjusted, d.tier, best.candidate.concept_id,
                            (best.candidate.domains or ("",))[0], d.span.start, d.span.end, best.signals,
                            [r.to_dict() for r in d.ranked[1:4]], list(d.span.reasons) + best.notes + d.blocked_by, sources, view.text[lo:hi], details)

    @staticmethod
    def _apply(texts: list[str], corrections: list[V4Correction]) -> list[str]:
        out = list(texts)
        for seg in sorted({c.segment for c in corrections}):
            text = out[seg]
            for c in sorted((c for c in corrections if c.segment == seg), key=lambda c: -c.start):
                text = text[:c.start] + c.replacement + text[c.end:]
            out[seg] = text
        return out
