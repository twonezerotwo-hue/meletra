"""The built-in study provider: rule based and offline. It needs no model, no download and no account.

It does not write new sentences. Summaries are the lecture's own sentences (tidied, in lecture order); flashcards and questions are
built from those sentences with fixed templates (term <-> definition, cause -> effect, fill-in-the-blank, true/false by swapping an
opposite word ...). Every item therefore carries the passage it comes from, and nothing is stated that the lecture did not say.
That is the SOURCE-BOUND default. A language model (``llm.py``) writes more fluent text, but needs setting up.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, Sequence

from . import text as T
from .index import DocumentIndex
from .models import (CARD_TYPES, QUESTION_TYPES, SOURCE_PDF, SOURCE_TRANSCRIPT, Chunk, Flashcard, Question, SourceRef, SummaryItem,
                     item_id)
from .i18n import strings
from .providers import NOT_IN_SOURCES, AIProvider, Evaluation, StudyContext

# --------------------------------------------------------------------------------------------------------------- text patterns
_LEAD_FILLER = re.compile(r"^(?:(?:so|okay|ok|well|now|and|but|alright|right|yeah|yes|then|um+|uh+|like|actually|basically)\b[,\s]*)+",
                          re.IGNORECASE)
_FILLER_WORD = re.compile(r"\b(?:um+|uh+|erm|hmm+)\b[,.]?\s*", re.IGNORECASE)
_PRONOUN_START = re.compile(r"^(?:it|this|that|these|those|there|here|we|they|you|i|he|she|which|what|who|when|where|how|why|if|"
                            r"bu|şu|o|bunlar|şunlar|onlar|biz|siz|ben|burada|şimdi)\b",
                            re.IGNORECASE)
_INTERROGATIVE = re.compile(r"^(?:why|what|how|where|when|who|which|is|are|do|does|did|can|could|should|"
                            r"neden|niçin|nasıl|nerede|ne|hangi|kim|kaç)\b", re.IGNORECASE)
_LEAD_ADVERB = re.compile(r"^(?:not surprisingly|importantly|therefore|thus|however|in this case|as a result|consequently|also|again|"
                          r"in addition|moreover|furthermore|finally|first|second)\b[,\s]*", re.IGNORECASE)
_STRUCTURE = re.compile(r"\b(?:structure of|composed of|consists? of|made of|formed by|subunits?|domains?|helix|binding site|active site)\b",
                        re.IGNORECASE)
_META_TALK = re.compile(r"^(?:i(?:'ll| will| would| think| believe| hope| mentioned| said| am going| want| do not| don't| can| have)|"
                        r"we will|we are going|let's|"
                        r"you have studied|you can see|you can find|you will see|as i said|as you can see|here we)\b", re.IGNORECASE)
_DEFINITION = re.compile(
    r"^(?P<term>[A-Za-z][\w\-/+'() ]{1,60}?)\s*,?\s+(?P<verb>is defined as|are defined as|refers to|is called|are called|is known as|"
    r"are known as|stands for|is a|is an|is the|are the|are a|denotes|means)\s+(?P<rest>.{12,})$", re.IGNORECASE)
_WEAK_VERBS = {"is a", "is an", "is the", "are the", "are a"}
_CAUSE_SPLIT = re.compile(r"\b(?P<verb>leads? to|results? in|resulting in|causes?|causing|triggers?|which causes|gives rise to|"
                          r"neden olur|neden olmaktadır|yol açar|yol açmaktadır|sonucunu doğurur)\b", re.IGNORECASE)
_BECAUSE = re.compile(r"\b(?P<verb>because|due to|as a result of|owing to|çünkü|nedeniyle|dolayı)\b", re.IGNORECASE)
_DENIR = re.compile(r"^(?P<rest>.{15,}?)\s+(?P<term>[\w\-]{4,40})\s+denir$", re.IGNORECASE)
_ACRONYM = re.compile(r"\b(?:[A-Z]{2,}[A-Za-z0-9+-]*|[a-z]{1,3}[A-Z][A-Za-z0-9]*|[A-Z][a-z]?\d[A-Za-z0-9]*)\b")
_MEDICAL_WORD = re.compile(
    r"\b[A-Za-z][A-Za-z-]{3,}(?:ase|osis|itis|emia|aemia|pathy|plasia|trophy|genesis|lysis|oma|iasis|uria|penia|cytosis|stasis|"
    r"mycin|cillin|statin|olol|pril|sartan|azole|mab|nib)s?\b", re.IGNORECASE)
_NOT_MEDICAL = frozenset("increase decrease disease release purchase database diseases increases decreases releases showcase".split())
_GENERIC = frozenset("lecture slide slides today thing things part parts type types kind case example examples point points time "
                     "way ways question questions answer number process level levels amount form side".split())
_ANTONYMS = (("increases", "decreases"), ("increase", "decrease"), ("increased", "decreased"), ("increasing", "decreasing"),
             ("activates", "inhibits"), ("activate", "inhibit"), ("activated", "inhibited"), ("activation", "inhibition"),
             ("stimulates", "inhibits"), ("promotes", "inhibits"), ("upregulates", "downregulates"), ("upregulated", "downregulated"),
             ("higher", "lower"), ("high", "low"), ("more", "less"), ("positive", "negative"), ("faster", "slower"),
             ("anabolic", "catabolic"), ("gain", "loss"), ("synthesis", "degradation"), ("before", "after"), ("enhances", "reduces"),
             ("elevated", "reduced"), ("excess", "deficiency"), ("deficiency", "excess"))
_ENZYME_WORD = re.compile(r"\b[A-Za-z][A-Za-z-]{3,}ases?\b")
_SENTENCE_END = (".", "!", "?", "…")


def tidy(sentence: str) -> str:
    """Drop spoken fillers at the start / inside ("So, um, ..."), start with a capital letter, end with a full stop."""
    s = T.clean(sentence)
    s = _FILLER_WORD.sub("", s)
    s = _LEAD_FILLER.sub("", s).strip(" ,;-–")
    if not s:
        return ""
    s = s[0].upper() + s[1:]
    return s if s.endswith(_SENTENCE_END) else s + "."


def _term_pattern(term: str) -> re.Pattern:
    return re.compile(r"(?<![\w-])" + re.escape(term) + r"(?:s|es)?(?![\w-])", re.IGNORECASE)


# ------------------------------------------------------------------------------------------------------------------- analysis
@dataclass
class Unit:
    pos: int
    text: str
    chunk: Chunk
    stems: frozenset
    n_words: int
    support: float = 0.0                         # how much of the sentence the linked slides repeat (0..1)
    cues: dict[str, int] = field(default_factory=dict)
    terms: list[str] = field(default_factory=list)
    meta: float = 0.0                            # 1.0 when the lecturer talks about the lecture ("I will not go into detail ...")

    @property
    def topic(self) -> str:
        return self.chunk.topic


@dataclass
class Term:
    key: str
    display: str
    kind: str                                    # enzyme | acronym | medical | phrase
    count: int = 0
    topics: int = 0
    score: float = 0.0
    units: list[int] = field(default_factory=list)


@dataclass
class Fact:
    kind: str                                    # definition | cause_effect | mechanism | clinical | comparison | structure | pathway | ...
    unit: Unit
    subject: str = ""
    left: str = ""
    right: str = ""
    weight: float = 0.0
    via: str = ""                                # cause_effect: "because" when the sentence reads "<effect> because <cause>"


def _pdf_units(chunk: Chunk) -> list[str]:
    """Slide text -> sentences: bullets are units; a line that continues the previous one (no end mark, lower-case start) is joined."""
    out: list[str] = []
    lines = [ln.lstrip("•-–·*▪●○ \t").strip() for ln in chunk.text.splitlines()]
    for line in (ln for ln in lines if ln):
        if out and not out[-1].endswith((".", ":", ";", "!", "?")) and line[:1].islower():
            out[-1] += " " + line
        else:
            out.append(line)
    if len(out) > 1 and chunk.title and out[0] == chunk.title:
        out = out[1:]                            # the slide title is a heading, not a statement
    result: list[str] = []
    for item in out:
        result.extend(T.sentences(item) if len(T.words(item)) > 30 else ([item] if len(T.words(item)) >= 3 else []))
    return result


def _cue_profile(text: str) -> dict[str, int]:
    return {
        "emphasis": T.cue_count(text, T.EMPHASIS_CUES), "exam": T.cue_count(text, T.EXAM_CUES),
        "mechanism": T.cue_count(text, T.MECHANISM_CUES), "cause": T.cue_count(text, T.CAUSE_EFFECT_CUES),
        "clinical": T.cue_count(text, T.CLINICAL_CUES), "comparison": T.cue_count(text, T.COMPARISON_CUES),
        "structure": T.cue_count(text, T.STRUCTURE_CUES), "pathway": T.cue_count(text, T.PATHWAY_CUES),
        "drug": T.cue_count(text, T.DRUG_CUES), "gene": T.cue_count(text, T.GENE_PROTEIN_CUES),
    }


class Analysis:
    """The whole lecture read once: sentence units, key terms and facts (cached in ``StudyContext.cache``)."""

    def __init__(self, index: DocumentIndex, language: str = "en") -> None:
        self.index = index
        self.language = language
        self.s = strings(language)                       # the fixed wording around the lecture's sentences
        self.units: list[Unit] = []
        self.by_chunk: dict[str, list[Unit]] = {}
        self._build_units()
        self.terms: dict[str, Term] = {}
        self._build_terms()
        self.facts: dict[str, list[Fact]] = {}
        self._build_facts()

    # ---- units
    def _build_units(self) -> None:
        slide_stems: dict[str, frozenset] = {c.id: frozenset(T.stem(w) for w in T.content_words(c.text)) for c in self.index.of_source(SOURCE_PDF)}
        for chunk in self.index.chunks:
            raw = _pdf_units(chunk) if chunk.source == SOURCE_PDF else T.sentences(chunk.text)
            linked = [slide_stems[s.id] for s in self.index.linked_slides(chunk) if s.id in slide_stems]
            made: list[Unit] = []
            for sentence in raw:
                text = tidy(sentence)
                stems = frozenset(T.stem(w) for w in T.content_words(text))
                n = len(T.words(text))
                if n < 4 or len(stems) < 2 or text.endswith("?") or _INTERROGATIVE.match(text):
                    continue                                                  # questions the lecturer asked aloud are not statements
                support = max((T.overlap(stems, ls) for ls in linked), default=0.0)
                made.append(Unit(pos=len(self.units) + len(made), text=text, chunk=chunk, stems=stems, n_words=n, support=support,
                                 cues=_cue_profile(text), meta=1.0 if _META_TALK.match(text) else 0.0))
            self.units.extend(made)
            self.by_chunk[chunk.id] = made

    # ---- terms
    def _candidates(self) -> Counter:
        found: Counter = Counter()
        display: dict[str, Counter] = {}
        kinds: dict[str, str] = {}

        def note(raw: str, kind: str) -> None:
            key = raw.casefold().rstrip("s") if raw.casefold().endswith("ases") else raw.casefold()
            if len(key) < 3 or key in T.STOPWORDS or key in _GENERIC or "'" in key or chr(0x2019) in key:
                return
            found[key] += 1
            display.setdefault(key, Counter())[raw] += 1
            kinds.setdefault(key, kind)

        for unit in self.units:
            body = unit.text
            for m in _ACRONYM.finditer(body):
                if m.start() == 0 and not (sum(c.isupper() for c in m.group()) >= 2 or any(c.isdigit() for c in m.group())):
                    continue
                note(m.group(), "acronym")
            for m in _MEDICAL_WORD.finditer(body):
                w = m.group()
                if w.casefold() in _NOT_MEDICAL:
                    continue
                note(w, "enzyme" if w.casefold().rstrip("s").endswith("ase") else "medical")
            tokens = [w for w in T.words(body)]
            for size in (3, 2):
                for i in range(len(tokens) - size + 1):
                    gram = tokens[i:i + size]
                    if all(len(w) >= 3 and w not in T.STOPWORDS and w not in _GENERIC and not w[0].isdigit() for w in gram):
                        note(" ".join(gram), "phrase")
        self._display = {k: c.most_common(1)[0][0] for k, c in display.items()}
        self._kinds = kinds
        return found

    def _build_terms(self) -> None:
        found = self._candidates()
        slide_text = " ".join(c.text for c in self.index.of_source(SOURCE_PDF)).casefold()
        titles = " ".join(c.title for c in self.index.chunks if c.title).casefold()
        glossary = {g.casefold() for g in getattr(self.index, "glossary", [])}
        total_topics = max(len(self.index.topics), 1)
        topic_of = [u.topic for u in self.units]
        min_count = 2 if len(self.units) > 30 else 1
        scored: list[Term] = []
        for key, count in found.items():
            kind = self._kinds[key]
            floor = min_count if kind != "phrase" else max(min_count, 3 if len(self.units) > 60 else 2)
            if count < floor:
                continue
            pattern = _term_pattern(key)
            units = [u.pos for u in self.units if pattern.search(u.text)]
            if not units:
                continue
            topics = len({topic_of[p] for p in units})
            score = math.log(1 + len(units)) * (1 + 0.5 * topics / total_topics)
            score *= {"enzyme": 1.5, "acronym": 1.3, "medical": 1.4, "phrase": 1.0}[kind]
            if key in slide_text:
                score *= 1.5
            if key in titles:
                score *= 1.5
            if key in glossary:
                score *= 1.4
            if kind == "phrase" and len(key.split()) == 3:
                score *= 1.2
            scored.append(Term(key, self._display[key], kind, len(units), topics, score, units))
        scored.sort(key=lambda t: t.score, reverse=True)
        kept: list[Term] = []
        for term in scored:                      # a phrase that is only part of a stronger term ("fatty acid" in "fatty acid synthesis") goes
            if any(term.key != other.key and term.key in other.key and other.count >= term.count * 0.7 for other in kept):
                continue
            kept.append(term)
            if len(kept) >= max(12, min(60, len(self.units) // 4)):
                break
        self.terms = {t.key: t for t in kept}
        for term in kept:
            for pos in term.units:
                self.units[pos].terms.append(term.key)

    def term_display(self, key: str) -> str:
        return self.terms[key].display if key in self.terms else key

    # ---- facts
    def _build_facts(self) -> None:
        facts: dict[str, list[Fact]] = {k: [] for k in
                                       ("definition", "cause_effect", "mechanism", "clinical", "comparison", "structure", "pathway",
                                        "enzyme", "drug", "gene", "emphasis", "concept")}
        for unit in self.units:
            text, cues = unit.text, unit.cues
            m = _DEFINITION.match(text.rstrip(".!"))
            if m and not _PRONOUN_START.match(m.group("term")) and len(T.words(m.group("term"))) <= 5:
                verb, term = m.group("verb").casefold(), m.group("term").strip(" ,")
                low = term.casefold()
                keyed = any(k in low or low in k for k in unit.terms)
                if verb not in _WEAK_VERBS or keyed:
                    facts["definition"].append(Fact("definition", unit, term, term, m.group("rest").strip(), 2.0 + (0.5 if keyed else 0)))
            denir = _DENIR.match(text.rstrip(".!"))
            if denir and not _PRONOUN_START.match(denir.group("term")):          # Turkish: "... glikoliz denir."
                term = denir.group("term")
                facts["definition"].append(Fact("definition", unit, term, term, denir.group("rest").strip(), 2.0 - unit.meta))
            split = _CAUSE_SPLIT.search(text)
            if split:
                left, right = text[:split.start()].strip(" ,;"), text[split.end():].strip(" ,;.")
                left = _LEAD_ADVERB.sub("", _LEAD_FILLER.sub("", left)).strip()
                if 2 <= len(T.words(left)) <= 16 and 2 <= len(T.words(right)) <= 18 and not _PRONOUN_START.match(left):
                    facts["cause_effect"].append(Fact("cause_effect", unit, left, left, right, 1.5 - unit.meta))
            else:
                because = _BECAUSE.search(text)
                if because:
                    effect, cause = text[:because.start()].strip(" ,;"), text[because.end():].strip(" ,;.")
                    if (3 <= len(T.words(effect)) <= 18 and 2 <= len(T.words(cause)) <= 18 and not _PRONOUN_START.match(effect)
                            and not _PRONOUN_START.match(cause)):
                        facts["cause_effect"].append(Fact("cause_effect", unit, cause, cause, effect, 1.2 - unit.meta, "because"))
            subject = self.subject_of(unit)
            for kind, cue_key, weight in (("mechanism", "mechanism", 1.0), ("clinical", "clinical", 1.0), ("comparison", "comparison", 1.0),
                                          ("structure", "structure", 1.0), ("pathway", "pathway", 1.0), ("drug", "drug", 1.0),
                                          ("gene", "gene", 0.8), ("emphasis", "emphasis", 1.5)):
                if kind == "structure":
                    if not _STRUCTURE.search(text) or not subject:
                        continue                               # "structure of <X>": the sentence must really talk about structure
                elif not cues.get(cue_key):
                    continue
                if kind == "gene" and (not subject or self.terms[subject].kind not in ("acronym", "enzyme")):
                    continue                                   # "role of <protein>": the subject must be a protein / acronym / enzyme
                if kind == "comparison" and len(unit.terms) < 2:
                    continue                                   # a comparison card needs the two things compared
                facts[kind].append(Fact(kind, unit, subject, weight=weight + 0.3 * cues.get(cue_key, 1) - unit.meta))
            for em in _ENZYME_WORD.finditer(text):
                if em.group().casefold() not in _NOT_MEDICAL:
                    facts["enzyme"].append(Fact("enzyme", unit, em.group(), weight=1.0 + 0.3 * cues.get("mechanism", 0)))
                    break
            if subject:
                facts["concept"].append(Fact("concept", unit, subject, weight=0.5 + 0.2 * len(unit.terms)))
        self.facts = facts

    def subject_of(self, unit: Unit) -> str:
        """The term a sentence is mostly about: the first specific term (not one that fills a quarter of the lecture) in reading order."""
        if not unit.terms:
            return ""
        low = unit.text.casefold()
        limit = max(0.12 * len(self.units), 4)

        def order(key: str) -> tuple[bool, int]:
            at = low.find(key)
            return self.terms[key].count > limit, at if at >= 0 else len(low)

        return min(unit.terms, key=order)

    # ---- scoring
    def importance(self, unit: Unit) -> float:
        c = unit.cues
        term_part = min(sum(self.terms[k].score for k in unit.terms if k in self.terms), 6.0) / 3.0
        cue_part = 1.5 * min(c["emphasis"], 2) + 1.0 * min(c["exam"], 1) + 0.6 * min(c["mechanism"] + c["cause"], 2) + 0.4 * min(c["clinical"], 2)
        length_part = 1.0 if 10 <= unit.n_words <= 35 else (0.3 if unit.n_words <= 45 else -2.0)
        return term_part + cue_part + 1.5 * unit.support + length_part - 3.0 * unit.meta

    def refs(self, unit: Unit) -> list[SourceRef]:
        out = [SourceRef.of(unit.chunk)]
        if unit.chunk.source == SOURCE_TRANSCRIPT:
            for slide in self.index.linked_slides(unit.chunk)[:2]:
                if T.overlap(unit.stems, {T.stem(w) for w in T.content_words(slide.text)}) >= 0.25:
                    out.append(SourceRef.of(slide))
                    break
        return out


def _analysis(ctx: StudyContext) -> Analysis:
    cached = ctx.cache.get("analysis")
    if cached is None or cached.index is not ctx.index or cached.language != ctx.language:
        cached = Analysis(ctx.index, ctx.language)
        ctx.cache["analysis"] = cached
    return cached


def _in_scope(analysis: Analysis, chunks: Sequence[Chunk]) -> list[Unit]:
    ids = {c.id for c in chunks}
    return [u for u in analysis.units if u.chunk.id in ids]


# ----------------------------------------------------------------------------------------------------------------------- provider
_SUMMARY_PICK = {                                # kind -> (cue to require or None, bonus function key)
    "quick": None, "detailed": None, "high_yield": "high", "key_concepts": "terms", "definitions": "definition",
    "mechanisms": "mechanism", "clinical": "clinical", "emphasis": "emphasis", "exam": "exam",
}


class ExtractiveProvider(AIProvider):
    name = "builtin"
    kind = "builtin"
    model = "extractive"

    # ---------------------------------------------------------------------------------------------------------------- summaries
    @staticmethod
    def _facts(ctx: StudyContext):
        """The fact-based study material (validated atomic facts, quality gate) is used for English lectures; other languages keep the sentence-based method below."""
        if (ctx.language or "en").lower().startswith("en"):
            from .factitems import study_of

            return study_of(ctx)
        return None

    def summarize(self, kind: str, chunks: Sequence[Chunk], *, topic: str, limit: int, ctx: StudyContext) -> list[SummaryItem]:
        facts = self._facts(ctx)
        if facts is not None:
            return facts.summarize(kind, chunks, topic, limit)
        an = _analysis(ctx)
        units = _in_scope(an, chunks)
        if not units or limit <= 0:
            return []
        units = [u for u in units if u.n_words <= 60] or units           # a 90-word run-on sentence is not a summary point
        if kind == "key_concepts":
            return self._key_concepts(an, units, topic, limit)
        if kind == "definitions":
            return self._definitions(an, units, topic, limit)
        scored = [(self._summary_score(an, kind, u), u) for u in units]
        scored = [(s, u) for s, u in scored if s > 0]
        picked: list[Unit] = []
        for _s, unit in sorted(scored, key=lambda x: x[0], reverse=True):
            if all(len(unit.stems & p.stems) / max(len(unit.stems), 1) < 0.6 for p in picked):      # no near-duplicates
                picked.append(unit)
            if len(picked) >= limit:
                break
        if kind == "quick" and not topic and limit > 1:                    # spread over the lecture: one strong sentence per topic first
            picked = self._spread(an, scored, limit)
        return [SummaryItem(text=u.text, sources=an.refs(u), topic=u.topic or topic) for u in sorted(picked, key=lambda u: u.pos)]

    def _summary_score(self, an: Analysis, kind: str, u: Unit) -> float:
        base = an.importance(u)
        c = u.cues
        if kind in ("quick", "detailed"):
            return base
        if kind == "high_yield":
            return base if (c["emphasis"] or c["exam"] or u.support >= 0.4 or c["clinical"] or c["cause"]) else 0.0
        if kind == "mechanisms":
            return base + 2 * c["mechanism"] + c["cause"] if (c["mechanism"] or c["cause"] or c["pathway"]) else 0.0
        if kind == "clinical":
            return base + 2 * c["clinical"] if c["clinical"] else 0.0
        if kind == "emphasis":
            return base + 3 * c["emphasis"] if c["emphasis"] else 0.0
        if kind == "exam":
            return base + 3 * c["exam"] + 2 * c["emphasis"] if (c["exam"] or c["emphasis"] or (u.support >= 0.5 and u.terms)) else 0.0
        return base

    @staticmethod
    def _spread(an: Analysis, scored: list[tuple[float, Unit]], limit: int) -> list[Unit]:
        best: dict[str, tuple[float, Unit]] = {}
        for s, u in scored:
            key = u.topic or "_"
            if key not in best or s > best[key][0]:
                best[key] = (s, u)
        firsts = sorted(best.values(), key=lambda x: x[0], reverse=True)[:limit]
        picked = [u for _s, u in firsts]
        for s, u in sorted(scored, key=lambda x: x[0], reverse=True):
            if len(picked) >= limit:
                break
            if u not in picked and all(len(u.stems & p.stems) / max(len(u.stems), 1) < 0.6 for p in picked):
                picked.append(u)
        return picked

    def _best_unit_for(self, an: Analysis, term: Term, units: Sequence[Unit]) -> Optional[Unit]:
        positions = set(term.units)
        candidates = [u for u in units if u.pos in positions]
        if not candidates:
            return None
        defs = {f.unit.pos for f in an.facts["definition"] if term.key in f.subject.casefold()}
        return max(candidates, key=lambda u: (u.pos in defs) * 3 + an.importance(u))

    def _key_concepts(self, an: Analysis, units: Sequence[Unit], topic: str, limit: int) -> list[SummaryItem]:
        in_scope = {u.pos for u in units}
        terms = [t for t in an.terms.values() if in_scope & set(t.units)]
        terms.sort(key=lambda t: t.score, reverse=True)
        out: list[SummaryItem] = []
        seen: set[int] = set()
        for term in terms:
            unit = self._best_unit_for(an, term, units)
            if unit is None or unit.pos in seen:
                continue
            seen.add(unit.pos)
            out.append(SummaryItem(text=unit.text, sources=an.refs(unit), topic=unit.topic or topic, term=T.title_case(term.display)))
            if len(out) >= limit:
                break
        return out

    def _definitions(self, an: Analysis, units: Sequence[Unit], topic: str, limit: int) -> list[SummaryItem]:
        in_scope = {u.pos for u in units}
        facts = sorted((f for f in an.facts["definition"] if f.unit.pos in in_scope), key=lambda f: f.weight + an.importance(f.unit) / 4,
                       reverse=True)
        out, seen = [], set()
        for fact in facts:
            if fact.subject.casefold() in seen:
                continue
            seen.add(fact.subject.casefold())
            out.append(fact)
            if len(out) >= limit:
                break
        out.sort(key=lambda f: f.unit.pos)
        return [SummaryItem(text=f.unit.text, sources=an.refs(f.unit), topic=f.unit.topic or topic, term=T.title_case(f.subject)) for f in out]

    # ---------------------------------------------------------------------------------------------------------------- flashcards
    def generate_flashcards(self, chunks: Sequence[Chunk], *, topic: str, types: Sequence[str], limit: int,
                            ctx: StudyContext) -> list[Flashcard]:
        facts = self._facts(ctx)
        if facts is not None:
            return facts.flashcards(chunks, topic, types, limit)
        an = _analysis(ctx)
        in_scope = {u.pos for u in _in_scope(an, chunks)}
        types = [t for t in (types or CARD_TYPES) if t in CARD_TYPES]
        pools = {t: self._card_pool(an, t, in_scope) for t in types}
        used_units: Counter = Counter()            # a sentence may feed at most two cards (for example a definition and a fill-in-the-blank)
        used_subjects: Counter = Counter()
        cards: list[Flashcard] = []
        while len(cards) < limit and any(pools.values()):
            progressed = False
            for card_type in types:
                pool = pools[card_type]
                while pool:
                    card = pool.pop(0)
                    unit_pos, subject = card[1], card[2].casefold()
                    if used_units[unit_pos] >= 2 or used_subjects[subject] >= 2:
                        continue
                    used_units[unit_pos] += 1
                    used_subjects[subject] += 1
                    cards.append(card[0])
                    progressed = True
                    break
                if len(cards) >= limit:
                    break
            if not progressed:
                break
        order = {u.chunk.id: u.pos for u in reversed(an.units)}
        cards.sort(key=lambda c: (order.get(c.sources[0].chunk_id, 0), c.id))
        return cards

    def _card(self, an: Analysis, card_type: str, fact: Fact, front: str, back: str, difficulty: Optional[str] = None) -> tuple[Flashcard, int, str]:
        unit = fact.unit
        diff = difficulty or self._difficulty(card_type, unit)
        card = Flashcard(id=item_id("c", card_type, front, back), type=card_type, front=front, back=back, sources=an.refs(unit),
                         difficulty=diff, topic=unit.topic)
        return card, unit.pos, fact.subject or front

    @staticmethod
    def _difficulty(card_type: str, unit: Unit) -> str:
        if card_type in ("definition", "concept") and unit.n_words <= 24:
            return "easy"
        if card_type in ("cause_effect", "comparison", "pathway") or unit.n_words > 34:
            return "hard"
        return "medium"

    def _card_pool(self, an: Analysis, card_type: str, in_scope: set[int]) -> list[tuple[Flashcard, int, str]]:
        if card_type == "integration":
            return []                                                          # a card that ties two facts together is written by a language model, not cut from one sentence
        facts = {
            "definition": "definition", "concept": "concept", "mechanism": "mechanism", "cause_effect": "cause_effect",
            "structure_function": "structure", "clinical": "clinical", "comparison": "comparison", "pathway": "pathway",
            "enzyme": "enzyme", "drug": "drug", "gene_protein": "gene", "exam_recall": "emphasis",
        }[card_type]
        pool = [f for f in an.facts[facts] if f.unit.pos in in_scope]
        pool.sort(key=lambda f: f.weight + an.importance(f.unit) / 3, reverse=True)
        out: list[tuple[Flashcard, int, str]] = []
        for fact in pool:
            card = self._make_card(an, card_type, fact)
            if card:
                out.append(card)
        return out

    def _make_card(self, an: Analysis, card_type: str, fact: Fact) -> Optional[tuple[Flashcard, int, str]]:
        unit = fact.unit
        subject = T.title_case(an.term_display(fact.subject.casefold()) if fact.subject.casefold() in an.terms else fact.subject)
        if card_type == "definition":
            return self._card(an, card_type, fact, an.s["what_is"].format(x=T.title_case(fact.subject)),
                              T.truncate(T.title_case(fact.right.rstrip(".")) + ".", 400))
        if card_type == "cause_effect":
            return self._card(an, card_type, fact, an.s["effect_of"].format(x=T.truncate(fact.left.rstrip(".").rstrip(), 120)),
                              T.truncate(T.title_case(fact.right.rstrip(".")) + ".", 300))
        if card_type == "exam_recall":
            blanked = None if unit.meta else self._blank(an, unit)
            if not blanked:
                return None
            return self._card(an, card_type, fact, an.s["fill_blank"].format(x=blanked[0]), f"{blanked[1]} - {unit.text}")
        if not fact.subject:
            return None
        if card_type == "comparison":
            other = next((k for k in unit.terms if k != fact.subject.casefold()), "")
            if not other:
                return None
            return self._card(an, card_type, fact, an.s["compare"].format(a=subject, b=T.title_case(an.term_display(other))), unit.text)
        key = {"concept": "explain", "mechanism": "mechanism", "structure_function": "structure", "clinical": "clinical",
               "pathway": "pathway", "enzyme": "enzyme", "drug": "drug", "gene_protein": "role"}[card_type]
        return self._card(an, card_type, fact, an.s[key].format(x=subject), unit.text)

    def _blank(self, an: Analysis, unit: Unit) -> Optional[tuple[str, str]]:
        """The sentence with its most important term replaced by a blank, and that term."""
        for key in unit.terms:
            term = an.terms.get(key)
            if not term:
                continue
            pattern = _term_pattern(term.display)
            hits = pattern.findall(unit.text)
            if len(hits) == 1 and unit.n_words >= 8:
                return pattern.sub("_____", unit.text, count=1), T.title_case(term.display) if term.kind == "phrase" else term.display
        return None

    # ---------------------------------------------------------------------------------------------------------------- questions
    def generate_questions(self, chunks: Sequence[Chunk], *, topic: str, types: Sequence[str], limit: int,
                           ctx: StudyContext) -> list[Question]:
        facts = self._facts(ctx)
        if facts is not None:
            return facts.questions(chunks, topic, types or QUESTION_TYPES, limit)
        an = _analysis(ctx)
        in_scope = {u.pos for u in _in_scope(an, chunks)}
        types = [t for t in (types or QUESTION_TYPES) if t in QUESTION_TYPES]
        builders = {
            "mcq": self._mcq, "true_false": self._true_false, "short_answer": self._short_answer, "explain_why": self._explain_why,
            "clinical_vignette": self._clinical, "mechanism": self._mechanism,
        }
        pools = {t: builders[t](an, in_scope) for t in types}
        out: list[Question] = []
        used: set[int] = set()
        while len(out) < limit and any(pools.values()):
            progressed = False
            for qtype in types:
                pool = pools[qtype]
                while pool:
                    question, pos = pool.pop(0)
                    if pos in used and qtype != "true_false":
                        continue
                    used.add(pos)
                    out.append(question)
                    progressed = True
                    break
                if len(out) >= limit:
                    break
            if not progressed:
                break
        order = {u.chunk.id: u.pos for u in reversed(an.units)}
        out.sort(key=lambda q: (order.get(q.sources[0].chunk_id, 0), q.id))
        return out

    def _question(self, an: Analysis, qtype: str, unit: Unit, stem: str, answer: str, explanation: str, options: Optional[list[str]] = None,
                  difficulty: Optional[str] = None) -> tuple[Question, int]:
        if options and qtype != "true_false":                      # True / False keep their natural order
            rnd = random.Random(item_id("q", qtype, stem))
            options = list(options)
            rnd.shuffle(options)
        q = Question(id=item_id("q", qtype, stem, answer), type=qtype, question=stem, answer=answer, explanation=explanation,
                     options=options or [], sources=an.refs(unit), difficulty=difficulty or self._difficulty("", unit), topic=unit.topic)
        return q, unit.pos

    def _distractor_terms(self, an: Analysis, correct: str, kind: str, near: Unit, count: int = 4) -> list[str]:
        same = [t for t in an.terms.values() if t.kind == kind and t.display.casefold() != correct.casefold()]
        other = [t for t in an.terms.values() if t.kind != kind and t.display.casefold() != correct.casefold()]
        rnd = random.Random(item_id("d", correct, str(near.pos)))
        pool = []
        for group in (same, other):
            group = [t for t in group if t.key not in near.terms and t.key not in correct.casefold() and correct.casefold() not in t.key]
            rnd.shuffle(group)
            pool.extend(group)
        seen: set[str] = set()
        picked: list[str] = []
        for t in pool:
            shown = T.title_case(t.display) if t.kind == "phrase" else t.display
            if shown.casefold() not in seen:
                seen.add(shown.casefold())
                picked.append(shown)
            if len(picked) == count:
                break
        return picked

    def _mcq(self, an: Analysis, in_scope: set[int]) -> list[tuple[Question, int]]:
        out: list[tuple[Question, int]] = []
        facts = [f for f in an.facts["definition"] if f.unit.pos in in_scope]
        facts.sort(key=lambda f: f.weight + an.importance(f.unit) / 3, reverse=True)
        for fact in facts:
            key = fact.subject.casefold()
            term = an.terms.get(key)
            correct = T.title_case(fact.subject)
            kind = term.kind if term else "phrase"
            wrong = self._distractor_terms(an, correct, kind, fact.unit)
            if len(wrong) < 4:
                continue
            stem = an.s["which_term"].format(x=T.truncate(T.title_case(fact.right.rstrip(".")), 220))
            out.append(self._question(an, "mcq", fact.unit, stem, correct, an.s["lecture_says"].format(x=fact.unit.text), [correct, *wrong[:4]]))
        by_pos = {q[1] for q in out}
        for unit in sorted((an.units[p] for p in in_scope), key=lambda u: an.importance(u), reverse=True):
            if unit.pos in by_pos or not unit.terms:
                continue
            blanked = self._blank(an, unit)
            if not blanked:
                continue
            sentence, correct = blanked
            term = an.terms.get(correct.casefold())
            wrong = self._distractor_terms(an, correct, term.kind if term else "phrase", unit)
            if len(wrong) < 4:
                continue
            out.append(self._question(an, "mcq", unit, an.s["complete"].format(x=sentence), correct,
                                      an.s["lecture_says"].format(x=unit.text), [correct, *wrong[:4]]))
        return out

    def _true_false(self, an: Analysis, in_scope: set[int]) -> list[tuple[Question, int]]:
        out: list[tuple[Question, int]] = []
        units = sorted((an.units[p] for p in in_scope), key=lambda u: an.importance(u), reverse=True)
        flip = False
        for unit in units:
            if unit.meta or not 8 <= unit.n_words <= 32 or not (unit.terms or unit.cues["emphasis"]):
                continue
            false_version = self._falsify(an, unit)
            yes, no = an.s["true"], an.s["false"]
            if flip and false_version:
                out.append(self._question(an, "true_false", unit, an.s["true_false"].format(x=false_version), no,
                                          f"{an.s['false_prefix']} {an.s['lecture_says'].format(x=unit.text)}", [yes, no], "medium"))
            else:
                out.append(self._question(an, "true_false", unit, an.s["true_false"].format(x=unit.text), yes,
                                          f"{an.s['true_prefix']} {an.s['lecture_says'].format(x=unit.text)}", [yes, no], "easy"))
            flip = not flip
        return out

    @staticmethod
    def _falsify(an: Analysis, unit: Unit) -> Optional[str]:
        for a, b in _ANTONYMS:
            m = re.search(r"(?<![\w-])" + a + r"(?![\w-])", unit.text, re.IGNORECASE)
            if m:
                swapped = b.capitalize() if m.group()[0].isupper() else b
                return unit.text[:m.start()] + swapped + unit.text[m.end():]
        return None

    def _short_answer(self, an: Analysis, in_scope: set[int]) -> list[tuple[Question, int]]:
        out = []
        facts = [f for f in an.facts["definition"] if f.unit.pos in in_scope]
        facts.sort(key=lambda f: f.weight + an.importance(f.unit) / 3, reverse=True)
        for fact in facts:
            out.append(self._question(an, "short_answer", fact.unit, an.s["own_words"].format(x=T.title_case(fact.subject)), fact.unit.text,
                                      an.s["lecture_says"].format(x=fact.unit.text)))
        seen = {q[1] for q in out}
        concepts = [f for f in an.facts["concept"] if f.unit.pos in in_scope and f.unit.pos not in seen and f.unit.n_words >= 8]
        concepts.sort(key=lambda f: f.weight + an.importance(f.unit) / 3, reverse=True)
        for fact in concepts:
            out.append(self._question(an, "short_answer", fact.unit, an.s["lecture_say"].format(x=T.title_case(an.term_display(fact.subject))),
                                      fact.unit.text, an.s["lecture_says"].format(x=fact.unit.text)))
        return out

    def _explain_why(self, an: Analysis, in_scope: set[int]) -> list[tuple[Question, int]]:
        out = []
        facts = [f for f in an.facts["cause_effect"] if f.unit.pos in in_scope]
        facts.sort(key=lambda f: f.weight + an.importance(f.unit) / 3, reverse=True)
        for fact in facts:
            if fact.via == "because":                                 # "<effect> because <cause>": ask for the cause, hide it
                stem = an.s["explain_why"].format(x=T.truncate(T.title_case(fact.right.rstrip(".")), 240))
                answer = T.title_case(fact.left.rstrip(".")) + "."
            else:                                                     # "<cause> leads to <effect>": the next sentence explains it
                later = [u for u in an.by_chunk.get(fact.unit.chunk.id, []) if u.pos == fact.unit.pos + 1]
                if not later:
                    continue                                          # nothing the lecture says would answer "why": no question
                stem = an.s["explain_why"].format(x=T.truncate(fact.unit.text.rstrip("."), 240))
                answer = later[0].text
            out.append(self._question(an, "explain_why", fact.unit, stem, answer, an.s["lecture_says"].format(x=fact.unit.text) +
                                      ("" if fact.via == "because" else f" {answer}"), difficulty="hard"))
        return out

    def _statement_options(self, an: Analysis, fact: Fact, kind_pool: list[Fact]) -> Optional[list[str]]:
        rnd = random.Random(item_id("s", fact.unit.text))
        others = [f for f in kind_pool if f.unit.pos != fact.unit.pos and f.unit.topic != fact.unit.topic and
                  len(f.unit.stems & fact.unit.stems) / max(len(fact.unit.stems), 1) < 0.5]
        if len(others) < 4:
            others += [f for f in an.facts["concept"] if f.unit.pos != fact.unit.pos and f.unit.topic != fact.unit.topic and
                       len(f.unit.stems & fact.unit.stems) / max(len(fact.unit.stems), 1) < 0.4]
        rnd.shuffle(others)
        chosen: list[str] = []
        for f in others:
            text = T.truncate(f.unit.text, 200)
            if text not in chosen:
                chosen.append(text)
            if len(chosen) == 4:
                break
        return [T.truncate(fact.unit.text, 200), *chosen] if len(chosen) == 4 else None

    def _statement_questions(self, an: Analysis, in_scope: set[int], fact_kind: str, qtype: str, stem_for) -> list[tuple[Question, int]]:
        pool = [f for f in an.facts[fact_kind]]
        mine = [f for f in pool if f.unit.pos in in_scope and f.subject]
        mine.sort(key=lambda f: f.weight + an.importance(f.unit) / 3, reverse=True)
        out = []
        for fact in mine:
            options = self._statement_options(an, fact, pool)
            if not options:
                continue
            out.append(self._question(an, qtype, fact.unit, stem_for(T.title_case(an.term_display(fact.subject.casefold()))), options[0],
                                      an.s["lecture_says"].format(x=fact.unit.text), options, "hard"))
        return out

    def _clinical(self, an: Analysis, in_scope: set[int]) -> list[tuple[Question, int]]:
        return self._statement_questions(
            an, in_scope, "clinical", "clinical_vignette", lambda term: an.s["scenario"].format(x=term))

    def _mechanism(self, an: Analysis, in_scope: set[int]) -> list[tuple[Question, int]]:
        return self._statement_questions(an, in_scope, "mechanism", "mechanism", lambda term: an.s["which_mechanism"].format(x=term))

    # ---------------------------------------------------------------------------------------------------------------- explain
    def explain(self, question: str, chunks: Sequence[Chunk], ctx: StudyContext) -> SummaryItem:
        an = _analysis(ctx)
        wanted = {T.stem(w) for w in T.content_words(question)}
        units = _in_scope(an, chunks) or an.units
        scored = []
        for u in units:
            hit = wanted & u.stems
            if wanted and len(hit) >= max(1, math.ceil(len(wanted) * 0.4)):
                scored.append((len(hit) / len(wanted) + 0.1 * an.importance(u), u))
        if not scored:
            return SummaryItem(text=NOT_IN_SOURCES, sources=[])
        best = [u for _s, u in sorted(scored, key=lambda x: x[0], reverse=True)[:3]]
        best.sort(key=lambda u: u.pos)
        refs: list[SourceRef] = []
        for u in best:
            for r in an.refs(u):
                if all(r.chunk_id != x.chunk_id for x in refs):
                    refs.append(r)
        return SummaryItem(text=" ".join(u.text for u in best), sources=refs, topic=best[0].topic)

    # ---------------------------------------------------------------------------------------------------------------- answers
    def evaluate_answer(self, question: Question, answer: str, chunks: Sequence[Chunk], ctx: StudyContext) -> Evaluation:
        given = T.clean(answer)
        if not given:
            return Evaluation(False, 0.0, "No answer was given.")
        if question.type in ("mcq", "true_false"):
            ok = given.casefold() == question.answer.casefold()
            return Evaluation(ok, 1.0 if ok else 0.0, question.explanation)
        asked = {T.stem(w) for w in T.content_words(question.question)}              # words of the question are given, not asked for
        words = [w for w in T.content_words(question.answer) if T.stem(w) not in asked] or T.content_words(question.answer)
        from .facts import VERB_FORMS

        nouns = [w for w in words if w not in VERB_FORMS]                              # a student who names the structures need not repeat the lecturer's verbs
        expected = {T.stem(w): w for w in (nouns if len(nouns) >= 3 else words)}
        mine = {T.stem(w) for w in T.content_words(given)}
        if not expected:
            return Evaluation(False, 0.0, "")
        recall = T.overlap(set(expected), mine)
        missing = [expected[s] for s in expected if s not in mine][:6]
        if recall >= 0.5:
            return Evaluation(True, round(recall, 2), question.explanation, missing)
        note = "Partly right. " if recall >= 0.25 else ""
        hint = (" Key words missing: " + ", ".join(missing) + ".") if missing else ""
        return Evaluation(False, round(recall, 2), note + question.explanation + hint, missing)
