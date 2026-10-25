"""Summaries, key concepts, flashcards and questions made from the validated facts of a lecture (``factbase.py``).

Nothing here reads the transcript on its own: every item is made from a :class:`~.facts.StudyFact` (one clean, grammatical, source-grounded statement) and carries the seconds it
was said. Every item then has to pass the quality gate (``quality.py``); an item that does not is dropped and counted (``FactStudy.gate``), never shown. The student gets fewer but
correct items rather than a full deck of malformed ones.

Wording around the lecture's own sentences is fixed and specific: "What is the neural tube?", "Where is the notochord located?", "How do X and Y differ?", a cloze with exactly one
blank. There is no "Explain: X", no "What does the lecture say about X?".
"""

from __future__ import annotations

import random
import re
from collections import Counter
from typing import Callable, Iterable, Optional, Sequence

from . import curate
from . import facts as F
from . import quality as Q
from . import recall as R
from . import text as T
from .factbase import FactBase, TermInfo, _term_pattern, build_factbase, fine_class, is_plural, singular_key, subject_term, term_category
from .models import CARD_TYPES, SOURCE_PDF, SOURCE_TRANSCRIPT, Chunk, Flashcard, Question, SourceRef, SummaryItem, item_id
from .providers import StudyContext

NARROW_MIN, NARROW_MAX = 5.0, 30.0              # seconds: a source reference points at the few seconds a statement was said, not at the whole chunk
BLANK = "_____"
MAX_CLOZE_WORDS = 28

_ANTONYMS = (("increases", "decreases"), ("increase", "decrease"), ("increased", "decreased"), ("increasing", "decreasing"), ("activates", "inhibits"),
             ("activate", "inhibit"), ("activated", "inhibited"), ("activation", "inhibition"), ("stimulates", "inhibits"), ("promotes", "inhibits"),
             ("upregulates", "downregulates"), ("upregulated", "downregulated"), ("higher", "lower"), ("anterior", "posterior"), ("dorsal", "ventral"), ("superior", "inferior"),
             ("medial", "lateral"), ("proximal", "distal"), ("cranial", "caudal"), ("anabolic", "catabolic"), ("gain", "loss"), ("synthesis", "degradation"),
             ("excitatory", "inhibitory"), ("sensory", "motor"), ("afferent", "efferent"), ("ascending", "descending"), ("before", "after"), ("enhances", "reduces"),
             ("elevated", "reduced"), ("excess", "deficiency"), ("deficiency", "excess"), ("opens", "closes"), ("open", "closed"))

_ANTONYM_MAP: dict[str, str] = {}
for _a, _b in _ANTONYMS:
    _ANTONYM_MAP.setdefault(_a, _b)
    _ANTONYM_MAP.setdefault(_b, _a)
_ANTONYM_RE = re.compile(r"(?<![\w-])(" + "|".join(sorted(map(re.escape, _ANTONYM_MAP), key=len, reverse=True)) + r")(?![\w-])", re.IGNORECASE)

_GENERIC_FIRST = frozenset("enzyme protein hormone gene receptor liver muscle brain kidney heart skin blood lung bone cell cells the this that these those".split())
_CAUSE = re.compile(r"\b(?P<verb>leads? to|results? in|resulting in|causes?|causing|triggers?|gives? rise to)\b", re.IGNORECASE)
_GIVES_RISE = re.compile(r"\b(?:gives?\s+rise\s+to|differentiates?\s+(?:into|to)|develops?\s+into|becomes?|turns?\s+into|forms?)\b", re.IGNORECASE)
_DERIVED = re.compile(r"\b(?:derived\s+from|arises?\s+from|develops?\s+from|originates?\s+from|differentiates?\s+from|comes?\s+from)\b", re.IGNORECASE)
_MADE_IN = re.compile(r"\b(?:is|are)\s+(?P<part>formed|synthesized|synthesised|produced|made|found|located|stored|released|secreted|expressed|absorbed|degraded)\s+(?:in|into|by|within)\b",
                      re.IGNORECASE)
#: "X is the most complex one", "X is a part of it": a predicate that defines nothing
_EMPTY_PREDICATE = re.compile(r"\b(?:is|are)\s+(?:the|a|an)\s+(?:[\w-]+\s+){0,3}?(?:one|ones|thing|things|part|example|kind|type)\s*[.!]?$", re.IGNORECASE)
_FRONT_ADVERBIALS = frozenset("in under during at after before when without with within through for on".split())     # "Under anaerobic conditions lactate dehydrogenase converts ..."
_FINITE_AUX = frozenset("is are was were can may will must should has have does do cannot".split())
_WH_NOUN = {"enzyme": "enzyme", "cell": "cell type", "structure": "structure", "process": "process", "disease": "condition", "drug": "drug", "molecule": "molecule", "gene": "gene"}
_SELF_REFERENCE = re.compile(r"\b(?:it|its|itself|they|their|this|these|those)\b", re.IGNORECASE)
_STATE_VERBS = frozenset("remain remains remained stay stays stayed seem seems seemed appear appears look looks exist exists".split())
_BECOMES = re.compile(r"\b(?:becomes?|remains?\s+as|turns?\s+into|will\s+become|will\s+remain\s+as)\b", re.IGNORECASE)
_DEFINITION = re.compile(r"\b(?:is|are)\s+(?:defined\s+as|called|known\s+as|the\s+so-called|a|an|the)\b|\brefers?\s+to\b|\bstands?\s+for\b|\bmeans\b", re.IGNORECASE)


class FactStudy:
    """The study material of one lecture, made from its fact base."""

    def __init__(self, ctx: StudyContext) -> None:
        self.ctx = ctx
        self.index = ctx.index
        self.base: FactBase = build_factbase(ctx.index, ctx.language)
        self.words = F.WordKnowledge(extra=list(getattr(ctx.index, "glossary", [])) + list(getattr(ctx.index, "lecture_terms", [])),
                                     counts=_word_counts(ctx.index))
        self.unresolved = list(getattr(ctx.index, "unresolved", []))
        self.gate = Q.Gate()
        self._chunk = {c.id: c for c in ctx.index.chunks}
        self.premium = bool(ctx.cache.get("premium"))               # the Automatic Study Engine: no fill-in-the-blank items, and every item also passes the premium gate (curate.py)

    # ==================================================================================================================== helpers
    def scope(self, chunks: Sequence[Chunk]) -> list[F.StudyFact]:
        """The facts of these chunks that study items may be made from: every HIGH fact, and MEDIUM facts with clear support (recall.usable)."""
        ids = {c.id for c in chunks}
        return [f for f in self.base.facts if f.chunk_id in ids and self.usable(f)]

    def usable(self, fact: F.StudyFact) -> bool:
        return R.usable(fact, self.known_term, lambda name: (self.term_info(name).category if self.term_info(name) else ""))

    def known_term(self, term: str) -> bool:
        return self.base.known(term)

    def refs(self, fact: F.StudyFact) -> list[SourceRef]:
        chunk = self._chunk.get(fact.chunk_id)
        if chunk is None:
            return []
        if chunk.source == SOURCE_PDF:
            return [SourceRef.of(chunk)]
        start, end = fact.source_start, fact.source_end
        if start is not None and end is not None:
            if end - start < NARROW_MIN:
                end = start + NARROW_MIN
            if end - start > NARROW_MAX:
                end = start + NARROW_MAX
        out = [SourceRef.of_span(chunk, start, end)]
        stems = frozenset(T.stem(w) for w in T.content_words(fact.statement))
        for slide in self.index.linked_slides(chunk)[:2]:
            if T.overlap(stems, {T.stem(w) for w in T.content_words(slide.text)}) >= 0.25:
                out.append(SourceRef.of(slide))
                break
        return out

    def plain_english(self, term: TermInfo) -> bool:
        """"origin", "ending", "outcome": a single ordinary English word of no medical kind is not something to test - only a term of a medical kind, an enzyme, an acronym
        or a word the medical vocabulary lists is blanked or made a key concept."""
        words = T.words(term.display)
        if len(words) != 1 or term.category != "other" or term.kind in ("enzyme", "acronym"):
            return False
        kb = F.default_knowledge()
        if kb is not None:
            try:
                if kb.exact(term.display):
                    return False
            except Exception:                                                  # noqa: BLE001 - the vocabulary is optional here
                pass
        return words[0] in self.words.common

    def term_info(self, name: str) -> Optional[TermInfo]:
        return self.base.terms.get(singular_key(name))

    def in_head(self, fact: F.StudyFact, term: str = "") -> bool:
        """The term is what the statement is ABOUT: it stands in the subject part of the sentence (before the first verb)."""
        name = term or fact.subtopic
        if not name:
            return False
        info = self.term_info(name)
        if info is None:
            return False
        chosen = subject_term(fact.statement, [info])
        return chosen is not None

    def display(self, term: str) -> str:
        info = self.term_info(term)
        return T.title_case(info.display if info else term)

    # ==================================================================================================================== overview
    def overview(self, pack) -> dict:
        """Title, subject, duration, the topics with their main terms and how much of the lecture could be turned into reliable statements."""
        meta = getattr(self.index, "meta", {}) or {}
        topics = []
        for name in pack.topics:
            in_topic = [t for t in self.base.terms.values() if name in t.where]
            in_topic.sort(key=lambda t: -t.score)
            topics.append({"name": name, "facts": sum(1 for f in self.base.facts if f.topic == name), "key_terms": [self.display(t.display) for t in in_topic[:5]]})
        ranked = sorted(self.base.terms.values(), key=lambda t: -t.score)
        types = Counter(f.fact_type for f in self.base.facts)
        report = self.base.reject_report()
        return {"title": meta.get("title") or pack.title, "subject": meta.get("subject") or "", "duration_seconds": meta.get("duration"), "language": pack.language,
                "source": meta.get("source_note", ""), "topics": topics, "key_terms": [self.display(t.display) for t in ranked[:12]],
                "facts": {"validated": report["facts"], "sentences": report["sentences"], "yield": round(report["facts"] / max(report["sentences"], 1), 2),
                          "high": report["high"], "medium": report["medium"], "used": sum(1 for f in self.base.facts if self.usable(f)),
                          "clauses_rescued": report["clauses_rescued"], "recovered": report["recovered"], "by_type": dict(types.most_common())}}

    # ==================================================================================================================== summaries
    def summarize(self, kind: str, chunks: Sequence[Chunk], topic: str, limit: int) -> list[SummaryItem]:
        facts = self.scope(chunks)
        if not facts or limit <= 0:
            return []
        if kind == "key_concepts":
            return self._key_concepts(facts, topic, limit)
        pool = [f for f in facts if self._wanted(kind, f)]
        pool.sort(key=lambda f: -f.importance)
        picked: list[F.StudyFact] = []
        per_subject: Counter = Counter()
        for fact in pool:
            if any(F.near_duplicate(fact.statement, p.statement) or F.subsumed(fact.statement, p.statement) for p in picked):
                continue
            if fact.subtopic and per_subject[fact.subtopic.casefold()] >= (2 if kind in ("detailed", "quick") else 1):
                continue
            verdict = Q.check_summary_point(fact.statement, unknown=self.words.unknown, unresolved=self.unresolved)
            if not self.gate.record("summary", verdict, fact.statement):
                continue
            picked.append(fact)
            per_subject[fact.subtopic.casefold()] += 1
            if len(picked) >= limit:
                break
        picked.sort(key=lambda f: f.position)
        term = kind == "definitions"
        return [SummaryItem(text=f.statement, sources=self.refs(f), topic=f.topic or topic, term=self.display(f.subtopic) if term and f.subtopic else "") for f in picked]

    @staticmethod
    def _wanted(kind: str, f: F.StudyFact) -> bool:
        if kind in ("quick", "detailed"):
            return True
        if kind == "high_yield":
            return f.teacher_emphasis or f.importance >= 3.0 or f.fact_type in ("definition", "clinical_relevance", "cause_effect", "mechanism")
        if kind == "definitions":
            return f.fact_type == "definition"
        if kind == "mechanisms":
            return f.fact_type in ("mechanism", "cause_effect", "sequence", "development", "derivative")
        if kind == "clinical":
            return f.fact_type == "clinical_relevance"
        if kind == "emphasis":
            return f.teacher_emphasis
        if kind == "exam":
            return f.teacher_emphasis or f.importance >= 3.5
        return True

    def _key_concepts(self, facts: Sequence[F.StudyFact], topic: str, limit: int) -> list[SummaryItem]:
        in_scope = {f.fact_id: f for f in facts}
        weight = {"definition": 3.0, "derivative": 2.2, "location": 2.0, "function": 2.0, "classification": 2.0, "structure_function": 2.0, "development": 1.6, "cause_effect": 1.4,
                  "mechanism": 1.4, "clinical_relevance": 1.2}
        out: list[SummaryItem] = []
        used: set[str] = set()
        terms = sorted((t for t in self.base.terms.values() if any(fid in in_scope for fid in t.facts)), key=lambda t: -t.score)
        for term in terms:
            if F.is_generic_label(term.display, self.known_term) or len(T.words(term.display)) > 4 or F.is_bare_generic(term.display) or self.plain_english(term):
                continue
            candidates = [in_scope[fid] for fid in term.facts if fid in in_scope and fid not in used and in_scope[fid].subtopic and singular_key(in_scope[fid].subtopic) == term.key]
            if not candidates:
                continue
            fact = max(candidates, key=lambda f: (weight.get(f.fact_type, 0.5) + (1.5 if f.subtopic.casefold() == term.key else 0.0) + f.importance / 4))
            label = self.display(term.display)
            verdict = Q.check_concept(label, fact.statement, known_term=self.known_term, unknown=self.words.unknown, unresolved=self.unresolved)
            if not self.gate.record("key_concept", verdict, f"{label}: {fact.statement}"):
                continue
            used.add(fact.fact_id)
            out.append(SummaryItem(text=fact.statement, sources=self.refs(fact), topic=fact.topic or topic, term=label, importance=self.importance_of(term, fact)))
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def importance_of(term: TermInfo, fact: F.StudyFact) -> str:
        """Why a key concept matters, from what the lecture did with it: stressed by the lecturer, named again and again, or one supporting term among others."""
        if fact.teacher_emphasis:
            return "High-yield - the lecturer stressed it"
        if term.count >= 8 or len(term.facts) >= 3:
            return f"Core term - named {term.count} times in the lecture"
        return f"Supporting term - named {term.count} time{'s' if term.count != 1 else ''}"

    # ==================================================================================================================== cloze and distractors
    def cloze(self, fact: F.StudyFact, prefer: str = "") -> Optional[tuple[str, str, TermInfo]]:
        """``(sentence with one blank, the blanked words, the term)``; None when no term can be blanked cleanly."""
        words = T.words(fact.statement)
        if not 8 <= len(words) <= MAX_CLOZE_WORDS:
            return None
        order = [t for t in (self.term_info(m) for m in ([prefer] if prefer else []) + [fact.subtopic] + fact.medical_terms) if t is not None]
        seen: set[str] = set()
        for term in order:
            if term.key in seen:
                continue
            seen.add(term.key)
            if term.category == "other" and term.kind == "phrase" or F.is_generic_label(term.display, self.known_term) or len(T.words(term.display)) > 3 \
                    or F.is_bare_generic(term.display) or self.plain_english(term):
                continue
            hits = list(_term_pattern(term.key).finditer(fact.statement))
            if len(hits) != 1:
                continue
            m = hits[0]
            if m.start() == 0 and len(fact.statement) - m.end() < 20:
                continue
            sentence = fact.statement[:m.start()] + BLANK + fact.statement[m.end():]
            if m.start() == 0:
                pass
            return sentence, m.group(0), term
        return None

    def distractors(self, answer: str, term: TermInfo, fact: F.StudyFact, count: int = 3) -> list[str]:
        """Terms of the SAME kind as the answer (cell / structure / process / ...) from the same lecture, none of them in the fact itself."""
        if term.category == "other":
            return []
        stems = {T.stem(w) for w in T.words(answer)}
        mine = {m.casefold() for m in fact.medical_terms}
        pool = []
        for other in self.base.terms.values():
            if other.key == term.key or other.category != term.category or other.display.casefold() in mine:
                continue
            if fine_class(other.display, other.category) != fine_class(term.display, term.category) and term.category in ("process", "disease"):
                continue                                                       # a wrong answer must look like the right one: a pathway among pathways, a syndrome among syndromes
            if other.key in term.key or term.key in other.key or {T.stem(w) for w in T.words(other.display)} & stems:
                continue                                                       # "neural tube" vs "neural tube formation" / "neural fold": too close to tell apart
            if F.is_generic_label(other.display, self.known_term) or len(T.words(other.display)) > 3 or F.is_bare_generic(other.display):
                continue
            if any(len(w) >= 5 and self.words.unknown(w) for w in T.words(other.display)) or not (other.facts or other.count >= 2):
                continue                                                       # a misheard spelling ("Kerbomoyl phosphate synthetase"), or a term said once in a sentence that was not used
            if _term_pattern(other.key).search(fact.statement):
                continue
            first = T.words(other.display)[0]
            if first in _GENERIC_FIRST or any(other.key != o.key and o.key.endswith(" " + other.key) for o in self.base.terms.values()):
                continue                                                       # "enzyme aldolase", "liver glucokinase"; and "isomerase" when the lecture says "triose phosphate isomerase"
            if term.where and other.where and not (term.where & other.where):
                continue                                                       # peers from the same part of the lecture, not from another subject
            pool.append(other)
        pool.sort(key=lambda t: -t.score)
        rnd = random.Random(item_id("d", answer, fact.fact_id))
        head = pool[:12]
        rnd.shuffle(head)
        out: list[str] = []
        used: set[str] = set()
        for other in head:
            shown = other.display if not answer[:1].isupper() else T.title_case(other.display)
            if answer.endswith("s") and not shown.endswith("s") and other.key + "s" in self.base.terms:
                shown = self.base.terms[other.key + "s"].display
            key = " ".join(T.stem(w) for w in T.words(shown))
            if key in used or not (0.4 <= len(shown) / max(len(answer), 1) <= 2.5):
                continue
            used.add(key)
            out.append(shown)
            if len(out) == count:
                break
        return out if len(out) == count else []

    # ==================================================================================================================== flashcards
    def flashcards(self, chunks: Sequence[Chunk], topic: str, types: Sequence[str], limit: int) -> list[Flashcard]:
        facts = sorted(self.scope(chunks), key=lambda f: -f.importance)
        wanted = [t for t in (types or CARD_TYPES) if t in CARD_TYPES]
        pools: dict[str, list[tuple[F.StudyFact, Flashcard]]] = {}
        for fact in facts:
            for card in self._cards_of(fact):
                if card.type not in wanted:
                    continue
                verdict = Q.check_card(card, unknown=self.words.unknown, unresolved=self.unresolved, known_term=self.known_term)
                if verdict.ok and self.premium:
                    verdict = self._premium(verdict, curate.review_card(card, self.index))
                if self.gate.record("flashcard", verdict, f"{card.front} -> {card.back}"):
                    pools.setdefault(card.type, []).append((fact, card))
        out: list[Flashcard] = []
        fronts: set[str] = set()
        used_facts: Counter = Counter()
        used_subjects: Counter = Counter()
        kinds_of: dict[str, set] = {}
        order = sorted(pools, key=lambda t: (t == "exam_recall", -len(pools[t])))            # a real question before a cloze of the same statement
        while len(out) < limit and any(pools.values()):
            progressed = False
            for ctype in order:
                pool = pools[ctype]
                while pool:
                    fact, card = pool.pop(0)
                    key = (fact.subtopic or card.front).casefold()
                    repeat = bool(used_facts[fact.fact_id]) and card.type in kinds_of.get(fact.fact_id, set())       # a second card of a statement only from another angle
                    if card.front.casefold() in fronts or used_facts[fact.fact_id] >= 2 or used_subjects[key] >= 2 or repeat:
                        continue
                    fronts.add(card.front.casefold())
                    used_facts[fact.fact_id] += 1
                    kinds_of.setdefault(fact.fact_id, set()).add(card.type)
                    used_subjects[key] += 1
                    out.append(card)
                    progressed = True
                    break
                if len(out) >= limit:
                    break
            if not progressed:
                break
        position = {f.fact_id: f.position for f in facts}
        out.sort(key=lambda c: (position.get(c.fact_id, 0), c.id))
        return out

    @staticmethod
    def _premium(verdict: Q.Verdict, review: curate.Review) -> Q.Verdict:
        """The premium gate's reasons replace an "ok" verdict when it refuses the item (an item must pass both gates)."""
        return verdict if review.ok else Q.Verdict(False, list(review.reasons), [], 0.0)

    def _card(self, fact: F.StudyFact, ctype: str, front: str, back: str) -> Flashcard:
        return Flashcard(id=item_id("c", ctype, front, back), type=ctype, front=front, back=back, sources=self.refs(fact), difficulty=fact.difficulty, topic=fact.topic, fact_id=fact.fact_id,
                         high_yield=fact.teacher_emphasis)

    def _cards_of(self, fact: F.StudyFact) -> list[Flashcard]:
        cards: list[Flashcard] = []
        subject = self.display(fact.subtopic) if fact.subtopic else ""
        text = fact.statement
        subj_info = self.term_info(fact.subtopic) if fact.subtopic else None
        subject_ok = bool(subject) and subj_info is not None and not F.is_generic_label(subject, self.known_term) and not F.is_bare_generic(subject) and self.in_head(fact)
        cat = subj_info.category if subj_info else "other"
        the = lambda name, c=cat, pr=(subj_info.proper if subj_info else False): _the(name, c, pr)                                                                                      # noqa: E731
        if subject_ok:
            plural = is_plural(subject)
            if fact.fact_type == "definition" and _DEFINITION.search(text) and not re.search(r"\b(?:is|are)\s+(?:called|known\s+as|the\s+so-called)\b", text, re.IGNORECASE) \
                    and not _EMPTY_PREDICATE.search(text):
                cards.append(self._card(fact, "definition", f"What {'are' if plural else 'is'} {the(subject)}?", text))
            elif fact.fact_type == "derivative":
                if _DERIVED.search(text):
                    cards.append(self._card(fact, "concept", f"What {'do' if plural else 'does'} {the(subject)} develop from?", text))
                elif _GIVES_RISE.search(text):
                    cards.append(self._card(fact, "concept", f"What {'do' if plural else 'does'} {the(subject)} give rise to?" if plural else f"What does {the(subject)} give rise to?", text))
            elif fact.fact_type == "location":
                cards.append(self._card(fact, "structure_function", f"Where {'are' if plural else 'is'} {the(subject)} located?", text))
            elif fact.fact_type in ("function", "structure_function"):
                cards.append(self._card(fact, "structure_function", f"What {'is' if not plural else 'are'} the function{'s' if plural else ''} of {the(subject)}?", text))
            elif fact.fact_type == "classification":
                cards.append(self._card(fact, "concept", f"How {'are' if plural else 'is'} {the(subject)} divided or classified?", text))
            elif fact.fact_type == "clinical_relevance":
                cards.append(self._card(fact, "clinical", f"What is the clinical relevance of {the(subject)}?", text))
            elif fact.fact_type == "comparison" and len(fact.medical_terms) >= 2:
                a, b = (self.display(t) for t in fact.medical_terms[:2])
                if a.casefold() != b.casefold():
                    cards.append(self._card(fact, "comparison", f"How do {the(a)} and {the(b)} differ?", text))
            elif fact.fact_type in ("cause_effect", "mechanism"):
                split = _CAUSE.search(text)
                if split:
                    cause, effect = text[:split.start()].strip(" ,;"), text[split.end():].strip(" ,;.")
                    if 3 <= len(T.words(cause)) <= 14 and 3 <= len(T.words(effect)) <= 20 and not F._PRONOUN_SUBJECT.match(cause):
                        clause = F._finite(cause.split())
                        front = f"What happens when {_lower_first(cause)}?" if clause else f"What is the result of {_lower_first(cause)}?"
                        cards.append(self._card(fact, "cause_effect", front, T.title_case(effect.rstrip(".")) + "."))
        made_in = _MADE_IN.search(text)
        tokens = text.split()
        verb_at = F._subject_end(tokens)
        main_passive = made_in is not None and 0 < verb_at < len(tokens) - 1 and tokens[verb_at].casefold() in ("is", "are")             and tokens[verb_at + 1].casefold().strip(",.") == made_in.group("part").casefold()
        if subject_ok and not cards and main_passive:
            plural = is_plural(subject)                                                                  # "The N-acetylglutamate is formed in mitochondria."
            cards.append(self._card(fact, "structure_function", f"Where {'are' if plural else 'is'} {the(subject)} {made_in.group('part')}?", text))
        if subject_ok and not cards and _BECOMES.search(text) and self._acts(fact):                      # a developmental fate: "The marginal zone remains as a white matter."
            plural = is_plural(subject)
            cards.append(self._card(fact, "concept", f"What {'do' if plural else 'does'} {the(subject)} become?", text))
        if subject_ok and not cards and fact.fact_type in ("relationship", "function", "mechanism", "development", "sequence", "structure_function", "cause_effect") \
                and self._acts(fact):
            plural = is_plural(subject)
            ctype = {"enzyme": "enzyme", "drug": "drug", "gene": "gene_protein", "molecule": "gene_protein"}.get(cat, "mechanism" if fact.fact_type == "mechanism" else "concept")
            cards.append(self._card(fact, ctype, f"What {'do' if plural else 'does'} {the(subject)} do?", text))
        blank = None if self.premium else self.cloze(fact)                      # a fill-in-the-blank card is never made for the Automatic Study Engine
        if blank:
            sentence, answer, _term = blank
            cards.append(self._card(fact, "exam_recall", f"Fill in the blank: {sentence}", f"{answer} - {text}"))
        return cards

    def _acts(self, fact: F.StudyFact) -> bool:
        """The subject term is followed directly by its verb ("Pyruvate kinase catalyzes ...", "Insulin stimulates ..."): a statement of what it does. A verb of state
        ("remains intact", "stays", "seems") says nothing it does."""
        words = fact.statement.split()
        end = F._subject_end(words)
        if end <= 0 or not fact.subtopic:
            return False
        verb = words[end].casefold().strip(",.")
        if verb in _STATE_VERBS and not _BECOMES.search(fact.statement):
            return False
        if verb in ("is", "are", "was", "were", "be", "been"):
            return False                                                       # "is formed", "are going to be quite bushy": a state or something done TO it, not what it does
        head = " ".join(words[:end]).casefold()
        m = _term_pattern(singular_key(fact.subtopic)).search(head)
        if not m or len(head) - m.end() > 2 or len(words) - end < 2:
            return False
        prefix = head[:m.start()].strip()
        if self.premium and prefix and prefix not in ("the", "a", "an") and prefix.split()[0] not in _FRONT_ADVERBIALS:
            return False                                                       # "Defective collagen cross linking is ...": the sentence is about something that merely contains the term
        return True

    # ==================================================================================================================== questions
    def questions(self, chunks: Sequence[Chunk], topic: str, types: Sequence[str], limit: int) -> list[Question]:
        facts = sorted(self.scope(chunks), key=lambda f: -f.importance)
        wanted = [t for t in (types or ()) if t]
        builders: dict[str, Callable[[F.StudyFact], Optional[Question]]] = {
            "mcq": self._mcq, "true_false": self._true_false, "short_answer": self._short_answer, "explain_why": self._explain_why,
            "clinical_vignette": self._sba, "mechanism": self._mechanism_mcq,
        }
        order = [t for t in ("mcq", "short_answer", "mechanism", "clinical_vignette", "explain_why", "true_false") if t in wanted]
        pools: dict[str, list[tuple[F.StudyFact, Question]]] = {t: [] for t in order}
        for fact in facts:
            for qtype in order:
                question = builders[qtype](fact)
                if question is None:
                    continue
                verdict = Q.check_question(question, unknown=self.words.unknown, unresolved=self.unresolved, same_category=self._same_category(fact))
                if verdict.ok and self.premium:
                    verdict = self._premium(verdict, curate.review_question(question, self.index))
                if self.gate.record("question", verdict, f"{question.question} -> {question.answer}"):
                    pools[qtype].append((fact, question))
        out: list[Question] = []
        seen: set[str] = set()
        used: Counter = Counter()
        tf_cap = max(1, round(0.35 * limit))
        while len(out) < limit and any(pools.values()):
            progressed = False
            for qtype in order:
                pool = pools[qtype]
                while pool:
                    fact, question = pool.pop(0)
                    if question.question.casefold() in seen or used[fact.fact_id] >= (1 if qtype == "true_false" else 2):
                        continue
                    if qtype == "true_false" and sum(1 for q in out if q.type == "true_false") >= tf_cap:
                        pool.clear()
                        break
                    if qtype == "true_false" and question.answer == "True" and not any(x[1].answer == "False" for x in pool) \
                            and not any(q.type == "true_false" and q.answer == "False" for q in out) and sum(1 for q in out if q.type == "true_false") >= 3:
                        pool.clear()                                              # no false version to balance them: three "True" statements are enough
                        break
                    if qtype == "true_false":
                        falses = sum(1 for q in out if q.type == "true_false" and q.answer == "False")
                        trues = sum(1 for q in out if q.type == "true_false" and q.answer == "True")
                        want = "False" if falses <= trues else "True"
                        if question.answer != want:
                            other = next((x for x in pool if x[1].answer == want and used[x[0].fact_id] < 1 and x[1].question.casefold() not in seen), None)
                            if other is not None:
                                pool.remove(other)
                                pool.insert(0, (fact, question))
                                fact, question = other
                    seen.add(question.question.casefold())
                    used[fact.fact_id] += 1
                    out.append(question)
                    progressed = True
                    break
                if len(out) >= limit:
                    break
            if not progressed:
                break
        position = {f.fact_id: f.position for f in facts}
        out.sort(key=lambda q: (position.get(q.fact_id, 0), q.id))
        return out

    def _same_category(self, fact: F.StudyFact):
        def same(answer: str, option: str) -> bool:
            a, o = self.term_info(answer), self.term_info(option)
            ca = a.category if a else term_category(answer)
            co = o.category if o else term_category(option)
            return ca == co and ca != "other"
        return same

    def _question(self, fact: F.StudyFact, qtype: str, stem: str, answer: str, explanation: str, options: Optional[list[str]] = None, difficulty: Optional[str] = None) -> Question:
        if options and qtype != "true_false":
            rnd = random.Random(item_id("q", qtype, stem))
            options = list(options)
            rnd.shuffle(options)
        return Question(id=item_id("q", qtype, stem, answer), type=qtype, question=stem, answer=answer, explanation=explanation, options=options or [], sources=self.refs(fact),
                        difficulty=difficulty or fact.difficulty, topic=fact.topic, fact_id=fact.fact_id, high_yield=fact.teacher_emphasis)

    def _says(self, fact: F.StudyFact) -> str:
        return f'The lecture says: "{fact.statement}"'

    def _mcq(self, fact: F.StudyFact) -> Optional[Question]:
        if fact.fact_type in ("mechanism", "cause_effect", "sequence", "development"):
            return None
        return self._wh_mcq(fact, "mcq") or self._cloze_mcq(fact, "mcq", fact.fact_type not in ("mechanism", "clinical_relevance"))

    def _mechanism_mcq(self, fact: F.StudyFact) -> Optional[Question]:
        if fact.fact_type not in ("mechanism", "cause_effect", "sequence", "development"):
            return None
        return self._wh_mcq(fact, "mechanism") or self._cloze_mcq(fact, "mechanism", True)

    def _sba(self, fact: F.StudyFact) -> Optional[Question]:
        return self._cloze_mcq(fact, "clinical_vignette", fact.fact_type == "clinical_relevance")           # a real vignette needs a patient scenario: only a language model writes those

    def _wh_mcq(self, fact: F.StudyFact, qtype: str) -> Optional[Question]:
        """"Which enzyme catalyzes the last step of glycolysis and produces pyruvate and ATP?": the subject of the lecture's own sentence is the answer, the rest of the sentence is the
        question, the wrong options are terms of the same kind from the same lecture. A real question, not a sentence with a hole."""
        if not fact.subtopic or not self.in_head(fact):
            return None
        info = self.term_info(fact.subtopic)
        noun = _WH_NOUN.get(info.category) if info is not None else None
        words = fact.statement.split()
        end = F._subject_end(words)
        if info is None or noun is None or end <= 0 or is_plural(self.display(fact.subtopic)):
            return None
        head = [w for w in words[:end]]
        while head and head[0].casefold() in ("the", "a", "an"):
            head = head[1:]
        term_stems, head_stems = [T.stem(w) for w in T.words(info.display)], [T.stem(w) for w in T.words(" ".join(head))]
        prefix = ""
        if head_stems != term_stems:
            n = len(term_stems)
            if len(head_stems) > n and head_stems[-n:] == term_stems and head[0].casefold() in _FRONT_ADVERBIALS:
                prefix = " ".join(head[:-n])                                       # "In the liver | glucokinase acts as ...": "Which enzyme in the liver acts as ...?"
                prefix = prefix[:1].lower() + prefix[1:]
            else:
                return None                                                      # the sentence is about something that merely contains the term ("Defective collagen cross linking is ...")
        rest = " ".join(words[end:]).rstrip(".!? ")
        first = rest.split()[0].casefold() if rest.split() else ""
        finite = first in _FINITE_AUX or (first.endswith(("s", "ed")) and not first.endswith(("ss", "us", "is")) and len(first) > 3)
        if not finite:
            return None                                                          # "Penicillin | binding proteins are ...": the subject goes on, the "verb" is a participle of the name
        if not 5 <= len(T.words(rest)) <= 26 or _SELF_REFERENCE.search(rest) or _term_pattern(info.key).search(rest) or T.stem(noun.split()[0]) in {T.stem(w) for w in T.words(rest)}:
            return None
        answer = self.display(fact.subtopic)
        wrong = self.distractors(answer, info, fact, 4) or (self.distractors(answer, info, fact, 3) if self.premium else [])      # five options as always; four only for the Automatic Study Engine
        if not wrong:
            return None
        stem = f"Which {noun} {prefix + ' ' if prefix else ''}{rest}?"
        return self._question(fact, qtype, stem, answer, self._says(fact), [answer, *wrong])

    def _cloze_mcq(self, fact: F.StudyFact, qtype: str, allowed: bool) -> Optional[Question]:
        if not allowed or self.premium:
            return None
        blank = self.cloze(fact)
        if not blank:
            return None
        sentence, answer, term = blank
        wrong = self.distractors(answer, term, fact, 4)                                                  # the multiple-choice format has five options
        if not wrong:
            return None
        stem = f"Which term correctly completes this statement? {sentence}"
        return self._question(fact, qtype, stem, answer, self._says(fact), [answer, *wrong])

    def _true_false(self, fact: F.StudyFact) -> Optional[Question]:
        words = T.words(fact.statement)
        if not 8 <= len(words) <= 28:
            return None
        false_version = self._falsify(fact.statement)
        if false_version and int(item_id("tf", fact.fact_id)[-1], 16) % 2 == 0:
            return self._question(fact, "true_false", f"True or false: {false_version}", "False", f"False. {self._says(fact)}", ["True", "False"], "medium")
        return self._question(fact, "true_false", f"True or false: {fact.statement}", "True", f"True. {self._says(fact)}", ["True", "False"], "easy")

    @staticmethod
    def _falsify(statement: str) -> Optional[str]:
        """Swap ONE word for its verified opposite ("anterior" / "posterior", "activates" / "inhibits"): the new statement is false by construction."""
        m = _ANTONYM_RE.search(statement)
        if not m:
            return None
        swapped = _ANTONYM_MAP[m.group(1).casefold()]
        swapped = swapped.capitalize() if m.group()[0].isupper() else swapped
        before = re.search(r"(?<![\w-])(a|an)\s+$", statement[:m.start()], re.IGNORECASE)                        # "a reduced" -> "an elevated", "an anabolic" -> "a catabolic"
        if before is not None:
            article = before.group(1)
            vowel = swapped[:1].casefold() in "aeiou" and not swapped.casefold().startswith(("eu", "uni", "one"))
            fixed = ("an" if vowel else "a")
            fixed = fixed.capitalize() if article[0].isupper() else fixed
            return statement[:before.start(1)] + fixed + " " + swapped + statement[m.end():]
        return statement[:m.start()] + swapped + statement[m.end():]

    def _short_answer(self, fact: F.StudyFact) -> Optional[Question]:
        if not fact.subtopic or fact.fact_type not in ("definition", "function", "location", "derivative", "classification", "structure_function") or not self.in_head(fact):
            return None
        info = self.term_info(fact.subtopic)
        if info is None or F.is_generic_label(info.display, self.known_term):
            return None
        subject = self.display(fact.subtopic)
        the = _the(subject, info.category, info.proper)
        plural = is_plural(subject)
        text = fact.statement
        if fact.fact_type == "definition":
            if re.search(r"\b(?:is|are)\s+(?:called|known\s+as|the\s+so-called)\b", text, re.IGNORECASE) or _EMPTY_PREDICATE.search(text):
                return None
            stem = f"What {'are' if plural else 'is'} {the}?"
        elif fact.fact_type == "location":
            stem = f"Where {'are' if plural else 'is'} {the} located?"
        elif fact.fact_type == "derivative":
            if _DERIVED.search(text):
                stem = f"What {'do' if plural else 'does'} {the} develop from?"
            elif _GIVES_RISE.search(text):
                stem = f"What {'do' if plural else 'does'} {the} give rise to?"
            else:
                return None
        elif fact.fact_type == "classification":
            stem = f"How {'are' if plural else 'is'} {the} divided or classified?"
        else:
            stem = f"What {'is' if not plural else 'are'} the function{'s' if plural else ''} of {the}?"
        return self._question(fact, "short_answer", stem, text, self._says(fact))

    def _explain_why(self, fact: F.StudyFact) -> Optional[Question]:
        if fact.fact_type != "cause_effect":
            return None
        m = re.search(r"\b(?P<verb>because|due to|as a result of)\b", fact.statement, re.IGNORECASE)
        if not m:
            return None
        effect, cause = fact.statement[:m.start()].strip(" ,;"), fact.statement[m.end():].strip(" ,;.")
        if not (4 <= len(T.words(effect)) <= 18 and 3 <= len(T.words(cause)) <= 18) or F._PRONOUN_SUBJECT.match(effect) or F._PRONOUN_SUBJECT.match(cause):
            return None
        return self._question(fact, "explain_why", f"Why is this the case: {effect}?", T.title_case(cause) + ".", self._says(fact), difficulty="hard")


def _the(term: str, category: str = "structure", proper: bool = False) -> str:
    """"the neural tube" for a structure or a cell; processes, diseases, molecules and genes take no article; acronyms and capitalised names are left as they are."""
    if not term or proper or term[:1].isupper() and (term.isupper() or any(c.isupper() for c in term[1:])):
        return term
    low = term[:1].lower() + term[1:]
    named_system = (T.words(low) or [""])[-1] in ("cycle", "pathway", "shunt", "cascade", "chain", "system", "axis", "loop", "barrier", "complex", "pump")
    if category not in ("structure", "cell") and not named_system:
        return low
    if low.casefold().startswith(("the ", "a ", "an ")):
        return low
    return "the " + low


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:]


def _word_counts(index) -> dict[str, int]:
    counts: Counter = Counter()
    for chunk in index.chunks:
        for w in T.words(chunk.text):
            counts[w] += 1
    return dict(counts)


def study_of(ctx: StudyContext) -> FactStudy:
    """The :class:`FactStudy` of a run (built once, kept in the run's cache)."""
    cached = ctx.cache.get("factstudy")
    if cached is None or cached.index is not ctx.index:
        cached = FactStudy(ctx)
        ctx.cache["factstudy"] = cached
    return cached
