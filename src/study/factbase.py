"""The fact base of a lecture: every sentence of the source turned into a validated, atomic, source-grounded :class:`~.facts.StudyFact`.

    source chunk -> sentences (with the seconds they were said) -> cleaned -> split into single propositions -> validated -> typed fact
                 -> terms (what each fact is about) -> importance (lecturer emphasis, slides, term weight) -> near-duplicates merged

Summary, key concepts, flashcards, questions and Study Mode are all made from the SAME facts (``factitems.py``), so they cannot contradict each other and a noisy sentence of
the recording never becomes study material only because it was said. Everything here is rule based and offline; nothing is added from outside the lecture.
Rejected sentences are counted by reason (``FactBase.rejected``): the student - and the audit - can see what the filter removed.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

from . import facts as F
from . import recall as R
from . import text as T
from .index import DocumentIndex
from .models import SOURCE_PDF, SOURCE_TRANSCRIPT, Chunk, SourceRef

# ======================================================================================================================== sentences with positions
_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+(?=[\"“'(\[]?[A-ZÀ-ÖØ-ÞİĞŞÇÖÜ0-9])")


def sentences_with_offsets(text: str) -> list[tuple[str, int, int]]:
    """``(sentence, first character, last character)`` for every sentence of ``text`` (fragments of fewer than 3 words are dropped)."""
    out: list[tuple[str, int, int]] = []
    at = 0
    for part in _SENT_SPLIT.split(text):
        start = text.find(part, at)
        if start < 0:
            start = at
        end = start + len(part)
        at = end
        if len(T.words(part)) >= 3:
            out.append((part.strip(), start, end))
    return out


def span_seconds(chunk: Chunk, first: int, last: int) -> tuple[Optional[float], Optional[float]]:
    """The seconds of the recording that carry the characters ``first:last`` of the chunk text (the segments that overlap them)."""
    segs = [s for s in (chunk.segs or []) if len(s) >= 4 and s[2] < last and first < s[3]]
    if segs:
        return float(min(s[0] for s in segs)), float(max(s[1] for s in segs))
    return chunk.start, chunk.end


def span_confidence(chunk: Chunk, first: int, last: int) -> tuple[Optional[float], set[str]]:
    """The lowest segment confidence and the content words the recogniser doubted, over the segments that carry the characters ``first:last``."""
    confs, low = [], set()
    for s in chunk.segs or []:
        if len(s) >= 4 and s[2] < last and first < s[3]:
            if len(s) > 4 and s[4] is not None:
                confs.append(float(s[4]))
            if len(s) > 5 and s[5]:
                low.update(str(w) for w in s[5])
    return (min(confs) if confs else None), low


MIN_SEGMENT_CONFIDENCE = 0.55


# ======================================================================================================================== what kind of thing a term is
_CATEGORY_HEADS = {
    "cell": ("cell", "cells", "neuron", "neurons", "glia", "cyte", "cytes", "blast", "blasts", "neuroblast", "fibroblast", "macrophage", "lymphocyte", "astrocyte", "microglia"),
    "structure": ("tube", "fold", "folds", "plate", "crest", "layer", "layers", "nerve", "nerves", "artery", "vein", "muscle", "gland", "duct", "tract", "nucleus", "ganglion",
                  "bone", "membrane", "cord", "vesicle", "vesicles", "zone", "organ", "tissue", "cavity", "canal", "arch", "pouch", "sinus", "cortex", "lobe", "gyrus", "sulcus",
                  "ventricle", "ligament", "valve", "vessel", "septum", "node", "somite", "somites", "notochord", "ectoderm", "mesoderm", "endoderm", "epithelium", "plexus",
                  "fibre", "fiber", "capsule", "lamina", "matrix", "wall", "border", "apparatus", "groove", "ridge", "bud", "disc", "pore", "neuropore", "lumen", "mantle",
                  "horn", "horns", "column", "columns", "root", "roots", "fissure", "foramen", "funiculus", "commissure", "lemniscus", "pole", "eminence", "vesicle",
                  "axon", "axons", "dendrite", "dendrites", "synapse", "synapses", "soma", "sheath", "hillock", "bouton", "boutons", "spine", "spines"),
    "disease": ("syndrome", "disease", "disorder", "deficiency", "failure", "infarction", "carcinoma", "lymphoma", "tumor", "tumour", "cancer", "infection", "anemia", "palsy",
                "dystrophy", "hernia", "fracture", "necrosis", "ischemia", "fever", "defect", "defects", "malformation", "atrophy", "hypertrophy", "dysplasia", "stenosis"),
    "process": ("formation", "development", "migration", "fusion", "closure", "differentiation", "proliferation", "induction", "signaling", "signalling", "transport", "synthesis",
                "metabolism", "regulation", "activation", "inhibition", "secretion", "absorption", "division", "mitosis", "meiosis", "replication", "transcription", "translation",
                "degradation", "reaction", "cycle", "pathway", "invagination", "elevation", "folding", "neurulation", "gastrulation", "patterning", "growth", "apoptosis"),
    "molecule": ("factor", "hormone", "protein", "receptor", "ligand", "kinase", "acid", "peptide", "toxin", "antibody", "cytokine", "transporter", "channel", "enzyme", "collagen",
                 "glycoprotein", "lipid", "steroid", "vitamin"),
}
_ENZYME = re.compile(r"ases?$", re.IGNORECASE)
_DRUG = re.compile(r"(?:olol|pril|sartan|statin|mab|cillin|mycin|azole|pine|zepam|oxacin|cycline|barbital|caine|dipine|prazole|vir|tidine|setron|triptan|gliptin|glitazone|mide)$",
                   re.IGNORECASE)
_DISEASE_END = re.compile(r"(?:itis|oma|pathy|emia|aemia|algia|osis|plegia|penia|uria|trophy|plasia)$", re.IGNORECASE)
_NOT_DISEASE_OSIS = frozenset("mitosis meiosis apoptosis phagocytosis exocytosis endocytosis osmosis symbiosis metamorphosis diagnosis prognosis synthesis analysis".split())


def fine_class(term: str, category: str) -> str:
    """A finer class than the category, for choosing look-alike wrong answers: processes by their ending (-lysis / -genesis / -ation ...), diseases by their head word."""
    words = T.words(term)
    head = words[-1] if words else ""
    if category == "process":
        for ending in ("lysis", "genesis", "synthesis", "ation", "ization", "osis", "ion"):
            if head.endswith(ending):
                return "process:" + ending
        return "process:" + head
    if category == "disease":
        for ending in ("syndrome", "disease", "deficiency", "disorder", "anemia", "anaemia", "itis", "oma", "pathy", "emia", "osis"):
            if head.endswith(ending):
                return "disease:" + ending
        return "disease:" + head
    return category


def term_category(term: str) -> str:
    """'cell' | 'structure' | 'disease' | 'process' | 'molecule' | 'enzyme' | 'drug' | 'gene' | 'other' - from the head word and the ending (no database)."""
    words = T.words(term)
    if not words:
        return "other"
    head = words[-1]
    raw = term.strip()
    if len(words) == 1 and re.fullmatch(r"[A-Z][A-Z0-9]{1,7}|[a-z]{1,3}[A-Z][A-Za-z0-9]*", raw):
        return "gene"
    if _ENZYME.search(head) and len(head) > 5 and head not in ("disease", "diseases", "increase", "decrease", "release", "base", "phase", "database"):
        return "enzyme"
    if _DRUG.search(head) and len(head) > 6:
        return "drug"
    if (_DISEASE_END.search(head) and head not in _NOT_DISEASE_OSIS) or head in _CATEGORY_HEADS["disease"]:
        return "disease"
    for category in ("cell", "structure", "process", "molecule"):
        if head in _CATEGORY_HEADS[category] or any(head.endswith(h) for h in _CATEGORY_HEADS[category] if len(h) >= 5 and category in ("cell",)):
            return category
    if re.search(r"(?:cyte|blast|glia)s?$", head):
        return "cell"
    if re.search(r"(?:tion|sion|genesis|lysis|ation)s?$", head):
        return "process"
    return "other"


# ======================================================================================================================== terms
_ORGANS = frozenset("liver muscle brain kidney heart skin blood lung bone intestine stomach pancreas spleen tissue cell cells".split())
_ENZYME_COMPOUND = re.compile(r"(?P<before>(?:[\w,;:-]+\s+){1,2})(?P<head>[A-Za-z-]{4,}ases?)\b", re.IGNORECASE)
_COMPOUND = re.compile(r"(?P<before>(?:[\w,;:-]+\s+){1,4})(?P<head>syndrome|disease|deficiency|disorder|effect|anemia|anaemia|carcinoma|lymphoma|infarction|dystrophy|palsy)\b", re.IGNORECASE)


def _surface(gram: str, body: str) -> str:
    """The words of ``gram`` as they are written in ``body`` (case and hyphens kept)."""
    found = list(re.finditer(re.escape(gram).replace(r"\ ", r"[\s-]+"), body, re.IGNORECASE))
    if not found:
        return gram
    inner = [m for m in found if m.start() > 0]
    return (inner[0] if inner else found[0]).group(0)


_ACRONYM = re.compile(r"\b(?:[A-Z]{2,}[A-Za-z0-9+-]*|[a-z]{1,3}[A-Z][A-Za-z0-9]*|[A-Z][a-z]?\d[A-Za-z0-9]*)\b")
_MEDICAL_WORD = re.compile(
    r"\b[A-Za-z][A-Za-z-]{3,}(?:ase|osis|itis|emia|aemia|pathy|plasia|trophy|genesis|lysis|oma|iasis|uria|penia|cytosis|stasis|mycin|cillin|statin|olol|pril|sartan|azole|mab|nib|"
    r"cyte|blast|derm|plate|tube|crest|fold|zone|ganglion|nucleus|cortex|notochord|somite|neuropore|epithelium|gyrus|sulcus|ventricle)s?\b", re.IGNORECASE)
_NOT_MEDICAL = frozenset("increase decrease disease release purchase database diseases increases decreases releases showcase".split())
_GENERIC = frozenset("lecture slide slides today thing things part parts type types kind case example examples point points time way ways question questions answer number process "
                     "level levels amount form side lot bit stuff something anything everything one ones".split())


@dataclass
class TermInfo:
    key: str
    display: str
    kind: str                                    # enzyme | acronym | medical | phrase
    category: str = "other"
    proper: bool = False                         # always written with a capital letter in the lecture (an eponym, a name): never lower-cased in a question
    count: int = 0
    topics: int = 0
    score: float = 0.0
    facts: list[str] = field(default_factory=list)       # ids of the facts that mention it
    where: set = field(default_factory=set)              # topics of those facts


_ADJECTIVE_END = re.compile(r"(?:al|ic|ous|ive|ary|ory|ar|oid|ile)$", re.IGNORECASE)
_NOUN_EXCEPTIONS = frozenset({"canal", "signal"})


def is_plural(term: str) -> bool:
    """The last word of the term is a plural ("neural folds", "neurons") - not "glycolysis" or "syndrome"."""
    words = T.words(term)
    return bool(words) and singular(words[-1]) != words[-1].casefold()


def singular(word: str) -> str:
    low = word.casefold()
    if low.endswith("ies") and len(low) > 5:
        return low[:-3] + "y"
    if low.endswith(("ses", "xes", "ches", "shes")) and len(low) > 5 and not low.endswith("sses"):
        return low[:-2]
    if low.endswith("s") and not low.endswith(("ss", "us", "is", "ics")) and len(low) > 4:
        return low[:-1]
    return low


def singular_key(term: str) -> str:
    """Key of a term that ignores the number: "neural folds" and "neural fold" are one concept."""
    parts = term.casefold().split()
    if parts:
        parts[-1] = singular(parts[-1])
    return " ".join(parts)


def term_pattern_text(term: str) -> str:
    parts = re.split(r"(\s+)", term)
    head = parts[-1] if parts else term
    if head.endswith("y") and len(head) > 3:
        parts[-1] = re.escape(head[:-1]) + r"(?:y|ies)"
        return "".join(p if i != len(parts) - 1 else p for i, p in enumerate(parts[:-1])) and "".join(re.escape(x) if i != len(parts) - 1 else x for i, x in enumerate(parts))
    return re.escape(term) + r"(?:s|es)?"


def _term_pattern(term: str) -> re.Pattern:
    return re.compile(r"(?<![\w-])" + term_pattern_text(term) + r"(?![\w-])", re.IGNORECASE)


def is_adjective_label(term: str) -> bool:
    """A single adjective ("epithelial", "neural") is not a concept a card can ask about."""
    words = T.words(term)
    return len(words) == 1 and bool(_ADJECTIVE_END.search(words[0])) and words[0] not in _NOUN_EXCEPTIONS and not _ENZYME.search(words[0])


# ======================================================================================================================== the fact base
@dataclass
class _Atom:
    chunk: Chunk
    text: str                                    # one cleaned proposition
    sentence: str                                # the sentence as it stands in the source
    emphasis: bool
    start: Optional[float]
    end: Optional[float]
    doubted: set


#: problems that a shorter clause of the same sentence may not have
RESCUABLE = frozenset({"too long", "two statements run together", "several statements in one sentence", "pronoun subject after an introductory phrase", "starts with a pronoun",
                       "broken grammar", "speaker's voice", "refers to the picture", "repeated words", "repeated phrase", "dependent clause only"})
_RELATIVE = re.compile(r"(?P<left>[^;]{6,200}?),\s+(?:which|that)\s+(?P<pred>(?:is|are|was|were|contains?|forms?|gives?|produces?|surrounds?|lines?|covers?|develops?|differentiates?|"
                       r"becomes?|has|have|carries|carry|releases?|secretes?|supplies|supply)\b[^,;]{6,120})", re.IGNORECASE)
_RUN_ON = re.compile(r",\s+(?=(?:The|A|An|In|At|On|During|After|Before|These|Those)\s)")         # "..., The folic acid supplementation was shown ...": a lost full stop
_COORDINATE = re.compile(r",?\s+(?:and|but|so)\s+(?=(?:the|a|an|these|those|some|each|all)\s)|;\s+|,\s+(?:and|but|so)\s+", re.IGNORECASE)


@dataclass
class FactBase:
    facts: list[F.StudyFact] = field(default_factory=list)
    terms: dict[str, TermInfo] = field(default_factory=dict)
    rejected: Counter = field(default_factory=Counter)
    examples: list[dict] = field(default_factory=list)          # a few rejected sentences with the reason (for the audit)
    sentences_seen: int = 0
    rescued: int = 0                                            # facts that are one clause of a sentence that failed as a whole
    medically_questioned: int = 0                               # statements the medical engine (v4, shadow) doubted: not used
    recovered: Counter = field(default_factory=Counter)         # statements made usable by recall.py, by rule
    by_id: dict[str, F.StudyFact] = field(default_factory=dict)

    def get(self, fact_id: str) -> Optional[F.StudyFact]:
        return self.by_id.get(fact_id)

    def of_topic(self, topic: str) -> list[F.StudyFact]:
        return [f for f in self.facts if not topic or f.topic == topic]

    def term(self, name: str) -> Optional[TermInfo]:
        return self.terms.get(singular_key(name))

    def known(self, name: str) -> bool:
        low = singular_key(name)
        return low in self.terms or any(low == k or low in k or k in low for k in self.terms if min(len(low), len(k)) >= 5)

    def reject_report(self) -> dict:
        tiers = Counter(f.tier for f in self.facts)
        return {"sentences": self.sentences_seen, "facts": len(self.facts), "high": tiers.get("HIGH", 0), "medium": tiers.get("MEDIUM", 0), "clauses_rescued": self.rescued,
                "recovered": dict(self.recovered.most_common()), "rejected": dict(self.rejected.most_common()), "examples": self.examples}


_COMPLEMENT = re.compile(r"\b(?:of|in|on|for|from|by|to|at|within|between|among|during|under|through)\b", re.IGNORECASE)
_LEADING_PHRASE = re.compile(r"^(?:in|on|at|within|during|under|through|for|with|from|by|after|before|among|between)\b", re.IGNORECASE)
_PATIENTS_WITH = re.compile(r"^(?:patients|children|individuals|people|infants|women|men)\s+with\s+", re.IGNORECASE)


def subject_term(statement: str, terms: Sequence[TermInfo]) -> Optional[TermInfo]:
    words = statement.split()
    end = F._subject_end(words)
    if end <= 0:
        return None
    head = " ".join(words[:end])
    if _PATIENTS_WITH.match(head):
        head = _PATIENTS_WITH.sub("", head, count=1)
    elif _LEADING_PHRASE.match(head):                                           # "In the liver glucokinase has ...": the subject is the term closest to the verb
        cands = [(m.end(), len(t.key), t) for t in terms if (m := _term_pattern(t.key).search(head))]
        return max(cands, key=lambda c: (c[0], c[1]))[2] if cands else None
    low = head.casefold()
    best: Optional[tuple[int, int, TermInfo]] = None
    for t in terms:
        m = _term_pattern(t.key).search(low)
        if not m:
            continue
        if _COMPLEMENT.search(low[:m.start()]):
            continue                                                            # "hydroxylation of PROLINE": a complement, not the subject
        if best is None or (m.start(), -len(t.key)) < (best[0], -best[1]):
            best = (m.start(), len(t.key), t)
    return best[2] if best else None


def medical_validation(statements: Sequence[str]) -> Optional[set[int]]:
    """Indices of the statements in which the v4 medical engine (shadow) would change or question a medical term: the recogniser probably wrote that term wrongly.
    None when the knowledge base is not installed (the statements are then judged by the word lists alone)."""
    try:
        from ..medical.corrector import SegmentView
        from ..medical_knowledge.runtime.engine import V4Engine

        kb = F.default_knowledge()
        if kb is None or not statements:
            return None
        segments = [SegmentView(i, i * 6.0, i * 6.0 + 5.5, text, "en", []) for i, text in enumerate(statements)]
        result = V4Engine(kb).run(segments)
        return {c.segment for c in list(result.corrections) + list(result.uncertain)}
    except Exception:                                            # noqa: BLE001 - the validation is an extra safeguard, never a reason to fail
        return None


class FactBuilder:
    """Builds the :class:`FactBase` of an index."""

    MAX_FACT_WORDS = 32

    def __init__(self, index: DocumentIndex, language: str = "en") -> None:
        self.index = index
        self.language = language
        counts: Counter = Counter()
        for chunk in index.chunks:
            for w in T.words(chunk.text):
                counts[w] += 1
        known_extra = list(getattr(index, "glossary", [])) + list(getattr(index, "lecture_terms", []))
        self.counts = dict(counts)
        self.words = F.WordKnowledge(extra=known_extra, counts=dict(counts))
        self.unresolved = list(getattr(index, "unresolved", []))
        self.glossary = {g.casefold() for g in known_extra}
        self._forced: set[str] = set()
        self._lexicon = getattr(self.words, "_index", None)
        self._lex_cache: dict[str, bool] = {}
        self.slides = {c.id: frozenset(T.stem(w) for w in T.content_words(c.text)) for c in index.of_source(SOURCE_PDF)}
        self.slide_text = " ".join(c.text for c in index.of_source(SOURCE_PDF)).casefold()
        self.titles = " ".join(c.title for c in index.chunks if c.title).casefold()

    # -------------------------------------------------------------------------------------------------------------------- sentences -> facts
    def build(self) -> FactBase:
        base = FactBase()
        atoms: list[_Atom] = []
        for chunk in self.index.chunks:
            units = self._pdf_sentences(chunk) if chunk.source == SOURCE_PDF else [(s, a, b) for s, a, b in sentences_with_offsets(chunk.text)]
            for sentence, first, last in units:
                base.sentences_seen += 1
                start, end = (chunk.start, chunk.end) if chunk.source == SOURCE_PDF else span_seconds(chunk, first, last)
                cleaned = F.clean_sentence(sentence)
                if not cleaned.text:
                    self._reject(base, "empty after cleaning", sentence)
                    continue
                conf, doubted = (None, set()) if chunk.source == SOURCE_PDF else span_confidence(chunk, first, last)
                if conf is not None and conf < MIN_SEGMENT_CONFIDENCE:
                    self._reject(base, "low recogniser confidence", sentence)
                    continue
                for piece in F.split_atomic(cleaned.text):
                    atoms.append(_Atom(chunk, F.demonstrative_to_article(piece), sentence, cleaned.emphasis, start, end, doubted))
        self._terms(base, [a.text for a in atoms], [a.chunk for a in atoms])
        raw: list[tuple[_Atom, str, str]] = []                                      # atom, statement, soft doubts
        validated: set[int] = set()
        for k, atom in enumerate(atoms):
            ok, reason, soft = self._validate(atom.text, atom.chunk, atom.doubted)
            if ok:
                raw.append((atom, atom.text, soft))
                validated.add(k)
                continue
            recovered = self._recover(k, atom, atoms, reason, base, validated)
            if recovered is not None:
                raw.append(recovered)
                validated.add(k)
                base.recovered[recovered[2].rsplit("|", 1)[-1]] += 1
                continue
            rescued = 0
            if reason in RESCUABLE or reason.startswith("recogniser doubted"):            # a clause that does not contain the doubted word is validated on its own
                for clause in self._clauses(atom.text, base):
                    ok2, _reason2, soft2 = self._validate(clause, atom.chunk, atom.doubted)
                    if ok2 and self._about_a_term(clause, base):
                        raw.append((atom, clause, "|".join(x for x in (soft2, "clause of a longer sentence") if x)))
                        rescued += 1
            if rescued:
                base.rescued += rescued
            else:
                self._reject(base, reason, atom.sentence)
        questioned = medical_validation([r[1] for r in raw]) or set()
        base.medically_questioned = len(questioned)
        kept_raw = []
        for k, item in enumerate(raw):
            if k in questioned:
                self._reject(base, "medical term questioned by the medical engine", item[0].sentence)
            else:
                kept_raw.append(item)
        for pos, (atom, statement, soft) in enumerate(kept_raw):
            fact = self._fact(base, pos, atom.chunk, statement, atom.sentence, atom.emphasis, atom.start, atom.end, soft)
            if fact is not None:
                base.facts.append(fact)
        base.facts = self._merge_duplicates(base.facts)
        for fact in base.facts:
            base.by_id[fact.fact_id] = fact
            for term in fact.medical_terms:
                info = base.terms.get(singular_key(term))
                if info is not None and fact.fact_id not in info.facts:
                    info.facts.append(fact.fact_id)
                    info.where.add(fact.topic)
        F.link_related(base.facts)
        return base

    def _validate(self, text: str, chunk: Chunk, doubted: set[str]) -> tuple[bool, str, str]:
        """``(usable, the first problem, soft doubts)`` for one candidate statement. A doubted word rejects it unless the rest of the lecture corroborates the word (recall.py)."""
        heard_doubt = [w for w in doubted if re.search(r"(?<![\w-])" + re.escape(w) + r"(?![\w-])", text, re.IGNORECASE)]
        hard = [w for w in heard_doubt if not R.doubt_is_corroborated(w, self.counts, self.words.unknown)]
        if hard:
            return False, f"recogniser doubted: {hard[0]}", ""
        check = F.sentence_issues(text, unknown=self.words.unknown, unresolved=self.unresolved, max_words=self.MAX_FACT_WORDS,
                                  min_words=5 if chunk.source == SOURCE_PDF else 6)
        if chunk.source == SOURCE_PDF:                                               # slide bullets are terse: a missing verb or pronoun start is normal there
            lenient = ("no verb", "starts with a pronoun", "too short")
            issues = [i for i in check.issues if i not in lenient]
            check = F.Check(not issues, issues, [i for i in check.issues if i in lenient[:2]] + list(check.soft))
        if not check.ok:
            return False, check.issues[0], ""
        corroborated = [f"recogniser doubted, corroborated: {w}" for w in heard_doubt if w.casefold() not in T.STOPWORDS and len(w) >= 4]
        return True, "", "|".join(list(check.soft) + corroborated)

    def _recover(self, k: int, atom: "_Atom", atoms: Sequence["_Atom"], reason: str, base: FactBase, validated: set[int]) -> Optional[tuple["_Atom", str, str]]:
        """A rejected statement made usable by recall.py: the lecturer's lead-in removed, or "It" / "They" replaced by the subject of the previous, VALIDATED sentence of the
        same passage (then the fact cites both sentences). The result is validated again like any statement; ``None`` when nothing applies."""
        if reason in ("speaker's voice", "lecturer talk"):
            text, emphatic = R.strip_lead_in(atom.text)
            if text:
                ok, _why, soft = self._validate(text, atom.chunk, atom.doubted)
                if ok:                                                     # the whole statement minus the lead-in (not a fragment): naming a term anywhere is enough
                    return replace(atom, emphasis=atom.emphasis or emphatic), text, "|".join(x for x in (soft, "lead-in removed") if x)
        if reason == "starts with a pronoun" and k > 0 and (k - 1) in validated:
            prev = atoms[k - 1]
            near = prev.start is not None and atom.start is not None and atom.start - (prev.end if prev.end is not None else prev.start) <= 15.0
            if prev.chunk is atom.chunk and near:
                subj = subject_term(prev.text, list(base.terms.values()))
                if subj is not None and not F.is_generic_label(subj.display, base.known):
                    article = "The" if subj.category in ("structure", "cell") and not subj.proper else ""
                    text = R.resolve_pronoun(atom.text, subj.display, is_plural(subj.display), article)
                    if text:
                        ok, _why, soft = self._validate(text, atom.chunk, atom.doubted | prev.doubted)
                        if ok:
                            merged = _Atom(atom.chunk, text, f"{prev.sentence} {atom.sentence}", atom.emphasis, prev.start, atom.end, atom.doubted)
                            return merged, text, "|".join(x for x in (soft, "pronoun resolved from the previous sentence") if x)
        return None

    # -------------------------------------------------------------------------------------------------------------------- clauses of a long spoken sentence
    def _clauses(self, text: str, base: FactBase) -> list[str]:
        """Single clauses of a sentence that is too long or too tangled: ``X, which is Y`` -> ``The X is Y``, and the pieces of ``A is B, and C is D``. Each piece is validated on its own."""
        body = text.rstrip(".!? ")
        out: list[str] = []
        for m in _RELATIVE.finditer(body):
            anchor = self._anchor(m.group("left"), base)
            if anchor:
                out.append(f"{anchor} {m.group('pred')}")
        for part in [p for piece in _RUN_ON.split(body) for p in _COORDINATE.split(piece)]:
            part = part.strip(" ,;")
            part = re.sub(r"^(?:and|but|so|then|because|while|whereas)\s+", "", part, flags=re.IGNORECASE)
            if part and part != body and len(T.words(part)) >= 6:
                out.append(part)
        cleaned: list[str] = []
        for clause in out:
            c = F.clean_sentence(clause).text
            c = F.demonstrative_to_article(F.fix_restatement(c)) if c else ""
            if c and c not in cleaned and c.rstrip(".") != body:
                cleaned.append(c)
        return cleaned

    def _anchor(self, left: str, base: FactBase) -> str:
        """The term that ends ``left`` (the noun a relative clause belongs to), with its article when it is a structure or a cell: 'The central canal'."""
        tail = left.rstrip(" ,")
        for info in sorted(base.terms.values(), key=lambda t: -len(t.key)):
            if len(info.key) < 5:
                continue
            m = re.search(r"(?<![\w-])(?:the\s+|a\s+|an\s+)?(" + term_pattern_text(info.key) + r")\s*$", tail, re.IGNORECASE)
            if m:
                surface = m.group(1)
                if info.category in ("structure", "cell") and not info.proper:
                    return "The " + surface
                return surface[:1].upper() + surface[1:]
        return ""

    @staticmethod
    def _about_a_term(clause: str, base: FactBase) -> bool:
        """A rescued clause is only trusted when it BEGINS with a known term (one article and up to two modifiers before it): it is about something."""
        for t in base.terms.values():
            if len(t.key) >= 4 and re.match(r"(?:(?:the|a|an|these|those)\s+)?(?:[\w-]+\s+){0,2}?" + term_pattern_text(t.key) + r"(?![\w-])", clause, re.IGNORECASE):
                return True
        return False

    @staticmethod
    def _pdf_sentences(chunk: Chunk) -> list[tuple[str, int, int]]:
        from .extractive import _pdf_units

        return [(s, 0, len(s)) for s in _pdf_units(chunk)]

    @staticmethod
    def _reject(base: FactBase, reason: str, sentence: str) -> None:
        base.rejected[reason] += 1
        if len(base.examples) < 40:
            base.examples.append({"reason": reason, "sentence": sentence[:200]})

    # -------------------------------------------------------------------------------------------------------------------- terms
    def _lexicon_has(self, gram: str) -> bool:
        hit = self._lex_cache.get(gram)
        if hit is None:
            try:
                hit = bool(self._lexicon.exact(gram))
            except Exception:                                            # pragma: no cover - the vocabulary is optional for the study side
                hit = False
            self._lex_cache[gram] = hit
        return hit

    def _candidates(self, statements: Sequence[str]) -> tuple[Counter, dict[str, str], dict[str, str], dict[str, bool]]:
        found: Counter = Counter()
        display: dict[str, Counter] = {}
        kinds: dict[str, str] = {}
        lower_seen: dict[str, bool] = {}

        def note(raw: str, kind: str, force: bool = False) -> None:
            low = raw.casefold()
            key = singular_key(low)
            if len(key) < 3 or key in T.STOPWORDS or key in _GENERIC or "'" in key or chr(0x2019) in key:
                return
            found[key] += 1
            display.setdefault(key, Counter())[raw] += 1
            kinds.setdefault(key, kind)
            lower_seen[key] = lower_seen.get(key, False) or raw == raw.lower()
            if force:
                self._forced.add(key)

        for body in statements:
            for m in _ACRONYM.finditer(body):
                if m.start() == 0 and not (sum(c.isupper() for c in m.group()) >= 2 or any(c.isdigit() for c in m.group())):
                    continue
                note(m.group(), "acronym")
            for m in _MEDICAL_WORD.finditer(body):
                w = m.group()
                if w.casefold() in _NOT_MEDICAL:
                    continue
                note(w, "enzyme" if w.casefold().rstrip("s").endswith("ase") else "medical")
            tokens = T.words(body)
            for size in (3, 2):
                for i in range(len(tokens) - size + 1):
                    gram = tokens[i:i + size]
                    if all(len(w) >= 3 and w not in T.STOPWORDS and w not in _GENERIC and w not in F.VERB_FORMS and not w[0].isdigit() for w in gram):
                        note(" ".join(gram), "phrase")
            if self._lexicon is not None:                                # terms of the medical vocabulary (the production Medical Expert's lists)
                for size in (3, 2, 1):
                    for i in range(len(tokens) - size + 1):
                        gram = " ".join(tokens[i:i + size])
                        if len(gram) >= 5 and gram not in T.STOPWORDS and gram not in _GENERIC and self._lexicon_has(gram):
                            note(_surface(gram, body), "medical")
            for m in _ENZYME_COMPOUND.finditer(body):                    # "Lysyl hydroxylase", "proline hydroxylase", "triose phosphate isomerase"
                before = m.group("before").split()
                kept = []
                for w in reversed(before):
                    plain = w.casefold().strip(",;:")
                    if w != w.strip(",;:") or len(plain) < 3 or plain in T.STOPWORDS or plain in _GENERIC or plain in F.VERB_FORMS or plain in _ORGANS:
                        break
                    if not (w[:1].isupper() or re.search(r"(?:yl|ate|ose|ine|ol|ide|ic|in)$", plain) or "-" in plain or self._lexicon_has(plain)):
                        break
                    kept.append(w)
                    if len(kept) == 2:
                        break
                if kept:
                    note(" ".join(reversed(kept)) + " " + m.group("head"), "enzyme", force=True)
            for m in _COMPOUND.finditer(body):                           # "Pyruvate kinase deficiency", "Vitamin C deficiency", "Ehlers Danlos syndrome", "Warburg effect"
                words_before = m.group("before").split()
                kept: list[str] = []
                for w in reversed(words_before):
                    plain = w.casefold().strip(",;:")
                    if ((len(plain) < 2 and not w.isupper()) or plain in T.STOPWORDS or plain in _GENERIC or (plain in F.VERB_FORMS and not w[:1].isupper()) or w != w.strip(",;:")):
                        break
                    if not (w[:1].isupper() or len(plain) == 1 or self._lexicon_has(plain) or _ENZYME.search(plain) or plain in self.glossary):
                        break
                    kept.append(w)
                    if len(kept) == 3:
                        break
                if kept:
                    note(" ".join(reversed(kept)) + " " + m.group("head"), "medical", force=True)
            for g in self.glossary:                                      # the lecture's own vocabulary (slides, glossary): always a candidate when it is said
                if len(g) >= 4 and g in body.casefold():
                    note(g, "medical")
        return found, {k: c.most_common(1)[0][0] for k, c in display.items()}, kinds, lower_seen

    @staticmethod
    def _case_of(key: str, show: str, statements: Sequence[str]) -> tuple[bool, str]:
        """``(is a name, how to write it)``: a name has a capital letter wherever it stands in the MIDDLE of a sentence (a capital at the start of a sentence says nothing)."""
        pattern = _term_pattern(key)
        capital, lower = Counter(), Counter()
        for text in statements:
            for m in pattern.finditer(text):
                if m.start() == 0:
                    continue
                (capital if m.group(0)[:1].isupper() else lower)[m.group(0)] += 1
        if capital and not lower:
            return True, capital.most_common(1)[0][0]
        if lower:
            return False, lower.most_common(1)[0][0]
        names = any(w[:1].isupper() for w in show.split()[1:]) or (show.isupper() and len(show) > 1)               # 'Ehlers Danlos syndrome' said only at the start of sentences
        return names, (show[:1].lower() + show[1:] if show[:1].isupper() and not any(c.isupper() for c in show[1:]) else show)

    def _terms(self, base: FactBase, statements: Sequence[str], chunks: Sequence[Chunk]) -> None:
        found, display, kinds, lower_seen = self._candidates(statements)
        total_topics = max(len(self.index.topics), 1)
        n = len(statements)
        min_count = 2 if n > 30 else 1
        scored: list[TermInfo] = []
        for key, count in found.items():
            kind = kinds[key]
            in_vocab = key in self._forced or key in self.glossary or key in self.slide_text or key in self.titles or (self._lexicon is not None and len(key) >= 5 and self._lexicon_has(key))
            floor = min_count if kind != "phrase" else max(min_count, 3 if n > 60 else 2)
            if in_vocab:
                floor = 1
            if count < floor:
                continue
            show = display[key]
            if F.is_generic_label(show, lambda t: t.casefold() in self.glossary or t.casefold() in self.slide_text) or is_adjective_label(show):
                continue
            pattern = _term_pattern(key)
            hit = [i for i, s in enumerate(statements) if pattern.search(s)]
            if not hit:
                continue
            topics = len({chunks[i].topic for i in hit})
            score = math.log(1 + len(hit)) * (1 + 0.5 * topics / total_topics)
            score *= {"enzyme": 1.5, "acronym": 1.3, "medical": 1.4, "phrase": 1.0}[kind]
            for bonus, flag in ((1.5, key in self.slide_text), (1.5, key in self.titles), (1.4, key in self.glossary), (1.2, kind == "phrase" and len(key.split()) == 3)):
                if flag:
                    score *= bonus
            if kind == "phrase" and not in_vocab and term_category(show) == "other" and len(hit) < 3:
                continue                                                 # an ordinary two-word phrase said twice names no concept
            proper, shown = self._case_of(key, show, [statements[i] for i in hit])
            scored.append(TermInfo(key, shown, kind, term_category(shown), proper and kind != "acronym", len(hit), topics, score))
        scored.sort(key=lambda t: t.score, reverse=True)
        kept: list[TermInfo] = []
        cap = max(60, min(400, 2 * n))
        for term in scored:
            if any(term.key != other.key and term.key in other.key and other.count >= term.count * 0.7 for other in kept):
                continue                                                 # "fatty acid" inside the stronger "fatty acid synthesis"
            kept.append(term)
            if len(kept) >= cap:
                break
        base.terms = {t.key: t for t in kept}

    # -------------------------------------------------------------------------------------------------------------------- one fact
    def _fact(self, base: FactBase, pos: int, chunk: Chunk, statement: str, source: str, emphasis: bool, start: Optional[float], end: Optional[float],
              soft: str) -> Optional[F.StudyFact]:
        low = statement.casefold()
        terms = [t for t in base.terms.values() if _term_pattern(t.key).search(statement)]
        terms.sort(key=lambda t: (low.find(t.key) if low.find(t.key) >= 0 else len(low), -len(t.key)))
        subject = self._subject(statement, terms)
        cues = {"emphasis": T.cue_count(statement, T.EMPHASIS_CUES), "exam": T.cue_count(statement, T.EXAM_CUES), "mechanism": T.cue_count(statement, T.MECHANISM_CUES),
                "cause": T.cue_count(statement, T.CAUSE_EFFECT_CUES), "clinical": T.cue_count(statement, T.CLINICAL_CUES), "comparison": T.cue_count(statement, T.COMPARISON_CUES)}
        base_kind = "concept"
        if _DEFINITION.match(statement.rstrip(".!")):
            base_kind = "definition"
        elif cues["comparison"] and len(terms) >= 2:
            base_kind = "comparison"
        elif cues["clinical"] >= 1 and re.search(r"\b(?:patients?|disease|syndrome|disorder|deficiency|treatment|clinical|symptom|tumou?r|cancer)\b", low):
            base_kind = "clinical"
        elif cues["cause"] and not cues["mechanism"]:
            base_kind = "cause_effect"
        elif cues["mechanism"]:
            base_kind = "mechanism"
        if not terms:
            self._reject(base, "names no medical term", source)
            return None
        fact_type = F.classify_statement(statement, base_kind)
        stems = frozenset(T.stem(w) for w in T.content_words(statement))
        linked = [self.slides[s.id] for s in self.index.linked_slides(chunk) if s.id in self.slides] if chunk.source == SOURCE_TRANSCRIPT else []
        support = max((T.overlap(stems, ls) for ls in linked), default=0.0)
        n_words = len(T.words(statement))
        term_part = min(sum(t.score for t in terms), 6.0) / 3.0
        cue_part = 1.5 * min(cues["emphasis"] + int(emphasis), 2) + 1.0 * min(cues["exam"], 1) + 0.6 * min(cues["mechanism"] + cues["cause"], 2) + 0.4 * min(cues["clinical"], 2)
        length_part = 1.0 if 10 <= n_words <= 30 else (0.3 if n_words <= 38 else -1.0)
        type_part = {"definition": 1.2, "derivative": 1.0, "cause_effect": 1.0, "mechanism": 0.9, "clinical_relevance": 0.9, "classification": 0.8, "function": 0.6, "location": 0.6,
                     "comparison": 0.9}.get(fact_type, 0.3)
        importance = round(term_part + cue_part + 1.5 * support + length_part + type_part, 3)
        difficulty = "easy" if fact_type == "definition" and n_words <= 20 else ("hard" if fact_type in ("cause_effect", "comparison", "mechanism") and n_words > 22 else "medium")
        soft_list = [s for s in soft.split("|") if s]
        back = R.points_back(statement)
        if back:
            soft_list.append(f"points back to an earlier sentence: {back}")
        uncertainty = min(1.0, 0.25 * len(soft_list) + (0.2 if not terms else 0.0))
        if chunk.source == SOURCE_PDF:
            source_type, page = "pdf", chunk.page
        else:
            source_type, page = "transcript", None
        return F.StudyFact(
            fact_id=F.fact_id_of(f"{chunk.id}:{pos}:{statement}"), statement=statement, topic=chunk.topic, subtopic=subject, fact_type=fact_type, importance=importance,
            difficulty=difficulty, source_type=source_type, source_start=start, source_end=end, source_page=page, source_excerpt=source, confidence=round(max(0.3, 1.0 - 0.15 * len(soft_list)), 2),
            medical_terms=[t.display for t in terms], teacher_emphasis=bool(emphasis or cues["emphasis"] or cues["exam"]), uncertainty=round(uncertainty, 2), chunk_id=chunk.id, position=pos,
            subject=subject, object="", tier=R.tier_of("|".join(soft_list), False), doubts=soft_list)

    def _subject(self, statement: str, terms: Sequence[TermInfo]) -> str:
        """The term the statement is ABOUT: the head noun of the subject part of the sentence (before the first verb), never a complement ("Hydroxylation of PROLINE needs ..."):
        a term that stands after of / in / for / from / by is not the subject. After an introductory phrase ("In the liver glucokinase ...") it is the one closest to the verb;
        "Patients with X" is about X. No such term: no subject (the statement can still be blanked, but no 'what is X' question is asked about it)."""
        found = subject_term(statement, terms)
        return found.display if found is not None else ""

    # -------------------------------------------------------------------------------------------------------------------- near-duplicates
    @staticmethod
    def _merge_duplicates(facts: list[F.StudyFact]) -> list[F.StudyFact]:
        kept: list[F.StudyFact] = []
        for fact in sorted(facts, key=lambda f: -f.importance):
            if any(F.near_duplicate(fact.statement, other.statement) for other in kept if other.chunk_id == fact.chunk_id or abs(other.position - fact.position) <= 40):
                continue
            kept.append(fact)
        kept.sort(key=lambda f: f.position)
        return kept


_DEFINITION = re.compile(
    r"^(?P<term>[A-Za-z][\w\-/+'() ]{1,60}?)\s*,?\s+(?P<verb>is defined as|are defined as|refers to|is called|are called|is known as|are known as|stands for|is a|is an|is the|are the|are a|"
    r"denotes|means)\s+(?P<rest>.{12,})$", re.IGNORECASE)
_PRONOUN_START = re.compile(r"^(?:it|this|that|these|those|there|here|we|they|you|i|he|she|which|what|who|when|where|how|why|if)\b", re.IGNORECASE)


def build_factbase(index: DocumentIndex, language: str = "en") -> FactBase:
    return FactBuilder(index, language).build()
