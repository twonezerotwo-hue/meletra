"""Small language helpers of the study engine: sentences, content words, stems and the cue words that mark definitions, mechanisms,
clinical statements and what a lecturer stresses. No external packages; English first, Turkish cues as far as they are cheap."""

from __future__ import annotations

import re
from typing import Iterable, Sequence

from ..language import _EN_STOPWORDS, _TR_STOPWORDS

_WORD_RE = re.compile(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*|\d+(?:[.,]\d+)?", re.UNICODE)
_SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+(?=[\"“'(\[]?[A-ZÀ-ÖØ-ÞİĞŞÇÖÜ0-9])")

STOPWORDS = frozenset(_EN_STOPWORDS | _TR_STOPWORDS | set(
    "so okay ok well now here there just like going gonna let lets know see look yes right really very much many some any "
    "all each other another also still even more most only then than too both such may might must shall do does did done "
    "get got make made one two three first second next last thing things way ways little bit actually basically "
    "um uh you your yours i me my mine he she him her his hers it its our ours us them theirs who whom whose where while "
    "if how why what which would should could can cannot able".split()))

EMPHASIS_CUES = (
    "important", "remember", "note that", "keep in mind", "key point", "key", "crucial", "essential", "you should know",
    "you need to know", "you have to know", "high yield", "high-yield", "don't forget", "do not forget", "exam", "tested",
    "main", "critical", "significant", "special", "pay attention", "take home", "mainly", "the point",
    "önemli", "dikkat", "sınav", "unutmayın", "hatırla",
)
EXAM_CUES = ("exam", "tested", "test", "question", "asked", "you should know", "you need to know", "must know", "sınav", "soru")
DEFINITION_VERBS = (
    r"is defined as", r"are defined as", r"refers to", r"is called", r"are called", r"is known as", r"are known as", r"means",
    r"is the so-called", r"stands for", r"is a", r"is an", r"is the", r"are the", r"are a", r"denotes", r"describes",
    r"denir", r"olarak adlandırılır", r"demektir",
)
MECHANISM_CUES = (
    "activates", "activated", "activation", "inhibits", "inhibited", "inhibition", "phosphorylat", "binds", "binding", "cleav",
    "converted", "converts", "catalyz", "catalys", "triggers", "stimulat", "blocks", "degrad", "synthes", "transport",
    "regulat", "signal", "recruit", "translocat", "ubiquitin", "cofactor", "substrate", "enzyme", "receptor", "produces",
    "releases", "oxidiz", "reduc", "transfer", "attaches", "attached",
)
CAUSE_EFFECT_CUES = (
    "causes", "cause", "leads to", "lead to", "results in", "result in", "due to", "because", "therefore", "as a result",
    "consequently", "so that", "which is why", "that's why", "that is why", "thus", "this will cause", "this causes",
    "neden olur", "yol açar", "sonucunda",
)
CLINICAL_CUES = (
    "patient", "patients", "disease", "syndrome", "deficiency", "treatment", "therapy", "diagnos", "symptom", "clinical",
    "disorder", "cancer", "diabetes", "atherosclerosis", "mutation", "infection", "inflammat", "drug", "inhibitor", "pathology",
    "pathogenesis", "hemolytic", "anemia", "acidosis", "toxicity", "cholesterol", "plaque", "cataract", "tumor",
    "hasta", "hastalık", "tedavi", "tanı",
)
COMPARISON_CUES = (
    "whereas", "in contrast", "compared to", "compared with", "on the other hand", "unlike", "differs from", "versus", " vs ",
    "while ", "as opposed to", "rather than", "the difference", "farklı", "oysa",
)
STRUCTURE_CUES = (
    "composed of", "consists of", "consist of", "domain", "subunit", "structure", "contains", "formed by", "made of",
    "binding site", "active site", "helix", "chain", "complex", "arranged", "attached to", "yapı",
)
PATHWAY_CUES = ("pathway", "cycle", "cascade", "signaling", "signal transduction", "step", "first,", "then ", "finally", "reaction")
DRUG_CUES = ("drug", "inhibitor", "agonist", "antagonist", "blocker", "therapy", "treatment", "statin", "antibiotic", "ilaç")
GENE_PROTEIN_CUES = ("protein", "gene", "kinase", "receptor", "factor", "transcription", "mtor", "ampk", "foxo", "ubiquitin",
                     "collagen", "insulin", "hormone", "peptide", "domain")


def words(text: str) -> list[str]:
    return [w.casefold() for w in _WORD_RE.findall(text)]


def stem(word: str) -> str:
    """Crude English suffix stripping, enough to match "phosphorylated" with "phosphorylation" and "enzymes" with "enzyme"."""
    w = word.casefold()
    for plural, singular in (("ices", "ex"), ("ae", "a"), ("ata", "a"), ("mina", "men"), ("ora", "us"), ("era", "us")):         # Latin plurals: cortices / cortex, foramina / foramen
        if w.endswith(plural) and len(w) - len(plural) >= 3:
            w = w[:-len(plural)] + singular
            break
    for suffix in ("ations", "ation", "ating", "ated", "ates", "ate", "ings", "ing", "ies", "ied", "ly", "ed", "es", "s", "al"):
        if len(w) - len(suffix) >= 4 and w.endswith(suffix):
            w = w[: -len(suffix)]
            break
    return w[:-1] if w.endswith("e") and len(w) > 4 else w                   # capsule / capsules


def content_words(text: str) -> list[str]:
    """Words that carry meaning: no stop words, no tiny words (numbers stay: "mTOR 2", "Table 1" differ)."""
    return [w for w in words(text) if (len(w) >= 3 and w not in STOPWORDS) or w[:1].isdigit()]


def content_stems(text: str) -> set[str]:
    return {stem(w) for w in content_words(text)}


def clean(text: str) -> str:
    """Collapse whitespace and the line breaks of subtitles / PDF text."""
    return re.sub(r"\s+", " ", text.replace("­", "")).strip()


def sentences(text: str) -> list[str]:
    """Sentences of a passage; fragments of fewer than 3 words are dropped."""
    out = []
    for part in _SENTENCE_RE.split(clean(text)):
        part = part.strip()
        if len(words(part)) >= 3:
            out.append(part)
    return out


def has_cue(text: str, cues: Iterable[str]) -> bool:
    low = " " + text.casefold() + " "
    return any(c in low for c in cues)


def cue_count(text: str, cues: Iterable[str]) -> int:
    low = " " + text.casefold() + " "
    return sum(1 for c in cues if c in low)


def overlap(a: Sequence[str] | set[str], b: Sequence[str] | set[str]) -> float:
    """Share of ``a`` that also occurs in ``b`` (0..1)."""
    a_set, b_set = set(a), set(b)
    return len(a_set & b_set) / len(a_set) if a_set else 0.0


def title_case(term: str) -> str:
    """Capitalise the first letter only (keep "mTOR", "DNA", "ATP"): used for headings and card fronts."""
    term = term.strip()
    if not term or (term[0].islower() and any(c.isupper() for c in term[1:])):          # "mTOR", "pH", "hnRNA": already written the way it is
        return term
    return term[0].upper() + term[1:]


def truncate(text: str, limit: int) -> str:
    text = clean(text)
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut.rstrip(",;:- ") + "..."
