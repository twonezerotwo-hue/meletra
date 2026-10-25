"""Optional lecture-slide (PDF) context: extract key medical terms to bias transcription.

No LLM is involved. Terms come from (1) dictionary matches and (2) simple regex candidates
(eponyms, medical suffixes) filtered by frequency. The feature is optional and never raises
into the transcription path - a failure just means "no extra context".
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol

from .vocabulary import VocabularyIndex, normalize_key

log = logging.getLogger(__name__)

_EPONYM = re.compile(
    r"\b([A-Z][a-zà-ÿ]+(?:-[A-Z][a-zà-ÿ]+)?(?:['’]s)?\s+"
    r"(?:disease|syndrome|area|nucleus|tract|law|sign|test|reflex|cells?|gland|ligament|nerve|duct|canal|"
    r"fossa|triangle|plexus|artery|vein|cycle|effect|equation|fibers?|body|layer|membrane|pathway))\b"
)
_SUFFIX_WORD = re.compile(
    r"\b([A-Za-z]{4,}(?:itis|osis|emia|aemia|pathy|ectomy|otomy|ostomy|plasty|oma|genesis|lysis|"
    r"penia|uria|cytosis|plasia|trophy|stasis|algia|megaly|graphy|gram|ase|ine|ide))\b"
)
_COMMON_FALSE_SUFFIX = {
    "the", "this", "these", "that", "case", "base", "phase", "release", "increase", "decrease", "disease",
    "please", "also", "because", "response", "course", "cause", "pause", "purpose", "false", "else",
    "machine", "define", "determine", "outline", "online", "line", "mine", "fine", "decide", "provide",
    "inside", "outside", "side", "wide", "guide", "slide", "divide", "include", "exclude", "alone", "hormone",
    "program", "diagram", "gram", "tone", "time", "name", "type", "share", "score", "more", "before",
}


class ContextProvider(Protocol):
    """Interface for anything that can supply extra context terms (PDF today, others later)."""

    def get_terms(self) -> list[str]:
        ...


@dataclass
class LectureContext:
    source: Optional[Path] = None
    terms: list[str] = field(default_factory=list)
    dictionary_hits: list[str] = field(default_factory=list)
    candidates: list[str] = field(default_factory=list)
    pages: int = 0
    characters: int = 0
    error: Optional[str] = None
    text: str = ""                                                   # full slide text (used for subject scoring)
    acronyms: dict[str, str] = field(default_factory=dict)           # "CSF" -> "cerebrospinal fluid"
    acronym_tokens: list[str] = field(default_factory=list)          # acronyms used without a definition
    rare_terms: list[str] = field(default_factory=list)              # capitalised proper terms outside the dictionary
    categories: dict[str, list[str]] = field(default_factory=dict)   # enzyme / protein / drug / anatomy ... -> terms

    def get_terms(self) -> list[str]:
        return list(self.terms)


def find_companion_pdf(media_path: Path) -> Optional[Path]:
    """``lecture.mp4`` -> ``lecture.pdf`` in the same folder (case-insensitive)."""
    media_path = Path(media_path)
    for candidate in media_path.parent.iterdir():
        if candidate.is_file() and candidate.suffix.lower() == ".pdf" and candidate.stem.lower() == media_path.stem.lower():
            return candidate
    return None


def extract_pdf_text(pdf_path: Path) -> tuple[str, int]:
    """Return ``(text, page_count)``. Raises ``ImportError`` if pypdf is not installed."""
    from pypdf import PdfReader  # optional dependency, imported lazily

    reader = PdfReader(str(pdf_path))
    pages = [(page.extract_text() or "") for page in reader.pages]
    return "\n".join(pages), len(pages)


def extract_candidate_terms(text: str, min_frequency: int = 2) -> list[str]:
    """Regex candidates (eponyms + medical-suffix words) seen at least ``min_frequency`` times."""
    counts: Counter[str] = Counter()
    display: dict[str, str] = {}
    for pattern in (_EPONYM, _SUFFIX_WORD):
        for match in pattern.finditer(text):
            word = match.group(1)
            key = normalize_key(word)
            if len(key) < 5 or word.casefold() in _COMMON_FALSE_SUFFIX:
                continue
            counts[key] += 1
            display.setdefault(key, word)
    ranked = [k for k, n in counts.most_common() if n >= min_frequency]
    return [display[k] for k in ranked]


def build_context(
    pdf_path: Path, vocab: Optional[VocabularyIndex], min_frequency: int = 2, max_terms: int = 25,
) -> LectureContext:
    """Extract lecture terms from ``pdf_path``. Never raises; problems land in ``error``."""
    context = LectureContext(source=Path(pdf_path))
    try:
        text, pages = extract_pdf_text(pdf_path)
    except ImportError:
        context.error = "pypdf is not installed (pip install pypdf); PDF context disabled"
        return context
    except Exception as exc:  # noqa: BLE001 - corrupt/encrypted PDFs must not stop transcription
        context.error = f"could not read PDF: {type(exc).__name__}: {exc}"
        return context
    context.pages, context.characters = pages, len(text)
    context.text = text
    if not text.strip():
        context.error = "PDF has no extractable text (scanned slides? OCR is not part of v1)"
        return context

    hits: Counter[str] = Counter(vocab.find_terms(text)) if vocab is not None else Counter()
    context.dictionary_hits = [term for term, _ in hits.most_common()]
    known = {normalize_key(t) for t in context.dictionary_hits}
    context.candidates = [
        t for t in extract_candidate_terms(text, min_frequency) if normalize_key(t) not in known
    ]
    _enrich(context, text, known, min_frequency)
    context.terms = _merge_terms(context, max_terms)
    log.info("PDF context from %s: %d pages, %d dictionary hits, %d candidates, %d acronyms, %d rare terms -> %d terms",
             Path(pdf_path).name, pages, len(context.dictionary_hits), len(context.candidates), len(context.acronyms),
             len(context.rare_terms), len(context.terms))
    return context


_RARE_PROPER = re.compile(r"\b([A-Z][a-z]{3,}(?:-[A-Z][a-z]{2,})?)\b")
_ACRONYM_TOKEN = re.compile(r"\b([A-Z][A-Z0-9]{1,6})\b")
_GROUPS_KEPT = ("enzyme", "protein", "drug", "gene", "receptor", "hormone", "neurotransmitter", "anatomy")


def _enrich(context: LectureContext, text: str, known_keys: set[str], min_frequency: int) -> None:
    """Acronyms (defined and bare), rare proper terms and enzyme/protein/drug/anatomy groups from the slide text."""
    from .acronyms import extract_definitions, looks_like_acronym
    from .entities import classify_term, known_drug_names
    from .knowledge_entities import COMMON_ENGLISH

    context.acronyms = extract_definitions(text)
    counts = Counter(m.group(1) for m in _ACRONYM_TOKEN.finditer(text))
    context.acronym_tokens = [a for a, n in counts.most_common()
                              if n >= min_frequency and a not in context.acronyms and looks_like_acronym(a)][:30]

    # Capitalised words in the middle of a sentence, seen at least twice, that are neither dictionary terms nor
    # everyday words (a capital at a sentence/line start proves nothing).
    mid_sentence: Counter[str] = Counter()
    for match in _RARE_PROPER.finditer(text):
        before = text[max(0, match.start() - 2):match.start()]
        if before.strip() in {"", ".", "!", "?", ":"} or "\n" in before:
            continue
        word = match.group(1)
        key = normalize_key(word)
        if key in known_keys or word.casefold() in COMMON_ENGLISH or len(key) < 5:
            continue
        mid_sentence[word] += 1
    context.rare_terms = [w for w, n in mid_sentence.most_common() if n >= min_frequency][:20]

    drugs = known_drug_names()
    groups: dict[str, list[str]] = {}
    for term in [*context.dictionary_hits, *context.candidates, *context.rare_terms]:
        for category in classify_term(term, drugs).categories:
            if category in _GROUPS_KEPT and term not in groups.setdefault(category, []):
                groups[category].append(term)
    context.categories = {k: v for k, v in groups.items() if v}


def _merge_terms(context: LectureContext, max_terms: int) -> list[str]:
    """Dictionary hits first, then enzyme/protein/drug/gene terms, rare proper terms, other candidates, acronyms."""
    ordered: list[str] = []
    seen: set[str] = set()
    special = [t for cat in ("enzyme", "protein", "drug", "gene") for t in context.categories.get(cat, [])]
    for term in [*context.dictionary_hits, *special, *context.rare_terms, *context.candidates, *context.acronyms]:
        key = normalize_key(term)
        if key and key not in seen:
            seen.add(key)
            ordered.append(term)
    return ordered[:max_terms]


class PdfContextProvider:
    """:class:`ContextProvider` backed by a lecture PDF."""

    def __init__(self, pdf_path: Path, vocab: Optional[VocabularyIndex], min_frequency: int = 2,
                 max_terms: int = 25) -> None:
        self.context = build_context(pdf_path, vocab, min_frequency, max_terms)

    def get_terms(self) -> list[str]:
        return self.context.get_terms()
