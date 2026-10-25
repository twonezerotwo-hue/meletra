"""Language-aware medical terminology: English <-> Turkish term pairs.

The pairs let the pipeline *recognise* both forms (``corticospinal tract`` / ``kortikospinal trakt``,
``glomerulus`` / ``glomerül``). Nothing is translated or substituted in a transcript.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from .vocabulary import DEFAULT_TERMS_DIR, VocabularyIndex, load_terms, normalize_key

PAIRS_FILE = "tr_en_pairs.tsv"


@dataclass
class TermPairs:
    pairs: list[tuple[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._en_to_tr = {normalize_key(en): tr for en, tr in self.pairs}
        self._tr_to_en = {normalize_key(tr): en for en, tr in self.pairs}
        self._en_canon = {normalize_key(en): en for en, _ in self.pairs}

    def __len__(self) -> int:
        return len(self.pairs)

    def to_turkish(self, term: str) -> Optional[str]:
        return self._en_to_tr.get(normalize_key(term))

    def to_english(self, term: str) -> Optional[str]:
        """English form for a Turkish term, or the term itself if it is already a known English one."""
        key = normalize_key(term)
        return self._tr_to_en.get(key) or self._en_canon.get(key)

    @property
    def turkish_terms(self) -> list[str]:
        return [tr for _, tr in self.pairs]

    @property
    def english_terms(self) -> list[str]:
        return [en for en, _ in self.pairs]


def load_pairs(terms_dir: Path = DEFAULT_TERMS_DIR) -> TermPairs:
    path = Path(terms_dir) / PAIRS_FILE
    pairs: list[tuple[str, str]] = []
    if path.is_file():
        for raw in path.read_text(encoding="utf-8-sig").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "\t" not in line:
                continue
            en, tr = (part.strip() for part in line.split("\t", 1))
            if en and tr:
                pairs.append((en, tr))
    return TermPairs(pairs)


def wants_turkish(language: Optional[str]) -> bool:
    """Turkish forms are recognised for Turkish runs and for auto-detect (the lecture may switch)."""
    return language in (None, "tr")


def with_turkish_forms(terms: Iterable[str], pairs: TermPairs) -> list[str]:
    """``terms`` plus the Turkish form of every term that has a pair (English kept, Turkish appended)."""
    out = list(terms)
    seen = {normalize_key(t) for t in out}
    for term in list(out):
        turkish = pairs.to_turkish(term)
        if turkish and normalize_key(turkish) not in seen:
            seen.add(normalize_key(turkish))
            out.append(turkish)
    return out


def build_language_index(language: Optional[str], terms_dir: Path = DEFAULT_TERMS_DIR,
                         extra_terms: Iterable[str] = ()) -> tuple[VocabularyIndex, TermPairs]:
    """Vocabulary index over all subjects, plus Turkish forms when the language calls for them."""
    pairs = load_pairs(terms_dir)
    terms = load_terms("all", language, terms_dir)
    if wants_turkish(language):
        terms = with_turkish_forms(terms, pairs)
        terms += [tr for _, tr in pairs.pairs if normalize_key(tr) not in {normalize_key(t) for t in terms}]
    terms += [t for t in extra_terms if t]
    return VocabularyIndex(terms), pairs


def localize_hotwords(terms: Iterable[str], language: Optional[str], pairs: TermPairs) -> list[str]:
    """Hotword spelling for the spoken language: Turkish runs get the Turkish form where a pair exists
    (English kept otherwise); every other language keeps the English terms."""
    out: list[str] = []
    seen: set[str] = set()
    for term in terms:
        word = (pairs.to_turkish(term) or term) if language == "tr" else term
        key = normalize_key(word)
        if key and key not in seen:
            seen.add(key)
            out.append(word)
    return out
