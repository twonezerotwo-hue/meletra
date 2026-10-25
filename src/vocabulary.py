"""Medical vocabulary: loading, subject selection, hotword building and fuzzy lookup."""

from __future__ import annotations

import difflib
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

from . import PROJECT_ROOT

log = logging.getLogger(__name__)

DEFAULT_TERMS_DIR = PROJECT_ROOT / "medical_terms"
SUBJECTS = (
    "anatomy", "histology", "physiology", "biochemistry", "molecular_biology",
    "neuroscience", "pathology", "pharmacology", "general_medical",
)
_WORD_RE = re.compile(r"[^\W\d_][\w'’\-]*", re.UNICODE)


class UnknownSubjectError(ValueError):
    """The requested subject has no vocabulary file."""


# Endings that turn one valid medical word into another (receptor/receptors, bradycardia/bradycardic,
# transcription/transcriptional ...). Words differing only by these are inflections, not mis-hearings.
_INFLECTION_ENDINGS = frozenset({
    "", "s", "es", "ed", "ing", "ic", "tic", "ical", "ically", "al", "ally", "ive", "ion", "ions", "ional",
    "ate", "ates", "ia", "ium", "us", "um", "ae", "a", "e", "y", "ist", "ists", "ism", "ous", "ity", "ar", "ary",
    "ose", "osis", "itis", "sis", "tis", "esis", "ide", "ine", "in",
    # adverb / agent / adjective forms of the same word (seen on a real lecture: relatively~relative, negatively~negative,
    # transporter~transport, exchange~exchanger, dominate~dominant): real English derivations, not mis-hearings
    "ly", "er", "ers", "or", "ors", "ant", "ent", "ance", "ence", "ness", "ment", "ments",
})


def is_inflection(word: str, candidate: str) -> bool:
    """True if ``word`` and ``candidate`` share a stem and differ only by a common word ending."""
    prefix = 0
    for a, b in zip(word, candidate):
        if a != b:
            break
        prefix += 1
    for stem_len in range(prefix, max(prefix - 4, 2), -1):
        if word[stem_len:] in _INFLECTION_ENDINGS and candidate[stem_len:] in _INFLECTION_ENDINGS:
            return True
    return False


def normalize_key(text: str) -> str:
    """Case/diacritic/punctuation-insensitive key (Turkish dotted/dotless i are unified)."""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    folded = stripped.casefold().replace("ı", "i")
    return re.sub(r"[^0-9a-z]+", "", folded)


def tokenize(text: str) -> list[str]:
    """Words of ``text`` (letters, with inner hyphens/apostrophes)."""
    return _WORD_RE.findall(text)


def available_subjects(terms_dir: Path = DEFAULT_TERMS_DIR) -> list[str]:
    return sorted(p.stem for p in Path(terms_dir).glob("*.txt"))


def parse_subjects(value: Optional[str | Sequence[str]], terms_dir: Path = DEFAULT_TERMS_DIR) -> list[str]:
    """Parse ``"neuroscience,anatomy"`` / ``"all"`` / a list into validated subject names."""
    available = available_subjects(terms_dir)
    if value is None:
        return []
    items = value.split(",") if isinstance(value, str) else list(value)
    names = [item.strip().lower() for item in items if item and item.strip()]
    if "all" in names:
        return available
    unknown = [n for n in names if n not in available]
    if unknown:
        raise UnknownSubjectError(
            f"Unknown subject(s): {', '.join(unknown)}. Available: {', '.join(available)}, all"
        )
    return list(dict.fromkeys(names))


def read_term_file(path: Path, language: Optional[str] = None) -> list[str]:
    """Read one term file. ``tr:`` / ``en:`` prefixes restrict a term to that language."""
    terms: list[str] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            match = re.match(r"^([a-z]{2}):\s*(.+)$", line)
            if match:
                prefix, term = match.group(1), match.group(2).strip()
                if language is not None and prefix != language:
                    continue
                terms.append(term)
            else:
                terms.append(line)
    return terms


def load_terms(
    subjects: str | Sequence[str] | None,
    language: Optional[str] = None,
    terms_dir: Path = DEFAULT_TERMS_DIR,
) -> list[str]:
    """Terms for the given subject(s), de-duplicated, original order kept.

    ``language`` ``None`` (auto) keeps terms of every language.
    """
    names = parse_subjects(subjects, terms_dir)
    seen: set[str] = set()
    merged: list[str] = []
    for name in names:
        for term in read_term_file(Path(terms_dir) / f"{name}.txt", language):
            key = normalize_key(term)
            if key and key not in seen:
                seen.add(key)
                merged.append(term)
    return merged


def build_hotwords(
    terms: Iterable[str],
    priority_terms: Iterable[str] = (),
    max_terms: int = 40,
    max_chars: int = 700,
    multiword_first: bool = False,
) -> list[str]:
    """Pick the terms handed to the ASR backend, in priority order.

    Order: ``priority_terms`` (PDF / terms heard in the lecture) as given, then the subject
    ``terms``. With ``multiword_first`` the subject terms are stably sorted so multi-word terms
    ("periaqueductal gray") precede single words: they are more specific, so they bias the model
    toward the lecture's vocabulary with fewer side effects on ordinary words.
    """
    subject_terms = list(terms)
    if multiword_first:
        subject_terms.sort(key=lambda t: 0 if len(tokenize(t)) > 1 else 1)
    chosen: list[str] = []
    seen: set[str] = set()
    used_chars = 0
    for term in [*priority_terms, *subject_terms]:
        key = normalize_key(term)
        if not key or key in seen:
            continue
        if len(chosen) >= max_terms or used_chars + len(term) + 2 > max_chars:
            break
        seen.add(key)
        chosen.append(term)
        used_chars += len(term) + 2
    return chosen


def hotwords_to_string(hotwords: Sequence[str]) -> str:
    return ", ".join(hotwords)


def fit_token_budget(
    terms: Sequence[str], count_tokens: Callable[[str], int], budget: int,
) -> tuple[list[str], int]:
    """Largest priority-ordered subset of ``terms`` whose joined prompt fits ``budget`` Whisper tokens.

    Whole terms only: faster-whisper itself would silently cut an over-long hotword string in the
    middle of a word, so we decide what is dropped. A term that does not fit is skipped and later
    (shorter) ones may still be added. Returns ``(kept, tokens_used)``.
    """
    kept: list[str] = []
    used = 0
    for term in terms:
        trial = [*kept, term]
        tokens = count_tokens(hotwords_to_string(trial))
        if tokens <= budget:
            kept, used = trial, tokens
    return kept, used


@dataclass
class VocabularyIndex:
    """Lookup structure over a term list (exact-key, per-word and fuzzy queries)."""

    terms: list[str]
    _by_key: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _words: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _word_list: list[str] = field(default_factory=list, init=False, repr=False)
    max_words: int = field(default=1, init=False)
    _fuzzy_cache: dict[tuple[str, float], Optional[str]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        for term in self.terms:
            self._by_key[normalize_key(term)] = term
            words = tokenize(term)
            self.max_words = max(self.max_words, len(words))
            for word in words:
                key = normalize_key(word)
                if key:
                    self._words.setdefault(key, word)
        self._word_list = sorted(self._words)

    def __len__(self) -> int:
        return len(self.terms)

    def canonical(self, text: str) -> Optional[str]:
        """Exact (spacing/hyphen/case-insensitive) dictionary term for ``text``, if any."""
        return self._by_key.get(normalize_key(text))

    def has_word(self, word: str) -> bool:
        return normalize_key(word) in self._words

    def near_miss(self, word: str, cutoff: float = 0.88, min_length: int = 7) -> Optional[str]:
        """A dictionary *word* that ``word`` resembles but does not equal (likely a mis-hearing)."""
        key = normalize_key(word)
        if len(key) < min_length or key in self._words:
            return None
        cache_key = (key, cutoff)
        if cache_key in self._fuzzy_cache:
            return self._fuzzy_cache[cache_key]
        result: Optional[str] = None
        for candidate in difflib.get_close_matches(key, self._word_list, n=3, cutoff=cutoff):
            if abs(len(candidate) - len(key)) > 2 or candidate[:2] != key[:2]:
                continue
            if is_inflection(key, candidate):
                result = None  # a valid inflected form of a dictionary word: not a mis-hearing
                break
            result = self._words[candidate]
            break
        self._fuzzy_cache[cache_key] = result
        return result

    @property
    def term_keys(self) -> dict[str, str]:
        """Normalised key -> canonical term (read-only view for fuzzy matching)."""
        return self._by_key

    def find_term_spans(self, words: Sequence[str]) -> list[tuple[int, int, str]]:
        """Dictionary terms in a word list as ``(first_word, end_word_exclusive, term)`` (longest first)."""
        spans: list[tuple[int, int, str]] = []
        i = 0
        while i < len(words):
            matched = False
            for size in range(min(self.max_words + 1, len(words) - i), 0, -1):
                term = self.canonical(" ".join(words[i:i + size]))
                if term is not None:
                    spans.append((i, i + size, term))
                    i += size
                    matched = True
                    break
            if not matched:
                i += 1
        return spans

    def find_terms(self, text: str) -> list[str]:
        """Dictionary terms occurring in ``text`` (longest-first sliding window, multiset order)."""
        return [term for _, _, term in self.find_term_spans(tokenize(text))]


def build_index(
    subjects: str | Sequence[str] | None,
    language: Optional[str] = None,
    terms_dir: Path = DEFAULT_TERMS_DIR,
    extra_terms: Iterable[str] = (),
) -> VocabularyIndex:
    terms = load_terms(subjects, language, terms_dir)
    terms.extend(t for t in extra_terms if t)
    return VocabularyIndex(terms)
