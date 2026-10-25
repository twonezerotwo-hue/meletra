"""Modular medical terminology: versioned domain folders, lazy loading, a sound-alike index.

``medical_vocab/<domain>/*.txt`` - one concept per line, ``|`` separated, everything after the term optional::

    Nissl body | Nissl bodies; Nissl substance | rough endoplasmic reticulum; ribosome; soma | nisal body | histology

    1  canonical spelling (correct capitalisation, diacritics)
    2  aliases: other correct forms (plurals, Latin / English pairs, spellings); matched exactly, never "corrected"
    3  cues: words and phrases that usually appear NEAR this term (the context that makes a correction believable)
    4  confusables: how ASR is known to mis-hear it (matched with the same context requirement as a sound-alike)
    5  kind: term | abbr | eponym | latin | gene | drug (default: term)

A domain folder may hold a ``_domain.json`` (title, version, related domains). Domains are loaded the first time they are asked for;
the sound-alike index is built per set of domains (the topic of the lecture decides which), so a lecture never pays for vocabulary
it does not need. Custom terms (settings) and slide terms are added at run time as extra domains ``custom`` / ``lecture``.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

from rapidfuzz.distance import Levenshtein

from .. import RESOURCE_ROOT
from .phonetics import letters, skeleton

log = logging.getLogger(__name__)

VOCAB_DIR = RESOURCE_ROOT / "medical_vocab"
KINDS = ("term", "abbr", "eponym", "latin", "gene", "drug")
_WORD = re.compile(r"[A-Za-zÀ-ɏ][A-Za-zÀ-ɏ'’-]*")
TRIGRAM = 3


def norm(text: str) -> str:
    """Matching form of a phrase: lower case, accents folded, hyphens / spaces / punctuation ignored."""
    import unicodedata

    plain = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]", "", plain.lower())


def words(text: str) -> list[str]:
    return [w.lower().strip("-'’") for w in _WORD.findall(text)]


_SUFFIXES = ("ations", "ation", "ings", "ing", "ical", "ally", "ies", "ied", "ic", "al", "es", "ed", "ly", "s")


@lru_cache(maxsize=32768)
def stem(word: str) -> str:
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def stem_phrase(text: str) -> str:
    """Lower-case word stems ("staining" -> "stain", "neuronal" -> "neuron"): cue words match their inflections."""
    return " ".join(stem(w) for w in words(text))


@dataclass
class Term:
    canonical: str
    domain: str
    kind: str = "term"
    aliases: tuple[str, ...] = ()
    cues: tuple[str, ...] = ()
    confusables: tuple[str, ...] = ()
    tokens: int = 1

    @property
    def surfaces(self) -> tuple[str, ...]:
        return (self.canonical, *self.aliases)

    @property
    def key(self) -> str:
        return norm(self.canonical)


@dataclass
class Domain:
    name: str
    version: str = "1"
    title: str = ""
    related: tuple[str, ...] = ()
    terms: list[Term] = field(default_factory=list)


def _plurals(canonical: str, kind: str) -> list[str]:
    """English plural of the last word of an ordinary term ("Nissl body" -> "Nissl bodies"); Latin / abbreviations / genes list theirs."""
    if kind != "term" or canonical.endswith("s") or canonical.isupper() or len(canonical) < 4:
        return []
    head, _, last = canonical.rpartition(" ")
    if not last.isalpha():
        return []
    if re.search(r"[^aeiou]y$", last):
        plural = last[:-1] + "ies"
    elif re.search(r"(s|x|z|ch|sh)$", last):
        plural = last + "es"
    elif last.endswith(("um", "us", "a", "ex", "is", "on")) and kind != "term":
        return []
    else:
        plural = last + "s"
    return [(head + " " if head else "") + plural]


def parse_line(line: str, domain: str) -> Optional[Term]:
    line = line.split("#", 1)[0].strip() if not line.lstrip().startswith("##") else line.strip()
    if not line or line.startswith("#"):
        return None
    parts = [p.strip() for p in line.split("|")]
    canonical = parts[0]
    if not canonical:
        return None

    def many(i: int) -> tuple[str, ...]:
        return tuple(x.strip() for x in parts[i].split(";") if x.strip()) if len(parts) > i else ()

    kind = parts[4].strip().lower() if len(parts) > 4 and parts[4].strip().lower() in KINDS else "term"
    aliases = many(1)
    extra = tuple(p for p in _plurals(canonical, kind) if p not in aliases)
    unique: list[str] = []
    seen = {norm(canonical)}
    for alias in aliases + extra:                                          # an alias twice (or the term itself) would be indexed twice
        if norm(alias) not in seen:
            seen.add(norm(alias))
            unique.append(alias)
    return Term(canonical, domain, kind, tuple(unique), many(2), many(3), tokens=len(words(canonical)))


def load_domain(folder: Path) -> Domain:
    meta: dict = {}
    if (folder / "_domain.json").is_file():
        try:
            meta = json.loads((folder / "_domain.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            log.warning("unreadable %s/_domain.json", folder)
    domain = Domain(folder.name, str(meta.get("version", "1")), meta.get("title", folder.name), tuple(meta.get("related", ())))
    owner: dict[str, Term] = {}                                       # normalised surface -> the term that owns it in this domain
    files = sorted(folder.glob("*.txt"), key=lambda path: (path.name == "baseline.txt", path.name))      # the starter list comes last
    for path in files:
        for raw in path.read_text(encoding="utf-8").splitlines():
            term = parse_line(raw, folder.name)
            if term is None:
                continue
            known = next((owner[norm(s)] for s in term.surfaces if norm(s) in owner), None)
            if known is None:
                domain.terms.append(term)
                for surface in term.surfaces:
                    owner.setdefault(norm(surface), term)
                continue
            # the same concept in two files, or a plural / spelling variant of it: merge instead of keeping two competing entries
            extra = tuple(s for s in term.surfaces if norm(s) not in {norm(x) for x in known.surfaces})
            known.aliases = _union(known.aliases, extra)
            known.cues = _union(known.cues, term.cues)
            known.confusables = _union(known.confusables, term.confusables)
            if known.kind == "term" and term.kind != "term":
                known.kind = term.kind
            for surface in known.surfaces:
                owner.setdefault(norm(surface), known)
    return domain


def _union(a: tuple[str, ...], b: tuple[str, ...]) -> tuple[str, ...]:
    seen = {norm(x) for x in a}
    return a + tuple(x for x in b if norm(x) not in seen and not seen.add(norm(x)))


def available_domains(root: Path = VOCAB_DIR) -> list[str]:
    return sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")) if root.is_dir() else []


class Lexicon:
    """Terms of the loaded domains + the lookup structures. ``subset`` builds an index over some domains only."""

    def __init__(self, root: Path = VOCAB_DIR) -> None:
        self.root = Path(root)
        self._domains: dict[str, Domain] = {}
        self._extra: dict[str, Domain] = {}
        self._indexes: dict[tuple, "Index"] = {}

    def fork(self) -> "Lexicon":
        """A lexicon that shares the loaded domains and the built indexes with this one but has its own run-time terms (slides, custom)."""
        other = Lexicon(self.root)
        other._domains, other._indexes = self._domains, self._indexes
        return other

    # ------------------------------------------------------------------ domains
    def domain(self, name: str) -> Optional[Domain]:
        if name in self._extra:
            return self._extra[name]
        if name not in self._domains:
            folder = self.root / name
            if not folder.is_dir():
                return None
            self._domains[name] = load_domain(folder)
        return self._domains[name]

    def names(self) -> list[str]:
        return sorted(set(available_domains(self.root)) | set(self._extra))

    def related(self, name: str) -> tuple[str, ...]:
        domain = self.domain(name)
        return domain.related if domain else ()

    def add_terms(self, domain: str, entries: Iterable[str], cues: Iterable[str] = ()) -> None:
        """Run-time vocabulary (``custom`` settings, ``lecture`` slides): plain terms, optionally with shared cue words."""
        target = self._extra.setdefault(domain, Domain(domain, title=domain))
        have = {t.key for t in target.terms}
        for entry in entries:
            term = parse_line(entry, domain)
            if term is not None and term.key not in have:
                term.cues = tuple(cues) + term.cues
                target.terms.append(term)
                have.add(term.key)

    def version(self) -> dict[str, str]:
        return {name: d.version for name in self.names() if (d := self.domain(name)) is not None}

    # ------------------------------------------------------------------ index
    def subset(self, names: Optional[Iterable[str]] = None) -> "Index":
        chosen = [n for n in (names if names is not None else self.names())]
        key = (tuple(sorted(chosen)), tuple((n, len(d.terms), d.terms[-1].key if d.terms else "") for n, d in sorted(self._extra.items())))
        cached = self._indexes.get(key)
        if cached is not None:
            return cached
        domains = [d for n in chosen if (d := self.domain(n)) is not None]
        domains += [d for n, d in self._extra.items() if n not in chosen]
        index = self._indexes[key] = Index(domains)
        return index


class Index:
    """Exact surfaces, confusables and a sound-alike (trigram) index over a set of domains."""

    def __init__(self, domains: list[Domain]) -> None:
        self.names = [d.name for d in domains]
        self.by_surface: dict[str, list[Term]] = {}
        self.by_confusable: dict[str, list[Term]] = {}
        self.surface_of: dict[str, str] = {}                      # normalised surface -> its display form
        self._sk: dict[str, list[tuple[str, Term]]] = {}          # skeleton -> [(surface, term)]
        self._tri: dict[str, set[str]] = {}                       # skeleton trigram -> skeletons
        for domain in domains:
            for term in domain.terms:
                for surface in term.surfaces:
                    key = norm(surface)
                    if not key:
                        continue
                    self.by_surface.setdefault(key, []).append(term)
                    self.surface_of.setdefault(key, surface)
                    sk = skeleton(surface)
                    if len(sk) >= 3:
                        self._sk.setdefault(sk, []).append((surface, term))
                for heard in term.confusables:
                    self.by_confusable.setdefault(norm(heard), []).append(term)
        for sk in self._sk:
            for i in range(max(len(sk) - TRIGRAM + 1, 1)):
                self._tri.setdefault(sk[i:i + TRIGRAM], set()).add(sk)
        self.known: set[str] = set()                             # stems of every word the vocabulary itself uses (surfaces and cues)
        self.token_freq: dict[str, int] = {}                     # word -> number of terms that contain it
        self.cue_df: dict[str, int] = {}                         # stemmed cue -> number of terms that list it (a cue shared by many says little)
        self._tail: dict[str, list[tuple[str, Term]]] = {}       # word ending -> single-word surfaces (compounds: "mitral glial" ~ microglial)
        self._head: dict[str, list[tuple[str, Term]]] = {}
        for key_terms in {id(t): t for ts in self.by_surface.values() for t in ts}.values():
            for word in {norm(w) for surface in key_terms.surfaces for w in surface.replace("-", " ").split()}:
                self.token_freq[word] = self.token_freq.get(word, 0) + 1
            for text in (*key_terms.surfaces, *key_terms.cues):
                self.known.update(stem(w) for w in words(text))
            for cue in {stem_phrase(c) for c in key_terms.cues if c}:
                self.cue_df[cue] = self.cue_df.get(cue, 0) + 1
            for surface in key_terms.surfaces:
                n = norm(surface)
                if " " not in surface and len(n) >= 7:
                    for k in range(4, len(n) - 2):
                        self._tail.setdefault(n[-k:], []).append((surface, key_terms))
                        self._head.setdefault(n[:k], []).append((surface, key_terms))
        self.max_tokens = max((t.tokens for ts in self.by_surface.values() for t in ts), default=1)

    def __len__(self) -> int:
        return len(self.by_surface)

    def containing(self, token: str) -> list[tuple[str, Term]]:
        """Single-word surfaces that begin or end with ``token`` (a covered word inside a longer compound term)."""
        key = norm(token)
        return self._tail.get(key, []) + self._head.get(key, []) if len(key) >= 4 else []

    def exact(self, phrase: str) -> list[Term]:
        return self.by_surface.get(norm(phrase), [])

    def candidates(self, span: str, minimum: float = 0.52, limit: int = 14) -> list[tuple[str, Term, float]]:
        """Surfaces that sound like ``span``: ``[(surface, term, skeleton similarity)]`` best first (the context decides afterwards)."""
        sk = skeleton(span)
        if len(sk) < 3:
            return []
        shared: dict[str, int] = {}
        grams = {sk[i:i + TRIGRAM] for i in range(max(len(sk) - TRIGRAM + 1, 1))}
        for gram in grams:
            for other in self._tri.get(gram, ()):
                shared[other] = shared.get(other, 0) + 1
        need = max(1, int(len(grams) * 0.34))
        out: list[tuple[str, Term, float]] = []
        for other, count in shared.items():
            if count < need or abs(len(other) - len(sk)) > max(3, len(sk) // 2):
                continue
            sim = Levenshtein.normalized_similarity(sk, other)
            if sim >= minimum:
                out.extend((surface, term, sim) for surface, term in self._sk[other])
        out.sort(key=lambda x: -x[2])
        return out[:limit]


_SHARED: dict[str, Lexicon] = {}


def shared_lexicon(root: Optional[Path] = None) -> Lexicon:
    """A fork of the process-wide lexicon for ``root`` (domains are read once per process, indexes are built once)."""
    key = str(Path(root or VOCAB_DIR).resolve())
    if key not in _SHARED:
        _SHARED[key] = Lexicon(Path(root or VOCAB_DIR))
    return _SHARED[key].fork()
