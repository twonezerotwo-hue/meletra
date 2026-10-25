"""``MedicalKnowledge``: the read-only API over a Medical Knowledge database.

    kb = MedicalKnowledge.open(path)
    kb.lookup("Remak")                    exact (normalised, spacing-insensitive) hits with concept, domains, trust
    kb.get_concept("MLT:remakbundle")     one concept with all its terms, domains and sources
    kb.related("MLT:schwanncell")         lightweight ontology neighbours
    kb.validate_term("Nissl body")        "is this a known medical term, in which fields, how far verified?"

The same API serves the correction engine now and later fact generation of the Study engine. Nothing is loaded at start-up: every call is an indexed
query on the database file, with small bounded caches in front of the frequent ones.
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

from ..database.schema import KNOWLEDGE_NAME, SCHEMA_VERSION
from ..database.store import Check, KnowledgeError, health, open_readonly, read_meta
from ..normalize.text import compact, normalize
from .cache import LRU

DB_ENV = "MLT_KNOWLEDGE_DB"
DEFAULT_NAMES = ("medical_knowledge_public.db", "medical_knowledge_full.db")


@dataclass(frozen=True)
class TermHit:
    term_id: int
    concept_id: str
    text: str
    language: str
    term_type: str
    is_preferred: bool
    trust: str
    preferred_name: str
    domains: tuple[str, ...]
    frequency: int = 0
    redistributable: bool = False
    match: str = "exact"                         # exact | prefix | phonetic | ...


@dataclass(frozen=True)
class TermInfo:
    text: str
    language: str
    term_type: str
    is_preferred: bool
    sources: tuple[str, ...]


@dataclass(frozen=True)
class Concept:
    concept_id: str
    preferred_name: str
    semantic_type: str
    kind: str
    trust: str
    domains: tuple[str, ...]
    terms: tuple[TermInfo, ...]
    sources: tuple[str, ...]

    def terms_in(self, language: str) -> list[str]:
        return [t.text for t in self.terms if t.language == language]

    @property
    def languages(self) -> tuple[str, ...]:
        return tuple(sorted({t.language for t in self.terms}))


@dataclass(frozen=True)
class Related:
    concept_id: str
    preferred_name: str
    relation: str
    weight: float
    direction: str                               # "out" (this -> other) or "in"


@dataclass(frozen=True)
class Validation:
    text: str
    known: bool
    concepts: tuple[str, ...] = ()
    names: tuple[str, ...] = ()
    domains: tuple[str, ...] = ()
    trust: str = ""
    is_abbreviation: bool = False
    expansions: tuple[str, ...] = ()
    in_domain: Optional[bool] = None             # when a domain was asked for: is the term known in that field (or a related one)?
    note: str = ""


def find_database(explicit: Optional[str | Path] = None, extra_dirs: Sequence[Path] = ()) -> Optional[Path]:
    """The database to use: explicit path, ``MLT_KNOWLEDGE_DB``, then the standard names in the app's ``data`` folder and the given folders."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get(DB_ENV):
        candidates.append(Path(os.environ[DB_ENV]))
    from ... import RESOURCE_ROOT

    for folder in (*extra_dirs, RESOURCE_ROOT / "data", RESOURCE_ROOT / "medical_knowledge"):
        candidates.extend(Path(folder) / name for name in DEFAULT_NAMES)
    return next((c for c in candidates if c.is_file()), None)


class MedicalKnowledge:
    def __init__(self, db: sqlite3.Connection, path: Path) -> None:
        self.db = db
        self.path = Path(path)
        self.meta = read_meta(db)
        self._terms: LRU = LRU(8192)
        self._concepts: LRU = LRU(4096)
        self._domains_of: LRU = LRU(8192)
        self._words: LRU = LRU(32768)
        self._levels: LRU = LRU(32768)
        self._token_terms: LRU = LRU(8192)
        self._types: LRU = LRU(16384)
        self._token_index: Optional[bool] = None
        self._long_tokens: Optional[dict[str, list[str]]] = None

    @classmethod
    def open(cls, path: Optional[str | Path] = None) -> "MedicalKnowledge":
        found = find_database(path)
        if found is None:
            raise KnowledgeError("no medical knowledge database found (set MLT_KNOWLEDGE_DB or build one with tools/build_medical_knowledge.py)")
        return cls(open_readonly(found), found)

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "MedicalKnowledge":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ information
    @property
    def profile(self) -> str:
        return self.meta.get("profile", "")

    @property
    def build_version(self) -> str:
        return self.meta.get("build_version", "")

    def counts(self) -> dict:
        return json.loads(self.meta.get("counts", "{}"))

    def info(self) -> dict:
        counts = self.counts()
        return {"name": KNOWLEDGE_NAME, "schema_version": int(self.meta.get("schema_version", 0)), "build_version": self.build_version,
                "build_date": self.meta.get("build_date", ""), "profile": self.profile, "file": str(self.path),
                "size_bytes": self.path.stat().st_size, "counts": counts,
                "sources": [dict(zip(("source_id", "name", "version", "license_status", "redistribution_allowed"), row)) for row in
                            self.db.execute("SELECT source_id, name, version, license_status, redistribution_allowed FROM source ORDER BY source_id")],
                "languages": counts.get("languages", {}), "phonetic_algorithms": self.meta.get("phonetic_algorithms", "").split(",")}

    def health(self) -> list[Check]:
        return health(self.db)

    # ------------------------------------------------------------------ lookup
    def _domains(self, concept_id: str) -> tuple[str, ...]:
        return self._domains_of.get(concept_id, lambda: tuple(r[0] for r in self.db.execute(
            "SELECT domain FROM concept_domain WHERE concept_id = ? ORDER BY weight DESC, domain", (concept_id,))))

    def _hit(self, row, match: str) -> TermHit:
        term_id, concept_id, text, language, term_type, preferred, trust, name, freq, redist = row
        return TermHit(term_id, concept_id, text, language, term_type or "", bool(preferred), trust, name, self._domains(concept_id), freq, bool(redist), match)

    _SELECT = ("SELECT t.term_id, t.concept_id, t.text, t.language, t.term_type, t.is_preferred, t.trust, c.preferred_name, t.frequency, t.redistributable "
               "FROM term t JOIN concept c ON c.concept_id = t.concept_id ")

    def lookup(self, text: str, *, domains: Optional[Sequence[str]] = None, language: Optional[str] = None, limit: int = 20,
               prefix: bool = False) -> list[TermHit]:
        """Terms whose normalised spelling equals ``text`` (spacing and hyphens ignored); ``prefix=True`` also returns terms that begin with it."""
        key = compact(text)
        if not key:
            return []
        cache_key = (key, tuple(domains or ()), language, limit, prefix)

        def compute() -> list[TermHit]:
            rows = self.db.execute(self._SELECT + "WHERE t.compact_text = ? " + ("AND t.language = ? " if language else "") + "ORDER BY t.is_preferred DESC, "
                                   "t.frequency DESC LIMIT ?", (key, *([language] if language else []), limit * 4)).fetchall()
            hits = [self._hit(r, "exact") for r in rows]
            if prefix and len(hits) < limit and len(key) >= 3:
                words = normalize(text).split()
                match = " ".join('"' + w + '"' for w in words[:-1]) + (" " if len(words) > 1 else "") + '"' + words[-1] + '"*'
                more = self.db.execute(self._SELECT + "WHERE t.term_id IN (SELECT rowid FROM term_fts WHERE term_fts MATCH ? LIMIT ?) AND t.compact_text != ? "
                                       "ORDER BY t.is_preferred DESC, t.token_count LIMIT ?", (match, limit * 4, key, limit)).fetchall()
                hits += [self._hit(r, "prefix") for r in more]
            if domains:
                wanted = set(domains)
                hits = [h for h in hits if wanted.intersection(h.domains)]
            return hits[:limit]

        return list(self._terms.get(cache_key, compute))

    def get_concept(self, concept_id: str) -> Optional[Concept]:
        def compute() -> Optional[Concept]:
            row = self.db.execute("SELECT concept_id, preferred_name, semantic_type, kind, trust FROM concept WHERE concept_id = ?", (concept_id,)).fetchone()
            if row is None:
                return None
            terms = []
            for term_id, text, language, term_type, preferred in self.db.execute(
                    "SELECT term_id, text, language, term_type, is_preferred FROM term WHERE concept_id = ? ORDER BY is_preferred DESC, language, text", (concept_id,)):
                srcs = tuple(r[0] for r in self.db.execute("SELECT DISTINCT source_id FROM term_source WHERE term_id = ?", (term_id,)))
                terms.append(TermInfo(text, language, term_type or "", bool(preferred), srcs))
            sources = tuple(sorted({s for t in terms for s in t.sources}))
            return Concept(row[0], row[1], row[2] or "", row[3], row[4], self._domains(concept_id), tuple(terms), sources)

        return self._concepts.get(concept_id, compute)

    def related(self, concept_id: str, relation: Optional[str] = None, limit: int = 20) -> list[Related]:
        out: list[Related] = []
        filt = " AND r.relation_type = ?" if relation else ""
        args = (concept_id, *([relation] if relation else []), limit)
        for row in self.db.execute("SELECT r.concept_b, c.preferred_name, r.relation_type, r.weight FROM relationship r JOIN concept c ON c.concept_id = r.concept_b "
                                   "WHERE r.concept_a = ?" + filt + " ORDER BY r.weight DESC, c.preferred_name LIMIT ?", args):
            out.append(Related(row[0], row[1], row[2], row[3], "out"))
        for row in self.db.execute("SELECT r.concept_a, c.preferred_name, r.relation_type, r.weight FROM relationship r JOIN concept c ON c.concept_id = r.concept_a "
                                   "WHERE r.concept_b = ?" + filt + " ORDER BY r.weight DESC, c.preferred_name LIMIT ?", args):
            out.append(Related(row[0], row[1], row[2], row[3], "in"))
        best: dict[tuple[str, str], Related] = {}
        for r in out:                                                       # the same neighbour can come from several sources / directions
            key = (r.concept_id, r.relation)
            if key not in best or r.weight > best[key].weight:
                best[key] = r
        return sorted(best.values(), key=lambda r: (-r.weight, r.preferred_name))[:limit]

    def related_ids(self, concept_id: str) -> set[str]:
        return {r.concept_id for r in self.related(concept_id, limit=200)}

    def validate_term(self, text: str, domain: Optional[str] = None) -> Validation:
        hits = self.lookup(text, limit=50)
        abbr = self.db.execute("SELECT long_form, concept_id, domain FROM abbreviation WHERE norm_short = ?", (compact(text),)).fetchall()
        if not hits and not abbr:
            return Validation(text, False, note="not in the knowledge base")
        concepts = tuple(dict.fromkeys(h.concept_id for h in hits))
        names = tuple(dict.fromkeys(h.preferred_name for h in hits))
        domains = tuple(dict.fromkeys(d for h in hits for d in h.domains))
        trust = hits[0].trust if hits else "CURATED"
        in_domain = None
        if domain:
            from ..domains import expand

            near = set(expand([domain]))
            in_domain = bool(near.intersection(domains)) or any(a[2] in near for a in abbr)
        return Validation(text, True, concepts, names, domains, trust, bool(abbr), tuple(dict.fromkeys(a[0] for a in abbr)), in_domain)

    # ------------------------------------------------------------------ small helpers used by retrieval / scoring
    def concept_domains(self, concept_id: str) -> tuple[str, ...]:
        return self._domains(concept_id)

    def concept_domains_many(self, concept_ids: Iterable[str]) -> dict[str, tuple[str, ...]]:
        """Fields of study of many concepts with one query for those not cached yet."""
        out: dict[str, tuple[str, ...]] = {}
        missing: list[str] = []
        for cid in dict.fromkeys(concept_ids):
            cached = self._domains_of._data.get(cid)
            if cached is not None:
                out[cid] = cached
            else:
                missing.append(cid)
        for start in range(0, len(missing), 400):
            chunk = missing[start:start + 400]
            marks = ",".join("?" * len(chunk))
            found: dict[str, list[str]] = {cid: [] for cid in chunk}
            for cid, domain in self.db.execute(f"SELECT concept_id, domain FROM concept_domain WHERE concept_id IN ({marks}) ORDER BY weight DESC, domain", chunk):
                found[cid].append(domain)
            for cid, domains in found.items():
                value = tuple(domains)
                out[cid] = value
                self._domains_of.get(cid, lambda value=value: value)
        return out

    def is_english_word(self, word: str, max_level: int = 70) -> bool:
        """True when the word is an ordinary English word of the dictionary lists (level 70 = about 110,000 words; 80 = about 240,000)."""
        key = word.lower()
        return self._words.get((key, max_level), lambda: self.db.execute("SELECT 1 FROM english_word WHERE word = ? AND level <= ?", (key, max_level)).fetchone() is not None)

    def has_form(self, phrase: str) -> bool:
        """True when the sources have exactly this phrase as a term, or its last word is a dictionary / vocabulary word (a verified number form: "synapses", "foramina")."""
        key = compact(phrase)
        if not key:
            return False
        if self.db.execute("SELECT 1 FROM term WHERE compact_text = ? LIMIT 1", (key,)).fetchone() is not None:
            return True
        last = normalize(phrase).split()[-1] if normalize(phrase).split() else ""
        return len(last) >= 4 and (self.is_english_word(last, 80) or self.token_frequency(last) >= 2)

    @property
    def has_token_index(self) -> bool:
        if self._token_index is None:
            names = {r[0] for r in self.db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
            self._token_index = "token_index" in names and "term_numeric" in names
        return self._token_index

    def semantic_type(self, concept_id: str) -> str:
        """The source's own type of a concept (GO: biological_process / molecular_function / cellular_component; HGNC: gene with protein product), '' when it has none. Cached."""
        return self._types.get(concept_id, lambda: (lambda r: r[0] if r and r[0] else "")(self.db.execute("SELECT semantic_type FROM concept WHERE concept_id = ?", (concept_id,)).fetchone()))

    def long_tokens(self) -> dict[str, list[str]]:
        """First three letters -> the longer (>= 7 letters) vocabulary words that are not ordinary English: the in-memory index for 'several words written apart = one rare word'."""
        if self._long_tokens is None:
            index: dict[str, list[str]] = {}
            if self.has_token_index:
                for token, pre in self.db.execute("SELECT token, pre FROM token_index WHERE len >= 7 AND (level = 0 OR level >= 70)"):
                    index.setdefault(pre, []).append(token)
            self._long_tokens = index
        return self._long_tokens

    def term_with_token(self, token: str):
        """The shortest term that contains the word (a row of the term + concept tables) - which concept a word-level candidate belongs to. Cached."""
        def load():
            row = self.db.execute("SELECT t.term_id, t.concept_id, t.text, t.language, t.term_type, t.trust, t.token_count, c.preferred_name, c.kind "
                                  "FROM term_fts f JOIN term t ON t.term_id = f.rowid JOIN concept c ON c.concept_id = t.concept_id WHERE term_fts MATCH ? "
                                  "ORDER BY t.token_count, t.is_preferred DESC LIMIT 1", ('"' + token + '"',)).fetchone()
            return row
        return self._token_terms.get(token, load)

    def word_level(self, word: str) -> Optional[int]:
        """SCOWL level of an ordinary English word (10 = the most common ... 80 = rare dictionary word), None when the lists do not have it."""
        key = word.lower()
        return self._levels.get(key, lambda: (lambda row: row[0] if row else None)(self.db.execute("SELECT level FROM english_word WHERE word = ?", (key,)).fetchone()))

    def token_frequency(self, token: str) -> int:
        row = self.db.execute("SELECT n FROM token_freq WHERE token = ?", (normalize(token),)).fetchone()
        return row[0] if row else 0

    def cache_stats(self) -> dict:
        return {"terms": self._terms.stats(), "concepts": self._concepts.stats(), "domains": self._domains_of.stats()}


def open_knowledge(path: Optional[str | Path] = None) -> Optional[MedicalKnowledge]:
    """The knowledge base if one is installed, else None (the v3.1.0 behaviour needs none)."""
    found = find_database(path)
    if found is None:
        return None
    return MedicalKnowledge(open_readonly(found), found)
