"""Builds a Medical Knowledge database from terminology sources.

    builder = KnowledgeBuilder(Path("data/medical_knowledge_public.db"), profile="public", build_version="2026.10")
    for source in sources: source.export(builder)
    report = builder.finalize()

* the license policy is enforced when a source is ADDED: in a public build a source that is not ALLOW + allowlisted raises ``LicenseViolation``
  (the build fails, the half-written file is removed); nothing is silently left out;
* every term row keeps its provenance in ``term_source``; ``term.redistributable`` is derived from the sources;
* the full-text / phonetic indexes are filled once at the end (``finalize``), then the database is checked, analysed and written next to a
  ``<name>.manifest.json`` (counts, source versions, license report, SHA-256).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

from ..database.schema import FTS, INDEXES, KNOWLEDGE_NAME, SCHEMA_VERSION, TABLES, TRUST_LEVELS
from ..domains import GENERAL
from ..indexes.phonetic import STORED, deletion_eligible, deletions, dm_key, keys_of, skeleton
from ..licenses.notices import write_notice
from ..licenses.policy import LicenseViolation, check_public, license_of, require_for_profile
from ..normalize.numbers import has_number, numeric_key
from ..normalize.text import compact, normalize, token_count
from ..sources.base import AbbreviationRecord, ConceptRecord, ConfusionRecord, CueRecord, RelationRecord, SourceInfo, TermRecord, WordRecord

BATCH = 20000


class KnowledgeBuilder:
    def __init__(self, output: Path, profile: str = "full", build_version: Optional[str] = None) -> None:
        self.output = Path(output)
        self.profile = profile
        self.build_version = build_version or time.strftime("%Y.%m")
        self.sources: dict[str, SourceInfo] = {}
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.tmp = self.output.with_suffix(self.output.suffix + ".building")
        for leftover in (self.tmp, Path(str(self.tmp) + "-journal")):
            leftover.unlink(missing_ok=True)
        self.db = sqlite3.connect(self.tmp)
        self.db.execute("PRAGMA journal_mode=OFF")
        self.db.execute("PRAGMA synchronous=OFF")
        self.db.execute("PRAGMA temp_store=FILE")
        self.db.execute("PRAGMA cache_size=-200000")
        self.db.executescript(TABLES)
        self._concepts: set[str] = set()
        self._finalized = False

    # ------------------------------------------------------------------ sources
    def add_source(self, info: SourceInfo) -> None:
        try:
            require_for_profile(info.source_id, self.profile)
        except LicenseViolation:
            self.abort()
            raise
        lic = license_of(info.source_id)
        self.sources[info.source_id] = info
        self.db.execute(
            "INSERT OR REPLACE INTO source VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (info.source_id, info.source_name, info.source_version, info.source_url, info.license_category, lic.status,
             1 if lic.redistribution_allowed else 0, 1 if info.requires_credentials or lic.requires_credentials else 0, info.language, info.release_date,
             info.notes))

    def _need_source(self, source_id: str) -> None:
        if source_id not in self.sources:
            raise ValueError(f"source {source_id} was not added first")

    # ------------------------------------------------------------------ records
    def add_concept(self, source_id: str, rec: ConceptRecord) -> None:
        self._need_source(source_id)
        if rec.trust not in TRUST_LEVELS:
            raise ValueError(f"unknown trust level {rec.trust}")
        domains = tuple(rec.domains) or (GENERAL,)
        existing = self.db.execute("SELECT 1 FROM concept WHERE concept_id=?", (rec.concept_id,)).fetchone()
        if existing is None:
            self.db.execute("INSERT INTO concept (concept_id, preferred_name, semantic_type, domain, subdomain, kind, trust) VALUES (?,?,?,?,?,?,?)",
                            (rec.concept_id, rec.preferred_name, rec.semantic_type, domains[0], rec.subdomain, rec.kind, rec.trust))
        for rank, domain in enumerate(domains):
            self.db.execute("INSERT OR IGNORE INTO concept_domain VALUES (?,?,?)", (rec.concept_id, domain, 1.0 if rank == 0 else 0.8))
        self._concepts.add(rec.concept_id)

    def add_term(self, source_id: str, rec: TermRecord) -> bool:
        """Insert a term (or add provenance to the term that already exists). False when the text is empty."""
        self._need_source(source_id)
        norm, comp = normalize(rec.text), compact(rec.text)
        if not comp:
            return False
        trust = rec.trust or self.db.execute("SELECT trust FROM concept WHERE concept_id=?", (rec.concept_id,)).fetchone()[0]
        row = self.db.execute(
            "INSERT INTO term (concept_id, text, normalized_text, compact_text, sk, dm, language, term_type, frequency, medical_frequency, is_preferred, "
            "token_count, trust) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT (concept_id, compact_text, language) DO UPDATE SET is_preferred = MAX(is_preferred, excluded.is_preferred), "
            "frequency = frequency + excluded.frequency RETURNING term_id",
            (rec.concept_id, rec.text, norm, comp, skeleton(norm), dm_key(norm), rec.language, rec.term_type, rec.frequency, rec.medical_frequency, 1 if rec.is_preferred else 0,
             token_count(rec.text), trust)).fetchone()
        self.db.execute("INSERT OR IGNORE INTO term_source VALUES (?,?,?,?,?,?)",
                        (row[0], source_id, rec.source_code, rec.source_concept_id, rec.term_type, rec.language))
        return True

    def add_relation(self, source_id: str, rec: RelationRecord) -> None:
        self._need_source(source_id)
        self.db.execute("INSERT OR IGNORE INTO relationship VALUES (?,?,?,?,?)", (rec.concept_a, rec.relation_type, rec.concept_b, source_id, rec.weight))

    def add_abbreviation(self, source_id: str, rec: AbbreviationRecord) -> None:
        self._need_source(source_id)
        self.db.execute("INSERT OR IGNORE INTO abbreviation (short_form, norm_short, long_form, concept_id, domain, language, source_id) VALUES (?,?,?,?,?,?,?)",
                        (rec.short_form, compact(rec.short_form), rec.long_form, rec.concept_id or None, rec.domain or None, rec.language, source_id))

    def add_confusion(self, source_id: str, rec: ConfusionRecord) -> None:
        self._need_source(source_id)
        self.db.execute("INSERT OR IGNORE INTO confusion VALUES (?,?,?,?)", (rec.concept_id, rec.heard, compact(rec.heard), source_id))

    def add_word(self, source_id: str, rec: WordRecord) -> None:
        self._need_source(source_id)
        self.db.execute("INSERT OR IGNORE INTO english_word VALUES (?,?,?)", (rec.word, rec.level, source_id))

    def add_cue(self, rec: CueRecord) -> None:
        self.db.execute("INSERT OR IGNORE INTO context_cue VALUES (?,?,?)", (rec.concept_id, normalize(rec.cue), rec.weight))

    # ------------------------------------------------------------------ finish
    def abort(self) -> None:
        try:
            self.db.close()
        finally:
            for leftover in (self.tmp, Path(str(self.tmp) + "-journal")):
                leftover.unlink(missing_ok=True)

    def finalize(self) -> dict:
        t0 = time.time()
        db = self.db
        db.commit()
        db.executescript(INDEXES)
        # redistributable flag of every term = at least one of its sources may be redistributed
        db.execute("UPDATE term SET redistributable = CASE WHEN EXISTS (SELECT 1 FROM term_source ts JOIN source s ON s.source_id = ts.source_id "
                   "WHERE ts.term_id = term.term_id AND s.redistribution_allowed = 1) THEN 1 ELSE 0 END")
        self._public_gate()
        db.executescript(FTS)
        self._phonetic_and_tokens()
        db.execute("INSERT INTO term_fts(term_fts) VALUES ('rebuild')")
        counts = self._counts()
        meta = {"schema_version": str(SCHEMA_VERSION), "name": KNOWLEDGE_NAME, "build_version": self.build_version,
                "build_date": time.strftime("%Y-%m-%dT%H:%M:%S"), "profile": self.profile,
                "source_versions": json.dumps({s.source_id: s.source_version for s in self.sources.values()}, sort_keys=True),
                "phonetic_algorithms": ",".join(STORED), "counts": json.dumps(counts, sort_keys=True)}
        db.executemany("INSERT INTO meta VALUES (?,?)", meta.items())
        db.commit()
        db.execute("ANALYZE")
        db.commit()
        db.close()
        self.output.unlink(missing_ok=True)
        self.tmp.replace(self.output)
        manifest = self._manifest(counts, meta, time.time() - t0)
        self.output.with_suffix(self.output.suffix + ".manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        write_notice(self.output, manifest)
        self._finalized = True
        return manifest

    def _public_gate(self) -> None:
        """A public database must contain nothing that is not redistributable - checked on the finished data, not only on the importers."""
        if self.profile != "public":
            return
        bad = self.db.execute("SELECT DISTINCT s.source_id FROM term_source ts JOIN source s ON s.source_id = ts.source_id "
                              "WHERE s.redistribution_allowed = 0").fetchall()
        problems = check_public(r[0] for r in self.db.execute("SELECT source_id FROM source").fetchall())
        if bad or problems:
            self.abort()
            raise LicenseViolation("PUBLIC KNOWLEDGE BUILD FAILS: " + "; ".join(problems or [f"{r[0]} is not redistributable" for r in bad]))

    def _phonetic_and_tokens(self) -> None:
        db = self.db
        freq: Counter = Counter()
        cursor = db.execute("SELECT term_id, normalized_text, token_count, dm, compact_text FROM term")
        while True:
            rows = cursor.fetchmany(BATCH)
            if not rows:
                break
            pk, dele, numeric = [], [], []
            for term_id, norm, tokens, dm, comp in rows:
                if has_number(norm):
                    nkey = numeric_key(norm)
                    if nkey != comp:
                        numeric.append((nkey, term_id))
                for word in set(norm.split()):
                    freq[word] += 1
                for algorithm, key in keys_of(norm, STORED).items():
                    pk.append((term_id, algorithm, key))
                if deletion_eligible(dm, tokens):
                    dele.extend((k, term_id) for k in deletions(dm))
            if pk:
                db.executemany("INSERT INTO phonetic_key VALUES (?,?,?)", pk)
            db.executemany("INSERT OR IGNORE INTO phonetic_del VALUES (?,?)", dele)
            if numeric:
                db.executemany("INSERT OR IGNORE INTO term_numeric VALUES (?,?)", numeric)
        db.executemany("INSERT INTO token_freq VALUES (?,?)", freq.items())
        levels = dict(db.execute("SELECT word, level FROM english_word").fetchall())
        words = [(w, n, len(w), w[:3], w[-4:], skeleton(w), dm_key(w), levels.get(w, 0)) for w, n in freq.items() if len(w) >= 4 and w.isalpha() and w.isascii()]
        db.executemany("INSERT INTO token_index VALUES (?,?,?,?,?,?,?,?)", words)

    def _counts(self) -> dict:
        q = lambda sql: self.db.execute(sql).fetchone()[0]                     # noqa: E731
        langs = dict(self.db.execute("SELECT language, COUNT(*) FROM term GROUP BY language").fetchall())
        return {"concepts": q("SELECT COUNT(*) FROM concept"), "terms": q("SELECT COUNT(*) FROM term"),
                "synonyms": q("SELECT COUNT(*) FROM term WHERE is_preferred = 0"), "abbreviations": q("SELECT COUNT(*) FROM abbreviation"),
                "relationships": q("SELECT COUNT(*) FROM relationship"), "cues": q("SELECT COUNT(*) FROM context_cue"),
                "confusions": q("SELECT COUNT(*) FROM confusion"), "english_words": q("SELECT COUNT(*) FROM english_word"),
                "languages": langs, "sources": q("SELECT COUNT(*) FROM source"),
                "redistributable_terms": q("SELECT COUNT(*) FROM term WHERE redistributable = 1"),
                "private_only_terms": q("SELECT COUNT(*) FROM term WHERE redistributable = 0")}

    def _manifest(self, counts: dict, meta: dict, seconds: float) -> dict:
        digest = hashlib.sha256()
        with self.output.open("rb") as handle:
            while block := handle.read(16 * 1024 * 1024):
                digest.update(block)
        report = []
        for s in self.sources.values():
            lic = license_of(s.source_id)
            report.append({"source_id": s.source_id, "name": s.source_name, "version": s.source_version, "license_name": lic.license_name,
                           "license_status": lic.status, "redistributable": lic.redistribution_allowed, "attribution": lic.attribution,
                           "license_url": lic.license_url})
        return {"name": KNOWLEDGE_NAME, "schema_version": SCHEMA_VERSION, "build_version": self.build_version, "build_date": meta["build_date"],
                "profile": self.profile, "file": self.output.name, "size_bytes": self.output.stat().st_size, "sha256": digest.hexdigest(),
                "build_seconds": round(seconds, 1), "counts": counts, "sources": report}
