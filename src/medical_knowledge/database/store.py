"""Opening, version checking and health checking of a Medical Knowledge database."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .schema import KNOWLEDGE_NAME, REQUIRED_TABLES, SCHEMA_VERSION


class KnowledgeError(Exception):
    """The knowledge database is missing, damaged or not usable."""


class IncompatibleKnowledgeBase(KnowledgeError):
    def __init__(self, found: Optional[str], expected: int) -> None:
        super().__init__(f"the medical knowledge database has schema version {found}, this program needs {expected}")
        self.found, self.expected = found, expected


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


def open_readonly(path: Path) -> sqlite3.Connection:
    path = Path(path)
    if not path.is_file():
        raise KnowledgeError(f"medical knowledge database not found: {path}")
    try:
        db = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, check_same_thread=False)
        db.execute("PRAGMA query_only=ON")
        db.execute("PRAGMA mmap_size=0")
        db.execute("PRAGMA cache_size=-24576")
        meta = dict(db.execute("SELECT key, value FROM meta").fetchall())
    except sqlite3.DatabaseError as exc:
        raise KnowledgeError(f"the medical knowledge database cannot be read: {exc}") from exc
    if meta.get("schema_version") != str(SCHEMA_VERSION):
        db.close()
        raise IncompatibleKnowledgeBase(meta.get("schema_version"), SCHEMA_VERSION)
    return db


def read_meta(db: sqlite3.Connection) -> dict[str, str]:
    return dict(db.execute("SELECT key, value FROM meta").fetchall())


def health(db: sqlite3.Connection) -> list[Check]:
    """The checks behind ``--medical-knowledge-info`` and the self-test. Every check reports instead of raising."""
    checks: list[Check] = []

    def run(name: str, fn) -> None:
        try:
            ok, detail = fn()
            checks.append(Check(name, bool(ok), detail))
        except Exception as exc:  # noqa: BLE001 - a failed check is a result
            checks.append(Check(name, False, f"{type(exc).__name__}: {exc}"))

    meta = read_meta(db)
    run("schema version", lambda: (meta.get("schema_version") == str(SCHEMA_VERSION), f"schema {meta.get('schema_version')} (need {SCHEMA_VERSION})"))
    run("name", lambda: (meta.get("name") == KNOWLEDGE_NAME, str(meta.get("name"))))
    names = {r[0] for r in db.execute("SELECT name FROM sqlite_master")}
    run("tables", lambda: (all(t in names for t in REQUIRED_TABLES), "all tables present" if all(t in names for t in REQUIRED_TABLES)
                           else "missing: " + ", ".join(t for t in REQUIRED_TABLES if t not in names)))
    run("integrity", lambda: (db.execute("PRAGMA quick_check").fetchone()[0] == "ok", "quick_check ok"))

    def exact() -> tuple[bool, str]:
        row = db.execute("SELECT text, compact_text FROM term ORDER BY term_id LIMIT 1 OFFSET (SELECT COUNT(*)/2 FROM term)").fetchone()
        if row is None:
            return False, "no terms"
        hit = db.execute("SELECT term_id FROM term WHERE compact_text = ?", (row[1],)).fetchone()
        return hit is not None, f"exact lookup of '{row[0]}'"

    def fts() -> tuple[bool, str]:
        row = db.execute("SELECT normalized_text FROM term ORDER BY term_id LIMIT 1 OFFSET (SELECT COUNT(*)/3 FROM term)").fetchone()
        word = row[0].split()[0]
        n = db.execute("SELECT COUNT(*) FROM term_fts WHERE term_fts MATCH ?", ('"' + word + '"',)).fetchone()[0]
        return n > 0, f"full-text search for '{word}': {n} terms"

    def phonetic() -> tuple[bool, str]:
        row = db.execute("SELECT dm, sk FROM term WHERE length(dm) BETWEEN 4 AND 20 ORDER BY term_id LIMIT 1 OFFSET (SELECT COUNT(*)/4 FROM term)").fetchone()
        n = db.execute("SELECT COUNT(*) FROM term WHERE sk = ?", (row[1],)).fetchone()[0]
        neighbour = row[0][:-1]
        m = db.execute("SELECT COUNT(*) FROM phonetic_del WHERE key = ?", (neighbour,)).fetchone()[0]
        return n > 0 and m > 0, f"skeleton '{row[1]}': {n} terms; Double Metaphone '{row[0]}', one letter fewer ('{neighbour}'): {m} entries"

    def relation() -> tuple[bool, str]:
        n = db.execute("SELECT COUNT(*) FROM relationship").fetchone()[0]
        return n > 0, f"{n} relationships"

    def licenses() -> tuple[bool, str]:
        rows = db.execute("SELECT source_id, license_status, license_category FROM source").fetchall()
        bad = [r[0] for r in rows if not r[1] or not r[2]]
        return bool(rows) and not bad, f"{len(rows)} sources with license metadata" if rows and not bad else f"missing license metadata: {bad or 'no sources'}"

    def provenance() -> tuple[bool, str]:
        n = db.execute("SELECT COUNT(*) FROM term t WHERE NOT EXISTS (SELECT 1 FROM term_source ts WHERE ts.term_id = t.term_id)").fetchone()[0]
        return n == 0, f"{n} terms without provenance"

    def public_clean() -> tuple[bool, str]:
        if meta.get("profile") != "public":
            return True, f"profile {meta.get('profile')} (not a public database)"
        n = db.execute("SELECT COUNT(*) FROM source WHERE redistribution_allowed IS NOT 1").fetchone()[0]
        t = db.execute("SELECT COUNT(*) FROM term WHERE redistributable IS NOT 1").fetchone()[0]
        s = db.execute("SELECT COUNT(*) FROM term_source ts JOIN source s ON s.source_id = ts.source_id WHERE s.redistribution_allowed IS NOT 1").fetchone()[0]
        return n == 0 and t == 0 and s == 0, f"{n} non-redistributable sources, {t} terms not marked redistributable, {s} term-source links to a restricted source in a public database"

    for name, fn in (("exact index", exact), ("full-text index", fts), ("phonetic index", phonetic), ("relationships", relation),
                     ("license metadata", licenses), ("provenance", provenance), ("public license filter", public_clean)):
        run(name, fn)
    return checks
