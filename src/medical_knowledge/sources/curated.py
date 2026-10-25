"""The project's own curated vocabulary (``medical_vocab/``) as source ``MLT_CURATED``.

One concept per canonical term (a term listed in several domain folders is ONE concept with several domains). Aliases become terms of that concept
(Latin forms get language ``la``); cue words become ``context_cue`` rows and - when a cue is itself a known term - ``related_to`` relationships;
the listed ASR mishearings (``confusables``) go to the ``confusion`` table (never into terms). Abbreviation lines (canonical = the short form,
aliases = the expansions) become ``abbreviation`` rows and terms of the expansion concept.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Iterator, Optional

from ...medical.lexicon import VOCAB_DIR, Term, _plurals, available_domains, load_domain
from ..domains import DOMAINS, SUPPORT_DOMAINS
from ..normalize.text import compact
from .base import (AbbreviationRecord, ConceptRecord, ConfusionRecord, CueRecord, Record, RelationRecord, SourceInfo, TermRecord, TerminologySource)

SOURCE_ID = "MLT_CURATED"
LATIN_ENDING = re.compile(r"(a|um|us|is|ex|ix|on|ae|i)$", re.I)                           # last word of a Latin-looking term: no English "-s" plural
_KIND_TERM_TYPE = {"term": "synonym", "abbr": "abbreviation", "eponym": "eponym", "latin": "latin", "gene": "gene", "drug": "drug"}


def concept_id(canonical: str) -> str:
    return "MLT:" + compact(canonical)


class CuratedSource(TerminologySource):
    def __init__(self, root: Optional[Path] = None, use_confusables: bool = True) -> None:
        self.root = Path(root or VOCAB_DIR)
        self.use_confusables = use_confusables
        manifest = self.root / "manifest.json"
        version = "1.0.0"
        if manifest.is_file():
            import json

            version = json.loads(manifest.read_text(encoding="utf-8")).get("version", version)
        self.info = SourceInfo(SOURCE_ID, "Project-authored medical vocabulary (medical_vocab)", version, "medical_vocab/SOURCES.json", "PROJECT_OWNED", True,
                               False, "en", "2026-10-11", "hand-written for this project; see medical_vocab/SOURCES.json")

    def verify(self) -> list[str]:
        return [] if self.root.is_dir() and available_domains(self.root) else [f"no vocabulary folders in {self.root}"]

    def parse(self) -> Iterator[dict]:
        for name in available_domains(self.root):
            domain = load_domain(self.root / name)
            for term in domain.terms:
                yield {"domain": name, "term": term}

    # ------------------------------------------------------------------
    def normalize(self, raw: Iterable[dict]) -> Iterator[Record]:
        rows = list(raw)
        # surface -> concept ids, to turn cue words that are terms into relationships
        known: dict[str, set[str]] = {}
        for row in rows:
            term: Term = row["term"]
            if term.kind == "abbr":
                continue
            cid = concept_id(term.canonical)
            for surface in term.surfaces:
                known.setdefault(compact(surface), set()).add(cid)
        related: set[tuple[str, str]] = set()
        for row in rows:
            term = row["term"]
            domain = row["domain"]
            if term.kind == "abbr":
                continue                                                    # abbreviations are attached to their expansions in the second pass
            cid = concept_id(term.canonical)
            yield ConceptRecord(cid, term.canonical, "", (domain,) if domain in DOMAINS else ("general_medical",), "", term.kind, "CURATED")
            language = "la" if term.kind == "latin" else "en"
            yield TermRecord(cid, term.canonical, language, "preferred", True, term.canonical, cid)
            generated = {p.lower() for p in _plurals(term.canonical, term.kind)}                  # plurals the vocabulary loader adds to every ordinary term
            for alias in term.aliases:
                if alias.lower() in generated:
                    if not LATIN_ENDING.search(term.canonical):                                      # "macula densas", "substantia nigras" are not plurals
                        yield TermRecord(cid, alias, language, "plural", False, term.canonical, cid)
                    continue
                yield TermRecord(cid, alias, language, _KIND_TERM_TYPE.get(term.kind, "synonym"), False, term.canonical, cid)
            for cue in term.cues:
                yield CueRecord(cid, cue)
                for other in known.get(compact(cue), ()):
                    if other != cid and (cid, other) not in related:
                        related.add((cid, other))
                        yield RelationRecord(cid, "related_to", other, 0.5)
            if self.use_confusables:
                for heard in term.confusables:
                    yield ConfusionRecord(cid, heard)

        for row in rows:                                                    # second pass: the concepts of the expansions exist now
            if row["term"].kind == "abbr":
                yield from self._abbreviation(row["term"], row["domain"], known)

    def _abbreviation(self, term: Term, domain: str, known: dict[str, set[str]]) -> Iterator[Record]:
        for expansion in term.aliases:
            targets = known.get(compact(expansion), set())
            if len(targets) == 1:                                           # the expansion is already a concept of the vocabulary: attach to it
                cid = next(iter(targets))
                field_domain = ""
            else:
                cid = concept_id(expansion)
                field_domain = "" if domain in SUPPORT_DOMAINS else domain
                yield ConceptRecord(cid, expansion, "", (field_domain,) if field_domain in DOMAINS else ("general_medical",), "", "term", "CURATED")
                yield TermRecord(cid, expansion, "en", "preferred", True, term.canonical, cid)
            yield TermRecord(cid, term.canonical, "en", "abbreviation", False, term.canonical, cid)
            yield AbbreviationRecord(term.canonical, expansion, cid, field_domain)
            for cue in term.cues:
                yield CueRecord(cid, cue)
