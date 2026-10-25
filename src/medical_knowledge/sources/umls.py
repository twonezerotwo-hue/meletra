"""UMLS Metathesaurus importer (source ``UMLS``, sub-sources ``UMLS/<SAB>``).

Reads a LOCALLY LICENSED copy of the Metathesaurus (the ``META`` folder with ``MRCONSO.RRF``, ``MRSTY.RRF``, optionally ``MRREL.RRF`` and ``MRSAB.RRF``).
It never downloads anything and never works around the license agreement: without the files it stops with instructions (``CredentialsRequired``).
UMLS-derived data is PRIVATE (``full`` profile only); each vocabulary (SAB) is its own source row, so a later review can allow single vocabularies.

Streaming and constant memory: MRCONSO and MRSTY are both sorted by CUI, so they are merged line by line (no per-CUI dictionaries); the CUIs that were
imported are kept as a sorted ``array('I')`` (4 bytes each) for the relationship pass.
"""

from __future__ import annotations

import re
from array import array
from bisect import bisect_left
from pathlib import Path
from typing import Iterable, Iterator, Optional

from ..normalize.text import normalize
from .base import (AbbreviationRecord, ConceptRecord, CredentialsRequired, Record, RelationRecord, SourceInfo, TermRecord, TerminologySource)
from .umls_types import domains_for, type_name

LANGUAGES = {"ENG": "en", "LAT": "la", "GER": "de", "HUN": "hu", "TUR": "tr", "SPA": "es", "FRE": "fr", "ITA": "it", "POR": "pt", "DUT": "nl", "RUS": "ru",
             "POL": "pl", "CZE": "cs", "SWE": "sv", "NOR": "no", "DAN": "da", "FIN": "fi", "JPN": "ja", "KOR": "ko", "CHI": "zh", "GRE": "el", "HEB": "he"}
DEFAULT_LANGUAGES = ("ENG", "LAT", "GER", "HUN", "TUR")
ABBREVIATION_TTY = {"AB", "ACR", "AA", "ABB"}
HOW_TO_GET = (
    "UMLS is licensed data and is NOT downloaded by this program. To use it: (1) create a free UMLS Terminology Services account and accept the UMLS "
    "Metathesaurus License Agreement at https://uts.nlm.nih.gov/ ; (2) download the 'UMLS Metathesaurus Full Subset' (or run MetamorphoSys on the Level 0-2 "
    "release) from https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html ; (3) point this importer at the META folder: "
    "python tools/import_umls.py --source <path-to-META>. Restricted vocabularies (e.g. SNOMED CT) stay private and never enter the public database.")


def _find(root: Path, name: str) -> Optional[Path]:
    for candidate in (root / name, root / "META" / name, root / "2" / name):
        if candidate.is_file():
            return candidate
    found = next(root.rglob(name), None) if root.is_dir() else None
    return found


def rows(path: Path, fields: int = 0) -> Iterator[list[str]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("|")
            if fields and len(parts) < fields:
                continue
            yield parts


def cui_number(cui: str) -> int:
    return int(cui[1:]) if len(cui) > 1 and cui[0] == "C" and cui[1:].isdigit() else -1


class UmlsSource(TerminologySource):
    def __init__(self, root: Path, languages: Iterable[str] = DEFAULT_LANGUAGES, sabs: Optional[Iterable[str]] = None, with_relations: bool = True,
                 max_related: int = 30, limit_concepts: int = 0) -> None:
        self.root = Path(root)
        self.languages = {l.upper() for l in languages}
        self.sabs = {s.upper() for s in sabs} if sabs else None
        self.with_relations = with_relations
        self.max_related = max_related
        self.limit_concepts = limit_concepts
        self.files = {name: _find(self.root, name) for name in ("MRCONSO.RRF", "MRSTY.RRF", "MRREL.RRF", "MRSAB.RRF")}
        match = re.search(r"(20\d\d[A-D][A-D])", str(self.root))
        self.info = SourceInfo("UMLS", "UMLS Metathesaurus (NLM)", match.group(1) if match else "unknown", "https://uts.nlm.nih.gov/", "UMLS_LICENSED", False, True,
                               "multi", "", "licensed local copy; private use only")
        self.skipped_types = 0
        self.imported: array = array("I")
        self.sab_info: dict[str, tuple[str, str, str]] = {}               # SAB -> (name, version, license category SRL)

    def download(self, destination: Path) -> None:
        raise CredentialsRequired(HOW_TO_GET)

    def verify(self) -> list[str]:
        problems = []
        if not self.root.is_dir():
            problems.append(f"{self.root} is not a folder. " + HOW_TO_GET)
            return problems
        for name in ("MRCONSO.RRF", "MRSTY.RRF"):
            path = self.files[name]
            if path is None:
                problems.append(f"{name} not found under {self.root}. " + HOW_TO_GET)
                continue
            first = next(rows(path), None)
            if first is None or len(first) < (18 if name == "MRCONSO.RRF" else 6):
                problems.append(f"{path.name} does not look like an RRF file ({len(first or [])} fields in the first line)")
        return problems

    # ------------------------------------------------------------------ parse: MRCONSO + MRSTY merged by CUI
    def _sabs(self) -> None:
        path = self.files["MRSAB.RRF"]
        if path is None:
            return
        for r in rows(path, 14):
            self.sab_info.setdefault(r[3].upper(), (r[4], r[6], r[13]))

    def parse(self) -> Iterator[dict]:
        self._sabs()
        sty = rows(self.files["MRSTY.RRF"], 4)
        pending = next(sty, None)
        current: Optional[str] = None
        atoms: list[list[str]] = []
        count = 0

        def finish(cui: str, group: list[list[str]]) -> Optional[dict]:
            nonlocal pending
            tuis: list[str] = []
            while pending is not None and pending[0] < cui:                # types of concepts that have no (kept) atoms
                pending = next(sty, None)
            while pending is not None and pending[0] == cui:
                tuis.append(pending[1])
                pending = next(sty, None)
            return {"cui": cui, "atoms": group, "tuis": tuis}

        for r in rows(self.files["MRCONSO.RRF"], 17):
            lat, suppress, sab = r[1], r[16], r[11].upper()
            if lat not in self.languages or suppress in ("O", "Y") or (self.sabs and sab not in self.sabs):
                continue
            if r[0] != current:
                if current is not None and atoms:
                    yield finish(current, atoms)
                    count += 1
                    if self.limit_concepts and count >= self.limit_concepts:
                        return
                current, atoms = r[0], []
            atoms.append(r)
        if current is not None and atoms and not (self.limit_concepts and count >= self.limit_concepts):
            yield finish(current, atoms)

    # ------------------------------------------------------------------ normalize
    def normalize(self, raw: Iterable[dict]) -> Iterator[Record]:
        announced: set[str] = set()

        def announce(sab: str) -> Iterator[SourceInfo]:
            if sab not in announced:
                announced.add(sab)
                name, version, category = self.sab_info.get(sab, (sab, self.info.source_version, "?"))
                yield SourceInfo(f"UMLS/{sab}", f"{name} (via UMLS)", version or self.info.source_version, "https://uts.nlm.nih.gov/", f"UMLS_LICENSED_SRL{category}",
                                 False, True, "multi", "", f"vocabulary {sab} inside UMLS {self.info.source_version}; UMLS license category {category}")

        for group in raw:
            cui, atoms, tuis = group["cui"], group["atoms"], group["tuis"]
            domains = domains_for(tuis)
            if not domains:
                self.skipped_types += 1
                continue
            preferred = next((a for a in atoms if a[1] == "ENG" and a[2] == "P" and a[4] == "PF" and a[6] == "Y"), None) \
                or next((a for a in atoms if a[1] == "ENG"), atoms[0])
            cid = f"UMLS:{cui}"
            number = cui_number(cui)
            if number >= 0:
                self.imported.append(number)
            yield ConceptRecord(cid, preferred[14], type_name(tuis[0]) if tuis else "", domains, "", "term", "ONTOLOGY_VERIFIED")
            for a in atoms:
                sab = a[11].upper()
                yield from announce(sab)
                lang = LANGUAGES.get(a[1], a[1].lower())
                is_pref = a is preferred
                tty = a[12]
                term_type = "preferred" if is_pref else ("abbreviation" if tty in ABBREVIATION_TTY else "synonym")
                yield TermRecord(cid, a[14], lang, term_type, is_pref, a[13], a[9] or a[10] or cui, 0, 0.0, "", f"UMLS/{sab}")
                if tty in ABBREVIATION_TTY and lang == "en" and a[14] != preferred[14]:
                    yield AbbreviationRecord(a[14], preferred[14], cid, domains[0], lang, f"UMLS/{sab}")
        if self.with_relations and self.files["MRREL.RRF"] is not None:
            yield from self._relations(announce)

    def _known(self, cui: str) -> bool:
        number = cui_number(cui)
        i = bisect_left(self.imported, number)
        return i < len(self.imported) and self.imported[i] == number

    def _relations(self, announce) -> Iterator[Record]:
        current, used = None, 0
        for r in rows(self.files["MRREL.RRF"], 15):
            rel, suppress = r[3], r[14]
            if rel not in ("PAR", "RB", "RO") or suppress in ("O", "Y") or r[0] == r[4]:
                continue
            if r[0] != current:
                current, used = r[0], 0
            if used >= self.max_related or not self._known(r[0]) or not self._known(r[4]):
                continue
            sab = r[10].upper()
            if self.sabs and sab not in self.sabs:
                continue
            relation = {"PAR": {"isa": "is_a", "part_of": "part_of"}.get(r[7], "broader"), "RB": "broader", "RO": "related_to"}[rel]
            yield from announce(sab)
            yield RelationRecord(f"UMLS:{r[0]}", relation, f"UMLS:{r[4]}", 0.4 if rel == "RO" else 0.8, f"UMLS/{sab}")
            used += 1
