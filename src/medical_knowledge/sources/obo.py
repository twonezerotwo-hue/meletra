"""OBO-format ontologies: Gene Ontology (GO), Human Phenotype Ontology (HPO), ChEBI (lite).

A small streaming ``[Term]`` reader plus one subclass per ontology that decides which terms are worth carrying, which fields of study they belong to and
which synonyms count. Obsolete terms are skipped. Relationships (``is_a``, ``part_of``) between kept terms become ``is_a`` / ``part_of`` relations.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Iterator, Optional

from .base import ConceptRecord, Record, RelationRecord, SourceInfo, TermRecord, TerminologySource

_SYN = re.compile(r'^synonym:\s+"((?:[^"\\]|\\.)*)"\s+(EXACT|RELATED|BROAD|NARROW)?\s*([A-Za-z_:/.\-]*)?\s*\[')
_NAME_OK = re.compile(r"^[A-Za-z][A-Za-z '’\-]*$")


def stanzas(path: Path) -> Iterator[dict]:
    """``[Term]`` stanzas as dicts: id, name, namespace, synonyms [(text, scope, type)], is_a [id], part_of [id], obsolete."""
    cur: Optional[dict] = None
    header: dict = {"data-version": ""}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line.startswith("["):
                if cur is not None:
                    yield cur
                cur = {"id": "", "name": "", "namespace": "", "synonyms": [], "is_a": [], "part_of": [], "obsolete": False} if line == "[Term]" else None
                continue
            if cur is None:
                if line.startswith("data-version:"):
                    header["data-version"] = line.split(":", 1)[1].strip()
                continue
            if line.startswith("id: "):
                cur["id"] = line[4:].strip()
            elif line.startswith("name: "):
                cur["name"] = line[6:].strip()
            elif line.startswith("namespace: "):
                cur["namespace"] = line[11:].strip()
            elif line.startswith("synonym: "):
                m = _SYN.match(line)
                if m:
                    cur["synonyms"].append((m.group(1).replace('\\"', '"'), m.group(2) or "RELATED", m.group(3) or ""))
            elif line.startswith("is_a: "):
                cur["is_a"].append(line[6:].split("!")[0].strip())
            elif line.startswith("relationship: part_of "):
                cur["part_of"].append(line[len("relationship: part_of "):].split("!")[0].strip())
            elif line.startswith("is_obsolete: true"):
                cur["obsolete"] = True
    if cur is not None:
        yield cur


def data_version(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("data-version:"):
                return line.split(":", 1)[1].strip()
            if line.startswith("["):
                break
    return "unknown"


class OboSource(TerminologySource):
    source_id = ""
    source_name = ""
    prefix = ""
    url = ""
    category = "OPEN"
    domains: tuple[str, ...] = ()
    synonym_scopes = ("EXACT",)
    max_tokens = 4
    max_chars = 60

    def __init__(self, path: Path, limit: int = 0) -> None:
        self.path = Path(path)
        self.limit = limit
        version = data_version(self.path) if self.path.is_file() else "unknown"
        self.info = SourceInfo(self.source_id, self.source_name, version, self.url, self.category, True, False, "en", "", f"{self.source_id} {version}")

    def verify(self) -> list[str]:
        return [] if self.path.is_file() else [f"{self.source_name}: file not found: {self.path} ({self.url})"]

    def parse(self) -> Iterator[dict]:
        for n, s in enumerate(stanzas(self.path)):
            if self.limit and n >= self.limit:
                return
            yield s

    # hooks ------------------------------------------------------------------
    def keep(self, s: dict) -> bool:
        return bool(s["id"].startswith(self.prefix) and s["name"] and not s["obsolete"] and self.name_ok(s["name"]))

    def name_ok(self, name: str) -> bool:
        return len(name) <= self.max_chars and len(name.split()) <= self.max_tokens and bool(_NAME_OK.match(name))

    def domains_of(self, s: dict) -> tuple[str, ...]:
        return self.domains

    def synonym_ok(self, text: str, scope: str, kind: str) -> bool:
        return scope in self.synonym_scopes and self.name_ok(text)

    def normalize(self, raw: Iterable[dict]) -> Iterator[Record]:
        kept: dict[str, dict] = {}
        for s in raw:
            if self.keep(s):
                kept[s["id"]] = s
        for sid, s in kept.items():
            domains = self.domains_of(s)
            if not domains:
                continue
            cid = f"{self.prefix.rstrip(':')}:{sid.split(':', 1)[1]}"
            yield ConceptRecord(cid, s["name"], s["namespace"], domains, "", "gene" if self.source_id == "HGNC" else "term", "ONTOLOGY_VERIFIED")
            yield TermRecord(cid, s["name"], "en", "preferred", True, sid, sid)
            for text, scope, kind in s["synonyms"]:
                if text != s["name"] and self.synonym_ok(text, scope, kind):
                    yield TermRecord(cid, text, "en", "abbreviation" if "abbrev" in kind.lower() else "synonym", False, sid, sid)
            for parent in s["is_a"]:
                if parent in kept:
                    yield RelationRecord(cid, "is_a", f"{self.prefix.rstrip(':')}:{parent.split(':', 1)[1]}", 0.8)
            for parent in s["part_of"]:
                if parent in kept:
                    yield RelationRecord(cid, "part_of", f"{self.prefix.rstrip(':')}:{parent.split(':', 1)[1]}", 0.8)


class GoSource(OboSource):
    source_id, source_name, prefix = "GO", "Gene Ontology", "GO:"
    url = "https://purl.obolibrary.org/obo/go/go-basic.obo"
    category = "CC_BY"
    max_tokens = 5
    _FAMILIES = ("regulation of", "positive regulation of", "negative regulation of", "response to", "cellular response to", "detection of", "establishment of",
                 "maintenance of", "organization of", "localization of", "process utilizing", "positive regulation", "negative regulation")

    def keep(self, s: dict) -> bool:
        return super().keep(s) and not s["name"].lower().startswith(self._FAMILIES)             # regulation / response families: thousands of rarely spoken phrases

    def domains_of(self, s: dict) -> tuple[str, ...]:
        return {"cellular_component": ("histology", "molecular_biology"), "molecular_function": ("biochemistry", "molecular_biology"),
                "biological_process": ("physiology", "molecular_biology", "biochemistry")}.get(s["namespace"], ())


class HpoSource(OboSource):
    source_id, source_name, prefix = "HPO", "Human Phenotype Ontology", "HP:"
    url = "https://purl.obolibrary.org/obo/hp.obo"
    category = "CUSTOM_OPEN"
    synonym_scopes = ("EXACT", "RELATED")
    domains = ("pathology", "internal_medicine")

    def domains_of(self, s: dict) -> tuple[str, ...]:
        n = s["name"].lower()
        extra = ("neurology",) if any(k in n for k in ("neuro", "seizure", "ataxia", "cerebr", "brain", "spastic")) else ("cardiology",) if any(k in n for k in ("cardi", "heart", "aort")) else ()
        return self.domains + extra


class ChebiSource(OboSource):
    source_id, source_name, prefix = "CHEBI", "ChEBI (Chemical Entities of Biological Interest)", "CHEBI:"
    url = "https://ftp.ebi.ac.uk/pub/databases/chebi/ontology/chebi_lite.obo"
    category = "CC_BY"
    domains = ("pharmacology", "biochemistry")
    max_tokens = 3
    max_chars = 40
    synonym_scopes = ("EXACT", "RELATED")

    def synonym_ok(self, text: str, scope: str, kind: str) -> bool:
        return "IUPAC" not in kind and super().synonym_ok(text, scope, kind)
