"""MeSH (Medical Subject Headings, NLM) - English descriptor file ``desc<year>.xml`` (or ``.gz``).

One concept per descriptor; every non-permuted English term of its concepts becomes a term (entry terms are the synonyms); the lexical tag tells abbreviations
(ABB / ACR) and eponyms (EPO). Fields of study come from the tree numbers (finer) and the UMLS semantic types MeSH attaches to its concepts. Hierarchy
(tree numbers) becomes ``broader`` relationships, "see related" becomes ``related_to``. Check the license status in ``licenses/registry.json`` before shipping:
attribution "Courtesy of the U.S. National Library of Medicine" and a currency notice are conditions of redistribution.
"""

from __future__ import annotations

import gzip
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, Iterator, Optional

from .base import AbbreviationRecord, ConceptRecord, Record, RelationRecord, SourceInfo, TermRecord, TerminologySource
from .umls_types import domains_for

SOURCE_ID = "MESH"
#: tree-number prefix -> fields of study (longest prefix wins); categories that hold no medical-lecture vocabulary map to ()
TREE_DOMAINS: dict[str, tuple[str, ...]] = {
    "A01": ("anatomy",), "A02": ("anatomy", "orthopedics"), "A03": ("anatomy", "gastroenterology"), "A04": ("anatomy", "pulmonology"), "A05": ("anatomy", "nephrology", "reproductive_medicine"),
    "A06": ("anatomy", "endocrinology"), "A07": ("anatomy", "cardiology"), "A08": ("neuroanatomy", "anatomy"), "A09": ("anatomy", "ophthalmology", "otolaryngology"),
    "A10": ("histology",), "A11": ("histology",), "A12": ("physiology",), "A13": (), "A14": ("anatomy", "otolaryngology"), "A15": ("immunology", "hematology"),
    "A16": ("embryology", "anatomy"), "A17": ("anatomy", "dermatology", "histology"), "A18": (), "A19": ("microbiology",), "A20": ("microbiology",), "A21": ("microbiology",),
    "B03": ("microbiology",), "B04": ("microbiology",), "B01": (), "B02": (), "B05": (), "B06": (), "B07": (),
    "C01": ("infectious_diseases", "microbiology"), "C02": ("infectious_diseases", "microbiology"), "C03": ("infectious_diseases", "microbiology"), "C04": ("oncology", "pathology"),
    "C05": ("orthopedics", "rheumatology", "pathology"), "C06": ("gastroenterology", "pathology"), "C07": ("otolaryngology", "pathology"), "C08": ("pulmonology", "pathology"),
    "C09": ("otolaryngology", "pathology"), "C10": ("neurology", "pathology"), "C11": ("ophthalmology", "pathology"), "C12": ("nephrology", "urology", "pathology"),
    "C13": ("reproductive_medicine", "pathology"), "C14": ("cardiology", "pathology"), "C15": ("hematology", "immunology", "pathology"), "C16": ("genetics", "pediatrics", "embryology"),
    "C17": ("dermatology", "pathology"), "C18": ("endocrinology", "biochemistry", "pathology"), "C19": ("endocrinology", "pathology"), "C20": ("immunology", "pathology"),
    "C21": ("emergency_medicine",), "C22": (), "C23": ("internal_medicine", "pathology"), "C24": (), "C25": ("pharmacology", "emergency_medicine"), "C26": ("emergency_medicine", "surgery"),
    "D01": ("biochemistry",), "D02": ("biochemistry", "pharmacology"), "D03": ("biochemistry", "pharmacology"), "D04": ("biochemistry",), "D05": ("biochemistry",), "D06": ("endocrinology", "biochemistry"),
    "D08": ("biochemistry",), "D09": ("biochemistry",), "D10": ("biochemistry",), "D12": ("biochemistry", "molecular_biology"), "D13": ("molecular_biology", "biochemistry"),
    "D20": ("pharmacology",), "D23": ("pharmacology", "physiology"), "D24": ("pharmacology",), "D25": ("pharmacology",), "D26": ("pharmacology",), "D27": ("pharmacology",),
    "E01": ("radiology", "internal_medicine"), "E02": ("internal_medicine", "surgery"), "E03": ("anesthesiology",), "E04": ("surgery",), "E05": ("molecular_biology", "laboratory_medicine"),
    "E06": (), "E07": (), "F01": ("psychiatry", "neuroscience"), "F02": ("psychiatry", "neuroscience"), "F03": ("psychiatry",), "F04": ("psychiatry",), "F05": ("psychiatry",),
    "G01": (), "G02": ("biochemistry",), "G03": ("biochemistry", "physiology"), "G04": ("histology", "biochemistry"), "G05": ("genetics", "molecular_biology"), "G06": ("microbiology",),
    "G07": ("physiology",), "G08": ("reproductive_medicine", "physiology"), "G09": ("physiology",), "G10": ("embryology",), "G11": ("physiology",), "G12": (), "G13": ("genetics",),
    "G14": (), "G15": (), "G16": (), "G17": (),
    "H": (), "I": (), "J": (), "K": (), "L": (), "M": (), "N": (), "V": (), "Z": (),
}
TERM_TYPE = {"ABB": "abbreviation", "ACR": "abbreviation", "EPO": "eponym", "NAM": "name", "TRD": "brand", "LAB": "synonym", "NON": "synonym"}


def tree_domains(trees: Iterable[str]) -> tuple[str, ...]:
    out: list[str] = []
    for tree in trees:
        key = next((tree[:n] for n in (3, 1) if tree[:n] in TREE_DOMAINS), None)
        for d in TREE_DOMAINS.get(key or "", ()):
            if d not in out:
                out.append(d)
    return tuple(out)


def _open(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open("r", encoding="utf-8")


def _text(node: Optional[ET.Element], path: str) -> str:
    found = node.find(path) if node is not None else None
    return (found.text or "").strip() if found is not None else ""


class MeshSource(TerminologySource):
    def __init__(self, path: Path, limit: int = 0) -> None:
        self.path = Path(path)
        self.limit = limit
        year = re.search(r"(20\d\d)", self.path.name)
        self.info = SourceInfo(SOURCE_ID, "Medical Subject Headings (NLM)", year.group(1) if year else "unknown", "https://www.nlm.nih.gov/databases/download/mesh.html",
                               "PUBLIC_DOMAIN_ATTRIBUTION", True, False, "en", year.group(1) + "-01-01" if year else "", "English descriptors and entry terms; permuted terms skipped")

    def verify(self) -> list[str]:
        return [] if self.path.is_file() else [f"MeSH descriptor file not found: {self.path} (https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/)"]

    def parse(self) -> Iterator[dict]:
        count = 0
        with _open(self.path) as handle:
            for _, elem in ET.iterparse(handle, events=("end",)):
                if elem.tag != "DescriptorRecord":
                    continue
                yield self._record(elem)
                elem.clear()
                count += 1
                if self.limit and count >= self.limit:
                    return

    @staticmethod
    def _record(elem: ET.Element) -> dict:
        concepts = []
        for c in elem.findall("ConceptList/Concept"):
            concepts.append({"preferred": c.get("PreferredConceptYN") == "Y", "ui": _text(c, "ConceptUI"), "name": _text(c, "ConceptName/String"),
                             "tuis": [_text(s, "SemanticTypeUI") for s in c.findall("SemanticTypeList/SemanticType")],
                             "terms": [{"ui": _text(t, "TermUI"), "text": _text(t, "String"), "tag": t.get("LexicalTag", ""), "permuted": t.get("IsPermutedTermYN") == "Y",
                                        "preferred": t.get("ConceptPreferredTermYN") == "Y"} for t in c.findall("TermList/Term")]})
        return {"ui": _text(elem, "DescriptorUI"), "name": _text(elem, "DescriptorName/String"),
                "trees": [(t.text or "").strip() for t in elem.findall("TreeNumberList/TreeNumber")], "concepts": concepts,
                "related": [_text(r, "DescriptorReferredTo/DescriptorUI") for r in elem.findall("SeeRelatedList/SeeRelatedDescriptor")],
                "actions": [_text(a, "DescriptorReferredTo/DescriptorUI") for a in elem.findall("PharmacologicalActionList/PharmacologicalAction")]}

    def normalize(self, raw: Iterable[dict]) -> Iterator[Record]:
        records = list(raw)
        tree_owner: dict[str, str] = {}
        for r in records:
            for t in r["trees"]:
                tree_owner[t] = r["ui"]
        known = {r["ui"] for r in records}
        for r in records:
            tuis = [t for c in r["concepts"] for t in c["tuis"]]
            domains = tree_domains(r["trees"]) or domains_for(tuis)
            if not domains:
                continue
            cid = f"MESH:{r['ui']}"
            semantic = next((c["tuis"][0] for c in r["concepts"] if c["tuis"]), "")
            yield ConceptRecord(cid, r["name"], semantic, domains, r["trees"][0] if r["trees"] else "", "term", "ONTOLOGY_VERIFIED")
            yield TermRecord(cid, r["name"], "en", "preferred", True, r["ui"], r["ui"])
            for c in r["concepts"]:
                for t in c["terms"]:
                    if t["permuted"] or not t["text"] or t["text"] == r["name"]:
                        continue
                    kind = TERM_TYPE.get(t["tag"], "synonym")
                    yield TermRecord(cid, t["text"], "en", kind, False, t["ui"], r["ui"])
                    if t["tag"] in ("ABB", "ACR"):
                        yield AbbreviationRecord(t["text"], r["name"], cid, domains[0])
            parents = {tree_owner.get(t.rsplit(".", 1)[0]) for t in r["trees"] if "." in t}
            for p in sorted(x for x in parents if x and x != r["ui"]):
                yield RelationRecord(cid, "broader", f"MESH:{p}", 0.8)
            for other in r["related"]:
                if other in known:
                    yield RelationRecord(cid, "related_to", f"MESH:{other}", 0.5)
            for other in r["actions"]:
                if other in known:
                    yield RelationRecord(cid, "related_to", f"MESH:{other}", 0.6)
