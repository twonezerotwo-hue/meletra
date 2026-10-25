"""HGNC approved human gene nomenclature (``hgnc_complete_set.txt``): symbols, names, aliases, previous symbols."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Iterator

from .base import AbbreviationRecord, ConceptRecord, Record, SourceInfo, TermRecord, TerminologySource

SOURCE_ID = "HGNC"
KEEP_LOCUS = ("protein-coding gene", "RNA, micro", "RNA, long non-coding", "pseudogene")


class HgncSource(TerminologySource):
    def __init__(self, path: Path, protein_coding_only: bool = True, limit: int = 0) -> None:
        self.path = Path(path)
        self.protein_coding_only = protein_coding_only
        self.limit = limit
        self.info = SourceInfo(SOURCE_ID, "HGNC gene nomenclature", self._version(), "https://www.genenames.org/download/", "CC0", True, False, "en", "",
                               "approved symbols and names; aliases and previous symbols as synonyms")

    def _version(self) -> str:
        import time
        return time.strftime("%Y-%m-%d", time.localtime(self.path.stat().st_mtime)) if self.path.is_file() else "unknown"

    def verify(self) -> list[str]:
        return [] if self.path.is_file() else [f"HGNC file not found: {self.path} (https://www.genenames.org/download/)"]

    def parse(self) -> Iterator[dict]:
        with self.path.open("r", encoding="utf-8", newline="") as handle:
            for n, row in enumerate(csv.DictReader(handle, delimiter="\t")):
                if self.limit and n >= self.limit:
                    return
                yield row

    def normalize(self, raw: Iterable[dict]) -> Iterator[Record]:
        for row in raw:
            if row.get("status") != "Approved" or not row.get("symbol"):
                continue
            if self.protein_coding_only and row.get("locus_group") != "protein-coding gene":
                continue
            hid, symbol, name = row["hgnc_id"], row["symbol"], row.get("name", "")
            cid = f"HGNC:{hid.split(':')[-1]}"
            yield ConceptRecord(cid, name or symbol, row.get("locus_type", ""), ("genetics", "molecular_biology"), "", "gene", "ONTOLOGY_VERIFIED")
            yield TermRecord(cid, symbol, "en", "gene", True, symbol, hid)
            if name:
                yield TermRecord(cid, name, "en", "preferred", False, symbol, hid)
                yield AbbreviationRecord(symbol, name, cid, "genetics")
            for field, kind in (("alias_symbol", "synonym"), ("prev_symbol", "synonym"), ("alias_name", "synonym"), ("prev_name", "synonym")):
                for alias in (row.get(field) or "").strip('"').split("|"):
                    if alias.strip():
                        yield TermRecord(cid, alias.strip().strip('"'), "en", kind, False, symbol, hid)
