"""Source ``SYNTHETIC_SCALE``: random Greek/Latin-looking terms to measure retrieval at millions of terms. NOT medical knowledge; private by registry."""

from __future__ import annotations

import random
from typing import Iterable, Iterator

from .base import ConceptRecord, Record, SourceInfo, TermRecord, TerminologySource

ROOTS = ("cardi", "neur", "gastr", "hepat", "nephr", "pulmon", "derm", "oste", "myo", "arthr", "cyt", "hist", "angi", "enter", "gloss", "pneum", "thromb", "leuk",
         "erythr", "lip", "gluc", "ren", "cerebr", "spin", "vascul", "thyr", "adren", "gonad", "ocul", "ot", "rhin", "laryng", "bronch", "pancreat", "splen", "lymph",
         "immun", "bacteri", "vir", "fung", "onc", "path", "pharm", "tox", "hem", "chol", "sten", "ather", "isch", "emb", "fibr", "scler", "ectas", "plas", "troph")
LINKS = ("o", "i", "o", "o", "")
ENDS = ("itis", "osis", "oma", "pathy", "ectomy", "plasty", "scopy", "algia", "emia", "uria", "penia", "genesis", "lysis", "stomy", "tomy", "cele", "rrhea", "megaly",
        "ia", "ic", "al", "ase", "in", "ide", "one", "ol")
ADJ = ("acute", "chronic", "benign", "primary", "congenital", "left", "right", "hereditary")
DOMAINS = ("anatomy", "pathology", "pharmacology", "biochemistry", "internal_medicine", "histology", "physiology", "genetics")


def make_term(rnd: random.Random) -> str:
    w = "".join(rnd.choice(ROOTS) + rnd.choice(LINKS) for _ in range(rnd.randint(1, 3)))
    w = w.rstrip("aeiou") + rnd.choice(ENDS)
    return rnd.choice(ADJ) + " " + w if rnd.random() < 0.35 else w


class SyntheticSource(TerminologySource):
    def __init__(self, count: int, seed: int = 11) -> None:
        self.count, self.seed = count, seed
        self.info = SourceInfo("SYNTHETIC_SCALE", "Synthetic terms (benchmark)", f"{count}-{seed}", "", "PROJECT_OWNED", False, False, "en", "", "benchmark only")

    def parse(self) -> Iterator[dict]:
        rnd = random.Random(self.seed)
        seen: set[str] = set()
        while len(seen) < self.count:
            w = make_term(rnd)
            if w not in seen:
                seen.add(w)
                yield {"term": w, "domain": rnd.choice(DOMAINS)}

    def normalize(self, raw: Iterable[dict]) -> Iterator[Record]:
        for i, row in enumerate(raw):
            cid = f"SYN:{i}"
            yield ConceptRecord(cid, row["term"], "", (row["domain"],), "", "term", "UNVERIFIED_CANDIDATE")
            yield TermRecord(cid, row["term"], "en", "preferred", True, str(i), cid)
