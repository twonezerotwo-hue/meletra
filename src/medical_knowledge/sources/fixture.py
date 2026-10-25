"""Source ``MLT_FIXTURE``: small hand-written additions used by the v4 tests and benchmarks.

1. ``ONTOLOGY``     typed relationships between concepts of the lecture material the project is tested with (is_a / part_of / related_to). They
                    are what "Remak, Schwann, unmyelinated fiber and PNS belong together" means to the scorer.
2. ``HOMOPHONES``   medical words that sound like ordinary English words (cell/sell, lumbar/lumber, vein/vain ...). They are the traps of the
                    false-positive benchmark: the engine must correct them only where the context is medical.

Nothing here is counted as medical coverage; the concepts exist so that the relations and traps are testable.
"""

from __future__ import annotations

from typing import Iterable, Iterator

from ..normalize.text import compact
from .base import ConceptRecord, CueRecord, Record, RelationRecord, SourceInfo, TermRecord, TerminologySource
from .curated import concept_id

SOURCE_ID = "MLT_FIXTURE"

#: (concept, relation, concept)  - names are canonical terms of the curated vocabulary (created here when missing)
ONTOLOGY: list[tuple[str, str, str, str]] = [
    ("Schwann cell", "part_of", "peripheral nervous system", "neuroanatomy"),
    ("Schwann cell", "related_to", "myelin", "neuroanatomy"),
    ("Remak bundle", "related_to", "Schwann cell", "neuroanatomy"),
    ("Remak bundle", "related_to", "unmyelinated axon", "neuroanatomy"),
    ("Remak bundle", "part_of", "peripheral nervous system", "neuroanatomy"),
    ("unmyelinated axon", "part_of", "peripheral nervous system", "neuroanatomy"),
    ("Nissl body", "part_of", "neuron", "histology"),
    ("Nissl body", "related_to", "rough endoplasmic reticulum", "histology"),
    ("Nissl body", "related_to", "ribosome", "histology"),
    ("cresyl violet", "related_to", "Nissl body", "histology"),
    ("tanycyte", "is_a", "ependymal cell", "neuroanatomy"),
    ("tanycyte", "related_to", "third ventricle", "neuroanatomy"),
    ("microglia", "is_a", "glial cell", "neuroanatomy"),
    ("astrocyte", "is_a", "glial cell", "neuroanatomy"),
    ("oligodendrocyte", "is_a", "glial cell", "neuroanatomy"),
    ("node of Ranvier", "part_of", "myelinated axon", "neuroanatomy"),
    ("saltatory conduction", "related_to", "node of Ranvier", "neuroanatomy"),
    ("saltatory conduction", "related_to", "myelin", "neuroanatomy"),
    ("axodendritic synapse", "is_a", "synapse", "neuroanatomy"),
    ("axosomatic synapse", "is_a", "synapse", "neuroanatomy"),
    ("gamma-aminobutyric acid", "is_a", "neurotransmitter", "neuroscience"),
    ("blood-brain barrier", "related_to", "astrocyte", "neuroanatomy"),
    ("blood-brain barrier", "related_to", "endothelial cell", "histology"),
    ("perikaryon", "part_of", "neuron", "neuroanatomy"),
    ("axon hillock", "part_of", "neuron", "neuroanatomy"),
]

#: (medical term, the ordinary English word it sounds like, domain, cue words of its medical use)
HOMOPHONES: list[tuple[str, str, str, tuple[str, ...]]] = [
    ("cell", "sell", "histology", ("nucleus", "membrane", "cytoplasm", "organelle", "tissue")),
    ("lumbar", "lumber", "anatomy", ("vertebra", "puncture", "spine", "nerve", "plexus")),
    ("pleural", "plural", "anatomy", ("lung", "cavity", "effusion", "membrane", "thorax")),
    ("vein", "vain", "anatomy", ("artery", "blood", "vessel", "valve", "venous")),
    ("cord", "chord", "anatomy", ("spinal", "nerve", "umbilical", "vocal", "tract")),
    ("humerus", "humorous", "anatomy", ("arm", "bone", "shoulder", "elbow", "fracture")),
    ("ileum", "ilium", "anatomy", ("small intestine", "jejunum", "bowel", "absorption", "Peyer")),
    ("sinus", "sign us", "anatomy", ("nasal", "venous", "cavity", "paranasal", "dural")),
    ("mucus", "mucous", "histology", ("goblet", "epithelium", "secretion", "membrane", "gland")),
    ("ulna", "ulna", "anatomy", ("radius", "forearm", "elbow", "bone")),
    ("serous", "serious", "histology", ("fluid", "membrane", "gland", "pleura", "peritoneum")),
    ("cilia", "silly", "histology", ("epithelium", "motile", "flagellum", "beat", "tracheal")),
    ("fascia", "fashion", "anatomy", ("muscle", "connective tissue", "layer", "plantar", "sheath")),
    ("dura", "doura", "neuroanatomy", ("mater", "meninges", "arachnoid", "sinus", "brain")),
    ("pons", "ponds", "neuroanatomy", ("brainstem", "medulla", "midbrain", "cerebellum", "nuclei")),
    ("glia", "glee a", "neuroanatomy", ("neuron", "astrocyte", "microglia", "oligodendrocyte", "support")),
    ("soma", "sofa", "neuroanatomy", ("neuron", "cell body", "dendrite", "axon", "perikaryon")),
    ("axon", "axe on", "neuroanatomy", ("neuron", "myelin", "dendrite", "terminal", "conduction")),
    ("myelin", "my a lin", "neuroanatomy", ("sheath", "oligodendrocyte", "Schwann", "axon", "insulation")),
    ("synapse", "sin apps", "neuroscience", ("neuron", "neurotransmitter", "cleft", "vesicle", "receptor")),
]


class FixtureSource(TerminologySource):
    info = SourceInfo(SOURCE_ID, "Synthetic test fixtures (relations and homophone traps)", "1.0.0", "", "PROJECT_OWNED", True, False, "en", "2026-10-13",
                      "authored for the v4 tests; not medical coverage")

    def parse(self) -> Iterator[dict]:
        for a, rel, b, domain in ONTOLOGY:
            yield {"kind": "relation", "a": a, "rel": rel, "b": b, "domain": domain}
        for term, word, domain, cues in HOMOPHONES:
            yield {"kind": "homophone", "term": term, "word": word, "domain": domain, "cues": cues}

    def normalize(self, raw: Iterable[dict]) -> Iterator[Record]:
        seen: set[str] = set()
        for row in raw:
            if row["kind"] == "relation":
                for name in (row["a"], row["b"]):
                    cid = concept_id(name)
                    if cid not in seen:                                  # the curated vocabulary normally has it; this keeps the relation from dangling
                        seen.add(cid)
                        yield ConceptRecord(cid, name, "", (row["domain"],), "", "term", "CURATED")
                        yield TermRecord(cid, name, "en", "preferred", True, name, cid)
                yield RelationRecord(concept_id(row["a"]), row["rel"], concept_id(row["b"]), 1.0)
            else:
                cid = concept_id(row["term"])
                if cid not in seen:
                    seen.add(cid)
                    yield ConceptRecord(cid, row["term"], "", (row["domain"],), "", "term", "CURATED")
                    yield TermRecord(cid, row["term"], "en", "preferred", True, row["term"], cid)
                for cue in row["cues"]:
                    yield CueRecord(cid, cue)
