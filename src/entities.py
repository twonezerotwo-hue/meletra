"""Medical entity classification (rule based): which kind of thing is a detected term?

Categories: anatomy, nerve, tract, nucleus, vessel, muscle, bone, enzyme, receptor, hormone,
neurotransmitter, drug, disease, pathology, gene, protein, cell type, biochemical pathway.

A term may carry two labels (``substantia nigra`` -> ``nucleus/anatomy``). The rules are heuristics over the
term's words and endings; ``rule`` in the output says which one fired so a reviewer can judge it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from .knowledge_entities import (
    BONE_WORDS, CELL_TYPE_SUFFIXES, CELL_TYPE_WORDS, DISEASE_SUFFIXES, DISEASE_WORDS, DRUG_SUFFIXES, ENZYME_EXCLUDE, NOT_DRUGS,
    ENZYME_WORDS, GENE_WORDS, HORMONES, MUSCLE_WORDS, NERVE_WORDS, NEUROTRANSMITTERS, NUCLEUS_WORDS,
    PATHOLOGY_WORDS, PATHWAY_SUFFIXES, PATHWAY_WORDS, PROTEIN_WORDS, RECEPTOR_WORDS, TRACT_WORDS, VESSEL_WORDS,
)
from .vocabulary import DEFAULT_TERMS_DIR, load_terms, normalize_key, tokenize

ANATOMY_SUBTYPES = {"nerve", "tract", "nucleus", "vessel", "muscle", "bone"}
_GENE_SYMBOL = re.compile(r"^(?=.*\d)[A-Za-z0-9\-]{2,9}$")


@dataclass
class Classification:
    categories: list[str]
    rule: str

    @property
    def label(self) -> str:
        return "/".join(self.categories)


def classify_term(term: str, known_drugs: frozenset[str] = frozenset(), vocab_files: Sequence[str] = ()) -> Classification:
    """Category labels for ``term`` (``["unclassified"]`` if no rule fits)."""
    words = [w.casefold() for w in tokenize(term)]
    raw_words = tokenize(term)
    if not words:
        return Classification(["unclassified"], "empty")
    head, whole = words[-1], " ".join(words)
    key = normalize_key(term)

    def labelled(cat: str, rule: str) -> Classification:
        return Classification([cat, "anatomy"] if cat in ANATOMY_SUBTYPES else [cat], rule)

    # 1. the head noun decides first ("dopamine receptor" is a receptor, not a transmitter)
    if head in RECEPTOR_WORDS:
        return labelled("receptor", "head noun 'receptor'")
    if head in TRACT_WORDS or any(w in {"tract", "fasciculus", "lemniscus"} for w in words):
        return labelled("tract", "tract/fasciculus/lemniscus")
    if head == "pathway" and "neuroscience" in vocab_files:       # "nigrostriatal pathway"; metabolic ones fall through
        return labelled("tract", "neural pathway")
    if head in NERVE_WORDS:
        return labelled("nerve", f"head noun '{head}'")
    if head in VESSEL_WORDS or any(w in {"artery", "vein", "aorta"} for w in words):
        return labelled("vessel", "vessel word")
    if head in MUSCLE_WORDS or any(w in MUSCLE_WORDS for w in words):
        return labelled("muscle", "muscle word")
    if head in BONE_WORDS or any(w in BONE_WORDS for w in words):
        return labelled("bone", "bone word")
    if head in {"nucleus", "nuclei"} or any(w in NUCLEUS_WORDS for w in words):
        return labelled("nucleus", "nucleus word")
    # 2. enzymes
    if head in ENZYME_WORDS or any(w in ENZYME_WORDS for w in words) or (
            head.endswith("ase") and len(head) > 5 and head not in ENZYME_EXCLUDE):
        return Classification(["enzyme"], "enzyme word / -ase ending")
    # 3. exact membership: hormones, neurotransmitters
    if whole in NEUROTRANSMITTERS or head in NEUROTRANSMITTERS:
        return Classification(["neurotransmitter"], "known neurotransmitter")
    if whole in HORMONES or head in HORMONES:
        return Classification(["hormone"], "known hormone")
    # 4. cell types
    if head in CELL_TYPE_WORDS or head.endswith(CELL_TYPE_SUFFIXES) or "cell" in words or "cells" in words:
        return Classification(["cell type"], "cell-type word / -cyte,-blast,-clast")
    # 5. genes and proteins
    if any(_GENE_SYMBOL.match(w) and w.upper() == w for w in raw_words) or head in GENE_WORDS:
        return Classification(["gene"], "gene symbol / gene word")
    if head in PROTEIN_WORDS or head.endswith("globin") or any(w in PROTEIN_WORDS for w in words):
        return Classification(["protein"], "protein word")
    # 6. biochemical pathways
    if head in PATHWAY_WORDS or head.endswith(PATHWAY_SUFFIXES) or any(w in {"cycle", "pathway", "shunt"} for w in words):
        return Classification(["biochemical pathway"], "pathway word")
    # 7. drugs
    if whole not in NOT_DRUGS and (
            key in known_drugs or whole in known_drugs or (len(head) >= 7 and head.endswith(DRUG_SUFFIXES))):
        return Classification(["drug"], "known drug / drug-name ending")
    # 8. disease vs. pathological process
    if head in DISEASE_WORDS or (len(head) > 5 and head.endswith(DISEASE_SUFFIXES)):
        return Classification(["disease"], "disease word / ending")
    if head in PATHOLOGY_WORDS or any(w in PATHOLOGY_WORDS for w in words):
        return Classification(["pathology"], "pathological process")
    if any(f in vocab_files for f in ("anatomy", "neuroscience", "histology")):
        return Classification(["anatomy"], "from an anatomical vocabulary file")
    return Classification(["unclassified"], "no rule matched")


@dataclass
class Entity:
    term: str
    categories: list[str]
    rule: str
    count: int
    first_start: float
    language_form: str = "en"
    english: Optional[str] = None
    turkish: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {"term": self.term, "category": "/".join(self.categories), "categories": self.categories, "rule": self.rule,
                "count": self.count, "first_start": round(self.first_start, 2), "language_form": self.language_form,
                "english": self.english, "turkish": self.turkish}


def known_drug_names(terms_dir: Path = DEFAULT_TERMS_DIR) -> frozenset[str]:
    """Single-word pharmacology entries that are drug names or drug classes (used to label drugs)."""
    return frozenset(normalize_key(t) for t in load_terms("pharmacology", None, terms_dir) if len(tokenize(t)) <= 2)


def build_entities(term_stats: Sequence[Any], terms_dir: Path = DEFAULT_TERMS_DIR, pairs: Any = None) -> list[Entity]:
    """Classify the detected terms (``TermStat`` objects from medical_report)."""
    drugs = known_drug_names(terms_dir)
    file_of: dict[str, list[str]] = {}
    from .vocabulary import available_subjects

    for subject in available_subjects(terms_dir):
        for term in load_terms(subject, None, terms_dir):
            file_of.setdefault(normalize_key(term), []).append(subject)
    entities: list[Entity] = []
    for stat in term_stats:
        english = pairs.to_english(stat.term) if pairs is not None else None
        lookup = english or stat.term
        result = classify_term(lookup, drugs, file_of.get(normalize_key(lookup), []))
        form = "tr" if (pairs is not None and english and normalize_key(english) != normalize_key(stat.term)) else "en"
        entities.append(Entity(
            term=stat.term, categories=result.categories, rule=result.rule, count=stat.count, first_start=stat.first_start,
            language_form=form, english=english or (stat.term if form == "en" else None),
            turkish=(pairs.to_turkish(lookup) if pairs is not None else None)))
    return entities


def entities_summary(entities: Sequence[Entity]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for e in entities:
        counts[e.categories[0]] = counts.get(e.categories[0], 0) + e.count
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
