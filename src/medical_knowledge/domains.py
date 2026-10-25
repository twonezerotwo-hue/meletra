"""Domain taxonomy: the medical-school / clinical fields a concept can belong to (a concept may belong to several).

Identifiers equal the folder names of ``medical_vocab/`` where a folder exists, so the curated vocabulary maps without a table. ``related`` lists
the fields whose terms are also plausible in a lecture of this field (used to widen the candidate domains, never to decide a correction).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Domain:
    id: str
    title: str
    group: str                      # basic | clinical | specialized | language | support
    related: tuple[str, ...] = ()


def _d(id: str, title: str, group: str, *related: str) -> Domain:
    return Domain(id, title, group, tuple(related))


_ALL = [
    # ---- basic sciences
    _d("anatomy", "Anatomy", "basic", "histology", "embryology", "physiology", "radiology", "surgery"),
    _d("neuroanatomy", "Neuroanatomy", "basic", "neuroscience", "histology", "anatomy", "physiology", "embryology", "neurology", "latin"),
    _d("histology", "Histology", "basic", "anatomy", "pathology", "embryology", "physiology", "molecular_biology"),
    _d("embryology", "Embryology", "basic", "anatomy", "histology", "genetics", "reproductive_medicine"),
    _d("physiology", "Physiology", "basic", "biochemistry", "anatomy", "neuroscience", "pharmacology", "internal_medicine"),
    _d("biochemistry", "Biochemistry", "basic", "molecular_biology", "physiology", "genetics", "pharmacology", "laboratory_medicine"),
    _d("molecular_biology", "Molecular biology", "basic", "biochemistry", "genetics", "immunology", "microbiology"),
    _d("genetics", "Genetics", "basic", "molecular_biology", "embryology", "clinical_genetics"),
    _d("microbiology", "Microbiology", "basic", "immunology", "infectious_diseases", "molecular_biology"),
    _d("immunology", "Immunology", "basic", "microbiology", "hematology", "rheumatology", "molecular_biology"),
    _d("pathology", "Pathology", "basic", "histology", "internal_medicine", "oncology", "laboratory_medicine"),
    _d("pharmacology", "Pharmacology", "basic", "physiology", "biochemistry", "internal_medicine", "psychiatry"),
    _d("neuroscience", "Neuroscience", "basic", "neuroanatomy", "physiology", "neurology", "psychiatry", "pharmacology"),
    # ---- clinical
    _d("internal_medicine", "Internal medicine", "clinical", "cardiology", "pulmonology", "gastroenterology", "nephrology", "endocrinology", "hematology",
       "rheumatology", "infectious_diseases", "physiology", "pathology"),
    _d("cardiology", "Cardiology", "clinical", "internal_medicine", "physiology", "pharmacology", "interventional_cardiology", "radiology"),
    _d("neurology", "Neurology", "clinical", "neuroanatomy", "neuroscience", "neurosurgery", "neuropathology", "neuroradiology", "psychiatry"),
    _d("neurosurgery", "Neurosurgery", "clinical", "neurology", "neuroanatomy", "surgery", "neuroradiology"),
    _d("surgery", "General surgery", "clinical", "anatomy", "orthopedics", "anesthesiology", "emergency_medicine"),
    _d("orthopedics", "Orthopedics", "clinical", "anatomy", "surgery", "radiology"),
    _d("radiology", "Radiology", "clinical", "anatomy", "neuroradiology", "interventional_radiology", "nuclear_medicine"),
    _d("anesthesiology", "Anesthesiology", "clinical", "pharmacology", "physiology", "surgery", "emergency_medicine"),
    _d("emergency_medicine", "Emergency medicine", "clinical", "internal_medicine", "surgery", "anesthesiology"),
    _d("pediatrics", "Pediatrics", "clinical", "internal_medicine", "genetics", "embryology", "infectious_diseases"),
    _d("reproductive_medicine", "Obstetrics, gynecology and reproductive medicine", "clinical", "embryology", "endocrinology", "genetics"),
    _d("psychiatry", "Psychiatry", "clinical", "neuroscience", "pharmacology", "neurology"),
    _d("ophthalmology", "Ophthalmology", "clinical", "anatomy", "neuroanatomy", "histology"),
    _d("otolaryngology", "Otolaryngology (ENT)", "clinical", "anatomy", "neuroanatomy", "surgery"),
    _d("urology", "Urology", "clinical", "nephrology", "surgery", "anatomy"),
    _d("nephrology", "Nephrology", "clinical", "physiology", "internal_medicine", "urology", "laboratory_medicine"),
    _d("pulmonology", "Pulmonology", "clinical", "physiology", "internal_medicine", "radiology"),
    _d("gastroenterology", "Gastroenterology", "clinical", "physiology", "internal_medicine", "pathology"),
    _d("endocrinology", "Endocrinology", "clinical", "physiology", "biochemistry", "internal_medicine", "reproductive_medicine"),
    _d("hematology", "Hematology", "clinical", "immunology", "internal_medicine", "oncology", "laboratory_medicine"),
    _d("oncology", "Oncology", "clinical", "pathology", "hematology", "pharmacology", "molecular_biology"),
    _d("rheumatology", "Rheumatology", "clinical", "immunology", "internal_medicine", "orthopedics"),
    _d("infectious_diseases", "Infectious diseases", "clinical", "microbiology", "immunology", "internal_medicine", "pharmacology"),
    _d("dermatology", "Dermatology", "clinical", "histology", "pathology", "immunology"),
    _d("laboratory_medicine", "Laboratory medicine", "clinical", "biochemistry", "hematology", "pathology", "microbiology"),
    # ---- specialized
    _d("neuropathology", "Neuropathology", "specialized", "neurology", "pathology", "neuroanatomy"),
    _d("neuroradiology", "Neuroradiology", "specialized", "radiology", "neurology", "neuroanatomy"),
    _d("interventional_radiology", "Interventional radiology", "specialized", "radiology", "cardiology", "surgery"),
    _d("interventional_cardiology", "Interventional cardiology", "specialized", "cardiology", "radiology"),
    _d("clinical_genetics", "Clinical genetics", "specialized", "genetics", "pediatrics", "molecular_biology"),
    _d("nuclear_medicine", "Nuclear medicine", "specialized", "radiology", "oncology", "biochemistry"),
    # ---- language / support (not a field of study, but terms that belong to no single field)
    _d("latin", "Latin and Greek anatomical terms", "language", "anatomy", "neuroanatomy"),
    _d("eponyms", "Eponyms", "support"),
    _d("abbreviations", "Abbreviations", "support"),
    _d("general_medical", "General medical vocabulary", "support"),
]

DOMAINS: dict[str, Domain] = {d.id: d for d in _ALL}
SUPPORT_DOMAINS = frozenset(d.id for d in _ALL if d.group in ("language", "support"))
GENERAL = "general_medical"


def known(domain: str) -> bool:
    return domain in DOMAINS


def related(domain: str) -> tuple[str, ...]:
    d = DOMAINS.get(domain)
    return d.related if d else ()


def expand(domains: list[str] | tuple[str, ...], depth: int = 1) -> list[str]:
    """The given fields plus their direct relatives (candidate retrieval is widened to these, never beyond)."""
    out = [d for d in domains if d in DOMAINS]
    seen = set(out)
    frontier = list(out)
    for _ in range(depth):
        nxt = []
        for d in frontier:
            for r in related(d):
                if r not in seen:
                    seen.add(r)
                    out.append(r)
                    nxt.append(r)
        frontier = nxt
    return out
