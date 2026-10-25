"""UMLS semantic types (TUI) -> fields of study of the knowledge base.

A concept is imported when at least one of its semantic types maps to a field. Types that only hold everyday vocabulary ("Qualitative Concept",
"Spatial Concept", "Temporal Concept", "Idea or Concept", organisations, occupations, geography ...) are SKIPPED: they contain ordinary English words
("between", "signal", "process") which would only add false-positive candidates to a lecture engine. The table is data, not logic: edit it freely.
"""

from __future__ import annotations

#: TUI -> (name, domains)  - an empty domain tuple means "skip concepts that only have this type"
TYPES: dict[str, tuple[str, tuple[str, ...]]] = {
    "T001": ("Organism", ("microbiology",)), "T002": ("Plant", ()), "T004": ("Fungus", ("microbiology",)), "T005": ("Virus", ("microbiology",)),
    "T007": ("Bacterium", ("microbiology",)), "T008": ("Animal", ()), "T010": ("Vertebrate", ()), "T011": ("Amphibian", ()), "T012": ("Bird", ()),
    "T013": ("Fish", ()), "T014": ("Reptile", ()), "T015": ("Mammal", ()), "T016": ("Human", ()), "T194": ("Archaeon", ("microbiology",)),
    "T204": ("Eukaryote", ()),
    "T017": ("Anatomical Structure", ("anatomy",)), "T018": ("Embryonic Structure", ("embryology", "anatomy")),
    "T019": ("Congenital Abnormality", ("embryology", "genetics", "pediatrics")), "T020": ("Acquired Abnormality", ("pathology",)),
    "T021": ("Fully Formed Anatomical Structure", ("anatomy",)), "T022": ("Body System", ("anatomy", "physiology")),
    "T023": ("Body Part, Organ, or Organ Component", ("anatomy",)), "T024": ("Tissue", ("histology",)), "T025": ("Cell", ("histology",)),
    "T026": ("Cell Component", ("histology", "molecular_biology")), "T028": ("Gene or Genome", ("genetics", "molecular_biology")),
    "T029": ("Body Location or Region", ("anatomy",)), "T030": ("Body Space or Junction", ("anatomy",)), "T031": ("Body Substance", ("physiology",)),
    "T032": ("Organism Attribute", ()), "T033": ("Finding", ("internal_medicine",)), "T034": ("Laboratory or Test Result", ("laboratory_medicine",)),
    "T037": ("Injury or Poisoning", ("emergency_medicine",)), "T038": ("Biologic Function", ("physiology",)), "T039": ("Physiologic Function", ("physiology",)),
    "T040": ("Organism Function", ("physiology",)), "T041": ("Mental Process", ("neuroscience", "psychiatry")), "T042": ("Organ or Tissue Function", ("physiology",)),
    "T043": ("Cell Function", ("histology", "biochemistry")), "T044": ("Molecular Function", ("biochemistry", "molecular_biology")),
    "T045": ("Genetic Function", ("genetics", "molecular_biology")), "T046": ("Pathologic Function", ("pathology",)),
    "T047": ("Disease or Syndrome", ("internal_medicine", "pathology")), "T048": ("Mental or Behavioral Dysfunction", ("psychiatry",)),
    "T049": ("Cell or Molecular Dysfunction", ("pathology", "molecular_biology")), "T050": ("Experimental Model of Disease", ("pathology",)),
    "T051": ("Event", ()), "T052": ("Activity", ()), "T053": ("Behavior", ()), "T054": ("Social Behavior", ()), "T055": ("Individual Behavior", ()),
    "T056": ("Daily or Recreational Activity", ()), "T057": ("Occupational Activity", ()), "T058": ("Health Care Activity", ()),
    "T059": ("Laboratory Procedure", ("laboratory_medicine",)), "T060": ("Diagnostic Procedure", ("radiology", "internal_medicine")),
    "T061": ("Therapeutic or Preventive Procedure", ("surgery", "internal_medicine")), "T062": ("Research Activity", ()),
    "T063": ("Molecular Biology Research Technique", ("molecular_biology",)), "T064": ("Governmental or Regulatory Activity", ()),
    "T065": ("Educational Activity", ()), "T066": ("Machine Activity", ()), "T067": ("Phenomenon or Process", ()), "T068": ("Human-caused Phenomenon or Process", ()),
    "T069": ("Environmental Effect of Humans", ()), "T070": ("Natural Phenomenon or Process", ()), "T071": ("Entity", ()), "T072": ("Physical Object", ()),
    "T073": ("Manufactured Object", ()), "T074": ("Medical Device", ("surgery",)), "T075": ("Research Device", ()), "T077": ("Conceptual Entity", ()),
    "T078": ("Idea or Concept", ()), "T079": ("Temporal Concept", ()), "T080": ("Qualitative Concept", ()), "T081": ("Quantitative Concept", ()),
    "T082": ("Spatial Concept", ()), "T083": ("Geographic Area", ()), "T085": ("Molecular Sequence", ("molecular_biology",)),
    "T086": ("Nucleotide Sequence", ("molecular_biology",)), "T087": ("Amino Acid Sequence", ("molecular_biology", "biochemistry")),
    "T088": ("Carbohydrate Sequence", ("biochemistry",)), "T089": ("Regulation or Law", ()), "T090": ("Occupation or Discipline", ()),
    "T091": ("Biomedical Occupation or Discipline", ()), "T092": ("Organization", ()), "T093": ("Health Care Related Organization", ()),
    "T094": ("Professional Society", ()), "T095": ("Self-help or Relief Organization", ()), "T096": ("Group", ()), "T097": ("Professional or Occupational Group", ()),
    "T098": ("Population Group", ()), "T099": ("Family Group", ()), "T100": ("Age Group", ()), "T101": ("Patient or Disabled Group", ()), "T102": ("Group Attribute", ()),
    "T103": ("Chemical", ("biochemistry",)), "T104": ("Chemical Viewed Structurally", ("biochemistry",)), "T109": ("Organic Chemical", ("biochemistry", "pharmacology")),
    "T110": ("Steroid", ("biochemistry", "endocrinology")), "T111": ("Eicosanoid", ("biochemistry",)), "T114": ("Nucleic Acid, Nucleoside, or Nucleotide", ("molecular_biology", "biochemistry")),
    "T115": ("Organophosphorus Compound", ("biochemistry",)), "T116": ("Amino Acid, Peptide, or Protein", ("biochemistry", "molecular_biology")),
    "T118": ("Carbohydrate", ("biochemistry",)), "T119": ("Lipid", ("biochemistry",)), "T120": ("Chemical Viewed Functionally", ("biochemistry",)),
    "T121": ("Pharmacologic Substance", ("pharmacology",)), "T122": ("Biomedical or Dental Material", ()), "T123": ("Biologically Active Substance", ("physiology", "pharmacology")),
    "T124": ("Neuroreactive Substance or Biogenic Amine", ("neuroscience", "pharmacology")), "T125": ("Hormone", ("endocrinology", "physiology")),
    "T126": ("Enzyme", ("biochemistry",)), "T127": ("Vitamin", ("biochemistry",)), "T129": ("Immunologic Factor", ("immunology",)),
    "T130": ("Indicator, Reagent, or Diagnostic Aid", ("laboratory_medicine",)), "T131": ("Hazardous or Poisonous Substance", ("pharmacology",)),
    "T167": ("Substance", ()), "T168": ("Food", ()), "T169": ("Functional Concept", ()), "T170": ("Intellectual Product", ()), "T171": ("Language", ()),
    "T184": ("Sign or Symptom", ("internal_medicine",)), "T185": ("Classification", ()), "T190": ("Anatomical Abnormality", ("pathology", "embryology")),
    "T191": ("Neoplastic Process", ("oncology", "pathology")), "T192": ("Receptor", ("pharmacology", "molecular_biology")), "T195": ("Antibiotic", ("pharmacology", "infectious_diseases")),
    "T196": ("Element, Ion, or Isotope", ("biochemistry",)), "T197": ("Inorganic Chemical", ("biochemistry",)), "T200": ("Clinical Drug", ("pharmacology",)),
    "T201": ("Clinical Attribute", ("laboratory_medicine",)), "T203": ("Drug Delivery Device", ()),
}


def domains_for(tuis: list[str]) -> tuple[str, ...]:
    """Fields of study of a concept from its semantic types (first type first, no duplicates). Empty = skip the concept."""
    out: list[str] = []
    for tui in tuis:
        for domain in TYPES.get(tui, ("", ()))[1]:
            if domain not in out:
                out.append(domain)
    return tuple(out)


def type_name(tui: str) -> str:
    return TYPES.get(tui, (tui, ()))[0]
