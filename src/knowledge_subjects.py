"""Static knowledge for the subject classifier: areas, indicative words, vocabulary files, file-name aliases.

Everything here is plain data so it can be reviewed and extended without touching the logic.
Keywords are lower-case; multi-word phrases are matched as phrases and count more than single words.
"""

from __future__ import annotations

# The 19 areas the classifier knows. Keys are used in reports and config.
AREAS: tuple[str, ...] = (
    "anatomy", "neuroanatomy", "neuroscience", "physiology", "histology", "pathology", "pharmacology",
    "biochemistry", "molecular_biology", "microbiology", "immunology", "cardiology", "respiratory",
    "nephrology", "gastroenterology", "endocrinology", "hematology", "reproductive", "embryology",
)

# Which medical_terms/*.txt files supply hotword candidates for an area (at most three, most specific first).
AREA_VOCAB_FILES: dict[str, tuple[str, ...]] = {
    "anatomy": ("anatomy",),
    "neuroanatomy": ("neuroscience", "anatomy"),
    "neuroscience": ("neuroscience",),
    "physiology": ("physiology",),
    "histology": ("histology",),
    "pathology": ("pathology",),
    "pharmacology": ("pharmacology",),
    "biochemistry": ("biochemistry",),
    "molecular_biology": ("molecular_biology",),
    "microbiology": ("pathology", "pharmacology"),
    "immunology": ("pathology", "molecular_biology"),
    "cardiology": ("physiology", "pathology", "pharmacology"),
    "respiratory": ("physiology", "pathology"),
    "nephrology": ("physiology", "pathology"),
    "gastroenterology": ("physiology", "pathology", "histology"),
    "endocrinology": ("physiology", "biochemistry", "pharmacology"),
    "hematology": ("histology", "pathology"),
    "reproductive": ("anatomy", "physiology", "histology"),
    "embryology": ("anatomy", "histology"),
}

# Words that point to an area. The area's own name (and close synonyms) is listed first on purpose:
# a lecturer who says "welcome to biochemistry" has told us the subject.
AREA_KEYWORDS: dict[str, tuple[str, ...]] = {
    "anatomy": (
        "anatomy", "anatomical", "bone", "muscle", "artery", "vein", "nerve", "ligament", "tendon", "joint",
        "fascia", "foramen", "fossa", "plexus", "vertebra", "thorax", "abdomen", "pelvis", "anterior", "posterior",
        "superior", "inferior", "medial", "lateral", "proximal", "distal", "dissection", "origin and insertion",
    ),
    "neuroanatomy": (
        "neuroanatomy", "tract", "nucleus", "nuclei", "gyrus", "sulcus", "lobe", "ganglion", "ganglia", "brainstem",
        "medulla", "pons", "midbrain", "cerebellum", "thalamus", "hypothalamus", "spinal cord", "cranial nerve",
        "white matter", "gray matter", "decussation", "pyramidal", "lemniscus", "fasciculus", "peduncle",
        "internal capsule", "corticospinal", "spinothalamic",
    ),
    "neuroscience": (
        "neuroscience", "neuron", "neurons", "synapse", "synaptic", "axon", "dendrite", "neurotransmitter",
        "action potential", "glia", "myelin", "plasticity", "dopamine", "serotonin", "glutamate", "gaba",
        "acetylcholine", "hippocampus", "amygdala", "basal ganglia", "nociception", "reflex", "consciousness",
        "long term potentiation", "excitatory", "inhibitory",
    ),
    "physiology": (
        "physiology", "physiological", "homeostasis", "feedback", "membrane potential", "cardiac output",
        "ventilation", "filtration", "reabsorption", "secretion", "osmolarity", "regulation", "compliance",
        "resistance", "blood pressure", "gradient", "transport", "contraction", "excitation",
    ),
    "histology": (
        "histology", "histological", "tissue", "epithelium", "epithelial", "connective tissue", "stain", "staining",
        "hematoxylin", "eosin", "microscope", "slide", "section", "basement membrane", "gland", "villi",
        "collagen fibers", "cytoplasm", "goblet cell", "lamina propria",
    ),
    "pathology": (
        "pathology", "pathological", "pathogenesis", "disease", "lesion", "necrosis", "inflammation", "neoplasm",
        "tumor", "carcinoma", "metastasis", "dysplasia", "hyperplasia", "atrophy", "hypertrophy", "ischemia",
        "infarct", "thrombus", "embolus", "biopsy", "etiology", "benign", "malignant", "granuloma",
    ),
    "pharmacology": (
        "pharmacology", "pharmacological", "drug", "drugs", "dose", "agonist", "antagonist", "pharmacokinetics",
        "pharmacodynamics", "half life", "bioavailability", "clearance", "cytochrome", "inhibitor", "side effect",
        "adverse effect", "toxicity", "therapeutic", "prodrug", "antibiotic", "analgesic", "anesthetic",
        "anticoagulant", "receptor blocker",
    ),
    "biochemistry": (
        "biochemistry", "biochemical", "enzyme", "enzymes", "substrate", "metabolism", "metabolic", "glycolysis",
        "krebs cycle", "citric acid cycle", "atp", "nadh", "fadh", "oxidation", "reduction", "phosphorylation",
        "amino acid", "amino acids", "protein", "lipid", "fatty acid", "cholesterol", "carbohydrate", "glycogen",
        "gluconeogenesis", "coenzyme", "vitamin", "kinase", "dehydrogenase", "mitochondria", "michaelis",
        "allosteric", "urea cycle", "ketone", "textbook harper", "harper", "metabolite", "pathway",
    ),
    "molecular_biology": (
        "molecular biology", "dna", "rna", "gene", "genes", "genome", "transcription", "translation", "replication",
        "mutation", "polymerase", "ribosome", "codon", "promoter", "plasmid", "pcr", "sequencing", "crispr",
        "chromosome", "allele", "epigenetic", "histone", "splicing", "protein synthesis", "gene expression",
        "genotype", "phenotype", "nucleotide",
    ),
    "microbiology": (
        "microbiology", "bacteria", "bacterium", "bacterial", "virus", "viral", "fungus", "fungal", "parasite",
        "protozoa", "gram positive", "gram negative", "staphylococcus", "streptococcus", "escherichia coli",
        "pseudomonas", "mycobacterium", "clostridium", "salmonella", "helicobacter", "spirochete", "chlamydia",
        "plasmodium", "candida", "aspergillus", "capsid", "virion", "bacteriophage", "endotoxin", "exotoxin",
        "pathogen", "virulence", "antibiotic resistance", "agar", "gram stain", "microbiota", "infection",
    ),
    "immunology": (
        "immunology", "immune", "immunity", "antibody", "antibodies", "antigen", "immunoglobulin", "lymphocyte",
        "t cell", "b cell", "macrophage", "cytokine", "interleukin", "complement", "mhc", "hla",
        "major histocompatibility", "innate immunity", "adaptive immunity", "hypersensitivity", "autoimmune",
        "tolerance", "vaccine", "vaccination", "plasma cell", "memory cell", "natural killer", "dendritic cell",
        "antigen presenting", "opsonization", "interferon", "immunodeficiency", "allergy",
    ),
    "cardiology": (
        "cardiology", "cardiac", "heart", "myocardium", "ventricle", "ventricular", "atrium", "atrial", "mitral",
        "tricuspid", "coronary", "ecg", "ekg", "arrhythmia", "fibrillation", "tachycardia", "bradycardia", "angina",
        "myocardial infarction", "heart failure", "ejection fraction", "stroke volume", "preload", "afterload",
        "systole", "diastole", "pericardium", "endocarditis", "cardiomyopathy", "murmur", "atherosclerosis", "qrs",
    ),
    "respiratory": (
        "respiratory", "pulmonary", "lung", "lungs", "alveoli", "alveolus", "bronchus", "bronchi", "bronchiole",
        "trachea", "ventilation perfusion", "oxygen", "carbon dioxide", "spirometry", "tidal volume", "vital capacity",
        "asthma", "copd", "emphysema", "pneumonia", "pleura", "pleural", "surfactant", "hypoxia", "hypercapnia",
        "respiratory rate", "diaphragm", "airway", "dyspnea",
    ),
    "nephrology": (
        "nephrology", "renal", "kidney", "kidneys", "nephron", "glomerulus", "glomerular", "tubule", "tubular",
        "loop of henle", "collecting duct", "filtration rate", "gfr", "creatinine", "nephritis", "nephrotic", "dialysis",
        "proteinuria", "hematuria", "renin", "angiotensin", "diuretic", "urine", "bladder", "ureter", "osmolality",
        "acid base",
    ),
    "gastroenterology": (
        "gastroenterology", "gastrointestinal", "stomach", "gastric", "intestine", "intestinal", "colon", "bowel",
        "duodenum", "jejunum", "ileum", "esophagus", "liver", "hepatic", "hepatocyte", "bile", "biliary",
        "gallbladder", "pancreas", "pancreatic", "digestion", "gastrin", "secretin", "cholecystokinin", "peristalsis",
        "crohn", "colitis", "ulcer", "cirrhosis", "hepatitis", "diarrhea", "malabsorption", "celiac",
    ),
    "endocrinology": (
        "endocrinology", "endocrine", "hormone", "hormonal", "pituitary", "thyroid", "thyroxine", "parathyroid",
        "adrenal", "cortisol", "insulin", "glucagon", "diabetes", "growth hormone", "prolactin", "acth", "tsh",
        "gonadotropin", "estrogen", "testosterone", "negative feedback", "hyperthyroidism", "hypothyroidism",
        "cushing", "addison", "goiter",
    ),
    "hematology": (
        "hematology", "haematology", "blood", "erythrocyte", "red blood cell", "hemoglobin", "hematocrit", "anemia",
        "platelet", "thrombocyte", "leukocyte", "white blood cell", "neutrophil", "coagulation", "clotting", "fibrin",
        "fibrinogen", "thrombin", "thrombosis", "hemophilia", "leukemia", "lymphoma", "bone marrow", "hematopoiesis",
        "erythropoietin", "ferritin", "transfusion", "sickle cell", "thalassemia",
    ),
    "reproductive": (
        "reproductive", "reproduction", "ovary", "ovarian", "uterus", "uterine", "testis", "testes", "sperm",
        "spermatogenesis", "oogenesis", "ovulation", "menstrual", "menstruation", "estrogen", "progesterone",
        "fertilization", "prostate", "fallopian", "endometrium", "placenta", "pregnancy", "lactation", "gonad",
        "puberty", "hcg",
    ),
    "embryology": (
        "embryology", "embryo", "embryonic", "fetus", "fetal", "gastrulation", "ectoderm", "mesoderm", "endoderm",
        "neural tube", "neural crest", "somite", "blastocyst", "zygote", "implantation", "morula", "germ layer",
        "notochord", "organogenesis", "teratogen", "congenital", "trophoblast", "primitive streak", "pharyngeal arch",
    ),
}

# Tokens in a media file name that hint at the subject (ported idea from the Cline reference project).
FILENAME_ALIASES: dict[str, str] = {
    "neuroscience": "neuroscience", "neuro": "neuroscience", "brain": "neuroscience",
    "neuroanatomy": "neuroanatomy", "neuroanat": "neuroanatomy",
    "anatomy": "anatomy", "anat": "anatomy",
    "physiology": "physiology", "physio": "physiology",
    "histology": "histology", "histo": "histology",
    "pathology": "pathology", "patho": "pathology",
    "pharmacology": "pharmacology", "pharma": "pharmacology", "pharm": "pharmacology",
    "biochemistry": "biochemistry", "biochem": "biochemistry",
    "molecularbiology": "molecular_biology", "molbio": "molecular_biology", "genetics": "molecular_biology",
    "microbiology": "microbiology", "micro": "microbiology", "bacteriology": "microbiology", "virology": "microbiology",
    "immunology": "immunology", "immuno": "immunology",
    "cardiology": "cardiology", "cardio": "cardiology", "cardiac": "cardiology",
    "respiratory": "respiratory", "pulmonology": "respiratory", "pulmo": "respiratory",
    "nephrology": "nephrology", "nephro": "nephrology", "renal": "nephrology",
    "gastroenterology": "gastroenterology", "gastro": "gastroenterology",
    "endocrinology": "endocrinology", "endocrine": "endocrinology", "endo": "endocrinology",
    "hematology": "hematology", "haematology": "hematology", "hema": "hematology",
    "reproductive": "reproductive", "repro": "reproductive",
    "embryology": "embryology", "embryo": "embryology",
}

# Turkish words in a file name (ders adı) -> area.
FILENAME_ALIASES_TR: dict[str, str] = {
    "anatomi": "anatomy", "fizyoloji": "physiology", "histoloji": "histology", "patoloji": "pathology",
    "farmakoloji": "pharmacology", "biyokimya": "biochemistry", "mikrobiyoloji": "microbiology",
    "immunoloji": "immunology", "kardiyoloji": "cardiology", "nefroloji": "nephrology",
    "embriyoloji": "embryology", "endokrinoloji": "endocrinology", "hematoloji": "hematology",
}
