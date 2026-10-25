"""Static knowledge for medical-entity classification and term scoring (plain data, easy to review)."""

from __future__ import annotations

# Single words that are too generic to help as hotwords (they are real English / basic science words).
GENERIC_TERMS: frozenset[str] = frozenset({
    "absorption", "distribution", "metabolism", "excretion", "clearance", "potency", "efficacy", "tolerance",
    "dependence", "withdrawal", "antagonism", "synergism", "transcription", "translation", "mutation", "deletion",
    "duplication", "inversion", "cloning", "exon", "intron", "codon", "fever", "shock", "coma", "stupor", "pelvis",
    "sternum", "inspection", "adaptation", "inflammation", "invasion", "congestion", "compliance", "contractility",
    "homeostasis", "peristalsis", "cilia", "lacuna", "platelet", "surfactant", "pigmentation", "calcification",
    "chemotherapy", "cytokine", "diuretic", "statin", "enzyme", "protein", "receptor", "hormone", "glucose", "blood",
    "pressure", "volume", "disease", "infection", "tumor", "cancer", "tissue", "muscle", "bone", "nerve", "artery",
    "vein", "heart", "lung", "kidney", "liver", "brain", "cell", "cells", "gene", "genes", "drug", "dose", "pain",
    "treatment", "diagnosis", "symptom", "syndrome", "complex", "pathway", "cycle", "synthesis", "reaction",
})

# Very common English words; used only to judge how "rare" a candidate term is.
COMMON_ENGLISH: frozenset[str] = frozenset("""
the of and to in is that it for as with was on be by this are from or an at which not have has had but they we you
all can will one would there their what so up out if about who get which go me when make can like time no just him
know take people into year your good some could them see other than then now look only come its over think also
back after use two how our work first well way even new want because any these give day most us system process
function level state order change point part number group case set area form field model side line type rate value
amount result effect factor level control pattern change increase decrease high low large small long short early
late main major minor common simple complex basic general specific normal total average different similar important
possible necessary available several many much more less each every both either neither another same other such
very quite rather almost often usually always never sometimes already still again further here where which while
chapter lecture slide slides textbook syllabus semester course student students question answer example case study
""".split())

# Head nouns / endings that make a phrase "anatomical" (raises medical specificity).
ANATOMICAL_HEADS: frozenset[str] = frozenset({
    "tract", "nucleus", "nuclei", "nerve", "ganglion", "ganglia", "artery", "vein", "plexus", "gyrus", "sulcus",
    "fasciculus", "lemniscus", "foramen", "fossa", "ligament", "tendon", "muscle", "cortex", "lobe", "duct", "canal",
    "membrane", "fiber", "fibers", "layer", "bone", "gland", "valve", "ventricle", "capsule", "pathway", "cycle",
    "receptor", "channel", "syndrome", "disease", "reductase", "synthase", "kinase", "dehydrogenase",
})

# Latin / Greek word endings and parts typical of medical terms.
LATINATE_SUFFIXES: tuple[str, ...] = (
    "itis", "osis", "emia", "aemia", "pathy", "ectomy", "otomy", "ostomy", "plasty", "oma", "genesis", "lysis",
    "penia", "uria", "cytosis", "plasia", "trophy", "stasis", "algia", "megaly", "graphy", "gram", "ase", "ine",
    "ide", "ate", "ium", "ule", "ula", "ular", "alis", "icus", "oid", "cyte", "blast", "clast", "philic", "tropic",
)
LATINATE_PREFIXES: tuple[str, ...] = (
    "hyper", "hypo", "neuro", "cardio", "gastro", "hepato", "nephro", "pulmo", "cortico", "spino", "cerebro", "myo",
    "osteo", "chondro", "angio", "hemo", "haemo", "immuno", "glyco", "gluco", "lipo", "phospho", "dehydro", "acetyl",
)

# ---------------------------------------------------------------- entity categories
CATEGORY_ORDER: tuple[str, ...] = (
    "anatomy", "nerve", "tract", "nucleus", "vessel", "muscle", "bone", "enzyme", "receptor", "hormone",
    "neurotransmitter", "drug", "disease", "pathology", "gene", "protein", "cell type", "biochemical pathway",
)

NERVE_WORDS = frozenset({"nerve", "nerves", "plexus", "ramus", "ganglion", "ganglia"})
TRACT_WORDS = frozenset({"tract", "fasciculus", "fasciculi", "lemniscus", "funiculus", "commissure", "peduncle",
                         "decussation", "fornix"})
NUCLEUS_WORDS = frozenset({"nucleus", "nuclei", "substantia", "locus", "pallidus", "putamen", "striatum", "caudate",
                           "amygdala", "thalamus", "hypothalamus", "hippocampus", "pulvinar", "colliculus"})
VESSEL_WORDS = frozenset({"artery", "arteries", "vein", "veins", "aorta", "vena", "cava", "capillary", "venule",
                          "arteriole", "sinus", "duct", "vessel", "vessels", "lymphatic"})
MUSCLE_WORDS = frozenset({
    "muscle", "muscles", "biceps", "triceps", "deltoid", "trapezius", "soleus", "iliopsoas", "gluteus", "gastrocnemius",
    "sternocleidomastoid", "quadriceps", "brachii", "femoris", "abdominis", "oblique", "diaphragm", "latissimus",
    "pectoralis", "rhomboid", "brachioradialis", "supraspinatus", "infraspinatus", "subscapularis", "sartorius",
})
BONE_WORDS = frozenset({
    "bone", "bones", "vertebra", "vertebrae", "femur", "tibia", "fibula", "humerus", "radius", "ulna", "scapula",
    "clavicle", "sternum", "pelvis", "skull", "cranium", "mandible", "maxilla", "patella", "calcaneus", "talus",
    "scaphoid", "lunate", "carpal", "metacarpal", "phalanges", "ilium", "ischium", "pubis", "sacrum", "coccyx", "rib", "ribs",
})
ENZYME_EXCLUDE = frozenset({"disease", "phase", "base", "release", "increase", "decrease", "case", "purchase", "database",
                            "ase", "please", "response", "cease", "lease", "grease", "crease", "decease", "tease"})
ENZYME_WORDS = frozenset({"enzyme", "kinase", "phosphatase", "polymerase", "helicase", "topoisomerase", "ligase", "lyase",
                          "synthase", "synthetase", "dehydrogenase", "reductase", "oxidase", "transferase", "hydrolase",
                          "isomerase", "carboxylase", "decarboxylase", "mutase", "aldolase", "enolase", "esterase",
                          "peptidase", "protease", "lipase", "amylase", "nuclease", "integrase", "transcriptase",
                          "cyclase", "hydroxylase", "transaminase", "aminotransferase", "cholinesterase"})
RECEPTOR_WORDS = frozenset({"receptor", "receptors", "adrenoceptor", "autoreceptor"})
HORMONES = frozenset({
    "insulin", "glucagon", "cortisol", "aldosterone", "thyroxine", "triiodothyronine", "testosterone", "estrogen",
    "estradiol", "progesterone", "oxytocin", "vasopressin", "adh", "prolactin", "acth", "tsh", "gnrh", "somatostatin",
    "calcitonin", "parathormone", "pth", "erythropoietin", "melatonin", "leptin", "ghrelin", "gastrin", "secretin",
    "cholecystokinin", "renin", "angiotensin", "epinephrine", "norepinephrine", "growth hormone", "hormone", "hcg",
    "fsh", "lh", "dhea", "thyrotropin", "corticotropin", "glucocorticoid", "mineralocorticoid",
})
NEUROTRANSMITTERS = frozenset({
    "dopamine", "serotonin", "gaba", "glutamate", "acetylcholine", "norepinephrine", "epinephrine", "histamine",
    "glycine", "aspartate", "substance p", "enkephalin", "endorphin", "neuropeptide", "adenosine", "anandamide",
    "neurotransmitter", "noradrenaline", "adrenaline",
})
DRUG_SUFFIXES: tuple[str, ...] = (
    "olol", "pril", "sartan", "statin", "mycin", "micin", "cillin", "azole", "vir", "mab", "nib", "prazole",
    "dipine", "thiazide", "afil", "azepam", "azolam", "barbital", "caine", "oxacin", "cycline", "floxacin", "tidine",
    "setron", "triptan", "gliptin", "glitazone", "gliflozin", "parin", "xaban", "gatran", "pamil", "semide",
    "terol", "lukast", "tropium", "sone", "olone", "zepam", "oxetine", "pramine", "triptyline", "peridol",
    "profen", "coxib", "mide", "fibrate", "formin", "glinide", "platin", "rubicin", "taxel", "tinib", "ciclib",
)
# Words that look like drugs (a "-terol" ending, a place in pharmacology.txt) but are not drug names. Seen on a real lecture:
# "absorption x12" and "cholesterol x4" were reported as drugs.
NOT_DRUGS = frozenset({
    "cholesterol", "ergosterol", "sitosterol", "absorption", "distribution", "metabolism", "excretion", "elimination",
    "withdrawal", "half life", "half-life", "clearance", "tolerance", "dependence", "potency", "efficacy", "bioavailability",
})
DISEASE_WORDS = frozenset({"disease", "syndrome", "disorder", "carcinoma", "sarcoma", "lymphoma", "leukemia", "cancer",
                           "infection", "failure", "stenosis", "insufficiency", "dystrophy", "myopathy", "neuropathy",
                           "diabetes", "hypertension", "hypotension", "asthma", "anemia", "arthritis", "sclerosis"})
DISEASE_SUFFIXES: tuple[str, ...] = ("itis", "oma", "pathy", "emia", "aemia", "penia", "uria", "algia", "plegia", "paresis")
PATHOLOGY_WORDS = frozenset({
    "necrosis", "apoptosis", "inflammation", "ischemia", "infarction", "hyperplasia", "hypertrophy", "atrophy",
    "metaplasia", "dysplasia", "neoplasia", "anaplasia", "fibrosis", "thrombosis", "embolism", "edema", "hemorrhage",
    "granuloma", "metastasis", "amyloidosis", "steatosis", "sclerosis", "calcification", "ulceration", "abscess",
    "cirrhosis", "stenosis", "aneurysm", "atherosclerosis", "tumor", "lesion",
})
PROTEIN_WORDS = frozenset({"protein", "proteins", "hemoglobin", "myoglobin", "collagen", "albumin", "actin", "myosin",
                           "troponin", "tropomyosin", "keratin", "elastin", "fibrin", "fibrinogen", "ferritin",
                           "transferrin", "immunoglobulin", "antibody", "cytochrome", "histone", "tubulin", "laminin",
                           "fibronectin", "integrin", "cadherin", "factor", "peptide", "ubiquitin", "chaperone"})
CELL_TYPE_SUFFIXES: tuple[str, ...] = ("cyte", "blast", "clast", "phage", "glia", "poietic")
CELL_TYPE_WORDS = frozenset({"neuron", "neurons", "cell", "cells", "astrocyte", "oligodendrocyte", "microglia", "macrophage",
                             "lymphocyte", "neutrophil", "eosinophil", "basophil", "monocyte", "platelet", "erythrocyte",
                             "hepatocyte", "podocyte", "keratinocyte", "melanocyte", "fibroblast", "osteoblast",
                             "osteoclast", "osteocyte", "chondrocyte", "adipocyte", "enterocyte", "myocyte", "pneumocyte"})
PATHWAY_WORDS = frozenset({"cycle", "pathway", "shunt", "oxidation", "synthesis", "cascade", "metabolism", "glycolysis",
                           "gluconeogenesis", "glycogenolysis", "glycogenesis", "lipolysis", "ketogenesis", "lipogenesis",
                           "phosphorylation", "transamination", "deamination", "proteolysis", "glycosylation"})
PATHWAY_SUFFIXES: tuple[str, ...] = ("lysis", "genesis", "oxidation", "synthesis")
GENE_WORDS = frozenset({"gene", "genes", "oncogene", "allele", "locus", "mutation"})
