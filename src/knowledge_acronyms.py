"""Medical acronyms, ambiguous abbreviations and gene/protein/drug name variants (plain data)."""

from __future__ import annotations

# acronym -> expansion (the unambiguous ones; ambiguous ones live in AMBIGUOUS below)
ACRONYMS: dict[str, str] = {
    "CNS": "central nervous system", "PNS": "peripheral nervous system", "ANS": "autonomic nervous system",
    "CSF": "cerebrospinal fluid", "BBB": "blood-brain barrier", "GABA": "gamma-aminobutyric acid",
    "NMDA": "N-methyl-D-aspartate", "AMPA": "alpha-amino-3-hydroxy-5-methyl-4-isoxazolepropionic acid",
    "ATP": "adenosine triphosphate", "ADP": "adenosine diphosphate", "AMP": "adenosine monophosphate",
    "cAMP": "cyclic adenosine monophosphate", "cGMP": "cyclic guanosine monophosphate",
    "GTP": "guanosine triphosphate", "GDP": "guanosine diphosphate", "NAD": "nicotinamide adenine dinucleotide",
    "NADH": "reduced nicotinamide adenine dinucleotide", "NADPH": "nicotinamide adenine dinucleotide phosphate",
    "FAD": "flavin adenine dinucleotide", "FADH2": "reduced flavin adenine dinucleotide", "CoA": "coenzyme A",
    "ECG": "electrocardiogram", "EKG": "electrocardiogram", "EEG": "electroencephalogram", "EMG": "electromyography",
    "MRI": "magnetic resonance imaging", "PET": "positron emission tomography",
    "DNA": "deoxyribonucleic acid", "RNA": "ribonucleic acid", "mRNA": "messenger RNA", "tRNA": "transfer RNA",
    "rRNA": "ribosomal RNA", "PCR": "polymerase chain reaction", "ELISA": "enzyme-linked immunosorbent assay",
    "HIV": "human immunodeficiency virus", "AIDS": "acquired immunodeficiency syndrome",
    "MHC": "major histocompatibility complex", "HLA": "human leukocyte antigen", "TNF": "tumor necrosis factor",
    "ACE": "angiotensin-converting enzyme", "ACh": "acetylcholine", "AChE": "acetylcholinesterase",
    "LTP": "long-term potentiation", "LTD": "long-term depression", "GFR": "glomerular filtration rate",
    "RAAS": "renin-angiotensin-aldosterone system", "ADH": "antidiuretic hormone",
    "ACTH": "adrenocorticotropic hormone", "TSH": "thyroid-stimulating hormone", "FSH": "follicle-stimulating hormone",
    "LH": "luteinizing hormone", "hCG": "human chorionic gonadotropin", "BMI": "body mass index",
    "ABG": "arterial blood gas", "CBC": "complete blood count", "WBC": "white blood cell",
    "RBC": "red blood cell", "INR": "international normalized ratio", "CRP": "C-reactive protein",
    "ESR": "erythrocyte sedimentation rate", "LDL": "low-density lipoprotein", "HDL": "high-density lipoprotein",
    "VLDL": "very low-density lipoprotein", "PFK": "phosphofructokinase", "LDH": "lactate dehydrogenase",
    "G6PD": "glucose-6-phosphate dehydrogenase", "COPD": "chronic obstructive pulmonary disease",
    "CHF": "congestive heart failure", "DVT": "deep vein thrombosis", "HTN": "hypertension",
    "CKD": "chronic kidney disease", "AKI": "acute kidney injury", "UTI": "urinary tract infection",
    "GERD": "gastroesophageal reflux disease", "IBD": "inflammatory bowel disease", "IBS": "irritable bowel syndrome",
    "COX": "cyclooxygenase", "NSAID": "nonsteroidal anti-inflammatory drug", "SSRI": "selective serotonin reuptake inhibitor",
    "MAOI": "monoamine oxidase inhibitor", "PPI": "proton pump inhibitor", "ARB": "angiotensin receptor blocker",
    "CVA": "cerebrovascular accident", "TIA": "transient ischemic attack", "ICP": "intracranial pressure",
    "ICU": "intensive care unit", "GI": "gastrointestinal", "ATPase": "adenosine triphosphatase",
    "PKA": "protein kinase A", "PKC": "protein kinase C", "IP3": "inositol trisphosphate", "DAG": "diacylglycerol",
    "ER": "endoplasmic reticulum", "TCA": "tricarboxylic acid", "ETC": "electron transport chain",
    "HMG-CoA": "3-hydroxy-3-methylglutaryl coenzyme A", "SAM": "S-adenosylmethionine",
    "PRPP": "phosphoribosyl pyrophosphate", "UDP": "uridine diphosphate", "VEGF": "vascular endothelial growth factor",
    "EGFR": "epidermal growth factor receptor", "mTOR": "mammalian target of rapamycin",
}
# Acronyms the glossary lists once but that are really ambiguous: handled by AMBIGUOUS instead.
ACRONYMS_THAT_ARE_AMBIGUOUS = frozenset({"ER", "TCA"})

# abbreviation -> {meaning: context words that favour it}. Scored by words around each occurrence.
AMBIGUOUS: dict[str, dict[str, tuple[str, ...]]] = {
    "MS": {
        "multiple sclerosis": ("myelin", "demyelination", "neurologic", "optic neuritis", "oligodendrocyte", "plaques",
                               "brain", "spinal", "relapsing", "remitting", "cns", "lesions", "mri", "interferon"),
        "mitral stenosis": ("mitral", "valve", "atrial", "murmur", "rheumatic", "heart", "diastolic", "left atrium", "cardiac"),
        "mass spectrometry": ("spectrometry", "mass", "peptide", "ionization", "analyte", "chromatography", "proteomics"),
        "morphine sulfate": ("morphine", "opioid", "analgesic", "pain", "dose", "sulfate"),
    },
    "PE": {
        "pulmonary embolism": ("pulmonary", "embolism", "clot", "dvt", "thrombus", "hypoxia", "lung", "anticoagulant", "d-dimer"),
        "physical examination": ("examination", "inspection", "palpation", "auscultation", "patient", "findings", "vitals"),
        "pleural effusion": ("pleural", "effusion", "fluid", "chest", "thoracentesis", "lung"),
    },
    "RA": {
        "rheumatoid arthritis": ("rheumatoid", "joint", "arthritis", "synovium", "autoimmune", "methotrexate", "hand", "rf"),
        "right atrium": ("atrium", "atrial", "heart", "vena cava", "tricuspid", "right", "cardiac", "pressure"),
        "room air": ("oxygen", "saturation", "spo2", "air", "breathing", "hypoxia"),
    },
    "ER": {
        "endoplasmic reticulum": ("endoplasmic", "reticulum", "ribosome", "golgi", "protein synthesis", "membrane", "organelle",
                                  "rough", "smooth", "calcium", "folding", "cell"),
        "estrogen receptor": ("estrogen", "receptor", "breast", "tamoxifen", "hormone", "cancer", "positive"),
        "emergency room": ("emergency", "patient", "triage", "admitted", "hospital", "ambulance"),
    },
    "CA": {
        "carcinoma / cancer": ("tumor", "malignant", "cancer", "carcinoma", "metastasis", "oncology", "chemotherapy"),
        "calcium": ("calcium", "serum", "ion", "hypercalcemia", "bone", "parathyroid", "channel", "mg"),
        "carbonic anhydrase": ("carbonic", "anhydrase", "bicarbonate", "enzyme", "zinc", "acetazolamide", "co2"),
    },
    "PD": {
        "Parkinson disease": ("parkinson", "dopamine", "substantia nigra", "tremor", "levodopa", "basal ganglia", "rigidity"),
        "pharmacodynamics": ("pharmacodynamics", "receptor", "drug", "effect", "dose response", "pharmacokinetics", "pk"),
        "peritoneal dialysis": ("peritoneal", "dialysis", "renal", "kidney", "catheter", "dialysate"),
    },
    "AD": {
        "Alzheimer disease": ("alzheimer", "dementia", "amyloid", "tau", "memory", "cholinesterase", "plaques", "neurodegeneration"),
        "autosomal dominant": ("autosomal", "dominant", "inheritance", "pedigree", "gene", "allele", "penetrance"),
        "atopic dermatitis": ("atopic", "dermatitis", "eczema", "skin", "ige", "itch"),
    },
    "CP": {
        "cerebral palsy": ("cerebral", "palsy", "spastic", "motor", "child", "birth", "hypoxia"),
        "chest pain": ("chest", "pain", "angina", "ecg", "troponin", "cardiac", "infarction"),
        "creatine phosphate": ("creatine", "phosphate", "muscle", "atp", "phosphocreatine", "energy"),
    },
    "TCA": {
        "tricarboxylic acid cycle": ("cycle", "krebs", "citrate", "mitochondria", "acetyl", "nadh", "oxaloacetate", "citric", "metabolism"),
        "tricyclic antidepressant": ("antidepressant", "amitriptyline", "depression", "overdose", "serotonin", "norepinephrine", "tricyclic"),
    },
    "DM": {
        "diabetes mellitus": ("diabetes", "insulin", "glucose", "hyperglycemia", "hba1c", "glycemic", "type 2", "pancreas"),
        "dermatomyositis": ("dermatomyositis", "muscle", "rash", "heliotrope", "inflammatory myopathy"),
        "myotonic dystrophy": ("myotonic", "dystrophy", "trinucleotide", "repeat", "muscle"),
    },
    "MI": {
        "myocardial infarction": ("myocardial", "infarction", "troponin", "coronary", "ecg", "st elevation", "heart", "occlusion", "necrosis"),
        "mitral insufficiency": ("mitral", "regurgitation", "valve", "murmur", "insufficiency", "left atrium"),
    },
    "AS": {
        "aortic stenosis": ("aortic", "stenosis", "valve", "murmur", "calcific", "syncope", "left ventricle"),
        "ankylosing spondylitis": ("ankylosing", "spondylitis", "spine", "hla-b27", "sacroiliac", "back pain"),
    },
    "HR": {
        "heart rate": ("heart", "pulse", "beats", "bpm", "tachycardia", "bradycardia", "cardiac output", "vital"),
        "hazard ratio": ("hazard", "survival", "confidence interval", "trial", "cox", "study", "risk"),
    },
    "PT": {
        "prothrombin time": ("prothrombin", "coagulation", "inr", "warfarin", "clotting", "factor vii", "extrinsic"),
        "physical therapy": ("physical", "therapy", "rehabilitation", "exercise", "patient", "mobility"),
    },
    "ALS": {
        "amyotrophic lateral sclerosis": ("amyotrophic", "motor neuron", "sclerosis", "riluzole", "weakness", "bulbar"),
        "advanced life support": ("resuscitation", "cardiac arrest", "defibrillation", "acls", "airway", "advanced"),
    },
}

# Canonical name -> spelling/format variants seen in lectures (drugs, genes, proteins, receptors). Used for the
# normalisation REPORT only; nothing is rewritten.
NAME_VARIANTS: dict[str, tuple[str, ...]] = {
    "TNF-alpha": ("tnf alpha", "tnf-a", "tnfα", "tnf α", "tumor necrosis factor alpha", "tumor necrosis factor-alpha", "tnf"),
    "IL-6": ("il6", "il 6", "interleukin 6", "interleukin-6", "interleukin six"),
    "IL-1 beta": ("il-1b", "il1 beta", "il 1 beta", "il-1β", "interleukin 1 beta", "interleukin-1 beta"),
    "IFN-gamma": ("ifn gamma", "ifn-g", "ifnγ", "interferon gamma", "interferon-gamma"),
    "GABA-A": ("gaba a", "gabaa", "gaba-a receptor", "gaba a receptor"),
    "GABA-B": ("gaba b", "gabab", "gaba-b receptor", "gaba b receptor"),
    "NMDA receptor": ("nmda", "n-methyl-d-aspartate receptor", "nmdar", "nmda-receptor"),
    "AMPA receptor": ("ampa", "ampar", "ampa-receptor"),
    "acetylcholinesterase": ("acetyl cholinesterase", "acetylcholine esterase", "acetylcholin esterase", "ache"),
    "butyrylcholinesterase": ("butyryl cholinesterase", "butyrylcholine esterase"),
    "phosphofructokinase": ("phospho fructo kinase", "phosphofructo kinase", "phosphofructokinase-1", "pfk-1", "pfk1"),
    "pyruvate dehydrogenase": ("pyruvate de hydrogenase", "pdh complex", "pdh"),
    "HMG-CoA reductase": ("hmg coa reductase", "hmg-coa-reductase", "hmgcr", "hmg co a reductase"),
    "glucose-6-phosphate dehydrogenase": ("glucose 6 phosphate dehydrogenase", "g6pd", "g-6-pd"),
    "p53": ("p 53", "tp53", "protein 53", "p-53"),
    "BRCA1": ("brca 1", "brca-1", "brca one"),
    "BRCA2": ("brca 2", "brca-2", "brca two"),
    "KRAS": ("k-ras", "k ras", "kras gene"),
    "EGFR": ("egfr", "epidermal growth factor receptor", "her1"),
    "HER2": ("her-2", "her 2", "erbb2", "her2/neu"),
    "VEGF": ("vegf", "vascular endothelial growth factor"),
    "NF-kB": ("nf kb", "nfkb", "nf-κb", "nuclear factor kappa b", "nuclear factor kappa-b"),
    "mTOR": ("m tor", "mammalian target of rapamycin"),
    "CFTR": ("c f t r", "cystic fibrosis transmembrane conductance regulator"),
    "HbA1c": ("hba1c", "hemoglobin a1c", "glycated hemoglobin", "glycosylated hemoglobin", "a1c"),
    "LDL receptor": ("ldl-receptor", "ldl r", "ldlr"),
    "acetaminophen": ("paracetamol",),
    "epinephrine": ("adrenaline",),
    "norepinephrine": ("noradrenaline",),
}
