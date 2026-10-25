"""One or two real-sounding cases per medical field: the vocabulary of each domain is used by the same context-aware correction."""

import pytest

from tests.medical_support import applied, correct

CASES = [
    # (subject, sentences, probabilities of doubted words, expected (heard, corrected))
    ("physiology", ["The stroke volume depends on the preload of the ventricle,", "which is described by the Frank Sterling law of the heart."],
     {1: {"sterling": 0.4}}, ("Sterling", "Starling")),
    ("biochemistry", ["The citric acid cycle starts when acetyl-CoA condenses with oxalacetate in the mitochondria,", "and NADH feeds the electron transport chain."],
     {}, ("oxalacetate", "oxaloacetate")),
    ("biochemistry", ["The urea cycle removes ammonia from the blood in the liver,", "and the first enzyme is the kerbamoyl phosphate synthetase in the mitochondria."],
     {}, ("kerbamoyl", "carbamoyl")),
    ("molecular_biology", ["During transcription the RNA polymerase reads the template strand,", "then the pre-mRNA is processed by the spliceosom and exported."],
     {1: {"spliceosom": 0.5}}, ("spliceosom", "spliceosome")),
    ("pathology", ["The plaque in the intima of the artery contains lipids and foam cells,", "this process is called athero sclerosis and narrows the lumen."],
     {1: {"athero": 0.45}}, ("athero sclerosis", "atherosclerosis")),
    ("microbiology", ["The gram-positive cocci grow in clusters and are catalase positive,", "this is Staphylococcus areus, a cause of abscesses and MRSA infection."],
     {1: {"areus": 0.4}}, ("areus", "aureus")),
    ("immunology", ["B cells differentiate into plasma cells that secrete antibodies,", "and the antibody class IgM is the first immunoglobulin of the primary response."],
     {}, None),
    ("histology", ["The simple columnar epithelium has microvilli and goblet cells,", "the cells rest on a basement membrane and are joined by tight junctions."],
     {}, None),
    ("anatomy", ["The brachial plexus gives rise to the median and the ulnar nerve,", "and the rotator cuff muscles stabilise the glenohumeral joint."],
     {}, None),
    ("cardiology", ["A patient with atrial fibrilation has an irregularly irregular pulse,", "and needs anticoagulation to prevent stroke."],
     {0: {"fibrilation": 0.4}}, ("fibrilation", "fibrillation")),
    ("neurology", ["Guillain-Barré syndrome causes an ascending paralysis with areflexia,", "after an infection such as Campylobacter."],
     {}, None),
    ("embryology", ["During gastrulation the primitive streak forms the three germ layers,", "and the notochord induces the neural plate to fold into the neural tube."],
     {}, None),
    ("genetics", ["Down syndrome is trisomy 21 caused by nondisjunction in meiosis,", "and cystic fibrosis follows autosomal recessive inheritance."],
     {}, None),
]


@pytest.mark.parametrize("subject,sentences,doubt,expected", CASES, ids=[f"{c[0]}-{i}" for i, c in enumerate(CASES)])
def test_each_field(subject, sentences, doubt, expected):
    result = correct(*sentences, subjects=[subject], probs=doubt)
    changes = applied(result)
    if expected is None:                                           # correct text stays exactly as spoken
        assert changes == [] and result.texts == sentences
    else:
        heard, wanted = expected
        done = [(c.original, c.corrected) for c in result.corrections if c.tier in ("high", "medium")]      # (spelling and word-split fixes included)
        assert any(heard in a and wanted in b for a, b in done) and any(wanted in text for text in result.texts), (done, result.texts)


def test_drug_names_stay_and_wrong_drug_spellings_follow_the_context():
    text = ["The patient was given warfarin and aspirin after the stroke,", "and the INR was checked because warfarin interacts with many drugs."]
    assert applied(correct(*text, subjects=["pharmacology"])) == []
    r = correct("The patient with a pulmonary embolism received warfrin as an oral anticoagulant,", "and the INR was monitored.", subjects=["pharmacology"],
                probs={0: {"warfrin": 0.4}})
    assert any(b == "warfarin" for _a, b in applied(r)), applied(r)
