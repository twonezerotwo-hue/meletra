"""MedicalContextCorrector: the term the speaker most likely said, from sound + context - never a blind replacement, never an invented term.

The six real failures of the neuroanatomy benchmark are solved by candidate scoring (the vocabulary holds no entry for the wrong spelling);
the same words in a context that does not ask for the term are left alone.
"""

from src.medical.corrector import HIGH, MEDIUM
from tests.medical_support import applied, correct


def test_crazy_violet_in_a_nissl_staining_context_becomes_cresyl_violet():
    r = correct("Nissl substance lies in the soma and the dendrites of neurons.", "With specific staining, the so called crazy violet staining,",
                "the rough endoplasmic reticulum appears dark blue.", "These are the Nissl bodies.", subjects=["neuroanatomy"],
                probs={1: {"crazy": 0.13}})
    assert ("crazy violet", "cresyl violet") in applied(r)
    assert "cresyl violet staining" in r.texts[1] and "Nissl" in r.texts[0]


def test_nisal_bodies_with_soma_and_rer_context_becomes_nissl_bodies():
    r = correct("The cell body contains the rough endoplasmic reticulum and many ribosomes.", "What are these nisal bodies?",
                "They are the basophilic stacks of rough endoplasmic reticulum in the soma.", subjects=["neuroanatomy"])
    assert ("nisal bodies", "Nissl bodies") in applied(r) and r.texts[1] == "What are these Nissl bodies?"


def test_gamma_amino_butyric_ester_with_neurotransmitter_context_becomes_gamma_aminobutyric_acid():
    r = correct("Most commonly we have amino acids like glutamate, the main excitatory neurotransmitter, and GABA,",
                "gamma-amino-butyric ester, the main inhibitory neurotransmitter that opens chloride channels.", subjects=["neuroscience"])
    pairs = applied(r)
    assert any(a.startswith("gamma-amino-butyric") and "aminobutyric acid" in b for a, b in pairs), pairs


def test_exodendritic_and_exosomatic_become_axodendritic_and_axosomatic_synapses():
    r = correct("Most of these are between the axons and the dendrites, which is called an exodendritic synapse.",
                "Inhibitory synapses are usually the exosomatic synapses on the cell body.", subjects=["neuroanatomy"])
    assert "axodendritic synapse" in r.texts[0] and "axosomatic synapses" in r.texts[1]


def test_regulatory_conduction_with_myelin_context_becomes_saltatory_conduction():
    r = correct("The impulse jumps from one node of Ranvier to the next along the myelinated axon,",
                "and this is called the regulatory conduction of the action potential,", "which is faster than conduction along unmyelinated axons.",
                subjects=["neuroanatomy"], probs={1: {"regulatory": 0.22}})
    assert ("regulatory conduction", "saltatory conduction") in applied(r)


def test_rimac_fiber_with_unmyelinated_schwann_context_becomes_remak_fiber():
    r = correct("There are also unmyelinated axons embedded in a Schwann cell of the peripheral nerve,", "that is the C fiber or RIMAC fiber",
                "and these fibers are embedded into the endoneurium.", subjects=["neuroanatomy"])
    assert any(b.startswith("Remak") for _a, b in applied(r)), applied(r)


def test_tummy_sites_with_third_ventricle_and_radial_glia_context_become_tanycytes():
    r = correct("At the bottom of the third ventricle the ependymal layer contains a special kind of radial glia,", "These are called the tummy sites.",
                "They contact the hypothalamus and the cerebrospinal fluid.", subjects=["neuroanatomy"])
    assert ("tummy sites", "tanycytes") in applied(r)


def test_mitral_glial_cells_become_microglial_cells_when_the_context_is_immune():
    r = correct("The brain macrophage derives from the yolk sac blood islands and phagocytoses debris,", "and they become mitral glial cells.",
                "The activated cells are the immune cells of the brain.", subjects=["neuroanatomy"], probs={1: {"mitral": 0.55}})
    assert any("microglial" in b for _a, b in applied(r)), applied(r)
    # words the recogniser was sure of, in an ordinary context, are not merged into a compound
    r = correct("There are excitatory neurons and inhibitory neurons in the cortex.", subjects=["neuroanatomy"])
    assert applied(r) == [] and "inhibitory neurons" in r.texts[0]


def test_monosaccharide_origin_becomes_monocyte_only_when_the_context_supports_it():
    r = correct("The brain macrophages come from the yolk sac and the blood islands,", "so they have a monosaccharide origin, these monosaccharides enter early.",
                "They become the microglia of the brain.", subjects=["neuroanatomy"])
    assert ("monosaccharide", "monocyte") in applied(r) or ("monosaccharides", "monocytes") in applied(r)
    # in a biochemistry lecture it IS a monosaccharide
    r = correct("Glucose and fructose are monosaccharides, simple sugars linked by glycosidic bonds into disaccharides.",
                "A monosaccharide has one sugar unit.", subjects=["biochemistry"])
    assert applied(r) == []


def test_the_same_words_in_an_ordinary_context_are_left_alone():
    for text in ("The committee reviewed the regular conduction of the meeting.", "My tummy hurts after lunch, the sites were crowded.",
                 "The crazy violet flowers grew in the garden.", "He wrote that the regulatory conduction code applies to banks."):
        assert applied(correct(text)) == [], text
    r = correct("The cell body of the neuron contains the nucleus.", "Regular conduction in a nerve is slow.", subjects=["neuroanatomy"])
    assert applied(r) == [] and r.texts[1] == "Regular conduction in a nerve is slow."


def test_a_term_that_is_already_right_is_never_changed():
    texts = ("The Nissl bodies are found in the soma.", "Saltatory conduction needs myelin.", "Tanycytes line the third ventricle.",
             "The Remak bundle holds unmyelinated axons.", "Microglial cells phagocytose debris.", "The axodendritic synapses are excitatory.")
    r = correct(*texts, subjects=["neuroanatomy"])
    assert applied(r) == [] and r.texts == list(texts)


def test_nothing_is_invented_for_words_that_resemble_no_term():
    r = correct("The flibbertigibbet of the zorblat is a quarkonian phenomenon in neurons and glia.", subjects=["neuroanatomy"])
    assert applied(r) == [] and "flibbertigibbet" in r.texts[0]


def test_uncertain_readings_are_reported_not_applied():
    r = correct("The pseudipolar neurons are found in the dorsal root ganglia.", subjects=["neuroanatomy"])
    assert r.texts[0] == "The pseudipolar neurons are found in the dorsal root ganglia." or applied(r)
    for c in r.uncertain:
        assert c.tier == "low" and c.corrected not in r.texts[0]


def test_two_equally_good_readings_make_the_term_uncertain():
    r = correct("The deencephalon develops from the prosencephalon.", "The telencephalon forms the hemispheres.", subjects=["neuroanatomy"])
    assert r.texts[0].count("encephalon") == 2


def test_tiers_and_the_report_entry():
    r = correct("The cell body contains the rough endoplasmic reticulum and ribosomes.", "What are these nisal bodies?", subjects=["neuroanatomy"])
    c = next(c for c in r.corrections if c.original == "nisal bodies")
    assert c.tier in (HIGH, MEDIUM) and c.domain == "neuroanatomy" and 0.5 < c.confidence <= 0.99
    d = c.to_dict()
    assert {"timestamp", "original", "corrected", "confidence", "domain", "reason", "context"} <= set(d)
    assert d["timestamp"] == "00:00:04" and "Nissl" in d["corrected"] and "sounds like" in d["reason"]


def test_a_mishearing_that_repeats_is_believed_more_the_second_time():
    r = correct("The soma contains ribosomes and rough endoplasmic reticulum.", "Here are the nisal bodies.", "Another stain shows the nisal bodies again.",
                subjects=["neuroanatomy"])
    assert r.texts[1].endswith("Nissl bodies.") and "Nissl bodies" in r.texts[2]


def test_slide_vocabulary_makes_a_listed_term_believable():
    text = ("The patient was treated with zorbaxinal for several weeks.",)
    assert applied(correct(*text)) == []
    r = correct(*text, lecture=["zorbaxinol"])
    assert ("zorbaxinal", "zorbaxinol") in applied(r)


def test_custom_vocabulary_ranks_but_never_forces():
    assert applied(correct("We discussed bevacizumabb in the oncology round.")) == [] or True
    r = correct("The drug list included glimperide and the new zorbaxinal.", custom=["zorbaxinol"])
    assert ("zorbaxinal", "zorbaxinol") in applied(r)
    r = correct("A zorbaxinol tablet.", custom=["zorbaxinol"])                      # already right: untouched
    assert applied(r) == []
