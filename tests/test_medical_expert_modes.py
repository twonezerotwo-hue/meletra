"""Expert modes of the corrector: Latin, abbreviations, eponyms, drug names, genes / proteins, Greek letters, numbers and units."""

import pytest

from src.medical.engine import capitalise_sentence_start, tidy
from tests.medical_support import applied, correct


def test_latin_terms_are_correct_as_they_are_and_never_translated():
    text = "The foramina of the skull transmit nerves, and the nuclei and ganglia lie near the septa and the rami."
    r = correct(text, subjects=["anatomy"])
    assert applied(r) == [] and r.texts[0] == text


def test_latin_and_english_forms_stay_separate():
    r = correct("The sulci and gyri of the cortex differ from the grooves and folds of the intestine.", subjects=["neuroanatomy"])
    assert applied(r) == [] and "sulci" in r.texts[0] and "grooves" in r.texts[0]


@pytest.mark.parametrize("heard,want", [("gfap", "GFAP"), ("mtor", "mTOR"), ("pka", "PKA"), ("gaba", "GABA"), ("dna", "DNA"), ("atp", "ATP")])
def test_abbreviations_get_their_exact_capitalisation(heard, want):
    r = correct(f"The role of {heard} in the signalling of the cell is important.", subjects=["molecular_biology"])
    assert want in r.texts[0].split() or f"{want}," in r.texts[0]


def test_ambiguous_abbreviations_need_the_context_and_ordinary_words_are_left():
    r = correct("The ms of the trial and the pd of the patient were no problem, as we were in a camp.")
    assert r.texts[0].startswith("The ms of the trial and the pd of the patient") and "camp" in r.texts[0]


def test_eponyms_keep_their_diacritics_and_spelling():
    r = correct("We owe the neuron doctrine to Santiago Ramon y Cajal and to Camillo Golgi, and the Purkinje cells are in the cerebellum.",
                subjects=["neuroanatomy"])
    assert "Santiago Ramón y Cajal" in r.texts[0] and "Golgi" in r.texts[0] and "Purkinje cells" in r.texts[0]


def test_misspelled_eponyms_follow_the_context():
    r = correct("In the peripheral nervous system the axons are wrapped by myelin,", "which is formed by a Schwan cell around one segment of the axon,",
                "and the nodes of Ranvier lie between neighbouring cells.", subjects=["neuroanatomy"])
    assert "Schwann cell" in r.texts[1]
    r = correct("The Meissner corpuscles sense light touch in the dermal papillae of the glabrous skin.")
    assert applied(r) == [] and "Meissner corpuscles" in r.texts[0]


def test_drug_names_are_never_swapped_for_other_drugs():
    text = "The patient takes metformin and hydralazine, not metronidazole or hydroxyzine, with clonidine and clonazepam on the list."
    r = correct(text, subjects=["pharmacology"])
    assert applied(r) == [] and r.texts[0] == text


def test_drug_names_need_stronger_evidence_than_other_terms():
    r = correct("He was given metoformin for his diabetes.", subjects=["pharmacology"])
    assert [(a, b) for a, b in applied(r) if b not in ("metformin",)] == []


def test_genes_and_proteins_keep_their_form():
    text = "Mutations of p53, BRCA1 and the Ras oncogene and the mTOR pathway are common."
    r = correct(text, subjects=["molecular_biology"])
    assert applied(r) == [] and r.texts[0] == text


def test_greek_letters_numbers_and_units_are_protected():
    for text in ("The alpha helix and the beta sheet differ from the gamma chain.", "The A-alpha and A-delta fibers are myelinated.",
                 "The dose was 5 mg every 8 hours and the pressure 120 mmHg with 140 mmol/L of sodium.",
                 "Concentrations of 10 µg per mL and 20 ng per mL were measured at 50 Hz and 70 mV."):
        r = correct(text, subjects=["physiology"])
        assert applied(r) == [] and r.texts[0] == text, text


def test_unit_symbols_after_numbers_are_formatted_without_touching_the_words():
    assert tidy("the dose was 5 mg and 10 ml of blood at 120 mmhg") == "the dose was 5 mg and 10 mL of blood at 120 mmHg"
    assert tidy("20 ug of drug , then 5 meq/l") == "20 µg of drug, then 5 mEq/L"
    assert tidy("an um 5 um thing") == "an um 5 um thing"                             # "um" is also a hesitation: not a unit
    assert tidy("the so -called myelin sheath") == "the so-called myelin sheath"
    assert tidy("minus -5 degrees") == "minus -5 degrees"


def test_sentence_capitals_follow_the_previous_segment_only():
    assert capitalise_sentence_start("the next sentence", "It ended.") == "The next sentence"
    assert capitalise_sentence_start("the next sentence", "and then it") == "the next sentence"
    assert capitalise_sentence_start("mTOR is a kinase", "Done.") == "mTOR is a kinase"
    assert capitalise_sentence_start("5 mg were given", "Done.") == "5 mg were given"
