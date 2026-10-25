"""More validated facts from a noisy lecture without lowering the bar (src/study/recall.py, factbase.py) and the defects found by the developer inspection of real packs."""

from types import SimpleNamespace

import pytest

from src.study import facts as F
from src.study import recall as R
from src.study.engine import StudyEngine
from src.study.factbase import build_factbase
from src.study.factitems import FactStudy, _the
from src.study.index import build_index
from src.study.providers import StudyContext


def segs(*sentences, seconds=12.0, low=None):
    out = []
    for i, s in enumerate(sentences):
        seg = {"start": i * seconds, "end": (i + 1) * seconds - 0.5, "text": s}
        if low and i in low:
            seg["low_words"] = low[i]
        out.append(seg)
    return out


# ======================================================================================================================== recall rules
def test_a_doubted_word_is_corroborated_only_when_it_is_known_and_the_lecture_repeats_it():
    counts = {"neural": 12, "tube": 9, "otocord": 3, "fate": 1}
    unknown = lambda w: w == "otocord"                                                                    # noqa: E731
    assert R.doubt_is_corroborated("this", counts, unknown)                                               # a function word
    assert R.doubt_is_corroborated("neural", counts, unknown) and R.doubt_is_corroborated("tube", counts, unknown)
    assert not R.doubt_is_corroborated("otocord", counts, unknown)                                       # unknown: a recogniser error until proven otherwise
    assert not R.doubt_is_corroborated("fate", counts, unknown)                                          # said once: nothing corroborates it


def test_the_lecturers_lead_in_is_removed_and_a_resumptive_pronoun_dropped():
    text, emphatic = R.strip_lead_in("Based on this picture, we can see that cells with high proliferation capacity, they locate near the central canal.")
    assert text == "Cells with high proliferation capacity locate near the central canal." and not emphatic
    text, emphatic = R.strip_lead_in("You have to remember that the notochord induces the neural plate.")
    assert text == "The notochord induces the neural plate." and emphatic
    assert R.strip_lead_in("The notochord induces the neural plate.") == ("", False)


def test_it_and_they_take_the_subject_of_the_previous_sentence_when_the_number_agrees():
    assert R.resolve_pronoun("It later forms the gray matter.", "intermediate zone", False, "The") == "The intermediate zone later forms the gray matter."
    assert R.resolve_pronoun("They fuse in the midline.", "neural folds", True, "The") == "The neural folds fuse in the midline."
    assert R.resolve_pronoun("They fuse in the midline.", "intermediate zone", False, "The") == ""          # number does not agree
    assert R.resolve_pronoun("It is important to remember this.", "neural tube", False, "The") == ""          # impersonal "it"
    assert R.resolve_pronoun("It can be 1 to 500 cases.", "neural tube", False, "The") == ""
    assert R.resolve_pronoun("It activates PFK-1.", "Insulin", False, "") == "Insulin activates PFK-1."     # a molecule takes no article


@pytest.mark.parametrize("statement, pointer", [
    ("Parallel with this process, a solution starts to fill this lumen.", "this process"),
    ("The spinal cord or the medulla oblongata will have the same organization.", "the same organization"),
    ("To prevent this, glutamate is kept under very low concentrations in the circulation.", "prevent this,"),
    ("The D-amino acid can bind to such enzymes and block it.", "such enzymes"),
    ("In this way, the NAD is regenerated.", "this way"),
    ("More ammonia reaches the pericentral hepatocytes.", "More"),
    ("The process is called primary neurulation.", "The process"),
    ("Not surprisingly, the enzyme is called N-acetylglutamate synthase.", "Not surprisingly,"),
])
def test_a_statement_that_points_back_is_found(statement, pointer):
    assert R.points_back(statement).startswith(pointer.split()[0])


@pytest.mark.parametrize("statement", ["Glycolysis is the pathway that converts glucose to pyruvate in the cytoplasm.", "The central canal is the lumen of the neural tube.",
                                       "The neural folds fuse and form the neural tube."])
def test_a_statement_that_stands_alone_is_not_flagged(statement):
    assert R.points_back(statement) == ""


def test_a_medium_fact_is_used_only_with_clear_support():
    known = lambda name: name in ("central canal", "human embryo")                                       # noqa: E731
    category = {"central canal": "structure", "human embryo": "other"}.get
    fact = lambda tier, doubts, subject="central canal", text="The central canal is the lumen of the neural tube.": SimpleNamespace(  # noqa: E731
        tier=tier, doubts=doubts, subtopic=subject, statement=text)
    assert R.usable(fact("HIGH", []), known, category)
    assert R.usable(fact("MEDIUM", ["clause of a longer sentence"]), known, category)
    assert not R.usable(fact("MEDIUM", ["clause of a longer sentence", "lead-in removed"]), known, category)               # two rescues: compounded doubt
    assert not R.usable(fact("MEDIUM", ["recogniser doubted, corroborated: posterior"]), known, category)                  # a heard doubt stays out of items
    assert not R.usable(fact("MEDIUM", ["points back to an earlier sentence: this fold"]), known, category)
    assert not R.usable(fact("MEDIUM", ["clause of a longer sentence"], "human embryo", "Human embryo is about the end of the 14th day."), known, category)


# ======================================================================================================================== grammar checks from the inspection
@pytest.mark.parametrize("sentence, issue", [
    ("The epithelial cells will reorganize the molecular and the cell membrane, cell adhesion molecules.", "broken grammar"),
    ("The epithelial cells will reorganize the molecular.", "broken grammar"),
    ("The dorsal horn, lateral horn, and the ventral horn develops from the intermediate zone of the spinal cord.", "subject and verb disagree"),
    ("The skeletal muscle is a site where.", "unfinished sentence"),
    ("The methylene tetrahydrofolate will be important.", "says only that it matters"),
    ("To regenerates the ornithine that is needed for the mitochondrial stage of the urea cycle.", "dependent clause only"),
    ("The gives the name of the group of amino acids.", "no subject"),
    ("In the tubular lumen, it binds protons, which cannot go back.", "pronoun subject after an introductory phrase"),
])
def test_garbled_statements_found_in_real_packs_are_refused(sentence, issue):
    assert issue in F.sentence_issues(sentence).issues


@pytest.mark.parametrize("sentence", ["The central and the peripheral nervous system develop from the neural tube.",
                                      "The neural tube contains the cells of the central nervous system and the peripheral nervous system.",
                                      "Insulin and glucagon regulate the blood glucose level in the body.", "The central canal is the lumen of the neural tube."])
def test_correct_sentences_are_not_caught_by_the_new_checks(sentence):
    assert F.sentence_issues(sentence).ok, F.sentence_issues(sentence).issues


def test_a_shorter_statement_inside_a_longer_one_is_subsumed_and_generic_nouns_are_bare():
    assert F.subsumed("The intermediate zone cells contain the precursors for the neurons.",
                      "The intermediate zone contains the precursors for the glia and neurons for the brain and the spinal cord.")
    assert not F.subsumed("The central canal is the lumen of the neural tube.", "The neural folds fuse and form the neural tube.")
    assert F.is_bare_generic("enzyme") and F.is_bare_generic("the proteins") and not F.is_bare_generic("N-acetylglutamate synthase")


def test_named_cycles_and_pathways_take_the_article():
    assert _the("urea cycle", "process") == "the urea cycle" and _the("citric acid cycle", "process") == "the citric acid cycle"
    assert _the("glycolysis", "process") == "glycolysis" and _the("neural tube", "structure") == "the neural tube"


# ======================================================================================================================== the builder
def test_a_pronoun_sentence_after_a_validated_one_becomes_a_medium_fact_citing_both():
    index = build_index(segs("The intermediate zone contains the neuronal cell bodies of the developing spinal cord.",
                             "It later forms the gray matter of the spinal cord.",
                             "The marginal zone contains the nerve fibers of the developing spinal cord.",
                             "The notochord induces the neural plate in the dorsal ectoderm of the embryo."), ())
    base = build_factbase(index)
    resolved = [f for f in base.facts if f.statement.startswith("The intermediate zone later forms")]
    assert resolved and resolved[0].tier == "MEDIUM" and "pronoun resolved from the previous sentence" in resolved[0].doubts
    assert "It later forms" in resolved[0].source_excerpt and "contains the neuronal cell bodies" in resolved[0].source_excerpt   # the fact cites both sentences
    assert all(f.tier == "HIGH" for f in base.facts if f.statement.startswith("The notochord induces"))
    assert base.reject_report()["medium"] >= 1


def test_a_fate_gets_a_become_card_and_a_passive_a_where_card():
    index = build_index(segs("The marginal zone remains as the white matter of the spinal cord in the adult.",
                             "The marginal zone contains the nerve fibers of the developing spinal cord.",
                             "Urea is formed in the liver from ammonia and carbon dioxide by the urea cycle.",
                             "The urea cycle converts toxic ammonia into urea in the liver cells.",
                             "The notochord induces the neural plate in the dorsal ectoderm of the embryo."), ())
    study = FactStudy(StudyContext(index=index))
    cards = study.flashcards(index.chunks, "", (), 20)
    fronts = [c.front for c in cards]
    assert any(c.front == "What does the marginal zone become?" and "white matter" in c.back for c in cards)
    assert not any(c.front == "What does the marginal zone do?" and "white matter" in c.back for c in cards)       # a fate is not something it does
    assert not any(f.startswith("What does urea do") for f in fronts)


def test_key_concepts_say_why_they_matter():
    index = build_index(segs(*(["The neural tube forms the brain and the spinal cord of the embryo."] + [f"The neural tube closes in week {n} of the development of the embryo." for n in (3, 4)]
                               + ["You have to remember that the notochord induces the neural plate in the dorsal ectoderm."])), ())
    pack = StudyEngine().generate(index, title="concepts")
    concepts = pack.summaries.get("key_concepts", [])
    assert concepts and all(c.importance for c in concepts)
    assert all(c.importance.startswith(("High-yield", "Core term", "Supporting term")) for c in concepts)


@pytest.mark.parametrize("sentence, issue", [
    ("The cells which make up the cortex and the cortical lamination.", "no main verb"),
    ("According to the direction of the fibers can individually be traced back.", "missing subject"),
    ("Without the cell bodies can be named pericardium, soma, cell body, several names.", "missing subject"),
    ("Gray matter, everybody knows what it means.", "speaker's voice"),
    ("These interneurons, in example one, they are majorly inhibitory neurons.", "pronoun subject after an introductory phrase"),
    ("One component is crossing, one part is uncrossed, right.", "spoken filler"),
])
def test_conversational_fragments_of_a_third_lecture_are_refused(sentence, issue):
    assert issue in F.sentence_issues(sentence).issues


@pytest.mark.parametrize("sentence", ["The neurons that line the canal divide rapidly in the embryo.", "The cells that form the neural crest migrate to many places in the body."])
def test_a_relative_clause_with_a_main_verb_after_it_is_fine(sentence):
    assert "no main verb" not in F.sentence_issues(sentence).issues


def test_otherwise_and_one_of_them_point_back():
    assert R.points_back("Otherwise, the muscle would be always in a tetanic cramp.").startswith("Otherwise")
    assert R.points_back("One is going up and ending up in the cortex via the thalamus.").startswith("One is")


def test_an_indefinite_article_before_a_plural_is_broken_grammar_but_a_series_of_is_not():
    assert "broken grammar" in F.sentence_issues("A dendritic spines fine on the other side so this is a spine with the neck.").issues
    assert F.sentence_issues("A series of reactions converts glucose to pyruvate in the cytoplasm.").ok


def test_an_ordinary_english_word_is_not_blanked_or_made_a_key_concept():
    index = build_index(segs("The axon takes its origin from the axon hillock of the nerve cell.", "The axon carries the action potential away from the cell body of the neuron.",
                             "The dendrites receive the synaptic input of the neuron from other cells.", "The axon hillock starts the action potential of the neuron."), ())
    study = FactStudy(StudyContext(index=index))
    cards = study.flashcards(index.chunks, "", (), 20)
    assert not any(c.back.startswith("origin -") for c in cards)


def test_anywhere_else_points_back():
    assert R.points_back("Urea is not synthesized anywhere else.") and R.points_back("Any surplus of amino acids above this is degraded.")
    assert R.points_back("The first task is in the intestine.") and R.points_back("The product is shown in the name of the enzyme, carbamoyl phosphate.")
