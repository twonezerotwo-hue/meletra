"""The scoring harness of the v4 benchmark: every region of difference between GOLD, RAW and ENGINE text is classified exactly once."""

from src.medical_knowledge.benchmarks.evaluate import evaluate, merge, words

GOLD = "The Nissl bodies contain rough endoplasmic reticulum and the Schwann cell wraps the axon."
TERMS = ["Nissl bodies", "rough endoplasmic reticulum", "Schwann cell", "axon"]


def kinds(ev):
    return sorted((r.kind, r.medical) for r in ev.regions)


def test_a_true_correction_is_counted_as_corrected_and_raises_term_recall():
    raw = "The nisal bodies contain rough endoplasmic reticulum and the Schwann cell wraps the axon."
    ev = evaluate(GOLD, raw, GOLD, TERMS)
    assert kinds(ev) == [("corrected", True)] and ev.summary()["precision"] == 1.0 and ev.summary()["medical_recall"] == 1.0
    assert ev.terms_total == 4 and ev.terms_raw_ok == 3 and ev.terms_engine_ok == 4


def test_an_error_the_engine_leaves_is_missed():
    raw = "The nisal bodies contain rough endoplasmic reticulum and the Schwann cell wraps the axon."
    ev = evaluate(GOLD, raw, raw, TERMS)
    assert kinds(ev) == [("missed", True)] and ev.summary()["precision"] is None and ev.summary()["medical_recall"] == 0.0


def test_a_change_to_the_wrong_term_is_wrong_not_corrected():
    raw = "The nisal bodies contain rough endoplasmic reticulum and the Schwann cell wraps the axon."
    engine = "The nasal bones contain rough endoplasmic reticulum and the Schwann cell wraps the axon."
    ev = evaluate(GOLD, raw, engine, TERMS)
    assert kinds(ev) == [("wrong", True)] and ev.summary()["precision"] == 0.0


def test_damaging_correct_speech_is_the_worst_class_and_is_counted_per_thousand_words():
    engine = "The Nissl bodies contain rough endoplasmic reticulum and the Swan cell wraps the axon."
    ev = evaluate(GOLD, GOLD, engine, TERMS)
    s = ev.summary()
    assert kinds(ev) == [("false", True)] and s["false"] == 1 and s["precision"] == 0.0 and s["false_per_1000_words"] > 50


def test_spacing_numbers_case_and_punctuation_are_not_errors():
    assert words("DNA polymerase 3, p53 and CRISPR-Cas9!") == ["dna", "polymerase", "three", "p", "fifty", "three", "and", "crispr", "cas", "nine"]
    ev = evaluate("The brainstem and the DNA polymerase three.", "The brain stem and the DNA polymerase 3.", "The brain stem and the DNA polymerase 3.", ["brainstem"])
    assert ev.regions == [] and ev.terms_total == 1                                  # a spacing variant is not an error region (term recall is stricter on purpose)


def test_a_mixed_clip_is_split_into_independent_regions():
    raw = "The nisal bodies contain rough endoplasmic reticulum and the Swan cell wraps the exon."
    engine = "The Nissl bodies contain rough endoplasmic reticulum and the Swan cell wraps the exon."
    ev = evaluate(GOLD, raw, engine, TERMS)
    assert kinds(ev) == [("corrected", True), ("missed", True), ("missed", True)]
    s = ev.summary()
    assert s["corrected"] == 1 and s["missed"] == 2 and s["precision"] == 1.0 and abs(s["medical_recall"] - 1 / 3) < 1e-3


def test_insertions_and_deletions_are_regions_too():
    ev = evaluate("the axon hillock starts it", "the axon starts it", "the axon hillock starts it", ["axon hillock"])
    assert kinds(ev) == [("corrected", True)]
    ev2 = evaluate("the axon starts it", "the axon hillock starts it", "the axon hillock starts it", [])
    assert kinds(ev2) == [("missed", False)]


def test_merging_clips_adds_everything_up():
    a = evaluate(GOLD, "The nisal bodies contain rough endoplasmic reticulum and the Schwann cell wraps the axon.", GOLD, TERMS)
    b = evaluate("a plain sentence here", "a plain sentence here", "a plain sentence there", [])
    total = merge([a, b])
    s = total.summary()
    assert s["corrected"] == 1 and s["false"] == 1 and s["precision"] == 0.5 and total.gold_words == 18


def test_a_neighbouring_error_does_not_turn_a_corrected_term_into_a_wrong_region():
    gold = "Langerhans in response to injury."
    raw = "longerhans and responds to injury."
    engine = "Langerhans and responds to injury."
    ev = evaluate(gold, raw, engine, ["Langerhans"])
    assert [(r.kind, r.medical) for r in ev.regions] == [("corrected", True), ("missed", False), ("missed", False)]


# ------------------------------------------------------------------------------------------------------------------ combined system and base-passage weights
def test_the_combined_system_adds_v4_where_production_changed_nothing_and_production_wins_conflicts():
    import sys
    from pathlib import Path
    from types import SimpleNamespace as NS

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    from v4_lib import overlay

    raw = ["the astrocite touches the sell body and the nisal bodies"]
    production = ["the astrocite touches the cell body and the nisal bodies"]               # production fixed "sell body" only
    v4 = [NS(segment=0, start=4, end=13, replacement="astrocyte"),                           # astrocite: production left it alone -> v4 adds it
          NS(segment=0, start=24, end=33, replacement="cell body")]                          # sell body: production already decided here -> dropped
    assert overlay(raw, production, v4) == ["the astrocyte touches the cell body and the nisal bodies"]
    assert overlay(raw, [], v4) == raw                                                       # no production text for the clip: nothing to overlay on


def test_base_passage_weights_count_every_passage_once():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    from v4_lib import _wsummary
    from src.medical_knowledge.benchmarks.evaluate import evaluate

    one = evaluate("the astrocyte is a glial cell", "the astrocite is a glial cell", "the astrocyte is a glial cell", ["astrocyte"])
    two = evaluate("the astrocyte is a glial cell", "the astrocite is a glial cell", "the astrocite is a glial cell", ["astrocyte"])
    assert _wsummary([one, two], [0.5, 0.5])["corrected"] == 0.5 and _wsummary([one, two], [0.5, 0.5])["missed"] == 0.5
