"""Term priority engine, token budgets, hotword safeguard and the Medical Context Engine."""

import pytest

from src.budget import (
    DECODER_CONTEXT, LIBRARY_PART_CAP, PREVIOUS_TEXT_MAX, compute_token_budgets, effective_hotword_budget,
)
from src.context_engine import MedicalContextEngine
from src.guard import HOTWORD_GUARD_NOTE, apply_hotword_guard, prepare_decode_profile
from src.priority import DEFAULT_WEIGHTS, TermPriorityEngine, medical_specificity, rarity
from src.vocabulary import DEFAULT_TERMS_DIR, fit_token_budget

NEURO = ("The corticospinal tract descends through the internal capsule. The medial lemniscus carries touch, and the "
         "spinothalamic tract carries pain. Lesions of the substantia nigra affect the basal ganglia.")


# ------------------------------------------------------------------------------------------- priority engine
def test_weights_follow_the_specification_and_sum_to_one():
    assert DEFAULT_WEIGHTS == {"subject_match": 0.30, "pdf_presence": 0.25, "early_transcript_presence": 0.15,
                               "frequency": 0.10, "rarity": 0.10, "medical_specificity": 0.10}
    engine = TermPriorityEngine({"subject_match": 3, "pdf_presence": 1})
    assert sum(engine.weights.values()) == pytest.approx(1.0)


def test_pdf_and_subject_evidence_outrank_unrelated_terms():
    engine = TermPriorityEngine()
    ranked = engine.rank(
        {"substantia nigra": {"neuroscience"}, "propranolol": {"pharmacology"}, "glucose": {"general_medical"}},
        {"neuroscience": 0.9, "pharmacology": 0.0, "general_medical": 0.25},
        pdf_terms=["substantia nigra"], pdf_text="the substantia nigra", early_counts={"substantia nigra": 3})
    assert [r.term for r in ranked][0] == "substantia nigra"
    top = ranked[0].components
    assert top["subject_match"] == 0.9 and top["pdf_presence"] == 1.0 and top["early_transcript_presence"] == 1.0


def test_multi_word_terms_get_a_bonus_over_the_same_evidence():
    engine = TermPriorityEngine()
    ranked = engine.rank({"periaqueductal gray": {"neuroscience"}, "periaqueductal": {"neuroscience"}},
                         {"neuroscience": 0.8})
    assert ranked[0].term == "periaqueductal gray" and ranked[0].words == 2
    assert ranked[0].components["multiword_bonus"] > ranked[1].components["multiword_bonus"] == 0.0


def test_generic_single_words_are_demoted():
    engine = TermPriorityEngine()
    ranked = engine.rank({"enzyme": {"biochemistry"}, "phosphofructokinase": {"biochemistry"}}, {"biochemistry": 0.8})
    assert ranked[0].term == "phosphofructokinase"


def test_rarity_and_specificity_are_bounded():
    for term in ("glucose", "acetylcholinesterase", "internal capsule", "x"):
        assert 0.0 <= rarity(term) <= 1.0 and 0.0 <= medical_specificity(term) <= 1.0
    assert rarity("acetylcholinesterase") > rarity("pressure")
    assert medical_specificity("corticospinal tract") > medical_specificity("the thing")


# ------------------------------------------------------------------------------------------- token budgets
def test_separate_prompt_and_hotword_budgets_are_clamped_and_explained():
    ok = compute_token_budgets({"prompt_token_budget": 280, "hotwords_token_budget": 160, "generation_reserve_tokens": 160})
    assert (ok.prompt, ok.hotwords, ok.notes) == (280, 160, ())
    greedy = compute_token_budgets({"prompt_token_budget": 600, "hotwords_token_budget": 400})
    assert greedy.prompt <= DECODER_CONTEXT - 4 - greedy.generation_reserve
    assert greedy.hotwords <= LIBRARY_PART_CAP and greedy.hotwords <= greedy.prompt
    assert len(greedy.notes) == 2


def test_legacy_single_budget_key_still_works():
    assert compute_token_budgets({"hotword_token_budget": 100}).hotwords == 100


def test_conditioning_shrinks_the_hotword_budget_so_the_window_cannot_fill_up():
    budgets = compute_token_budgets({"prompt_token_budget": 280, "hotwords_token_budget": 160})
    assert effective_hotword_budget(budgets, conditioning=False) == (160, None)
    allowed, note = effective_hotword_budget(budgets, conditioning=True)
    assert allowed == budgets.prompt - PREVIOUS_TEXT_MAX and "previous text" in note
    tight = compute_token_budgets({"prompt_token_budget": 200, "hotwords_token_budget": 150})
    assert effective_hotword_budget(tight, True)[0] == 0


def test_token_budget_keeps_whole_terms_highest_priority_first():
    count = lambda text: len(text.split())          # noqa: E731 - stand-in tokenizer
    kept, used = fit_token_budget(["a b c", "d e f", "g", "h i"], count, 5)
    assert kept == ["a b c", "g"] and used == 4     # a term that does not fit is skipped whole, never cut in half


# ------------------------------------------------------------------------------------------- hotword safeguard
@pytest.fixture
def profile():
    return {"beam_size": 5, "temperature": 0, "condition_on_previous_text": True, "vad_filter": True}


def test_guard_switches_conditioning_off_whenever_hotwords_are_sent(cfg, profile):
    assert apply_hotword_guard(profile, ["substantia nigra"], cfg) == HOTWORD_GUARD_NOTE
    assert profile["condition_on_previous_text"] is False


def test_guard_leaves_a_run_without_hotwords_alone(cfg, profile):
    assert apply_hotword_guard(profile, [], cfg) is None and profile["condition_on_previous_text"] is True


def test_prepare_profile_never_mutates_the_base_profile_and_reports_notes(cfg, profile):
    budgets = compute_token_budgets(cfg["context"])
    prepared, budget, notes = prepare_decode_profile(profile, ["a", "b"], cfg, budgets)
    assert prepared["condition_on_previous_text"] is False and profile["condition_on_previous_text"] is True
    assert budget == budgets.hotwords and notes == [HOTWORD_GUARD_NOTE]
    _, none_budget, none_notes = prepare_decode_profile(profile, [], cfg, budgets)
    assert none_budget is None and none_notes == []


def test_even_with_the_guard_disabled_the_budget_protects_the_window(cfg, profile):
    relaxed = {**cfg, "context": {**cfg["context"], "hotwords_disable_conditioning": False}}
    budgets = compute_token_budgets(relaxed["context"])
    prepared, budget, notes = prepare_decode_profile(profile, ["a"], relaxed, budgets)
    assert prepared["condition_on_previous_text"] is True
    assert budget < budgets.hotwords and any("previous text" in n for n in notes)


# ------------------------------------------------------------------------------------------- context engine
def test_neuroscience_context_does_not_offer_pharmacology_terms(cfg):
    engine = MedicalContextEngine(cfg, DEFAULT_TERMS_DIR, "en", filename="neuro_week2.mp4")
    context = engine.build(NEURO)
    top = [t.lower() for t in context.hotword_candidates(30)]
    assert "corticospinal tract" in top and "substantia nigra" in top
    assert "propranolol" not in top
    assert context.selected_areas and context.confident


def test_pdf_terms_are_ranked_high_and_marked_as_pdf_evidence(cfg):
    engine = MedicalContextEngine(cfg, DEFAULT_TERMS_DIR, "en", pdf_terms=["Kluver-Bucy syndrome"],
                                  pdf_text="Kluver-Bucy syndrome follows bilateral amygdala lesions. Kluver-Bucy syndrome.")
    context = engine.build("")
    ranked = {r.term: r for r in context.ranked}
    assert "Kluver-Bucy syndrome" in ranked
    assert ranked["Kluver-Bucy syndrome"].components["pdf_presence"] == 1.0
    assert context.hotword_candidates(5).count("Kluver-Bucy syndrome") == 1


def test_user_subject_is_honoured_even_without_other_evidence(cfg):
    engine = MedicalContextEngine(cfg, DEFAULT_TERMS_DIR, "en", user_subjects=["pharmacology"])
    context = engine.build("")
    assert "pharmacology" in context.selected_areas
    assert "propranolol" in " ".join(t.lower() for t in context.hotword_candidates(300))


def test_rolling_adaptation_adds_terms_heard_in_a_chunk_without_exceeding_the_limit(cfg):
    engine = MedicalContextEngine(cfg, DEFAULT_TERMS_DIR, "en", filename="neuro.mp4")
    engine.build(NEURO)
    before = set(engine.context.hotword_candidates(60))
    entry = engine.update_with_heard("Now the periaqueductal gray and the red nucleus matter, and the red nucleus again.", 0)
    assert {"periaqueductal gray", "red nucleus"} <= set(entry["new_terms"])
    assert len(entry["new_terms"]) <= cfg["context_engine"]["rolling_new_terms"]
    after = engine.context.hotword_candidates(60)
    assert "periaqueductal gray" in after and "periaqueductal gray" not in before
    assert engine.context.rolling[-1]["chunk"] == 0
    assert "periaqueductal gray" in engine.heard_terms()


def test_learned_acronyms_become_candidates(cfg):
    engine = MedicalContextEngine(cfg, DEFAULT_TERMS_DIR, "en")
    engine.build("")
    engine.add_acronyms([{"acronym": "CSF", "expansion": "cerebrospinal fluid", "occurrences": 4}])
    assert "CSF" in {r.term for r in engine.context.ranked}


def test_context_report_is_json_serialisable(cfg):
    import json

    engine = MedicalContextEngine(cfg, DEFAULT_TERMS_DIR, "en", filename="neuro.mp4")
    data = engine.build(NEURO).to_dict()
    json.dumps(data)
    assert data["selected_areas"] and data["ranked_terms"][0]["components"]
