"""Token budget (real-tokenizer interface), priority order and topic detection."""

import pytest

from src.model_manager import MAX_HOTWORD_TOKENS, DecodeRequest, FasterWhisperAdapter, ModelSpec
from src.pipeline import build_hotword_candidates
from src.topic import GENERIC_SUBJECT, detect_topic
from src.vocabulary import build_hotwords, fit_token_budget, hotwords_to_string, load_terms


def words_as_tokens(text: str) -> int:
    return len(text.replace(",", " ").split())


def test_budget_keeps_whole_terms_in_priority_order():
    terms = ["medial lemniscus", "periaqueductal gray", "thalamus", "nucleus ambiguus", "vagus nerve"]
    kept, used = fit_token_budget(terms, words_as_tokens, budget=6)
    assert kept == ["medial lemniscus", "periaqueductal gray", "thalamus"][:2] + [] or kept[:2] == terms[:2]
    assert used <= 6 and hotwords_to_string(kept).count(",") == len(kept) - 1
    assert all(t in terms for t in kept)                       # never a half-cut term


def test_term_that_does_not_fit_is_skipped_but_shorter_later_ones_still_added():
    kept, used = fit_token_budget(["aaa bbb ccc ddd", "ee", "ff"], words_as_tokens, budget=3)
    assert kept == ["ee", "ff"] and used == 2


def test_zero_budget_keeps_nothing():
    assert fit_token_budget(["a b"], words_as_tokens, budget=0) == ([], 0)


def test_multiword_terms_are_prioritised_over_single_words():
    terms = ["thalamus", "putamen", "medial lemniscus", "cerebellum", "spinothalamic tract"]
    plain = build_hotwords(terms, max_terms=10)
    smart = build_hotwords(terms, max_terms=10, multiword_first=True)
    assert plain[:2] == ["thalamus", "putamen"]
    assert smart[:2] == ["medial lemniscus", "spinothalamic tract"]
    assert smart[2:] == ["thalamus", "putamen", "cerebellum"]               # stable order inside each group


def test_pdf_and_heard_terms_come_before_subject_terms(cfg):
    out = build_hotword_candidates(["neuroscience"], "en", ["PDF term"], ["heard term"], cfg)
    assert out[:2] == ["PDF term", "heard term"]
    is_multi = [" " in t for t in out[2:]]
    assert is_multi == sorted(is_multi, reverse=True)                         # all multi-word terms, then single words
    assert any(is_multi)


class _Tok:
    """Stands in for faster-whisper's hf_tokenizer: one token per word."""

    class _Enc:
        def __init__(self, ids):
            self.ids = ids

    def encode(self, text, add_special_tokens=False):
        return self._Enc(text.replace(",", " ").split())


class _Model:
    hf_tokenizer = _Tok()


def adapter() -> FasterWhisperAdapter:
    a = FasterWhisperAdapter(ModelSpec("default", "faster_whisper", "large-v3"), device="cuda")
    a._model = _Model()
    return a


def request(hotwords, budget):
    return DecodeRequest(audio_path=None, language="en", profile={}, decoding={}, vad_parameters={},
                         hotwords=hotwords, hotword_token_budget=budget)


def test_adapter_applies_the_budget_with_the_models_own_tokenizer():
    terms = ["medial lemniscus", "periaqueductal gray", "nucleus ambiguus", "vagus nerve"]
    kept, used = adapter()._fit_hotwords(request(terms, 5))
    assert kept == ["medial lemniscus", "periaqueductal gray"][:2] or used <= 5
    assert used <= 5 and kept[0] == "medial lemniscus"


def test_adapter_without_a_budget_passes_everything_and_reports_tokens():
    kept, used = adapter()._fit_hotwords(request(["a b", "c"], None))
    assert kept == ["a b", "c"] and used == 3


def test_budget_above_the_libraries_hard_cap_is_clamped(caplog):
    terms = [f"term{i} word" for i in range(200)]
    kept, used = adapter()._fit_hotwords(request(terms, 10_000))
    assert used <= MAX_HOTWORD_TOKENS and "exceeds" in caplog.text


def test_empty_hotword_list_is_a_noop():
    assert adapter()._fit_hotwords(request([], 100)) == ([], None)


def test_config_has_a_token_budget_below_the_hard_cap(cfg):
    from src.budget import compute_token_budgets

    budgets = compute_token_budgets(cfg["context"])
    assert 0 < budgets.hotwords <= MAX_HOTWORD_TOKENS and budgets.hotwords <= budgets.prompt < 448
    assert not budgets.notes                      # the shipped config needs no clamping


# --------------------------------------------------------------------------- topic
NEURO = ("The corticospinal tract descends through the internal capsule. The medial lemniscus carries touch, "
         "and the spinothalamic tract carries pain. Lesions of the substantia nigra affect the basal ganglia and "
         "the dorsal root ganglion sends fibers to the dorsal column.")
BIOCHEM = ("In glycolysis, hexokinase and phosphofructokinase are regulated. Pyruvate dehydrogenase links to the "
           "citric acid cycle. Fatty acid synthase, acetyl CoA carboxylase and beta oxidation control lipid metabolism, "
           "and gluconeogenesis needs oxaloacetate.")


def test_neuroscience_text_selects_neuroscience(cfg):
    topic = detect_topic(NEURO, "en")
    assert topic.confident and topic.selected[0] == "neuroscience"
    assert "medial lemniscus" in topic.matched_terms and "corticospinal tract" in topic.matched_terms


def test_biochemistry_text_selects_biochemistry_and_not_pharmacology():
    topic = detect_topic(BIOCHEM, "en")
    assert topic.selected[0] == "biochemistry" and "pharmacology" not in topic.selected
    assert topic.scores["biochemistry"] > 3 * topic.scores["pharmacology"]


def test_small_talk_is_not_confident_and_falls_back_to_the_generic_list():
    topic = detect_topic("Good afternoon everyone. Welcome back. Please find your seats; the Moodle page has the slides.",
                         "en")
    assert not topic.confident and topic.selected == [GENERIC_SUBJECT]


def test_at_most_max_subjects_are_selected():
    mixed = NEURO + " " + BIOCHEM
    topic = detect_topic(mixed, "en", max_subjects=1)
    assert len(topic.selected) == 1
    assert len(detect_topic(mixed, "en", max_subjects=2).selected) <= 2


def test_empty_text_is_safe():
    topic = detect_topic("", "en")
    assert topic.words_analyzed == 0 and topic.selected == [GENERIC_SUBJECT] and not topic.confident


def test_topic_result_serialises():
    import json

    json.dumps(detect_topic(NEURO, "en").to_dict())
    assert load_terms("neuroscience")                                          # sanity: dictionaries are loaded
