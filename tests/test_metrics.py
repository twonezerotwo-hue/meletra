import pytest

from src.metrics import medical_term_scores, normalize_for_wer, word_error_rate
from src.vocabulary import VocabularyIndex, load_terms


@pytest.fixture(scope="module")
def vocab():
    return VocabularyIndex(load_terms("all"))


def test_wer_basic():
    assert word_error_rate("the cat sat on the mat", "the cat sat on the mat") == 0.0
    assert word_error_rate("the cat sat", "the dog sat") == pytest.approx(1 / 3)
    assert word_error_rate("a b c d", "") == 1.0


def test_wer_ignores_case_and_punctuation():
    assert word_error_rate("The Thalamus, relays!", "the thalamus relays") == 0.0


def test_wer_turkish_dotted_dotless_i():
    assert normalize_for_wer("IŞIK İLE", "tr") == "ışık ile"
    assert word_error_rate("IŞIK İLE", "ışık ile", "tr") == 0.0


def test_wer_empty_reference_rejected():
    with pytest.raises(ValueError):
        word_error_rate("", "anything")


def test_medical_term_recall_and_precision(vocab):
    ref = "The substantia nigra and the thalamus connect with the cerebellum."
    hyp = "The substance ignorant and the thalamus connect with the cerebellum and the amygdala."
    scores = medical_term_scores(ref, hyp, vocab)
    assert scores["reference_terms"] == 3
    assert scores["matched_terms"] == 2
    assert scores["recall"] == pytest.approx(2 / 3)
    assert scores["precision"] == pytest.approx(2 / 3)   # hyp has thalamus, cerebellum, amygdala
    assert scores["missed"] == ["substantia nigra"]
    assert scores["extra"] == ["amygdala"]


def test_medical_scores_handle_no_terms(vocab):
    scores = medical_term_scores("nothing special here", "nothing special here", vocab)
    assert scores["recall"] is None and scores["precision"] is None


def test_perfect_hypothesis_scores_one(vocab):
    text = "Broca area and Wernicke area are connected by the arcuate fasciculus."
    scores = medical_term_scores(text, text, vocab)
    assert scores["recall"] == 1.0 and scores["precision"] == 1.0 and scores["f1"] == 1.0
