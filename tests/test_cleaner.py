import copy

import pytest

from src.cleaner import clean_segments, strip_markers, word_keys
from src.exporters import raw_transcript_text
from src.uncertainty import UncertaintyConfig, annotate_segments
from src.vocabulary import VocabularyIndex, load_terms
from tests.conftest import make_segment


@pytest.fixture(scope="module")
def vocab():
    return VocabularyIndex(load_terms("all"))


def test_spacing_and_eponym_case_fixes(cfg, vocab):
    seg = make_segment(" The spino thalamic tract and the broca area .", 0, 4)
    result = clean_segments([seg], vocab, cfg["cleaner"])
    assert "spinothalamic tract" in result.text
    assert "Broca area" in result.text
    assert "area." in result.text                      # space before punctuation tidied
    kinds = {c["type"] for c in result.changes}
    assert kinds == {"term_spacing", "eponym_case"}


def test_existing_correct_terms_and_sentence_capitals_are_left_alone(cfg, vocab):
    seg = make_segment("Substantia nigra produces dopamine. Broca area speaks.", 0, 4)
    result = clean_segments([seg], vocab, cfg["cleaner"])
    assert result.text.strip() == "Substantia nigra produces dopamine. Broca area speaks."
    assert result.changes == []


def test_raw_segments_are_never_modified(cfg, vocab):
    segs = [make_segment(" spino thalamic tract damage.", 0, 2), make_segment(" broca area lesion", 2, 4)]
    before_text = [s.text for s in segs]
    before_raw = raw_transcript_text(segs)
    snapshot = copy.deepcopy(segs)
    clean_segments(segs, vocab, cfg["cleaner"])
    assert [s.text for s in segs] == before_text
    assert raw_transcript_text(segs) == before_raw
    assert segs == snapshot


def test_cleaning_never_deletes_or_adds_words(cfg, vocab):
    segs = [
        make_segment("So  um the thalamus relays   sensory input , right?", 0, 3),
        make_segment("The the the the the lesion.", 3, 6),
        make_segment("Uh.", 6, 7),
    ]
    annotate_segments(segs, UncertaintyConfig.from_config(cfg), vocab)
    result = clean_segments(segs, vocab, cfg["cleaner"])
    original_words = [k for s in segs for k in word_keys(s.text)]
    cleaned_words = word_keys(strip_markers(result.text))
    assert cleaned_words == original_words


def test_uncertain_segments_are_marked_not_removed(cfg, vocab):
    seg = make_segment("Probably something about the nerve.", 0, 3, avg_logprob=-1.6)
    annotate_segments([seg], UncertaintyConfig.from_config(cfg), vocab)
    result = clean_segments([seg], vocab, cfg["cleaner"])
    assert result.text.strip() == "[UNCERTAIN: Probably something about the nerve.]"


def test_near_miss_word_gets_inline_marker_but_stays(cfg, vocab):
    seg = make_segment("The spinothalmic tract carries pain.", 0, 3)
    annotate_segments([seg], UncertaintyConfig.from_config(cfg), vocab)
    text = clean_segments([seg], vocab, cfg["cleaner"]).text
    assert 'spinothalmic [UNCERTAIN: resembles "spinothalamic"]' in text
    assert not text.startswith("[UNCERTAIN: The")


def test_paragraphs_follow_pauses(cfg, vocab):
    segs = [make_segment("First idea here.", 0, 3), make_segment("Same paragraph.", 3.5, 6),
            make_segment("New topic after a pause.", 12, 15)]
    text = clean_segments(segs, vocab, cfg["cleaner"]).text
    assert text == "First idea here. Same paragraph.\n\nNew topic after a pause.\n"


def test_turkish_text_passes_through(cfg, vocab):
    seg = make_segment("Şişman bağırsağın ileum kısmı.", 0, 3, language="tr")
    assert clean_segments([seg], vocab, cfg["cleaner"]).text.strip() == "Şişman bağırsağın ileum kısmı."
