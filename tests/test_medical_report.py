import copy

import pytest

from src.exporters import possible_errors_text, terms_report_text
from src.medical_report import build_report, detect_candidate_terms, detect_terms, scan_possible_errors
from src.schema import Word
from src.vocabulary import VocabularyIndex, load_terms
from tests.conftest import make_segment


@pytest.fixture(scope="module")
def vocab():
    return VocabularyIndex(load_terms("all", "en"))


def test_dictionary_terms_are_counted_with_first_timestamp(vocab):
    segs = [make_segment("Intro without terms.", 0, 5),
            make_segment("The thalamus relays input to the cortex.", 100, 106),
            make_segment("The thalamus and the substantia nigra differ.", 200, 206)]
    stats = {t.term: t for t in detect_terms(segs, vocab)}
    assert stats["thalamus"].count == 2 and stats["thalamus"].first_start >= 100
    assert stats["substantia nigra"].first_start >= 200


def test_cortical_spinal_tract_is_suggested_as_corticospinal_tract(vocab):
    seg = make_segment("Lesions of the cortical spinal tract cause spasticity.", 125, 131)
    [err] = scan_possible_errors([seg], vocab)
    assert err.original == "cortical spinal tract" and err.suggestion == "corticospinal tract"
    assert err.kind == "phrase" and 0.86 <= err.confidence <= 1.0
    assert 125 <= err.start <= 131


def test_misspelled_single_words_are_found(vocab):
    segs = [make_segment("The medial lemoniscus carries touch.", 10, 14),
            make_segment("A spinothalmic pathway carries pain.", 20, 24)]
    found = {(e.original, e.suggestion) for e in scan_possible_errors(segs, vocab)}
    assert ("lemoniscus", "lemniscus") in found or ("medial lemoniscus", "medial lemniscus") in found
    assert ("spinothalmic", "spinothalamic") in found


def test_exact_terms_variants_and_inflections_are_not_reported(vocab):
    segs = [make_segment("The corticospinal tract and the medial lemniscus and the Broca area.", 0, 5),
            make_segment("The spino thalamic tract is a valid spacing variant.", 5, 9),
            make_segment("Beta receptors and antagonists differ.", 9, 12)]
    assert scan_possible_errors(segs, vocab) == []


def test_the_analyzer_never_edits_the_transcript(vocab):
    segs = [make_segment(" The cortical spinal tract and a medial lemoniscus.", 0, 5)]
    before = copy.deepcopy(segs)
    report = build_report(segs, vocab)
    assert segs == before and segs[0].text == " The cortical spinal tract and a medial lemoniscus."
    assert report.errors                                                        # but it does report them


def test_word_timestamps_are_used_for_the_error_time(vocab):
    seg = make_segment("Hello there the cortical spinal tract is long.", 0, 20)
    words = seg.text.split()
    seg.words = [Word(i * 2.0, i * 2.0 + 1.0, w, 0.9) for i, w in enumerate(words)]
    [err] = scan_possible_errors([seg], vocab)
    assert err.start == pytest.approx(6.0)                                       # 4th word ("cortical") starts at 6 s


def test_candidates_outside_the_dictionary_need_repetition(vocab):
    segs = [make_segment("Thyroiditis is common.", 5, 8), make_segment("Again thyroiditis, and glomerulitis once.", 50, 55),
            make_segment("Pancreatitis is already in the dictionary and twice pancreatitis.", 60, 65)]
    names = {c.term.lower() for c in detect_candidate_terms(segs, vocab, min_count=2)}
    assert "thyroiditis" in names and "glomerulitis" not in names              # needs 2+ occurrences
    assert "pancreatitis" not in names                                          # dictionary terms are not "candidates"


def test_report_files_have_the_requested_fields(vocab):
    seg = make_segment("The cortical spinal tract.", 3725, 3730)
    report = build_report([seg], vocab)
    text = possible_errors_text(report.errors)
    for label in ("Original phrase:", "Possible medical term:", "Timestamp: 01:02:0", "Similarity:", "Subject:",
                  "PDF evidence:", "Reason:", "Confidence:"):   # 3725 s ~ 01:02:05
        assert label in text
    assert "SUGGESTIONS only" in text
    assert "MEDICAL TERMS DETECTED" in terms_report_text(report.terms, report.candidates)
