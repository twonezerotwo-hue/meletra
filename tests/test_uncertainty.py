import pytest

from src.uncertainty import (
    UncertaintyConfig, annotate_segments, compute_confidence, find_repetition,
)
from src.vocabulary import VocabularyIndex, load_terms
from tests.conftest import make_segment


@pytest.fixture
def ucfg(cfg):
    return UncertaintyConfig.from_config(cfg)


@pytest.fixture(scope="module")
def vocab():
    return VocabularyIndex(load_terms("neuroscience"))


def codes(segment):
    return {flag.code for flag in segment.flags}


def test_clean_segment_is_not_flagged(ucfg, vocab):
    seg = make_segment("The corticospinal tract descends through the internal capsule.")
    assert annotate_segments([seg], ucfg, vocab) == 0
    assert not seg.uncertain and seg.flags == []


def test_low_avg_logprob(ucfg):
    seg = make_segment("Something muddy was said here.", avg_logprob=-1.4)
    annotate_segments([seg], ucfg)
    assert "low_avg_logprob" in codes(seg)


def test_threshold_is_strictly_below(ucfg):
    seg = make_segment("Borderline value is fine here.", avg_logprob=-1.0)
    annotate_segments([seg], ucfg)
    assert "low_avg_logprob" not in codes(seg)


def test_high_no_speech_probability(ucfg):
    seg = make_segment("Thank you for watching everyone.", no_speech_prob=0.85)
    annotate_segments([seg], ucfg)
    assert "high_no_speech_prob" in codes(seg)


def test_high_compression_ratio(ucfg):
    seg = make_segment("blah blah something", compression_ratio=3.1)
    annotate_segments([seg], ucfg)
    assert "high_compression_ratio" in codes(seg)


def test_thresholds_are_read_from_config_not_hardcoded(cfg):
    cfg2 = {**cfg, "uncertainty": {**cfg["uncertainty"], "avg_logprob_threshold": -0.1}}
    strict = UncertaintyConfig.from_config(cfg2)
    seg = make_segment("A perfectly normal sentence about physiology.", avg_logprob=-0.3)
    annotate_segments([seg], strict)
    assert "low_avg_logprob" in codes(seg)


def test_too_short_segment(ucfg):
    seg = make_segment("Uh.")
    annotate_segments([seg], ucfg)
    assert "too_short" in codes(seg)
    long_single = make_segment("Thalamus")
    annotate_segments([long_single], ucfg)
    assert "too_short" not in codes(long_single)


def test_repetition_and_hallucination_loops(ucfg):
    seg = make_segment("the the the the the nerve")
    annotate_segments([seg], ucfg)
    assert "repetition" in codes(seg)
    assert find_repetition("it goes up and down up and down up and down up and down", 4, 4)
    assert find_repetition("a normal sentence with no loops at all", 4, 4) is None


def test_identical_consecutive_segments_are_flagged(ucfg):
    segs = [make_segment("Subtitles by the community.", i * 2, i * 2 + 2) for i in range(3)]
    annotate_segments(segs, ucfg)
    assert all("repeated_segment" in codes(s) for s in segs)


def test_truncated_long_segment_is_flagged_but_normal_speech_is_not(ucfg):
    cut = make_segment("The substantia nigra pars compactus supplies dopamine to the stri", 73.3, 100.1)
    normal = make_segment("The corticospinal tract originates in the primary motor cortex and descends through "
                          "the internal capsule and the cerebral peduncle.", 0.0, 9.5)
    short = make_segment("Okay so let us continue.", 10.0, 12.5)
    annotate_segments([cut, normal, short], ucfg)
    assert "low_speech_density" in codes(cut)
    assert "low_speech_density" not in codes(normal) and "low_speech_density" not in codes(short)


def test_sentence_repeated_a_few_segments_later_is_flagged(ucfg):
    line = "The subthalamic nucleus is an excitatory structure, and"
    segs = [make_segment(line, 0, 3), make_segment("A different sentence about the thalamus here.", 3, 6),
            make_segment("Another unrelated sentence on the cerebellum follows.", 6, 9), make_segment(line, 9, 12)]
    annotate_segments(segs, ucfg)
    assert "duplicate_nearby" in codes(segs[3]) and "duplicate_nearby" not in codes(segs[0])
    # short stock phrases may legitimately repeat
    stock = [make_segment("Thank you.", 0, 1), make_segment("Next slide please.", 1, 2), make_segment("Thank you.", 2, 3)]
    annotate_segments(stock, ucfg)
    assert not any("duplicate_nearby" in codes(s) for s in stock)


def test_hotword_echo_detected(ucfg):
    hot = ["corticospinal tract", "spinothalamic tract", "medial lemniscus", "substantia nigra"]
    echo = make_segment("spinothalamic tract, medial lemniscus, substantia nigra")
    annotate_segments([echo], ucfg, hotwords=hot)
    assert "hotword_echo" in codes(echo)
    real = make_segment("The medial lemniscus carries touch.")
    annotate_segments([real], ucfg, hotwords=hot)
    assert "hotword_echo" not in codes(real)


def test_near_miss_term_is_term_scoped(ucfg, vocab):
    seg = make_segment("The spinothalmic tract carries pain.")
    annotate_segments([seg], ucfg, vocab)
    flag = next(f for f in seg.flags if f.code == "near_miss_term")
    assert flag.scope == "term" and flag.token == "spinothalmic" and flag.suggestion == "spinothalamic"


def test_confidence_heuristic(ucfg):
    good = make_segment("x y z", avg_logprob=-0.1, no_speech_prob=0.0)
    assert compute_confidence(good, ucfg) == pytest.approx(0.9048, abs=1e-3)
    silent = make_segment("x y z", avg_logprob=-0.1, no_speech_prob=0.5)
    assert compute_confidence(silent, ucfg) == pytest.approx(0.4524, abs=1e-3)
    loopy = make_segment("x y z", avg_logprob=-0.1, no_speech_prob=0.0, compression_ratio=3.0)
    assert compute_confidence(loopy, ucfg) == pytest.approx(0.4524, abs=1e-3)


def test_confidence_is_none_without_decode_statistics(ucfg):
    seg = make_segment("text only", avg_logprob=None, no_speech_prob=None, compression_ratio=None)
    annotate_segments([seg], ucfg)
    assert seg.confidence is None
    assert not seg.flags or all(f.code in ("too_short",) for f in seg.flags)
