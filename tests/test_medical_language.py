"""Language stability: the soft lock decides the primary language, real multilingual lectures stay multilingual, drift is repaired conservatively."""

from src.medical import language as lm
from tests.medical_support import seg

EN = "The neural tube closes and the neural crest cells migrate to form the peripheral nervous system."
SL = "Zdravstveno v tem semestru mikroskopična anatomija bo o neuroanatomiji."


def summary(*pairs):
    return {"samples": [[i * 100.0, lang, p] for i, (lang, p) in enumerate(pairs)]}


# ----------------------------------------------------------------------------------------------------- what is English text
def test_english_text_is_recognised_with_medical_terms_and_latin():
    assert lm.language_fit(EN, "en") > 0.8
    assert lm.language_fit("foramina nuclei ganglia septa", "en", lambda w: w in {"foramina", "nuclei", "ganglia", "septa"}) == 1.0
    assert lm.language_fit(SL, "en") < 0.3 and lm.language_fit("Beszélünk a türelmi célokról és a kommunikációról.", "en") < 0.3


def test_english_words_forced_into_another_language_do_not_fit_that_language():
    assert lm.language_fit(EN, "sl") < 0.5 < lm.language_fit(EN, "en")


# ----------------------------------------------------------------------------------------------------- soft lock decisions
def test_accented_english_with_weak_detections_is_not_mixed():
    s = summary(("sl", 0.55), ("hr", 0.5), ("en", 0.52), ("sr", 0.45), ("sl", 0.61), ("bs", 0.4), ("en", 0.7), ("hu", 0.38), ("sl", 0.66), ("en", 0.5))

    def decode(language, start):
        return (EN, 0.9) if language == "en" else (EN, 0.6)                       # English words come out whatever the forced language
    v = lm.stabilize(s, decode)
    assert v.primary == "en" and not v.multilingual and v.arbitrated and v.confidence > 0.3
    assert "arbitration" in v.reason and v.windows


def test_a_language_with_several_strong_windows_is_locked_without_decoding():
    calls = []
    v = lm.stabilize(summary(*[("en", 0.97)] * 14, ("hu", 0.55), ("sl", 0.4)), lambda l, s: calls.append(l) or ("", None))
    assert v.primary == "en" and not v.multilingual and calls == [] and not v.arbitrated


def test_a_really_multilingual_lecture_stays_multilingual():
    v = lm.stabilize(summary(*[("en", 0.97)] * 14, *[("hu", 0.93)] * 6))
    assert v.multilingual and v.primary == "en" and set(v.strong) == {"en", "hu"}


def test_one_strong_odd_window_does_not_switch_the_language():
    v = lm.stabilize(summary(*[("en", 0.97)] * 20, ("tr", 0.95)))
    assert v.primary == "en" and not v.multilingual


def test_a_recording_that_opens_in_another_language_is_a_real_switch():
    v = lm.stabilize(summary(*[("en", 0.97)] * 20), probe=("hu", 0.92))
    assert v.multilingual and "first minutes are hu" in v.reason
    assert not lm.stabilize(summary(*[("en", 0.97)] * 20), probe=("hu", 0.55)).multilingual


def test_a_turkish_lecture_is_locked_to_turkish():
    tr = "Bu derste sinir sistemi ve nöronların yapısı hakkında konuşacağız, çünkü bu çok önemli bir konu."

    def decode(language, start):
        return (tr, 0.9) if language == "tr" else (tr, 0.4)
    v = lm.stabilize(summary(("tr", 0.6), ("az", 0.5), ("tr", 0.7), ("en", 0.4)), decode)
    assert v.primary == "tr" and not v.multilingual


def test_without_a_decoder_the_detector_majority_decides_and_nothing_fails():
    v = lm.stabilize(summary(("sl", 0.6), ("sl", 0.5), ("en", 0.4)))
    assert v.primary == "sl" and not v.arbitrated
    assert lm.stabilize({"samples": []}).primary is None


def test_a_failing_decoder_never_breaks_the_decision():
    def boom(language, start):
        raise RuntimeError("out of memory")
    v = lm.stabilize(summary(("sl", 0.6), ("en", 0.5)), boom)
    assert v.primary in ("sl", "en") and not v.multilingual


def test_cpu_arbitration_decodes_less():
    calls = []

    def decode(language, start):
        calls.append((language, start))
        return EN, 0.9
    lm.stabilize(summary(*[("sl", 0.5)] * 6), decode, on_cpu=True)
    cpu = len(calls)
    calls.clear()
    lm.stabilize(summary(*[("sl", 0.5)] * 6), decode, on_cpu=False)
    assert 0 < cpu < len(calls)


def test_the_verdict_is_written_into_the_sampling_summary():
    s = summary(*[("en", 0.97)] * 10)
    s.update(mixed=True, dominant="sl", reason="old")
    lm.apply_to_summary(s, lm.stabilize(s))
    assert s["dominant"] == "en" and s["mixed"] is False and s["soft_lock"]["primary"] == "en"


# ----------------------------------------------------------------------------------------------------- drift repair
def test_an_isolated_drifted_segment_is_redecoded_in_the_primary_language():
    segs = [seg(EN, 0, 5), seg(SL, 5, 10), seg(EN, 10, 15)]
    seen = []

    def redecode(start, end, language):
        seen.append((round(start, 1), round(end, 1), language))
        return "The microscopic anatomy of the nervous system is the subject of this semester.", 0.9
    texts, repairs, drifted = lm.repair_drift(segs, "en", redecode)
    assert drifted == [1] and texts[0] == EN and texts[2] == EN and texts[1].startswith("The microscopic anatomy")
    assert seen == [(4.7, 10.3, "en")] and repairs[0].redecoded and repairs[0].replacement == texts[1]
    assert repairs[0].to_dict()["original"] == SL                                  # the original stays in the report


def test_a_redecode_that_is_not_clearly_better_is_not_used():
    segs = [seg(EN, 0, 5), seg(SL, 5, 10), seg(EN, 10, 15)]
    texts, repairs, _ = lm.repair_drift(segs, "en", lambda s, e, l: ("Dobro jutro in dobrodošli.", 0.8))
    assert texts[1] == SL and repairs[0].replacement is None and "not clearly better" in repairs[0].reason


def test_a_long_run_in_another_language_is_a_real_passage_and_left_alone():
    segs = [seg(EN, 0, 5)] + [seg(SL, 5 + i * 5, 10 + i * 5) for i in range(6)] + [seg(EN, 35, 40)]
    called = []
    texts, repairs, drifted = lm.repair_drift(segs, "en", lambda s, e, l: called.append(1) or (EN, 0.9))
    assert len(drifted) == 6 and called == [] and texts[1] == SL
    assert all("run of 6" in r.reason for r in repairs)


def test_latin_and_medical_terms_are_not_drift():
    segs = [seg("foramina nuclei ganglia plexus", 0, 4)]
    med = lambda w: w in {"foramina", "nuclei", "ganglia", "plexus"}                # noqa: E731
    assert lm.repair_drift(segs, "en", None, med)[2] == []


def test_a_failing_redecode_keeps_the_text():
    segs = [seg(EN, 0, 5), seg(SL, 5, 10), seg(EN, 10, 15)]

    def boom(s, e, l):
        raise RuntimeError("no audio")
    texts, repairs, _ = lm.repair_drift(segs, "en", boom)
    assert texts[1] == SL and "re-decode failed" in repairs[0].reason


def test_no_primary_language_means_no_repair():
    assert lm.repair_drift([seg(SL, 0, 5)], None)[2] == []
