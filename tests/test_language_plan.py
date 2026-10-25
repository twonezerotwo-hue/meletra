"""Planning languages: when a dense scan is worth it, music suspicion, bounded verification of unsure runs (fake model functions, no audio)."""

from __future__ import annotations

import wave

import numpy as np
import pytest

from src.multilingual import plan as pl
from src.multilingual import timeline as tl
from src.multilingual.timeline import LanguagePlan, LanguageRun


def write_wav(path, seconds, amplitude_fn):
    rate = 16000
    t = np.arange(int(seconds * rate)) / rate
    data = (np.clip(amplitude_fn(t), -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(data.tobytes())


def test_activity_profile_measures_loudness_per_second(tmp_path):
    path = tmp_path / "a.wav"
    write_wav(path, 10, lambda t: np.where(t < 5, 0.0, 0.3 * np.sin(2 * np.pi * 220 * t)))                      # 5 s of silence, 5 s of a loud tone
    activity = pl.activity_profile(path)
    assert len(activity.rms) == 10 and activity.rms[0] < 0.001 and activity.rms[9] > 0.1
    assert 0.45 <= activity.loud_share <= 0.55
    assert activity.loud_regions(min_seconds=3.0) == [(5.0, 10.0)] and activity.rms_between(0, 4) < 0.001


def test_music_is_suspected_when_loud_time_is_not_speech_to_the_filter():
    loud = pl.Activity(rms=[0.2] * 600)
    assert pl.music_suspected(loud, [(10.0, 30.0)], 600.0)                                                      # loud all the time, 3 % speech: singing / music
    assert not pl.music_suspected(loud, [(0.0, 560.0)], 600.0)                                                  # loud and all speech: a lecture
    quiet_lecture = pl.Activity(rms=[0.001] * 300 + [0.08] * 300)
    assert not pl.music_suspected(quiet_lecture, [(300.0, 600.0)], 600.0)
    assert not pl.music_suspected(None, [], 600.0)


def test_escalation_reasons():
    en = [(i * 30.0, "en", 0.95) for i in range(24)]
    quiet = pl.Activity(rms=[0.05] * 720)
    assert pl.escalation_reason(en, activity=quiet, speech=[(0, 700)], duration=720) is None                  # one clear language: the cheap sampling is enough
    assert pl.escalation_reason(en, activity=quiet, speech=[(0, 700)], duration=720, setting="off") is None
    assert "requested" in pl.escalation_reason(en, activity=quiet, speech=[(0, 700)], duration=720, setting="always")
    two = en[:20] + [(600.0 + i * 30.0, "ko", 0.93) for i in range(4)]
    assert "en" in pl.escalation_reason(two, activity=quiet, speech=[(0, 700)], duration=720) and "ko" in pl.escalation_reason(two, activity=quiet, speech=[(0, 700)], duration=720)
    brief = en[:23] + [(690.0, "hi", 0.96)]
    assert "hi" in pl.escalation_reason(brief, activity=quiet, speech=[(0, 700)], duration=720)               # one sure window in another language is enough to look closer
    accented = [(i * 30.0, "sl" if i % 2 else "hr", 0.35) for i in range(24)]
    assert "weak" in pl.escalation_reason(accented, activity=quiet, speech=[(0, 700)], duration=720)
    assert "singing" in pl.escalation_reason(en, activity=pl.Activity(rms=[0.2] * 720), speech=[(0, 20)], duration=720)


def make_plan(statuses):
    runs = [LanguageRun(i * 60.0, (i + 1) * 60.0, lang, None if status == "unsure" else 0.9, status, candidates=["ta"], windows=2) for i, (lang, status) in enumerate(statuses)]
    return LanguagePlan(runs=runs, languages=[l for l, _s in statuses], mode="music")


class FakeScorer:
    TEXT = {"ko": "안녕하세요 여러분 오늘은 신경해부학에 대해 이야기해 봅시다 감사합니다", "hi": "नमस्ते दोस्तों आज हम तंत्रिका तंत्र के बारे में बात करेंगे धन्यवाद",
            "en": "hello everybody today we will talk about the nervous system thank you"}

    def __init__(self, truth_by_start):
        self.calls = []
        self.truth = truth_by_start

    def __call__(self, start, seconds, languages):
        self.calls.append((round(start), tuple(languages)))
        truth = next(lang for (a, b), lang in self.truth.items() if a <= start < b)
        out = {}
        for language in languages:
            good = language == truth
            out[language] = {"avg_logprob": -0.3 if good else -1.1, "no_speech_prob": 0.1, "text": self.TEXT.get(language, "x") if good else "x"}
        return out


def test_verification_replaces_an_unsure_guess_by_the_language_with_the_best_likelihood():
    plan = make_plan([("ta", "unsure"), ("en", "confident"), ("ml", "unsure")])
    scorer = FakeScorer({(0, 60): "hi", (120, 180): "ko"})
    report = pl.verify_runs(plan, scorer, duration=180.0)
    assert [r.language for r in plan.runs] == ["hi", "en", "ko"]
    assert [r.status for r in plan.runs] == ["verified", "confident", "verified"] and plan.runs[0].source == "verified" and plan.runs[1].source == "detector"
    assert all(r.confidence is None for r in (plan.runs[0], plan.runs[2]))                                         # a likelihood comparison is not a probability: none is claimed
    assert report["samples"] >= 2 and all(set(c[1]) <= set(tl.SUPPORTED_SPOKEN) for c in scorer.calls)             # only languages the program offers are compared
    assert set(plan.languages) == {"hi", "en", "ko"} and plan.mode == "multilingual"


def test_verification_is_bounded_and_never_touches_confident_runs():
    plan = make_plan([("ta", "unsure")] * 1)
    plan.runs[0].end = 3600.0
    scorer = FakeScorer({(0, 3600): "hi"})
    pl.verify_runs(plan, scorer, duration=3600.0, max_samples=4)
    assert len(scorer.calls) == 4                                                                                   # an hour of unsure audio costs four windows, not sixty
    confident = make_plan([("en", "confident")])
    scorer2 = FakeScorer({(0, 60): "en"})
    pl.verify_runs(confident, scorer2, duration=60.0)
    assert scorer2.calls == [] and confident.runs[0].language == "en"


def test_a_tie_keeps_the_run_unsure_and_lists_alternatives():
    plan = make_plan([("ta", "unsure")])

    def flat(start, seconds, languages):
        return {l: {"avg_logprob": -0.7, "no_speech_prob": 0.1, "text": FakeScorer.TEXT.get(l, FakeScorer.TEXT["en"])} for l in languages}

    pl.verify_runs(plan, flat, duration=60.0)
    assert plan.runs[0].status == "unsure" and plan.runs[0].candidates                                              # the uncertainty is kept, not hidden


def test_a_failing_window_leaves_the_run_as_it_was():
    plan = make_plan([("ta", "unsure")])

    def broken(start, seconds, languages):
        raise RuntimeError("out of memory")

    report = pl.verify_runs(plan, broken, duration=60.0)
    assert plan.runs[0].language == "ta" and plan.runs[0].status == "unsure" and report["samples"] == 0


def test_stopping_is_honoured_between_windows():
    plan = make_plan([("ta", "unsure")])
    scorer = FakeScorer({(0, 60): "hi"})
    pl.verify_runs(plan, scorer, duration=60.0, should_stop=lambda: True)
    assert scorer.calls == []


def test_the_language_score_is_the_models_own_log_probability_gated_against_outputs_that_are_not_speech():
    korean = "안녕하세요 여러분 오늘은 신경해부학에 대해 이야기해 봅시다 감사합니다"
    good = pl.language_score("ko", {"avg_logprob": -0.4, "text": korean})
    better = pl.language_score("ko", {"avg_logprob": -0.1, "text": korean})
    worse = pl.language_score("ko", {"avg_logprob": -1.2, "text": korean})
    assert better > good > worse                                                                              # the true language explains the audio best (measured on 19 known windows)
    wrong_script = pl.language_score("ko", {"avg_logprob": -0.4, "text": "hello everybody today we will talk about the nervous system thank you"})
    tiny = pl.language_score("ko", {"avg_logprob": -0.1, "text": "네"})
    stock = pl.language_score("tr", {"avg_logprob": -0.1, "text": "Altyazı M.K."})
    loop = pl.language_score("en", {"avg_logprob": -0.5, "text": "la la la la la la la la la la la la la la"})
    assert good > wrong_script + 1.5 and good > tiny + 0.5 and good > stock + 0.5 and good > loop + 0.5                 # a phrase on music, a few letters, a loop, the wrong writing system never win
    assert pl.language_score("en", {"text": korean}) <= -3.0 + 0.0 or pl.language_score("en", {"text": korean}) < -3.0     # no log-probability at all is the worst case, not a free pass


def test_the_candidate_pool_always_tries_one_language_per_writing_system():
    run = LanguageRun(0, 60, "ta", None, "unsure", candidates=["ml", "en"])
    pool = pl.candidate_pool(run, [])
    assert pool[0] == "en" and {"ko", "hi"} <= set(pool) and len(pool) <= pl.MAX_CANDIDATES and set(pool) <= set(tl.SUPPORTED_SPOKEN)


def test_scan_windows_joins_detector_output_with_speech_and_loudness():
    def detect(starts, size, progress):
        return [(s, {"ko": 0.9}) for s in starts]

    activity = pl.Activity(rms=[0.1] * 100)
    windows = pl.scan_windows(detect, [0.0, 30.0], 100.0, [(0.0, 20.0)], activity)
    assert [round(w.speech_seconds) for w in windows] == [20, 0] and windows[0].rms == pytest.approx(0.1) and windows[1].end == 60.0


def test_an_unsure_stretch_is_split_where_the_verified_language_changes():
    plan = LanguagePlan(runs=[LanguageRun(0.0, 300.0, "ta", None, "unsure", candidates=["en"], windows=15)], languages=["ta"], mode="music")
    scorer = FakeScorer({(0, 140): "ko", (140, 300): "hi"})
    report = pl.verify_runs(plan, scorer, duration=300.0, max_samples=10)
    assert [r.language for r in plan.runs] == ["ko", "hi"] and plan.mode == "multilingual" and set(plan.languages) == {"ko", "hi"}
    assert 100 <= plan.runs[0].end <= 180 and plan.runs[0].end == plan.runs[1].start                                   # the change is placed between the windows that disagree
    assert report["samples"] == 10 and len(report["runs"][0]["pieces"]) == 2


def test_a_lone_window_does_not_cut_a_passage_in_two():
    rows = [{"ko": 0.5, "hi": 0.1}] * 4 + [{"ko": 0.2, "hi": 0.28}] + [{"ko": 0.5, "hi": 0.1}] * 4
    assert set(pl.smooth_scores(rows, ["ko", "hi"])) == {"ko"}
    rows = [{"ko": 0.5, "hi": 0.1}] * 4 + [{"ko": 0.1, "hi": 0.6}] * 4
    assert pl.smooth_scores(rows, ["ko", "hi"]) == ["ko"] * 4 + ["hi"] * 4
