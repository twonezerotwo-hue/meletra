"""v4: targeted re-decode - the audio confirms or rejects MEDIUM proposals, within a budget, with an injected decoder (no model needed)."""

import pytest

from src.medical.corrector import SegmentView
from src.medical_knowledge.runtime.engine import V4Correction, V4Result
from src.medical_knowledge.runtime.redecode import (CONFIRMED, REJECTED, UNDECIDED, RedecodeBudget, TargetedRedecoder, judge, window_for)


def correction(original="tanisites", replacement="tanycytes", segment=0, start=30, end=39, confidence=0.7):
    return V4Correction(segment, 10.0, original, replacement, confidence, "medium", "c1", "neuroanatomy", start, end, {}, [], [], [], "")


def result(*items):
    r = V4Result.__new__(V4Result)
    r.corrections, r.uncertain = [], list(items)
    return r


VIEWS = [SegmentView(0, 10.0, 20.0, "Next we look at the cells called tanisites in the third ventricle floor.", "en", []),
         SegmentView(1, 20.0, 30.0, "And then the other cells and so on and so forth for a long time.", "en", [])]


def test_window_is_around_the_span_and_never_longer_than_the_limit():
    w = window_for(correction(start=30, end=39), VIEWS[0], RedecodeBudget(pad=1.0, max_window=8.0))
    assert 10.0 <= w.start < w.end <= 21.5 and w.seconds <= 8.0 and w.hotwords == ["tanycytes"]
    long = SegmentView(0, 0.0, 100.0, "x" * 100, "en", [])
    assert window_for(correction(start=0, end=100), long, RedecodeBudget(max_window=6.0)).seconds <= 6.0


@pytest.mark.parametrize("decoded, verdict", [
    ("the cells called tanycytes in the third ventricle", CONFIRMED),
    ("the cells called Tanny cytes in the third", CONFIRMED),                      # sounds the same
    ("the cells called tanisites in the third ventricle", REJECTED),               # the audio says what was heard before
    ("the sells called tonic acid in the", UNDECIDED),
    ("", UNDECIDED),
])
def test_judging_what_the_second_listening_produced(decoded, verdict):
    assert judge(correction(), decoded)[0] == verdict


def test_the_redecoder_promotes_only_confirmed_proposals_and_reports_everything():
    answers = {"tanycytes": "the cells called tanycytes in the", "nucleus": "the cells called nuclear in the"}
    calls = []

    def decode(start, end, hotwords):
        calls.append((start, end, tuple(hotwords)))
        return answers.get(hotwords[0], "something else entirely")

    items = [correction(), correction("nuclear", "nucleus", 0, 40, 47, 0.65), correction("axon hill", "axon hillock", 1, 4, 13, 0.6)]
    report = TargetedRedecoder(decode).run(result(*items), VIEWS)
    assert len(calls) == 3 and report.count(CONFIRMED) == 1 and report.count(UNDECIDED) >= 1
    assert [c.replacement for c in report.promoted()] == ["tanycytes"]
    data = report.to_dict()
    assert data["windows"] == 3 and data["audio_seconds"] > 0 and all("verdict" in o for o in data["outcomes"])


def test_the_budget_limits_windows_and_audio_seconds():
    items = [correction(replacement=f"term{i}", confidence=0.7 - i * 0.01) for i in range(10)]
    report = TargetedRedecoder(lambda s, e, h: "", RedecodeBudget(max_windows=3)).run(result(*items), VIEWS)
    assert len(report.outcomes) == 3 and report.skipped_by_budget == 7
    report = TargetedRedecoder(lambda s, e, h: "", RedecodeBudget(max_seconds=5.0)).run(result(*items), VIEWS)
    assert report.seconds_used <= 5.0 and report.skipped_by_budget > 0


def test_a_failing_decoder_leaves_the_proposal_undecided():
    def broken(start, end, hotwords):
        raise RuntimeError("GPU busy")

    report = TargetedRedecoder(broken).run(result(correction()), VIEWS)
    assert report.count(UNDECIDED) == 1 and "decoder failed" in report.outcomes[0].why and report.promoted() == []
