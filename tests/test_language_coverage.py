"""Speech coverage measured on the audio: loud stretches without words are found, decoded again within a budget, and the rest is reported (no invented text)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.multilingual import coverage as cv
from src.multilingual import pipeline_hook as hook
from src.multilingual.plan import Activity
from src.multilingual.timeline import LanguagePlan, LanguageRun
from src.schema import Segment


def seg(start, end, text="word " * 5, language="en"):
    return Segment(start, end, text, language, "m", "fake")


def loud(seconds_by_state):
    """An Activity from ``[(seconds, rms), ...]``."""
    rms = []
    for seconds, level in seconds_by_state:
        rms.extend([level] * int(seconds))
    return Activity(frame=1.0, rms=rms)


def test_the_threshold_follows_the_loudness_of_the_words_that_exist():
    activity = loud([(30, 0.10), (30, 0.01)])
    transcribed = [seg(0, 30)]
    assert cv.audible_threshold(activity, transcribed) == pytest.approx(0.04)
    quiet = loud([(60, 0.001)])
    assert cv.audible_threshold(quiet, [seg(0, 60)]) == cv.LEVEL_FLOOR                                          # room noise is never "speech that is missing"


def test_a_loud_stretch_without_words_is_a_gap_and_a_pause_is_not():
    activity = loud([(20, 0.1), (3, 0.001), (20, 0.1), (60, 0.1), (20, 0.1)])
    segments = [seg(0, 20), seg(23, 43)]                                                                       # 20 s transcribed, 3 s quiet pause, 20 s transcribed, then 80 s loud and unwritten
    gaps = cv.find_gaps(activity, segments)
    assert len(gaps) == 1 and gaps[0].start == pytest.approx(43.5, abs=1.5) and gaps[0].end == pytest.approx(123.0, abs=1.5)


def test_numbers_are_measured_on_the_audio():
    activity = loud([(40, 0.1), (60, 0.1)])
    numbers = cv.coverage_numbers(activity, [seg(0, 40)])
    assert numbers["audible_seconds"] == 100.0 and 0.38 <= numbers["share"] <= 0.45


def test_recovery_decodes_the_longest_gap_first_within_the_budget_and_marks_the_result():
    gaps = [cv.Gap(100, 110), cv.Gap(0, 60), cv.Gap(200, 230)]
    calls = []

    def decode(start, end, *_):
        calls.append((start, end))
        return [seg(start, end, "recovered text here now", "ko")]

    new = cv.recover_gaps(gaps, decode, budget=95.0, piece=60.0)
    assert calls[0] == (0, 60) and (200, 230) in calls                                                          # longest first
    assert [g.status for g in gaps] == ["skipped", "recovered", "recovered"] and gaps[0].reason                  # the 10 s gap did not fit the budget: reported as skipped, not forgotten
    assert all(s.language_meta["status"] == "repaired" for s in new)


def test_a_gap_that_yields_nothing_is_reported_unreadable_not_filled_in():
    gaps = [cv.Gap(0, 30)]
    new = cv.recover_gaps(gaps, lambda a, b, *_: [], budget=100.0)
    assert new == [] and gaps[0].status == "unreadable" and "no speech" in gaps[0].reason


def test_a_failing_decode_never_fails_the_recording():
    gaps = [cv.Gap(0, 30)]

    def broken(a, b, *_):
        raise RuntimeError("out of memory")

    assert cv.recover_gaps(gaps, broken, budget=100.0) == [] and gaps[0].status == "unreadable" and "failed" in gaps[0].reason


def test_long_gaps_are_decoded_in_pieces():
    gaps = [cv.Gap(0, 200)]
    calls = []
    cv.recover_gaps(gaps, lambda a, b, *_: calls.append((a, b)) or [], budget=1000.0, piece=90.0)
    assert calls == [(0, 90), (90, 180), (180, 200)]


class FakeTranscriber:
    adapter = SimpleNamespace(device="cpu")

    def __init__(self, texts):
        self.texts, self.requests = texts, []

    def transcribe(self, request):
        self.requests.append(request)
        text = self.texts.pop(0) if self.texts else ""
        from src.schema import TranscriptionResult

        segments = [seg(request.start_offset, request.start_offset + request.max_seconds, text, request.language)] if text else []
        return TranscriptionResult(segments, request.language, None, 0.0, "m", "fake", "int8")


CFG = {"decoding": {}, "vad": {}, "multilingual": {"coverage_share": 1.0}}


def test_the_pipeline_step_decodes_gaps_without_the_filter_in_the_language_of_the_surrounding_run():
    activity = loud([(30, 0.1), (60, 0.1)])
    segments = [seg(0, 30, "hello everybody today we", "en")]
    plan = LanguagePlan(runs=[LanguageRun(0, 40, "en", 0.9, "confident"), LanguageRun(40, 90, "ko", 0.9, "confident")], languages=["en", "ko"], mode="multilingual")
    transcriber = FakeTranscriber(["안녕하세요 여러분 오늘은 신경해부학에 대해 이야기해 봅시다"])
    merged, record = hook.recover_coverage(transcriber=transcriber, wav="x.wav", segments=segments, activity=activity, plan=plan, language=None, profile={"beam_size": 1},
                                           cfg=CFG, duration=90.0)
    [request] = transcriber.requests
    assert request.language == "ko" and request.vad_override is False and request.hotwords == []
    assert len(merged) == 2 and merged[1].language == "ko" and merged[1].language_meta["status"] == "repaired"
    assert record["gaps_recovered"] == 1 and record["covered_share_after"] > record["covered_share_before"] and record["checked"] is True


def test_music_tags_and_stock_phrases_are_not_added_as_speech():
    activity = loud([(10, 0.1), (60, 0.1)])
    transcriber = FakeTranscriber(["[Music]"])
    merged, record = hook.recover_coverage(transcriber=transcriber, wav="x.wav", segments=[seg(0, 10, "hello there all of you", "en")], activity=activity, plan=None, language="en",
                                           profile={}, cfg=CFG, duration=70.0)
    assert len(merged) == 1 and record["gaps_unreadable"] == 1 and record["unrecovered"][0]["status"] == "unreadable"


def test_a_line_the_transcript_already_holds_a_chorus_of_times_is_not_brought_back_by_a_re_read():
    line = "उनको लाने उरगुम इरविल"
    existing = [seg(i * 10.0, i * 10.0 + 8.0, line, "hi") for i in range(4)]                   # the main decode kept the line as often as a chorus repeats it
    activity = loud([(40, 0.1), (30, 0.1)])
    transcriber = FakeTranscriber([line])                                                        # the gap decodes to the very same line
    merged, record = hook.recover_coverage(transcriber=transcriber, wav="x.wav", segments=existing, activity=activity, plan=None, language="hi", profile={}, cfg=CFG, duration=70.0)
    assert len(merged) == 4 and record["gaps_recovered"] == 0 and record["repeated_lines_refused"] >= 1
    assert record["unrecovered"][0]["status"] == "unreadable"


def test_a_line_that_is_new_or_short_is_still_recovered_after_a_chorus():
    line = "उनको लाने उरगुम इरविल"
    existing = [seg(i * 10.0, i * 10.0 + 8.0, line, "hi") for i in range(4)]
    activity = loud([(40, 0.1), (30, 0.1)])
    merged, record = hook.recover_coverage(transcriber=FakeTranscriber(["तुम्हें देखा तो मैंने दिल में कहीं"]), wav="x.wav", segments=existing, activity=activity, plan=None,
                                           language="hi", profile={}, cfg=CFG, duration=70.0)
    assert len(merged) == 5 and record["gaps_recovered"] == 1 and "repeated_lines_refused" not in record
    merged, record = hook.recover_coverage(transcriber=FakeTranscriber(["ठीक है"]), wav="x.wav", segments=[seg(0, 8, "ठीक है", "hi") for _ in range(5)] + [seg(10, 40, "word " * 5)],
                                           activity=loud([(40, 0.1), (30, 0.1)]), plan=None, language="hi", profile={}, cfg=CFG, duration=70.0)
    assert record["gaps_recovered"] == 1                                                            # two words: not a line that can be a stuck decoder


def test_the_step_can_be_switched_off_and_needs_loudness():
    segments = [seg(0, 10)]
    off_cfg = {"decoding": {}, "vad": {}, "multilingual": {"coverage": "off"}}
    assert hook.recover_coverage(transcriber=None, wav="x", segments=segments, activity=loud([(30, 0.1)]), plan=None, language=None, profile={}, cfg=off_cfg, duration=30.0)[1]["checked"] is False
    assert hook.recover_coverage(transcriber=None, wav="x", segments=segments, activity=None, plan=None, language=None, profile={}, cfg=CFG, duration=30.0)[1]["checked"] is False


def test_a_recording_without_gaps_is_left_alone():
    activity = loud([(30, 0.1)])
    segments = [seg(0, 30)]
    merged, record = hook.recover_coverage(transcriber=None, wav="x", segments=segments, activity=activity, plan=None, language=None, profile={}, cfg=CFG, duration=30.0)
    assert merged == segments and record["gaps_found"] == 0 and record["checked"] is True


def test_a_much_quieter_stretch_is_looked_for_too_but_steady_room_noise_is_not():
    activity = loud([(20, 0.1), (3, 0.0005), (14, 0.002), (3, 0.0005), (10, 0.1), (20, 0.0003)])           # words, a pause, a speaker far from the microphone for 14 s, a pause, words, quiet room
    segments = [seg(0, 20), seg(40, 50)]
    gaps = cv.find_gaps(activity, segments)
    assert [(g.kind, round(g.start), round(g.end)) for g in gaps] == [("quiet", 23, 37)]
    noisy = loud([(20, 0.1), (60, 0.0004)])
    assert cv.find_gaps(noisy, [seg(0, 20)]) == []                                                                 # below the floor: room noise, not a speaker
    short = loud([(20, 0.1), (4, 0.002), (4, 0.1)])
    assert cv.find_gaps(short, [seg(0, 20), seg(24, 28)]) == []                                                    # a 4 s murmur is not worth a decode


def test_a_quiet_gap_is_decoded_with_the_volume_raised_and_the_times_stay_absolute(tmp_path):
    import numpy as np
    import soundfile as sf

    rate = 16000
    t = np.arange(rate * 40) / rate
    quiet = np.where((t >= 10) & (t < 30), 0.003 * np.sin(2 * np.pi * 300 * t), 0.0).astype("float32")
    path = tmp_path / "quiet.wav"
    sf.write(str(path), quiet, rate, subtype="PCM_16")
    boosted = hook.boosted_slice(path, 10.0, 30.0)
    data, _rate = sf.read(str(boosted), dtype="float32")
    assert len(data) == 20 * rate and 0.04 < float(np.abs(data).max()) < 1.0                                       # about 100 times louder, limited
    boosted.unlink()

    activity = loud([(10, 0.1), (20, 0.002), (10, 0.1)])
    transcriber = FakeTranscriber(["안녕하세요 여러분 오늘은 신경해부학에 대해 이야기해 봅시다"])
    merged, record = hook.recover_coverage(transcriber=transcriber, wav=path, segments=[seg(0, 10), seg(30, 40)], activity=activity, plan=None, language="ko", profile={}, cfg=CFG,
                                           duration=40.0)
    [request] = transcriber.requests
    assert request.start_offset == 0.0 and request.max_seconds == pytest.approx(19.0, abs=2.0) and str(request.audio_path).endswith(".wav") and request.audio_path != path
    recovered = [s for s in merged if s.language_meta.get("status") == "repaired"]
    assert recovered and recovered[0].start == pytest.approx(10.0, abs=1.6) and recovered[0].end <= 31.0               # absolute times, not relative to the slice
    assert record["quiet_gaps_found"] == 1 and record["gaps_recovered"] == 1
