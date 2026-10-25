import numpy as np
import pytest

from src import quality
from src.model_manager import DecodeRequest, FasterWhisperAdapter
from src.quality import DEFAULTS, analyze_quality, find_speech_onset, missing_speech, timestamp_issues, window_stats
from src.schema import UncertaintyFlag, Word
from src.transcriber import ProgressReporter
from tests.conftest import make_segment, write_silent_wav

CFG = dict(DEFAULTS)


def test_probe_starts_at_sustained_speech_not_at_a_blip():
    speech = [(3.0, 3.6), (347.0, 380.0), (381.0, 440.0)]      # a click at 3 s, the lecture really starts at 5:47
    assert find_speech_onset(speech) == pytest.approx(346.0)
    assert find_speech_onset([]) == 0.0
    assert find_speech_onset([(1.0, 3.0)]) == 0.0              # never enough speech: fall back to the start


def test_timestamp_anomalies_are_all_detected():
    segs = [make_segment("ok sentence here", 0, 10),
            make_segment("overlapping with previous one", 9.0, 15),
            make_segment("goes backwards in time again", 5.0, 8),
            make_segment("collapsed to zero length", 20.0, 20.01),
            make_segment("reversed times", 30.0, 25.0),
            make_segment("past the end of the file", 95.0, 120.0),
            make_segment("way too long for whisper", 130.0, 175.0)]
    codes = {i["code"] for i in timestamp_issues(segs, 100.0, CFG)}
    assert {"overlap_with_previous", "non_monotonic", "zero_length", "reversed_or_negative", "beyond_audio_end",
            "overlong_segment"} <= codes


def test_clean_timestamps_produce_no_issues():
    segs = [make_segment("a b c", i * 5.0, i * 5.0 + 4.9) for i in range(10)]
    assert timestamp_issues(segs, 60.0, CFG) == []


def test_word_outside_its_segment_is_flagged():
    seg = make_segment("hello world", 10, 12)
    seg.words = [Word(10, 10.5, "hello", 0.9), Word(40, 41, "world", 0.9)]
    assert any(i["code"] == "word_outside_segment" for i in timestamp_issues([seg], 100.0, CFG))


def test_hallucination_stock_phrase_and_text_in_silence():
    speech = [(0.0, 10.0), (50.0, 60.0)]
    segs = [make_segment("Thanks for watching, and please subscribe!", 12, 15),
            make_segment("This is real lecture speech about the thalamus.", 0, 8),
            make_segment("Some invented words in a dead stretch of audio.", 20, 28)]
    flags = {(f["code"], f["segment"]) for f in quality.hallucination_flags(segs, speech, CFG)}
    assert ("stock_phrase", 0) in flags and ("text_in_non_speech", 2) in flags
    assert not any(seg_index == 1 for _, seg_index in flags)


def test_turkish_stock_phrase_is_known():
    flags = quality.hallucination_flags([make_segment("Abone ol abone ol", 0, 3)], [(0, 3)], CFG)
    assert [f["code"] for f in flags] == ["stock_phrase"]


def test_repetition_flags_come_from_the_uncertainty_codes():
    seg = make_segment("the same sentence again and again", 0, 5)
    seg.flags = [UncertaintyFlag("duplicate_nearby", "x"), UncertaintyFlag("low_avg_logprob", "y")]
    assert [f["code"] for f in quality.repetition_flags([seg])] == ["duplicate_nearby"]


def test_missing_speech_reports_uncovered_gaps_and_coverage():
    speech = [(0.0, 100.0)]
    segs = [make_segment("first part", 0, 40), make_segment("last part", 70, 100)]       # 30 s hole
    gaps, coverage = missing_speech(segs, speech, min_gap_s=8.0)
    assert gaps == [{"start": 40.0, "end": 70.0, "seconds": 30.0}]
    assert coverage == pytest.approx(0.70)
    assert missing_speech(segs, speech, min_gap_s=60.0)[0] == []


def test_windows_cover_beginning_middle_and_end():
    segs = [make_segment("word " * 20, t, t + 20, avg_logprob=-0.2) for t in range(0, 3600, 30)]
    stats = window_stats(segs, 3600.0, 10.0, [(0.0, 3600.0)])
    assert set(stats) == {"beginning", "middle", "end"}
    assert stats["beginning"]["from"] == 0 and stats["end"]["to"] == 3600
    assert 1500 <= stats["middle"]["from"] < 1700
    assert stats["middle"]["words_per_minute"] > 0


def test_full_audit_summary_counts():
    segs = [make_segment("good lecture sentence about enzymes", 0, 10), make_segment("Thanks for watching", 12, 14)]
    report = analyze_quality(segs, 20.0, [(0.0, 11.0)])
    s = report["summary"]
    # "Thanks for watching" is both a stock phrase and sits outside the VAD speech region: two signals
    assert s["hallucination_flags"] == 2 and s["segments"] == 2 and 0 < s["speech_coverage"] <= 1


def test_progress_line_has_every_requested_field():
    reporter = ProgressReporter(gpu_stats=lambda: {"dedicated_mib": 4259.0, "shared_mib": 78.0})
    line = reporter.line(1920.0, 5241.0)
    for part in ("PROGRESS", "36.6%", "32:00 / 1:27:21", "elapsed", "ETA", "RTF", "GPU dedicated 4259 MiB", "shared 78 MiB"):
        assert part in line


def test_progress_after_resume_measures_only_this_runs_audio(monkeypatch):
    t = iter([100.0, 110.0, 110.0])
    monkeypatch.setattr("src.transcriber.time.perf_counter", lambda: next(t))
    reporter = ProgressReporter(start_offset=1000.0)         # 1000 s were finished by an earlier run
    line = reporter.line(1100.0, 2000.0)                    # 100 s decoded in 10 s of this run -> RTF 0.1
    assert "RTF 0.100" in line and "ETA 1:30" in line


def test_audio_slicing_for_resume_and_probe(tmp_path):
    wav = write_silent_wav(tmp_path / "a.wav", seconds=30)
    whole, offset = FasterWhisperAdapter._audio_input(DecodeRequest(wav, None, {}, {}, {}))
    assert whole == str(wav) and offset == 0.0                                  # no copy when no slicing is needed
    part, offset = FasterWhisperAdapter._audio_input(DecodeRequest(wav, None, {}, {}, {}, start_offset=10.0, max_seconds=5.0))
    assert offset == 10.0 and len(part) == 5 * 16000
    tail, offset = FasterWhisperAdapter._audio_input(DecodeRequest(wav, None, {}, {}, {}, start_offset=25.0))
    assert offset == 25.0 and len(tail) == 5 * 16000


class _Raw:
    def __init__(self, start, end, text):
        self.start, self.end, self.text = start, end, text
        self.avg_logprob, self.no_speech_prob, self.compression_ratio = -0.2, 0.01, 1.2
        self.words = [Word(start, start + 0.5, "w", 0.9)]


class _Info:
    duration, language = 30.0, "en"


def test_collected_segments_get_absolute_timestamps_and_checkpoint_callback():
    adapter = FasterWhisperAdapter(__import__("src.model_manager", fromlist=["ModelSpec"]).ModelSpec(
        "default", "faster_whisper", "large-v3"), device="cuda")
    seen, progress = [], []
    request = DecodeRequest(None, "en", {}, {}, {}, start_offset=600.0, total_duration=1000.0,
                            on_segment=seen.append, progress=lambda done, total: progress.append((done, total)))
    segs = adapter._collect(iter([_Raw(0.0, 5.0, " first"), _Raw(25.0, 99.0, " beyond")]), _Info(), request, 600.0)
    assert [(s.start, s.end) for s in segs] == [(600.0, 605.0), (625.0, 630.0)]     # offset added, clamped to the slice
    assert segs[0].words[0].start == 600.0
    assert seen == segs and progress[-1] == (630.0, 1000.0)


def test_subtitle_credits_in_korean_chinese_and_hindi_are_flagged_but_a_real_thank_you_is_not():
    credits = [make_segment(text, i * 5, i * 5 + 3) for i, text in enumerate(["한글자막 by 한효정", "中文字幕志愿者 李宗盛", "सब्सक्राइब करें"])]
    flags = quality.hallucination_flags(credits, [(0, 20)], CFG)
    assert sorted((f["code"], f["segment"]) for f in flags) == [("stock_phrase", 0), ("stock_phrase", 1), ("stock_phrase", 2)]
    speech = [make_segment("감사합니다", 0, 3), make_segment("오늘은 신경계의 구조에 대해 이야기하겠습니다.", 3, 8), make_segment("Thank you.", 8, 10)]
    assert quality.hallucination_flags(speech, [(0, 10)], CFG) == []
