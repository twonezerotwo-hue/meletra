"""Language "Auto" looks at the whole recording, mixed-language recordings are decoded window by window, hotwords only when supported.

Found with a real recording: an English lecture that opens with small talk in Hungarian was decoded as Hungarian from start to end
(language taken from the first minutes only): English speech came out as Hungarian word salad with decoder loops."""

import json
import wave

import numpy as np
import pytest

from src import language as lang
from src.model_manager import ASRAdapter, Capabilities, DecodeRequest, ModelSpec, read_wav_slice
from src.pipeline import RunOptions, RunPlan, _gate_hotwords, process_files
from src.schema import Segment, TranscriptionResult
from src.transcriber import Transcriber
from tests.conftest import write_silent_wav


# ----------------------------------------------------------------------------------------------- sampling windows
def test_windows_are_spread_evenly_over_the_speech_and_fit_inside_the_file():
    starts = lang.sampling_starts([(0.0, 600.0)], 600.0, count=10, seconds=30.0)
    assert len(starts) == 10 and starts == sorted(starts) and all(0.0 <= s <= 570.0 for s in starts)
    assert starts[0] < 40 and starts[-1] > 520


def test_silence_is_skipped_when_choosing_windows():
    speech = [(0.0, 100.0), (1000.0, 1100.0)]
    starts = lang.sampling_starts(speech, 1200.0, count=4, seconds=30.0)
    assert all(s < 100 or 1000 <= s <= 1100 for s in starts) and sum(s >= 1000 for s in starts) == 2


def test_without_vad_the_whole_file_is_used():
    starts = lang.sampling_starts([], 3000.0, count=6, seconds=30.0)
    assert len(starts) == 6 and starts[-1] > 2400


def test_degenerate_inputs():
    assert lang.sampling_starts([(0, 10)], 0.0) == [] and lang.sampling_starts([(0, 10)], 100.0, count=0) == []
    assert lang.sampling_starts([(0, 5)], 20.0, count=3, seconds=30.0) == [0.0, 0.0, 0.0]


# ----------------------------------------------------------------------------------------------- one language or mixed?
def samples(*items):
    return [(float(i * 100), code, p) for i, (code, p) in enumerate(items)]


def test_one_language_is_not_mixed():
    out = lang.summarize_language_samples(samples(*[("en", 0.99)] * 24))
    assert out["dominant"] == "en" and not out["mixed"] and out["dominant_share"] == 1.0


def test_one_unsure_window_does_not_make_a_lecture_mixed():
    out = lang.summarize_language_samples(samples(("uk", 0.47), *[("en", 1.0)] * 23))
    assert out["dominant"] == "en" and not out["mixed"]


def test_two_deviating_windows_make_it_mixed():
    out = lang.summarize_language_samples(samples(("hu", 0.6), ("tr", 0.55), *[("en", 1.0)] * 22))
    assert out["mixed"] and "2 of 24" in out["reason"]


def test_a_clearly_different_language_makes_it_mixed():
    out = lang.summarize_language_samples(samples(("hu", 0.95), *[("en", 1.0)] * 23))
    assert out["mixed"] and "hu" in out["reason"]


def test_the_real_case_the_first_minutes_disagree_with_the_rest():
    out = lang.summarize_language_samples(samples(("uk", 0.47), *[("en", 1.0)] * 23), probe=("hu", 0.92))
    assert out["mixed"] and out["dominant"] == "en" and "first minutes are hu" in out["reason"]


def test_an_unsure_probe_is_ignored():
    out = lang.summarize_language_samples(samples(*[("en", 1.0)] * 24), probe=("hu", 0.55))
    assert not out["mixed"]


def test_no_usable_samples():
    out = lang.summarize_language_samples(samples(("en", 0.2), ("tr", 0.1)))
    assert out["dominant"] is None and not out["mixed"] and out["counts"] == {}
    assert lang.summarize_language_samples([])["windows"] == 0


# ----------------------------------------------------------------------------------------------- hotword gate
def plan(**kw) -> RunPlan:
    base = dict(language="en", language_source="user", topic={"confident": True, "selected": ["biochemistry"]})
    base.update(kw)
    return RunPlan(**base, hotword_candidates=["glycolysis", "hexokinase"])


def gate(p, **opts):
    _gate_hotwords(p, RunOptions(**opts))
    return p


def test_hotwords_stay_when_language_and_subject_are_supported():
    p = gate(plan())
    assert p.hotword_candidates and p.hotwords_off_reason == ""


def test_mixed_recordings_get_no_hotwords():
    p = gate(plan(language=None, multilingual=True, language_dominant="en"))
    assert p.hotword_candidates == [] and "mixes languages" in p.hotwords_off_reason


def test_a_language_without_a_vocabulary_gets_no_hotwords():
    p = gate(plan(language="hu"))
    assert p.hotword_candidates == [] and "'hu'" in p.hotwords_off_reason


def test_an_unrecognised_subject_gets_no_hotwords_unless_the_user_or_the_slides_say_otherwise():
    p = gate(plan(topic={"confident": False, "selected": []}))
    assert p.hotword_candidates == [] and "could not be recognised" in p.hotwords_off_reason
    assert gate(plan(topic={"confident": False, "selected": []}), subjects=["neuroscience"]).hotword_candidates
    assert gate(plan(topic={"confident": False, "selected": []}, pdf_terms=["glycolysis"])).hotword_candidates


# ----------------------------------------------------------------------------------------------- the pipeline
class LangFake(ASRAdapter):
    backend_name = "fake"
    capabilities = Capabilities(hotwords=True, vad=True, word_timestamps=True, language_detection=True, multilingual=True,
                                compute_types=True)
    detections: list[tuple[str, float]] = []          # answers of detect_language_at, in order of the calls
    requests: list[DecodeRequest] = []
    calls = 0

    def __init__(self) -> None:
        super().__init__(ModelSpec("default", "faster_whisper", "large-v3"))

    def load(self, compute_type):
        self.compute_type = compute_type

    def detect_language_at(self, wav, start, seconds=30.0):
        answer = LangFake.detections[min(LangFake.calls, len(LangFake.detections) - 1)]
        LangFake.calls += 1
        return answer

    def transcribe(self, request):
        LangFake.requests.append(request)
        total = request.total_duration
        end = min(total, request.start_offset + request.max_seconds) if request.max_seconds else total
        segs, t = [], request.start_offset
        while t < end - 0.5:
            seg = Segment(t, min(t + 10, end), f" glycolysis step {chr(97 + int(t // 10) % 26)} words here", request.language or "en",
                          "large-v3", "fake", avg_logprob=-0.2, no_speech_prob=0.01, compression_ratio=1.1)
            segs.append(seg)
            if request.on_segment:
                request.on_segment(seg)
            t += 10
        return TranscriptionResult(segs, request.language or "en", 0.99, total, "large-v3", "fake", self.compute_type,
                                   hotwords_applied=list(request.hotwords[:5]), hotword_tokens=10)

    def close(self):
        pass


@pytest.fixture
def lang_fake(monkeypatch):
    LangFake.requests, LangFake.calls, LangFake.detections = [], 0, [("en", 0.99)]
    monkeypatch.setattr("src.transcriber.create_adapter", lambda *a, **k: LangFake())
    monkeypatch.setattr("src.pipeline.resolve_device", lambda requested="auto", index=0: ("cuda", ""))


def run(tmp_path, cfg, **kw):
    media = write_silent_wav(tmp_path / "lecture.wav", seconds=60)
    base = dict(outputs_root=tmp_path / "out", quality_audit=False, auto_topic=True, subjects=["biochemistry"],
                chunk_minutes=1.0, checkpoint_root=tmp_path / "ck", temp_root=tmp_path / "tmp")
    base.update(kw)
    [summary] = process_files([media], RunOptions(**base), cfg)
    meta = json.loads((tmp_path / "out" / "lecture" / "metadata.json").read_text(encoding="utf-8"))
    return summary, meta


@pytest.mark.usefixtures("fake_ffmpeg")
def test_auto_takes_the_dominant_language_of_the_whole_file(tmp_path, cfg, lang_fake, monkeypatch):
    monkeypatch.setattr("src.quality.compute_speech_ranges", lambda wav, vad: [(0.0, 60.0)])
    summary, meta = run(tmp_path, cfg)
    assert meta["language"]["source"] == "sampled" and meta["language"]["forced"] == "en" and not meta["language"]["multilingual"]
    assert meta["language"]["sampling"]["windows"] == lang.SAMPLE_WINDOWS and not meta["language"]["sampling"]["mixed"]
    main = [r for r in LangFake.requests if r.max_seconds is None or r.max_seconds > 100] or LangFake.requests[-1:]
    assert all(r.language == "en" and not r.multilingual for r in LangFake.requests)


@pytest.mark.usefixtures("fake_ffmpeg")
def test_a_mixed_recording_is_decoded_window_by_window_without_hotwords(tmp_path, cfg, lang_fake, monkeypatch):
    monkeypatch.setattr("src.quality.compute_speech_ranges", lambda wav, vad: [(0.0, 60.0)])
    LangFake.detections = [("en", 1.0)] * 20 + [("hu", 0.97)] * 4 + [("en", 1.0)]             # 24 samples + the first-speech probe
    summary, meta = run(tmp_path, cfg)
    assert meta["language"]["source"] == "mixed" and meta["language"]["multilingual"] is True and meta["language"]["forced"] is None
    assert meta["language"]["dominant"] == "en" and meta["language"]["sampling"]["mixed"]
    main = [r for r in LangFake.requests if r.start_offset == 0.0 and not r.max_seconds or (r.max_seconds or 0) > 30]
    assert LangFake.requests and all(r.language in (None, "en") for r in LangFake.requests)
    assert any(r.multilingual and r.language is None for r in LangFake.requests)                 # the main decode
    assert all(not r.hotwords for r in LangFake.requests)                                        # never biased
    assert "mixes languages" in meta["subject"]["hotwords_off_reason"] and meta["subject"]["hotwords_applied"] == []


@pytest.mark.usefixtures("fake_ffmpeg")
def test_the_first_minutes_in_another_language_make_the_recording_mixed(tmp_path, cfg, lang_fake, monkeypatch):
    monkeypatch.setattr("src.quality.compute_speech_ranges", lambda wav, vad: [(0.0, 60.0)])
    LangFake.detections = [("en", 1.0)] * 24 + [("hu", 0.92)]          # all samples English, the first window Hungarian
    summary, meta = run(tmp_path, cfg)
    assert meta["language"]["multilingual"] is True and "first minutes are hu" in meta["language"]["sampling"]["reason"]


@pytest.mark.usefixtures("fake_ffmpeg")
def test_a_language_chosen_by_the_user_is_never_second_guessed(tmp_path, cfg, lang_fake, monkeypatch):
    LangFake.detections = [("hu", 0.99)]
    summary, meta = run(tmp_path, cfg, language="en")
    assert meta["language"]["source"] == "user" and meta["language"]["sampling"] is None and LangFake.calls == 0


@pytest.mark.usefixtures("fake_ffmpeg")
def test_the_mixed_option_forces_per_window_decoding(tmp_path, cfg, lang_fake, monkeypatch):
    monkeypatch.setattr("src.quality.compute_speech_ranges", lambda wav, vad: [(0.0, 60.0)])
    summary, meta = run(tmp_path, cfg, multilingual=True)
    assert meta["language"]["multilingual"] is True and meta["language"]["source"] == "mixed"
    assert any(r.multilingual and r.language is None for r in LangFake.requests)


@pytest.mark.usefixtures("fake_ffmpeg")
def test_a_language_without_a_vocabulary_runs_without_hotwords(tmp_path, cfg, lang_fake, monkeypatch):
    monkeypatch.setattr("src.quality.compute_speech_ranges", lambda wav, vad: [(0.0, 60.0)])
    LangFake.detections = [("hu", 0.99)]
    summary, meta = run(tmp_path, cfg)
    assert meta["language"]["forced"] == "hu" and "no medical vocabulary for language 'hu'" in meta["subject"]["hotwords_off_reason"]
    assert all(not r.hotwords for r in LangFake.requests)


@pytest.mark.usefixtures("fake_ffmpeg")
def test_a_backend_that_cannot_detect_languages_keeps_the_old_behaviour(tmp_path, cfg, monkeypatch):
    class Plain(LangFake):
        detect_language_at = None

    monkeypatch.setattr("src.transcriber.create_adapter", lambda *a, **k: Plain())
    monkeypatch.setattr("src.pipeline.resolve_device", lambda requested="auto", index=0: ("cuda", ""))
    LangFake.requests = []
    summary, meta = run(tmp_path, cfg)
    assert meta["language"]["sampling"] is None


# ----------------------------------------------------------------------------------------------- Transcriber.detect_languages
def test_detect_languages_returns_nothing_for_a_backend_without_detection(monkeypatch):
    class Mute(LangFake):
        detect_language_at = None

    monkeypatch.setattr("src.transcriber.create_adapter", lambda *a, **k: Mute())
    tr = Transcriber(ModelSpec("default", "faster_whisper", "large-v3"), {"compute_type": "float16"}, "cuda", 0, False)
    assert tr.detect_languages("x.wav", [0.0, 30.0]) == []


def test_one_failing_window_does_not_lose_the_others(monkeypatch):
    class Flaky(LangFake):
        def detect_language_at(self, wav, start, seconds=30.0):
            if start == 30.0:
                raise RuntimeError("unreadable")
            return ("tr", 0.9)

    monkeypatch.setattr("src.transcriber.create_adapter", lambda *a, **k: Flaky())
    tr = Transcriber(ModelSpec("default", "faster_whisper", "large-v3"), {"compute_type": "float16"}, "cuda", 0, False)
    assert tr.detect_languages("x.wav", [0.0, 30.0, 60.0]) == [(0.0, "tr", 0.9), (60.0, "tr", 0.9)]


def test_a_model_that_cannot_load_skips_sampling_quietly(monkeypatch):
    class NoLoad(LangFake):
        def load(self, compute_type):
            raise RuntimeError("out of memory")

    monkeypatch.setattr("src.transcriber.create_adapter", lambda *a, **k: NoLoad())
    tr = Transcriber(ModelSpec("default", "faster_whisper", "large-v3"), {"compute_type": "float16"}, "cuda", 0, False)
    assert tr.detect_languages("x.wav", [0.0]) == []


# ----------------------------------------------------------------------------------------------- reading a slice of the 16 kHz wav
def test_wav_slice_matches_the_full_decode(tmp_path):
    path = tmp_path / "a.wav"
    rate = 16000
    tone = (np.sin(np.arange(rate * 10) * 0.05) * 12000).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(tone.tobytes())
    part = read_wav_slice(path, 2.0, 3.0)
    assert part.dtype == np.float32 and len(part) == 3 * rate
    assert np.allclose(part, tone[2 * rate:5 * rate] / 32768.0, atol=1e-4)
    rest = read_wav_slice(path, 8.0, None)
    assert len(rest) == 2 * rate


def test_a_wav_with_another_sample_rate_still_reads_through_the_fallback(tmp_path):
    path = tmp_path / "b.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(22050)
        w.writeframes((np.sin(np.arange(22050 * 4) * 0.03) * 9000).astype("<i2").tobytes())
    part = read_wav_slice(path, 1.0, 2.0)
    assert abs(len(part) - 2 * 16000) < 400                       # resampled to 16 kHz by the full decode
