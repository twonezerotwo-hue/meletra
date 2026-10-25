"""The ASR backend abstraction and the macOS-optimized backend (MLX). The MLX package is replaced by a fake on this Windows machine;
what is checked is the contract: the same Segment / Word / TranscriptionResult objects and the same exports as the faster-whisper backend."""

import platform
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from src import asr
from src.asr import (ASRBackend, BackendUnavailable, FasterWhisperBackend, MacOptimizedBackend, TransformersBackend, create_backend,
                     default_engine, describe_engines)
from src.asr import mac as mac_module
from src.exporters import segments_to_srt, segments_to_vtt, transcript_json
from src.model_manager import ASRAdapter, DecodeRequest, GpuOutOfMemoryError, ModelSpec, TransformersAdapter
from src.platform import MacPlatform, WindowsPlatform
from src.schema import Segment, TranscriptionResult, Word
from src.transcriber import Transcriber
from tests.conftest import write_silent_wav
from tests.test_platform import mac

SPEC = ModelSpec("mlx_large_v3", "mlx_whisper", "C:/models/whisper-large-v3-mlx")


def decode_request(path, **kw):
    defaults = dict(audio_path=path, language="en", profile={"beam_size": 5, "temperature": [0.0, 0.2], "compute_type": "float16",
                                                             "vad_filter": False, "condition_on_previous_text": True, "word_timestamps": True},
                    decoding={"no_speech_threshold": 0.6, "log_prob_threshold": -1.0, "compression_ratio_threshold": 2.4},
                    vad_parameters={})
    defaults.update(kw)
    return DecodeRequest(**defaults)


class FakeMlx:
    """Stands in for the ``mlx_whisper`` package (and its small helpers) with the dict format of OpenAI's Whisper."""

    def __init__(self):
        self.calls = []
        self.reply = {"language": "en", "segments": [
            {"id": 0, "start": 0.0, "end": 2.5, "text": " Hexokinase phosphorylates glucose.", "avg_logprob": -0.21, "no_speech_prob": 0.01,
             "compression_ratio": 1.4, "words": [{"word": " Hexokinase", "start": 0.0, "end": 0.9, "probability": 0.98},
                                                  {"word": " phosphorylates", "start": 0.9, "end": 1.8, "probability": 0.91}]},
            {"id": 1, "start": 2.5, "end": 99.0, "text": "   ", "avg_logprob": -0.5, "no_speech_prob": 0.5, "compression_ratio": 1.0},
            {"id": 2, "start": 3.0, "end": 400.0, "text": " It traps glucose.", "avg_logprob": -0.3, "no_speech_prob": 0.02, "compression_ratio": 1.2},
        ]}
        self.error = None
        self.detect_probabilities = {"en": 0.9, "tr": 0.1}

    def install(self, monkeypatch):
        main = types.ModuleType("mlx_whisper")
        main.transcribe = self._transcribe
        audio = types.ModuleType("mlx_whisper.audio")
        audio.N_FRAMES, audio.N_SAMPLES = 3000, 480000
        audio.log_mel_spectrogram = lambda a, n_mels=128, padding=0: ("mel", len(a))
        audio.pad_or_trim = lambda mel, frames, axis=-1: types.SimpleNamespace(astype=lambda dtype: ("segment", mel))
        holder = types.ModuleType("mlx_whisper.transcribe")
        outer = self

        class Model:
            dims = types.SimpleNamespace(n_mels=128)

            def detect_language(self, segment):
                outer.calls.append(("detect", segment))
                return None, outer.detect_probabilities

        holder.ModelHolder = types.SimpleNamespace(get_model=lambda path, dtype: Model())
        tokenizer = types.ModuleType("mlx_whisper.tokenizer")
        tokenizer.get_tokenizer = lambda multilingual, **kw: types.SimpleNamespace(encode=lambda text: text.split())
        core = types.ModuleType("mlx.core")
        core.float16 = "float16"
        core.clear_cache = lambda: None
        pkg = types.ModuleType("mlx")
        pkg.core = core
        for name, module in (("mlx_whisper", main), ("mlx_whisper.audio", audio), ("mlx_whisper.transcribe", holder),
                             ("mlx_whisper.tokenizer", tokenizer), ("mlx", pkg), ("mlx.core", core)):
            monkeypatch.setitem(sys.modules, name, module)
        return self

    def _transcribe(self, audio, **kw):
        self.calls.append(("transcribe", len(audio), kw))
        if self.error:
            raise self.error
        return self.reply


@pytest.fixture()
def fake_mlx(monkeypatch, tmp_path):
    fake = FakeMlx().install(monkeypatch)
    m, _ = mac(tmp_path)                                               # an arm64 Mac
    monkeypatch.setattr(mac_module, "current", lambda: m)
    return fake


@pytest.fixture()
def loaded(fake_mlx):
    backend = create_backend(SPEC, "cpu")
    backend.load("float16")
    fake_mlx.calls.clear()
    return backend, fake_mlx


# ---------------------------------------------------------------------------------------------------------------- the abstraction
def test_every_backend_is_an_asr_backend_with_the_same_interface():
    assert ASRBackend is ASRAdapter
    for cls in (FasterWhisperBackend, TransformersBackend, MacOptimizedBackend):
        assert issubclass(cls, ASRBackend)
        for method in ("load", "transcribe", "close"):
            assert callable(getattr(cls, method))
        assert cls.backend_name and cls.capabilities is not None
    assert asr.ENGINES == {"faster_whisper": FasterWhisperBackend, "transformers": TransformersBackend, "mlx_whisper": MacOptimizedBackend}


def test_the_windows_backend_is_the_one_of_version_011():
    spec = ModelSpec("default", "faster_whisper", "large-v3")
    backend = create_backend(spec, "cuda", 0, False)
    assert type(backend) is FasterWhisperBackend and backend.backend_name == "faster-whisper" and backend.device == "cuda"
    caps = backend.capabilities
    assert caps.hotwords and caps.word_timestamps and caps.vad and caps.segment_stats and caps.multilingual and caps.compute_types


def test_the_mac_backend_declares_what_it_cannot_do():
    backend = create_backend(SPEC, "cpu")
    assert type(backend) is MacOptimizedBackend and backend.device == "metal" and backend.backend_name == "mlx-whisper"
    caps = backend.capabilities
    assert caps.hotwords and caps.word_timestamps and caps.vad and caps.segment_stats and caps.language_detection
    assert not caps.multilingual and not caps.compute_types


def test_the_engine_a_computer_should_use(tmp_path):
    arm, _ = mac(tmp_path, machine="arm64")
    intel, _ = mac(tmp_path, machine="x86_64")
    assert default_engine(arm) == "mlx_whisper" and default_engine(intel) == "faster_whisper"
    assert default_engine(WindowsPlatform(env={}, home=Path("C:/Users/x"))) == "faster_whisper"
    expected = "mlx_whisper" if sys.platform == "darwin" and platform.machine() == "arm64" else "faster_whisper"
    assert default_engine() == expected


def test_describe_engines_says_why_an_engine_cannot_run(tmp_path, monkeypatch):
    rows = {r["engine"]: r for r in describe_engines()}
    if sys.platform == "darwin" and platform.machine() == "arm64":
        assert rows["mlx_whisper"]["ready"] == "yes"
    else:
        assert rows["mlx_whisper"]["ready"] == "no" and "Apple Silicon" in rows["mlx_whisper"]["note"]
    assert rows["faster_whisper"]["ready"] == "yes"
    arm, _ = mac(tmp_path)
    monkeypatch.setattr("importlib.util.find_spec", lambda name, *a: None if name == "mlx_whisper" else object())
    assert {r["engine"]: r for r in describe_engines(arm)}["mlx_whisper"]["note"] == "mlx-whisper is not installed"


# ---------------------------------------------------------------------------------------------------------------------- loading
def test_loading_needs_apple_silicon_and_the_package(monkeypatch, tmp_path):
    backend = create_backend(SPEC, "cpu")
    if sys.platform != "darwin" or platform.machine() != "arm64":
        with pytest.raises(BackendUnavailable, match="Apple Silicon"):
            backend.load("float16")
    intel, _ = mac(tmp_path, machine="x86_64")
    monkeypatch.setattr(mac_module, "current", lambda: intel)
    with pytest.raises(BackendUnavailable, match="Apple Silicon"):
        backend.load("float16")
    arm, _ = mac(tmp_path)
    monkeypatch.setattr(mac_module, "current", lambda: arm)
    monkeypatch.setitem(sys.modules, "mlx_whisper", None)                 # "import mlx_whisper" raises ImportError
    with pytest.raises(BackendUnavailable, match="not installed"):
        backend.load("float16")


def test_loading_warms_the_model_and_records_the_precision(fake_mlx):
    backend = create_backend(SPEC, "cpu")
    backend.load("float16")
    assert backend.compute_type == "float16"
    kind, samples, kw = fake_mlx.calls[0]
    assert kind == "transcribe" and samples == 16000 and kw["path_or_hf_repo"] == SPEC.model and kw["fp16"] is True
    backend.close()
    with pytest.raises(RuntimeError, match="not loaded"):
        backend.transcribe(decode_request(Path("x.wav")))


def test_out_of_gpu_memory_is_translated(fake_mlx):
    fake_mlx.error = RuntimeError("[METAL] Command buffer execution failed: Insufficient Memory (kIOGPUCommandBufferCallbackErrorOutOfMemory)")
    backend = create_backend(SPEC, "cpu")
    with pytest.raises(GpuOutOfMemoryError):
        backend.load("float16")


# ----------------------------------------------------------------------------------------------------------------- the same results
def test_segments_words_language_and_stats_come_back_in_the_shared_schema(loaded, tmp_path):
    backend, fake = loaded
    wav = write_silent_wav(tmp_path / "a.wav", seconds=10)
    seen, progress = [], []
    request = decode_request(wav, on_segment=seen.append, progress=lambda done, total: progress.append((done, total)))
    result = backend.transcribe(request)
    assert isinstance(result, TranscriptionResult) and result.backend == "mlx-whisper" and result.language == "en"
    assert result.compute_type == "float16" and result.duration == pytest.approx(10.0) and result.language_probability is None
    first, second = result.segments                                       # the blank segment is dropped
    assert isinstance(first, Segment) and first.text == " Hexokinase phosphorylates glucose." and first.backend == "mlx-whisper"
    assert (first.start, first.end) == (0.0, 2.5)
    assert first.avg_logprob == -0.21 and first.no_speech_prob == 0.01 and first.compression_ratio == 1.4
    assert [type(w) for w in first.words] == [Word, Word] and first.words[1].word == " phosphorylates" and first.words[1].probability == 0.91
    assert second.end == pytest.approx(10.0)                              # a timestamp past the audio is clamped, like faster-whisper's
    assert seen == result.segments and progress[-1] == (pytest.approx(10.0), pytest.approx(10.0))
    assert result.capabilities_used == {"hotwords": False, "word_timestamps": True, "vad": False, "multilingual": False}
    assert any("greedy" in note for note in result.notes)


def test_the_exports_are_identical_in_form_to_faster_whispers(loaded, tmp_path):
    backend, _ = loaded
    wav = write_silent_wav(tmp_path / "a.wav", seconds=10)
    result = backend.transcribe(decode_request(wav))
    srt, vtt = segments_to_srt(result.segments), segments_to_vtt(result.segments)
    assert "00:00:00,000 --> " in srt and "Hexokinase phosphorylates glucose." in srt and vtt.startswith("WEBVTT")
    data = transcript_json(result.segments, language=result.language, language_probability=result.language_probability, model=result.model,
                           backend=result.backend, duration=result.duration, source="a.wav")
    assert set(data) >= {"language", "segments"}
    keys = set(data["segments"][0])
    assert {"start", "end", "text", "confidence", "language", "model", "backend", "avg_logprob", "no_speech_prob", "compression_ratio",
            "uncertain", "uncertainty_reasons", "words"} <= keys
    assert set(data["segments"][0]["words"][0]) == {"start", "end", "word", "probability"}


def test_resume_offsets_make_timestamps_absolute_and_only_the_slice_is_decoded(loaded, tmp_path):
    backend, fake = loaded
    wav = write_silent_wav(tmp_path / "a.wav", seconds=60)
    result = backend.transcribe(decode_request(wav, start_offset=30.0, max_seconds=20.0, total_duration=60.0))
    (_kind, samples, _kw), = fake.calls
    assert samples == 20 * 16000
    assert result.segments[0].start == 30.0 and result.segments[0].words[0].start == 30.0
    assert result.duration == pytest.approx(50.0)


def test_decode_settings_are_translated_for_mlx(loaded, tmp_path):
    backend, fake = loaded
    wav = write_silent_wav(tmp_path / "a.wav", seconds=5)
    backend.transcribe(decode_request(wav, decoding={"no_speech_threshold": 0.5, "log_prob_threshold": -0.8,
                                                     "compression_ratio_threshold": 2.2, "hallucination_silence_threshold": 2.0}))
    kw = fake.calls[0][2]
    assert kw["language"] == "en" and kw["task"] == "transcribe" and kw["temperature"] == (0.0, 0.2) and kw["word_timestamps"] is True
    assert kw["no_speech_threshold"] == 0.5 and kw["logprob_threshold"] == -0.8 and kw["compression_ratio_threshold"] == 2.2
    assert kw["hallucination_silence_threshold"] == 2.0 and "beam_size" not in kw and "initial_prompt" not in kw


def test_hotwords_become_the_prompt_within_the_token_budget(loaded, tmp_path):
    backend, fake = loaded
    wav = write_silent_wav(tmp_path / "a.wav", seconds=5)
    terms = ["hexokinase", "glucokinase", "phosphofructokinase", "enolase"]
    result = backend.transcribe(decode_request(wav, hotwords=terms, hotword_token_budget=3))
    assert result.hotwords_applied == terms[:3] and fake.calls[0][2]["initial_prompt"] == "hexokinase, glucokinase, phosphofructokinase"
    assert result.capabilities_used["hotwords"] is True and result.hotword_tokens == 3


def test_language_is_detected_once_when_not_given(loaded, tmp_path):
    backend, fake = loaded
    wav = write_silent_wav(tmp_path / "a.wav", seconds=5)
    result = backend.transcribe(decode_request(wav, language=None))
    assert result.language == "en" and result.language_probability == pytest.approx(0.9)
    assert fake.calls[0][0] == "detect" and fake.calls[1][2]["language"] == "en"
    assert backend.detect_language_at(wav, 0.0, 5.0) == ("en", 0.9)
    fake.detect_probabilities = {"tr": 0.8, "en": 0.2}
    assert backend.detect_language_at(wav, 0.0, 5.0) == ("tr", 0.8)


def test_voice_activity_windows_are_decoded_one_by_one(loaded, tmp_path, monkeypatch):
    backend, fake = loaded
    wav = write_silent_wav(tmp_path / "a.wav", seconds=40)
    monkeypatch.setattr(TransformersAdapter, "speech_windows",
                        staticmethod(lambda audio, rate, params, max_window_s=29.0, max_gap_s=2.0: [(0, 10 * rate), (20 * rate, 35 * rate)]))
    request = decode_request(wav, profile={"beam_size": 5, "temperature": 0.0, "compute_type": "float16", "vad_filter": True,
                                           "condition_on_previous_text": True, "word_timestamps": False})
    result = backend.transcribe(request)
    sizes = [c[1] for c in fake.calls if c[0] == "transcribe"]
    assert sizes == [10 * 16000, 15 * 16000]
    starts = [s.start for s in result.segments]
    assert starts[:1] == [0.0] and 20.0 in starts                         # the second window's segments start at its own offset
    assert result.capabilities_used["vad"] is True and fake.calls[0][2]["temperature"] == (0.0,)


def test_the_pipeline_runs_the_mac_backend_like_any_other(fake_mlx, tmp_path, cfg):
    transcriber = Transcriber(SPEC, cfg["quality"], "metal")
    assert transcriber.compute_types == ["float16"] and transcriber.adapter.device == "metal"
    wav = write_silent_wav(tmp_path / "a.wav", seconds=10)
    events = []
    transcriber.events = lambda name, **kw: events.append(name)
    result = transcriber.transcribe(decode_request(wav))
    assert [s.text.strip() for s in result.segments][0] == "Hexokinase phosphorylates glucose." and events[:2] == ["model_loading", "model_loaded"]
    assert transcriber.detect_languages(wav, [0.0, 3.0])[0][1] == "en"
    transcriber.close()


def test_the_worker_never_asks_cuda_questions_for_mlx(monkeypatch, cfg):
    import src.pipeline as pipeline

    made = {}

    class Dummy:
        def __init__(self, spec, profile, device, index, offline):
            made["device"] = device

        def close(self):
            pass

    monkeypatch.setattr(pipeline, "Transcriber", Dummy)
    monkeypatch.setattr(pipeline, "resolve_device", lambda *a, **k: pytest.fail("CUDA was consulted for an MLX model"))
    options = pipeline.RunOptions(model_spec=SPEC, device="auto")
    pipeline.process_files([], options, cfg)
    assert made["device"] == "metal"
