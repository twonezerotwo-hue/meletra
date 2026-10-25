"""TransformersAdapter behaviour that does not need torch weights (fake pipeline)."""

import numpy as np

from src.model_manager import DecodeRequest, ModelSpec, TransformersAdapter, load_model_specs
from tests.conftest import write_silent_wav

RATE = 16000


def make_adapter(**options) -> TransformersAdapter:
    return TransformersAdapter(ModelSpec("t", "transformers", "x/y", options=options), device="cuda")


def request(wav, hotwords=(), vad=False):
    return DecodeRequest(audio_path=wav, language="tr", profile={"beam_size": 5, "vad_filter": vad}, decoding={},
                         vad_parameters={"threshold": 0.5, "min_silence_duration_ms": 500}, hotwords=list(hotwords))


class FakePipe:
    """Mimics the HF ASR pipeline: one dict in -> one dict out, list in -> list out."""

    device = "cpu"

    def __init__(self, chunks_per_call):
        self.chunks_per_call, self.calls, self.inputs = list(chunks_per_call), [], []

    def __call__(self, inputs, **kwargs):
        self.calls.append(kwargs)
        batch = inputs if isinstance(inputs, list) else [inputs]
        self.inputs.append([len(b["raw"]) for b in batch])
        outs = [{"text": "", "chunks": self.chunks_per_call.pop(0)} for _ in batch]
        return outs if isinstance(inputs, list) else outs[0]


def test_prompt_hotwords_are_opt_in_per_model():
    assert make_adapter().capabilities.hotwords is False
    assert make_adapter(prompt_hotwords=True).capabilities.hotwords is True
    specs = load_model_specs()
    assert TransformersAdapter(specs["turkish_medical"]).capabilities.hotwords is False


def test_backend_capabilities_are_reported_honestly():
    caps = make_adapter().capabilities
    assert caps.vad                                  # provided by Silero VAD from faster-whisper
    assert not caps.word_timestamps and not caps.segment_stats and not caps.compute_types


def test_missing_or_reversed_end_times_are_repaired_and_clamped(tmp_path):
    wav = write_silent_wav(tmp_path / "a.wav", seconds=30)
    adapter = make_adapter()
    adapter._pipe = FakePipe([[
        {"timestamp": (0.0, 4.0), "text": " first"},
        {"timestamp": (4.0, None), "text": " second"},          # end missing -> next start
        {"timestamp": (9.0, 8.0), "text": " third"},            # end < start -> next start
        {"timestamp": (12.0, 99.0), "text": " fourth"},         # beyond audio -> clamped
        {"timestamp": (20.0, 21.0), "text": "   "},             # empty -> dropped
        {"timestamp": (25.0, None), "text": " fifth"},          # last, no end -> audio duration
    ]])
    adapter._torch = None
    result = adapter.transcribe(request(wav))
    spans = [(round(s.start, 1), round(s.end, 1)) for s in result.segments]
    assert spans == [(0.0, 4.0), (4.0, 9.0), (9.0, 12.0), (12.0, 30.0), (25.0, 30.0)]
    assert result.segments[0].confidence is None and result.segments[0].avg_logprob is None
    assert result.capabilities_used["vad"] is False and "hallucinated" in result.notes[0]


def test_sequential_longform_is_default_and_chunking_is_opt_in(tmp_path):
    wav = write_silent_wav(tmp_path / "a.wav", seconds=2)
    for options, expect_chunk in (({}, False), ({"chunk_length_s": 30}, True)):
        adapter = make_adapter(**options)
        adapter._pipe = FakePipe([[{"timestamp": (0.0, 1.0), "text": " x"}]])
        adapter.transcribe(request(wav))
        assert ("chunk_length_s" in adapter._pipe.calls[0]) is expect_chunk
        assert adapter._pipe.calls[0]["return_timestamps"] is True


def test_speech_windows_group_chunks_and_respect_gaps(monkeypatch):
    import faster_whisper.vad as vad

    def chunk(a, b):
        return {"start": int(a * RATE), "end": int(b * RATE)}

    monkeypatch.setattr(vad, "get_speech_timestamps", lambda *a, **k: [
        chunk(1, 5), chunk(6, 12),          # gap 1 s  -> same window
        chunk(20, 24),                      # gap 8 s  -> new window
        chunk(25, 40),                      # would make the window 20 s..40 s = 20 s -> still fits
        chunk(45, 52),                      # window would span 20..52 = 32 s > 29 -> new window
    ])
    windows = TransformersAdapter.speech_windows(np.zeros(60 * RATE, dtype="float32"), RATE, {"threshold": 0.5})
    assert [(round(s / RATE), round(e / RATE)) for s, e in windows] == [(1, 12), (20, 40), (45, 52)]


def test_vad_windows_are_decoded_separately_and_timestamps_get_the_window_offset(tmp_path, monkeypatch):
    wav = write_silent_wav(tmp_path / "a.wav", seconds=60)
    adapter = make_adapter(batch_size=2)
    adapter._pipe = FakePipe([
        [{"timestamp": (0.0, 3.0), "text": " first window"}],
        [{"timestamp": (1.0, 4.0), "text": " second window"}, {"timestamp": (4.0, None), "text": " tail"}],
    ])
    adapter._torch = None
    monkeypatch.setattr(TransformersAdapter, "speech_windows",
                        staticmethod(lambda *a, **k: [(10 * RATE, 20 * RATE), (40 * RATE, 50 * RATE)]))
    progress = []
    req = request(wav, vad=True)
    req.progress = lambda done, total: progress.append((round(done), round(total)))
    result = adapter.transcribe(req)
    assert adapter._pipe.inputs == [[10 * RATE, 10 * RATE]]              # one batch of two windows, silence not decoded
    spans = [(s.text.strip(), round(s.start), round(s.end)) for s in result.segments]
    assert spans == [("first window", 10, 13), ("second window", 41, 44), ("tail", 44, 50)]
    assert result.capabilities_used["vad"] is True
    assert progress[-1] == (50, 60)


def test_no_speech_means_no_segments_and_no_model_call(tmp_path, monkeypatch):
    wav = write_silent_wav(tmp_path / "a.wav", seconds=5)
    adapter = make_adapter()
    adapter._pipe = FakePipe([])
    adapter._torch = None
    monkeypatch.setattr(TransformersAdapter, "speech_windows", staticmethod(lambda *a, **k: []))
    result = adapter.transcribe(request(wav, vad=True))
    assert result.segments == [] and adapter._pipe.calls == []
