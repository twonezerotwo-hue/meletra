"""A video must never sit at 0%: the job reports REAL progress for every step before the first transcript segment (reading, extracting
the audio, finding speech, detecting the language, listening to the first minutes), and a pause / cancel is honoured in each of them."""

import pytest

from src.model_manager import ASRAdapter, Capabilities, ModelSpec

from src.pipeline import RunOptions, process_files
from src.schema import Segment, TranscriptionResult
from tests.test_ingest_router import encode


class StagedFake(ASRAdapter):
    """Language detection per window, a probe and a decode that report progress the way the real adapter does."""

    backend_name = "fake"
    capabilities = Capabilities(hotwords=True, vad=True, word_timestamps=True, language_detection=True, multilingual=True, compute_types=True)
    detections = 0
    progress_calls = 0
    on_detect = None
    on_progress = None

    def __init__(self) -> None:
        super().__init__(ModelSpec("default", "faster_whisper", "large-v3"))

    def load(self, compute_type):
        self.compute_type = compute_type

    def detect_language_at(self, wav, start, seconds=30.0):
        StagedFake.detections += 1
        if StagedFake.on_detect:
            StagedFake.on_detect(StagedFake.detections)
        return "en", 0.97

    def transcribe(self, request):
        total = request.total_duration
        end = min(total, request.start_offset + request.max_seconds) if request.max_seconds else total
        segs, t = [], request.start_offset
        while t < end - 0.5:
            seg = Segment(t, min(t + 5, end), " glycolysis converts glucose to pyruvate in the cytoplasm", "en", "large-v3", "fake",
                          avg_logprob=-0.2, no_speech_prob=0.01, compression_ratio=1.1)
            segs.append(seg)
            if request.on_segment:
                request.on_segment(seg)
            if request.progress:
                StagedFake.progress_calls += 1
                if StagedFake.on_progress:
                    StagedFake.on_progress(StagedFake.progress_calls)
                request.progress(seg.end, total)
            t += 5
        return TranscriptionResult(segs, "en", 0.99, total, "large-v3", "fake", self.compute_type)

    def close(self):
        pass


@pytest.fixture
def staged(monkeypatch):
    StagedFake.detections = StagedFake.progress_calls = 0
    StagedFake.on_detect = StagedFake.on_progress = None
    monkeypatch.setattr("src.transcriber.create_adapter", lambda *a, **k: StagedFake())


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    return encode(tmp_path_factory.mktemp("v") / "Ders Kaydi çş.mp4", "mp4", "aac", "mpeg4", seconds=60.0)


def run(tmp_path, cfg, source, **kw):
    events: list[dict] = []
    options = RunOptions(outputs_root=tmp_path / "out", temp_root=tmp_path / "tmp", quality_audit=False, language=None, events=events.append, **kw)
    summaries = process_files([source], options, cfg)
    return summaries[0], events


def kinds(events):
    return [e["type"] for e in events]


# ----------------------------------------------------------------------------------------------- progress comes before the first words
def test_every_preparation_step_reports_real_progress_before_the_first_segment(tmp_path, cfg, staged, video):
    summary, events = run(tmp_path, cfg, video)
    assert summary.ok
    stages = [(i, e) for i, e in enumerate(events) if e["type"] == "stage"]
    names = [e["stage"] for _, e in stages]
    assert names[0] == "probing" and {"extracting", "speech_analysis", "language", "analysis"} <= set(names)
    first_words = kinds(events).index("chunk_progress")
    assert all(i < first_words for i, e in stages if e["stage"] != "transcribing")                   # all of it before the first segment
    extracting = [e["percent"] for _, e in stages if e["stage"] == "extracting"]
    assert extracting[0] == 0.0 and extracting == sorted(extracting) and extracting[-1] == pytest.approx(100.0, abs=3.0)
    language = [e for _, e in stages if e["stage"] == "language"]
    assert language and language[-1]["done"] < language[-1]["total"] == StagedFake.detections - 1      # k of N windows, measured
    analysis = [e["percent"] for _, e in stages if e["stage"] == "analysis"]
    assert analysis[0] == 0.0 and analysis == sorted(analysis) and analysis[-1] > 0     # a fast fake: the 0.5 s throttle keeps only a few


def test_the_audio_is_prepared_before_anything_else_and_leaves_no_temporary_file(tmp_path, cfg, staged, video):
    _, events = run(tmp_path, cfg, video)
    order = kinds(events)
    assert order.index("stage") < order.index("audio_preparing") < order.index("audio_ready") < order.index("chunk_progress")
    assert not any((tmp_path / "tmp").iterdir())                                                      # the extracted audio is gone again


def test_a_recording_that_is_already_audio_still_reports_the_same_steps(tmp_path, cfg, staged):
    from tests.conftest import write_silent_wav
    _, events = run(tmp_path, cfg, write_silent_wav(tmp_path / "a.wav", 90))
    assert [e["stage"] for e in events if e["type"] == "stage"][:2] == ["probing", "extracting"]


# ----------------------------------------------------------------------------------------------- a pause / cancel is honoured in each step
def paused_cleanly(summary, events, tmp_path):
    assert summary.paused and not summary.ok and summary.state == "paused" and summary.segments == 0
    assert "paused" in kinds(events) and "chunk_progress" not in kinds(events) and "completed" not in kinds(events)
    assert not any((tmp_path / "tmp").iterdir())                                                      # temporary audio cleaned up


def test_cancel_before_anything_is_prepared(tmp_path, cfg, staged, video):
    summary, events = run(tmp_path, cfg, video, should_stop=lambda: True)
    paused_cleanly(summary, events, tmp_path)
    assert StagedFake.detections == 0


def test_cancel_while_the_language_is_being_detected(tmp_path, cfg, staged, video):
    flag = []
    StagedFake.on_detect = lambda n: flag.append(1) if n >= 2 else None
    summary, events = run(tmp_path, cfg, video, should_stop=lambda: bool(flag))
    paused_cleanly(summary, events, tmp_path)
    assert 2 <= StagedFake.detections < 6 and "analysis" not in [e.get("stage") for e in events]


def test_cancel_while_the_first_minutes_are_analysed(tmp_path, cfg, staged, video):
    flag = []
    StagedFake.on_progress = lambda n: flag.append(1) if n >= 3 else None
    summary, events = run(tmp_path, cfg, video, should_stop=lambda: bool(flag))
    paused_cleanly(summary, events, tmp_path)
    assert StagedFake.progress_calls < 12


def test_cancel_while_the_audio_is_extracted(tmp_path, cfg, staged, video, monkeypatch):
    from src import audio as audio_mod
    calls = []
    real = audio_mod.extract_audio_pyav

    def stopping_after_a_while():
        calls.append(1)
        return len(calls) > 5
    monkeypatch.setattr(audio_mod, "locate_ffmpeg_optional", lambda cfg=None: (None, None))          # the PyAV path, deterministic
    summary, events = run(tmp_path, cfg, video, should_stop=stopping_after_a_while)
    paused_cleanly(summary, events, tmp_path)
    assert "language" not in [e.get("stage") for e in events] and real is audio_mod.extract_audio_pyav


# ----------------------------------------------------------------------------------------------- failures are clear, never a silent wait
def test_a_video_without_sound_stops_with_a_clear_message(tmp_path, cfg, staged):
    silent = encode(tmp_path / "silent.mp4", "mp4", "", "mpeg4", seconds=5.0)
    with pytest.raises(Exception, match="no audio"):
        run(tmp_path, cfg, silent)


def test_a_damaged_video_stops_with_a_clear_message(tmp_path, cfg, staged):
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"garbage" * 500)
    with pytest.raises(Exception):
        run(tmp_path, cfg, broken)
