"""Crash -> resume of a long recording through the real pipeline with a fake ASR backend."""

import json

import pytest

from src.model_manager import ASRAdapter, Capabilities, DecodeRequest, ModelSpec
from src.pipeline import RunOptions, process_files
from src.schema import Segment, TranscriptionResult
from tests.conftest import write_silent_wav

pytestmark = pytest.mark.usefixtures("fake_ffmpeg")


class ResumableFake(ASRAdapter):
    """Emits one 10-second segment at a time (absolute timestamps), like the real adapter after the offset fix."""

    backend_name = "fake"
    capabilities = Capabilities(hotwords=True, vad=True, word_timestamps=True, compute_types=True)
    crash_after: int | None = None
    start_offsets: list[float] = []

    def __init__(self) -> None:
        super().__init__(ModelSpec("default", "faster_whisper", "large-v3"))

    def load(self, compute_type: str) -> None:
        self.compute_type = compute_type

    def transcribe(self, request: DecodeRequest) -> TranscriptionResult:
        ResumableFake.start_offsets.append(request.start_offset)
        total, t, made, out = request.total_duration, request.start_offset, 0, []
        while t < total - 0.5:
            seg = Segment(t, min(t + 10.0, total), f" sentence at {int(t)}", "en", "large-v3", "fake",
                          avg_logprob=-0.2, no_speech_prob=0.01, compression_ratio=1.1)
            out.append(seg)
            if request.on_segment:
                request.on_segment(seg)
            made += 1
            if ResumableFake.crash_after is not None and made >= ResumableFake.crash_after:
                ResumableFake.crash_after = None          # crash only the first time
                raise RuntimeError("simulated crash (CUDA error / power cut)")
            t += 10.0
        return TranscriptionResult(out, "en", 1.0, total, "large-v3", "fake", self.compute_type)

    def close(self) -> None:
        pass


@pytest.fixture
def fake(monkeypatch):
    ResumableFake.crash_after, ResumableFake.start_offsets = None, []
    monkeypatch.setattr("src.transcriber.create_adapter", lambda *a, **k: ResumableFake())


def no_retry(cfg):
    """A crash that is not retried inside the run, so the SAME command has to resume it next time."""
    return {**cfg, "long_audio": {**cfg["long_audio"], "chunk_retries": 0}}


def test_chunk_is_retried_automatically_and_continues_after_the_saved_segments(tmp_path, cfg, fake):
    media = write_silent_wav(tmp_path / "retry.wav", seconds=60)
    options = RunOptions(outputs_root=tmp_path / "out", language="en", quality_audit=False, auto_topic=False)
    ResumableFake.crash_after = 3
    [summary] = process_files([media], options, cfg)
    assert summary.ok
    assert ResumableFake.start_offsets == [0.0, 30.0]                 # retry resumed after the 3 segments already saved
    raw = (tmp_path / "out" / "retry" / "local_raw_transcript.txt").read_text(encoding="utf-8").splitlines()
    assert [line.strip() for line in raw] == [f"sentence at {t}" for t in range(0, 60, 10)]


def test_crash_keeps_finished_work_and_the_same_command_resumes(tmp_path, cfg, fake):
    cfg = no_retry(cfg)
    media = write_silent_wav(tmp_path / "long lecture.wav", seconds=60)          # a space in the name on purpose
    options = RunOptions(outputs_root=tmp_path / "out", language="en", quality_audit=False, auto_topic=False)
    out = tmp_path / "out" / "long lecture"

    ResumableFake.crash_after = 3
    with pytest.raises(RuntimeError, match="simulated crash"):
        process_files([media], options, cfg)
    chunk_dir = out / "checkpoint" / "chunks" / "chunk_0000"
    assert (chunk_dir / "partial_segments.jsonl").read_text(encoding="utf-8").count("\n") == 3
    partial = (out / "partial_transcript.txt").read_text(encoding="utf-8")        # partial recovery
    assert "sentence at 0" in partial and "sentence at 20" in partial and "sentence at 30" not in partial
    assert not (out / "local_raw_transcript.txt").exists()                               # no half-finished "final" files

    [summary] = process_files([media], options, cfg)                                # same command again
    assert ResumableFake.start_offsets == [0.0, 30.0]                              # resumed right after the last saved segment
    assert summary.ok and summary.resumed_segments == 3
    raw = (out / "local_raw_transcript.txt").read_text(encoding="utf-8").splitlines()
    assert [line.strip() for line in raw] == [f"sentence at {t}" for t in range(0, 60, 10)]    # each exactly once
    data = json.loads((out / "local_transcript.json").read_text(encoding="utf-8"))
    assert [round(s["start"]) for s in data["segments"]] == [0, 10, 20, 30, 40, 50]            # absolute timestamps
    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert meta["resume"]["resumed_segments"] == 3
    assert not (out / "checkpoint").exists()                                       # cleaned after success


def test_changed_settings_do_not_resume_a_stale_checkpoint(tmp_path, cfg, fake):
    cfg = no_retry(cfg)
    media = write_silent_wav(tmp_path / "a.wav", seconds=40)
    ResumableFake.crash_after = 2
    with pytest.raises(RuntimeError):
        process_files([media], RunOptions(outputs_root=tmp_path / "o", language="en", quality_audit=False, auto_topic=False), cfg)
    process_files([media], RunOptions(outputs_root=tmp_path / "o", language="en", mode="fast", quality_audit=False, auto_topic=False), cfg)
    assert ResumableFake.start_offsets[-1] == 0.0                                  # fast != quality: started over


def test_no_resume_flag_starts_from_zero(tmp_path, cfg, fake):
    cfg = no_retry(cfg)
    media = write_silent_wav(tmp_path / "a.wav", seconds=40)
    options = RunOptions(outputs_root=tmp_path / "o", language="en", quality_audit=False, auto_topic=False)
    ResumableFake.crash_after = 2
    with pytest.raises(RuntimeError):
        process_files([media], options, cfg)
    options.resume = False
    process_files([media], options, cfg)
    assert ResumableFake.start_offsets[-1] == 0.0


def test_fully_checkpointed_file_needs_no_decode(tmp_path, cfg, fake):
    media = write_silent_wav(tmp_path / "a.wav", seconds=30)
    options = RunOptions(outputs_root=tmp_path / "o", language="en", quality_audit=False, keep_checkpoint=True, auto_topic=False)
    process_files([media], options, cfg)
    calls_before = len(ResumableFake.start_offsets)
    [summary] = process_files([media], options, cfg)
    assert summary.ok and len(ResumableFake.start_offsets) == calls_before         # nothing re-decoded
    raw = (tmp_path / "o" / "a" / "local_raw_transcript.txt").read_text(encoding="utf-8").splitlines()
    assert len(raw) == 3


def test_probe_then_transcribe_reuses_the_saved_plan(tmp_path, cfg, monkeypatch):
    """`probe` decides language/subject once; the following run must not probe again."""
    probes = []

    class ProbeFake(ResumableFake):
        def transcribe(self, request):
            if request.max_seconds:                                              # the probe call
                probes.append(request.start_offset)
                seg = Segment(0, 5, " the glycolysis pathway uses hexokinase and phosphofructokinase "
                                    "and pyruvate kinase and the citric acid cycle", "en", "large-v3", "fake")
                return TranscriptionResult([seg], "en", 0.99, 5.0, "large-v3", "fake", self.compute_type)
            return super().transcribe(request)

    ResumableFake.crash_after, ResumableFake.start_offsets = None, []
    monkeypatch.setattr("src.transcriber.create_adapter", lambda *a, **k: ProbeFake())
    media = write_silent_wav(tmp_path / "lec.wav", seconds=30)
    common = dict(outputs_root=tmp_path / "o", quality_audit=False, subjects=["general_medical"])
    process_files([media], RunOptions(auto_topic=True, probe_only=True, **common), cfg)
    plan = json.loads((tmp_path / "o" / "lec" / "probe_plan.json").read_text(encoding="utf-8"))
    assert plan["language"] == "en" and plan["language_source"] == "probe"
    assert "biochemistry" in plan["subjects"]
    assert len(probes) == 1

    process_files([media], RunOptions(auto_topic=True, **common), cfg)
    assert len(probes) == 1                                                       # saved plan reused, no second probe
    meta = json.loads((tmp_path / "o" / "lec" / "metadata.json").read_text(encoding="utf-8"))
    assert "biochemistry" in meta["subject"]["subjects"]
    assert meta["subject"]["hotwords"][0].lower() in {"hexokinase", "phosphofructokinase", "pyruvate kinase",
                                                      "citric acid cycle", "glycolysis"}     # heard terms come first
