"""Full pipeline with a fake ASR backend: real FFmpeg extraction, real exporters, no GPU/model."""

import json
import logging

import pytest

from src import pipeline
from src.model_manager import ASRAdapter, Capabilities, DecodeRequest, ModelSpec
from src.pipeline import RunOptions, process_files
from src.schema import Segment, TranscriptionResult
from tests.conftest import write_silent_wav
from tests.test_pdf_context import make_pdf

pytestmark = pytest.mark.usefixtures("fake_ffmpeg")


class FakeAdapter(ASRAdapter):
    backend_name = "fake"
    capabilities = Capabilities(hotwords=True, word_timestamps=False, vad=False, compute_types=True)
    last_request: DecodeRequest | None = None

    def __init__(self) -> None:
        super().__init__(ModelSpec("default", "faster_whisper", "large-v3"))

    def load(self, compute_type: str) -> None:
        self.compute_type = compute_type

    def transcribe(self, request: DecodeRequest) -> TranscriptionResult:
        FakeAdapter.last_request = request
        segs = [
            Segment(0.0, 3.0, " The spino thalamic tract carries pain.", "en", "large-v3", "fake",
                    avg_logprob=-0.2, no_speech_prob=0.01, compression_ratio=1.1),
            Segment(3.0, 6.0, " Uh huh.", "en", "large-v3", "fake",
                    avg_logprob=-1.7, no_speech_prob=0.2, compression_ratio=1.0),
        ]
        return TranscriptionResult(segs, "en", 0.98, 6.0, "large-v3", "fake", self.compute_type,
                                   capabilities_used={"hotwords": bool(request.hotwords)})

    def close(self) -> None:
        pass


@pytest.fixture
def fake_backend(monkeypatch):
    monkeypatch.setattr("src.transcriber.create_adapter", lambda *a, **k: FakeAdapter())


def test_single_file_produces_every_output(tmp_path, cfg, fake_backend):
    media = write_silent_wav(tmp_path / "lecture.wav", seconds=2)
    make_pdf(tmp_path / "lecture.pdf", ["The substantia nigra projects to the putamen. Substantia nigra again."])
    options = RunOptions(mode="quality", subjects=["neuroscience"], outputs_root=tmp_path / "out", language="en")
    [summary] = process_files([media], options, cfg)

    out = tmp_path / "out" / "lecture"
    assert summary.ok and summary.segments == 2 and summary.uncertain == 1
    for name in ("local_raw_transcript.txt", "local_clean_transcript.txt", "local_transcript_with_timestamps.txt",
                 "local_transcript.srt", "local_transcript.vtt", "local_transcript.json", "medical_terms_detected.txt",
                 "medical_entities.json", "possible_medical_errors.txt", "medical_acronyms.txt", "medical_context.json",
                 "uncertain_segments.json", "metadata.json", "processing.log"):
        assert (out / name).is_file(), name
    assert not (out / "deepgram_raw_transcript.txt").exists()                  # local is the default: nothing online

    raw = (out / "local_raw_transcript.txt").read_text(encoding="utf-8")
    assert raw == "The spino thalamic tract carries pain.\nUh huh.\n"          # untouched ASR text
    clean = (out / "local_clean_transcript.txt").read_text(encoding="utf-8")
    assert "spinothalamic tract" in clean and "[UNCERTAIN: Uh huh.]" in clean

    data = json.loads((out / "local_transcript.json").read_text(encoding="utf-8"))
    assert data["segments"][0]["backend"] == "fake" and data["segments"][0]["confidence"] is not None
    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert meta["confidence"]["is_model_confidence"] is False
    assert meta["pdf_context"]["source"] == "lecture.pdf"
    assert "substantia nigra" in [h.lower() for h in meta["subject"]["hotwords"][:5]]        # PDF term ranks near the top
    assert "locally" in meta["privacy"]
    assert meta["model"]["compute_type_used"] == "float16"
    uncertain = json.loads((out / "uncertain_segments.json").read_text(encoding="utf-8"))
    assert uncertain[0]["index"] == 1


def test_hotwords_switch_off_condition_on_previous_text(tmp_path, cfg, fake_backend):
    media = write_silent_wav(tmp_path / "c.wav")
    process_files([media], RunOptions(subjects=["neuroscience"], outputs_root=tmp_path / "o", language="en"), cfg)
    request = FakeAdapter.last_request
    assert request.hotwords and request.profile["condition_on_previous_text"] is False
    meta = json.loads((tmp_path / "o" / "c" / "metadata.json").read_text(encoding="utf-8"))
    assert meta["profile"]["condition_on_previous_text"] is False
    assert any("condition_on_previous_text" in n for n in meta["capability_notes"])


def test_conditioning_kept_when_no_hotwords_or_guard_disabled(tmp_path, cfg, fake_backend):
    media = write_silent_wav(tmp_path / "d.wav")
    process_files([media], RunOptions(outputs_root=tmp_path / "o", language="en", use_hotwords=False), cfg)
    assert FakeAdapter.last_request.hotwords == [] and FakeAdapter.last_request.profile["condition_on_previous_text"] is True
    relaxed = {**cfg, "context": {**cfg["context"], "hotwords_disable_conditioning": False}}
    process_files([media], RunOptions(subjects=["neuroscience"], outputs_root=tmp_path / "o2", language="en"), relaxed)
    assert FakeAdapter.last_request.hotwords and FakeAdapter.last_request.profile["condition_on_previous_text"] is True


def test_pdf_can_be_disabled_and_hotwords_follow_subject(tmp_path, cfg, fake_backend):
    media = write_silent_wav(tmp_path / "a.wav")
    make_pdf(tmp_path / "a.pdf", ["Substantia nigra substantia nigra"])
    options = RunOptions(subjects=["pharmacology"], outputs_root=tmp_path / "out", use_pdf=False, language="en")
    process_files([media], options, cfg)
    hot = FakeAdapter.last_request.hotwords
    from src.vocabulary import load_terms

    pharmacology = {t.lower() for t in load_terms("pharmacology", "en")}
    assert hot and sum(h.lower() in pharmacology for h in hot) >= len(hot) // 2      # follows the requested subject
    assert "substantia nigra" not in [h.lower() for h in hot]                         # PDF was disabled


def test_batch_continues_after_a_failing_file(tmp_path, cfg, fake_backend, caplog):
    good = write_silent_wav(tmp_path / "good.wav")
    bad = tmp_path / "bad.mp3"
    bad.write_bytes(b"not audio")
    options = RunOptions(outputs_root=tmp_path / "out")
    with caplog.at_level(logging.ERROR):
        summaries = process_files([bad, good], options, cfg)
    assert [s.ok for s in summaries] == [False, True]
    assert "bad.mp3" in summaries[0].media.name and summaries[0].error
    assert (tmp_path / "out" / "good" / "local_transcript.json").is_file()


def test_single_failing_file_raises(tmp_path, cfg, fake_backend):
    bad = tmp_path / "bad.mp3"
    bad.write_bytes(b"not audio")
    with pytest.raises(Exception):
        process_files([bad], RunOptions(outputs_root=tmp_path / "out"), cfg)


def test_same_stem_different_extension_do_not_collide(tmp_path, cfg, fake_backend):
    a = write_silent_wav(tmp_path / "talk.wav")
    b = tmp_path / "talk.flac"
    import soundfile as sf

    data, rate = sf.read(str(a))
    sf.write(str(b), data, rate, format="FLAC")
    summaries = process_files([a, b], RunOptions(outputs_root=tmp_path / "out"), cfg)
    assert all(s.ok for s in summaries)
    assert {p.name for p in (tmp_path / "out").iterdir()} == {"talk_wav", "talk_flac"}


def test_capability_check_disables_unsupported_features(cfg):
    caps = Capabilities(hotwords=False, vad=False, word_timestamps=False)
    profile, notes = pipeline._effective_profile(cfg["quality"], caps)
    assert not profile["vad_filter"] and not profile["word_timestamps"] and not profile["use_hotwords"]
    assert len(notes) == 3
