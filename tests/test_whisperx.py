"""Optional WhisperX integration, tested without installing WhisperX (API mirrored with fakes)."""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

from src import whisperx_extra
from src.pipeline import RunOptions, process_files
from src.schema import Word
from src.whisperx_extra import WhisperXUnavailableError, refine_segments
from tests.conftest import make_segment, write_silent_wav
from tests.test_pipeline import FakeAdapter

ROOT = Path(__file__).resolve().parent.parent


def load_worker():
    spec = importlib.util.spec_from_file_location("whisperx_worker", ROOT / "scripts" / "whisperx_worker.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeWhisperX(types.ModuleType):
    """Mirrors the whisperx 3.8.6 signatures that the worker relies on."""

    def __init__(self):
        super().__init__("whisperx")
        self.calls = []
        self.diarize = types.ModuleType("whisperx.diarize")
        outer = self

        class DiarizationPipeline:
            def __init__(self, model_name=None, token=None, device="cpu", cache_dir=None):
                outer.calls.append(("DiarizationPipeline", token, device))

            def __call__(self, audio, num_speakers=None, min_speakers=None, max_speakers=None, **_):
                outer.calls.append(("diarize", min_speakers, max_speakers))
                return "DIAR_DF"

        self.diarize.DiarizationPipeline = DiarizationPipeline

    def load_audio(self, path):
        self.calls.append(("load_audio", path))
        return [0.0]

    def load_align_model(self, language_code, device, model_name=None, model_dir=None, model_cache_only=False):
        self.calls.append(("load_align_model", language_code, device))
        return "MODEL_A", {"meta": 1}

    def align(self, transcript, model, align_model_metadata, audio, device, interpolate_method="nearest",
              return_char_alignments=False, print_progress=False, combined_progress=False, progress_callback=None):
        self.calls.append(("align", model, return_char_alignments, device))
        segs = [{**s, "words": [{"word": "The", "start": s["start"], "end": s["start"] + 0.4, "score": 0.9},
                                {"word": "42", "score": None}]} for s in transcript]
        return {"segments": segs, "word_segments": []}

    def assign_word_speakers(self, diarize_df, transcript_result, speaker_embeddings=None, fill_nearest=False):
        self.calls.append(("assign_word_speakers", diarize_df))
        for seg in transcript_result["segments"]:
            seg["speaker"] = "SPEAKER_00"
        return transcript_result


@pytest.fixture
def fake_whisperx(monkeypatch):
    fake = FakeWhisperX()
    monkeypatch.setitem(sys.modules, "whisperx", fake)
    monkeypatch.setitem(sys.modules, "whisperx.diarize", fake.diarize)
    return fake


REQUEST = {"audio_path": "a.wav", "language": "en", "device": "cuda", "diarize": False,
           "segments": [{"start": 0.0, "end": 2.0, "text": "The thalamus."}]}


def test_worker_alignment_only_never_touches_text_or_diarization(fake_whisperx, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    result = load_worker().run(dict(REQUEST))
    assert result["aligned"] and not result["diarized"]
    assert result["segments"][0]["text"] == "The thalamus."
    assert [c[0] for c in fake_whisperx.calls] == ["load_audio", "load_align_model", "align"]
    assert ("load_align_model", "en", "cuda") in fake_whisperx.calls


def test_worker_diarization_uses_token_and_assigns_speakers(fake_whisperx, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    result = load_worker().run({**REQUEST, "diarize": True, "min_speakers": 1, "max_speakers": 2})
    assert result["diarized"] and result["segments"][0]["speaker"] == "SPEAKER_00"
    assert ("DiarizationPipeline", "hf_test", "cuda") in fake_whisperx.calls
    assert ("diarize", 1, 2) in fake_whisperx.calls


def test_worker_refuses_diarization_without_token(fake_whisperx, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        load_worker().run({**REQUEST, "diarize": True})


def test_worker_main_writes_error_json(tmp_path, monkeypatch):
    monkeypatch.delitem(sys.modules, "whisperx", raising=False)
    monkeypatch.setitem(sys.modules, "whisperx", None)  # makes `import whisperx` fail
    request, out = tmp_path / "in.json", tmp_path / "out.json"
    request.write_text(json.dumps(REQUEST), encoding="utf-8")
    assert load_worker().main(["worker", str(request), str(out)]) == 1
    assert "error" in json.loads(out.read_text(encoding="utf-8"))


def test_merge_replaces_words_and_adds_speakers_without_changing_text():
    segs = [make_segment(" The thalamus.", 0, 2), make_segment(" Next.", 2, 4)]
    segs[0].words = [Word(0, 1, "old", 0.1)]
    notes = whisperx_extra._merge(segs, {"aligned": True, "diarized": True, "segments": [
        {"words": [{"word": "The", "start": 0.0, "end": 0.4, "score": 0.9}, {"word": "42", "start": None, "end": None}],
         "speaker": "SPEAKER_01"},
        {"words": [], "speaker": None},
    ]})
    assert segs[0].text == " The thalamus."
    assert [w.word for w in segs[0].words] == ["The"] and segs[0].speaker == "SPEAKER_01"
    assert segs[1].speaker is None
    assert len(notes) == 2


def test_merge_skips_when_segment_count_differs():
    segs = [make_segment("a b", 0, 1)]
    notes = whisperx_extra._merge(segs, {"segments": []})
    assert "skipped" in notes[0] and segs[0].words == []


def test_missing_environment_gives_install_hint(monkeypatch, tmp_path):
    monkeypatch.setattr(whisperx_extra, "DEFAULT_PYTHON", tmp_path / "none.exe")
    monkeypatch.setattr(whisperx_extra, "PROJECT_ROOT", tmp_path)
    with pytest.raises(WhisperXUnavailableError, match="-WithWhisperX"):
        refine_segments([make_segment("x", 0, 1)], Path("a.wav"), "en")


def test_diarization_without_token_explains_how_to_get_one(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(WhisperXUnavailableError, match="huggingface.co/settings/tokens"):
        refine_segments([make_segment("x", 0, 1)], Path("a.wav"), "en", diarize=True)


def test_pipeline_survives_missing_whisperx_and_records_it(tmp_path, cfg, monkeypatch, fake_ffmpeg):
    monkeypatch.setattr("src.transcriber.create_adapter", lambda *a, **k: FakeAdapter())
    monkeypatch.setattr(whisperx_extra, "DEFAULT_PYTHON", tmp_path / "none.exe")
    monkeypatch.setattr(whisperx_extra, "PROJECT_ROOT", tmp_path)
    media = write_silent_wav(tmp_path / "x.wav")
    [summary] = process_files([media], RunOptions(outputs_root=tmp_path / "o", language="en", align=True), cfg)
    assert summary.ok
    meta = json.loads((tmp_path / "o" / "x" / "metadata.json").read_text(encoding="utf-8"))
    assert meta["whisperx"]["status"] == "failed" and "setup_windows.ps1" in meta["whisperx"]["error"]


def test_diarization_is_off_by_default():
    assert RunOptions().diarize is False and RunOptions().align is False
