"""The whole pipeline on a recording that changes language (Korean -> Hindi -> English) with a fake speech model that behaves like Whisper does on singing:

* the language detector names the wrong language for the Korean part (it never names Korean there), so the plan has to be verified by decoding;
* a forced WRONG language writes a short filler, the right language writes the sentence;
* the canonical source transcript keeps every language; the translation is made from that text (no speech model call), or - without the translation pack - by Whisper's translate
  task stretch by stretch in the stretch's own language.
"""

from __future__ import annotations

import json
import math
import wave
from pathlib import Path

import numpy as np
import pytest

from src.model_manager import ASRAdapter, Capabilities, DecodeRequest, ModelSpec
from src.multilingual import finalize as ml_finalize
from src.multilingual import source as source_mod
from src.pipeline import RunOptions, process_files
from src.schema import Segment, TranscriptionResult
from src.translation import METHOD_M2M100, METHOD_WHISPER
from src.translation.m2m100 import MODEL_FOLDER
from src.translation.service import TranslationService

pytestmark = pytest.mark.usefixtures("fake_ffmpeg")

TRUTH = [(0.0, 24.0, "ko"), (24.0, 48.0, "hi"), (48.0, 72.0, "en")]
TEXT = {"ko": "안녕하세요 여러분 오늘은 신경해부학에 대해 이야기해 봅시다", "hi": "नमस्ते दोस्तों आज हम तंत्रिका तंत्र के बारे में बात करेंगे",
        "en": "hello everybody today we will talk about the nervous system"}
ENGLISH_OF = {"ko": "Hello everyone, today we talk about neuroanatomy.", "hi": "Hello friends, today we talk about the nervous system.", "en": TEXT["en"]}


def truth_at(t: float) -> str:
    return next(lang for a, b, lang in TRUTH if a <= t < b)


class SongAdapter(ASRAdapter):
    backend_name = "fake"
    capabilities = Capabilities(hotwords=True, word_timestamps=False, vad=False, compute_types=True, multilingual=True, language_detection=True)
    requests: list[DecodeRequest] = []

    def __init__(self) -> None:
        super().__init__(ModelSpec("default", "faster_whisper", "large-v3"), device="cpu")

    def load(self, compute_type: str) -> None:
        self.compute_type = compute_type

    def close(self) -> None:
        pass

    # -- language identification -------------------------------------------------------------------------------------
    def detect_language_probs(self, wav, start, seconds=30.0, top=5):
        truth = truth_at(min(start + seconds / 2.0, 71.0))
        if truth == "ko":                                      # sung Korean: the detector is lost and names an Indian language it does not believe in
            return [("ta", 0.45), ("ml", 0.2), ("en", 0.1)]
        return [(truth, 0.95), ("en" if truth != "en" else "de", 0.03)]

    def detect_language_at(self, wav, start, seconds=30.0):
        top = self.detect_language_probs(wav, start, seconds)[0]
        return top[0], top[1]

    def score_languages(self, wav, start, seconds, languages, max_tokens=32):
        truth = truth_at(min(start + seconds / 2.0, 71.0))
        return {lang: ({"avg_logprob": -0.3, "no_speech_prob": 0.1, "text": TEXT[lang]} if lang == truth else {"avg_logprob": -1.2, "no_speech_prob": 0.1, "text": "la"}) for lang in languages}

    # -- decoding ----------------------------------------------------------------------------------------------------
    def transcribe(self, request: DecodeRequest) -> TranscriptionResult:
        SongAdapter.requests.append(request)
        lo = float(request.start_offset or 0.0)
        hi = lo + float(request.max_seconds) if request.max_seconds else float(request.total_duration or 72.0)
        segments = []
        t = lo
        while t < hi - 0.5:
            end = min(t + 6.0, hi)
            truth = truth_at(min(t + 0.1, 71.0))
            language = request.language or self.detect_language_probs(None, t, 6.0)[0][0]               # no language given: Whisper decides per window, and on singing it is lost
            right = language == truth
            if request.task == "translate":
                text = f"{ENGLISH_OF[truth]} ({int(t)})"
            else:
                text = f"{TEXT[language]} {int(t)}" if right else "la"
            segments.append(Segment(t, end, " " + text, language, "large-v3", "fake", avg_logprob=-0.2 if right else -1.4, no_speech_prob=0.01, compression_ratio=1.1))
            t = end
        return TranscriptionResult(segments, request.language or truth_at(min(lo + 0.1, 71.0)), 0.9, hi, "large-v3", "fake", self.compute_type)


class FakeEngine:
    id = "fake-m2m:1"

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int]] = []

    def translate(self, texts, source, target):
        self.calls.append((source, target, len(texts)))
        return [f"[{source}>{target}] " + ENGLISH_OF.get(source, "text") if target == "en" else f"[{source}>{target}] {t}" for t in texts]


@pytest.fixture
def song_backend(monkeypatch):
    SongAdapter.requests = []
    monkeypatch.setattr("src.transcriber.create_adapter", lambda *a, **k: SongAdapter())


def write_loud_wav(path: Path, seconds: float = 72.0, rate: int = 16000) -> Path:
    t = np.arange(int(seconds * rate)) / rate
    data = (0.25 * np.sin(2 * math.pi * 220 * t) * (0.7 + 0.3 * np.sin(2 * math.pi * 0.5 * t)) * 32767).astype(np.int16)        # loud, but not speech to the voice filter
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(data.tobytes())
    return path


def install_pack(folder: Path) -> Path:
    pack = folder / MODEL_FOLDER
    pack.mkdir(parents=True, exist_ok=True)
    for name in ("model.bin", "config.json", "sentencepiece.bpe.model", "shared_vocabulary.json"):
        (pack / name).write_bytes(b"x")
    return folder


def run(tmp_path: Path, cfg, output_language: str, models: Path | None = None, **extra):
    media = write_loud_wav(tmp_path / "songs.wav")
    options = RunOptions(mode="quality", subjects=["general_medical"], outputs_root=tmp_path / "out", language=None, output_language=output_language,
                         translation_model_dirs=[models] if models else [], **extra)
    [summary] = process_files([media], options, cfg)
    return summary, tmp_path / "out" / "songs"


@pytest.fixture
def engine(monkeypatch):
    fake = FakeEngine()
    real = TranslationService
    monkeypatch.setattr(ml_finalize, "TranslationService", lambda dirs, device="auto": real(dirs, engine=fake, device=device))
    return fake


def test_every_spoken_language_is_kept_in_order_and_the_timeline_comes_from_the_audio(tmp_path, cfg, song_backend, engine):
    summary, out = run(tmp_path, cfg, "en", install_pack(tmp_path / "models"))
    assert summary.ok
    segments = json.loads((out / "local_transcript.json").read_bytes().decode("utf-8"))["segments"]
    languages = [s["language"] for s in segments]
    assert [l for i, l in enumerate(languages) if i == 0 or l != languages[i - 1]] == ["ko", "hi", "en"]            # three languages, in the order they were spoken
    for s in segments:
        assert TEXT[s["language"]] in s["text"], (s["start"], s["language"], s["text"])                              # each part was decoded in ITS language, not in the dominant one
    plan = source_mod.read_file(out).engine["language_plan"]
    assert plan["mode"] in ("music", "multilingual") and {r["language"] for r in plan["runs"]} == {"ko", "hi", "en"}


def test_the_canonical_source_transcript_holds_all_languages_and_the_lecture_facts(tmp_path, cfg, song_backend, engine):
    summary, out = run(tmp_path, cfg, "en", install_pack(tmp_path / "models"))
    source = source_mod.read_file(out)
    assert source is not None and source.multilingual and set(source.detected_languages) == {"ko", "hi", "en"}
    assert all(s.language_known and s.language_source in source_mod.LANGUAGE_SOURCES and s.language_source != "unknown" for s in source.segments)
    metadata = json.loads((out / "metadata.json").read_bytes().decode("utf-8"))
    record = metadata["multilingual"]
    assert set(record["detected_languages"]) == {"ko", "hi", "en"} and record["input_language_mode"] == "auto" and record["source_transcript_revision"] == source.revision
    assert "en" in record["output_languages_available"] and record["translation_engine_versions"]["en"] == "fake-m2m:1"
    assert metadata["language"]["multilingual"] is not None and record["coverage"].get("checked") is True


def test_the_translation_is_made_from_the_saved_text_and_the_speech_model_is_not_used_for_it(tmp_path, cfg, song_backend, engine):
    summary, out = run(tmp_path, cfg, "en", install_pack(tmp_path / "models"))
    assert summary.translation["method"] == METHOD_M2M100 and summary.translation["output_language"] == "en"
    assert not any(r.task == "translate" for r in SongAdapter.requests)                                                 # no Whisper translate pass
    sources = {s for s, _t, _n in engine.calls}
    assert sources == {"ko", "hi"}                                                                                      # English was spoken in English: copied, never "translated"
    doc = json.loads((out / "translations" / "en.json").read_bytes().decode("utf-8"))
    assert doc["status"] == "complete" and {s["translation_status"] for s in doc["segments"]} == {"translated", "same_language"}
    assert (out / "translated_en_transcript.txt").is_file() and (out / "translated_en_transcript.srt").is_file()


def test_another_language_can_be_added_later_without_the_recording_or_the_speech_model(tmp_path, cfg, song_backend, engine):
    summary, out = run(tmp_path, cfg, "en", install_pack(tmp_path / "models"))
    calls_before = len(SongAdapter.requests)
    (tmp_path / "songs.wav").unlink()                                                                                   # the original media is gone
    service = TranslationService([tmp_path / "models"], engine=engine)
    doc = service.translate(out, "tr")
    assert doc.target == "tr" and doc.complete and len(SongAdapter.requests) == calls_before                              # the recogniser was not called
    assert {s for s, t, _n in engine.calls if t == "tr"} == {"ko", "hi", "en"}                                          # direct translation from every spoken language (no pivot through English)
    assert not any(t == "tr" and s == "en" and False for s, t, _n in engine.calls)
    assert sorted(service.stored_targets(out, any_engine=True)) == ["en", "tr"]                                         # the first translation is still there
    again = TranslationService([tmp_path / "models"], engine=engine).translate(out, "tr")
    assert again.created_at == doc.created_at                                                                           # the second request is answered from the cache


def test_without_the_pack_english_is_made_stretch_by_stretch_in_the_stretchs_own_language(tmp_path, cfg, song_backend):
    summary, out = run(tmp_path, cfg, "en")
    assert summary.ok and summary.translation["method"] == METHOD_WHISPER
    translate = [r for r in SongAdapter.requests if r.task == "translate"]
    assert sorted(r.language for r in translate) == ["hi", "ko"]                                                        # Korean as Korean, Hindi as Hindi, English not at all
    assert all(r.start_offset is not None and r.max_seconds for r in translate)
    lines = (out / "translated_en_transcript.txt").read_text(encoding="utf-8").splitlines()
    assert any(ENGLISH_OF["ko"] in l for l in lines) and any(ENGLISH_OF["hi"] in l for l in lines) and any(TEXT["en"] in l for l in lines)
    doc = json.loads((out / "translations" / "en.json").read_bytes().decode("utf-8"))
    assert doc["engine"].startswith("whisper-translate")


def test_another_language_without_the_pack_is_reported_and_the_transcript_is_complete(tmp_path, cfg, song_backend):
    summary, out = run(tmp_path, cfg, "de")
    assert summary.ok and summary.translation["method"] == "model_missing" and not (out / "translations").exists()
    assert set(source_mod.read_file(out).detected_languages) == {"ko", "hi", "en"}


def test_transcribing_the_same_recording_again_keeps_the_earlier_results(tmp_path, cfg, song_backend, engine):
    _summary, out = run(tmp_path, cfg, "en", install_pack(tmp_path / "models"))
    first = (out / "local_raw_transcript.txt").read_text(encoding="utf-8")
    revision = source_mod.read_file(out).revision
    (out / "study").mkdir()
    (out / "study" / "study_pack.json").write_text("{}", encoding="utf-8")
    _summary, out2 = run(tmp_path, cfg, "en", install_pack(tmp_path / "models"))
    assert out2 == out
    kept = sorted((out / "previous_versions").iterdir())
    assert len(kept) == 1
    assert (kept[0] / "local_raw_transcript.txt").read_text(encoding="utf-8") == first and (kept[0] / "translations" / "en.json").is_file()
    assert source_mod.read_file(kept[0]).revision == revision                                                      # the earlier canonical transcript is intact
    assert (out / "study" / "study_pack.json").is_file() and (out / "local_raw_transcript.txt").is_file()          # study material stays; the new results are in place


def test_on_a_gpu_the_speech_model_is_freed_before_the_translation_model_is_used(tmp_path):
    from src.transcriber import Transcriber
    from src.translation.outputs import TranslationResult

    folder = tmp_path / "lecture"
    folder.mkdir()
    rows = [source_mod.SourceSegment(1, 0, 5, TEXT["ko"], detected_language="ko", language_source="detector")]
    source_mod.write(folder, source_mod.SourceTranscript(lecture_id="l", segments=rows))
    order: list[str] = []

    class Engine:
        id = "e:1"

        def translate(self, texts, source, target):
            order.append("translate")
            return list(texts)

    real = ml_finalize.TranslationService
    ml_finalize.TranslationService = lambda dirs, device="auto": real(dirs, engine=Engine(), device=device)
    try:
        result = ml_finalize.run_text_translation(folder, TranslationResult(METHOD_M2M100, None, "en", "x", deferred=True), models_dirs=[], device="cuda", emit=lambda *a, **k: None,
                                                  stop_point=lambda: None, release=lambda: order.append("release"))
    finally:
        ml_finalize.TranslationService = real
    assert order == ["release", "translate"] and result.stored_by_service

    adapter = SongAdapter()
    transcriber = Transcriber.__new__(Transcriber)
    transcriber.adapter, transcriber._loaded = adapter, True
    closed = []
    adapter.close = lambda: closed.append(True)
    transcriber.unload()
    assert closed == [True] and transcriber._loaded is False                                                    # the next decode loads the model again


def test_a_chorus_in_a_song_is_not_taken_for_a_decoder_loop(tmp_path, cfg, monkeypatch):
    class Chorus(SongAdapter):
        def transcribe(self, request):
            result = super().transcribe(request)
            for s in result.segments:                                                                           # the same line every time: a chorus
                s.text = " " + TEXT.get(s.language or "en", TEXT["en"])
            return result

    SongAdapter.requests = []
    monkeypatch.setattr("src.transcriber.create_adapter", lambda *a, **k: Chorus())
    _summary, out = run(tmp_path, cfg, "")
    metadata = json.loads((out / "metadata.json").read_bytes().decode("utf-8"))
    assert metadata["long_audio"]["loop_repairs"] == []                                                         # no loop repair was started for the music plan
    text = (out / "local_raw_transcript.txt").read_text(encoding="utf-8")
    assert text.count(TEXT["hi"]) >= 3 and text.count(TEXT["ko"]) >= 2                                         # the repeated lines are all in the transcript


def test_nothing_is_sent_anywhere_while_a_recording_is_transcribed_and_translated(tmp_path, cfg, song_backend, engine, monkeypatch):
    """Privacy: the whole multilingual path (scan, verification, decoding, coverage, source transcript, translation, archive) runs with every network connection refused."""
    import socket

    attempts = []

    def refuse(self, address, *a, **k):
        attempts.append(address)
        raise OSError("network access is not allowed in this test")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    summary, out = run(tmp_path, cfg, "en", install_pack(tmp_path / "models"))
    assert summary.ok and (out / "translations" / "en.json").is_file() and attempts == []
    assert not (out / "sharing").exists()                                                                       # transcript sharing stays off (nothing is queued)


def test_the_quality_report_states_the_coverage_of_the_audible_time_next_to_the_voice_filter_figure(tmp_path, cfg, song_backend, engine, monkeypatch):
    monkeypatch.setattr("src.quality.compute_speech_ranges", lambda wav, vad_cfg: [(0.0, 3.0)])                # the voice filter hears three seconds of a sung recording
    summary, out = run(tmp_path, cfg, "en", install_pack(tmp_path / "models"))
    quality = json.loads((out / "quality_report.json").read_bytes().decode("utf-8"))["summary"]
    coverage = json.loads((out / "metadata.json").read_bytes().decode("utf-8"))["multilingual"]["coverage"]
    assert quality["audible_coverage"] == coverage["covered_share_after"] and quality["audible_seconds"] == coverage["audible_seconds"]
    assert quality["unrecovered_audible_gaps"] == len(coverage.get("unrecovered") or []) and "voice-activity filter" in quality["coverage_basis"]
