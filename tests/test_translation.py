"""Output Language: the decision, the translated files, Whisper's translate pass (English), the offline M2M100 engine (every other language), and the pipeline around them.
The spoken-language transcript is never touched."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.model_manager import ASRAdapter, Capabilities, DecodeRequest, ModelSpec
from src.pipeline import PauseRequested, RunOptions, process_files
from src.schema import Segment, TranscriptionResult
from src.translation import METHOD_FAILED, METHOD_M2M100, METHOD_MISSING, METHOD_NONE, METHOD_WHISPER
from src.translation import hook as translation_hook
from src.translation.m2m100 import MODEL_FOLDER, M2M100Engine, TranslationUnavailable, find_model, lang_token, translate_segments
from src.translation.outputs import TranslationResult, file_stem, read_record, translated_text_file
from src.translation.plan import decide
from src.translation.whisper_pass import translate_to_english
from tests.conftest import write_silent_wav

pytestmark = pytest.mark.usefixtures("fake_ffmpeg")


def seg(text, start=0.0, end=2.0, lang="tr"):
    return Segment(start, end, text, lang, "m", "fake")


# ------------------------------------------------------------------------------------------------------------------ the decision
@pytest.mark.parametrize("source, target, ready, method", [
    ("tr", "tr", True, METHOD_NONE), ("en", "en", False, METHOD_NONE), ("tr", None, True, METHOD_NONE), ("tr", "", True, METHOD_NONE),
    ("tr", "en", False, METHOD_WHISPER), ("de", "en", True, METHOD_M2M100), (None, "en", False, METHOD_WHISPER), (None, "en", True, METHOD_M2M100),
    ("en", "tr", True, METHOD_M2M100), ("hu", "de", True, METHOD_M2M100), ("en", "tr", False, METHOD_MISSING), (None, "tr", True, METHOD_M2M100), (None, "tr", False, METHOD_MISSING)])
def test_the_method_depends_on_the_languages_and_the_model(source, target, ready, method):
    plan = decide(source, target, ready)
    assert plan.method == method and plan.target == (target or None)


def test_the_text_translation_is_the_way_when_the_model_is_there_and_english_has_the_audio_fallback():
    assert decide("tr", "en", m2m_ready=False).method == METHOD_WHISPER and decide("tr", "en", m2m_ready=True).method == METHOD_M2M100
    for code in ("tr", "de", "hu", "ko", "zh", "hi", "nl", "fr", "es"):
        assert decide("en", code, m2m_ready=False).method == METHOD_MISSING and decide("en", code, m2m_ready=True).method == METHOD_M2M100


def test_a_recording_in_several_languages_is_translated_from_each_segments_own_language():
    spoken = ["ko", "hi", "en"]
    assert decide(None, "en", True, spoken).method == METHOD_M2M100 and decide(None, "tr", True, spoken).method == METHOD_M2M100
    assert decide(None, "en", False, spoken).method == METHOD_WHISPER and decide(None, "tr", False, spoken).method == METHOD_MISSING
    assert decide(None, "en", True, ["en"]).method == METHOD_NONE                                              # every recognised segment is already in the output language


# ------------------------------------------------------------------------------------------------------------------ files and record
def test_translated_files_sit_next_to_the_transcript_and_the_record_says_what_happened(tmp_path):
    result = TranslationResult(METHOD_WHISPER, "tr", "en", "x", [seg("Hello there.", 0, 2), seg("The heart has four chambers.", 2, 5)], seconds=3.2)
    files = result.write(tmp_path)
    assert [f.name for f in files] == ["translated_en_transcript.txt", "translated_en_transcript.srt", "translated_en_transcript.json"]
    assert (tmp_path / "translated_en_transcript.txt").read_text(encoding="utf-8") == "Hello there.\nThe heart has four chambers.\n"
    assert "00:00:00,000 --> " in (tmp_path / "translated_en_transcript.srt").read_text(encoding="utf-8")
    data = json.loads((tmp_path / "translated_en_transcript.json").read_text(encoding="utf-8"))
    assert data["language"] == "en" and data["source_language"] == "tr" and len(data["segments"]) == 2 and "unchanged" in data["note"]
    assert result.record()["files"] == ["translated_en_transcript.txt", "translated_en_transcript.srt", "translated_en_transcript.json"]
    assert translated_text_file(tmp_path, "en") == tmp_path / "translated_en_transcript.txt" and translated_text_file(tmp_path, "de") is None
    nothing = TranslationResult(METHOD_MISSING, "en", "tr", "the translation model is not installed")
    assert nothing.write(tmp_path / "x") == [] and nothing.record()["files"] == [] and not (tmp_path / "x").exists()
    assert file_stem("de") == "translated_de_transcript" and read_record(tmp_path) is None


# ------------------------------------------------------------------------------------------------------------------ Whisper's translate pass
class Recorder:
    """A transcriber that records the DecodeRequests it gets."""

    def __init__(self):
        self.requests = []

    def transcribe(self, request):
        self.requests.append(request)
        for s in (seg("Hello", 0, 2, "en"), seg("world", 2, 4, "en")):
            if request.on_segment is not None:
                request.on_segment(s)
        return TranscriptionResult([seg("Hello", 0, 2, "en"), seg("world", 2, 4, "en")], "en", 1.0, 4.0, "m", "fake", "int8")


def test_the_whisper_pass_asks_for_the_translate_task_without_hotwords():
    transcriber, progress = Recorder(), []
    cfg = {"decoding": {"log_prob_threshold": -1.0}, "vad": {"min_silence_duration_ms": 500}}
    segments = translate_to_english(transcriber, Path("lecture.wav"), "tr", {"beam_size": 5, "use_hotwords": True, "condition_on_previous_text": True, "word_timestamps": True},
                                    cfg, 4.0, progress=progress.append)
    [request] = transcriber.requests
    assert request.task == "translate" and request.language == "tr" and request.hotwords == [] and request.multilingual is False
    assert request.profile["condition_on_previous_text"] is False and request.profile["use_hotwords"] is False
    assert [s.text for s in segments] == ["Hello", "world"] and progress == [0.5, 1.0]
    translate_to_english(transcriber, Path("lecture.wav"), None, {"beam_size": 5}, cfg, 4.0)                      # a mixed-language recording: Whisper decides per window
    assert transcriber.requests[-1].language is None and transcriber.requests[-1].multilingual is True


def test_pause_during_the_translation_is_not_swallowed():
    class Pause(Exception):
        pass

    def stop():
        raise Pause()

    with pytest.raises(Pause):
        translate_to_english(Recorder(), Path("x.wav"), "tr", {}, {"decoding": {}, "vad": {}}, 4.0, should_stop=stop)


# ------------------------------------------------------------------------------------------------------------------ the offline model
class FakeTokenizer:
    def encode(self, text, out_type=str):
        return text.split()

    def decode(self, pieces):
        return " ".join(pieces)


class FakeTranslator:
    def __init__(self):
        self.calls = []

    def translate_batch(self, batch, target_prefix=None, **kwargs):
        self.calls.append((batch, target_prefix, kwargs))

        class Hyp:
            def __init__(self, tokens):
                self.hypotheses = [tokens]

        return [Hyp([target_prefix[i][0]] + [f"<{t}>" for t in sentence[1:-1]]) for i, sentence in enumerate(batch)]


def test_the_engine_tokenises_with_the_language_tokens_and_strips_the_target_prefix():
    translator = FakeTranslator()
    engine = M2M100Engine(None, translator=translator, tokenizer=FakeTokenizer())
    out = engine.translate(["the heart", "", "pumps blood"], "en", "tr")
    batch, prefix, kwargs = translator.calls[0]
    assert batch == [["__en__", "the", "heart", "</s>"], ["__en__", "pumps", "blood", "</s>"]] and prefix == [["__tr__"], ["__tr__"]]
    assert out == ["<the> <heart>", "", "<pumps> <blood>"] and kwargs["beam_size"] == 4                            # an empty segment stays empty, nothing is sent for it
    assert lang_token("zh") == "__zh__"


def test_auto_translation_retries_on_cpu_when_cuda_fails_during_first_inference(monkeypatch, tmp_path):
    class BrokenCuda:
        def translate_batch(self, *args, **kwargs):
            raise RuntimeError("cublas64_12.dll is not found")

    cpu = FakeTranslator()
    monkeypatch.setitem(sys.modules, "ctranslate2", SimpleNamespace(Translator=lambda *a, **kw: cpu))
    engine = M2M100Engine(tmp_path, device="auto", translator=BrokenCuda(), tokenizer=FakeTokenizer())
    engine._active_device = "cuda"
    assert engine.translate(["the heart"], "en", "tr") == ["<the> <heart>"]
    assert engine._active_device == "cpu" and len(cpu.calls) == 1


def test_translating_segments_keeps_every_time_and_reports_progress():
    segments = [seg(f"sentence {i}", i, i + 1, "en") for i in range(40)]
    engine = M2M100Engine(None, translator=FakeTranslator(), tokenizer=FakeTokenizer())
    seen, stops = [], []
    out = translate_segments(engine, segments, "en", "tr", progress=seen.append, should_stop=lambda: stops.append(1))
    assert [(s.start, s.end) for s in out] == [(s.start, s.end) for s in segments] and out[0].text == "<sentence> <0>" and out[0].language == "tr"
    assert seen[-1] == 1.0 and len(stops) == 3 and len(seen) == 3                                                  # 40 segments = 3 batches of 16


def test_a_missing_or_damaged_model_is_reported_not_crashed(tmp_path):
    assert find_model(tmp_path, None) is None
    folder = tmp_path / MODEL_FOLDER
    folder.mkdir()
    for name in ("model.bin", "config.json"):
        (folder / name).write_bytes(b"x")
    assert find_model(tmp_path) is None                                                                            # an incomplete folder is not a model
    (folder / "sentencepiece.bpe.model").write_bytes(b"x")
    (folder / "shared_vocabulary.json").write_text("[]", encoding="utf-8")
    assert find_model(tmp_path) == folder
    with pytest.raises(TranslationUnavailable):
        M2M100Engine(folder).load()                                                                                 # garbage files: a clear error, not a traceback
    with pytest.raises(TranslationUnavailable):
        M2M100Engine(None).load()


# ------------------------------------------------------------------------------------------------------------------ the hook
def run_hook(output, source, tmp_path, transcriber=None, mixed=False, model=False, **kw):
    folder = tmp_path / "models"
    if model:
        (folder / MODEL_FOLDER).mkdir(parents=True, exist_ok=True)
        for name in ("model.bin", "config.json", "sentencepiece.bpe.model", "shared_vocabulary.json"):
            (folder / MODEL_FOLDER / name).write_bytes(b"x")
    events = []
    result = translation_hook.run_for_pipeline(
        output_language=output, source_language=source, mixed=mixed, segments=[seg("Merhaba", 0, 2), seg("Kalp", 2, 4)], transcriber=transcriber or Recorder(), wav=Path("a.wav"),
        profile={"beam_size": 5}, cfg={"decoding": {}, "vad": {}}, duration=4.0, model_parent_dirs=[folder], emit=lambda kind, **d: events.append((kind, d)),
        stop_point=lambda: None, **kw)
    return result, events


def test_the_hook_does_nothing_for_the_same_language_and_says_when_the_model_is_missing(tmp_path):
    for output, source, method in (("tr", "tr", METHOD_NONE), (None, "tr", METHOD_NONE), ("de", "en", METHOD_MISSING)):
        result, events = run_hook(output, source, tmp_path)
        assert result.method == method and result.segments == [] and events == [] and not result.written
    assert run_hook("de", "en", tmp_path)[0].reason == "the translation model is not installed"


def test_the_hook_runs_whisper_for_english_without_the_model_and_leaves_the_text_translation_for_later_with_it(tmp_path):
    result, events = run_hook("en", "tr", tmp_path)
    assert result.method == METHOD_WHISPER and [s.text for s in result.segments] == ["Hello", "world"] and result.written
    assert [kind for kind, _ in events][:1] == ["translation_started"] and events[-1][0] == "translation_finished" and events[-1][1]["segments"] == 2

    result, events = run_hook("de", "en", tmp_path, model=True)                    # the model is installed: the text translation is made from the saved transcript, no audio is touched here
    assert result.method == METHOD_M2M100 and result.deferred and result.segments == [] and events == [] and not result.written
    result, _events = run_hook("en", "tr", tmp_path, model=True)
    assert result.method == METHOD_M2M100 and result.deferred and not Recorder().requests                          # English too: one direct translation, Whisper's translate task is only the fallback
    assert run_hook("de", "en", tmp_path, model=True, mixed=True)[0].deferred                                       # a mixed recording is translated segment by segment, each from its own language


def test_a_mixed_recording_without_the_model_is_translated_to_english_stretch_by_stretch(tmp_path):
    recorder = Recorder()
    segments = [seg("Merhaba", 0, 2, "tr"), seg("Hello there", 2, 4, "en"), seg("Kalp", 4, 6, "tr"), seg("Doctor", 6, 8, "en")]
    events = []
    result = translation_hook.run_for_pipeline(output_language="en", source_language=None, mixed=True, segments=segments, transcriber=recorder, wav=Path("a.wav"), profile={"beam_size": 5},
                                               cfg={"decoding": {}, "vad": {}}, duration=8.0, model_parent_dirs=[tmp_path], emit=lambda k, **d: events.append(k), stop_point=lambda: None)
    assert result.method == METHOD_WHISPER and [(r.language, r.task, r.start_offset) for r in recorder.requests] == [("tr", "translate", 0.0), ("tr", "translate", 3.7)]
    assert [s.start for s in result.segments] == sorted(s.start for s in result.segments)
    assert [s.text for s in result.segments if s.language_meta.get("source") == "copied"] == ["Hello there", "Doctor"]            # the English segments are copied as they were spoken


def test_a_translation_error_never_costs_the_spoken_transcript(tmp_path):
    class Broken(Recorder):
        def transcribe(self, request):
            raise RuntimeError("CUDA out of memory")

    result, events = run_hook("en", "tr", tmp_path, transcriber=Broken())
    assert result.method == METHOD_FAILED and "out of memory" in result.error and result.segments == [] and not result.written
    assert events[-1][0] == "translation_finished"


def test_a_pause_request_passes_through_the_hook(tmp_path):
    class Pausing(Recorder):
        def transcribe(self, request):
            raise PauseRequested()

    with pytest.raises(PauseRequested):
        run_hook("en", "tr", tmp_path, transcriber=Pausing())


# ------------------------------------------------------------------------------------------------------------------ the whole pipeline
class TurkishAdapter(ASRAdapter):
    backend_name = "fake"
    capabilities = Capabilities(hotwords=True, word_timestamps=False, vad=False, compute_types=True)
    requests: list = []

    def __init__(self) -> None:
        super().__init__(ModelSpec("default", "faster_whisper", "large-v3"))

    def load(self, compute_type: str) -> None:
        self.compute_type = compute_type

    def transcribe(self, request: DecodeRequest) -> TranscriptionResult:
        TurkishAdapter.requests.append(request)
        if request.task == "translate":
            segs = [Segment(0.0, 3.0, " The heart pumps blood.", "tr", "large-v3", "fake"), Segment(3.0, 6.0, " The lungs exchange gas.", "tr", "large-v3", "fake")]
        else:
            segs = [Segment(0.0, 3.0, " Kalp kani pompalar.", "tr", "large-v3", "fake", avg_logprob=-0.2, no_speech_prob=0.01, compression_ratio=1.1),
                    Segment(3.0, 6.0, " Akciger gaz alisverisi yapar.", "tr", "large-v3", "fake", avg_logprob=-0.2, no_speech_prob=0.01, compression_ratio=1.1)]
        return TranscriptionResult(segs, "tr", 0.98, 6.0, "large-v3", "fake", self.compute_type)

    def close(self) -> None:
        pass


@pytest.fixture
def turkish_backend(monkeypatch):
    TurkishAdapter.requests = []
    monkeypatch.setattr("src.transcriber.create_adapter", lambda *a, **k: TurkishAdapter())


def run_pipeline(tmp_path, cfg, output_language, language=None, **extra):
    media = write_silent_wav(tmp_path / "lecture.wav", seconds=2)
    options = RunOptions(mode="quality", subjects=["general_medical"], outputs_root=tmp_path / "out", language=language, output_language=output_language, **extra)
    [summary] = process_files([media], options, cfg)
    return summary, tmp_path / "out" / "lecture"


def test_a_turkish_lecture_with_output_english_keeps_the_turkish_and_adds_the_english(tmp_path, cfg, turkish_backend):
    summary, out = run_pipeline(tmp_path, cfg, "en")
    assert summary.ok and summary.translation["method"] == METHOD_WHISPER and summary.translation["output_language"] == "en"
    assert (out / "local_raw_transcript.txt").read_text(encoding="utf-8").splitlines()[0] == "Kalp kani pompalar."          # the spoken-language transcript: untouched
    assert (out / "translated_en_transcript.txt").read_text(encoding="utf-8").splitlines() == ["The heart pumps blood.", "The lungs exchange gas."]
    record = read_record(out)
    assert record["method"] == METHOD_WHISPER and record["source_language"] == "tr" and "translated_en_transcript.srt" in record["files"]
    tasks = [r.task for r in TurkishAdapter.requests]
    assert tasks[0] == "transcribe" and tasks.count("translate") == 1                                           # exactly one extra pass, the Whisper translate pass
    assert [r for r in TurkishAdapter.requests if r.task == "translate"][0].hotwords == []


def test_the_spoken_language_as_output_makes_no_translation_and_no_extra_decode(tmp_path, cfg, turkish_backend):
    summary, out = run_pipeline(tmp_path, cfg, "tr")
    assert summary.translation["method"] == METHOD_NONE and not list(out.glob("translated_*")) and all(r.task == "transcribe" for r in TurkishAdapter.requests)


def test_another_language_without_the_model_keeps_the_transcript_and_says_why(tmp_path, cfg, turkish_backend):
    summary, out = run_pipeline(tmp_path, cfg, "de")
    assert summary.ok and summary.translation["method"] == METHOD_MISSING and not list(out.glob("translated_*"))
    assert read_record(out)["reason"] == "the translation model is not installed" and (out / "local_raw_transcript.txt").is_file()


def test_no_output_language_means_the_old_behaviour(tmp_path, cfg, turkish_backend):
    summary, out = run_pipeline(tmp_path, cfg, None)
    assert summary.translation["method"] == METHOD_NONE and read_record(out)["method"] == METHOD_NONE


def test_the_job_record_learns_the_languages_and_the_translation(tmp_path):
    from src.services.job_service import JobStore

    store = JobStore(tmp_path / "jobs")
    store.create_or_get("j1", tmp_path / "a.mp4", output_language="en", state="running")
    store.apply_event("j1", {"type": "completed", "language": "tr", "audio_seconds": 2700.0, "translation": {"method": "whisper"}, "output_dir": str(tmp_path)})
    record = store.get("j1")
    assert (record.detected_language, record.translation, record.duration_seconds, record.output_language, record.state) == ("tr", "whisper", 2700.0, "en", "completed")


def test_the_study_material_follows_the_output_language(tmp_path):
    from src.services.settings_service import SettingsService
    from src.services.study_service import StudyService

    folder = tmp_path / "lecture"
    folder.mkdir()
    (folder / "metadata.json").write_text(json.dumps({"language": {"detected": "tr"}, "translation": {"output_language": "en", "method": "whisper"}}), encoding="utf-8")
    (folder / "local_transcript.json").write_text(json.dumps({"segments": [{"start": 0, "end": 2, "text": "Kalp kani pompalar."}]}), encoding="utf-8")
    service = StudyService(SettingsService(tmp_path / "s.json"), credentials=None)
    assert service.study_language(folder) == "tr" and service.study_transcript(folder).name == "local_transcript.json"       # no translated file yet: the spoken one
    (folder / "translated_en_transcript.json").write_text(json.dumps({"language": "en", "segments": [{"start": 0, "end": 2, "text": "The heart pumps blood."}]}), encoding="utf-8")
    assert service.study_language(folder) == "en" and service.study_transcript(folder).name == "translated_en_transcript.json"
    service.settings.update(study_language="tr")                                                                    # an explicit choice wins
    assert service.study_language(folder) == "tr" and service.study_transcript(folder).name == "local_transcript.json"
