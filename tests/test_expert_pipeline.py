"""The default (expert) path end to end with a fake ASR: probe -> context -> chunk decode -> reports.
Also: one hallucination/uncertainty engine, and the rule that no report ever rewrites the transcript."""

import json

import pytest

from src import quality
from src.model_manager import ASRAdapter, Capabilities, DecodeRequest, ModelSpec
from src.pipeline import RunOptions, process_files, resolve_subjects
from src.schema import Segment, TranscriptionResult
from tests.conftest import make_segment, write_silent_wav
from tests.test_pdf_context import make_pdf

pytestmark = pytest.mark.usefixtures("fake_ffmpeg")

LECTURE = [
    "Welcome back. Today the corticospinal tract and the spinothalamic tract. The CSF stays clear.",
    "Damage to the cortical spinal tract causes weakness. Lesions of the substantia nigra reduce dopamine.",
    "The periaqueductal gray, the medial lemniscus and the internal capsule are also important.",
]


class LectureFake(ASRAdapter):
    backend_name = "fake"
    capabilities = Capabilities(hotwords=True, vad=True, word_timestamps=True, compute_types=True)
    requests: list[DecodeRequest] = []

    def __init__(self) -> None:
        super().__init__(ModelSpec("default", "faster_whisper", "large-v3"))

    def load(self, compute_type: str) -> None:
        self.compute_type = compute_type

    def transcribe(self, request: DecodeRequest) -> TranscriptionResult:
        LectureFake.requests.append(request)
        total = request.total_duration
        end = min(total, request.start_offset + request.max_seconds) if request.max_seconds else total
        segs, t, i = [], request.start_offset, 0
        while t < end - 0.5:
            seg = Segment(t, min(t + 8.0, end), " " + LECTURE[i % len(LECTURE)], "en", "large-v3", "fake",
                          avg_logprob=-0.2, no_speech_prob=0.01, compression_ratio=1.2)
            segs.append(seg)
            if request.on_segment:
                request.on_segment(seg)
            if request.progress:
                request.progress(seg.end, total)
            t, i = t + 8.0, i + 1
        return TranscriptionResult(segs, "en", 0.99, total, "large-v3", "fake", self.compute_type,
                                   hotwords_applied=request.hotwords[:10], hotword_tokens=30)

    def close(self) -> None:
        pass


@pytest.fixture
def fake(monkeypatch):
    LectureFake.requests = []
    monkeypatch.setattr("src.transcriber.create_adapter", lambda *a, **k: LectureFake())


def run(tmp_path, cfg, seconds=60, **kw):
    media = write_silent_wav(tmp_path / "week2 lecture.wav", seconds=seconds)
    options = RunOptions(outputs_root=tmp_path / "out", quality_audit=False, **kw)
    [summary] = process_files([media], options, cfg)
    return summary, tmp_path / "out" / "week2 lecture"


def test_default_path_probes_the_start_detects_the_subject_and_uses_only_relevant_hotwords(tmp_path, cfg, fake):
    summary, out = run(tmp_path, cfg, language=None)                  # nothing but the file: no --subject, no flags
    probe, main = LectureFake.requests[0], LectureFake.requests[-1]
    assert probe.max_seconds and probe.hotwords == []                 # the probe decodes without bias ...
    assert probe.profile["condition_on_previous_text"] is False       # ... and without conditioning (measured: loops)
    assert main.hotwords and main.profile["condition_on_previous_text"] is False
    lowered = [h.lower() for h in main.hotwords]
    assert "corticospinal tract" in lowered and "propranolol" not in lowered
    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert meta["language"]["source"] == "probe" and meta["language"]["forced"] == "en"
    assert meta["subject"]["engine"] == "expert" and set(meta["subject"]["subjects"]) & {"neuroanatomy", "neuroscience"}
    assert meta["subject"]["token_budget"]["hotwords_token_budget"] <= 223
    assert "locally" in meta["privacy"]
    assert summary.deepgram == "not requested"


def test_medical_context_json_explains_the_decisions(tmp_path, cfg, fake):
    _, out = run(tmp_path, cfg, language=None)
    context = json.loads((out / "medical_context.json").read_text(encoding="utf-8"))
    assert context["engine"] == "expert" and context["areas"]["selected"]
    assert context["initial_context"]["ranked_terms"][0]["components"]["subject_match"] > 0
    assert context["token_budget"]["prompt_token_budget"] == cfg["context"]["prompt_token_budget"]
    assert "never edits the transcript" in context["note"]


def test_subject_flag_names_an_area_that_has_no_vocabulary_file_of_its_own(tmp_path, cfg, fake):
    assert resolve_subjects("cardiology", cfg) == ["cardiology"]
    run(tmp_path, cfg, language="en", subjects=["cardiology"], auto_topic=False)
    assert LectureFake.requests[-1].hotwords                          # physiology/pathology/pharmacology terms were offered


def test_legacy_engine_is_still_available_for_comparison(tmp_path, cfg, fake):
    _, out = run(tmp_path, cfg, language="en", subjects=["neuroscience"], auto_topic=False, context_engine="legacy")
    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert meta["subject"]["engine"] == "legacy" and LectureFake.requests[-1].hotwords


def test_no_report_rewrites_the_transcript_but_the_near_miss_is_reported_with_evidence(tmp_path, cfg, fake):
    make_pdf(tmp_path / "week2 lecture.pdf", ["The corticospinal tract is the main descending motor pathway. "
                                              "The corticospinal tract again. Cerebrospinal fluid (CSF) is clear."])
    _, out = run(tmp_path, cfg, language="en", auto_topic=False, subjects=["neuroscience"])
    raw = (out / "local_raw_transcript.txt").read_text(encoding="utf-8")
    assert "Damage to the cortical spinal tract causes weakness." in raw          # the ASR wording survives verbatim
    errors = (out / "possible_medical_errors.txt").read_text(encoding="utf-8")
    for label in ("Timestamp:", "Original phrase: cortical spinal tract", "Possible medical term: corticospinal tract",
                  "Similarity:", "Subject:", "PDF evidence:", "Reason:", "Confidence:"):
        assert label in errors, label
    assert "SUGGESTIONS only" in errors
    json.loads((out / "medical_entities.json").read_text(encoding="utf-8"))
    acronyms = (out / "medical_acronyms.txt").read_text(encoding="utf-8")
    assert "CSF" in acronyms and "cerebrospinal fluid" in acronyms.lower()    # learned from the slide definition
    clean = (out / "local_clean_transcript.txt").read_text(encoding="utf-8")
    assert "cortical spinal tract" in clean                                      # even the clean text does not guess


def test_every_documented_output_is_written(tmp_path, cfg, fake):
    _, out = run(tmp_path, cfg, language="en")
    for name in ("local_raw_transcript.txt", "local_clean_transcript.txt", "local_transcript_with_timestamps.txt",
                 "local_transcript.srt", "local_transcript.vtt", "local_transcript.json", "medical_terms_detected.txt",
                 "medical_entities.json", "possible_medical_errors.txt", "medical_acronyms.txt", "drug_gene_protein_report.txt",
                 "medical_context.json", "uncertain_segments.json", "metadata.json", "processing.log"):
        assert (out / name).is_file() and (out / name).stat().st_size > 0, name
    assert (out / "local_transcript.vtt").read_text(encoding="utf-8").startswith("WEBVTT")


# ------------------------------------------------------------------------------------------- one flag engine
def test_timestamp_anomalies_and_hallucination_signals_become_segment_flags():
    ok = make_segment("A normal sentence about the heart.", 0, 4)
    backwards = make_segment("Reversed times here.", 10, 6)
    stock = make_segment("Thanks for watching and please subscribe to the channel.", 12, 16)
    segments = [ok, backwards, stock]
    added = quality.add_engine_flags(segments, 20.0, [(0.0, 20.0)])
    assert added >= 2
    assert any(f.code == "timestamp_anomaly" for f in backwards.flags)
    assert any(f.code == "hallucination_stock_phrase" for f in stock.flags)
    assert ok.flags == [] and quality.add_engine_flags(segments, 20.0, [(0.0, 20.0)]) == 0      # idempotent


def test_text_over_non_speech_is_flagged_only_with_vad_evidence():
    seg = make_segment("A long invented sentence spoken over complete silence in the recording.", 0, 12)
    quality.add_engine_flags([seg], 20.0, [(15.0, 20.0)])
    assert any(f.code == "hallucination_text_in_non_speech" for f in seg.flags)
    other = make_segment("A long invented sentence spoken over complete silence in the recording.", 0, 12)
    quality.add_engine_flags([other], 20.0, [])
    assert not any(f.code == "hallucination_text_in_non_speech" for f in other.flags)


def test_the_single_engine_reports_every_signal_family_in_uncertain_segments_json(tmp_path, cfg, monkeypatch):
    class Messy(LectureFake):
        def transcribe(self, request):
            def seg(start, end, text, logprob=-0.2, no_speech=0.01):
                return Segment(start, end, text, "en", "large-v3", "fake", avg_logprob=logprob, no_speech_prob=no_speech,
                               compression_ratio=1.1)

            repeated = " The corticospinal tract descends through the internal capsule."
            segs = [seg(0, 4, " Low confidence words here.", logprob=-1.8), seg(4, 14, " Hmm.", no_speech=0.9),
                    seg(14, 18, repeated), seg(18, 22, repeated), seg(22, 26, " Thanks for watching."),
                    seg(40, 30, " Time runs backwards.")]
            return TranscriptionResult(segs, "en", 0.99, request.total_duration, "large-v3", "fake", self.compute_type)

    monkeypatch.setattr("src.transcriber.create_adapter", lambda *a, **k: Messy())
    media = write_silent_wav(tmp_path / "m.wav", seconds=60)
    process_files([media], RunOptions(outputs_root=tmp_path / "o", language="en", auto_topic=False), cfg)
    entries = json.loads((tmp_path / "o" / "m" / "uncertain_segments.json").read_text(encoding="utf-8"))
    codes = {r["code"] for e in entries for r in e["reasons"]}
    assert {"low_avg_logprob", "high_no_speech_prob", "low_speech_density", "duplicate_nearby",
            "hallucination_stock_phrase", "timestamp_anomaly"} <= codes


# ------------------------------------------------------------------------------------------- Turkish medical backend shape
def test_turkish_backend_without_hotword_support_still_gets_chunking_reports_and_turkish_terms(tmp_path, cfg, monkeypatch):
    class TurkishFake(LectureFake):
        capabilities = Capabilities(hotwords=False, word_timestamps=False, vad=True, segment_stats=False, compute_types=False)

        def transcribe(self, request):
            LectureFake.requests.append(request)
            seg = Segment(request.start_offset, request.start_offset + 5, " Kortikospinal trakt motor liflerini taşır.",
                          "tr", "turkish-medical", "fake")
            return TranscriptionResult([seg], "tr", 0.99, request.total_duration, "turkish-medical", "fake", None)

    LectureFake.requests = []
    monkeypatch.setattr("src.transcriber.create_adapter", lambda *a, **k: TurkishFake())
    media = write_silent_wav(tmp_path / "ders.wav", seconds=20)
    [summary] = process_files([media], RunOptions(outputs_root=tmp_path / "o", language="tr", quality_audit=False,
                                                  auto_topic=False, subjects=["neuroscience"]), cfg)
    request = LectureFake.requests[-1]
    assert summary.ok and request.hotwords == []                          # the capability check switched hotwords off
    assert request.profile["condition_on_previous_text"] is True          # no hotwords -> nothing to guard against
    out = tmp_path / "o" / "ders"
    terms = (out / "medical_terms_detected.txt").read_text(encoding="utf-8")
    assert "Kortikospinal trakt".lower() in terms.lower()                 # Turkish form recognised, not translated
    assert "Kortikospinal trakt" in (out / "local_raw_transcript.txt").read_text(encoding="utf-8")
    entities = json.loads((out / "medical_entities.json").read_text(encoding="utf-8"))
    assert any(e["language_form"] == "tr" and e["english"] == "corticospinal tract" for e in entities["entities"])
