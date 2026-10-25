"""Optional Deepgram backend, compare mode and consensus - tested with a fake HTTP opener (no network, no key spent)."""

import io
import json
import urllib.error

import pytest

from src import second_opinion as so
from src.consensus import compare_transcripts, consensus_text, disagreement_report_text
from src.deepgram_backend import (
    DeepgramAuthError, DeepgramQuotaError, DeepgramRequestError, DeepgramTranscriber, DeepgramUnavailable,
    classify_http_error, get_api_key, load_env_file, parse_response, suspicious_local_transcript,
)
from src.pipeline import RunOptions, process_files
from src.vocabulary import DEFAULT_TERMS_DIR, build_index
from tests.conftest import make_segment, write_silent_wav
from tests.test_pipeline import FakeAdapter

pytestmark = pytest.mark.usefixtures("fake_ffmpeg")


def dg_payload(words, transcript=None):
    """A minimal Deepgram /v1/listen response with utterances."""
    entries = [{"word": w.lower(), "punctuated_word": w, "start": start, "end": start + 0.4, "confidence": 0.98}
               for w, start in words]
    text = transcript or " ".join(w for w, _ in words)
    return {"metadata": {"request_id": "req-1", "duration": 6.0, "language": "en"},
            "results": {"channels": [{"alternatives": [{"transcript": text, "confidence": 0.98, "words": entries}]}],
                        "utterances": [{"start": entries[0]["start"], "end": entries[-1]["end"], "transcript": text,
                                        "confidence": 0.98, "words": entries}]}}


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def opener_returning(payload, seen=None):
    def opener(request, timeout=None):
        if seen is not None:
            seen.append(request)
        return FakeResponse(json.dumps(payload).encode("utf-8"))
    return opener


# ------------------------------------------------------------------------------------------- the client
def test_without_a_key_the_client_refuses_before_any_network_call():
    with pytest.raises(DeepgramUnavailable, match=".env"):
        DeepgramTranscriber(None)


def test_request_goes_to_nova_3_medical_with_keyterms_and_the_key_in_a_header_only(tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF....WAVE")
    seen = []
    client = DeepgramTranscriber("secret-key-123", opener=opener_returning(dg_payload([("Hello", 0.0)]), seen))
    result = client.transcribe_file(audio, "en", ["substantia nigra", "corticospinal tract"])
    request = seen[0]
    assert "model=nova-3-medical" in request.full_url and "keyterm=substantia+nigra" in request.full_url
    assert "secret-key-123" not in request.full_url                                  # never in the URL
    assert request.get_header("Authorization") == "Token secret-key-123"
    assert request.get_header("Content-type") == "audio/wav"
    assert result.segments and result.meta["model"] == "nova-3-medical"


def test_key_is_never_part_of_error_text_or_results(tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")

    def failing(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, io.BytesIO(b'{"err":"bad key"}'))

    client = DeepgramTranscriber("secret-key-123", opener=failing, sleep=lambda s: None)
    with pytest.raises(DeepgramAuthError) as info:
        client.transcribe_file(audio)
    assert "secret-key-123" not in str(info.value)


def test_http_errors_are_classified():
    assert isinstance(classify_http_error(401), DeepgramAuthError)
    assert isinstance(classify_http_error(402), DeepgramQuotaError)
    assert "429" in str(classify_http_error(429)) and isinstance(classify_http_error(500, "oops"), DeepgramRequestError)


def test_transient_failures_are_retried_then_succeed(tmp_path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    calls = []

    def flaky(request, timeout=None):
        calls.append(1)
        if len(calls) < 3:
            raise urllib.error.URLError("temporary")
        return FakeResponse(json.dumps(dg_payload([("Hi", 0.0)])).encode())

    client = DeepgramTranscriber("k", opener=flaky, sleep=lambda s: None, max_retries=3)
    assert client.transcribe_file(audio).meta["attempts"] == 3


def test_api_key_comes_from_the_environment_or_dotenv_and_placeholders_do_not_count(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# comment\nDEEPGRAM_API_KEY="abc123"\nOTHER=1\n', encoding="utf-8")
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    assert load_env_file(env)["DEEPGRAM_API_KEY"] == "abc123" and get_api_key(env) == "abc123"
    monkeypatch.setenv("DEEPGRAM_API_KEY", "from-env")
    assert get_api_key(env) == "from-env"
    monkeypatch.delenv("DEEPGRAM_API_KEY")
    env.write_text("DEEPGRAM_API_KEY=your_key_here\n", encoding="utf-8")
    assert get_api_key(env) is None
    assert get_api_key(tmp_path / "missing.env") is None


def test_dotenv_is_git_ignored_and_an_example_ships():
    from src import PROJECT_ROOT

    assert ".env" in (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "DEEPGRAM_API_KEY" in (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")


def test_response_without_utterances_is_grouped_from_words():
    payload = dg_payload([("The", 0.0), ("tract", 0.5), ("Next", 5.0), ("one", 5.4)])
    payload["results"].pop("utterances")
    result = parse_response(payload)
    assert [s.text for s in result.segments] == ["The tract", "Next one"] and result.language == "en"


# ------------------------------------------------------------------------------------------- when to ask
def test_default_never_asks_deepgram():
    assert so.should_ask_deepgram("local", False, 50, 100, 20, 20, {})[0] is False


def test_explicit_backends_always_ask():
    assert so.should_ask_deepgram("compare", False, 0, 100, 0, 0, {})[:2] == (True, "compare")
    assert so.should_ask_deepgram("deepgram", False, 0, 100, 0, 0, {})[:2] == (True, "backend")


def test_if_suspect_only_asks_when_the_local_transcript_looks_doubtful():
    thresholds = {"uncertain_ratio": 0.10, "loop_flags": 5}
    assert so.should_ask_deepgram("local", True, 2, 100, 0, 0, thresholds)[0] is False
    ask, by, why = so.should_ask_deepgram("local", True, 30, 100, 0, 0, thresholds)
    assert ask and by == "if-suspect" and "uncertain" in why
    assert suspicious_local_transcript(0, 100, 3, 3, thresholds)[0] is True


# ------------------------------------------------------------------------------------------- comparison / consensus
LOCAL = [make_segment("The cortical spinal tract and the substantia nigra.", 0.0, 4.0),
         make_segment("Patients receive propranolol daily.", 5.0, 8.0)]
DEEPGRAM = parse_response(dg_payload(
    [("The", 0.0), ("corticospinal", 0.4), ("tract", 0.9), ("and", 1.4), ("the", 1.8), ("substantia", 2.0),
     ("nigra.", 2.6), ("Patients", 5.0), ("receive", 5.4), ("propranolol", 5.9), ("daily.", 6.5)]))


def test_comparison_reports_medical_disagreement_and_consensus_without_overwriting_either_transcript():
    index = build_index("all", None, DEFAULT_TERMS_DIR)
    before = [s.text for s in LOCAL]
    comparison = compare_transcripts(LOCAL, DEEPGRAM.segments, index, pdf_terms=["corticospinal tract"],
                                     pdf_text="the corticospinal tract")
    assert [s.text for s in LOCAL] == before
    spelled = [d for d in comparison.disagreements if d.kind == "spelling"]
    assert spelled and spelled[0].dictionary_term.lower() == "corticospinal tract"
    assert spelled[0].status == "REVIEW_REQUIRED" and spelled[0].in_pdf
    assert {"dictionary", "pdf", "deepgram"} <= set(spelled[0].sources_for_candidate)
    agreed = {c.term.lower(): c for c in comparison.consensus}
    assert agreed["substantia nigra"].status == "HIGH_CONFIDENCE" and "propranolol" in agreed
    assert "corticospinal tract" not in agreed                       # the engines disagreed on it
    assert 0 < comparison.statistics["token_agreement"] < 1


def test_report_files_name_both_engines_and_say_nothing_was_overwritten():
    index = build_index("all", None, DEFAULT_TERMS_DIR)
    comparison = compare_transcripts(LOCAL, DEEPGRAM.segments, index)
    text = disagreement_report_text(comparison)
    assert "Local" in text and "Deepgram" in text and "Nothing was overwritten" in text
    assert "HIGH_CONFIDENCE" in consensus_text(comparison)
    json.dumps(comparison.to_dict())


def test_identical_transcripts_have_no_disagreements():
    index = build_index("all", None, DEFAULT_TERMS_DIR)
    same = compare_transcripts([make_segment("The substantia nigra.", 0, 3)],
                               parse_response(dg_payload([("The", 0.0), ("substantia", 0.5), ("nigra.", 1.0)])).segments, index)
    assert same.disagreements == [] and same.statistics["token_agreement"] == 1.0


# ------------------------------------------------------------------------------------------- through the pipeline
@pytest.fixture
def fake_backend(monkeypatch):
    monkeypatch.setattr("src.transcriber.create_adapter", lambda *a, **k: FakeAdapter())


def patch_deepgram(monkeypatch, payload, key="test-key"):
    seen = []
    monkeypatch.setattr(so, "get_api_key", lambda *a, **k: key)
    monkeypatch.setattr(so, "DeepgramTranscriber",
                        lambda api_key, **kw: DeepgramTranscriber(api_key, opener=opener_returning(payload, seen),
                                                                  sleep=lambda s: None, **kw))
    return seen


def test_local_is_the_default_and_nothing_is_uploaded(tmp_path, cfg, fake_backend, monkeypatch):
    seen = patch_deepgram(monkeypatch, dg_payload([("x", 0.0)]))
    media = write_silent_wav(tmp_path / "lec.wav", seconds=3)
    [summary] = process_files([media], RunOptions(outputs_root=tmp_path / "o", language="en"), cfg)
    assert summary.ok and summary.deepgram == "not requested" and seen == []
    meta = json.loads((tmp_path / "o" / "lec" / "metadata.json").read_text(encoding="utf-8"))
    assert "locally" in meta["privacy"] and meta["second_opinion"]["status"] == "not requested"


def test_compare_backend_writes_the_deepgram_and_comparison_files(tmp_path, cfg, fake_backend, monkeypatch):
    seen = patch_deepgram(monkeypatch, dg_payload([("The", 0.0), ("spinothalamic", 0.4), ("tract", 1.0),
                                                   ("carries", 1.6), ("pain.", 2.0)]))
    media = write_silent_wav(tmp_path / "lec.wav", seconds=6)
    [summary] = process_files([media], RunOptions(outputs_root=tmp_path / "o", language="en", backend="compare"), cfg)
    out = tmp_path / "o" / "lec"
    assert summary.ok and summary.deepgram == "ok" and len(seen) == 1
    for name in ("local_raw_transcript.txt", "deepgram_raw_transcript.txt", "deepgram_transcript.json",
                 "deepgram_raw_response.json", "medical_disagreement_report.txt", "comparison.json",
                 "consensus_medical_terms.txt"):
        assert (out / name).is_file(), name
    raw = (out / "local_raw_transcript.txt").read_text(encoding="utf-8")
    assert raw.startswith("The spino thalamic tract carries pain.")                   # local transcript untouched by Deepgram
    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert "DEEPGRAM WAS USED" in meta["privacy"]
    assert "test-key" not in (out / "metadata.json").read_text(encoding="utf-8")
    assert "test-key" not in (out / "processing.log").read_text(encoding="utf-8")


def test_compare_without_a_key_falls_back_to_local_only_and_says_so(tmp_path, cfg, fake_backend, monkeypatch):
    monkeypatch.setattr(so, "get_api_key", lambda *a, **k: None)
    media = write_silent_wav(tmp_path / "lec.wav", seconds=3)
    [summary] = process_files([media], RunOptions(outputs_root=tmp_path / "o", language="en", backend="compare"), cfg)
    assert summary.ok and summary.deepgram.startswith("skipped")
    out = tmp_path / "o" / "lec"
    assert (out / "local_raw_transcript.txt").is_file() and not (out / "deepgram_raw_transcript.txt").exists()


def test_deepgram_only_backend_skips_the_local_model(tmp_path, cfg, fake_backend, monkeypatch):
    patch_deepgram(monkeypatch, dg_payload([("Hello", 0.0), ("there.", 0.5)]))
    FakeAdapter.last_request = None
    media = write_silent_wav(tmp_path / "lec.wav", seconds=3)
    [summary] = process_files([media], RunOptions(outputs_root=tmp_path / "o", language="en", backend="deepgram"), cfg)
    out = tmp_path / "o" / "lec"
    assert summary.ok and FakeAdapter.last_request is None
    assert (out / "deepgram_raw_transcript.txt").read_text(encoding="utf-8").strip() == "Hello there."
    assert not (out / "local_raw_transcript.txt").exists()


def test_deepgram_backend_without_a_key_runs_the_local_model_instead(tmp_path, cfg, fake_backend, monkeypatch):
    monkeypatch.setattr(so, "get_api_key", lambda *a, **k: None)
    media = write_silent_wav(tmp_path / "lec.wav", seconds=3)
    [summary] = process_files([media], RunOptions(outputs_root=tmp_path / "o", language="en", backend="deepgram"), cfg)
    assert summary.ok and (tmp_path / "o" / "lec" / "local_raw_transcript.txt").is_file()
    assert summary.deepgram.startswith("skipped")


def test_if_suspect_stays_quiet_for_a_healthy_transcript(tmp_path, cfg, fake_backend, monkeypatch):
    seen = patch_deepgram(monkeypatch, dg_payload([("x", 0.0)]))
    media = write_silent_wav(tmp_path / "lec.wav", seconds=3)
    relaxed = {**cfg, "deepgram": {**cfg["deepgram"], "suspect_thresholds": {"uncertain_ratio": 0.99, "loop_flags": 99}}}
    [summary] = process_files([media], RunOptions(outputs_root=tmp_path / "o", language="en", deepgram_if_suspect=True), relaxed)
    assert summary.ok and seen == [] and summary.deepgram == "not requested"


def test_if_suspect_asks_when_the_local_transcript_is_doubtful(tmp_path, cfg, fake_backend, monkeypatch):
    seen = patch_deepgram(monkeypatch, dg_payload([("Hello", 0.0), ("there.", 0.5)]))
    media = write_silent_wav(tmp_path / "lec.wav", seconds=3)
    eager = {**cfg, "deepgram": {**cfg["deepgram"], "suspect_thresholds": {"uncertain_ratio": 0.01, "loop_flags": 99}}}
    [summary] = process_files([media], RunOptions(outputs_root=tmp_path / "o", language="en", deepgram_if_suspect=True), eager)
    assert summary.ok and len(seen) == 1 and summary.deepgram == "ok"      # the fake transcript has an uncertain segment
