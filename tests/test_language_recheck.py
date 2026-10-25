"""Segment language check: a passage in another language that the window scan cannot see is found from its own audio and read again in its language."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.model_manager import ASRAdapter, Capabilities, ModelSpec
from src.multilingual import recheck as rc
from src.multilingual import source as source_mod
from src.pipeline import RunOptions, process_files
from src.schema import Segment, TranscriptionResult
from tests.test_multilingual_pipeline import write_loud_wav

KO = "안녕하세요 여러분 오늘은 신경해부학에 대해 이야기해 봅시다"
EN = "hello everybody today we will talk about the nervous system"


def seg(start, end, text, language="en", logprob=-0.2, compression=1.2, **kw):
    return Segment(start, end, text, language, "m", "fake", avg_logprob=logprob, no_speech_prob=0.01, compression_ratio=compression, **kw)


class FakeTranscriber:
    """Audio truth: Korean between 40 and 52 s, English elsewhere."""

    def __init__(self, device="cuda"):
        self.adapter = SimpleNamespace(detect_language_probs=lambda *a, **k: [], device=device)
        self.detected, self.scored, self.requests = [], [], []

    def truth(self, t):
        return "ko" if 40.0 <= t < 52.0 else "en"

    def detect_language_windows(self, wav, starts, seconds=30.0, *a, **k):
        self.detected.append((starts[0], seconds))
        language = self.truth(starts[0] + seconds / 2.0)
        return [(starts[0], {language: 0.93, "de": 0.03})]

    def score_languages(self, wav, start, seconds, languages, max_tokens=24):
        self.scored.append((start, tuple(languages)))
        truth = self.truth(start + seconds / 2.0)
        return {l: ({"avg_logprob": -0.3, "no_speech_prob": 0.1, "text": KO if l == "ko" else EN} if l == truth else {"avg_logprob": -1.3, "no_speech_prob": 0.1, "text": "x"})
                for l in languages}

    def transcribe(self, request):
        self.requests.append(request)
        lo = request.start_offset
        text = KO if request.language == "ko" else EN
        return TranscriptionResult([seg(lo + 0.2, lo + request.max_seconds - 0.2, text, request.language)], request.language, 0.9, lo + request.max_seconds, "m", "fake", "int8")


CFG = {"decoding": {}, "vad": {}}


def english_with_an_echo():
    segs = [seg(i * 8.0, i * 8.0 + 7.0, f"{EN} {i}") for i in range(5)]                                   # 0-39 s: English
    segs.append(seg(41.0, 51.0, f"{EN} 0", "en", logprob=-0.18))                                         # the Korean passage, "decoded" as an echo of earlier English
    segs += [seg(53.0 + i * 8.0, 60.0 + i * 8.0, f"{EN} {i + 10}") for i in range(3)]
    return segs


def test_a_passage_in_another_language_is_found_from_its_own_audio_and_read_again_in_it():
    transcriber = FakeTranscriber()
    segs = english_with_an_echo()
    fixed, report = rc.recheck_languages(transcriber=transcriber, wav="x.wav", segments=segs, duration=90.0, profile={}, cfg=CFG, on_gpu=True)
    assert report["changed"] == 1 and report["changes"][0]["from"] == "en" and report["changes"][0]["to"] == "ko"
    korean = [s for s in fixed if s.language == "ko"]
    assert len(korean) == 1 and korean[0].text == KO and korean[0].language_meta["source"] == "recheck" and korean[0].language_meta["candidates"] == ["en"]
    assert [s.text for s in fixed if s.language == "en"] == [s.text for s in segs if s.start < 40 or s.start > 52]       # every other segment is exactly as it was
    [request] = transcriber.requests
    assert request.language == "ko" and request.hotwords == [] and request.profile["condition_on_previous_text"] is False


def test_a_detector_guess_that_the_likelihood_does_not_support_changes_nothing():
    class Doubtful(FakeTranscriber):
        def detect_language_windows(self, wav, starts, seconds=30.0, *a, **k):
            return [(starts[0], {"de": 0.7, "en": 0.2})]                                                   # names German for English speech (an accent)

        def score_languages(self, wav, start, seconds, languages, max_tokens=24):
            return {"en": {"avg_logprob": -0.3, "no_speech_prob": 0.1, "text": EN}, "de": {"avg_logprob": -1.2, "no_speech_prob": 0.1, "text": "x"}}

    transcriber = Doubtful()
    segs = [seg(i * 8.0, i * 8.0 + 7.0, f"{EN} {i}") for i in range(4)]
    fixed, report = rc.recheck_languages(transcriber=transcriber, wav="x.wav", segments=segs, duration=40.0, profile={}, cfg=CFG, on_gpu=True)
    assert report["changed"] == 0 and fixed == segs and transcriber.requests == []                         # the likelihood decides, not the detector alone


def test_a_cpu_run_checks_only_the_most_suspicious_segments():
    transcriber = FakeTranscriber(device="cpu")
    segs = [seg(i * 8.0, i * 8.0 + 7.0, f"{EN} number {i}") for i in range(40)]
    segs[17] = seg(17 * 8.0, 17 * 8.0 + 7.0, f"{EN} number 3", "en", compression=2.6)                 # an echo with odd statistics
    _fixed, report = rc.recheck_languages(transcriber=transcriber, wav="x.wav", segments=segs, duration=330.0, profile={}, cfg=CFG, on_gpu=False)
    assert report["checked"] == rc.CPU_BUDGET and report["candidates"] == 40
    assert any(abs(start - 17 * 8.0) < 4 for start, _l in transcriber.detected)                          # the suspicious one was among them


def test_segments_decided_by_evidence_or_too_short_are_not_checked():
    transcriber = FakeTranscriber()
    segs = [seg(0.0, 9.0, EN, language_meta={"source": "verified"}), seg(10.0, 11.0, "ok"), seg(12.0, 20.0, EN)]
    _fixed, report = rc.recheck_languages(transcriber=transcriber, wav="x.wav", segments=segs, duration=40.0, profile={}, cfg=CFG, on_gpu=True)
    assert report["candidates"] == 1 and len(transcriber.detected) == 1


def test_a_reading_in_the_wrong_writing_system_is_not_taken():
    class Broken(FakeTranscriber):
        def transcribe(self, request):
            result = super().transcribe(request)
            result.segments[0].text = "latin letters only"                                                  # Korean was asked for; Hangul did not come back
            return result

    segs = english_with_an_echo()
    fixed, report = rc.recheck_languages(transcriber=Broken(), wav="x.wav", segments=segs, duration=90.0, profile={}, cfg=CFG, on_gpu=True)
    assert report["changed"] == 0 and fixed == segs


def test_without_a_language_detector_nothing_is_attempted():
    transcriber = SimpleNamespace(adapter=SimpleNamespace(device="cuda"))
    segs = english_with_an_echo()
    fixed, report = rc.recheck_languages(transcriber=transcriber, wav="x.wav", segments=segs, duration=90.0, profile={}, cfg=CFG, on_gpu=True)
    assert fixed is segs and "cannot identify" in report["skipped"]


# ------------------------------------------------------------------------------------------------------------------ the whole pipeline
class LectureAdapter(ASRAdapter):
    """English lecture with ten seconds of Korean in the middle. The window detector sees only English (30 s windows are mostly English) so there is no plan; the single-language
    decode writes an echo of earlier English for the Korean part - exactly what the real recogniser did."""

    backend_name = "fake"
    capabilities = Capabilities(hotwords=True, word_timestamps=False, vad=False, compute_types=True, multilingual=True, language_detection=True)
    requests: list = []

    def __init__(self):
        super().__init__(ModelSpec("default", "faster_whisper", "large-v3"), device="cpu")

    def load(self, compute_type):
        self.compute_type = compute_type

    def close(self):
        pass

    @staticmethod
    def truth(t):
        return "ko" if 40.0 <= t < 50.0 else "en"

    def detect_language_probs(self, wav, start, seconds=30.0, top=5):
        if seconds >= 29.0:                                                                                 # a long window is dominated by English
            return [("en", 0.95), ("ko", 0.02)]
        language = self.truth(start + seconds / 2.0)
        return [(language, 0.93), ("de", 0.02)]

    def detect_language_at(self, wav, start, seconds=30.0):
        return self.detect_language_probs(wav, start, seconds)[0]

    def score_languages(self, wav, start, seconds, languages, max_tokens=24):
        truth = self.truth(start + seconds / 2.0)
        return {l: ({"avg_logprob": -0.3, "no_speech_prob": 0.1, "text": KO if l == "ko" else EN} if l == truth else {"avg_logprob": -1.3, "no_speech_prob": 0.1, "text": "x"})
                for l in languages}

    def transcribe(self, request):
        LectureAdapter.requests.append(request)
        lo = float(request.start_offset or 0.0)
        hi = lo + float(request.max_seconds) if request.max_seconds else float(request.total_duration or 80.0)
        segments, t = [], lo
        while t < hi - 0.5:
            end = min(t + 8.0, hi)
            truth = self.truth(t + 0.1)
            language = request.language or "en"
            if language == truth:
                text = f"{KO if language == 'ko' else EN} {int(t)}"
            else:
                text = f"{EN} 0"                                                                          # forced English on Korean: an echo of earlier English
            segments.append(Segment(t, end, " " + text, language, "large-v3", "fake", avg_logprob=-0.2, no_speech_prob=0.01, compression_ratio=1.2))
            t = end
        return TranscriptionResult(segments, request.language or "en", 0.9, hi, "large-v3", "fake", self.compute_type)


@pytest.fixture
def lecture_backend(monkeypatch):
    LectureAdapter.requests = []
    monkeypatch.setattr("src.transcriber.create_adapter", lambda *a, **k: LectureAdapter())


@pytest.mark.usefixtures("fake_ffmpeg")
def test_the_pipeline_recovers_a_brief_passage_in_another_language(tmp_path, cfg, lecture_backend):
    media = write_loud_wav(tmp_path / "lecture.wav", seconds=80.0)
    options = RunOptions(mode="quality", subjects=["general_medical"], outputs_root=tmp_path / "out", language=None, output_language="")
    [summary] = process_files([media], options, cfg)
    out = tmp_path / "out" / "lecture"
    assert summary.ok
    source = source_mod.read_file(out)
    assert set(source.detected_languages) == {"en", "ko"}
    korean = [s for s in source.segments if s.detected_language == "ko"]
    assert korean and all(KO in s.source_text for s in korean) and all(s.language_source == "recheck" for s in korean)
    metadata = json.loads((out / "metadata.json").read_bytes().decode("utf-8"))
    assert metadata["multilingual"]["language_check"]["changed"] >= 1


@pytest.mark.usefixtures("fake_ffmpeg")
def test_a_language_chosen_by_the_student_is_never_second_guessed(tmp_path, cfg, lecture_backend):
    media = write_loud_wav(tmp_path / "lecture.wav", seconds=80.0)
    options = RunOptions(mode="quality", subjects=["general_medical"], outputs_root=tmp_path / "out", language="en", output_language="")
    process_files([media], options, cfg)
    source = source_mod.read_file(tmp_path / "out" / "lecture")
    assert source.detected_languages == ["en"] and all(r.language == "en" for r in LectureAdapter.requests)
