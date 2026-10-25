"""The medical expert layer inside the pipeline: additive (raw / clean / SRT / VTT / JSON byte-identical), soft language lock, separate outputs."""

import hashlib
import json

import pytest

from src import language as lang
from tests.test_language_sampling import LangFake, lang_fake, run  # noqa: F401 - fixture + helpers of the language tests

MEDICAL_FILES = ("medical_corrected_transcript.txt", "medical_corrected_transcript_timestamped.txt", "medical_corrected_segments.json",
                 "correction_report.json", "medical_quality_report.json")
UNTOUCHED = ("local_raw_transcript.txt", "local_clean_transcript.txt", "local_transcript.srt", "local_transcript.vtt", "local_transcript.json")


def folder(tmp_path, name):
    path = tmp_path / name
    path.mkdir()
    return path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.usefixtures("fake_ffmpeg")
def test_the_medical_outputs_are_added_and_nothing_else_changes(tmp_path, cfg, lang_fake, monkeypatch):
    monkeypatch.setattr("src.quality.compute_speech_ranges", lambda wav, vad: [(0.0, 60.0)])
    summary, meta = run(folder(tmp_path, "with"), cfg)                                          # medical_expert is on by default
    out_with = tmp_path / "with" / "out" / "lecture"
    for name in MEDICAL_FILES:
        assert (out_with / name).is_file(), name
    assert meta["medical_expert"]["raw_transcript_modified"] is False and meta["medical_expert"]["total_segments"] == summary.segments
    LangFake.requests, LangFake.calls = [], 0
    summary2, meta2 = run(folder(tmp_path, "without"), cfg, medical_expert=False)
    out_without = tmp_path / "without" / "out" / "lecture"
    assert meta2["medical_expert"] is None and not any((out_without / n).exists() for n in MEDICAL_FILES)
    for name in UNTOUCHED:
        assert digest(out_with / name) == digest(out_without / name), name


@pytest.mark.usefixtures("fake_ffmpeg")
def test_weak_language_detection_is_soft_locked_instead_of_called_mixed(tmp_path, cfg, lang_fake, monkeypatch):
    monkeypatch.setattr("src.quality.compute_speech_ranges", lambda wav, vad: [(0.0, 60.0)])
    weak = [("sl", 0.55), ("hr", 0.5), ("en", 0.52), ("sr", 0.45), ("hu", 0.4), ("sl", 0.6)] * 4 + [("en", 0.5)]      # accented English
    LangFake.detections = weak
    summary, meta = run(folder(tmp_path, "soft"), cfg)
    sampling = meta["language"]["sampling"]
    assert meta["language"]["multilingual"] is False and meta["language"]["forced"] == "en"
    assert sampling["soft_lock"]["primary"] == "en" and sampling["soft_lock"]["arbitrated"] and not sampling["mixed"]
    assert meta["medical_expert"]["primary_language"] == "en"
    LangFake.requests, LangFake.calls = [], 0
    LangFake.detections = weak
    summary, meta = run(folder(tmp_path, "old"), cfg, medical_expert=False)                      # the old rule: the same detections mean "mixed"
    assert meta["language"]["multilingual"] is True


@pytest.mark.usefixtures("fake_ffmpeg")
def test_a_real_multilingual_recording_is_still_mixed_with_the_medical_layer(tmp_path, cfg, lang_fake, monkeypatch):
    monkeypatch.setattr("src.quality.compute_speech_ranges", lambda wav, vad: [(0.0, 60.0)])
    LangFake.detections = [("en", 1.0)] * 18 + [("hu", 0.97)] * 6 + [("en", 1.0)]
    summary, meta = run(folder(tmp_path, "real"), cfg)
    assert meta["language"]["multilingual"] is True and meta["language"]["sampling"]["soft_lock"]["multilingual"] is True
    assert (tmp_path / "real" / "out" / "lecture" / "medical_quality_report.json").is_file()


@pytest.mark.usefixtures("fake_ffmpeg")
def test_custom_terms_reach_the_pass(tmp_path, cfg, lang_fake, monkeypatch):
    monkeypatch.setattr("src.quality.compute_speech_ranges", lambda wav, vad: [(0.0, 60.0)])
    summary, meta = run(folder(tmp_path, "custom"), cfg, custom_medical_terms=["glycolysiss"])
    report = json.loads((tmp_path / "custom" / "out" / "lecture" / "medical_quality_report.json").read_text(encoding="utf-8"))
    assert report["total_segments"] == summary.segments and report["primary_language"] == "en"
