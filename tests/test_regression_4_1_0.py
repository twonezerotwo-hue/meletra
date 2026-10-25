"""The 4.1.0 failure, reproduced and fixed: a recording that changes language (Korean -> Hindi -> English) lost its languages because ONE language was chosen for all of it
(docs/v4_1_1/ROOT_CAUSE.md). With the multilingual scan switched off the pipeline behaves like 4.1.0 and loses them; with the default settings it keeps them all."""

from __future__ import annotations

import copy
import json

import pytest

from tests.test_multilingual_pipeline import SongAdapter, TEXT, run, song_backend  # noqa: F401 - the fake speech model and the runner

pytestmark = pytest.mark.usefixtures("fake_ffmpeg")


def spoken_languages(out) -> set[str]:
    segments = json.loads((out / "local_transcript.json").read_bytes().decode("utf-8"))["segments"]
    return {s["language"] for s in segments if s["text"].strip()}


def test_with_the_scan_off_the_sung_korean_part_is_lost_as_in_4_1_0(tmp_path, cfg, song_backend):
    old = copy.deepcopy(cfg)
    old["multilingual"] = {"scan": "off", "coverage": "off"}
    _summary, out = run(tmp_path, old, "")
    text = (out / "local_raw_transcript.txt").read_text(encoding="utf-8")
    assert TEXT["ko"] not in text                                                                               # the bug: the Korean part never reached the transcript ...
    assert "ko" not in spoken_languages(out)                                                                    # ... and no segment says it was Korean (the detector named another language)


def test_with_the_default_settings_every_language_is_kept(tmp_path, cfg, song_backend):
    _summary, out = run(tmp_path, cfg, "")
    assert spoken_languages(out) == {"ko", "hi", "en"}
    text = (out / "local_raw_transcript.txt").read_text(encoding="utf-8")
    assert all(t in text for t in TEXT.values())


def test_a_manual_language_is_respected_exactly_as_before(tmp_path, cfg, song_backend):
    from src.pipeline import RunOptions, process_files
    from tests.test_multilingual_pipeline import write_loud_wav

    media = write_loud_wav(tmp_path / "songs.wav")
    options = RunOptions(mode="quality", subjects=["general_medical"], outputs_root=tmp_path / "out", language="hi", output_language="")
    [summary] = process_files([media], options, cfg)
    assert summary.ok
    forced = {r.language for r in SongAdapter.requests if r.task == "transcribe" and r.language_plan is None and r.start_offset in (None, 0.0)}
    assert "hi" in forced and not any(r.language_plan for r in SongAdapter.requests)                            # the student's choice: no plan, no guessing
