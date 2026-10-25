"""Two different recordings with the same file name must never overwrite each other's results."""

import json
import types
from pathlib import Path

import pytest

from src.model_manager import ModelSpec
from src.pipeline import OUTPUT_MARKER, job_key, mark_output_owner
from src.services import transcription_service as tsvc
from src.services.job_service import JobStore
from src.services.model_service import ModelService
from src.services.paths import AppPaths
from src.services.transcription_service import JobRequest, resolve_output_name, run_worker
from tests.conftest import write_silent_wav
from tests.test_expert_pipeline import LectureFake


def test_a_fresh_name_is_used_as_it_is(tmp_path):
    (tmp_path / "in").mkdir()
    source = write_silent_wav(tmp_path / "in" / "Biochemistry 3.wav", 1)
    assert resolve_output_name(tmp_path / "out", source) == "Biochemistry 3"


def test_an_empty_existing_folder_is_reused(tmp_path):
    (tmp_path / "out" / "lecture").mkdir(parents=True)
    source = write_silent_wav(tmp_path / "lecture.wav", 1)
    assert resolve_output_name(tmp_path / "out", source) == "lecture"


def test_the_same_recording_gets_its_folder_back(tmp_path):
    source = write_silent_wav(tmp_path / "lecture.wav", 2)
    folder = tmp_path / "out" / "lecture"
    folder.mkdir(parents=True)
    (folder / "local_raw_transcript.txt").write_text("old", encoding="utf-8")
    mark_output_owner(folder, source)
    assert (folder / OUTPUT_MARKER).read_text(encoding="utf-8") == job_key(source)
    assert resolve_output_name(tmp_path / "out", source) == "lecture"


def test_a_different_recording_with_the_same_name_gets_a_numbered_folder(tmp_path):
    for name in ("a", "b", "c"):
        (tmp_path / name).mkdir()
    first = write_silent_wav(tmp_path / "a" / "lecture.wav", 2)
    second = write_silent_wav(tmp_path / "b" / "lecture.wav", 3)
    third = write_silent_wav(tmp_path / "c" / "lecture.wav", 4)
    out = tmp_path / "out"
    for source in (first, second):
        name = resolve_output_name(out, source)
        folder = out / name
        folder.mkdir(parents=True)
        (folder / "local_raw_transcript.txt").write_text("text", encoding="utf-8")
        mark_output_owner(folder, source)
    assert (out / "lecture").is_dir() and (out / "lecture (2)").is_dir()
    assert resolve_output_name(out, first) == "lecture" and resolve_output_name(out, second) == "lecture (2)"
    assert resolve_output_name(out, third) == "lecture (3)"


def test_a_results_folder_without_an_owner_marker_is_never_overwritten(tmp_path):
    folder = tmp_path / "out" / "lecture"
    folder.mkdir(parents=True)
    (folder / "local_raw_transcript.txt").write_text("made by an older version or by hand", encoding="utf-8")
    source = write_silent_wav(tmp_path / "lecture.wav", 1)
    assert resolve_output_name(tmp_path / "out", source) == "lecture (2)"


def test_unsafe_characters_in_the_name_are_replaced_consistently(tmp_path):
    source = write_silent_wav(tmp_path / "Ders 3 kaydı şğü.wav", 1)
    source = source.rename(source.with_name("Ders 3 kaydı şğü.wav"))
    name = resolve_output_name(tmp_path / "out", source)
    assert name == "Ders 3 kaydı şğü"                                          # Turkish characters are kept
    from src.exporters import safe_name

    assert safe_name("a:b?c") == "a_b_c"                                       # characters Windows forbids are replaced


@pytest.fixture
def two_lectures(tmp_path, monkeypatch):
    paths = AppPaths(tmp_path / "data").ensure()
    monkeypatch.setattr(ModelService, "model_spec", lambda self, key: ModelSpec(key, "faster_whisper", "large-v3"))
    monkeypatch.setattr(tsvc, "STOP_POLL_SECONDS", 0.0)
    monkeypatch.setattr("src.transcriber.create_adapter", lambda *a, **k: LectureFake())
    (tmp_path / "x").mkdir()
    (tmp_path / "y").mkdir()
    sources = [write_silent_wav(tmp_path / "x" / "lecture.wav", 30), write_silent_wav(tmp_path / "y" / "lecture.wav", 40)]
    return types.SimpleNamespace(paths=paths, sources=sources, out=tmp_path / "out", store=JobStore(paths.jobs))


def run_job(env, source: Path) -> Path:
    jid = job_key(source)
    request = JobRequest(source=str(source), output_dir=str(env.out), job_id=jid, data_root=str(env.paths.root), language="en",
                         chunk_minutes=1.0, subject="neuroscience")
    request_file = env.paths.jobs / f"{jid}.request.json"
    request.save(request_file)
    env.store.create_or_get(jid, source)
    assert run_worker(request_file) == tsvc.EXIT_COMPLETED
    return Path(env.store.get(jid).summary["output_dir"])


@pytest.mark.usefixtures("fake_ffmpeg")
def test_two_lectures_called_the_same_end_up_in_two_folders_and_a_rerun_goes_back_to_its_own(two_lectures):
    env = two_lectures
    first = run_job(env, env.sources[0])
    second = run_job(env, env.sources[1])
    assert first.name == "lecture" and second.name == "lecture (2)" and first != second
    first_text = (first / "local_raw_transcript.txt").read_text(encoding="utf-8")
    assert first_text and (second / "local_raw_transcript.txt").is_file()
    again = run_job(env, env.sources[0])                                   # the same recording again: its own folder, not a third one
    assert again == first and not (env.out / "lecture (3)").exists()
    meta = json.loads((first / "metadata.json").read_text(encoding="utf-8"))
    assert meta["source"]["size_bytes"] == env.sources[0].stat().st_size   # and the first folder still describes the first file
