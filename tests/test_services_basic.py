"""Desktop services that hold state: paths, settings, credentials, jobs."""

import json
import os
import sys
from pathlib import Path

import pytest

from src.services.credential_service import (
    CredentialService, MemoryCredentialBackend, WindowsCredentialBackend,
)
from src.services.job_service import JobRecord, JobStore
from src.services.paths import AppPaths, PORTABLE_FLAG, SUBFOLDERS, default_output_dir, get_paths, resolve_user_data_root
from src.services.settings_service import SCHEMA_VERSION, Settings, SettingsService, migrate, sanitize


# ----------------------------------------------------------------------------------------------- paths
@pytest.mark.windows_platform
def test_user_data_lives_below_localappdata_by_default(tmp_path):
    root = resolve_user_data_root({"LOCALAPPDATA": str(tmp_path / "Local")}, app_dir=tmp_path / "app")
    assert root == tmp_path / "Local" / "MedicalLectureTranscriber"


def test_env_override_beats_everything(tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / PORTABLE_FLAG).write_text("")
    root = resolve_user_data_root({"MLT_DATA_DIR": str(tmp_path / "custom"), "LOCALAPPDATA": "x"}, app_dir=tmp_path / "app")
    assert root == tmp_path / "custom"


def test_portable_flag_keeps_everything_next_to_the_exe(tmp_path):
    app = tmp_path / "MedicalLectureTranscriber-Portable"
    app.mkdir()
    (app / PORTABLE_FLAG).write_text("")
    assert resolve_user_data_root({"LOCALAPPDATA": str(tmp_path / "Local")}, app_dir=app) == app / "data"


def test_ensure_creates_the_documented_folders(tmp_path):
    paths = get_paths({"MLT_DATA_DIR": str(tmp_path / "d")})
    for name in SUBFOLDERS:
        assert (paths.root / name).is_dir(), name
    assert paths.settings_file.parent == paths.settings and paths.app_log.parent == paths.logs
    assert paths.job_log("abc").name == "job-abc.log"


def test_model_search_order_user_then_app_then_extra(tmp_path):
    paths = AppPaths(tmp_path / "data")
    dirs = paths.model_search_dirs(["D:/models", ""], app_dir=tmp_path / "app")
    assert dirs == [tmp_path / "data" / "models", tmp_path / "app" / "models", Path("D:/models")]


def test_default_output_is_next_to_the_source(tmp_path):
    source = tmp_path / "lectures" / "bio.mp4"
    source.parent.mkdir()
    assert default_output_dir(source) == (tmp_path / "lectures" / "Medical Transcription").resolve()


@pytest.mark.windows_platform
def test_nothing_is_written_below_the_install_folder(tmp_path):
    app = tmp_path / "Program Files" / "MedicalLectureTranscriber"
    app.mkdir(parents=True)
    paths = get_paths({"LOCALAPPDATA": str(tmp_path / "Local")}, app_dir=app)
    assert not any(app.iterdir()) and str(paths.root).startswith(str(tmp_path / "Local"))


# ----------------------------------------------------------------------------------------------- settings
def test_defaults_are_quality_auto_local(tmp_path):
    s = SettingsService(tmp_path / "settings.json").settings
    assert (s.mode, s.language, s.subject, s.device, s.model_key) == ("quality", "auto", "auto", "auto", "whisper-large-v3")
    assert s.cloud_enabled is False and s.unload_ollama is False and s.medical_context and s.pdf_context
    assert s.schema_version == SCHEMA_VERSION


def test_save_is_atomic_and_round_trips(tmp_path):
    service = SettingsService(tmp_path / "settings" / "settings.json")
    service.update(language="tr", beam_size=3, theme="dark", output_dir="D:/out")
    assert not list((tmp_path / "settings").glob("*.tmp"))                      # no half-written leftovers
    again = SettingsService(tmp_path / "settings" / "settings.json").settings
    assert (again.language, again.beam_size, again.theme, again.output_dir) == ("tr", 3, "dark", "D:/out")
    assert json.loads((tmp_path / "settings" / "settings.json").read_text())["schema_version"] == SCHEMA_VERSION


def test_old_settings_files_are_migrated(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"output_folder": "E:/x", "model": "turkish_medical", "quality_mode": "fast"}))
    s = SettingsService(path).settings
    assert (s.output_dir, s.model_key, s.mode) == ("E:/x", "turkish-medical-large-v3", "fast")
    assert migrate({"model_key": "default"})["model_key"] == "whisper-large-v3"


def test_a_corrupt_file_never_stops_the_app(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{this is not json")
    s = SettingsService(path).settings
    assert s == Settings()
    assert (tmp_path / "settings.json.corrupt").is_file()


def test_invalid_values_fall_back_to_defaults_and_ranges_are_clamped():
    s = sanitize({"language": "klingon", "mode": 7, "beam_size": 999, "chunk_minutes": -3, "cpu_threads": "many",
                  "output_formats": ["pdf", "txt"], "theme": None, "cloud_enabled": "yes", "extra_model_dirs": "nope"})
    assert s.language == "auto" and s.mode == "quality" and s.beam_size == 20 and s.chunk_minutes == 1.0
    assert s.cpu_threads == 0 and s.output_formats == ["txt"] and s.theme == "dark" and s.cloud_enabled is False
    assert s.extra_model_dirs == []


def test_unknown_setting_names_are_rejected(tmp_path):
    with pytest.raises(KeyError, match="nope"):
        SettingsService(tmp_path / "s.json").update(nope=1)


def test_reset_keeps_first_run_flag(tmp_path):
    service = SettingsService(tmp_path / "s.json")
    service.update(first_run_done=True, language="tr")
    assert service.reset().language == "auto" and service.settings.first_run_done is True


def test_no_secret_field_exists_in_settings():
    assert not any("key" in name and "model" not in name for name in Settings().to_dict()), "keys belong in Credential Manager"


# ----------------------------------------------------------------------------------------------- credentials
def test_key_round_trip_and_masking():
    service = CredentialService(MemoryCredentialBackend(), target="t")
    assert not service.has_deepgram_key() and service.masked_deepgram_key() == ""
    service.set_deepgram_key("  abcd1234efgh5678  ")
    assert service.get_deepgram_key() == "abcd1234efgh5678" and service.has_deepgram_key()
    assert service.masked_deepgram_key().endswith("5678") and "abcd" not in service.masked_deepgram_key()
    assert service.delete_deepgram_key() and not service.has_deepgram_key()


@pytest.mark.parametrize("bad", ["", "   ", "abc def", "abc\ndef"])
def test_bad_keys_are_rejected(bad):
    with pytest.raises(ValueError):
        CredentialService(MemoryCredentialBackend()).set_deepgram_key(bad)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Credential Manager")
def test_windows_credential_manager_really_stores_and_removes_a_secret():
    backend = WindowsCredentialBackend()
    target = f"MedicalLectureTranscriber-test-{os.getpid()}"
    service = CredentialService(backend, target=target)
    try:
        assert service.secure_storage and not service.has_deepgram_key()
        service.set_deepgram_key("unit-test-secret-0123456789")
        assert service.get_deepgram_key() == "unit-test-secret-0123456789"
        assert backend.get(target) == "unit-test-secret-0123456789"
    finally:
        service.delete_deepgram_key()
    assert backend.get(target) is None


def test_the_key_is_never_part_of_the_settings_file(tmp_path):
    service = SettingsService(tmp_path / "s.json")
    CredentialService(MemoryCredentialBackend()).set_deepgram_key("top-secret-key-123")
    service.update(cloud_enabled=True)
    assert "top-secret" not in (tmp_path / "s.json").read_text(encoding="utf-8")


# ----------------------------------------------------------------------------------------------- jobs
@pytest.fixture
def store(tmp_path):
    return JobStore(tmp_path / "jobs", alive=lambda pid: pid == 4242)


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "lecture.mp4"
    path.write_bytes(b"x" * 1234)
    return path


def test_create_get_update(store, source):
    record = store.create_or_get("job1", source, model_key="whisper-large-v3", language="en", output_dir="out")
    assert record.state == "queued" and record.source_size == 1234 and record.source_name == "lecture.mp4"
    store.update("job1", progress=42.5, chunk=2, chunks=6)
    again = store.get("job1")
    assert again.progress == 42.5 and (again.chunk, again.chunks) == (2, 6) and again.language == "en"
    assert store.create_or_get("job1", source).progress == 42.5                     # same job id -> same job


def test_states_are_validated(store, source):
    store.create_or_get("job1", source)
    with pytest.raises(ValueError):
        store.update("job1", state="exploding")


def test_unfinished_jobs_and_interrupted_ones_are_found_after_a_crash(store, source, tmp_path):
    store.create_or_get("alive", source)
    store.mark_running("alive", pid=4242)
    other = tmp_path / "b.wav"
    other.write_bytes(b"y" * 10)
    store.create_or_get("dead", other)
    store.mark_running("dead", pid=999)                                              # worker died with the app
    store.create_or_get("done", other)
    store.mark_completed("done")
    store.create_or_get("bad", other)
    store.mark_failed("bad", "boom")
    states = {r.job_id: r.state for r in store.unfinished()}
    assert states == {"alive": "running", "dead": "paused", "bad": "failed"}
    assert "Interrupted" in store.get("dead").last_error


def test_the_same_file_finds_its_unfinished_job(store, source):
    store.create_or_get("job1", source)
    store.mark_paused("job1", position_seconds=300.0)
    found = store.find_for_source(source)
    assert found is not None and found.job_id == "job1" and found.position_seconds == 300.0
    source.write_bytes(b"changed size!!")                                            # a different file now
    assert store.find_for_source(source) is None
    store.mark_completed("job1")
    source.write_bytes(b"x" * 1234)
    assert store.find_for_source(source) is None                                     # finished jobs are not offered for resume


def test_pipeline_events_update_the_record(store, source):
    store.create_or_get("job1", source)
    store.apply_event("job1", {"type": "chunk_progress", "percent": 64.04, "processed": 3350.0, "total": 5241.0,
                               "chunk": 4, "chunks": 6})
    r = store.get("job1")
    assert (r.progress, r.chunk, r.chunks, r.position_seconds) == (64.0, 4, 6, 3350.0)
    store.apply_event("job1", {"type": "paused", "position_seconds": 3400.0})
    assert store.get("job1").state == "paused"
    store.apply_event("job1", {"type": "completed", "words": 7400, "output_dir": "x"})
    done = store.get("job1")
    assert done.state == "completed" and done.progress == 100.0 and done.summary["words"] == 7400
    store.apply_event("job1", {"type": "failed", "error": "disk full"})
    assert store.get("job1").last_error == "disk full" and store.get("job1").state == "failed"


def test_unknown_fields_in_a_job_file_are_ignored(store, source, tmp_path):
    store.create_or_get("job1", source)
    path = tmp_path / "jobs" / "job1.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["future_field"] = 1
    path.write_text(json.dumps(data), encoding="utf-8")
    assert isinstance(store.get("job1"), JobRecord)


def test_old_completed_jobs_can_be_pruned(store, source):
    for i in range(5):
        store.create_or_get(f"j{i}", source)
        store.mark_completed(f"j{i}")
    assert store.prune(keep_completed=2) == 3 and len(store.list(["completed"])) == 2


# ----------------------------------------------------------------------------------------------- job files that another process has open (Windows)
def test_a_job_update_survives_a_reader_holding_the_file(tmp_path, monkeypatch):
    import os as _os

    from src.services import job_service as js

    store = js.JobStore(tmp_path)
    source = tmp_path / "a.wav"
    source.write_bytes(b"x" * 10)
    store.create_or_get("job1", source, state="running")
    real = _os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] <= 3:
            raise PermissionError(13, "The process cannot access the file because it is being used by another process")
        real(src, dst)

    monkeypatch.setattr(js.os, "replace", flaky)
    monkeypatch.setattr(js, "REPLACE_PAUSE_SECONDS", 0.0)
    store.mark_completed("job1", {"words": 5})
    assert calls["n"] == 4 and store.get("job1").state == "completed"


def test_a_job_update_that_never_gets_through_raises_and_leaves_no_temp_file(tmp_path, monkeypatch):
    from src.services import job_service as js

    store = js.JobStore(tmp_path)
    source = tmp_path / "a.wav"
    source.write_bytes(b"x")
    store.create_or_get("job1", source, state="running")
    monkeypatch.setattr(js.os, "replace", lambda a, b: (_ for _ in ()).throw(PermissionError(13, "locked")))
    monkeypatch.setattr(js, "REPLACE_PAUSE_SECONDS", 0.0)
    with pytest.raises(PermissionError):
        store.mark_paused("job1")
    assert not list(tmp_path.glob("*.tmp")) and store.get("job1").state == "running"


def test_reading_a_job_that_is_being_replaced_retries(tmp_path, monkeypatch):
    from src.services import job_service as js

    store = js.JobStore(tmp_path)
    source = tmp_path / "a.wav"
    source.write_bytes(b"x")
    store.create_or_get("job1", source, state="paused")
    real = js.Path.read_text
    calls = {"n": 0}

    def flaky(self, *a, **k):
        if self.name == "job1.json" and calls["n"] < 2:
            calls["n"] += 1
            raise PermissionError(13, "locked")
        return real(self, *a, **k)

    monkeypatch.setattr(js.Path, "read_text", flaky)
    monkeypatch.setattr(js, "REPLACE_PAUSE_SECONDS", 0.0)
    assert store.get("job1").state == "paused" and calls["n"] == 2
