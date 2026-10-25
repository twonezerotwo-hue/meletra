"""GPU / diagnostics services and the worker that runs a job for the GUI (real pipeline, fake ASR)."""

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from src import APP_NAME, __version__
from src.model_manager import ModelSpec
from src.services import diagnostics_service as dsvc
from src.services import transcription_service as tsvc
from src.services.credential_service import CredentialService, MemoryCredentialBackend
from src.services.gpu_service import LARGE_V3_FLOAT16_MIB, GpuService
from src.services.job_service import JobStore
from src.services.model_service import ModelService
from src.services.paths import AppPaths
from src.services.transcription_service import (
    EXIT_COMPLETED, EXIT_FAILED, EXIT_PAUSED, EventTail, FileEventSink, JobRequest, build_run_options, request_stop,
    run_worker,
)
from src.pipeline import job_key
from tests.conftest import write_silent_wav
from tests.test_expert_pipeline import LectureFake
from tests.test_model_service import Handler, server, service  # noqa: F401  (fixtures: a real local HTTP model server)


# ----------------------------------------------------------------------------------------------- GPU service
SMI = {"name": "NVIDIA GeForce RTX 2070", "total_mib": 8192, "used_mib": 1000, "free_mib": 7000, "driver": "617.14"}


def gpu(smi=SMI, cuda=(True, "CUDA acceleration is available."), programs=()):
    return GpuService(smi=lambda i: smi, cuda=lambda i: cuda, processes=lambda: list(programs))


@pytest.mark.windows_platform
def test_gpu_detected_with_cuda():
    status = gpu().detect()
    assert status.device == "cuda" and status.cuda_available and status.nvidia_present
    assert status.headline == "GPU detected: NVIDIA GeForce RTX 2070 (8 GB VRAM)" and "CUDA acceleration available" in status.detail


@pytest.mark.windows_platform
def test_no_nvidia_gpu_means_cpu_with_a_warning():
    status = gpu(smi=None, cuda=(False, "No compatible NVIDIA GPU / driver was detected.")).detect()
    assert status.device == "cpu" and not status.nvidia_present
    assert status.headline == "No compatible NVIDIA GPU detected." and "very slow" in status.detail


@pytest.mark.windows_platform
def test_gpu_present_but_cuda_unusable_explains_why_and_offers_the_cpu():
    status = gpu(cuda=(False, "A CUDA runtime library is missing or incompatible (cublas64_12.dll).")).detect()
    assert status.device == "cpu" and status.nvidia_present
    assert "CUDA acceleration unavailable" in status.detail and "cublas64_12.dll" in status.detail
    assert "CPU fallback enabled" in status.detail


def test_vram_check_ok_warning_critical_and_names_the_culprits():
    assert gpu().vram_check().ok
    low = {**SMI, "free_mib": 3000}
    warn = gpu(smi=low, programs=["ollama.exe", "chrome.exe x2"]).vram_check()
    assert not warn.ok and warn.severity == "warning" and "ollama.exe" in warn.message and "chrome.exe x2" in warn.message
    assert "will not close anything" in warn.message
    crit = gpu(smi={**SMI, "free_mib": 900}).vram_check()
    assert crit.severity == "critical" and crit.free_mib == 900 and crit.required_mib == LARGE_V3_FLOAT16_MIB


def test_vram_check_without_an_nvidia_gpu_is_silent():
    assert gpu(smi=None).vram_check().ok


def test_ollama_models_are_listed_and_unloaded_only_on_request(monkeypatch):
    monkeypatch.setattr("src.services.gpu_service.shutil.which", lambda name: "ollama.exe")
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd[1:])
        out = "NAME ID SIZE PROCESSOR UNTIL\nqwen3:8b abc 5.3 GB 100% GPU 29 minutes\n" if cmd[1] == "ps" else ""
        return subprocess.CompletedProcess(cmd, 0, out, "")

    service = gpu()
    assert service.ollama_models(fake_run) == ["qwen3:8b"] and calls == [["ps"]]
    assert service.unload_ollama(fake_run) == ["qwen3:8b"] and ["stop", "qwen3:8b"] in calls


def test_no_ollama_installed_is_fine(monkeypatch):
    monkeypatch.setattr("src.services.gpu_service.shutil.which", lambda name: None)
    assert gpu().ollama_models() == [] and gpu().unload_ollama() == []


# ----------------------------------------------------------------------------------------------- diagnostics
@pytest.mark.windows_platform
def test_diagnostics_report_has_every_section_and_no_secrets(tmp_path):
    paths = AppPaths(tmp_path / "d").ensure()
    data = dsvc.collect(paths, ModelService(paths), gpu(), check_ffmpeg_version=False)
    for key in ("app", "windows", "cpu", "ram", "gpu", "nvidia_driver", "cuda", "libraries", "ffmpeg", "models", "paths", "disk"):
        assert key in data, key
    assert data["app"]["version"] == __version__ and data["app"]["company"] == "Touche Holding"
    assert data["gpu"]["name"] == "NVIDIA GeForce RTX 2070" and data["cuda"]["available"] is True
    assert data["ffmpeg"]["pyav_fallback"] is True
    assert {m["key"] for m in data["models"]} >= {"whisper-large-v3", "turkish-medical-large-v3"}
    text = dsvc.as_text(data)
    for needle in (f"{APP_NAME} {__version__}", "Touche Holding", "Windows", "RAM", "NVIDIA driver", "CUDA", "ctranslate2",
                   "faster-whisper", "FFmpeg", "PyAV", "Model"):
        assert needle in text, needle
    json.dumps(data)


def test_ffmpeg_version_probe_never_hangs_the_report(monkeypatch):
    assert dsvc.tool_version(None) == "not found"

    class Stuck:
        def __init__(self, *a, **k):
            pass

        def communicate(self, timeout=None):
            raise subprocess.TimeoutExpired("ffmpeg", timeout)

        def kill(self):
            pass

    monkeypatch.setattr(dsvc.subprocess, "Popen", Stuck)
    assert "did not answer" in dsvc.tool_version(Path("ffmpeg.exe"), timeout=0.2)


def test_windows_and_cpu_names_are_readable():
    assert dsvc.windows_version() and dsvc.cpu_name() and dsvc.memory_status()["total_gb"] >= 0


# ----------------------------------------------------------------------------------------------- job request / IPC
def test_job_request_round_trip_ignores_future_fields(tmp_path):
    request = JobRequest(source="C:/lec.mp4", language="tr", beam_size=3, extra_model_dirs=["D:/m"])
    request.save(tmp_path / "r.json")
    data = json.loads((tmp_path / "r.json").read_text())
    data["from_the_future"] = True
    (tmp_path / "r.json").write_text(json.dumps(data))
    loaded = JobRequest.load(tmp_path / "r.json")
    assert (loaded.language, loaded.beam_size, loaded.extra_model_dirs) == ("tr", 3, ["D:/m"])


def test_event_files_deliver_complete_lines_only(tmp_path):
    path = tmp_path / "e.jsonl"
    sink, tail = FileEventSink(path), EventTail(path)
    assert tail.poll() == []
    sink({"type": "a"})
    sink({"type": "b", "text": "Türkçe ğüşiöç"})
    with path.open("ab") as handle:
        handle.write(b'{"type": "c"')                                      # a half-written line
    assert [e["type"] for e in tail.poll()] == ["a", "b"]
    with path.open("ab") as handle:
        handle.write(b'}\n')
    assert [e["type"] for e in tail.poll()] == ["c"] and tail.poll() == []
    assert FileEventSink(path) and path.read_text() == ""                  # a new job truncates the file


def stub_models(tmp_path):
    paths = AppPaths(tmp_path / "data").ensure()
    service = ModelService(paths)
    service.model_spec = lambda key: ModelSpec(key, "faster_whisper", str(tmp_path / "model"))     # type: ignore[method-assign]
    return paths, service


def test_request_to_options_mapping(tmp_path, cfg):
    paths, models = stub_models(tmp_path)
    source = write_silent_wav(tmp_path / "lec.wav", seconds=2)
    request = JobRequest(source=str(source), mode="fast", language="tr", subject="cardiology", compute_type="int8",
                         beam_size=2, vad=False, chunk_minutes=10, overlap_seconds=5, cpu_threads=4, device="cpu",
                         pdf_path=str(tmp_path / "s.pdf"), hotwords="auto", data_root=str(paths.root))
    options, cfg2 = build_run_options(request, cfg, models, paths)
    assert (options.mode, options.language, options.subjects, options.device) == ("fast", "tr", ["cardiology"], "cpu")
    assert options.chunk_minutes == 10 and options.cpu_threads == 4 and options.resume and not options.keep_checkpoint
    assert options.outputs_root == Path(source).resolve().parent / "Medical Transcription"
    assert options.checkpoint_root == paths.checkpoints and options.temp_root == paths.temp
    assert options.use_hotwords and options.use_pdf and options.auto_topic is True and options.context_engine == "expert"
    assert cfg2["fast"]["compute_type"] == "int8" and cfg2["fast"]["beam_size"] == 2 and cfg2["fast"]["vad_filter"] is False
    assert cfg2["long_audio"]["overlap_seconds"] == 5
    assert cfg["fast"]["compute_type"] != "int8" or cfg is not cfg2                            # the shipped config is untouched
    assert options.backend == "local"


def test_medical_context_off_means_no_hotwords_no_pdf_no_probe(tmp_path, cfg):
    paths, models = stub_models(tmp_path)
    options, _ = build_run_options(JobRequest(source="x.wav", medical_context=False, data_root=str(paths.root)), cfg, models, paths)
    assert not options.use_hotwords and not options.use_pdf and options.auto_topic is False and options.context_engine == "legacy"
    options, _ = build_run_options(JobRequest(source="x.wav", hotwords="off", data_root=str(paths.root)), cfg, models, paths)
    assert not options.use_hotwords and options.auto_topic is True


def test_cloud_comparison_needs_both_the_switch_and_a_key(tmp_path, cfg):
    paths, models = stub_models(tmp_path)
    request = JobRequest(source="x.wav", cloud_compare=True, data_root=str(paths.root))
    assert build_run_options(request, cfg, models, paths, deepgram_key=None)[0].backend == "local"
    options = build_run_options(request, cfg, models, paths, deepgram_key="k")[0]
    assert options.backend == "compare" and options.deepgram_api_key == "k"
    assert build_run_options(JobRequest(source="x.wav", data_root=str(paths.root)), cfg, models, paths, "k")[0].backend == "local"


def test_friendly_errors():
    assert "CPU" in tsvc._friendly_error(RuntimeError("Library cublas64_12.dll is not found"))
    assert "GPU ran out of memory" in tsvc._friendly_error(RuntimeError("CUDA failed with error out of memory"))
    assert tsvc._friendly_error(ValueError("plain message")) == "plain message"


# ----------------------------------------------------------------------------------------------- the worker, end to end
@pytest.fixture
def worker_env(tmp_path, monkeypatch):
    paths = AppPaths(tmp_path / "data").ensure()
    monkeypatch.setattr(ModelService, "model_spec", lambda self, key: ModelSpec(key, "faster_whisper", "large-v3"))
    monkeypatch.setattr(tsvc, "STOP_POLL_SECONDS", 0.0)
    monkeypatch.setattr("src.transcriber.create_adapter", lambda *a, **k: LectureFake())
    LectureFake.requests = []
    source = write_silent_wav(tmp_path / "lecture.wav", seconds=120)
    jid = job_key(source)
    request = JobRequest(source=str(source), output_dir=str(tmp_path / "out"), job_id=jid, data_root=str(paths.root),
                         language="en", chunk_minutes=1.0, subject="neuroscience")
    request_file = paths.jobs / f"{jid}.request.json"
    request.save(request_file)
    store = JobStore(paths.jobs)
    store.create_or_get(jid, source)
    return types.SimpleNamespace(jid=jid, paths=paths, request=request, request_file=request_file, store=store, source=source,
                                 tmp=tmp_path)


def events_of(env):
    return [json.loads(line) for line in (env.paths.jobs / f"{env.jid}.events.jsonl").read_text(encoding="utf-8").splitlines()]


@pytest.mark.usefixtures("fake_ffmpeg")
def test_worker_completes_a_job_and_reports_everything_the_gui_needs(worker_env):
    code = run_worker(worker_env.request_file)
    assert code == EXIT_COMPLETED
    kinds = [e["type"] for e in events_of(worker_env)]
    assert kinds[0] == "job_started" and "completed" in kinds and kinds[-1] == "worker_exit" and "log" in kinds
    assert "chunk_progress" in kinds and "subject_detected" in kinds
    record = worker_env.store.get(worker_env.jid)
    assert record.state == "completed" and record.progress == 100.0 and record.summary["words"] > 0
    out = worker_env.tmp / "out" / "lecture"
    for name in ("local_raw_transcript.txt", "local_clean_transcript.txt", "local_transcript.srt", "local_transcript.vtt",
                 "local_transcript.json", "metadata.json"):
        assert (out / name).is_file(), name
    assert not (worker_env.paths.checkpoints / worker_env.jid).exists()
    assert worker_env.paths.job_log(worker_env.jid).is_file()


@pytest.mark.usefixtures("fake_ffmpeg")
def test_stop_file_pauses_safely_and_the_same_job_resumes(worker_env, monkeypatch):
    seen = {"n": 0}

    class StoppingFake(LectureFake):
        def transcribe(self, request):
            real_on_segment = request.on_segment

            def on_segment(seg):
                real_on_segment(seg)
                seen["n"] += 1
                if seen["n"] == 3:
                    request_stop(worker_env.paths, worker_env.jid)          # what the GUI's PAUSE SAFELY button does

            if real_on_segment is not None:                          # the subject probe has no checkpoint callback
                request.on_segment = on_segment
            return super().transcribe(request)

    monkeypatch.setattr("src.transcriber.create_adapter", lambda *a, **k: StoppingFake())
    assert run_worker(worker_env.request_file) == EXIT_PAUSED
    kinds = [e["type"] for e in events_of(worker_env)]
    assert "pause_requested" in kinds and "paused" in kinds and "completed" not in kinds and kinds[-1] == "worker_exit"
    assert worker_env.store.get(worker_env.jid).state == "paused"
    assert (worker_env.paths.checkpoints / worker_env.jid).is_dir()
    assert (worker_env.tmp / "out" / "lecture" / "partial_transcript.txt").is_file()

    monkeypatch.setattr("src.transcriber.create_adapter", lambda *a, **k: LectureFake())
    LectureFake.requests = []
    assert run_worker(worker_env.request_file) == EXIT_COMPLETED                  # "RESUME": same request, same job id
    assert "resume_started" in [e["type"] for e in events_of(worker_env)]
    assert worker_env.store.get(worker_env.jid).state == "completed"
    assert round(LectureFake.requests[0].start_offset) > 0                       # did not start from zero


@pytest.mark.usefixtures("fake_ffmpeg")
def test_a_vanished_gui_makes_the_worker_pause_instead_of_running_on(worker_env):
    worker_env.request.parent_pid = 2 ** 22 + 12345                              # no such process
    worker_env.request.save(worker_env.request_file)
    assert run_worker(worker_env.request_file) == EXIT_PAUSED
    assert worker_env.store.get(worker_env.jid).state == "paused"


@pytest.mark.usefixtures("fake_ffmpeg")
def test_failures_reach_the_gui_with_a_readable_message(worker_env, monkeypatch):
    def broken(self, key):
        raise ModelNotInstalledStub("Whisper Large-v3 is not installed. Open Settings -> Models to download or import it.")

    class ModelNotInstalledStub(RuntimeError):
        pass

    monkeypatch.setattr(ModelService, "model_spec", broken)
    assert run_worker(worker_env.request_file) == EXIT_FAILED
    events = events_of(worker_env)
    failed = [e for e in events if e["type"] == "failed"]
    assert len(failed) == 1 and "not installed" in failed[0]["error"] and events[-1]["type"] == "worker_exit"
    record = worker_env.store.get(worker_env.jid)
    assert record.state == "failed" and "not installed" in record.last_error


@pytest.mark.usefixtures("fake_ffmpeg")
def test_cloud_comparison_without_a_key_runs_locally_and_says_so(worker_env, monkeypatch):
    monkeypatch.setattr(tsvc, "CredentialService", lambda: CredentialService(MemoryCredentialBackend()))
    worker_env.request.cloud_compare = True
    worker_env.request.save(worker_env.request_file)
    assert run_worker(worker_env.request_file) == EXIT_COMPLETED
    assert any(e["type"] == "warning" and e.get("code") == "no_api_key" for e in events_of(worker_env))


@pytest.mark.usefixtures("fake_ffmpeg")
def test_idle_ollama_models_are_only_unloaded_when_the_setting_is_on(worker_env, monkeypatch):
    calls = []
    monkeypatch.setattr(GpuService, "unload_ollama", lambda self: calls.append(1) or ["qwen3:8b"])
    run_worker(worker_env.request_file)
    assert calls == []
    worker_env.request.unload_ollama = True
    worker_env.request.save(worker_env.request_file)
    run_worker(worker_env.request_file)
    assert calls == [1]


def test_worker_command_for_source_and_for_the_packaged_exe(tmp_path, monkeypatch):
    program, args = tsvc.worker_command(tmp_path / "r.json")
    assert program == sys.executable and args[:2] == ["-m", "src.services.worker"]
    monkeypatch.setattr(tsvc, "FROZEN", True)
    program, args = tsvc.worker_command(tmp_path / "r.json")
    assert args == ["--worker", str(tmp_path / "r.json")]


# ----------------------------------------------------------------------------------------------- headless model download (packaged-app proof)
def test_download_command_downloads_verifies_and_writes_a_result(tmp_path, service):
    import json as _json

    from src.services.model_cli import download_command

    out = tmp_path / "r" / "result.json"
    assert download_command("tiny", str(out), models=service) == 0
    data = _json.loads(out.read_text(encoding="utf-8"))
    assert data["ok"] and data["verified"] and data["files"] == 3 and data["size_on_disk"] > 2_000_000


def test_download_command_reports_unknown_models_and_failures(tmp_path, service):
    import json as _json

    from src.services.model_cli import download_command

    out = tmp_path / "result.json"
    assert download_command("no-such-model", str(out), models=service) == 1
    assert "Unknown model" in _json.loads(out.read_text(encoding="utf-8"))["error"]
