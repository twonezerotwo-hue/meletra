"""CUDA-unavailable handling and the float16 -> int8_float16 fallback policy (no GPU needed)."""

import logging

import pytest

from src import diagnostics
from src.diagnostics import CudaUnavailableError, GpuMemoryMonitor, require_cuda
from src.model_manager import (
    ASRAdapter, Capabilities, DecodeRequest, FasterWhisperAdapter, GpuOutOfMemoryError, ModelSpec,
    TransformersAdapter,
)
from src.schema import Segment, TranscriptionResult
from src.transcriber import Transcriber


class _NoCuda:
    """Stand-in for ctranslate2 reporting zero CUDA devices."""

    __version__ = "0"

    @staticmethod
    def get_cuda_device_count() -> int:
        return 0


@pytest.fixture
def no_cuda(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "ctranslate2", _NoCuda)


def test_require_cuda_raises_actionable_error_when_no_gpu(no_cuda):
    with pytest.raises(CudaUnavailableError) as info:
        require_cuda(0)
    message = str(info.value)
    assert "CPU" in message and "silently" in message
    assert "cu126" in message and "nvidia-smi" in message


def test_faster_whisper_adapter_refuses_to_fall_back_to_cpu(no_cuda):
    adapter = FasterWhisperAdapter(ModelSpec("default", "faster_whisper", "large-v3"), device="cuda")
    with pytest.raises(CudaUnavailableError):
        adapter.load("float16")


def test_transformers_adapter_refuses_to_fall_back_to_cpu(no_cuda):
    adapter = TransformersAdapter(ModelSpec("tm", "transformers", "x/y"), device="cuda")
    with pytest.raises(CudaUnavailableError):
        adapter.load("float16")


def test_cpu_requires_an_explicit_device_flag(no_cuda, caplog):
    adapter = FasterWhisperAdapter(ModelSpec("default", "faster_whisper", "large-v3"), device="cpu")
    with caplog.at_level(logging.WARNING):
        adapter._require_gpu()  # explicit CPU is allowed, but loudly
    assert "MUCH slower" in caplog.text


def test_nvidia_smi_missing_is_reported_not_raised(monkeypatch):
    monkeypatch.setattr(diagnostics, "find_executable", lambda *a, **k: None)
    assert diagnostics.query_nvidia_smi() is None
    assert diagnostics.check_nvidia_smi().status == diagnostics.FAIL
    assert diagnostics.vram_warnings() == []


def test_vram_warning_names_competing_processes(monkeypatch):
    monkeypatch.setattr(diagnostics, "query_nvidia_smi", lambda index=0: {"free_mib": 1500, "total_mib": 8192})
    monkeypatch.setattr(diagnostics, "gpu_compute_processes", lambda: ["llama-server.exe x2", "chrome.exe"])
    warnings = diagnostics.vram_warnings()
    assert "1500 MiB" in warnings[0]
    assert any("llama-server.exe x2" in w for w in warnings)


def test_gpu_program_list_is_deduplicated_and_heavy_first(monkeypatch):
    class Proc:
        stdout = ("1, C:\\Windows\\explorer.exe\n2, C:\\a\\chrome.exe\n3, C:\\b\\chrome.exe\n"
                  "4, C:\\Ollama\\lib\\llama-server.exe\n")

    monkeypatch.setattr(diagnostics, "find_executable", lambda *a, **k: "nvidia-smi")
    monkeypatch.setattr(diagnostics, "_hidden_run", lambda *a, **k: Proc())
    assert diagnostics.gpu_compute_processes() == ["llama-server.exe", "chrome.exe x2", "explorer.exe"]


def test_process_gpu_counter_parses_typeperf_csv_in_any_locale():
    mib = 1024 * 1024
    header = ('"(PDH-CSV 4.0)","\\\\PC\\GPU Process Memory(pid_7_luid_a_phys_0)\\Dedicated Usage",'
              '"\\\\PC\\GPU Process Memory(pid_7_luid_b_phys_0)\\Dedicated Usage",'
              '"\\\\PC\\GPU Process Memory(pid_7_luid_a_phys_0)\\Shared Usage"\n')

    def row(ts, a, b, c):
        return f'"{ts}","{a * mib:.6f}","{b * mib:.6f}","{c * mib:.6f}"\n'

    class FakeProc:
        stdout = iter([header, row("t1", 100, 0, 5), row("t2", 3000, 20, 700), row("t3", 2900, 0, 650),
                       'Exiting, please wait...\n'])

    counter = diagnostics.ProcessGpuCounter(pid=7)
    counter._read(FakeProc())
    result = {"d": round(counter.peak_dedicated_mib), "s": round(counter.peak_shared_mib)}
    assert result == {"d": 3020, "s": 700}      # dedicated = sum over adapters; peaks, not last values


def test_process_gpu_counter_is_inert_without_typeperf(monkeypatch):
    monkeypatch.setattr(diagnostics.shutil, "which", lambda name: None)
    counter = diagnostics.ProcessGpuCounter().start()
    assert counter.stop() == {"process_peak_dedicated_mib": None, "process_peak_shared_mib": None}


def test_gpu_memory_monitor_tracks_peak(monkeypatch):
    readings = iter([2000, 2000, 5200, 4100, 4100, 4100])
    monkeypatch.setattr(
        diagnostics, "query_nvidia_smi", lambda index=0: {"used_mib": next(readings, 4100)}
    )
    monitor = GpuMemoryMonitor(interval=0.01)
    monitor.start()
    import time

    time.sleep(0.15)
    result = monitor.stop()
    assert result["baseline_mib"] == 2000
    assert result["peak_mib"] == 5200
    assert result["peak_delta_mib"] == 3200


class _FakeAdapter(ASRAdapter):
    """Adapter whose first compute type runs out of memory."""

    backend_name = "fake"
    capabilities = Capabilities(compute_types=True)

    def __init__(self, fail_on: set[str]) -> None:
        super().__init__(ModelSpec("default", "faster_whisper", "large-v3"))
        self.fail_on = fail_on
        self.loaded: list[str] = []

    def load(self, compute_type: str) -> None:
        self.loaded.append(compute_type)
        self.compute_type = compute_type

    def transcribe(self, request: DecodeRequest) -> TranscriptionResult:
        if self.compute_type in self.fail_on:
            raise GpuOutOfMemoryError("CUDA failed with error out of memory")
        seg = Segment(0, 1, "ok", "en", "large-v3", "fake")
        return TranscriptionResult([seg], "en", 1.0, 1.0, "large-v3", "fake")

    def close(self) -> None:
        pass


def _transcriber(monkeypatch, cfg, fail_on, profile="quality"):
    fake = _FakeAdapter(fail_on)
    monkeypatch.setattr("src.transcriber.create_adapter", lambda *a, **k: fake)
    return Transcriber(ModelSpec("default", "faster_whisper", "large-v3"), cfg[profile]), fake


def _request():
    return DecodeRequest(audio_path=None, language=None, profile={}, decoding={}, vad_parameters={})


def test_float16_oom_falls_back_to_int8_float16_and_logs(monkeypatch, cfg, caplog):
    transcriber, fake = _transcriber(monkeypatch, cfg, {"float16"})
    with caplog.at_level(logging.WARNING):
        result = transcriber.transcribe(_request())
    assert fake.loaded == ["float16", "int8_float16"]
    assert result.compute_type == "int8_float16" and result.fallback_used
    assert "FALLBACK" in caplog.text and "CPU is never used" in caplog.text
    assert any("fell back" in n for n in result.notes)


def test_fallback_is_sticky_for_following_files(monkeypatch, cfg):
    transcriber, fake = _transcriber(monkeypatch, cfg, {"float16"})
    transcriber.transcribe(_request())
    transcriber.transcribe(_request())
    assert fake.loaded == ["float16", "int8_float16"]  # second file did not reload / retry float16


def test_no_fallback_needed_keeps_float16(monkeypatch, cfg):
    transcriber, fake = _transcriber(monkeypatch, cfg, set())
    result = transcriber.transcribe(_request())
    assert result.compute_type == "float16" and not result.fallback_used


def test_oom_in_every_compute_type_raises_with_advice(monkeypatch, cfg):
    transcriber, _ = _transcriber(monkeypatch, cfg, {"float16", "int8_float16", "int8"})
    with pytest.raises(GpuOutOfMemoryError, match="ollama stop"):
        transcriber.transcribe(_request())


def test_last_resort_is_int8_after_float16_and_int8_float16_do_not_fit(monkeypatch, cfg):
    transcriber, fake = _transcriber(monkeypatch, cfg, {"float16", "int8_float16"})
    result = transcriber.transcribe(_request())
    assert fake.loaded == ["float16", "int8_float16", "int8"]
    assert result.compute_type == "int8" and result.fallback_used


def test_fast_mode_ends_on_int8_and_fails_cleanly(monkeypatch, cfg):
    transcriber, fake = _transcriber(monkeypatch, cfg, {"int8_float16", "int8"}, profile="fast")
    with pytest.raises(GpuOutOfMemoryError):
        transcriber.transcribe(_request())
    assert fake.loaded == ["int8_float16", "int8"]                  # the fast profile ends on the same last resort
