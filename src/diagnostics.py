"""GPU / environment diagnostics and VRAM monitoring."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .platform_utils import find_executable, register_cuda_dll_directories

log = logging.getLogger(__name__)

PASS, FAIL, WARN, SKIP = "PASS", "FAIL", "WARN", "SKIP"

CUDA_HELP = (
    "Possible causes / fixes:\n"
    "  1. NVIDIA driver missing or too old  -> install/update the Game Ready or Studio driver (nvidia-smi must work).\n"
    "  2. CPU-only PyTorch wheel installed   -> reinstall with: pip install torch==2.14.1+cu126 "
    "--index-url https://download.pytorch.org/whl/cu126\n"
    "  3. cuBLAS 12 / cuDNN 9 DLLs not found -> the cu126 PyTorch wheel bundles them; or pip install "
    "nvidia-cublas-cu12 nvidia-cudnn-cu12\n"
    "  4. GPU out of memory                  -> close other GPU programs (e.g. `ollama stop <model>`), "
    "or use --mode fast.\n"
    "Run CHECK_GPU.bat for a full report."
)


class CudaUnavailableError(RuntimeError):
    """Raised when a CUDA GPU is required but cannot be used. Never falls back to CPU silently."""


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)


def _hidden_run(cmd: list[str], timeout: float = 15.0) -> subprocess.CompletedProcess[str]:
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=timeout, creationflags=flags, check=False,
    )


def query_nvidia_smi(index: int = 0) -> Optional[dict[str, Any]]:
    """Return name/driver/VRAM from ``nvidia-smi`` or ``None`` if it is unavailable."""
    exe = find_executable("nvidia-smi") or None
    if exe is None:
        return None
    cmd = [str(exe), "--query-gpu=name,memory.total,memory.used,memory.free,driver_version",
           "--format=csv,noheader,nounits", "-i", str(index)]
    try:
        proc = _hidden_run(cmd)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    parts = [p.strip() for p in proc.stdout.strip().splitlines()[0].split(",")]
    if len(parts) < 5:
        return None
    try:
        return {
            "name": parts[0], "total_mib": int(parts[1]), "used_mib": int(parts[2]),
            "free_mib": int(parts[3]), "driver": parts[4],
        }
    except ValueError:
        return None


HEAVY_GPU_PROGRAMS = ("ollama", "llama", "lmstudio", "python", "comfy", "stable", "blender", "obs", "vlc", "game")


def gpu_compute_processes(limit: int = 8) -> list[str]:
    """Distinct program names using the GPU (best effort), likely heavy ones first.

    On Windows (WDDM) nearly every desktop app shows up, so the list is de-duplicated and capped.
    """
    exe = find_executable("nvidia-smi")
    if exe is None:
        return []
    try:
        proc = _hidden_run([str(exe), "--query-compute-apps=pid,process_name", "--format=csv,noheader"])
    except (OSError, subprocess.SubprocessError):
        return []
    names: dict[str, int] = {}
    for line in proc.stdout.splitlines():
        parts = line.strip().split(",", 1)
        if len(parts) == 2 and parts[1].strip():
            base = parts[1].strip().replace("\\", "/").rsplit("/", 1)[-1]
            names[base] = names.get(base, 0) + 1
    ordered = sorted(names, key=lambda n: (not any(h in n.lower() for h in HEAVY_GPU_PROGRAMS), n.lower()))
    shown = [f"{n} x{names[n]}" if names[n] > 1 else n for n in ordered[:limit]]
    if len(ordered) > limit:
        shown.append(f"... +{len(ordered) - limit} more")
    return shown


def check_nvidia_smi() -> CheckResult:
    info = query_nvidia_smi()
    if info is None:
        return CheckResult("nvidia-smi", FAIL, "nvidia-smi not found or returned an error (driver missing?)")
    detail = f"{info['name']} | driver {info['driver']} | {info['total_mib']} MiB total, {info['free_mib']} MiB free"
    return CheckResult("nvidia-smi", PASS, detail, info)


def check_torch() -> CheckResult:
    try:
        register_cuda_dll_directories()
        import torch
    except Exception as exc:  # noqa: BLE001 - report any import problem
        return CheckResult("PyTorch CUDA", FAIL, f"cannot import torch: {exc}")
    data: dict[str, Any] = {"torch": torch.__version__, "cuda_build": torch.version.cuda}
    if not torch.cuda.is_available():
        reason = "CPU-only PyTorch build" if torch.version.cuda is None else "CUDA runtime not usable"
        return CheckResult("PyTorch CUDA", FAIL, f"torch {torch.__version__}: cuda unavailable ({reason})", data)
    props = torch.cuda.get_device_properties(0)
    data.update(
        device=torch.cuda.get_device_name(0),
        capability=f"{props.major}.{props.minor}",
        vram_gb=round(props.total_memory / 1024**3, 2),
        cudnn=torch.backends.cudnn.version(),
    )
    detail = (f"torch {torch.__version__} (CUDA {torch.version.cuda}, cuDNN {data['cudnn']}) | "
              f"{data['device']} | sm_{props.major}{props.minor} | {data['vram_gb']} GB")
    return CheckResult("PyTorch CUDA", PASS, detail, data)


def check_ctranslate2() -> CheckResult:
    try:
        register_cuda_dll_directories()
        import ctranslate2
    except Exception as exc:  # noqa: BLE001
        return CheckResult("CTranslate2 CUDA", FAIL, f"cannot import ctranslate2: {exc}")
    data: dict[str, Any] = {"ctranslate2": ctranslate2.__version__}
    try:
        count = ctranslate2.get_cuda_device_count()
    except Exception as exc:  # noqa: BLE001
        return CheckResult("CTranslate2 CUDA", FAIL, f"ctranslate2 {ctranslate2.__version__}: {exc}", data)
    data["devices"] = count
    if count < 1:
        return CheckResult("CTranslate2 CUDA", FAIL,
                           f"ctranslate2 {ctranslate2.__version__}: no CUDA device visible", data)
    types = sorted(ctranslate2.get_supported_compute_types("cuda"))
    data["compute_types"] = types
    return CheckResult("CTranslate2 CUDA", PASS,
                       f"ctranslate2 {ctranslate2.__version__} | {count} device(s) | compute types: {', '.join(types)}",
                       data)


def check_whisper_gpu_inference(model_size: str = "tiny", compute_type: str = "float16") -> CheckResult:
    """Run a tiny real Whisper inference on the GPU (downloads ~75 MB the first time)."""
    try:
        register_cuda_dll_directories()
        import numpy as np
        from faster_whisper import WhisperModel
    except Exception as exc:  # noqa: BLE001
        return CheckResult("Whisper GPU inference", FAIL, f"cannot import faster-whisper: {exc}")
    try:
        started = time.perf_counter()
        model = WhisperModel(model_size, device="cuda", compute_type=compute_type)
        load_s = time.perf_counter() - started
        rng = np.random.default_rng(0)
        audio = (rng.standard_normal(16000 * 3) * 0.01).astype("float32")
        started = time.perf_counter()
        segments, info = model.transcribe(audio, beam_size=1, vad_filter=False)
        list(segments)  # force the lazy generator -> runs encoder + decoder on the GPU
        infer_s = time.perf_counter() - started
        del model
    except Exception as exc:  # noqa: BLE001
        return CheckResult("Whisper GPU inference", FAIL, f"{type(exc).__name__}: {exc}")
    return CheckResult("Whisper GPU inference", PASS,
                       f"model '{model_size}' {compute_type}: load {load_s:.1f}s, inference {infer_s:.2f}s",
                       {"model": model_size, "compute_type": compute_type})


def check_ffmpeg(explicit: Optional[str] = None) -> CheckResult:
    exe = find_executable("ffmpeg", explicit)
    if exe is None:
        return CheckResult("FFmpeg", FAIL, "ffmpeg not found (install: winget install Gyan.FFmpeg)")
    try:
        first = _hidden_run([str(exe), "-version"]).stdout.splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError) as exc:
        return CheckResult("FFmpeg", FAIL, f"{exe}: {exc}")
    return CheckResult("FFmpeg", PASS, f"{first} ({exe})", {"path": str(exe)})


def vram_warnings(required_mib: int = 5000) -> list[str]:
    """Human-readable warnings if the GPU is crowded by other programs."""
    info = query_nvidia_smi()
    if info is None or info["free_mib"] >= required_mib:
        return []
    others = gpu_compute_processes()
    msg = [f"Only {info['free_mib']} MiB VRAM free (large-v3 float16 needs ~{required_mib} MiB)."]
    if others:
        msg.append("Programs using the GPU: " + ", ".join(others))
    msg.append("Close GPU programs (e.g. `ollama stop <model>`) or use --mode fast / int8_float16.")
    return msg


def require_cuda(device_index: int = 0) -> None:
    """Raise :class:`CudaUnavailableError` with an actionable message if CUDA is unusable."""
    try:
        register_cuda_dll_directories()
        import ctranslate2
    except Exception as exc:  # noqa: BLE001
        raise CudaUnavailableError(f"CTranslate2 is not importable: {exc}\n{CUDA_HELP}") from exc
    try:
        count = ctranslate2.get_cuda_device_count()
    except Exception as exc:  # noqa: BLE001
        raise CudaUnavailableError(f"CUDA query failed: {exc}\n{CUDA_HELP}") from exc
    if count <= device_index:
        raise CudaUnavailableError(
            f"No CUDA device #{device_index} visible to CTranslate2 (found {count}). "
            f"The app does not fall back to CPU silently.\n{CUDA_HELP}"
        )


def cuda_status(device_index: int = 0) -> tuple[bool, str]:
    """``(available, reason)`` without raising: can CTranslate2 use NVIDIA GPU ``device_index``? The reason is user-readable."""
    try:
        register_cuda_dll_directories()
        import ctranslate2
    except Exception as exc:  # noqa: BLE001
        return False, f"The speech engine (CTranslate2) could not be loaded: {exc}"
    try:
        count = ctranslate2.get_cuda_device_count()
    except Exception as exc:  # noqa: BLE001
        text = str(exc)
        if any(tag in text.lower() for tag in ("cublas", "cudnn", "cudart", ".dll")):
            return False, f"A CUDA runtime library is missing or incompatible ({text.splitlines()[0][:140]})."
        return False, f"CUDA could not be queried: {text.splitlines()[0][:160] if text else type(exc).__name__}"
    if count <= device_index:
        if query_nvidia_smi(device_index) is None:
            return False, "No compatible NVIDIA GPU / driver was detected."
        return False, "An NVIDIA GPU was found but its driver does not provide CUDA 12 (update the NVIDIA driver)."
    return True, "CUDA acceleration is available."


def resolve_device(requested: str = "auto", device_index: int = 0) -> tuple[str, Optional[str]]:
    """Map the user's device choice to ``cuda`` / ``cpu``. ``auto`` uses CUDA when it works and otherwise falls back to
    the CPU, returning the reason; an explicit ``cuda`` never falls back silently (it raises later, as the CLI always did)."""
    if requested in ("cpu", "cuda"):
        return requested, None
    ok, reason = cuda_status(device_index)
    return ("cuda", None) if ok else ("cpu", reason)


def run_all_checks(with_inference: bool = True) -> list[CheckResult]:
    results = [check_nvidia_smi(), check_torch(), check_ctranslate2()]
    if with_inference:
        if all(r.status == PASS for r in results[1:]):
            results.append(check_whisper_gpu_inference())
        else:
            results.append(CheckResult("Whisper GPU inference", SKIP, "skipped: earlier CUDA checks failed"))
    return results


def overall_status(results: list[CheckResult]) -> str:
    """READY only when every check passed (SKIP and FAIL both block readiness)."""
    return "READY" if results and all(r.status in (PASS, WARN) for r in results) else "NOT READY"


class ProcessGpuCounter:
    """Per-process GPU memory via the Windows "GPU Process Memory" performance counters.

    ``nvidia-smi`` cannot report per-process memory under WDDM, and the whole-GPU figure is
    misleading when the driver spills allocations into shared system memory (CUDA sysmem
    fallback) instead of failing. This reads ``typeperf`` for our own PID: dedicated (real VRAM)
    and shared (spilled to system RAM). Returns ``None`` values where counters are unavailable.
    """

    RETRY_SECONDS = 2.0

    def __init__(self, pid: Optional[int] = None) -> None:
        self.pid = pid or os.getpid()
        self.peak_dedicated_mib: Optional[float] = None
        self.peak_shared_mib: Optional[float] = None
        self.last_dedicated_mib: Optional[float] = None     # latest sample (for live progress lines)
        self.last_shared_mib: Optional[float] = None
        self._proc: Optional[subprocess.Popen[str]] = None
        self._thread: Optional[threading.Thread] = None
        self._last_attempt = 0.0
        self._exe = shutil.which("typeperf") if os.name == "nt" else None

    def start(self) -> "ProcessGpuCounter":
        self.poll()
        return self

    def poll(self) -> None:
        """(Re)start ``typeperf`` if needed. The counter instance only exists once this process has
        allocated GPU memory, so an early start exits immediately and is retried later."""
        if self._exe is None:
            return
        if self._proc is not None and self._proc.poll() is None:
            return
        now = time.monotonic()
        if now - self._last_attempt < self.RETRY_SECONDS:
            return
        self._last_attempt = now
        cmd = [self._exe, f"\\GPU Process Memory(pid_{self.pid}*)\\Dedicated Usage",
               f"\\GPU Process Memory(pid_{self.pid}*)\\Shared Usage", "-si", "1"]
        try:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, text=True,
                encoding="utf-8", errors="replace", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError:
            self._exe = None
            return
        self._thread = threading.Thread(target=self._read, args=(self._proc,), name="gpu-proc-counter", daemon=True)
        self._thread.start()

    def _read(self, proc: "subprocess.Popen[str]") -> None:
        assert proc.stdout is not None
        columns: list[str] = []
        for line in proc.stdout:
            fields = [f.strip().strip('"') for f in line.strip().split('","')]
            if not fields or not fields[0]:
                continue
            if fields[0].startswith("(PDH-CSV"):
                columns = fields[1:]
                continue
            dedicated = shared = 0.0
            parsed = False
            for name, value in zip(columns, fields[1:]):
                try:
                    number = float(value.replace(",", ".")) / (1024 * 1024)
                except ValueError:
                    continue
                parsed = True
                if name.lower().endswith("dedicated usage"):
                    dedicated += number
                elif name.lower().endswith("shared usage"):
                    shared += number
            if parsed:
                self.last_dedicated_mib, self.last_shared_mib = dedicated, shared
                self.peak_dedicated_mib = max(self.peak_dedicated_mib or 0.0, dedicated)
                self.peak_shared_mib = max(self.peak_shared_mib or 0.0, shared)

    def stop(self) -> dict[str, Optional[float]]:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread is not None:
            self._thread.join(timeout=3)
        def rounded(v: Optional[float]) -> Optional[float]:
            return None if v is None else round(v)

        return {"process_peak_dedicated_mib": rounded(self.peak_dedicated_mib),
                "process_peak_shared_mib": rounded(self.peak_shared_mib)}


class GpuMemoryMonitor:
    """Record peak GPU memory during a run.

    Two views are combined: the whole-GPU figure from ``nvidia-smi`` (``peak_mib`` minus
    ``baseline_mib`` = ``peak_delta_mib``, polled in a thread) and, where Windows exposes it, this
    process's own dedicated / shared (spilled-to-RAM) usage (``process_peak_*``). The per-process
    numbers are the trustworthy ones when other programs (e.g. Ollama) share the GPU.
    """

    def __init__(self, interval: float = 0.5, index: int = 0, per_process: bool = True) -> None:
        self.interval = interval
        self.index = index
        self.baseline_mib: Optional[int] = None
        self.peak_mib: Optional[int] = None
        self._process_counter: Optional[ProcessGpuCounter] = ProcessGpuCounter() if per_process else None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _sample(self) -> None:
        info = query_nvidia_smi(self.index)
        if info is None:
            return
        used = info["used_mib"]
        if self.peak_mib is None or used > self.peak_mib:
            self.peak_mib = used

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self._sample()
            if self._process_counter is not None:
                self._process_counter.poll()

    def latest(self) -> dict[str, Optional[float]]:
        """Most recent per-process GPU memory (dedicated / shared) for live progress lines."""
        counter = self._process_counter
        return {"dedicated_mib": counter.last_dedicated_mib if counter else None,
                "shared_mib": counter.last_shared_mib if counter else None}

    def start(self) -> "GpuMemoryMonitor":
        info = query_nvidia_smi(self.index)
        if info is not None:
            self.baseline_mib = info["used_mib"]
            self.peak_mib = info["used_mib"]
        if self._process_counter is not None:
            self._process_counter.start()
        self._thread = threading.Thread(target=self._loop, name="gpu-mem-monitor", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> dict[str, Optional[int]]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._sample()
        delta = None
        if self.peak_mib is not None and self.baseline_mib is not None:
            delta = max(0, self.peak_mib - self.baseline_mib)
        result: dict[str, Optional[int]] = {"baseline_mib": self.baseline_mib, "peak_mib": self.peak_mib,
                                            "peak_delta_mib": delta}
        if self._process_counter is not None:
            result.update(self._process_counter.stop())
        return result

    def __enter__(self) -> "GpuMemoryMonitor":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()


def environment_summary() -> dict[str, Any]:
    """Package versions for metadata.json (imports are lazy and failure-tolerant)."""
    import sys
    from importlib import metadata

    summary: dict[str, Any] = {"python": sys.version.split()[0]}
    for pkg in ("faster-whisper", "ctranslate2", "torch", "transformers", "numpy", "pyyaml", "pypdf"):
        try:
            summary[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            summary[pkg] = None
    return summary
