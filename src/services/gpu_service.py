"""GPU detection, VRAM protection and (opt-in) unloading of idle Ollama models.

The app never closes other programs on its own. It tells the user who is using the GPU and lets them decide; the only
automatic action is the OPTIONAL "unload idle Ollama models" setting (default OFF), which runs ``ollama stop <model>`` -
never kills a process or touches the Ollama service.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

from .. import diagnostics

log = logging.getLogger(__name__)

#: free VRAM that Whisper large-v3 in float16 needs (measured 4.2-4.3 GB dedicated on an RTX 2070)
LARGE_V3_FLOAT16_MIB = 4500
LARGE_V3_INT8_MIB = 2600


@dataclass
class GpuStatus:
    name: str = ""
    driver: str = ""
    total_mib: int = 0
    free_mib: int = 0
    used_mib: int = 0
    nvidia_present: bool = False
    cuda_available: bool = False
    cuda_reason: str = ""
    device: str = "cpu"                         # what "auto" will use
    headline: str = ""
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VramCheck:
    ok: bool
    severity: str = "ok"                        # ok | warning | critical
    free_mib: int = 0
    required_mib: int = 0
    programs: list[str] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GpuService:
    def __init__(self, smi: Callable[[int], Optional[dict[str, Any]]] = diagnostics.query_nvidia_smi,
                 cuda: Callable[[int], tuple[bool, str]] = diagnostics.cuda_status,
                 processes: Callable[[], list[str]] = diagnostics.gpu_compute_processes) -> None:
        self._smi, self._cuda, self._processes = smi, cuda, processes

    def detect(self, index: int = 0) -> GpuStatus:
        from ..platform import current

        platform = current()
        if platform.name == "macos":                      # no NVIDIA / CUDA question on a Mac
            return self._detect_mac(platform)
        info = self._smi(index)
        ok, reason = self._cuda(index)
        status = GpuStatus(nvidia_present=info is not None or ok, cuda_available=ok, cuda_reason=reason,
                           device="cuda" if ok else "cpu")
        if info:
            status.name, status.driver = info["name"], info["driver"]
            status.total_mib, status.free_mib, status.used_mib = info["total_mib"], info["free_mib"], info["used_mib"]
        if ok and info:
            status.headline = f"GPU detected: {info['name']} ({info['total_mib'] / 1024:.0f} GB VRAM)"
            status.detail = "CUDA acceleration available"
        elif ok:                                  # CUDA works but nvidia-smi could not be queried (no name / memory figures)
            status.headline = "NVIDIA GPU detected"
            status.detail = "CUDA acceleration available"
        elif info:
            status.headline = f"GPU detected: {info['name']}"
            status.detail = f"CUDA acceleration unavailable. Reason: {reason} CPU fallback enabled."
        else:
            status.headline = "No compatible NVIDIA GPU detected."
            status.detail = "CPU mode will be used. Large-v3 may be very slow."
        return status

    @staticmethod
    def _detect_mac(platform) -> GpuStatus:
        if platform.is_apple_silicon:
            return GpuStatus(name="Apple Silicon GPU", device="metal", headline="Apple Silicon detected",
                             detail="The optimized speech engine uses this Mac's GPU (Metal) and its unified memory.")
        return GpuStatus(device="cpu", headline="Intel Mac detected",
                         detail="The CPU will be used. Large-v3 can take several times the length of the recording.")

    def vram_check(self, required_mib: int = LARGE_V3_FLOAT16_MIB, index: int = 0) -> VramCheck:
        """Is there enough free VRAM for the job? The answer names the programs that hold the GPU."""
        info = self._smi(index)
        if info is None:
            return VramCheck(True, "ok", 0, required_mib, [], "")
        free = info["free_mib"]
        if free >= required_mib:
            return VramCheck(True, "ok", free, required_mib, [], "")
        programs = self._processes()
        severity = "critical" if free < LARGE_V3_INT8_MIB else "warning"
        who = f" Detected: {', '.join(programs)}." if programs else ""
        message = (f"Another application is using significant GPU memory. Only {free / 1024:.1f} GB of "
                   f"{info['total_mib'] / 1024:.1f} GB VRAM are free (about {required_mib / 1024:.1f} GB are needed).{who} "
                   f"The program will not close anything by itself. Transcription still works but may be slower.")
        return VramCheck(False, severity, free, required_mib, programs, message)

    # ------------------------------------------------------------------ Ollama (opt-in)
    @staticmethod
    def ollama_models(run: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> list[str]:
        exe = shutil.which("ollama")
        if not exe:
            return []
        try:
            proc = run([exe, "ps"], capture_output=True, text=True, timeout=10, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except (OSError, subprocess.SubprocessError):
            return []
        lines = proc.stdout.strip().splitlines()[1:]
        return [line.split()[0] for line in lines if line.strip()]

    def unload_ollama(self, run: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> list[str]:
        """``ollama stop <model>`` for every loaded model. Called ONLY when the user enabled the setting."""
        stopped = []
        exe = shutil.which("ollama")
        for model in self.ollama_models(run):
            try:
                run([exe, "stop", model], capture_output=True, text=True, timeout=20,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                stopped.append(model)
            except (OSError, subprocess.SubprocessError) as exc:
                log.warning("Could not unload Ollama model %s: %s", model, exc)
        return stopped
