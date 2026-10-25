"""One report that answers "why does it not work on my PC?": app, Windows, CPU, RAM, GPU, CUDA, libraries, FFmpeg, models.

Collected without blocking: nothing here launches a program that could hang the window (ffmpeg's version is read with a hard
timeout and abandoned if it does not answer).
"""

from __future__ import annotations

import ctypes
import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Optional

from .. import APP_NAME, COMPANY, FROZEN, __version__, build_info
from ..platform_utils import bundled_bin_dir, bundled_cuda_dir, find_executable
from .gpu_service import GpuService
from .model_service import ModelService, human_bytes
from .paths import AppPaths


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def memory_status() -> dict[str, float]:
    from ..platform import current

    return current().memory_status()


def cpu_name() -> str:
    from ..platform import current

    return current().cpu_name()


def windows_version() -> str:
    """The operating system line of the report (named for history: it is "macOS 14.5 (arm64)" on a Mac)."""
    from ..platform import current

    return current().os_description()


def _platform_facts() -> dict[str, Any]:
    from ..platform import current

    try:
        return current().describe()
    except Exception:                                      # noqa: BLE001 - a report must never fail
        return {"name": "unknown"}


def _engines() -> list[dict[str, str]]:
    try:
        from ..asr import describe_engines

        return describe_engines()
    except Exception:                                      # noqa: BLE001
        return []


def tool_version(exe: Optional[Path], timeout: float = 6.0) -> str:
    """First line of ``<tool> -version``, or a note. The process is abandoned (never waited for) if it hangs."""
    if exe is None:
        return "not found"
    box: dict[str, str] = {}

    def run() -> None:
        try:
            proc = subprocess.Popen([str(exe), "-version"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            out, _ = proc.communicate(timeout=timeout)
            box["v"] = (out or "").splitlines()[0] if out else "no output"
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except OSError:
                pass
            box["v"] = f"did not answer within {timeout:.0f} s (the app decodes audio with PyAV instead)"
        except (OSError, IndexError) as exc:
            box["v"] = f"cannot start: {exc}"

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(timeout + 2)
    return box.get("v", f"did not answer within {timeout:.0f} s")


def collect(paths: AppPaths, models: ModelService, gpu: Optional[GpuService] = None, check_ffmpeg_version: bool = True
            ) -> dict[str, Any]:
    gpu = gpu or GpuService()
    status = gpu.detect()
    ffmpeg = find_executable("ffmpeg")
    ffprobe = find_executable("ffprobe")
    cuda_dir = bundled_cuda_dir()
    data: dict[str, Any] = {
        "app": {"name": APP_NAME, "version": __version__, "company": COMPANY, "packaged": FROZEN,
                "commit": build_info.commit(short=False), "build_date": build_info.load().get("build_date", ""),
                "python_required": False if FROZEN else True, "executable": sys.executable},
        "windows": windows_version(),                      # the operating system line, whatever the system (kept key name of 0.1.1)
        "platform": _platform_facts(),
        "engines": _engines(),
        "cpu": {"name": cpu_name(), "logical_cores": os.cpu_count()},
        "ram": memory_status(),
        "gpu": status.to_dict(),
        "nvidia_driver": status.driver or "not detected",
        "cuda": {"available": status.cuda_available, "reason": status.cuda_reason,
                 "bundled_runtime_dir": str(cuda_dir), "bundled_runtime_present": cuda_dir.is_dir() and any(cuda_dir.glob("*.dll"))},
        "libraries": {"ctranslate2": _version("ctranslate2"), "faster-whisper": _version("faster-whisper"),
                      "av (PyAV)": _version("av"), "PySide6": _version("PySide6-Essentials") if _version("PySide6-Essentials") != "not installed" else _version("PySide6"),
                      "numpy": _version("numpy"), "onnxruntime": _version("onnxruntime"),
                      "mlx-whisper": _version("mlx-whisper")},
        "ffmpeg": {"path": str(ffmpeg) if ffmpeg else None, "bundled": bool(ffmpeg and ffmpeg.parent == bundled_bin_dir()),
                   "ffprobe": str(ffprobe) if ffprobe else None,
                   "version": tool_version(ffmpeg) if check_ffmpeg_version else "not checked",
                   "pyav_fallback": True},
        "models": [{"key": s.key, "title": s.title, "installed": s.installed, "path": str(s.path) if s.path else None,
                    "size": human_bytes(s.size_on_disk) if s.installed else "-", "verified": s.verified,
                    "runtime_ok": s.runtime_ok} for s in models.list_models(include_hidden=True)],
        "paths": {"data": str(paths.root), "models": str(paths.models), "logs": str(paths.logs),
                  "checkpoints": str(paths.checkpoints), "settings": str(paths.settings_file)},
        "disk": {"free_gb": round(shutil.disk_usage(paths.root).free / 1024 ** 3, 1)},
    }
    return data


def _os_label(data: dict[str, Any]) -> str:
    return {"windows": "Windows", "macos": "macOS", "linux": "Linux"}.get((data.get("platform") or {}).get("name"), "Windows")


def as_text(data: dict[str, Any]) -> str:
    """Plain-text version for the COPY DIAGNOSTICS button (no secrets are ever collected)."""
    app, gpu, cuda, ff = data["app"], data["gpu"], data["cuda"], data["ffmpeg"]
    lines = [
        f"{app['name']} {app['version']} ({app['company']})  packaged={app['packaged']}  python required: {'no' if not app['python_required'] else 'yes (source run)'}",
        f"{_os_label(data):<13}: {data['windows']}",
        f"CPU          : {data['cpu']['name']} ({data['cpu']['logical_cores']} logical cores)",
        f"RAM          : {data['ram']['total_gb']} GB total, {data['ram']['free_gb']} GB free",
        f"GPU          : {gpu['name'] or 'none'}  VRAM {gpu['total_mib']} MiB total / {gpu['free_mib']} MiB free",
        f"NVIDIA driver: {data['nvidia_driver']}",
        f"CUDA         : {'available' if cuda['available'] else 'unavailable - ' + (cuda['reason'] or 'unknown')}  "
        f"(bundled runtime: {'yes' if cuda['bundled_runtime_present'] else 'no'})",
    ]
    lines += [f"{name:<13}: {version}" for name, version in data["libraries"].items() if version != "not installed" or name != "mlx-whisper"]
    lines += [f"Speech engine: {e['engine']} ({e['backend']}): {e['note']}" for e in data.get("engines", [])]
    lines += [f"FFmpeg       : {ff['path'] or 'not found'} ({'bundled' if ff['bundled'] else 'system'}) - {ff['version']}",
              "Audio decoder: FFmpeg, with PyAV (built-in libraries) as automatic fallback"]
    for m in data["models"]:
        lines.append(f"Model        : {m['title']}: {'installed ' + m['size'] if m['installed'] else 'not installed'}")
    lines += [f"Data folder  : {data['paths']['data']}", f"Free disk    : {data['disk']['free_gb']} GB"]
    return "\n".join(lines) + "\n"
