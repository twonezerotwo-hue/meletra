"""Windows / CUDA specific helpers, kept out of the core logic."""

from __future__ import annotations

import glob
import importlib.util
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

from . import APP_DIR, FROZEN, PROJECT_ROOT

log = logging.getLogger(__name__)

_registered_dirs: list[Path] = []


def configure_utf8_console() -> None:
    """Make stdout/stderr UTF-8 so Turkish characters survive the Windows console."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


def _package_dir(name: str) -> Optional[Path]:
    """Locate an installed package directory without importing it."""
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None
    if spec is None:
        return None
    if spec.submodule_search_locations:
        locations = list(spec.submodule_search_locations)
        return Path(locations[0]) if locations else None
    return Path(spec.origin).parent if spec.origin else None


def bundled_bin_dir() -> Path:
    """Where the app ships ffmpeg.exe / ffprobe.exe: ``<app>/bin`` (installed/portable) or ``vendor/ffmpeg/bin`` (source tree)."""
    return APP_DIR / "bin" if FROZEN else PROJECT_ROOT / "vendor" / "ffmpeg" / "bin"


def bundled_cuda_dir() -> Path:
    """Where the app ships the CUDA 12 runtime DLLs CTranslate2 needs (cuBLAS, cuDNN): ``<app>/libs/cuda``."""
    return APP_DIR / "libs" / "cuda" if FROZEN else PROJECT_ROOT / "vendor" / "cuda"


def candidate_cuda_dll_dirs() -> list[Path]:
    """Directories that may hold cuBLAS 12 / cuDNN 9 DLLs.

    The app's own ``libs/cuda`` comes first (so a packaged build never depends on the machine), then the PyTorch cu12x
    wheels (``torch/lib``) and the ``nvidia-*-cu12`` wheels (``nvidia/<lib>/bin``) for source installs.
    """
    dirs: list[Path] = [bundled_cuda_dir()]
    torch_dir = _package_dir("torch")
    if torch_dir is not None:
        dirs.append(torch_dir / "lib")
    nvidia_dir = _package_dir("nvidia")
    if nvidia_dir is not None:
        dirs.extend(sorted(nvidia_dir.glob("*/bin")))
    return [d for d in dirs if d.is_dir()]


def register_cuda_dll_directories() -> list[Path]:
    """Expose bundled CUDA DLLs to CTranslate2 (Windows only; no-op elsewhere).

    CTranslate2's wheel does not ship cuBLAS/cuDNN. We add the directories both to the
    DLL search path and to ``PATH`` so ``LoadLibrary`` finds them regardless of flags.
    Must run BEFORE ``import ctranslate2`` / ``import faster_whisper``.
    """
    if os.name != "nt":
        return []
    added: list[Path] = []
    for directory in candidate_cuda_dll_dirs():
        if directory in _registered_dirs:
            continue
        try:
            os.add_dll_directory(str(directory))
        except OSError as exc:  # pragma: no cover - defensive
            log.warning("Could not add DLL directory %s: %s", directory, exc)
            continue
        os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")
        _registered_dirs.append(directory)
        added.append(directory)
        log.debug("Registered CUDA DLL directory: %s", directory)
    return added


def _winget_ffmpeg_candidates() -> list[Path]:
    local = os.environ.get("LOCALAPPDATA")
    if not local:
        return []
    pattern = os.path.join(local, "Microsoft", "WinGet", "Packages", "Gyan.FFmpeg*", "**", "bin", "ffmpeg.exe")
    links = Path(local) / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe"
    found = [Path(p) for p in glob.glob(pattern, recursive=True)]
    if links.is_file():
        found.insert(0, links)
    return found


def find_executable(name: str, explicit: Optional[str | Path] = None) -> Optional[Path]:
    """Find ``ffmpeg``/``ffprobe``: explicit path, env override, PATH, then WinGet's install dir."""
    if name in ("ffmpeg", "ffprobe") and os.environ.get("MLT_DISABLE_FFMPEG") == "1":
        return None                      # troubleshooting switch: decode with the built-in PyAV libraries only
    from .platform import current

    suffix = current().exe_suffix                      # ".exe" on Windows, nothing on macOS
    bundled = bundled_bin_dir() / f"{name}{suffix}"
    if not explicit and bundled.is_file():             # the bundled binary always wins: no dependence on the system PATH
        return bundled
    if explicit:
        path = Path(explicit)
        if path.is_file():
            return path
        # allow pointing at a folder that contains the binaries
        for candidate in (path / f"{name}{suffix}", path / name):
            if candidate.is_file():
                return candidate
    env_value = os.environ.get(f"{name.upper()}_PATH")
    if env_value and Path(env_value).is_file():
        return Path(env_value)
    on_path = shutil.which(name)
    if on_path:
        return Path(on_path)
    if os.name == "nt":
        for ffmpeg_path in _winget_ffmpeg_candidates():
            sibling = ffmpeg_path.with_name(f"{name}.exe")
            if sibling.is_file():
                return sibling
    return None
