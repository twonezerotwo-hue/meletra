"""The platform interface and the fallback for other POSIX systems (Linux: development and CI only)."""

from __future__ import annotations

import os
import platform as _stdlib_platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

APP_ID = "MedicalLectureTranscriber"

Runner = Callable[..., Any]          # subprocess.run-like: runner(args, **kwargs) -> CompletedProcess


@dataclass(frozen=True)
class AcceleratorInfo:
    """The hardware that speeds speech recognition up (shown only in advanced diagnostics)."""

    kind: str               # "cuda" (NVIDIA) | "apple" (Apple Silicon) | "none"
    name: str = ""
    detail: str = ""


class Platform:
    """Everything that differs between operating systems. Subclasses override what they do differently."""

    name = "generic"                      # windows | macos | linux
    display_name = "this computer"
    exe_suffix = ""
    secure_store_label = "memory only (this system has no secure store)"

    def __init__(self, env: Optional[Mapping[str, str]] = None, home: Optional[Path] = None, run: Optional[Runner] = None,
                 machine: Optional[str] = None) -> None:
        self.env: Mapping[str, str] = os.environ if env is None else env
        self.home = Path(home) if home is not None else Path.home()
        self._run: Runner = run or subprocess.run
        self.machine = machine if machine is not None else _stdlib_platform.machine()

    # ------------------------------------------------------------------ folders
    def user_data_dir(self) -> Path:
        """Models, settings, jobs, checkpoints, temporary files (and, where the OS has no better place, cache and logs)."""
        base = Path(self.env["XDG_DATA_HOME"]) if self.env.get("XDG_DATA_HOME") else self.home / ".local" / "share"
        return base / APP_ID

    def cache_dir(self) -> Optional[Path]:
        """Disposable caches; ``None`` = inside the data folder."""
        return None

    def logs_dir(self) -> Optional[Path]:
        """Log files; ``None`` = inside the data folder."""
        return None

    # ------------------------------------------------------------------ shell
    def open_path(self, path: Path) -> None:
        self._run(["xdg-open", str(path)], check=False)

    def reveal_path(self, path: Path) -> None:
        """Show ``path`` in the file manager (selected when the OS can)."""
        self.open_path(Path(path).parent if Path(path).is_file() else Path(path))

    # ------------------------------------------------------------------ power
    def keep_awake(self, on: bool) -> bool:
        return False

    # ------------------------------------------------------------------ system facts
    def memory_status(self) -> dict[str, float]:
        return {"total_gb": 0.0, "free_gb": 0.0}

    def cpu_name(self) -> str:
        return _stdlib_platform.processor() or "unknown"

    def os_description(self) -> str:
        return _stdlib_platform.platform()

    def accelerator(self) -> AcceleratorInfo:
        return AcceleratorInfo("none")

    @property
    def is_apple_silicon(self) -> bool:
        return False

    # ------------------------------------------------------------------ process / security
    def prepare_process(self, app_id: str) -> None:
        """Once per process, before the first window (taskbar identity on Windows)."""

    def process_flags(self) -> int:
        """``creationflags`` for helper programs started from the windowed app (no console window on Windows)."""
        return 0

    def credential_backend(self) -> Any:
        from ..services.credential_service import MemoryCredentialBackend

        return MemoryCredentialBackend()

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "os": self.os_description(), "machine": self.machine, "apple_silicon": self.is_apple_silicon}

    # ------------------------------------------------------------------ helpers for subclasses
    def _capture(self, args: Sequence[str], timeout: float = 5.0) -> str:
        """stdout of a short helper command, or ``""`` if it is missing or fails (never raises)."""
        try:
            proc = self._run(list(args), capture_output=True, text=True, timeout=timeout, check=False)
        except (OSError, subprocess.SubprocessError):
            return ""
        return (getattr(proc, "stdout", "") or "").strip()


class PosixPlatform(Platform):
    name = "linux"
    display_name = "Linux"
