"""macOS (Apple Silicon first, Intel second): Library folders, ``open``, ``caffeinate``, ``sysctl``/``sw_vers``, Keychain.

None of this can run on the Windows machine the program is built on; every command goes through ``self._run`` so the tests drive it
with a fake runner, and the first real run happens on a Mac (see DEVELOPMENT_STATUS.md, "macOS: not tested on hardware").
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any, Optional

from .common import APP_ID, AcceleratorInfo, Platform


class MacPlatform(Platform):
    name = "macos"
    display_name = "macOS"
    exe_suffix = ""
    secure_store_label = "macOS Keychain"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._awake: Optional[Any] = None                   # the running `caffeinate` process

    # ------------------------------------------------------------------ folders
    def user_data_dir(self) -> Path:
        return self.home / "Library" / "Application Support" / APP_ID

    def cache_dir(self) -> Path:
        return self.home / "Library" / "Caches" / APP_ID

    def logs_dir(self) -> Path:
        return self.home / "Library" / "Logs" / APP_ID

    # ------------------------------------------------------------------ shell
    def open_path(self, path: Path) -> None:
        self._run(["open", str(path)], check=False)

    def reveal_path(self, path: Path) -> None:
        path = Path(path)
        self._run(["open", "-R", str(path)] if path.is_file() else ["open", str(path)], check=False)

    # ------------------------------------------------------------------ power
    def keep_awake(self, on: bool) -> bool:
        """``caffeinate -i`` (no idle sleep) for as long as it runs; released by ending the helper."""
        if on:
            if self._awake is not None:
                return True
            try:
                self._awake = subprocess.Popen(["caffeinate", "-i"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except (OSError, subprocess.SubprocessError):
                self._awake = None
                return False
            return True
        helper, self._awake = self._awake, None
        if helper is not None:
            try:
                helper.terminate()
            except OSError:
                pass
        return True

    # ------------------------------------------------------------------ system facts
    @property
    def is_apple_silicon(self) -> bool:
        return self.machine.lower() in ("arm64", "aarch64")

    def memory_status(self) -> dict[str, float]:
        gb = 1024 ** 3
        total = self._capture(["sysctl", "-n", "hw.memsize"])
        vm = self._capture(["vm_stat"])
        total_gb = round(int(total) / gb, 1) if total.isdigit() else 0.0
        page = re.search(r"page size of (\d+) bytes", vm)
        size = int(page.group(1)) if page else 4096
        free_pages = 0
        for label in ("Pages free", "Pages inactive", "Pages speculative"):
            match = re.search(rf"{label}:\s+(\d+)", vm)
            if match:
                free_pages += int(match.group(1))
        return {"total_gb": total_gb, "free_gb": round(free_pages * size / gb, 1)}

    def cpu_name(self) -> str:
        return self._capture(["sysctl", "-n", "machdep.cpu.brand_string"]) or ("Apple Silicon" if self.is_apple_silicon else "Intel")

    def os_description(self) -> str:
        version = self._capture(["sw_vers", "-productVersion"])
        return f"macOS {version} ({self.machine})" if version else f"macOS ({self.machine})"

    def accelerator(self) -> AcceleratorInfo:
        if self.is_apple_silicon:
            return AcceleratorInfo("apple", self.cpu_name(), "Metal / unified memory")
        return AcceleratorInfo("none", self.cpu_name(), "Intel Mac: the CPU is used")

    # ------------------------------------------------------------------ security
    def credential_backend(self) -> Any:
        from ..services.credential_service import KeyringBackend, MacKeychainBackend, MemoryCredentialBackend

        backend = KeyringBackend.try_create()
        if backend is not None:
            return backend
        try:
            return MacKeychainBackend(run=self._run)
        except OSError:
            return MemoryCredentialBackend()
