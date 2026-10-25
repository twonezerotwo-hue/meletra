"""Windows 10/11: %LOCALAPPDATA%, os.startfile, SetThreadExecutionState, Credential Manager, GlobalMemoryStatusEx."""

from __future__ import annotations

import ctypes
import os
import platform as _stdlib_platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from .common import APP_ID, AcceleratorInfo, Platform

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
CREATE_NO_WINDOW = 0x08000000


class WindowsPlatform(Platform):
    name = "windows"
    display_name = "Windows"
    exe_suffix = ".exe"
    secure_store_label = "Windows Credential Manager"

    # ------------------------------------------------------------------ folders
    def user_data_dir(self) -> Path:
        local = self.env.get("LOCALAPPDATA")
        base = Path(local) if local else self.home / "AppData" / "Local"
        return base / APP_ID

    # ------------------------------------------------------------------ shell
    def open_path(self, path: Path) -> None:
        os.startfile(str(path))                                         # noqa: S606 - the user asked to open this file

    def reveal_path(self, path: Path) -> None:
        path = Path(path)
        if path.is_file():
            self._run(["explorer", f"/select,{path}"], check=False)
        else:
            self.open_path(path)

    # ------------------------------------------------------------------ power
    def keep_awake(self, on: bool) -> bool:
        try:
            flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED if on else ES_CONTINUOUS
            return bool(int(ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint(flags))))
        except Exception:  # noqa: BLE001 - cosmetic feature, never a reason to fail
            return False

    # ------------------------------------------------------------------ system facts
    def memory_status(self) -> dict[str, float]:
        class _Status(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                        ("avail", ctypes.c_ulonglong), ("page_total", ctypes.c_ulonglong), ("page_avail", ctypes.c_ulonglong),
                        ("virt_total", ctypes.c_ulonglong), ("virt_avail", ctypes.c_ulonglong), ("ext", ctypes.c_ulonglong)]

        if sys.platform != "win32":
            return {"total_gb": 0.0, "free_gb": 0.0}
        status = _Status()
        status.length = ctypes.sizeof(_Status)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        gb = 1024 ** 3
        return {"total_gb": round(status.total / gb, 1), "free_gb": round(status.avail / gb, 1)}

    def cpu_name(self) -> str:
        if sys.platform == "win32":
            try:
                import winreg

                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                    return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
            except OSError:
                pass
        return _stdlib_platform.processor() or "unknown"

    def os_description(self) -> str:
        if sys.platform != "win32":
            return _stdlib_platform.platform()
        build = sys.getwindowsversion().build
        name = "Windows 11" if build >= 22000 else "Windows 10"
        return f"{name} (build {build}, {self.machine})"

    def accelerator(self) -> AcceleratorInfo:
        from ..diagnostics import query_nvidia_smi

        info = query_nvidia_smi(0)
        if info:
            return AcceleratorInfo("cuda", info["name"], f"{info['total_mib'] / 1024:.0f} GB video memory")
        return AcceleratorInfo("none")

    # ------------------------------------------------------------------ process / security
    def prepare_process(self, app_id: str) -> None:
        """Own taskbar identity (icon, grouping) instead of python.exe's."""
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
        except Exception:  # noqa: BLE001 - cosmetic
            pass

    def process_flags(self) -> int:
        return getattr(subprocess, "CREATE_NO_WINDOW", CREATE_NO_WINDOW)

    def credential_backend(self) -> Any:
        from ..services.credential_service import MemoryCredentialBackend, WindowsCredentialBackend

        try:
            return WindowsCredentialBackend()
        except OSError:
            return MemoryCredentialBackend()
