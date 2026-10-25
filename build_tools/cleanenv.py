"""An environment that looks like a plain Windows session but whose PATH holds only Windows itself.

Used to prove the packaged app does not borrow anything from the Python install, the project, CUDA or FFmpeg elsewhere on the PC.
(nvidia-smi needs the usual Windows variables - ProgramFiles, ProgramData, SystemDrive - or it cannot start NVML.)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Optional

KEEP = ("SystemRoot", "windir", "SystemDrive", "ProgramFiles", "ProgramFiles(x86)", "ProgramW6432", "CommonProgramFiles",
        "CommonProgramFiles(x86)", "CommonProgramW6432", "ProgramData", "ALLUSERSPROFILE", "PUBLIC", "COMPUTERNAME", "USERNAME",
        "USERDOMAIN", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "PROCESSOR_IDENTIFIER", "OS", "PATHEXT", "COMSPEC")


def windows_only_env(work: Path, extra: Optional[Mapping[str, str]] = None, source: Optional[Mapping[str, str]] = None) -> dict[str, str]:
    source = os.environ if source is None else source
    windir = source.get("SystemRoot", r"C:\Windows")
    env = {k: source[k] for k in KEEP if k in source}
    env.update({"SystemRoot": windir, "PATH": f"{windir}\\System32;{windir}", "TEMP": str(work / "tmp"), "TMP": str(work / "tmp"),
                "LOCALAPPDATA": str(work / "local"), "APPDATA": str(work / "roaming"), "USERPROFILE": str(work / "profile"),
                "MLT_DATA_DIR": str(work / "data"), "MLT_ALLOW_MULTIPLE": "1"})
    env.update(extra or {})
    for name in ("tmp", "local", "roaming", "profile", "data"):
        (Path(work) / name).mkdir(parents=True, exist_ok=True)
    return env
