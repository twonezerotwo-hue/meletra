"""Smoke test of the INSTALLED program: silent per-user install of the real Setup, the full packaged smoke (scripts/smoke_packaged.py) against the installed exe, silent uninstall.

    python scripts/smoke_installed.py --setup release/v4.0.0/Windows/MedicalLectureTranscriber-Setup-Windows-x64.exe --model-dir C:/.../models --report release/v4.0.0/Windows/installed_smoke_report.json

Nothing of the source tree is used: the installed copy runs from a temporary folder with a Windows-only PATH (the smoke test builds that environment). The installation is removed
at the end whatever the result was; the exit code is the smoke test's.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXE_NAME = "MELETRA.exe"


def silent(args: list[str], timeout: float) -> int:
    return subprocess.run(args, timeout=timeout, check=False).returncode


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    a = ap.parse_args(argv)
    target = Path(tempfile.mkdtemp(prefix="mlt_installed_")) / "MELETRA"
    log = target.parent / "install.log"
    print(f"[installed] installing {a.setup.name} into {target} (per user, silent)")
    code = silent([str(a.setup.resolve()), "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/CURRENTUSER", "/NOICONS", f"/DIR={target}", f"/LOG={log}"], 1800)
    exe = target / EXE_NAME
    if code != 0 or not exe.is_file():
        print(f"[installed] FAILED: the installer exited with {code}; {exe} exists: {exe.is_file()}")
        return 1
    try:
        result = subprocess.run([sys.executable, str(ROOT / "scripts" / "smoke_packaged.py"), "--exe", str(exe), "--model-dir", str(a.model_dir), "--report", str(a.report)],
                                check=False).returncode
    finally:
        uninstaller = next(iter(sorted(target.glob("unins*.exe"))), None)
        if uninstaller is not None:
            silent([str(uninstaller), "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART"], 600)
            deadline = time.monotonic() + 60
            while exe.exists() and time.monotonic() < deadline:
                time.sleep(1.0)
        print(f"[installed] uninstalled: {not exe.exists()}")
    print("[installed] " + ("ALL PASSED" if result == 0 else f"smoke test failed (exit {result})"))
    return result


if __name__ == "__main__":
    sys.exit(main())
