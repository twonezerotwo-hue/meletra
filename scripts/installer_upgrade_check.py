"""Upgrade check of build_tools/installer.iss WITHOUT the 1.3 GB program: tiny stand-in programs, the REAL installer script, a replica of the 4.0.0 installer script.

    python scripts/installer_upgrade_check.py [--keep] [--out DIR]

1. the installer script of the tag ``desktop-app-v4.0.0`` is compiled with a tiny ``MedicalLectureTranscriber.exe`` (the 4.0.0 product name, Start menu folder and uninstaller),
2. build_tools/installer.iss is compiled with a tiny ``MELETRA.exe``,
3. scripts/installer_upgrade_check.ps1 installs the first (per user, silent), installs the second over it, starts the program and tries again, and uninstalls,
   and the result lines are checked here. It needs Windows, Inno Setup (found like build_installer.py does) and the csc.exe that ships with Windows.

The user's real data folder is only listed before and after (never written); the installs go to a temp folder and %LOCALAPPDATA%\\Programs\\MELETRA, and are removed again.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_installer import find_iscc   # noqa: E402
from src import __version__   # noqa: E402

CSC = Path(r"C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe")
TAG = "desktop-app-v4.0.0"
FAKE = 'using System; using System.Threading; class P { static int Main(string[] a) { Console.WriteLine("stand-in program"); if (a.Length > 0 && a[0] == "--wait") Thread.Sleep(600000); return 0; } }'

EXPECTED = [
    ("old product installed", "registry: Medical Lecture Transcriber (Current user) 4.0.0 at "),
    ("old program file there", "old exe present: True"),
    ("upgrade exit code 0", None),
    (f"MELETRA registered, version {__version__}, in a folder of the new name", f"registry: MELETRA (Current user) {__version__} at "),
    ("new program file", "new exe present: True  (folder name: MELETRA)"),
    ("old folder removed", "old folder gone: True"),
    ("old program file removed", "old exe anywhere: False"),
    ("new files in, old files out", "new payload file: True  old payload file: False"),
    ("old Start menu folder removed", "old start menu folder gone: True"),
    ("exactly one program listed", "programs listed for this user: 1 (MELETRA"),
    ("a running program stops the upgrade", "exit with the program running: 7"),
    ("... and no file is touched", "files intact: True"),
    ("uninstall removes the registration and the folder", "registry key gone: True; folder gone: True"),
    ("the user's data folder is untouched", "real data folder unchanged: True"),
]


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, errors="replace", **kw)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--keep", action="store_true", help="keep the work folder")
    parser.add_argument("--out", type=Path, default=None, help="where to write the result (default: the work folder)")
    args = parser.parse_args()
    iscc = find_iscc()
    if not iscc or not CSC.is_file():
        print("SKIPPED: Inno Setup or the C# compiler is not available")
        return 2
    work = Path(tempfile.mkdtemp(prefix="mlt_upgrade_"))
    try:
        old_iss = work / "old.iss"
        done = run(["git", "show", f"{TAG}:build_tools/installer.iss"], cwd=str(ROOT))
        if done.returncode != 0:
            print(f"SKIPPED: the tag {TAG} is not in this clone")
            return 2
        old_iss.write_text(done.stdout, encoding="utf-8")
        (work / "fake.cs").write_text(FAKE, encoding="utf-8")
        for folder, exe, payload in (("old_dist", "MedicalLectureTranscriber.exe", "old.txt"), ("new_dist", "MELETRA.exe", "new.txt")):
            (work / folder / "sub").mkdir(parents=True)
            built = run([str(CSC), "/nologo", f"/out:{work / folder / exe}", str(work / "fake.cs")])
            assert built.returncode == 0, built.stdout + built.stderr
            (work / folder / "sub" / payload).write_text(payload, encoding="utf-8")
        out = work / "out"
        out.mkdir()
        icon = ROOT / "assets" / "brand" / "meletra.ico"
        notice = ROOT / "build_tools" / "installer_notice.txt"
        for script, version, folder, base, extra in ((old_iss, "4.0.0", "old_dist", "old-setup", []), (ROOT / "build_tools" / "installer.iss", __version__, "new_dist", "new-setup", [f"/DNoticeFile={notice}"])):
            compiled = run([str(iscc), "/Q", f"/DAppVersion={version}", f"/DSourceDir={work / folder}", f"/DOutputDir={out}", f"/DOutputBase={base}", f"/DIconFile={icon}", *extra, str(script)])
            if compiled.returncode != 0:
                print(compiled.stdout[-1500:], compiled.stderr[-500:])
                return 1
        result = (args.out or work) / "upgrade_check.txt"
        result.parent.mkdir(parents=True, exist_ok=True)
        ran = run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "scripts" / "installer_upgrade_check.ps1"), "-Work", str(work), "-Out", str(result)], timeout=900)
        text = result.read_text(encoding="utf-8", errors="replace") if result.is_file() else ran.stdout + ran.stderr
        failed = []
        exits = [line for line in text.splitlines() if line.startswith("exit ")]
        for label, needle in EXPECTED:
            ok = (len(exits) >= 2 and exits[1] == "exit 0") if needle is None else any(needle in line for line in text.splitlines())
            print(("PASS " if ok else "FAIL ") + label)
            if not ok:
                failed.append(label)
        if failed:
            print(text)
        print("UPGRADE_CHECK_OK" if not failed else "UPGRADE_CHECK_FAILED")
        return 1 if failed else 0
    finally:
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
