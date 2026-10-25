"""Build release\\MELETRA-Setup-Windows-x64.exe with Inno Setup (run build_app.py first).

    python scripts/build_installer.py            build, then verify (silent install into a temp folder, self-test, uninstall)
    python scripts/build_installer.py --no-verify

Inno Setup is installed per-user with winget when it is missing. The compiler is stopped if it prints nothing for
``--stall-seconds`` (default 900; LZMA compression of the CUDA DLLs is slow but never silent).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from build_tools import assemble                                      # noqa: E402
from build_tools.cleanenv import windows_only_env                     # noqa: E402
from build_tools.layout import base_version, dist_dir, release_dir   # noqa: E402
from src import PRODUCT_ID, __version__                                   # noqa: E402

DIST = dist_dir(__version__, ROOT)
RELEASE = release_dir(__version__, "Windows", ROOT)
SCRIPT = ROOT / "build_tools" / "installer.iss"
ICON = ROOT / "build" / "assets" / "app.ico"
OUTPUT_BASE = f"{PRODUCT_ID}-Setup-Windows-x64"
LOG = ROOT / "logs" / "build_installer.log"


def find_iscc() -> Path | None:
    candidates = [Path(p) for p in (os.environ.get("ISCC_PATH"), shutil.which("iscc")) if p]
    local = os.environ.get("LOCALAPPDATA")
    for base in (local and Path(local) / "Programs", Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")),
                 Path(os.environ.get("ProgramFiles", r"C:\Program Files"))):
        if base:
            candidates.append(base / "Inno Setup 6" / "ISCC.exe")
    return next((c for c in candidates if c.is_file()), None)


def ensure_inno_setup(timeout: float = 900.0) -> Path:
    found = find_iscc()
    if found:
        return found
    print("[installer] Inno Setup not found - installing it for this user with winget ...")
    proc = subprocess.Popen(["winget", "install", "--id", "JRSoftware.InnoSetup", "-e", "--scope", "user", "--silent",
                             "--accept-package-agreements", "--accept-source-agreements"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    end = time.monotonic() + timeout
    while proc.poll() is None and time.monotonic() < end:
        time.sleep(2)
    if proc.poll() is None:
        proc.kill()
        raise SystemExit("winget did not finish installing Inno Setup in time; install it from https://jrsoftware.org/isdl.php and retry")
    found = find_iscc()
    if not found:
        raise SystemExit("Inno Setup could not be installed automatically; install it from https://jrsoftware.org/isdl.php and retry")
    return found


def compile_installer(iscc: Path, stall_seconds: float) -> Path:
    RELEASE.mkdir(parents=True, exist_ok=True)
    target = RELEASE / f"{OUTPUT_BASE}.exe"
    target.unlink(missing_ok=True)
    cmd = [str(iscc), f"/DAppVersion={base_version(__version__)}", f"/DSourceDir={DIST}", f"/DOutputDir={RELEASE}", f"/DOutputBase={OUTPUT_BASE}",
           f"/DIconFile={ICON}", f"/DNoticeFile={ROOT / 'build_tools' / 'installer_notice.txt'}",
           *[f"/DWizard{name}={wizard_images(kind)}" for name, kind in (("Large", "large"), ("Small", "small")) if wizard_images(kind)], str(SCRIPT)]
    LOG.parent.mkdir(parents=True, exist_ok=True)
    last = [time.monotonic()]
    started = time.monotonic()
    with open(LOG, "w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")

        def pump() -> None:
            for line in proc.stdout:                                  # type: ignore[union-attr]
                log.write(line)
                log.flush()
                last[0] = time.monotonic()

        thread = threading.Thread(target=pump, daemon=True)
        thread.start()
        while proc.poll() is None:
            time.sleep(2)
            if time.monotonic() - last[0] > stall_seconds:
                proc.kill()
                raise SystemExit(f"Inno Setup printed nothing for {stall_seconds:.0f} s - stopped (see {LOG})")
        thread.join(5)
    if proc.returncode != 0 or not target.is_file():
        tail = LOG.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]
        raise SystemExit("Inno Setup failed:\n" + "\n".join(tail))
    print(f"[installer] built in {time.monotonic() - started:.0f} s: {target}  ({assemble.human_size(target.stat().st_size)})")
    return target


def run_with_timeout(cmd: list[str], timeout: float, env: dict[str, str] | None = None, cwd: Path | None = None) -> int | None:
    proc = subprocess.Popen(cmd, env=env, cwd=str(cwd) if cwd else None, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    end = time.monotonic() + timeout
    while proc.poll() is None and time.monotonic() < end:
        time.sleep(1)
    if proc.poll() is None:
        proc.kill()
        return None
    return proc.returncode


def wizard_images(kind: str) -> str:
    """The brand pictures of the setup wizard (100 / 150 / 200 % scale, comma separated); empty when they are not there (Inno Setup then uses its own)."""
    folder = ROOT / "assets" / "brand" / "installer"
    files = [folder / f"wizard_{kind}_{tag}.bmp" for tag in ("100", "150", "200")]
    return ",".join(str(f) for f in files) if all(f.is_file() for f in files) else ""


def verify_installer(installer: Path, install_timeout: float = 900.0) -> None:
    """Silent per-user install into a temp folder -> run the installed program's self-test -> silent uninstall."""
    work = Path(tempfile.mkdtemp(prefix="mlt_install_check_"))
    target = work / "MELETRA"
    log = work / "install.log"
    print(f"[installer] verifying: silent install into {target} ...")
    try:
        code = run_with_timeout([str(installer), "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/CURRENTUSER", "/NOICONS",
                                 f"/DIR={target}", f"/LOG={log}"], install_timeout)
        if code != 0:
            tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-10:] if log.exists() else []
            raise SystemExit(f"the silent install failed (exit code {code}):\n" + "\n".join(tail))
        exe = target / f"{PRODUCT_ID}.exe"
        uninstaller = next(iter(sorted(target.glob("unins*.exe"))), None)
        if not exe.is_file() or uninstaller is None:
            raise SystemExit("the installed folder lacks the program or the uninstaller")
        problems = assemble.verify_dist(target, need_cuda=True, need_ffmpeg=True)
        if problems:
            raise SystemExit("the installed program is incomplete: " + "; ".join(problems))
        out = work / "self.json"
        env = windows_only_env(work)
        code = run_with_timeout([str(exe), "--selftest", str(out)], 240, env=env, cwd=work)
        report = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {"ok": False}
        if not report.get("ok"):
            raise SystemExit(f"self-test of the INSTALLED program failed (exit code {code})")
        print("[installer] installed program passed its self-test; uninstalling ...")
        code = run_with_timeout([str(uninstaller), "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART"], 300)
        deadline = time.monotonic() + 30
        while exe.exists() and time.monotonic() < deadline:                # the uninstaller finishes its own cleanup a moment later
            time.sleep(1)
        if exe.exists():
            raise SystemExit("the uninstaller left the program behind")
        print("[installer] verified: install -> run -> uninstall all work")
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--verify-only", action="store_true", help="only test the existing installer (install, self-test, uninstall)")
    parser.add_argument("--stall-seconds", type=float, default=900.0)
    args = parser.parse_args(argv)
    if args.verify_only:
        verify_installer(RELEASE / f"{OUTPUT_BASE}.exe")
        return 0
    problems = assemble.verify_dist(DIST, need_cuda=(DIST / "libs" / "cuda").is_dir(), need_ffmpeg=(DIST / "bin").is_dir())
    if problems:
        raise SystemExit("dist is incomplete (run BUILD_APP.bat):\n  " + "\n  ".join(problems))
    if not ICON.is_file():
        from build_tools.icon import write_icon

        write_icon(ICON)
    installer = compile_installer(ensure_inno_setup(), args.stall_seconds)
    if not args.no_verify:
        verify_installer(installer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
