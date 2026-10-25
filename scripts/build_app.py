"""Build the packaged Windows app:  ``python scripts/build_app.py [--no-cuda] [--no-ffmpeg] [--skip-pyinstaller]``

Steps: icon + version resource -> PyInstaller (one-folder, no UPX) -> CUDA runtime -> FFmpeg -> notices -> completeness check
-> the built .exe runs its own ``--selftest`` from a neutral folder with a clean PATH.

Nothing here waits blindly: PyInstaller output is streamed to a log and the build is stopped if it prints nothing for
``--stall-seconds`` (default 600); the self-test has a hard timeout.
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

from build_tools import assemble                                   # noqa: E402
from build_tools import lgpl_media                                 # noqa: E402
from build_tools.layout import base_version, dist_dir, dist_parent  # noqa: E402
from build_tools.cleanenv import windows_only_env                  # noqa: E402
from build_tools.icon import write_icon                            # noqa: E402
from build_tools.versioninfo import write_version_file             # noqa: E402
from src import APP_NAME, APP_TAGLINE, COMPANY, PRODUCT_ID, __version__             # noqa: E402

DIST = dist_dir(__version__, ROOT)
BUILD = ROOT / "build"
SPEC = ROOT / "build_tools" / f"{PRODUCT_ID}.spec"
LOG = ROOT / "logs" / "build_app.log"
FFMPEG_VENDOR = ROOT / "vendor" / "ffmpeg" / "bin"
LGPL_CACHE = ROOT / "build_cache" / "ffmpeg-lgpl"


def run_streaming(cmd: list[str], log_path: Path, stall_seconds: float, cwd: Path = ROOT, env: dict | None = None) -> int:
    """Run ``cmd``, copy its output to ``log_path``; kill it if it prints nothing for ``stall_seconds``."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    last = [time.monotonic()]
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace")

        def pump() -> None:
            for line in proc.stdout:                                  # type: ignore[union-attr]
                log.write(line)
                log.flush()
                last[0] = time.monotonic()

        thread = threading.Thread(target=pump, daemon=True)
        thread.start()
        while proc.poll() is None:
            time.sleep(1.0)
            if time.monotonic() - last[0] > stall_seconds:
                proc.kill()
                log.write(f"\n[build_app] no output for {stall_seconds:.0f} s - the build was stopped.\n")
                return 124
        thread.join(5)
    return proc.returncode


def prepare_assets() -> None:
    assets = BUILD / "assets"
    write_icon(assets / "app.ico")
    write_version_file(assets / "version_info.txt", base_version(__version__), COMPANY, APP_NAME, f"{APP_NAME} - {APP_TAGLINE}",
                       f"{PRODUCT_ID}.exe")
    print(f"[build] icon and version resource written to {assets}")


def build_pyinstaller(stall_seconds: float) -> None:
    if DIST.exists():
        shutil.rmtree(DIST)                                          # our own output folder only
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--distpath", str(dist_parent(__version__, ROOT)),
           "--workpath", str(BUILD / "pyinstaller"), str(SPEC)]
    print("[build] PyInstaller ... (log: logs/build_app.log)")
    # Do not let unrelated programs on the caller's PATH satisfy DLL imports while PyInstaller analyses the app.
    # In particular, a Poppler ICU build on PATH can be mistaken for Windows' ICU and make Qt fail only after packaging.
    code = run_streaming(cmd, LOG, stall_seconds, env=windows_only_env(BUILD / "pyinstaller_env"))
    if code != 0:
        tail = LOG.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
        raise SystemExit("PyInstaller failed (exit %s):\n%s" % (code, "\n".join(tail)))
    print(f"[build] PyInstaller finished: {DIST}")


def add_ffmpeg() -> bool:
    ffmpeg, ffprobe = (FFMPEG_VENDOR / "ffmpeg.exe", FFMPEG_VENDOR / "ffprobe.exe")
    if not (ffmpeg.is_file() and ffprobe.is_file()):
        print("[build] no bundled FFmpeg found in vendor/ffmpeg/bin - the app will decode with PyAV only")
        return False
    assemble.copy_ffmpeg(DIST, ffmpeg, ffprobe)
    licence = FFMPEG_VENDOR.parent / "LICENSE.txt"
    if licence.is_file():
        shutil.copy2(licence, DIST / "bin" / "FFMPEG_LICENSE.txt")
    print(f"[build] FFmpeg added: {ffmpeg}")
    return True


def add_lgpl_media(allow_gpl: bool) -> dict | None:
    """PyAV's own FFmpeg libraries are a GPL build (libx264 / libx265): replace them with the pinned LGPL-only build of the same release (build_tools/lgpl_media.py)."""
    internal = DIST / "_internal"
    try:
        archive = lgpl_media.pinned_archive(LGPL_CACHE)
    except Exception as exc:                                      # noqa: BLE001
        if allow_gpl:
            print(f"[build] WARNING: no LGPL FFmpeg build ({exc}) - this TEST build ships PyAV's GPL-class FFmpeg")
            return None
        raise SystemExit(f"the LGPL FFmpeg build is not available ({exc}); a release build never ships PyAV's GPL-class FFmpeg "
                         "(--allow-gpl-media makes a local test build only)")
    report = lgpl_media.swap_pyav(internal / "av", internal / "av.libs", archive)
    licences = DIST / "licenses"
    licences.mkdir(exist_ok=True)
    (licences / "FFMPEG_LGPL_LICENSE.txt").write_text(lgpl_media.license_text_from_zip(archive), encoding="utf-8")
    left = lgpl_media.gpl_class_files(DIST)
    if left:
        raise SystemExit("GPL-class media libraries are still in the build: " + ", ".join(left[:5]))
    (DIST / "media_libraries.json").write_text(json.dumps({**lgpl_media.PINNED, **report}, indent=1), encoding="utf-8")
    print(f"[build] LGPL FFmpeg {lgpl_media.PINNED['version']} replaces PyAV's GPL build: {len(report['removed'])} files out, {len(report['installed'])} in, "
          f"{report['patched_modules']} PyAV modules relinked; no GPL-class file left")
    return lgpl_media.PINNED


def add_cuda() -> bool:
    source = assemble.find_cuda_source()
    if source is None:
        print("[build] no CUDA 12 runtime found - building a CPU-only package")
        return False
    copied = assemble.copy_cuda(DIST, source)
    print(f"[build] CUDA runtime added from {source}: {len(copied)} DLLs, {assemble.human_size(sum(p.stat().st_size for p in copied))}")
    return True


def clean_env(data_dir: Path) -> dict[str, str]:
    """A PATH that holds only Windows itself: proves the build does not lean on Python, CUDA or FFmpeg installed elsewhere."""
    return windows_only_env(data_dir)


def packaged_selftest(timeout: float = 180.0) -> dict:
    exe = DIST / f"{PRODUCT_ID}.exe"
    work = Path(tempfile.mkdtemp(prefix="mlt_selftest_"))
    out = work / "selftest.json"
    started = time.monotonic()
    proc = subprocess.Popen([str(exe), "--selftest", str(out)], cwd=str(work), env=clean_env(work),
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    while proc.poll() is None and time.monotonic() - started < timeout:
        time.sleep(0.5)
    if proc.poll() is None:
        proc.kill()
        raise SystemExit(f"the packaged self-test did not finish within {timeout:.0f} s")
    if not out.exists():
        raise SystemExit(f"the packaged self-test wrote no report (exit code {proc.returncode})")
    return json.loads(out.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--no-ffmpeg", action="store_true")
    parser.add_argument("--skip-pyinstaller", action="store_true", help="re-assemble an existing dist folder")
    parser.add_argument("--stall-seconds", type=float, default=600.0)
    parser.add_argument("--allow-gpl-media", action="store_true", help="local test build without the LGPL FFmpeg swap (never for a release)")
    args = parser.parse_args(argv)
    started = time.monotonic()
    prepare_assets()
    if not args.skip_pyinstaller:
        build_pyinstaller(args.stall_seconds)
    media = add_lgpl_media(args.allow_gpl_media)
    has_ffmpeg = False if args.no_ffmpeg else add_ffmpeg()
    has_cuda = False if args.no_cuda else add_cuda()
    assemble.write_notices(DIST, media)
    assemble.write_extra_licenses(DIST)                                 # SentencePiece (Apache-2.0) and the optional M2M100 pack (MIT): licence texts next to the others
    shutil.copy2(ROOT / "README_APP.md", DIST / "README.txt")           # the user guide travels with the program
    problems = assemble.verify_dist(DIST, need_cuda=has_cuda, need_ffmpeg=has_ffmpeg)
    if problems:
        raise SystemExit("the build is incomplete:\n  " + "\n  ".join(problems))
    report = packaged_selftest()
    for check in report["checks"]:
        print(f"[selftest] {'ok  ' if check['ok'] else 'FAIL'} {check['name']}: {check['detail'][:110]}")
    if not report["ok"]:
        raise SystemExit("the packaged self-test failed")
    size = assemble.dir_size(DIST)
    print(f"[build] DONE in {time.monotonic() - started:.0f} s: {DIST / (PRODUCT_ID + '.exe')}  ({assemble.human_size(size)} unpacked)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
