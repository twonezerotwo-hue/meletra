"""Build release/MELETRA-Portable-Windows-x64.zip from dist/MELETRA (run build_app.py first).

The zip unpacks to ``MELETRA/`` and contains ``portable.flag`` (settings, models, logs and unfinished work then
live in ``data/`` next to the .exe) plus a short README. It is verified by unzipping and running the self-test from the copy.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from build_tools import assemble                                      # noqa: E402
from build_tools.cleanenv import windows_only_env                     # noqa: E402
from build_tools.packages import PORTABLE_FLAG, PORTABLE_README, extra_text_files, portable_entries, write_zip   # noqa: E402
from build_tools.layout import dist_dir, release_dir                 # noqa: E402
from src import PRODUCT_ID, __version__                                   # noqa: E402

DIST = dist_dir(__version__, ROOT)
RELEASE = release_dir(__version__, "Windows", ROOT)
NAME = f"{PRODUCT_ID}-Portable-Windows-x64.zip"


def build(dist: Path = DIST, destination: Path | None = None, verify: bool = True) -> Path:
    problems = assemble.verify_dist(dist, need_cuda=(dist / "libs" / "cuda").is_dir(), need_ffmpeg=(dist / "bin").is_dir())
    if problems:
        raise SystemExit("dist is incomplete (run BUILD_APP.bat):\n  " + "\n  ".join(problems))
    destination = destination or RELEASE / NAME
    stage = Path(tempfile.mkdtemp(prefix="mlt_portable_stage_"))
    entries = portable_entries(dist)
    entries += extra_text_files(stage, PRODUCT_ID, {PORTABLE_FLAG: "portable\n", "README_PORTABLE.txt": PORTABLE_README.format(exe=f"{PRODUCT_ID}.exe"),
                                                "models/README.txt": "Models placed here (one folder per model) are found automatically.\n"})
    started = time.monotonic()
    last = [0.0]

    def progress(i: int, n: int) -> None:
        if time.monotonic() - last[0] > 15:
            last[0] = time.monotonic()
            print(f"[portable] {i}/{n} files ...", flush=True)

    print(f"[portable] zipping {len(entries)} files ...")
    write_zip(destination, entries, progress=progress)
    shutil.rmtree(stage, ignore_errors=True)
    print(f"[portable] {destination}  {assemble.human_size(destination.stat().st_size)}  in {time.monotonic() - started:.0f} s")
    if verify:
        verify_zip(destination)
    return destination


def verify_zip(archive: Path, timeout: float = 240.0) -> None:
    """Unzip the package somewhere else and run its own self-test: proves the zip is complete and portable mode works."""
    work = Path(tempfile.mkdtemp(prefix="mlt_portable_check_"))
    try:
        with zipfile.ZipFile(archive) as z:
            bad = z.testzip()
            if bad:
                raise SystemExit(f"the zip is corrupt at {bad}")
            z.extractall(work / "unzipped")
        app = work / "unzipped" / PRODUCT_ID
        if not (app / PORTABLE_FLAG).is_file():
            raise SystemExit("portable.flag is missing from the package")
        env = windows_only_env(work)
        env.pop("MLT_DATA_DIR")                                       # portable mode must find its own data folder
        out = work / "self.json"
        proc = subprocess.Popen([str(app / f"{PRODUCT_ID}.exe"), "--selftest", str(out)], cwd=str(work), env=env,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        end = time.monotonic() + timeout
        while proc.poll() is None and time.monotonic() < end:
            time.sleep(0.5)
        if proc.poll() is None:
            proc.kill()
            raise SystemExit("the unzipped portable app did not finish its self-test in time")
        report = json.loads(out.read_text(encoding="utf-8"))
        if not report["ok"]:
            raise SystemExit("self-test of the unzipped portable app failed: " + json.dumps(report["checks"])[:600])
        data_dir = app / "data"
        if not (data_dir / "settings").is_dir():
            raise SystemExit("portable mode did not create its data folder next to the program")
        print(f"[portable] verified: unzipped copy runs, data folder: {data_dir.name}/ next to the program")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()
    build(verify=not args.no_verify)
