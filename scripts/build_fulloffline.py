"""Build release/MELETRA-FullOffline-x64.zip: the portable package plus the Whisper large-v3 model.

The model (Systran/faster-whisper-large-v3, MIT licence) is taken from the installed models; its files are re-verified against
the pinned SHA-256 list first, so a damaged copy is never shipped. If no verified model is installed the package is NOT created.
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
from build_tools.packages import (                                    # noqa: E402
    MODEL_LICENSE, OFFLINE_README, PORTABLE_FLAG, extra_text_files, model_entries, portable_entries, write_zip,
)
from scripts.build_portable import DIST, RELEASE, verify_zip         # noqa: E402,F401
from src import PRODUCT_ID                                                # noqa: E402
from src.services.model_service import ModelService                   # noqa: E402
from src.services.paths import get_paths                              # noqa: E402

NAME = f"{PRODUCT_ID}-FullOffline-x64.zip"
MODEL_KEY = "whisper-large-v3"


def verify(archive: Path, audio: Path | None = None, timeout: float = 600.0) -> None:
    """Unzip the package into a temp folder and transcribe a short clip with the model that is INSIDE it: no data folder, no extra
    model folders, no internet needed. Proves the package is complete and works offline."""
    work = Path(tempfile.mkdtemp(prefix="mlt_offline_check_"))
    audio = audio or ROOT / "benchmarks" / "audio" / "biochem_clean.wav"
    try:
        print("[offline] verifying: unzipping ...", flush=True)
        with zipfile.ZipFile(archive) as z:
            z.extractall(work / "unzipped")
        app = work / "unzipped" / PRODUCT_ID
        env = windows_only_env(work)
        env.pop("MLT_DATA_DIR")                                       # portable mode: the data folder is next to the program
        result = work / "result.json"
        settings = app / "data" / "settings"
        settings.mkdir(parents=True, exist_ok=True)
        (settings / "settings.json").write_text(json.dumps({"schema_version": 2, "first_run_done": True, "open_folder_after": False,
                                                            "output_dir": str(work / "out")}), encoding="utf-8")
        env["QT_QPA_PLATFORM"] = "windows"
        proc = subprocess.Popen([str(app / f"{PRODUCT_ID}.exe"), "--autorun", str(audio), "--result", str(result), "--timeout", str(int(timeout - 30)),
                                 "--language", "en"], cwd=str(work), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        end = time.monotonic() + timeout
        while proc.poll() is None and time.monotonic() < end:
            time.sleep(1)
        if proc.poll() is None:
            proc.kill()
            raise SystemExit("the Full Offline package did not finish a transcription in time")
        data = json.loads(result.read_text(encoding="utf-8")) if result.exists() else {}
        if data.get("state") != "completed":
            raise SystemExit(f"the Full Offline package failed to transcribe: {data}")
        raw = next(Path(data["output_dir"]).glob("local_raw_transcript.txt")).read_text(encoding="utf-8").lower()
        if "glycolysis" not in raw:
            raise SystemExit("the Full Offline transcript does not contain the expected speech")
        print(f"[offline] verified: transcribed {data.get('audio_seconds', 0):.0f} s on {data.get('device')} with the bundled model, "
              f"{len(raw.split())} words, no internet and no other model folder", flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify(RELEASE / NAME)
        return 0
    problems = assemble.verify_dist(DIST, need_cuda=(DIST / "libs" / "cuda").is_dir(), need_ffmpeg=(DIST / "bin").is_dir())
    if problems:
        print("dist is incomplete (run BUILD_APP.bat):\n  " + "\n  ".join(problems))
        return 1
    models = ModelService(get_paths())
    info = models.info(MODEL_KEY)
    folder = models.locate(MODEL_KEY)
    if folder is None:
        print(f"NOT CREATED: {info.title} is not installed (download it in the app, or run scripts/download_model.py).")
        return 1
    print(f"[offline] verifying {folder} against the pinned checksums ...")
    check = models.verify(MODEL_KEY)
    if not check.ok:
        print("NOT CREATED: the installed model does not match the tested version: " + "; ".join(check.problems[:3]))
        return 1
    stage = Path(tempfile.mkdtemp(prefix="mlt_offline_stage_"))
    entries = portable_entries(DIST)
    entries += extra_text_files(stage, PRODUCT_ID, {PORTABLE_FLAG: "portable\n", "README_FULL_OFFLINE.txt": OFFLINE_README.format(exe=f"{PRODUCT_ID}.exe"),
                                                f"models/{info.folder}/LICENSE_MODEL.txt": MODEL_LICENSE})
    entries += model_entries(folder, info.folder)
    destination = RELEASE / NAME
    started = time.monotonic()
    last = [0.0]

    def progress(i: int, n: int) -> None:
        if time.monotonic() - last[0] > 20:
            last[0] = time.monotonic()
            print(f"[offline] {i}/{n} files ...", flush=True)

    print(f"[offline] zipping {len(entries)} files ...")
    write_zip(destination, entries, progress=progress)
    shutil.rmtree(stage, ignore_errors=True)
    print(f"[offline] {destination}  {assemble.human_size(destination.stat().st_size)}  in {time.monotonic() - started:.0f} s")
    if not args.no_verify:
        verify(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
