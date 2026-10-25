"""Final acceptance of the PACKAGED program: every page photographed from the real window, plus the accessibility tree and the native drop path.

    python scripts/packaged_final_check.py --exe dist/v4.1.0/MELETRA/MELETRA.exe --out OUT_DIR [--sizes 1366x768,1440x900,1920x1080]

For each size the packaged program is started with a private data folder and ``--ui-check`` (see ``src/gui/ui_check.py``). It prepares one page after the other and writes
``<page>.ready`` (the handle of the window to photograph); this script takes the picture with the operating system (``PrintWindow``: the window's own pixels, never the screen) and
writes ``<page>.captured``. Nothing is transcribed and nothing is sent. The report (``packaged_final_report.json``) holds the accessibility tree, the Tab order, the native drop
result, the list of pictures and the verdict.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import packaged_visual_check as pvc  # noqa: E402

SIZES = "1366x768,1440x900,1920x1080"
STEPS = ["home", "progress", "queue", "result", "transcript", "study", "files", "settings", "dropdown", "about"]            # the steps of ``ui_check`` that photograph a page
PAGES = ["home", "drag_overlay", "progress", "queue", "result", "transcript_translated", "study_summary", "study_flashcards", "study_questions", "files", "settings",
         "dropdown", "about"]                                                                                          # the pictures they make (the overlay comes from native_drop)


def make_lecture_folders(work: Path) -> tuple[Path, Path]:
    """A synthetic lecture (two topics) for Study and a lecture with a translated transcript; no audio, no model."""
    from tests.study_support import write_lecture_folder
    from tools.gui_shots import translated_lecture

    return write_lecture_folder(work / "lecture_rich"), translated_lecture(work / "lecture_translated")


def run_size(exe: Path, size: str, out: Path, work: Path, lecture: Path, translated: Path, pages: str) -> dict:
    out_dir = work / f"ui_{size}"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    env = {k: v for k, v in os.environ.items() if k not in ("QT_QPA_PLATFORM", "QT_QPA_FONTDIR")}          # (importing tools.gui_shots sets the offscreen platform: the program must run on the real one)
    env.update({"MLT_DATA_DIR": str(work / "data"), "MLT_ALLOW_MULTIPLE": "1"})
    command = [str(exe), "--ui-check", str(out_dir), "--size", size, "--lecture", str(lecture), "--translated", str(translated)]
    if pages:
        command += ["--ui-pages", pages]
    proc = subprocess.Popen(command, env=env, cwd=str(work))
    screens: list[str] = []
    deadline = time.monotonic() + 420
    started = time.monotonic()
    while proc.poll() is None and time.monotonic() < deadline:
        for ready in sorted(out_dir.glob("*.ready")):
            name = ready.stem
            if (out_dir / f"{name}.captured").exists():
                continue
            try:
                handle = int(ready.read_text(encoding="utf-8").strip())
            except ValueError:
                continue
            time.sleep(0.4)                                                           # the window has painted its new page
            target = out / f"packaged_{name}_{size}.png"
            try:
                pvc.capture(handle, target)
                screens.append(target.name)
            except Exception as exc:  # noqa: BLE001
                screens.append(f"{target.name}: FAILED {type(exc).__name__}: {exc}")
                print(f"[{size}] {target.name}: FAILED {type(exc).__name__}: {exc}", flush=True)
            (out_dir / f"{name}.captured").write_text("1", encoding="utf-8")
        time.sleep(0.15)
    timed_out = proc.poll() is None
    if timed_out:
        proc.kill()
    report_file = out_dir / "ui_check.json"
    report = json.loads(report_file.read_text(encoding="utf-8")) if report_file.exists() else {"errors": ["no ui_check.json"]}
    report["screens"] = screens
    report["exit_code"] = proc.returncode
    report["timed_out"] = timed_out
    report["seconds"] = round(time.monotonic() - started, 1)
    return report


def verdict(reports: dict[str, dict], pages: list[str]) -> dict:
    checks: dict[str, bool] = {}
    for size, report in reports.items():
        checks[f"{size}: ran to the end without errors"] = not report.get("errors") and not report.get("timed_out") and report.get("exit_code") == 0
        pictures = [s for s in report.get("screens", []) if "FAILED" not in s]
        wanted = [p for p in pages if p in PAGES]
        checks[f"{size}: every page photographed"] = all(any(s.startswith(f"packaged_{p}_") for s in pictures) for p in wanted)
    first = next(iter(reports.values()), {})
    accessibility = first.get("accessibility", {}).get("checks", {})
    checks["accessibility: all checks"] = bool(accessibility) and all(v is not False for v in accessibility.values())
    drop = first.get("native_drop", {})
    checks["native drop: overlay, drop once, unsupported file"] = bool(drop.get("ok"))
    return checks


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sizes", default=SIZES)
    parser.add_argument("--pages", default="", help="comma separated; default: accessibility, native_drop and every page")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="wm_final_"))
    pvc.seed(work / "data")
    lecture, translated = make_lecture_folders(work)
    reports: dict[str, dict] = {}
    for size in args.sizes.split(","):
        # the accessibility tree and the drop are the same at every size: they are measured once, at the first
        pages = args.pages or ("accessibility," if not reports else "") + "native_drop," + ",".join(STEPS)
        reports[size] = run_size(args.exe, size, args.out, work, lecture, translated, pages)
        print(f"[{size}] {len(reports[size].get('screens', []))} pictures, errors: {reports[size].get('errors')}", flush=True)
    checks = verdict(reports, (args.pages.split(",") if args.pages else PAGES))
    result = {"exe": str(args.exe), "checks": checks, "ok": all(checks.values()), "reports": reports}
    (args.out / "packaged_final_report.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    print(json.dumps(checks, indent=1))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
