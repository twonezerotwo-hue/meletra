"""The INSTALLED MELETRA against the student's real profile: archive, rename, pages, About, a short transcription, and a read-only start on the real data.

    python scripts/installed_local_check.py --exe C:/Users/<you>/AppData/Local/Programs/MELETRA/MELETRA.exe --out OUT_DIR

Everything that writes happens on a COPY of the profile (``%LOCALAPPDATA%\\MedicalLectureTranscriber`` without its models, temp and cache; the speech model is found through the
"extra model folder" setting of the copy, so nothing is copied or linked): the pages are read through the accessibility tree and photographed from the program's own window, a lecture
is renamed through the real window with UI Automation, and a short English recording is transcribed with the real model. At the end the program is started ONCE on the real data,
only to read the archive and the window (nothing is clicked or typed); before and after, every file of the real profile is hashed and the difference is reported.
"""

from __future__ import annotations

import argparse
import hashlib
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
from src import __version__   # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))

import packaged_final_check as pfc            # noqa: E402
import packaged_visual_check as pvc            # noqa: E402
import smoke_packaged as sp                    # noqa: E402

REAL = Path(os.environ.get("LOCALAPPDATA", "")) / "MedicalLectureTranscriber"
SKIP = {"models", "temp", "cache"}


def snapshot(folder: Path, skip: set[str] = SKIP | {"logs"}) -> dict[str, str]:
    out = {}
    for path in sorted(folder.rglob("*")):
        if path.is_file() and not (set(path.relative_to(folder).parts[:1]) & skip):
            out[path.relative_to(folder).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def copy_profile(target: Path) -> None:
    shutil.copytree(REAL, target, ignore=shutil.ignore_patterns(*SKIP))
    settings_file = target / "settings" / "settings.json"
    settings = json.loads(settings_file.read_text(encoding="utf-8-sig"))
    settings["extra_model_dirs"] = [str(REAL / "models")]                                       # the copy finds the real model where it is
    settings_file.write_text(json.dumps(settings, indent=2), encoding="utf-8")


def main_window(pid: int, timeout: float = 60.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        for handle, title in sp.windows_of(pid):
            if title.startswith("MELETRA") and min(sp.window_size(handle)) >= 300:
                return handle, title
        time.sleep(0.4)
    return None, ""


def close(proc: subprocess.Popen, hwnd) -> int | None:
    sp.user32.PostMessageW(hwnd, 0x0010, 0, 0)
    try:
        return proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        return None


def entries(tree: list[dict], cls: str) -> list[dict]:
    return [e for e in tree if e.get("class") == cls]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-transcription", action="store_true")
    parser.add_argument("--rename-source", default="", help="rename the lecture whose recording has exactly this file name (default: the first lecture of the archive that has a job record)")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    exe = args.exe.resolve()
    checks: dict[str, bool] = {}
    notes: dict = {}
    real_before = snapshot(REAL)
    notes["real_files_before"] = len(real_before)

    work = Path(tempfile.mkdtemp(prefix="meletra_local_"))
    data = work / "data"
    copy_profile(data)
    jobs_before = {p.name: json.loads(p.read_text(encoding="utf-8")) for p in (data / "jobs").glob("*.json") if "events" not in p.name and "request" not in p.name}
    lecture, translated = pfc.make_lecture_folders(work)

    # --- 1. the pages of the installed program on the copy of the real profile
    report = pfc.run_size(exe, "1366x768", args.out, work, lecture, translated, "accessibility,home,files,settings,about")
    tree = report.get("accessibility", {})
    home, files = tree.get("home", []), tree.get("files", [])
    cards = sorted({e["name"] for e in entries(home, "LectureCard") + entries(files, "LectureCard")})
    combo = {e["name"]: e.get("value") for e in home if e.get("role") == "ComboBox"}
    about_text = " ".join(str(e.get("name", "")) + " " + str(e.get("value", "")) for e in tree.get("about", []))
    settings_text = " ".join(str(e.get("name", "")) for e in tree.get("settings", []))
    notes.update({"lecture_cards": cards, "combos": combo, "ui_check_errors": report.get("errors"), "screens": report.get("screens")})
    checks["the installed program ran its page check to the end without errors"] = not report.get("errors") and not report.get("timed_out") and report.get("exit_code") == 0
    checks["Home, Files, Settings and About were photographed from the program's own window"] = sum(1 for s in report.get("screens", []) if "FAILED" not in s) >= 4
    checks["the window and the sidebar say MELETRA"] = any(e.get("class") == "MainWindow" and e.get("name") == "MELETRA" for e in home) and any("MELETRA" in str(e.get("name", "")) for e in home)
    saved = json.loads((REAL / "settings" / "settings.json").read_text(encoding="utf-8-sig"))
    names = {"auto": "Auto Detect", "en": "English"}
    notes["saved_languages"] = {"language": saved.get("language"), "output_language": saved.get("output_language", "en")}
    checks["Home shows the languages saved in the profile (a new profile starts with Auto Detect / English)"] = (
        combo.get("Input language") == names.get(saved.get("language", "auto"), combo.get("Input language")) and combo.get("Output language") == names.get(saved.get("output_language", "en"), "English"))
    checks["the privacy chip says Local processing"] = any(e.get("name") == "Privacy status: Local processing" for e in home)
    checks["the Recent lectures / Files archive lists the lectures of the real profile"] = len(cards) >= len(jobs_before)
    checks[f"About says MELETRA, Version {__version__}"] = "MELETRA" in about_text and __version__ in about_text

    # --- 2. rename through the real window (UI Automation), on the copy
    env = {k: v for k, v in os.environ.items() if k not in ("QT_QPA_PLATFORM", "QT_QPA_FONTDIR")}
    env.update({"MLT_DATA_DIR": str(data), "MLT_ALLOW_MULTIPLE": "1"})
    proc = subprocess.Popen([str(exe)], env=env, cwd=str(work))
    hwnd, title = main_window(proc.pid)
    new_name = "Renamed in the installed check"
    home_cards = [e["name"] for e in entries(files, "LectureCard")]                                     # the archive on the Files page (Home shows only the newest few)
    target_job, target_source = None, ""
    for card in home_cards:                                                                               # the first card that belongs to a real job record (not a folder adopted into the archive)
        for job_file, rec in jobs_before.items():
            source = rec.get("source_name") or ""
            if source and card.startswith(f"Lecture {source},") and (not args.rename_source or source == args.rename_source):
                target_job, target_source = job_file, source
                break
        if target_job:
            break
    notes["rename_target"] = {"job_file": target_job, "source_name": target_source}
    renamed_line = ""
    if hwnd:
        time.sleep(2.0)
        done = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "scripts" / "uia_rename.ps1"), "-Hwnd", str(hwnd), "-NewName", new_name, "-Match", f"Actions for {target_source}*", "-Page", "Files"],
                              capture_output=True, text=True, timeout=180)
        renamed_line = (done.stdout or done.stderr).strip().splitlines()[-1] if (done.stdout or done.stderr).strip() else ""
        time.sleep(1.5)
        code = close(proc, hwnd)
    else:
        proc.kill()
        code = None
    jobs_after = {p.name: json.loads(p.read_text(encoding="utf-8")) for p in (data / "jobs").glob("*.json") if "events" not in p.name and "request" not in p.name}
    named = [n for n, rec in jobs_after.items() if rec.get("display_name") == new_name and n == target_job]
    notes["rename"] = {"ui_automation": renamed_line, "records_with_the_new_name": named}
    checks["the lecture was renamed through the window (card menu -> Rename -> Enter) and the program closed cleanly"] = bool(named) and code == 0
    checks["a rename changes the display name only: the recording's file name and the lecture folder are as before"] = bool(named) and all(
        jobs_after[n].get("source_name") == jobs_before[n].get("source_name") and jobs_after[n].get("output_dir") == jobs_before[n].get("output_dir") for n in named)

    # --- 3. a short real transcription with the real model, on the copy
    if not args.skip_transcription:
        audio = sp.ensure_audio()["gpu"]
        result_file = work / "result.json"
        started = time.monotonic()
        done = subprocess.run([str(exe), "--autorun", str(audio), "--result", str(result_file), "--timeout", "600", "--language", "en"], env=env, cwd=str(work), capture_output=True, text=True, timeout=700)
        result = json.loads(result_file.read_text(encoding="utf-8")) if result_file.is_file() else {}
        ok, detail = sp.check_outputs(Path(result["output_dir"]), ["glycolysis", "hexokinase", "pyruvate", "ATP"]) if result.get("output_dir") else (False, "no result")
        notes["transcription"] = {"state": result.get("state"), "device": result.get("device"), "seconds": round(time.monotonic() - started), "detail": detail}
        checks["a short English recording was transcribed by the installed program with the real model (all outputs written)"] = result.get("state") == "completed" and ok

    # --- 4. the real data: one read-only start
    env_real = {k: v for k, v in env.items() if k != "MLT_DATA_DIR"}
    proc = subprocess.Popen([str(exe)], env=env_real, cwd=str(work))
    hwnd, title = main_window(proc.pid)
    real_cards: list[str] = []
    if hwnd:
        time.sleep(3.0)
        pvc.capture(hwnd, args.out / "installed_on_real_data_home.png")
        dump = work / "real_home.json"
        subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "scripts" / "uia_dump.ps1"), "-Hwnd", str(hwnd), "-Out", str(dump)], capture_output=True, timeout=120)
        rows = json.loads(dump.read_text(encoding="utf-8-sig")) if dump.is_file() else []
        real_cards = sorted({r["name"] for r in rows if r.get("class") == "LectureCard"})
        real_title = rows[0]["name"] if rows else title
        code = close(proc, hwnd)
    else:
        proc.kill()
        real_title, code = "", None
    real_after = snapshot(REAL)
    changed = sorted(name for name in set(real_before) | set(real_after) if real_before.get(name) != real_after.get(name))
    notes["real_start"] = {"title": real_title, "lecture_cards": real_cards, "profile_files_changed_by_the_start": changed}
    checks["the installed program starts on the real data, shows the window MELETRA and the real archive, and closes cleanly"] = real_title == "MELETRA" and len(real_cards) >= 3 and code == 0
    checks["starting it changed no job record, transcript or Study file of the real profile"] = not [n for n in changed if n.startswith(("jobs/", "study/", "Transcripts/"))]

    result = {"exe": str(exe), "checks": checks, "ok": all(checks.values()), "notes": notes}
    (args.out / "installed_local_report.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    for name, ok in checks.items():
        print(("PASS " if ok else "FAIL ") + name)
    print(json.dumps({k: notes[k] for k in ("lecture_cards", "combos", "rename", "transcription", "real_start") if k in notes}, indent=1, default=str)[:3000])
    shutil.rmtree(work, ignore_errors=True)
    print("INSTALLED_LOCAL_CHECK_OK" if result["ok"] else "INSTALLED_LOCAL_CHECK_FAILED")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
