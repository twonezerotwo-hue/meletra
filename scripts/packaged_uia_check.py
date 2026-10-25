"""The packaged program as seen by a real Windows UI Automation client (names, control types, keyboard focus).

    python scripts/packaged_uia_check.py --exe dist/v4.1.0/MELETRA/MELETRA.exe --out OUT_DIR

Starts the program with a private, seeded data folder, reads the UIA tree of the start screen with ``scripts/uia_dump.ps1`` and checks the names an assistive technology
needs ("Browse lecture files", "Input language", "Lecture ..., completed", the sidebar). A UIA client is NOT a screen reader: how NVDA / JAWS / Narrator speak the page was not tested.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import packaged_visual_check as pvc  # noqa: E402

TYPE_BUTTON, TYPE_COMBO, TYPE_EDIT = "ControlType.Button", "ControlType.ComboBox", "ControlType.Edit"


def read_tree(hwnd: int, out_file: Path) -> list[dict]:
    done = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "scripts" / "uia_dump.ps1"),
                           "-Hwnd", str(hwnd), "-Out", str(out_file)], capture_output=True, text=True, timeout=240)
    if not out_file.exists():
        raise SystemExit(f"UIA dump failed: {done.stderr[-800:]}")
    data = json.loads(out_file.read_text(encoding="utf-8-sig"))
    return data if isinstance(data, list) else [data]


def find(rows: list[dict], name: str, kind: str = "") -> dict | None:
    return next((r for r in rows if (r.get("name") or "").startswith(name) and (not kind or r.get("type") == kind)), None)


def wait_window_any(title_part: str, timeout: float = 90.0):
    """The window of any process (a virtualenv's python.exe starts a second process for the real one)."""
    import ctypes
    from ctypes import wintypes

    user32 = pvc.user32
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        found = []

        @pvc.EnumProc
        def callback(hwnd, _lparam):
            if user32.IsWindowVisible(hwnd):
                buffer = ctypes.create_unicode_buffer(256)
                user32.GetWindowTextW(hwnd, buffer, 256)
                if title_part.lower() in buffer.value.lower() and min(pvc.client_size(hwnd)) > 300:
                    found.append((hwnd, buffer.value))
            return True

        user32.EnumWindows(callback, 0)
        if found:
            return found[0]
        time.sleep(0.5)
    return None, ""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--launch-args", nargs="*", default=[], help="arguments in front of the program (to try it with `python launcher.py` before it is built)")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="wm_uia_"))
    pvc.seed(work / "data")
    env = {**os.environ, "MLT_DATA_DIR": str(work / "data"), "MLT_ALLOW_MULTIPLE": "1"}
    proc = subprocess.Popen([str(args.exe), *args.launch_args], env=env, cwd=str(work))
    report: dict = {"exe": str(args.exe), "checks": {}}
    try:
        hwnd, title = wait_window_any("MELETRA") if args.launch_args else pvc.wait_window(proc.pid, "MELETRA")
        if not hwnd:
            raise SystemExit("the window did not open")
        time.sleep(4.0)                                                                  # the window has its recent lectures
        rows = read_tree(hwnd, args.out / "uia_tree_home.json")
        report["elements"] = len(rows)
        checks = report["checks"]
        browse = find(rows, "Browse lecture files", TYPE_BUTTON)
        checks["browse_lecture_files_is_a_focusable_button"] = bool(browse and browse["focusable"])
        pdf = find(rows, "Add lecture PDF", TYPE_BUTTON)
        checks["add_lecture_pdf_is_a_button"] = bool(pdf)
        combo = find(rows, "Input language", TYPE_COMBO)
        checks["input_language_is_a_combo_box"] = bool(combo and combo["focusable"])
        checks["output_language_is_a_combo_box"] = bool(find(rows, "Output language", TYPE_COMBO))
        checks["transcribe_is_a_button"] = bool(find(rows, "TRANSCRIBE", TYPE_BUTTON))
        cards = [r for r in rows if (r.get("name") or "").startswith("Lecture ") and ", " in r["name"] and r.get("class") == "LectureCard"]
        checks["lecture_cards_are_named_buttons"] = bool(cards) and all(r["type"] == TYPE_BUTTON and r["focusable"] for r in cards)
        report["lecture_cards"] = [r["name"] for r in cards]
        checks["card_menus_are_named"] = bool(find(rows, "Actions for "))
        checks["sidebar_items_are_named"] = all(find(rows, n) for n in ("Home", "Transcription", "Study", "Files", "Settings"))
        checks["window_is_named_meletra"] = (rows[0].get("name") or "") == "MELETRA"
        unnamed = [r for r in rows if r.get("focusable") and r.get("enabled") and not (r.get("name") or "").strip() and r.get("type") not in ("ControlType.Text", "ControlType.Pane", "ControlType.Window", "ControlType.Custom")]
        checks["no_focusable_control_without_a_name"] = not unnamed
        report["unnamed_focusable"] = [{k: r.get(k) for k in ("type", "class", "automationId")} for r in unnamed][:10]
        user32 = pvc.user32
        user32.PostMessageW(hwnd, pvc.WM_CLOSE, 0, 0)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            pass
    finally:
        if proc.poll() is None:
            proc.kill()
    report["ok"] = all(report["checks"].values())
    (args.out / "packaged_uia_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({"checks": report["checks"], "lecture_cards": report.get("lecture_cards"), "unnamed_focusable": report.get("unnamed_focusable")}, indent=1))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
