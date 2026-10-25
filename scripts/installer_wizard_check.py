"""Walks the REAL Inno Setup wizard of the built installer page by page and photographs each page (the window's own pixels, never the screen).

    python scripts/installer_wizard_check.py --setup release/v4.1.0/Windows/MELETRA-Setup-Windows-x64.exe --out OUT_DIR

The installer is run for the current user only (``/CURRENTUSER``: no administrator prompt) into a temporary folder; at the end the program is uninstalled again
(``unins000.exe /VERYSILENT``) and the folder is checked to be gone. Pages: Welcome, Your privacy (the notice), Select Destination, Additional Tasks, Ready, Installing, Finish.
Nothing is started: the "Start MELETRA" box of the last page is cleared before Finish.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from ctypes import wintypes
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import packaged_visual_check as pvc  # noqa: E402

user32 = ctypes.WinDLL("user32", use_last_error=True)
EnumProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
BM_CLICK = 0x00F5
WM_KEYDOWN, WM_KEYUP, VK_SPACE = 0x0100, 0x0101, 0x20
LB_SETCURSEL = 0x0186
WM_GETTEXTLENGTH = 0x000E


def text_of(hwnd) -> str:
    buffer = ctypes.create_unicode_buffer(512)
    user32.GetWindowTextW(hwnd, buffer, 512)
    return buffer.value


def class_of(hwnd) -> str:
    buffer = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(hwnd, buffer, 256)
    return buffer.value


def process_of(hwnd) -> int:
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return pid.value


def top_windows() -> list[int]:
    found: list[int] = []

    @EnumProc
    def callback(hwnd, _):
        if user32.IsWindowVisible(hwnd):
            found.append(hwnd)
        return True

    user32.EnumWindows(callback, 0)
    return found


def children(hwnd) -> list[int]:
    found: list[int] = []

    @EnumProc
    def callback(child, _):
        found.append(child)
        return True

    user32.EnumChildWindows(hwnd, callback, 0)
    return found


def find_wizard():
    """The visible wizard window ("Setup - MELETRA", class TWizardForm) of any process."""
    for hwnd in top_windows():
        if class_of(hwnd) == "TWizardForm" and "Setup" in text_of(hwnd):
            return hwnd
    return None


def find_button(wizard, *labels: str):
    for child in children(wizard):
        if class_of(child) == "TNewButton" and user32.IsWindowVisible(child) and user32.IsWindowEnabled(child):
            label = text_of(child).replace("&", "")
            if any(label.startswith(wanted) for wanted in labels):
                return child
    return None


def page_texts(wizard) -> list[str]:
    return [t for t in (text_of(c) for c in children(wizard) if user32.IsWindowVisible(c)) if t.strip()]


def press(wizard, *labels: str, timeout: float = 30.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        button = find_button(wizard, *labels)
        if button:
            user32.PostMessageW(button, BM_CLICK, 0, 0)
            return True
        time.sleep(0.3)
    return False


def shot(wizard, out: Path, name: str, report: dict) -> None:
    user32.SetForegroundWindow(wizard)                                                  # a window behind others may not have painted its pictures yet
    user32.RedrawWindow(wizard, None, None, 0x0001 | 0x0004 | 0x0080 | 0x0100)         # INVALIDATE | ERASE | ALLCHILDREN | UPDATENOW
    time.sleep(2.0)
    path = out / f"installer_{name}.png"
    pvc.capture(wizard, path)
    report["pages"].append({"page": name, "image": path.name, "texts": page_texts(wizard)[:12]})


def wait_page(wizard, marker: str, timeout: float = 60.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if any(marker.lower() in t.lower() for t in page_texts(wizard)):
            return True
        time.sleep(0.3)
    return False


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    target = Path(tempfile.mkdtemp(prefix="wm_wizard_")) / "MELETRA"
    log = args.out / "installer_wizard.log"
    report: dict = {"setup": str(args.setup), "pages": [], "checks": {}}
    proc = subprocess.Popen([str(args.setup), "/CURRENTUSER", f"/DIR={target}", "/NOICONS", f"/LOG={log}"])
    try:
        wizard = None
        for _ in range(120):
            wizard = find_wizard()
            if wizard and min(pvc.client_size(wizard)) > 200:
                break
            time.sleep(0.5)
        report["checks"]["wizard_opened"] = bool(wizard)
        if not wizard:
            raise SystemExit("the installer wizard did not open")
        report["window_title"] = text_of(wizard)
        report["checks"]["title_is_meletra"] = "MELETRA" in report["window_title"] and "Medical Lecture" not in report["window_title"] and "Whisper" not in report["window_title"]
        wait_page(wizard, "Welcome")
        shot(wizard, args.out, "1_welcome", report)
        press(wizard, "Next")
        wait_page(wizard, "privacy")
        shot(wizard, args.out, "2_privacy_notice", report)
        press(wizard, "Next")
        wait_page(wizard, "Destination")
        shot(wizard, args.out, "3_location", report)
        press(wizard, "Next")
        wait_page(wizard, "Additional")
        shot(wizard, args.out, "4_tasks", report)
        press(wizard, "Next")
        wait_page(wizard, "Ready")
        shot(wizard, args.out, "5_ready", report)
        press(wizard, "Install")
        wait_page(wizard, "Installing")
        time.sleep(4.0)                                                                         # some progress on the bar
        shot(wizard, args.out, "6_installing", report)
        report["checks"]["no_inner_file_names_while_installing"] = not any("_internal" in t or ".dll" in t.lower() or ".pyd" in t.lower() for t in report["pages"][-1]["texts"])
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline and not wait_page(wizard, "is ready", timeout=2.0):
            if proc.poll() is not None:
                break
        shot(wizard, args.out, "7_finish", report)
        for child in children(wizard):                                                          # clear "Start MELETRA": nothing may be launched by this check
            if class_of(child) == "TNewCheckListBox" and user32.IsWindowVisible(child):
                user32.SendMessageW(child, LB_SETCURSEL, 0, 0)                                   # the list has no selected item until one is chosen: the space bar toggles the selected one
                user32.PostMessageW(child, WM_KEYDOWN, VK_SPACE, 0)
                user32.PostMessageW(child, WM_KEYUP, VK_SPACE, 0)
        time.sleep(0.6)
        press(wizard, "Finish")
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            pass
        exe = target / "MELETRA.exe"
        report["checks"]["installed"] = exe.is_file()
        report["checks"]["nothing_started"] = "MELETRA.exe" not in subprocess.run(["tasklist", "/FI", "IMAGENAME eq MELETRA.exe"], capture_output=True, text=True).stdout          # the program, not the Setup that is still closing
        uninstaller = target / "unins000.exe"
        if uninstaller.is_file():
            subprocess.run([str(uninstaller), "/VERYSILENT", "/SUPPRESSMSGBOXES"], timeout=300)
            time.sleep(3)
        report["checks"]["uninstalled_cleanly"] = not exe.exists()
    finally:
        subprocess.run(["powershell", "-NoProfile", "-Command", f"Get-Process -Name MELETRA -ErrorAction SilentlyContinue | Where-Object {{ $_.Path -like '{target}\\*' }} | Stop-Process -Force"],
                       capture_output=True, timeout=60)                                    # never leave a copy of THIS test's program running (only one that lives in the test's own folder)
        if proc.poll() is None:
            proc.kill()
        shutil.rmtree(target.parent, ignore_errors=True)
    report["ok"] = all(report["checks"].values()) and len(report["pages"]) == 7
    (args.out / "installer_wizard_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report["checks"], indent=1))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
