"""Open the PACKAGED Windows app with sample lectures and look at it: window title, icons, Home, Files, Settings at 1366x768, 1440x900 and 1920x1080.

    python scripts/packaged_visual_check.py --exe dist/v4.1.0/MELETRA/MELETRA.exe --out OUT_DIR

Only the app's OWN window is captured (``PrintWindow`` of its handle) - never the screen. The app runs with a private data folder (``MLT_DATA_DIR``) that this script fills with
a few finished and running sample lectures; the student's real data is not touched. Clicks are posted to the app's window (the sidebar), so nothing else on the desktop is used.
Writes OUT_DIR/*.png and OUT_DIR/packaged_visual_report.json; exit code 0 = the window opened, had its title and icon, every page was captured and the app closed cleanly.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import tempfile
import time
from ctypes import wintypes
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
shell32 = ctypes.WinDLL("shell32", use_last_error=True)

WM_CLOSE, WM_GETICON, WM_MOUSEMOVE, WM_LBUTTONDOWN, WM_LBUTTONUP = 0x0010, 0x007F, 0x0200, 0x0201, 0x0202
ICON_SMALL, ICON_BIG = 0, 1
GCLP_HICON, GCLP_HICONSM = -14, -34
PW_RENDERFULLCONTENT = 2
SIZES = ((1366, 768), (1440, 900), (1920, 1080))
NAV_Y = {"home": 116, "transcription": 160, "study": 204, "files": 248, "settings": 292}       # logical pixels in the window, measured on the rendered sidebar
NAV_X = 100

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:  # noqa: BLE001
    user32.SetProcessDPIAware()

EnumProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG), ("biHeight", wintypes.LONG), ("biPlanes", wintypes.WORD), ("biBitCount", wintypes.WORD),
                ("biCompression", wintypes.DWORD), ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG), ("biYPelsPerMeter", wintypes.LONG),
                ("biClrUsed", wintypes.DWORD), ("biClrImportant", wintypes.DWORD)]


def windows_of(pid: int) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []

    @EnumProc
    def callback(hwnd, _lparam):
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value == pid and user32.IsWindowVisible(hwnd):
            buffer = ctypes.create_unicode_buffer(256)
            user32.GetWindowTextW(hwnd, buffer, 256)
            found.append((hwnd, buffer.value))
        return True

    user32.EnumWindows(callback, 0)
    return found


def wait_window(pid: int, title_part: str, timeout: float = 60.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        for hwnd, title in windows_of(pid):
            if title_part.lower() in title.lower() and min(client_size(hwnd)) > 300:      # (a tiny temporary window with the same title shows up first while the app starts)
                return hwnd, title
        time.sleep(0.25)
    return None, ""


def client_size(hwnd) -> tuple[int, int]:
    rect = wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(rect))
    return rect.right - rect.left, rect.bottom - rect.top


def capture(hwnd, path: Path, proc=None):
    """``capture_once`` with a few retries (a window that is just being resized may report an empty client area for a moment)."""
    for attempt in range(4):
        try:
            return capture_once(hwnd, path)
        except ValueError:
            time.sleep(1.0)
    return capture_once(hwnd, path)


def capture_once(hwnd, path: Path):
    """The window's own pixels (PrintWindow renders it even when it is behind something): saved as PNG; returns (width, height)."""
    from PIL import Image

    for _ in range(20):                                                               # a window that is minimised or still opening has no client area yet
        if user32.IsIconic(hwnd):
            user32.ShowWindow(hwnd, 9)                                                    # SW_RESTORE
        width, height = client_size(hwnd)
        if width > 50 and height > 50:
            break
        time.sleep(0.5)
    window_dc = user32.GetDC(hwnd)
    mem_dc = gdi32.CreateCompatibleDC(window_dc)
    bitmap = gdi32.CreateCompatibleBitmap(window_dc, width, height)
    gdi32.SelectObject(mem_dc, bitmap)
    user32.PrintWindow(hwnd, mem_dc, PW_RENDERFULLCONTENT | 1)                       # 1 = PW_CLIENTONLY
    info = BITMAPINFOHEADER(ctypes.sizeof(BITMAPINFOHEADER), width, -height, 1, 32, 0, 0, 0, 0, 0, 0)
    buffer = ctypes.create_string_buffer(width * height * 4)
    gdi32.GetDIBits(mem_dc, bitmap, 0, height, buffer, ctypes.byref(info), 0)
    image = Image.frombuffer("RGBA", (width, height), buffer, "raw", "BGRA", 0, 1).convert("RGB")
    image.save(path)
    gdi32.DeleteObject(bitmap)
    gdi32.DeleteDC(mem_dc)
    user32.ReleaseDC(hwnd, window_dc)
    return width, height


def icon_to_image(hicon, size: int = 32):
    from PIL import Image

    screen = user32.GetDC(0)
    mem_dc = gdi32.CreateCompatibleDC(screen)
    bitmap = gdi32.CreateCompatibleBitmap(screen, size, size)
    old = gdi32.SelectObject(mem_dc, bitmap)
    user32.DrawIconEx(mem_dc, 0, 0, hicon, size, size, 0, 0, 3)                      # DI_NORMAL
    info = BITMAPINFOHEADER(ctypes.sizeof(BITMAPINFOHEADER), size, -size, 1, 32, 0, 0, 0, 0, 0, 0)
    buffer = ctypes.create_string_buffer(size * size * 4)
    gdi32.GetDIBits(mem_dc, bitmap, 0, size, buffer, ctypes.byref(info), 0)
    gdi32.SelectObject(mem_dc, old)
    gdi32.DeleteObject(bitmap)
    gdi32.DeleteDC(mem_dc)
    user32.ReleaseDC(0, screen)
    return Image.frombuffer("RGBA", (size, size), buffer, "raw", "BGRA", 0, 1).convert("RGB")


def has_content(image) -> bool:
    """True when an icon image is not blank: more than one colour and a coloured (teal / gold) pixel."""
    pixels = list(image.getdata())
    return len(set(pixels)) > 12 and any(abs(r - b) > 40 or abs(g - b) > 40 for r, g, b in pixels)


def click(hwnd, x: int, y: int) -> None:
    lparam = (y << 16) | (x & 0xFFFF)
    user32.PostMessageW(hwnd, WM_MOUSEMOVE, 0, lparam)
    user32.PostMessageW(hwnd, WM_LBUTTONDOWN, 1, lparam)
    user32.PostMessageW(hwnd, WM_LBUTTONUP, 0, lparam)


def resize_client(hwnd, width: int, height: int) -> None:
    style = user32.GetWindowLongW(hwnd, -16)
    rect = wintypes.RECT(0, 0, width, height)
    user32.AdjustWindowRect(ctypes.byref(rect), style, False)
    user32.SetWindowPos(hwnd, 0, 40, 40, rect.right - rect.left, rect.bottom - rect.top, 0x0004 | 0x0010)    # SWP_NOZORDER | SWP_NOACTIVATE
    time.sleep(1.0)


def screen_point(hwnd, x: int, y: int) -> tuple[int, int]:
    point = wintypes.POINT(x, y)
    user32.ClientToScreen(hwnd, ctypes.byref(point))
    return point.x, point.y


def other_windows(pid: int, main: int) -> list[tuple[int, str]]:
    return [(h, t) for h, t in windows_of(pid) if h != main and min(client_size(h)) > 20]


def drag_overlay(hwnd, work: Path, out: Path, size: str, report: dict) -> None:
    """A real OS drag of one audio file over the window; the app's own window is captured while the cursor is over it, then the drag is cancelled (nothing is dropped)."""
    import wave

    sample = work / "drag_sample.wav"
    with wave.open(str(sample), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 16000)
    left, top = screen_point(hwnd, 0, 0)
    width, height = client_size(hwnd)
    target_x, target_y = screen_point(hwnd, width // 2 + 100, height // 2)
    flag = str(work / "drag_flag")
    for suffix in (".hover", ".done"):
        Path(flag + suffix).unlink(missing_ok=True)
    helper = subprocess.Popen(["powershell", "-NoProfile", "-STA", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "scripts" / "packaged_drag_helper.ps1"), "-File", str(sample),
                               "-FormX", str(left + 380), "-FormY", str(top + 6), "-TargetX", str(target_x), "-TargetY", str(target_y), "-Flag", flag])
    end = time.monotonic() + 25
    while time.monotonic() < end and not Path(flag + ".hover").exists():
        time.sleep(0.1)
    ok = Path(flag + ".hover").exists()
    if ok:
        time.sleep(1.0)
        path = out / f"packaged_drag_overlay_{size}.png"
        capture(hwnd, path)
        report["screens"].append(path.name)
    Path(flag + ".done").write_text("1")
    try:
        helper.wait(timeout=20)
    except subprocess.TimeoutExpired:
        helper.kill()
    report["checks"]["drag_overlay_captured"] = ok


def dropdown_and_about(hwnd, pid: int, out: Path, dpi: float, report: dict) -> None:
    """The Input Language list opened, and the About box (Settings -> Advanced -> About)."""
    click(hwnd, int(NAV_X * dpi), int(NAV_Y["home"] * dpi))
    time.sleep(1.0)
    click(hwnd, int(415 * dpi), int(658 * dpi))                                           # the Input Language box of the start screen at 1366 x 768
    time.sleep(1.2)
    popups = other_windows(pid, hwnd)
    if popups:
        path = out / "packaged_dropdown_input_language.png"
        capture(popups[0][0], path)
        report["screens"].append(path.name)
        user32.PostMessageW(popups[0][0], 0x0100, 0x1B, 0)                                 # Esc closes the list
        user32.PostMessageW(popups[0][0], 0x0101, 0x1B, 0)
    report["checks"]["dropdown_captured"] = bool(popups)
    time.sleep(0.8)
    click(hwnd, int(NAV_X * dpi), int(NAV_Y["settings"] * dpi))
    time.sleep(1.5)
    dialogs = [(h, t) for h, t in windows_of(pid) if h != hwnd and t.strip().lower() == "settings"]
    if not dialogs:
        report["checks"]["about_captured"] = False
        return
    dialog = dialogs[0][0]
    click(dialog, 100, 231)                                                              # the Advanced group
    time.sleep(1.0)
    click(dialog, 344, 370)                                                              # "About MELETRA ..."
    time.sleep(1.5)
    about = [(h, t) for h, t in windows_of(pid) if h not in (hwnd, dialog) and "about" in t.lower()]
    if about:
        path = out / "packaged_about.png"
        capture(about[0][0], path)
        report["screens"].append(path.name)
        user32.PostMessageW(about[0][0], WM_CLOSE, 0, 0)
        time.sleep(0.8)
    report["checks"]["about_captured"] = bool(about)
    user32.PostMessageW(dialog, WM_CLOSE, 0, 0)
    time.sleep(0.8)


def seed(data_dir: Path) -> None:
    """A private data folder with finished, running and failed sample lectures (the same JSON the app writes)."""
    from src.services.job_service import JobStore
    from src.services.paths import AppPaths
    from src.services.settings_service import SettingsService

    paths = AppPaths(data_dir).ensure()
    settings = SettingsService(paths.settings_file)
    settings.update(first_run_done=True, share_notice_shown=True, theme="dark")
    store = JobStore(paths.jobs)
    lectures = data_dir / "lectures"

    def lecture(stem, **fields):
        folder = lectures / stem
        folder.mkdir(parents=True, exist_ok=True)
        text = "Glycolysis converts glucose into pyruvate.\nHexokinase phosphorylates glucose.\nPyruvate kinase makes the last ATP of the pathway.\n"
        (folder / "local_clean_transcript.txt").write_text(text, encoding="utf-8")
        (folder / "local_raw_transcript.txt").write_text(text, encoding="utf-8")
        (folder / "local_transcript.json").write_text(json.dumps({"language": "en", "segments": [{"start": 0.0, "end": 4.0, "text": text.splitlines()[0]}]}), encoding="utf-8")
        (folder / "metadata.json").write_text(json.dumps({"language": {"detected": "en"}}), encoding="utf-8")
        record = store.create_or_get("job-" + stem, data_dir / f"{stem}.mp4", output_dir=str(folder), state="completed", progress=100, **fields)
        store.update(record.job_id, state="completed")

    lecture("FIXTURE_0004", display_name="Neuroanatomy - Ventricular System", language="auto", detected_language="en", output_language="tr", translation="m2m100", duration_seconds=2700)
    lecture("FIXTURE_0003", language="auto", detected_language="en", output_language="en", translation="none", duration_seconds=1860)
    lecture("Pharma_lecture3", display_name="Pharmacology: autonomic drugs", language="tr", detected_language="tr", output_language="en", translation="whisper", duration_seconds=3300)
    lecture("Histology_week2", language="de", detected_language="de", output_language="en", translation="whisper", duration_seconds=4020)
    store.create_or_get("job-failed", data_dir / "Anatomy_thorax.mp4", state="failed", language="auto", output_language="en", duration_seconds=1500)
    store.create_or_get("job-queued", data_dir / "Physiology_renal.mp4", state="queued", language="auto", output_language="en", duration_seconds=2400)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sizes", default=",".join(f"{w}x{h}" for w, h in SIZES))
    ap.add_argument("--drag", action="store_true", help="also try a real OS drag over the window (moves the mouse pointer for about three seconds); off by default: synthesised input is not accepted by the OS drag loop on the machine this was written on")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="wm_visual_"))
    seed(work / "data")
    env = {**os.environ, "MLT_DATA_DIR": str(work / "data"), "MLT_ALLOW_MULTIPLE": "1"}
    report: dict = {"exe": str(args.exe), "checks": {}, "screens": []}
    proc = subprocess.Popen([str(args.exe)], env=env, cwd=str(work))
    report["pid"] = proc.pid
    try:
        hwnd, title = wait_window(proc.pid, "MELETRA")
        report["window_title"] = title
        report["checks"]["window_opened"] = bool(hwnd)
        if not hwnd:
            raise SystemExit("the packaged window did not open within 60 s")
        for _ in range(60):                                                               # wait until the window is laid out and has its icons (a cold start can take a while)
            if min(client_size(hwnd)) > 300 and (user32.SendMessageW(hwnd, WM_GETICON, ICON_BIG, 0) or user32.GetClassLongPtrW(hwnd, GCLP_HICON)):
                break
            time.sleep(0.5)
        time.sleep(1.5)
        small = user32.SendMessageW(hwnd, WM_GETICON, ICON_SMALL, 0) or user32.GetClassLongPtrW(hwnd, GCLP_HICONSM)
        big = user32.SendMessageW(hwnd, WM_GETICON, ICON_BIG, 0) or user32.GetClassLongPtrW(hwnd, GCLP_HICON)
        for name, handle in (("title_bar_icon", small), ("taskbar_icon", big)):
            if handle:
                image = icon_to_image(handle, 32)
                image.save(args.out / f"{name}.png")
                report["checks"][name] = has_content(image)
            else:
                report["checks"][name] = False
        large, small_exe = ctypes.c_void_p(), ctypes.c_void_p()
        count = shell32.ExtractIconExW(str(args.exe), 0, ctypes.byref(large), ctypes.byref(small_exe), 1)
        if count and large.value:
            image = icon_to_image(large.value, 32)
            image.save(args.out / "exe_icon.png")
            report["checks"]["exe_icon"] = has_content(image)
        else:
            report["checks"]["exe_icon"] = False
        dpi = user32.GetDpiForWindow(hwnd) / 96.0 if hasattr(user32, "GetDpiForWindow") else 1.0
        report["dpi_scale"] = dpi
        for size in args.sizes.split(","):
            width, height = (int(v) for v in size.split("x"))
            resize_client(hwnd, width, height)
            for page in ("home", "files", "settings"):
                click(hwnd, int(NAV_X * dpi), int(NAV_Y[page] * dpi))
                time.sleep(1.5)
                if page == "settings":
                    target = None
                    for _ in range(20):                                               # the dialog needs a moment to get its real size
                        dialogs = [(h, t) for h, t in windows_of(proc.pid) if h != hwnd and t.strip().lower() == "settings"]
                        if dialogs and min(client_size(dialogs[0][0])) > 50:
                            target = dialogs[0][0]
                            break
                        time.sleep(0.5)
                    if target is not None:
                        path = args.out / f"packaged_settings_{size}.png"
                        capture(target, path)
                        report["screens"].append(path.name)
                        user32.PostMessageW(target, WM_CLOSE, 0, 0)
                        time.sleep(1.0)
                    else:
                        report["checks"].setdefault("settings_opened", False)
                        continue
                    report["checks"]["settings_opened"] = True
                else:
                    path = args.out / f"packaged_{page}_{size}.png"
                    capture(hwnd, path)
                    report["screens"].append(path.name)
        first = args.sizes.split(",")[0]
        resize_client(hwnd, *(int(v) for v in first.split("x")))
        click(hwnd, int(NAV_X * dpi), int(NAV_Y["home"] * dpi))
        time.sleep(1.0)
        if args.drag:
            drag_overlay(hwnd, work, args.out, first, report)
        dropdown_and_about(hwnd, proc.pid, args.out, dpi, report)
        user32.PostMessageW(hwnd, WM_CLOSE, 0, 0)
        try:
            code = proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            code = None
            proc.kill()
        report["exit_code"] = code
        report["checks"]["closed_cleanly"] = code == 0
    finally:
        if proc.poll() is None:
            proc.kill()
    report["ok"] = all(report["checks"].values()) and len(report["screens"]) >= 3 * len(args.sizes.split(","))
    (args.out / "packaged_visual_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report["checks"], indent=1))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
