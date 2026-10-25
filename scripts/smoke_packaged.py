"""Smoke-test the PACKAGED app (dist\\MedicalLectureTranscriber\\MedicalLectureTranscriber.exe) from outside the project.

Every step starts the .exe in a neutral working folder with a PATH that holds only Windows itself, so nothing can be borrowed from
the Python install, the project, CUDA or FFmpeg elsewhere on this PC.

    python scripts/smoke_packaged.py            all steps (GUI, GPU run, CPU fallback, PyAV fallback)
    python scripts/smoke_packaged.py --quick    start-up checks only (self-test, window opens and closes, diagnostics) - no model needed
    python scripts/smoke_packaged.py --exe PATH --model-dir DIR --report FILE

Hang rule: each step has its own timeout; a step that shows no progress is killed and reported as FAILED (never waited for).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import shutil
import os
import subprocess
import sys
import tempfile
import time
from ctypes import wintypes
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from build_tools.cleanenv import windows_only_env                           # noqa: E402
from build_tools.layout import dist_dir, release_dir                        # noqa: E402
from src import PRODUCT_ID, APP_NAME, __version__                               # noqa: E402

DEFAULT_EXE = dist_dir(__version__, ROOT) / f"{PRODUCT_ID}.exe"
DEFAULT_REPORT = release_dir(__version__, "Windows", ROOT) / "packaged_smoke_report.json"
AUDIO_DIR = ROOT / "benchmarks" / "audio"
OUTPUT_FILES = ["local_clean_transcript.txt", "local_raw_transcript.txt", "local_transcript_with_timestamps.txt",
                "local_transcript.srt", "local_transcript.vtt", "local_transcript.json"]


class Step:
    def __init__(self, name: str) -> None:
        self.name, self.ok, self.detail, self.seconds = name, False, "", 0.0

    def as_dict(self) -> dict:
        return {"name": self.name, "ok": self.ok, "detail": self.detail, "seconds": round(self.seconds, 1)}


def clean_env(work: Path, extra: dict[str, str] | None = None) -> dict[str, str]:
    return windows_only_env(work, {"QT_QPA_PLATFORM": "windows", **(extra or {})})


def write_settings(work: Path, **values) -> None:
    folder = work / "data" / "settings"
    folder.mkdir(parents=True, exist_ok=True)
    # vram_warnings off: when another program (the owner's llama-server) holds the GPU, the app asks "GPU memory is low - Continue Anyway?" and an unattended run would wait for a click for ever
    (folder / "settings.json").write_text(json.dumps({"schema_version": 2, "first_run_done": True, "vram_warnings": False, **values}), encoding="utf-8")


def run_exe(exe: Path, args: list[str], env: dict[str, str], cwd: Path, timeout: float) -> tuple[int | None, bool]:
    """Start, wait up to ``timeout`` s, kill on timeout. Returns ``(exit code or None, timed_out)``."""
    proc = subprocess.Popen([str(exe), *args], cwd=str(cwd), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    end = time.monotonic() + timeout
    while proc.poll() is None and time.monotonic() < end:
        time.sleep(0.5)
    if proc.poll() is None:
        proc.kill()
        return None, True
    return proc.returncode, False


# ----------------------------------------------------------------------------------------------- windows (GUI launch / close)
user32 = ctypes.windll.user32 if sys.platform == "win32" else None


def windows_of(pid: int) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def callback(hwnd, _lparam):
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value == pid and user32.IsWindowVisible(hwnd):
            length = user32.GetWindowTextLengthW(hwnd)
            buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buffer, length + 1)
            found.append((hwnd, buffer.value))
        return True

    user32.EnumWindows(callback_type(callback), 0)
    return found


def window_size(hwnd) -> tuple[int, int]:
    rect = wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    return rect.right - rect.left, rect.bottom - rect.top


QUESTION_TITLE = "Your transcripts and your privacy"                       # the first-launch question about sharing transcript text with MELETRA


def step_gui_launch(exe: Path, work: Path, first_launch: bool = True) -> Step:
    """Start the real program. The FIRST launch of a profile asks the sharing question once: it must appear, closing it (the X button) means "keep transcripts private" and is saved;
    on the next launch of the same profile it is not asked again. Then the window closes cleanly."""
    step = Step("GUI launch, window shown, the sharing question answered, clean close" if first_launch else "GUI launch again: the question is not asked twice, clean close")
    started = time.monotonic()
    if first_launch:
        write_settings(work)                                              # a restart keeps the profile exactly as the first launch left it (the saved answer included)
    proc = subprocess.Popen([str(exe)], cwd=str(work), env=clean_env(work), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        hwnd = None
        end = time.monotonic() + 60
        while time.monotonic() < end and proc.poll() is None:
            for handle, title in windows_of(proc.pid):
                if title.startswith(APP_NAME) and min(window_size(handle)) >= 200:                   # not the tiny temporary window the program shows while it starts
                    hwnd = handle
            if hwnd:
                break
            time.sleep(0.5)
        if hwnd is None:
            step.detail = f"no '{APP_NAME}' window within 60 s (process {'exited with ' + str(proc.poll()) if proc.poll() is not None else 'still running'})"
            return step
        shown_after = time.monotonic() - started
        time.sleep(2.0)                                                   # let the start-up GPU check finish
        asked = ""
        if first_launch:
            question = None
            end = time.monotonic() + 20
            while time.monotonic() < end and question is None and proc.poll() is None:
                question = next((h for h, t in windows_of(proc.pid) if t == QUESTION_TITLE), None)
                time.sleep(0.5)
            if question is None:
                step.detail = "the sharing question did not appear on the first launch"
                return step
            user32.PostMessageW(question, 0x0010, 0, 0)                   # WM_CLOSE: the X button - no consent was given
            time.sleep(2.0)
            saved = json.loads((work / "data" / "settings" / "settings.json").read_text(encoding="utf-8"))
            if not (saved.get("share_choice_made") is True and saved.get("share_transcripts") is False and saved.get("share_notice_shown") is False):
                step.detail = f"closing the question did not leave sharing off and answered: {({k: saved.get(k) for k in ('share_choice_made', 'share_transcripts', 'share_notice_shown')})}"
                return step
            asked = "; the sharing question appeared, was closed (= keep private) and saved"
        else:
            time.sleep(3.0)
            if any(t == QUESTION_TITLE for _h, t in windows_of(proc.pid)):
                step.detail = "the sharing question was asked again on the second launch"
                return step
            asked = "; the sharing question was not asked again"
        user32.PostMessageW(hwnd, 0x0010, 0, 0)                           # WM_CLOSE
        try:
            code = proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            step.detail = "the window did not close within 30 s"
            return step
        step.ok = code == 0
        step.detail = f"window appeared after {shown_after:.1f} s, closed cleanly (exit code {code}){asked}"
        return step
    finally:
        if proc.poll() is None:
            proc.kill()
        step.seconds = time.monotonic() - started


def step_selftest(exe: Path, work: Path) -> Step:
    step = Step("self-test (PyAV decode, model discovery, GPU detection, CPU fallback, output folder)")
    started = time.monotonic()
    out = work / "selftest.json"
    code, timed_out = run_exe(exe, ["--selftest", str(out)], clean_env(work), work, 180)
    step.seconds = time.monotonic() - started
    if timed_out or not out.exists():
        step.detail = "timed out" if timed_out else f"no report written (exit code {code})"
        return step
    report = json.loads(out.read_text(encoding="utf-8"))
    bad = [f"{c['name']}: {c['detail']}" for c in report["checks"] if not c["ok"]]
    step.ok = report["ok"] and code == 0
    step.detail = "; ".join(f"{c['name']} ok" for c in report["checks"] if c["ok"])[:300] if step.ok else "; ".join(bad)
    (work / "selftest_report.txt").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return step


def step_diagnostics(exe: Path, work: Path) -> Step:
    step = Step("diagnostics report")
    started = time.monotonic()
    out = work / "diagnostics.txt"
    code, timed_out = run_exe(exe, ["--diagnostics", str(out)], clean_env(work), work, 120)
    step.seconds = time.monotonic() - started
    if timed_out or not out.exists():
        step.detail = "timed out" if timed_out else f"no report (exit code {code})"
        return step
    text = out.read_text(encoding="utf-8")
    step.ok = all(w in text for w in ("Windows", "CPU", "GPU", "CUDA", "FFmpeg", "Model", "Data folder"))
    step.detail = " | ".join(line.strip() for line in text.splitlines() if line.startswith(("GPU", "CUDA", "FFmpeg", "Audio decoder")))[:400]
    return step


def step_download(exe: Path, work: Path) -> Step:
    step = Step("REAL model download over HTTPS (tiny test model, ~75 MB) + SHA-256 verification")
    started = time.monotonic()
    out = work / "download.json"
    code, timed_out = run_exe(exe, ["--download-model", "whisper-tiny-test", "--result", str(out)], clean_env(work), work, 600)
    step.seconds = time.monotonic() - started
    if timed_out or not out.exists():
        step.detail = "timed out after 600 s" if timed_out else f"no result written (exit code {code})"
        return step
    data = json.loads(out.read_text(encoding="utf-8"))
    step.ok = bool(data.get("ok")) and code == 0 and data.get("size_on_disk", 0) > 10_000_000
    step.detail = (f"{data.get('files')} files, {data.get('size_on_disk', 0) / 1024 ** 2:.0f} MB, verified={data.get('verified')} in "
                   f"{data.get('seconds')} s" if step.ok else str(data.get("error") or data.get("problems")))
    return step


# ----------------------------------------------------------------------------------------------- real transcription
def ensure_audio() -> dict[str, Path]:
    wanted = {"gpu": AUDIO_DIR / "biochem_clean.wav", "cpu": AUDIO_DIR / "neuro_hard.wav", "pyav": AUDIO_DIR / "pharma_hard.wav"}
    if not all(p.is_file() for p in wanted.values()):
        subprocess.run([sys.executable, str(ROOT / "scripts" / "make_benchmark_audio.py")], cwd=str(ROOT), timeout=300, check=False)
    missing = [str(p) for p in wanted.values() if not p.is_file()]
    if missing:
        raise SystemExit("benchmark audio is missing and could not be generated: " + ", ".join(missing))
    return wanted


def to_mp3(wav: Path, target: Path) -> Path:
    """Compress with PyAV (libmp3lame) so the packaged app has to decode a real MP3, not just a WAV."""
    import av

    with av.open(str(wav)) as src, av.open(str(target), "w") as dst:
        stream_in = src.streams.audio[0]
        out = dst.add_stream("mp3", rate=16000)
        out.layout = "mono"
        for frame in src.decode(stream_in):
            frame.pts = None
            for packet in out.encode(frame):
                dst.mux(packet)
        for packet in out.encode(None):
            dst.mux(packet)
    return target


def check_outputs(out_dir: Path, expect_words: list[str]) -> tuple[bool, str]:
    missing = [n for n in OUTPUT_FILES if not (out_dir / n).is_file() or (out_dir / n).stat().st_size == 0]
    if missing:
        return False, "missing/empty outputs: " + ", ".join(missing)
    data = json.loads((out_dir / "local_transcript.json").read_text(encoding="utf-8"))
    segments = data.get("segments") or []
    srt = (out_dir / "local_transcript.srt").read_text(encoding="utf-8")
    vtt = (out_dir / "local_transcript.vtt").read_text(encoding="utf-8")
    raw = (out_dir / "local_raw_transcript.txt").read_text(encoding="utf-8").lower()
    problems = []
    if not segments:
        problems.append("JSON has no segments")
    if "-->" not in srt or not srt.lstrip().startswith("1"):
        problems.append("SRT is malformed")
    if not vtt.startswith("WEBVTT"):
        problems.append("VTT is malformed")
    found = [w for w in expect_words if w.lower() in raw]
    if len(found) < max(1, len(expect_words) // 2):
        problems.append(f"expected terms not heard: {sorted(set(expect_words) - set(found))}")
    words = len(raw.split())
    return (not problems), (f"{words} words, {len(segments)} segments; terms heard: {', '.join(found)}" if not problems else "; ".join(problems))


def step_transcribe(name: str, exe: Path, work: Path, audio: Path, model_dir: Path, settings: dict, env_extra: dict[str, str],
                    expect_words: list[str], expect_device: str | None, timeout: float, pause_at: float = 0.0) -> Step:
    step = Step(name)
    started = time.monotonic()
    out_dir = work / "out"
    write_settings(work, output_dir=str(out_dir), extra_model_dirs=[str(model_dir)], open_folder_after=False, **settings)
    result = work / "result.json"
    result.unlink(missing_ok=True)
    args = ["--autorun", str(audio), "--result", str(result), "--timeout", str(int(timeout - 20)), "--language", "en"]
    if pause_at:
        args += ["--pause-at", str(pause_at)]
    code, timed_out = run_exe(exe, args, clean_env(work, env_extra), work, timeout)
    step.seconds = time.monotonic() - started
    if timed_out and not result.exists():
        step.detail = f"no result within {timeout:.0f} s (process killed)"
        return step
    if not result.exists():
        step.detail = f"the app exited (code {code}) without a result"
        return step
    data = json.loads(result.read_text(encoding="utf-8"))
    if data.get("state") != "completed":
        step.detail = f"{data.get('state')}: {data.get('error', '')}"[:400]
        return step
    ok, detail = check_outputs(Path(data["output_dir"]), expect_words)
    device = data.get("device")
    timing = data.get("timing") or {}
    rtf = timing.get("real_time_factor_effective") or timing.get("real_time_factor")
    if expect_device and device != expect_device:
        ok, detail = False, f"ran on '{device}', expected '{expect_device}'. " + detail
    if pause_at:
        if not (data.get("pause_requested") and data.get("resumed")):
            ok, detail = False, "the job finished before PAUSE SAFELY could be pressed (pause/resume not exercised). " + detail
        else:
            detail = f"PAUSE SAFELY at {data.get('paused_at_percent')}%, RESUME, completed; " + detail
    step.ok = ok
    step.detail = f"device={device}/{data.get('compute_type')}, audio {data.get('audio_seconds')} s, wall {step.seconds:.0f} s, RTF {rtf}; {detail}"
    return step


def long_clip(audio: dict[str, Path], target: Path) -> Path:
    """The three clean benchmark clips back to back (about 2.5 minutes): long enough to press PAUSE SAFELY in the middle."""
    import wave

    with wave.open(str(target), "wb") as out:
        for i, key in enumerate(("gpu", "cpu", "pyav")):
            with wave.open(str(audio[key]), "rb") as src:
                if i == 0:
                    out.setparams(src.getparams())
                out.writeframes(src.readframes(src.getnframes()))
    return target


def find_model_dir() -> Path:
    from src.services.paths import get_paths

    return get_paths().models


# ----------------------------------------------------------------------------------------------- v3: batch queue, documents, OCR
def make_text_pdf(path: Path, pages: list[str]) -> Path:
    """A small valid PDF with real embedded text (no library needed)."""
    objects: dict[int, bytes] = {1: b"<< /Type /Catalog /Pages 2 0 R >>", 3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"}
    kids = []
    for i, text in enumerate(pages):
        number = 4 + 2 * i
        kids.append(f"{number} 0 R")
        stream = f"BT /F1 14 Tf 40 700 Td ({text}) Tj ET".encode("latin-1")
        objects[number] = (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {number + 1} 0 R "
                           f"/Resources << /Font << /F1 3 0 R >> >> >>").encode()
        objects[number + 1] = b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"
    objects[2] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>".encode()
    out, offsets = bytearray(b"%PDF-1.4\n"), {}
    for number in sorted(objects):
        offsets[number] = len(out)
        out += b"%d 0 obj\n" % number + objects[number] + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for number in sorted(objects):
        out += b"%010d 00000 n \n" % offsets[number]
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    path.write_bytes(bytes(out))
    return path


def make_scan(path: Path, text: str) -> None:
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (1400, 240), "white")
    ImageDraw.Draw(image).text((30, 90), text, fill="black", font=ImageFont.load_default(size=52))
    if path.suffix == ".pdf":
        image.save(path, "PDF")
    else:
        image.save(path)


def step_queue(name: str, exe: Path, work: Path, files: list[Path], model_dir: Path, timeout: float, env_extra: dict[str, str] | None = None
               ) -> tuple[Step, dict]:
    """The real window, several files at once: CONVERT TO TEXT -> queue -> result file. Returns the step and the parsed result."""
    step, started = Step(name), time.monotonic()
    out_dir = work / "out"
    write_settings(work, output_dir=str(out_dir), extra_model_dirs=[str(model_dir)], open_folder_after=False, device="auto")
    result = work / "result.json"
    result.unlink(missing_ok=True)
    args = ["--autorun-queue", "|".join(str(f) for f in files), "--result", str(result), "--timeout", str(int(timeout - 20)), "--language", "en"]
    code, timed_out = run_exe(exe, args, clean_env(work, env_extra), work, timeout)
    step.seconds = time.monotonic() - started
    if not result.exists():
        step.detail = f"no result (exit {code}, timed out: {timed_out})"
        return step, {}
    data = json.loads(result.read_text(encoding="utf-8"))
    if data.get("state") != "completed":
        step.detail = f"{data.get('state')}: {data.get('error') or [(i['name'], i['state'], i['error']) for i in data.get('items', [])]}"[:500]
    else:
        step.ok = True
    return step, data


def newest_output(root: Path) -> Path | None:
    """The newest transcript folder (the one holding local_transcript.json) below an output root."""
    found = sorted(root.rglob("local_transcript.json"), key=lambda f: f.stat().st_mtime)
    return found[-1].parent if found else None


def step_intelligence_on(exe: Path, work: Path, audio: Path, model_dir: Path) -> Step:
    """Medical Intelligence v4 ON in the installed program (the opt-in): the layer runs, the raw transcript stays untouched, both reports carry its status."""
    step = step_transcribe("Medical Intelligence v4 ON (opt-in) through the window", exe, work, audio, model_dir, {"device": "auto"}, {"MLT_V4_MODE": "on"},
                           ["glycolysis", "hexokinase", "pyruvate"], None, 420)
    if not step.ok:
        return step
    folder = newest_output(work / "out")
    if folder is None:
        step.ok, step.detail = False, "no output folder"
        return step
    sys.path.insert(0, str(ROOT / "scripts"))
    from ci_validate_packaged import intelligence_run_check

    ok, note = intelligence_run_check(folder)
    step.ok, step.detail = ok, note + "; " + step.detail
    return step


def step_study(exe: Path, work: Path, lecture: Path) -> Step:
    """Study generation in the installed program from a finished lecture folder: the new quality engine made it, items are tagged with facts, nothing is malformed."""
    step, started = Step("Study workspace from a finished lecture: new quality engine, no malformed item"), time.monotonic()
    sys.path.insert(0, str(ROOT / "tools"))
    import study_integration_check as C

    report, folder = C.run_packaged(exe, lecture, work)
    step.seconds = time.monotonic() - started
    pack_file = folder / "study" / "study_pack.json"
    if not pack_file.is_file():
        step.detail = f"no study pack was written ({report.get('error') or (report.get('ui_study') or {}).get('error')})"
        return step
    pack = json.loads(pack_file.read_text(encoding="utf-8"))
    engine = (report.get("ui_study") or {}).get("engine") or {}
    scan, counts = C.defect_scan(pack), C.counts(pack)
    problems = []
    if "NEW QUALITY ENGINE" not in str(engine.get("engine")):
        problems.append(f"engine {engine.get('engine')!r}")
    if not (counts["flashcards"] and counts["questions"] and counts["summary"]):
        problems.append(f"empty areas {counts}")
    if scan["total"] or C.mcq_kind_problems(pack):
        problems.append(f"malformed items: {scan['total']} found by the scan, {C.mcq_kind_problems(pack)} broken multiple-choice questions")
    step.ok = not problems
    step.detail = "; ".join(problems) if problems else f"{engine.get('engine')}; delivered {counts}; known defects 0; facts {engine.get('facts', {}).get('validated')} of {engine.get('facts', {}).get('sentences')} sentences"
    return step


MEDIA_FILES = ("speech.wav", "speech.mp3", "speech.m4a", "speech.aac", "video_h264.mp4", "video_h264.mov", "video_hevc.mp4")


def make_media_matrix(speech: Path, target: Path, seconds: float = 20.0) -> list[Path]:
    """One spoken passage in every container / codec the program must read (scripts/make_media_matrix.py; the encoders belong to the development environment, not to the program)."""
    done = subprocess.run([sys.executable, str(ROOT / "scripts" / "make_media_matrix.py"), str(target), "--speech", str(speech), "--seconds", str(seconds)],
                          capture_output=True, text=True, timeout=300)
    made = [target / n for n in MEDIA_FILES if (target / n).is_file()]
    if done.returncode != 0 or len(made) != len(MEDIA_FILES):
        raise SystemExit(f"the media fixtures could not be made ({len(made)} of {len(MEDIA_FILES)}): {done.stderr[-300:]}")
    return made


def check_media(items: list[dict], terms: list[str]) -> tuple[bool, str]:
    by_name = {i["name"]: i for i in items}
    problems, notes = [], []
    for name in MEDIA_FILES:
        item = by_name.get(name)
        if not item or item["state"] != "completed":
            problems.append(f"{name}: {item['state'] if item else 'missing'} {item.get('error', '')[:80] if item else ''}")
            continue
        raw = Path(item["files"]["raw"]).read_text(encoding="utf-8").lower() if item["files"].get("raw") else ""
        words = len(raw.split())
        if words < 15 or not any(t.lower() in raw for t in terms):
            problems.append(f"{name}: {words} words, none of {terms} heard")
        else:
            notes.append(f"{name.rsplit('.', 1)[0].replace('speech', 'audio')}.{name.rsplit('.', 1)[1]}={words}w")
    return (not problems), ("; ".join(problems) if problems else ", ".join(notes))


def check_documents(items: list[dict]) -> tuple[bool, str]:
    by_name = {i["name"]: i for i in items}
    problems, notes = [], []
    for name, source, engine in (("embedded.pdf", "embedded_text", ""), ("scan.pdf", "ocr", "Windows text recognition"),
                                 ("slide.png", "ocr", "Windows text recognition"), ("notes.txt", "text_file", "")):
        item = by_name.get(name)
        if not item or item["state"] != "completed":
            problems.append(f"{name}: not completed")
            continue
        data = json.loads(Path(item["files"]["json"]).read_text(encoding="utf-8"))
        text = Path(item["files"]["text"]).read_text(encoding="utf-8").lower()
        if data["text_source"] != source:
            problems.append(f"{name}: text source {data['text_source']!r}, expected {source!r}")
        if source in ("embedded_text", "text_file") and data["confidence"] is not None:
            problems.append(f"{name}: an invented confidence value ({data['confidence']})")
        if engine and engine not in data["label"]:
            problems.append(f"{name}: engine label {data['label']!r}")
        if not any(w in text for w in ("afferent", "arteriole", "nephron", "kidney")):
            problems.append(f"{name}: expected words not found in {text[:80]!r}")
        notes.append(f"{name}={data['text_source']}")
    return (not problems), ("; ".join(problems) if problems else ", ".join(notes))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exe", type=Path, default=DEFAULT_EXE)
    parser.add_argument("--quick", action="store_true", help="start-up checks only (no model, no transcription)")
    parser.add_argument("--model-dir", type=Path, default=None, help="folder that contains the Large-v3 model folder")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--skip-cpu", action="store_true")
    parser.add_argument("--skip-download", action="store_true", help="leave out the step that downloads a small test model from the internet (the offline check uses this)")
    args = parser.parse_args(argv)
    exe = args.exe.resolve()
    if not exe.is_file():
        print(f"FAILED: {exe} does not exist - run BUILD_APP.bat first")
        return 1
    base = Path(tempfile.mkdtemp(prefix="mlt_smoke_"))                     # outside the project
    print(f"[smoke] exe: {exe}\n[smoke] working folder: {base}")
    steps: list[Step] = []

    def record(step: Step) -> None:
        steps.append(step)
        print(f"[smoke] {'PASS' if step.ok else 'FAIL'}  {step.name}  ({step.seconds:.0f} s)\n         {step.detail}")

    record(step_selftest(exe, base / "selftest"))
    record(step_diagnostics(exe, base / "diag"))
    record(step_gui_launch(exe, base / "gui"))
    docs = base / "docs_in"
    docs.mkdir()
    make_text_pdf(docs / "embedded.pdf", ["The nephron is the functional unit of the kidney and filters blood plasma in the glomerulus."])
    make_scan(docs / "scan.pdf", "The afferent arteriole regulates glomerular filtration")
    make_scan(docs / "slide.png", "The afferent arteriole regulates glomerular filtration")
    (docs / "notes.txt").write_text("Kidney notes: the nephron filters plasma.", encoding="utf-8")
    step, queue_result = step_queue("Documents through the queue: embedded PDF, scanned PDF (OCR), picture (OCR), text file", exe, base / "docs",
                                    [docs / n for n in ("embedded.pdf", "scan.pdf", "slide.png", "notes.txt")], base, 180)
    if step.ok:
        step.ok, step.detail = check_documents(queue_result["items"])
    record(step)
    if not args.quick:
        if not args.skip_download:
            record(step_download(exe, base / "download"))
        model_dir = args.model_dir or find_model_dir()
        audio = ensure_audio()
        terms_bio = ["glycolysis", "hexokinase", "pyruvate", "ATP"]
        mp3 = to_mp3(audio["pyav"], base / "pharma_hard.mp3")
        three = long_clip(audio, base / "three_lectures.wav")
        record(step_transcribe("REAL transcription, GPU (CUDA from libs/cuda), Large-v3", exe, base / "gpu", audio["gpu"], model_dir,
                               {"device": "auto"}, {}, terms_bio, "cuda", 420))
        if not args.skip_cpu:
            record(step_transcribe("REAL transcription, CPU fallback (no CUDA visible)", exe, base / "cpu", audio["cpu"], model_dir,
                                   {"device": "auto"}, {"CUDA_VISIBLE_DEVICES": "-1"}, ["corticospinal", "tract", "substantia"], "cpu", 900))
        batch_dir = base / "batch_in"
        batch_dir.mkdir()
        first, second = batch_dir / "Lecture_01.wav", batch_dir / "Lecture_02.mp3"
        shutil.copyfile(audio["gpu"], first)
        shutil.copyfile(mp3, second)
        step, batch = step_queue("REAL batch of 2 recordings (WAV + MP3) and a PDF through the queue, GPU", exe, base / "batch",
                                 [first, second, docs / "embedded.pdf"], model_dir, 900)
        if step.ok:
            names = {Path(f).name for i in batch["items"] for f in i["files"].values()}
            wanted = {"Lecture_01_raw.txt", "Lecture_01_clean.txt", "Lecture_01_timestamped.txt", "Lecture_01.srt", "Lecture_01.vtt",
                      "Lecture_01.json", "Lecture_02_raw.txt", "Lecture_02.srt", "Lecture_02.json"}
            missing = sorted(wanted - names)
            heard = Path(batch["items"][0]["files"]["raw"]).read_text(encoding="utf-8").lower() if batch["items"][0]["files"].get("raw") else ""
            if missing or "glycolysis" not in heard or not Path(batch["root"]).name.startswith("Batch_"):
                step.ok, step.detail = False, f"missing named files {missing}; first recording heard glycolysis: {'glycolysis' in heard}"
            else:
                step.detail = f"3 items completed in {Path(batch['root']).name}; named outputs present; terms heard"
        record(step)
        record(step_transcribe("REAL PAUSE SAFELY + RESUME through the window, then completion (3 clips, 2.7 min)", exe, base / "pause", three,
                               model_dir, {"device": "auto", "chunk_minutes": 15.0}, {}, ["glycolysis", "corticospinal", "propranolol"], "cuda",
                               480, pause_at=25.0))
        record(step_intelligence_on(exe, base / "v4on", audio["gpu"], model_dir))
        lecture = newest_output(base / "pause" / "out")
        record(step_study(exe, base / "study", lecture) if lecture is not None else Step("Study workspace from a finished lecture: no lecture folder from the pause step"))
        restart = step_gui_launch(exe, base / "gui", first_launch=False)               # the same profile folder as the first launch: settings, models, logs already there
        restart.name = "RESTART: the program starts again on the same profile and closes cleanly"
        record(restart)
        media = make_media_matrix(audio["gpu"], base / "media_in")
        for label, extra, folder in (("with the bundled FFmpeg", {}, "media_ffmpeg"), ("with PyAV only (FFmpeg disabled)", {"MLT_DISABLE_FFMPEG": "1"}, "media_pyav")):
            step, matrix = step_queue(f"MEDIA FORMATS {label}: WAV, MP3, M4A, AAC, MP4 H.264, MOV H.264, MP4 HEVC through the queue", exe, base / folder, media, model_dir, 900, extra)
            if step.ok:
                step.ok, step.detail = check_media(matrix["items"], terms_bio)
            record(step)
        record(step_transcribe("REAL transcription of an MP3 with PyAV only (FFmpeg disabled)", exe, base / "pyav", mp3, model_dir,
                               {"device": "auto"}, {"MLT_DISABLE_FFMPEG": "1"}, ["propranolol", "beta", "metoprolol"], None, 420))
    ok = all(s.ok for s in steps)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"exe": str(exe), "ok": ok, "steps": [s.as_dict() for s in steps]}, indent=2), encoding="utf-8")
    print(f"\n[smoke] {'ALL PASSED' if ok else 'FAILED'} - report: {args.report}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
