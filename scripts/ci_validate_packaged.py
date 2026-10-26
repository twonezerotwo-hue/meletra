"""Test the BUILT, INSTALLED-LIKE app on a CI macOS machine. Writes ci_out/packaged_validation.json.

    python scripts/ci_validate_packaged.py --app "<copy of MELETRA.app>" [--dmg <the DMG>] [--out ci_out/packaged_validation.json]

Run from a directory unrelated to the repository; the app is a COPY taken out of the mounted DMG after it was ejected. Every run of the app
happens under an operating-system rule that denies all network access (sandbox-exec), so a hidden download cannot succeed - it would make the
test fail. What cannot be looked at on a CI machine is read from the window itself (MLT_UI_TRACE=1: the progress texts with their times,
navigation, the first-launch model screen, the Study workspace) - see src/gui/diagnostics_probe.py.

This proves the build works on the hosted macOS architecture recorded in the report. It says nothing about a physical user Mac.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_tools import macos_intel as intel                           # noqa: E402
from validate_macos_intel import (                                     # noqa: E402
    EXPECTED_TERMS, MEDICAL_SENTENCE, Session, check_documents_mac, make_speech, make_video, no_dev_files, pick_voice, sha256, video_progress_ok,
)

PROFILE = '(version 1)(allow default)(deny network*)(allow network* (remote unix-socket))(allow network* (remote ip "localhost:*"))'
TITLES = [("Preparing video", "Preparing video"), ("Extracting audio", "Extracting audio"), ("Analyzing", "Analyzing"), ("Transcribing", "Transcribing"),
          ("Finalizing", "Finalizing")]


def say(message: str) -> None:
    print(f"[ci-validate] {message}", flush=True)


# ------------------------------------------------------------------------------------------------------------ pure helpers
def stage_check(result: dict, expect_video: bool = True) -> tuple[bool, str]:
    """From the window's own trace: the first title is "Preparing ...", then Extracting audio, Analyzing, Transcribing, Finalizing in that order,
    and the percent label moved off 0% while the job ran (no frozen 0%)."""
    rows = result.get("ui_trace") or []
    titles = [(r[0], r[2]) for r in rows if r[1] == "active.title"]
    if not titles:
        return False, "the window never showed a progress title"
    seen = []
    for needle, label in TITLES:
        if needle == "Preparing video" and not expect_video:
            needle, label = "Preparing", "Preparing"
        index = next((i for i, (_t, text) in enumerate(titles) if needle in text), None)
        if index is None:
            return False, f"never showed '{label}' (saw: {', '.join(t for _s, t in titles)})"
        seen.append(index)
    if seen != sorted(seen):
        return False, f"stages out of order: {[t for _s, t in titles]}"
    percents = [r[2] for r in rows if r[1] == "active.percent"]
    if not any(p not in ("0%", "") for p in percents):
        return False, f"the percent stayed at 0% ({percents[:6]})"
    first_gap = titles[0][0]
    return True, f"{' > '.join(t for _s, t in titles)}; first title at +{first_gap:.1f} s; percents {percents[:8]}"


def queue_check(result: dict) -> tuple[bool, str]:
    """1 of 2 -> 2 of 2 -> Done on the batch screen; the first file's result screen never covers the second file; the second file's progress shows."""
    rows = result.get("ui_trace") or []
    titles = [(r[0], r[2]) for r in rows if r[1] == "batch.title"]
    names = [t for _s, t in titles]
    def at(text: str) -> Optional[float]:
        return next((s for s, t in titles if t == text), None)
    first, second, done = at("Converting 1 of 2"), at("Converting 2 of 2"), at("Done")
    if None in (first, second, done) or not (first < second < done):
        return False, f"batch titles: {names}"
    early_result = [r for r in rows if r[1] == "result.title" and r[0] < done]
    if early_result:
        return False, f"a result screen ({early_result[0][2]}) showed at +{early_result[0][0]:.1f} s, before the queue was done"
    second_percent = [r for r in rows if r[1] == "batch.percent" and second <= r[0] < done]
    if not second_percent:
        return False, "the second file's progress was never visible"
    items = [i["state"] for i in result.get("items", [])]
    return items == ["completed", "completed"], f"{' > '.join(names)}; items {items}; second-file progress rows {len(second_percent)}"


def medical_outputs_check(folder: Path) -> tuple[bool, str]:
    needed = ("local_raw_transcript.txt", "local_clean_transcript.txt", "medical_corrected_transcript.txt", "correction_report.json", "medical_quality_report.json")
    missing = [n for n in needed if not (folder / n).is_file()]
    if missing:
        return False, "missing " + ", ".join(missing)
    report = json.loads((folder / "correction_report.json").read_text(encoding="utf-8"))
    quality = json.loads((folder / "medical_quality_report.json").read_text(encoding="utf-8"))
    raw = (folder / "local_raw_transcript.txt").read_text(encoding="utf-8")
    corrected = (folder / "medical_corrected_transcript.txt").read_text(encoding="utf-8")
    if report.get("raw_transcript_modified"):
        return False, "the report says the raw transcript was modified"
    if not quality.get("medical_terms_detected") or not quality.get("detected_subject"):
        return False, f"no vocabulary effect (terms detected {quality.get('medical_terms_detected')}, subject {quality.get('detected_subject')})"
    found = [t for t in EXPECTED_TERMS if t in corrected.lower()]
    changed = [t for t in EXPECTED_TERMS if t in corrected.lower() and t not in raw.lower()]
    return True, f"subject {quality.get('detected_subject')}; {quality['medical_terms_detected']} terms detected; fixture terms in the corrected text: {found}; made by correction: {changed}"


def intelligence_resources_check(selftest_checks: dict) -> tuple[bool, str]:
    """The packaged self-test's "Medical knowledge" check: the bundled database opened, matched its schema and answered its queries (a skipped / missing one is NOT a pass here)."""
    check = selftest_checks.get("Medical knowledge", {})
    detail = check.get("detail", "")
    return bool(check.get("ok")) and "concepts" in detail and "not installed" not in detail.lower() and "skipped" not in detail.lower(), detail[:200] or "the self-test has no 'Medical knowledge' check"


def intelligence_run_check(folder: Path) -> tuple[bool, str]:
    """A transcription run with Medical Intelligence ON, from the packaged binary: the v4 layer ran (status ok), the raw transcript is untouched, the reports carry its result."""
    report = json.loads((folder / "correction_report.json").read_text(encoding="utf-8"))
    quality = json.loads((folder / "medical_quality_report.json").read_text(encoding="utf-8"))
    v4 = report.get("medical_intelligence_v4") or {}
    if v4.get("mode") != "on" or v4.get("status") != "ok":
        return False, f"v4 mode {v4.get('mode')!r}, status {v4.get('status')!r}"
    if report.get("raw_transcript_modified"):
        return False, "the report says the raw transcript was modified"
    if quality.get("v4_status") != "ok":
        return False, f"medical_quality_report v4_status {quality.get('v4_status')!r}"
    return True, f"v4 on, status ok; {len(v4.get('applied', []))} correction(s) applied, {len(v4.get('suggested', []))} suggestion(s); raw transcript untouched"


def study_check(result: dict) -> tuple[bool, str]:
    study = result.get("ui_study") or {}
    if study.get("error") or not study:
        return False, f"no Study report ({result.get('error') or study.get('error')})"
    tabs = study.get("tabs", [])
    content = study.get("tab_content", {})
    cards = set((content.get("Flashcards") or {}).get("buttons", []))
    labels = " | ".join((content.get("Flashcards") or {}).get("labels", []))
    problems = []
    if tabs != ["Overview", "Summary", "Key Concepts", "Flashcards", "Questions", "Study Mode"]:
        problems.append(f"tabs {tabs}")
    if not {"Previous", "Next", "Know", "Unsure", "Don't know"} <= cards:
        problems.append(f"flashcard buttons {sorted(cards)}")
    if not any(re.match(r"^(Transcript|Slide|Page)\b", b) for b in cards):
        problems.append("no source link on the card")
    if not all(w in (content.get("Flashcards") or {}).get("labels", []) for w in ("Topic", "Type", "Level")):
        problems.append("filters missing")
    if not all(w in labels for w in ("Key concepts", "Question preview", "Study progress")):
        problems.append("right information rail missing")
    stats = study.get("stats", {})
    if not {"concepts", "flashcards", "questions", "mastery"} <= set(stats) or not int(stats.get("flashcards", "0") or 0):
        problems.append(f"stat cards {stats}")
    if not {"Back", "Regenerate", "Export"} <= set(study.get("actions", [])):
        problems.append(f"actions {study.get('actions')}")
    engine = study.get("engine") or {}
    if "NEW QUALITY ENGINE" not in str(engine.get("engine", "")):
        problems.append(f"the pack was not made by the new quality engine ({engine.get('engine')!r})")
    elif not (engine.get("flashcards", {}).get("from_validated_facts") and engine.get("questions", {}).get("from_validated_facts")):
        problems.append(f"cards / questions are not tagged with validated facts ({engine.get('flashcards')}, {engine.get('questions')})")
    return not problems, ("; ".join(problems) if problems else f"tabs {len(tabs)}, stats {stats}, card buttons {sorted(cards)}; {engine.get('engine')}")


def frontend_check(result: dict) -> tuple[bool, str]:
    home = result.get("ui_home") or {}
    nav = home.get("navigation", [])
    texts = " ".join((home.get("home") or {}).get("labels", []) + (home.get("home") or {}).get("buttons", []))
    problems = []
    if nav != ["Home", "Transcription", "Study", "Files", "Settings"]:
        problems.append(f"navigation {nav}")
    if home.get("privacy", "") not in ("Local processing", "Sharing enabled", "Cloud comparison on"):
        problems.append(f"no privacy status chip ({home.get('privacy', '')!r})")
    if "TRANSCRIBE" not in texts or "Browse files" not in texts or "Drop lecture files here" not in texts:
        problems.append("drop area / buttons missing")
    return not problems, ("; ".join(problems) if problems else f"navigation {nav}; privacy '{home.get('privacy')}'")


def first_launch_check(result: dict) -> tuple[bool, str]:
    facts = result.get("ui_first_launch") or {}
    text = facts.get("model_text", "")
    ok = "is ready" in text and facts.get("download_button_hidden") is True and facts.get("import_folder_button_hidden") is True
    return ok, f"'{text[:120]}'; download button hidden: {facts.get('download_button_hidden')}; folder picker button hidden: {facts.get('import_folder_button_hidden')}"


# ------------------------------------------------------------------------------------------------------------ the run
def network_block() -> tuple[list[str], str]:
    """The sandbox rule that denies the network, proven: an outside address works without it and fails with it."""
    wrapper = ["/usr/bin/sandbox-exec", "-p", PROFILE]
    snippet = [sys.executable, "-c", "import socket; socket.create_connection(('1.1.1.1', 443), 5).close()"]
    control = subprocess.run(snippet, capture_output=True, text=True, check=False)
    alive = subprocess.run(wrapper + ["/usr/bin/true"], capture_output=True, text=True, check=False)
    blocked = subprocess.run(wrapper + snippet, capture_output=True, text=True, check=False)
    if alive.returncode != 0:
        return [], f"sandbox-exec could not run ({alive.stderr.strip()[:120]})"
    if blocked.returncode == 0:
        return [], "the sandbox rule did not block the network"
    return wrapper, f"outside address reachable without the rule: {control.returncode == 0}; with it: blocked"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--app", type=Path, required=True)
    parser.add_argument("--dmg", type=Path)
    parser.add_argument("--out", type=Path, default=Path("ci_out/packaged_validation.json"))
    args = parser.parse_args(argv)
    if sys.platform != "darwin":
        print("this runs on macOS only", file=sys.stderr)
        return 2
    app = args.app.resolve()
    checks: dict[str, str] = {}
    notes: dict[str, str] = {}

    def record(key: str, ok: bool, note: str = "") -> None:
        checks[key] = "PASS" if ok else "FAIL"
        notes[key] = note
        say(f"{key}: {checks[key]}  {note[:240]}")

    work = Path(tempfile.mkdtemp(prefix="mlt_ci_"))
    os.chdir(work)                                                       # nothing below may rely on the repository as the working folder
    wrapper, why = network_block()
    record("network_blocked_for_app_runs", bool(wrapper), why)
    session = Session(app, work, wrapper=wrapper, extra_env={"QT_QPA_PLATFORM": "offscreen", "MLT_UI_TRACE": "1", "MLT_SELFTEST_DEEP": "1"})
    report: dict[str, Any] = {"provider": "GitHub Actions", "runner": os.environ.get("RUNNER_NAME", ""), "runner_image": os.environ.get("ImageVersion", ""),
                              "runner_label": os.environ.get("MLT_CI_RUNNER_LABEL", "macos"), "machine": platform.machine(), "macos": platform.mac_ver()[0], "real_hardware": False,
                              "physical_user_mac_tested": False, "app": str(app), "checks": checks, "notes": notes,
                              "dmg_sha256": sha256(args.dmg) if args.dmg and args.dmg.is_file() else ""}
    opened: set[str] = set()
    try:
        record("app_is_a_copy_outside_the_repo", str(ROOT) not in str(app) and "/Volumes/" not in str(app), str(app))
        (work / "tmp").mkdir(exist_ok=True)

        # self-test (deep) and diagnostics
        out = work / "selftest.json"
        done = subprocess.run(session.wrapper + [str(session.exe), "--selftest", str(out)], env=session.env(), cwd=work, check=False, timeout=1500)
        st = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {"ok": False, "checks": []}
        by = {c["name"]: c for c in st.get("checks", [])}
        failed = [f"{c['name']}: {c['detail'][:100]}" for c in st.get("checks", []) if not c["ok"]]
        record("packaged_selftest", done.returncode == 0 and bool(st.get("ok")), "; ".join(failed) or f"{len(by)} checks ok")
        resources = by.get("Resources", {}).get("detail", "")
        record("medical_vocab", by.get("Resources", {}).get("ok", False) and "Medical Expert vocabulary" in resources, resources[:200])
        model_detail = by.get("Model discovery", {}).get("detail", "")
        bundled_title = ("Whisper Large-v3: installed" in model_detail or
                         "Whisper Large-v3 (Apple Silicon): installed" in model_detail)
        record("bundled_large_v3", by.get("Resources", {}).get("ok", False) and "offline release" in resources and
               by.get("Model discovery", {}).get("ok", False) and bundled_title, model_detail[:160])
        record("model_checksum", "speech model, text recognition models and licence files present" in resources, "deep self-test: every file of the bundled model hashed")
        ok, note = intelligence_resources_check(by)
        record("medical_intelligence_resources", ok, note)
        record("ocr_resources", "text recognition models" in resources and by.get("Text recognition", {}).get("ok", False), by.get("Text recognition", {}).get("detail", ""))
        record("medical_expert_selftest", by.get("Medical Expert", {}).get("ok", False), by.get("Medical Expert", {}).get("detail", ""))
        diag = work / "diagnostics.txt"
        subprocess.run(session.wrapper + [str(session.exe), "--diagnostics", str(diag)], env=session.env(), cwd=work, check=False, timeout=300)
        record("diagnostics", diag.is_file() and "MELETRA" in diag.read_text(encoding="utf-8", errors="replace"), f"{diag.stat().st_size if diag.is_file() else 0} bytes")

        # text to speech: the Mac's own voices, nothing downloaded
        voices = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, check=False).stdout.splitlines()
        english = [l.split()[0] for l in voices if re.search(r"\ben[_-]", l)]
        turkish = [l.split()[0] for l in voices if re.search(r"\btr[_-]TR\b", l)]
        record("tts_english", bool(english) and by.get("Read aloud", {}).get("ok", False) and not by.get("Read aloud", {}).get("skipped", False),
               f"{len(english)} English system voices; {by.get('Read aloud', {}).get('detail', '')}")
        report["turkish_builtin_voice"] = "AVAILABLE: " + ", ".join(turkish[:3]) if turkish else "NOT AVAILABLE"

        # real transcription on the runner's own speech (offline voice -> WAV)
        session.write_settings(model_key="whisper-large-v3-mlx" if platform.machine() == "arm64" else "whisper-large-v3")
        speech = make_speech(work, "The glycolysis pathway converts glucose to pyruvate. Hexokinase phosphorylates glucose and ATP is produced.", "audio")
        result = work / "audio.json"
        r = session.run(["--autorun", str(speech), "--result", str(result), "--timeout", "1500", "--language", "en"], 1600)
        data = json.loads(result.read_text(encoding="utf-8")) if result.exists() else {}
        raw = ""
        if data.get("output_dir"):
            f = Path(data["output_dir"]) / "local_raw_transcript.txt"
            raw = f.read_text(encoding="utf-8").lower() if f.is_file() else ""
        seconds = float(data.get("audio_seconds", 0) or 0)
        keywords = ("glucose", "glycolysis", "pyruvate", "hexokinase", "phosphorylates", "atp")
        record("audio_stt", data.get("state") == "completed" and len(raw.split()) >= 8 and any(k in raw for k in keywords),
               f"{data.get('state')} {seconds:.1f} s of audio; error: {data.get('error', '')}; text: {raw[:110]!r}")
        report["probes"] = {"home": data.get("ui_home"), "first_launch": data.get("ui_first_launch"), "audio_ui_trace": (data.get("ui_trace") or [])[:60]}
        backend = "MLX Whisper (Apple Silicon)" if platform.machine() == "arm64" else "faster-whisper / CTranslate2 (CPU)"
        report["performance"] = {"audio_seconds": seconds, "processing_seconds": round(r["seconds"], 1), "real_time_factor": intel.realtime_factor(r["seconds"], seconds),
                                 "peak_ram_gb": round((r["peak_bytes"] or 0) / 1024 ** 3, 2), "backend": backend, "device": data.get("device"),
                                 "compute_type": data.get("compute_type"), "includes": "start-up, model load, transcription, medical pass, writing files"}
        opened |= set(r["opened_files"])
        ok, note = frontend_check(data)
        record("new_frontend", ok, note)
        ok, note = first_launch_check(data)
        record("first_launch_no_downloader", ok, note)
        models_dir = session.data_dir / "models"
        left = sorted(p.name for p in models_dir.iterdir()) if models_dir.is_dir() else []
        record("no_model_download", not left and not (models_dir / ".partial").exists(), "user data models folder: " + (", ".join(left) or "empty - the bundled model is used in place"))

        # video
        mp4 = make_video(speech, work / "short.mp4")
        result_v = work / "video.json"
        rv = session.run(["--autorun", str(mp4), "--result", str(result_v), "--timeout", "1500", "--language", "en"], 1600, measure=False)
        dv = json.loads(result_v.read_text(encoding="utf-8")) if result_v.exists() else {}
        record("video_stt", dv.get("state") == "completed", f"{dv.get('state')} {dv.get('audio_seconds')} s")
        ok, note = stage_check(dv)
        record("video_progress", ok, note)
        opened |= set(rv["opened_files"])

        # the decoders a licence-clean media stack must still have: the files are made HERE with the CI environment's PyAV (it has the encoders), the app only decodes them
        media_notes, media_ok = [], True
        try:
            import make_media_matrix as MM

            matrix = work / "media_matrix"
            MM.main([str(matrix), "--speech", str(speech), "--seconds", "6"])
            for media_file in sorted(matrix.glob("*.*")):
                out_json = work / f"media_{media_file.stem}_{media_file.suffix[1:]}.json"
                rm = session.run(["--autorun", str(media_file), "--result", str(out_json), "--timeout", "900", "--language", "en"], 1000, measure=False)
                dm = json.loads(out_json.read_text(encoding="utf-8")) if out_json.exists() else {}
                words = 0
                if dm.get("output_dir"):
                    raw_file = Path(dm["output_dir"]) / "local_raw_transcript.txt"
                    words = len(raw_file.read_text(encoding="utf-8").split()) if raw_file.is_file() else 0
                good = dm.get("state") == "completed" and words >= 3
                media_ok &= good
                media_notes.append(f"{media_file.name}={'ok' if good else dm.get('state') or 'no result'}({words}w)")
                opened |= set(rm["opened_files"])
        except Exception as exc:                                                   # noqa: BLE001 - reported as a failed check, not a crash of the whole validation
            media_ok = False
            media_notes.append(f"{type(exc).__name__}: {exc}")
        record("media_formats", media_ok and len(media_notes) >= 6, "WAV, MP3, M4A, AAC, MP4 H.264, MOV H.264, MP4 HEVC through the packaged app: " + ", ".join(media_notes))

        # queue of two
        second = make_speech(work, "The citric acid cycle takes place in the mitochondria.", "second")
        result_q = work / "queue.json"
        rq = session.run(["--autorun-queue", f"{speech}|{second}", "--result", str(result_q), "--timeout", "1500", "--language", "en"], 1600, measure=False)
        dq = json.loads(result_q.read_text(encoding="utf-8")) if result_q.exists() else {}
        ok, note = queue_check(dq)
        record("queue", ok, note)
        opened |= set(rq["opened_files"])

        # Medical Expert on speech
        med = make_speech(work, MEDICAL_SENTENCE, "medical")
        result_m = work / "medical.json"
        session.run(["--autorun", str(med), "--result", str(result_m), "--timeout", "1500", "--language", "en"], 1600, measure=False)
        dm = json.loads(result_m.read_text(encoding="utf-8")) if result_m.exists() else {}
        ok, note = medical_outputs_check(Path(dm["output_dir"])) if dm.get("output_dir") else (False, "no output folder")
        record("medical_expert", ok and checks.get("medical_expert_selftest") == "PASS", note)

        # Medical Intelligence (v4) ON in the packaged app: the same recording again
        session.extra_env["MLT_V4_MODE"] = "on"
        result_i = work / "intelligence.json"
        session.run(["--autorun", str(med), "--result", str(result_i), "--timeout", "1500", "--language", "en"], 1600, measure=False)
        session.extra_env.pop("MLT_V4_MODE", None)
        di = json.loads(result_i.read_text(encoding="utf-8")) if result_i.exists() else {}
        ok, note = intelligence_run_check(Path(di["output_dir"])) if di.get("output_dir") else (False, "no output folder")
        record("medical_intelligence_on", ok, note)

        # documents
        from scripts.smoke_packaged import make_scan, make_text_pdf
        docs = work / "docs"
        docs.mkdir()
        make_text_pdf(docs / "embedded.pdf", ["The nephron is the functional unit of the kidney and filters blood plasma in the glomerulus."])
        make_scan(docs / "scan.pdf", "The afferent arteriole regulates glomerular filtration")
        make_scan(docs / "slide.png", "The afferent arteriole regulates glomerular filtration")
        (docs / "notes.txt").write_text("Kidney notes: the nephron filters plasma.", encoding="utf-8")
        result_d = work / "docs.json"
        session.run(["--autorun-queue", "|".join(str(docs / n) for n in ("embedded.pdf", "scan.pdf", "slide.png", "notes.txt")), "--result", str(result_d),
                     "--timeout", "600", "--language", "en"], 700, measure=False)
        dd = json.loads(result_d.read_text(encoding="utf-8")) if result_d.exists() else {}
        ok, note = check_documents_mac(dd.get("items", []))
        items = {i["name"]: i for i in dd.get("items", [])}
        record("pdf", items.get("embedded.pdf", {}).get("state") == "completed" and "embedded.pdf:" not in note, note)
        record("ocr", ok, note)

        # Study workspace of the packaged build (a finished synthetic lecture, no network, built-in method)
        from tests.study_support import write_lecture_folder
        lecture = write_lecture_folder(work / "lecture")
        result_s = work / "study.json"
        session.run(["--autorun-study", str(lecture), "--result", str(result_s), "--timeout", "300"], 400, measure=False)
        ds = json.loads(result_s.read_text(encoding="utf-8")) if result_s.exists() else {}
        ok, note = study_check(ds)
        record("study_workspace", ok, note)
        report["probes"]["study"] = ds.get("ui_study")
        report["probes"]["video_ui_trace"] = (dv.get("ui_trace") or [])[:80]
        report["probes"]["queue_ui_trace"] = (dq.get("ui_trace") or [])[:80]

        # no development environment, restart
        ok, note = no_dev_files(opened)
        record("no_dev_environment", ok, note)
        out2 = work / "selftest2.json"
        done2 = subprocess.run(session.wrapper + [str(session.exe), "--selftest", str(out2)], env=session.env(), cwd=work, check=False, timeout=1500)
        record("app_restart", done2.returncode == 0 and out2.exists(), "second start with the same user data folder")

        # the real cocoa window (informational on a CI machine: a hosted VM may not give a window server to a launched app)
        env = {k: v for k, v in session.env().items() if k != "QT_QPA_PLATFORM"}
        proc = subprocess.Popen(session.wrapper + [str(session.exe)], env=env, cwd=work, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        time.sleep(20)
        alive = proc.poll() is None
        if alive:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
        stderr = (proc.communicate(timeout=30)[1] or "")[-300:]
        record("gui_launch_cocoa", alive, f"stayed up for 20 s: {alive}; exit {proc.returncode}; {stderr.strip()[:120]}")
        # what a downloaded copy would meet: the quarantine flag + the Gatekeeper assessment (informational: ad-hoc signed, not notarized)
        subprocess.run(["xattr", "-w", "com.apple.quarantine", f"0083;{int(time.time()):x};Safari;", str(app)], check=False)
        gk = subprocess.run(["spctl", "--assess", "--type", "execute", "-vv", str(app)], capture_output=True, text=True, check=False)
        report["gatekeeper_quarantined_copy"] = {"accepted": gk.returncode == 0, "output": ((gk.stdout or "") + (gk.stderr or "")).strip()[:300]}
        say(f"gatekeeper (informational): accepted={gk.returncode == 0} {report['gatekeeper_quarantined_copy']['output'][:160]}")
    finally:
        logs = list(session.home.rglob("app.log"))
        if logs:
            report["app_log_tail"] = logs[0].read_text(encoding="utf-8", errors="replace")[-12000:]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        shutil.rmtree(work, ignore_errors=True)
    informational = {"gui_launch_cocoa"}
    bad = [k for k, v in checks.items() if v != "PASS" and k not in informational]
    say("ALL CHECKS PASS" if not bad else "FAILING: " + ", ".join(bad))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
