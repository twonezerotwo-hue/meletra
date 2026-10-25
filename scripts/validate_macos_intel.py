"""Validate the INSTALLED Intel-Mac offline app on a REAL Intel Mac, with the network switched off. Writes validation_report.json.

Run it in Terminal ON THE MAC'S OWN DESKTOP (the app needs a window server), after
    1. the DMG was downloaded / copied to this Mac,
    2. Wi-Fi and Ethernet were switched OFF,
    3. you dragged "MELETRA.app" from the mounted DMG into /Applications and ejected the DMG:

    .venv-macos-intel/bin/python scripts/validate_macos_intel.py --dmg ~/Downloads/MELETRA-Intel-Mac-Offline.dmg \\
        [--app "/Applications/MELETRA.app"] [--attest frontend,study_workspace,tts]

What it checks by itself: the network really is off; the Gatekeeper verdict of the quarantined DMG; the deep self-test; real transcription of
speech made by the Mac's own voice (audio, a video made with PyAV, a 2-file queue); the progress events of the video; the Medical Expert
outputs and that the raw transcript is untouched; PDF and scanned-page OCR; that nothing was downloaded; that the app opens no file of this
repository / a virtualenv / Homebrew; a second launch. Checks that need eyes (the look of Home and the Study workspace, hearing the voice) are
only "PASS" when you attest them with --attest - the report records that they were attested, not measured.

The finalizer on Windows (scripts/finalize_mac_dmg.py) refuses to call the DMG ready unless this report says: real_hardware, machine x86_64,
network_disconnected, the SHA-256 of THIS dmg, and every check PASS.
Nothing in this file was run on a Mac when it was written; its pure parts are unit-tested.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from build_tools import macos_intel as intel                          # noqa: E402
from src import PRODUCT_ID                                                # noqa: E402

DEFAULT_APP = Path("/Applications") / intel.APP_FOLDER
MEDICAL_SENTENCE = ("In neuroanatomy, the Nissl bodies are found in the neuronal soma. Schwann cells form the myelin sheath of the peripheral "
                    "nervous system. Tanycytes line the third ventricle. The node of Ranvier allows saltatory conduction along a myelinated axon. "
                    "Remak bundles contain several unmyelinated axons. An axodendritic synapse connects an axon to a dendrite, and an axosomatic "
                    "synapse ends on the cell body.")
EXPECTED_TERMS = ("nissl", "schwann", "tanycyte", "remak", "saltatory", "axodendritic", "axosomatic")


def say(message: str) -> None:
    print(f"[validate] {message}", flush=True)


# ------------------------------------------------------------------------------------------------------------ pure helpers
def network_is_off(timeout: float = 3.0) -> tuple[bool, str]:
    """True only when a public address cannot be reached AND a public name cannot be resolved (so a Wi-Fi that is 'on but captive' fails)."""
    notes = []
    try:
        socket.create_connection(("1.1.1.1", 443), timeout).close()
        notes.append("connected to 1.1.1.1:443")
    except OSError:
        pass
    try:
        socket.getaddrinfo("huggingface.co", 443)
        notes.append("resolved huggingface.co")
    except OSError:
        pass
    return (not notes), ("; ".join(notes) or "no route to the internet")


def machine_facts() -> dict[str, Any]:
    def sysctl(name: str) -> str:
        return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, check=False).stdout.strip() if sys.platform == "darwin" else ""

    features = (sysctl("machdep.cpu.features") + " " + sysctl("machdep.cpu.leaf7_features")).split()
    return {"machine": platform.machine(), "model": sysctl("hw.model"), "cpu": sysctl("machdep.cpu.brand_string"),
            "ram_gb": round(int(sysctl("hw.memsize") or 0) / 1024 ** 3, 1), "macos": platform.mac_ver()[0],
            "cpu_features": [f for f in features if f in ("SSE4.2", "AVX1.0", "AVX2", "AVX512F", "FMA")]}


def real_hardware() -> bool:
    """A Mac that is not a virtual machine (hw.model is 'VirtualMac*' / 'VMware*' / 'Parallels*' in VMs; kern.hv_vmm_present is 1)."""
    if sys.platform != "darwin":
        return False
    vm = subprocess.run(["sysctl", "-n", "kern.hv_vmm_present"], capture_output=True, text=True, check=False).stdout.strip()
    model = subprocess.run(["sysctl", "-n", "hw.model"], capture_output=True, text=True, check=False).stdout.strip()
    return vm != "1" and not re.search(r"vmware|parallels|virtual|vbox", model, re.I)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def events_of(data_dir: Path) -> list[dict]:
    rows = []
    for f in sorted((data_dir / "jobs").glob("*.events.jsonl")) if (data_dir / "jobs").is_dir() else []:
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    return rows


def video_progress_ok(events: list[dict]) -> tuple[bool, str]:
    """The job reports every step before the first words: its first step arrives within 3 s of the job start (the window itself shows
    "Preparing video ..." at once - that part is looked at), audio preparing -> audio ready -> (language / model loading) -> first progress."""
    def first(kind: str) -> Optional[float]:
        return next((float(e["time"]) for e in events if e.get("type") == kind and "time" in e), None)

    start, preparing, ready, progress = first("job_started"), first("audio_preparing"), first("audio_ready"), first("chunk_progress")
    if None in (start, preparing, ready):
        return False, f"missing events (job_started={start}, audio_preparing={preparing}, audio_ready={ready})"
    early = min(t for t in (first("source_validated"), preparing) if t is not None)
    if early - start > 3.0:
        return False, f"the first step came {early - start:.1f} s after the start"
    if progress is not None and not (preparing <= ready <= progress):
        return False, "the steps came in the wrong order"
    steps = [k for k in ("source_validated", "audio_preparing", "audio_ready", "language_detecting", "model_loading", "chunk_progress") if first(k) is not None]
    return True, "steps seen: " + " > ".join(steps)


def medical_check(folder: Path) -> tuple[bool, str]:
    needed = ("local_raw_transcript.txt", "local_clean_transcript.txt", "medical_corrected_transcript.txt", "correction_report.json", "medical_quality_report.json")
    missing = [n for n in needed if not (folder / n).is_file()]
    if missing:
        return False, "missing " + ", ".join(missing)
    report = json.loads((folder / "correction_report.json").read_text(encoding="utf-8"))
    quality = json.loads((folder / "medical_quality_report.json").read_text(encoding="utf-8"))
    if report.get("raw_transcript_modified"):
        return False, "the report says the raw transcript was modified"
    if not quality.get("medical_terms_detected"):
        return False, "no medical term was detected - the vocabulary is probably not in the app"
    text = (folder / "medical_corrected_transcript.txt").read_text(encoding="utf-8").lower()
    found = [t for t in EXPECTED_TERMS if t in text]
    return len(found) >= 3, f"{quality['medical_terms_detected']} medical terms detected; expected terms found: {found}"


def check_documents_mac(items: list[dict]) -> tuple[bool, str]:
    by_name = {i["name"]: i for i in items}
    problems, notes = [], []
    for name, source, engine in (("embedded.pdf", "embedded_text", ""), ("scan.pdf", "ocr", "RapidOCR"), ("slide.png", "ocr", "RapidOCR"),
                                 ("notes.txt", "text_file", "")):
        item = by_name.get(name)
        if not item or item["state"] != "completed":
            problems.append(f"{name}: not completed")
            continue
        data = json.loads(Path(item["files"]["json"]).read_text(encoding="utf-8"))
        text = Path(item["files"]["text"]).read_text(encoding="utf-8").lower()
        if data["text_source"] != source:
            problems.append(f"{name}: text source {data['text_source']!r}, expected {source!r}")
        if engine and engine not in data["label"]:
            problems.append(f"{name}: engine label {data['label']!r}")
        if not any(w in text for w in ("afferent", "arteriole", "nephron", "kidney")):
            problems.append(f"{name}: expected words not found")
        notes.append(f"{name}={data['text_source']}")
    return (not problems), ("; ".join(problems) if problems else ", ".join(notes))


# ------------------------------------------------------------------------------------------------------------ running the app
class Session:
    def __init__(self, app: Path, work: Path, wrapper: Optional[list[str]] = None, extra_env: Optional[dict[str, str]] = None) -> None:
        self.app, self.work = app, work
        self.wrapper = list(wrapper or [])                 # e.g. sandbox-exec -p '(deny network*)': the app runs under it
        self.extra_env = dict(extra_env or {})
        self.home = work / "home"
        self.home.mkdir(parents=True, exist_ok=True)
        self.exe = app / "Contents" / "MacOS" / PRODUCT_ID
        self.data_dir = self.home / "Library" / "Application Support" / "MedicalLectureTranscriber"

    def env(self) -> dict[str, str]:
        return {"HOME": str(self.home), "PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "LANG": "en_US.UTF-8", "MLT_ALLOW_MULTIPLE": "1",
                "TMPDIR": str(self.work / "tmp"), **self.extra_env}

    def write_settings(self, **values: Any) -> None:
        folder = self.data_dir / "settings"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "settings.json").write_text(json.dumps({"schema_version": 2, "first_run_done": True, "open_folder_after": False,
                                                          "output_dir": str(self.work / "out"), **values}), encoding="utf-8")

    def run(self, args: list[str], timeout: float, measure: bool = True) -> dict[str, Any]:
        (self.work / "tmp").mkdir(exist_ok=True)
        argv = self.wrapper + (["/usr/bin/time", "-l"] if measure else []) + [str(self.exe), *args]
        started = time.monotonic()
        proc = subprocess.Popen(argv, cwd=self.work, env=self.env(), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        opened: set[str] = set()
        try:
            while proc.poll() is None and time.monotonic() - started < timeout:
                opened |= self._open_files(proc.pid)
                time.sleep(2.0)
            timed_out = proc.poll() is None
            if timed_out:
                proc.kill()
            stderr = proc.communicate(timeout=30)[1] or ""
        finally:
            if proc.poll() is None:
                proc.kill()
        return {"seconds": time.monotonic() - started, "code": proc.returncode, "timed_out": timed_out,
                "peak_bytes": intel.parse_peak_memory(stderr), "opened_files": opened}

    @staticmethod
    def _open_files(pid: int) -> set[str]:
        done = subprocess.run(["lsof", "-p", str(pid), "-Fn"], capture_output=True, text=True, check=False)
        return {line[1:] for line in done.stdout.splitlines() if line.startswith("n/")}


def no_dev_files(opened: set[str]) -> tuple[bool, str]:
    bad = sorted(p for p in opened if (str(ROOT) in p or "/.venv" in p or "/opt/homebrew" in p or "/usr/local/Cellar" in p or "/site-packages/" in p)
                 and f"/{intel.APP_FOLDER}/" not in p and f"/{intel.LEGACY_APP_FOLDER}/" not in p)
    return (not bad), ("opened " + ", ".join(bad[:3])) if bad else f"{len(opened)} files opened, none from the repository, a virtualenv or Homebrew"


NOVELTY_VOICES = {"Albert", "Bad News", "Bahh", "Bells", "Boing", "Bubbles", "Cellos", "Deranged", "Good News", "Hysterical", "Jester", "Organ",
                  "Superstar", "Trinoids", "Whisper", "Wobble", "Zarvox", "Fred", "Junior", "Kathy", "Princess", "Ralph"}
PREFERRED_VOICES = ("Samantha", "Alex", "Daniel", "Karen", "Moira", "Tessa", "Rishi")


def pick_voice(say_listing: list[str]) -> str:
    """A clear English voice from ``say -v ?`` lines ("Name   en_US   # sample"): a preferred one, else any that is not a novelty voice."""
    voices = []
    for line in say_listing:
        m = re.match(r"^(?P<name>.+?)\s+(?P<loc>[a-z]{2,3}[_-][A-Za-z]{2,4})\s+#", line)
        if m and m["loc"].lower().startswith("en"):
            voices.append(m["name"].strip())
    for name in PREFERRED_VOICES:
        if name in voices:
            return name
    return next((v for v in voices if v not in NOVELTY_VOICES and "(" not in v), "")


def make_speech(work: Path, text: str, name: str) -> Path:
    """Speech from the Mac's own built-in voice (offline): say -> AIFF -> 16 kHz WAV."""
    voice = pick_voice(subprocess.run(["say", "-v", "?"], capture_output=True, text=True, check=False).stdout.splitlines())
    aiff, wav = work / f"{name}.aiff", work / f"{name}.wav"
    subprocess.run(["say", *(["-v", voice] if voice else []), "-o", str(aiff), text], check=True)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(wav)], check=True)
    return wav


def make_video(wav: Path, mp4: Path) -> Path:
    """A short real MP4 (mpeg4 video + aac audio) made with PyAV from the build virtualenv - the app under test is not involved."""
    import av
    import numpy as np

    with av.open(str(wav)) as src, av.open(str(mp4), "w") as out:
        video = out.add_stream("mpeg4", rate=15)
        video.width, video.height, video.pix_fmt = 640, 360, "yuv420p"
        audio = out.add_stream("aac", rate=16000)
        audio.layout = "mono"
        seconds = 0.0
        for frame in src.decode(audio=0):
            frame.pts = None
            for packet in audio.encode(frame):
                out.mux(packet)
            seconds += frame.samples / frame.sample_rate
        for packet in audio.encode(None):
            out.mux(packet)
        for i in range(int(seconds * 15)):
            image = np.full((360, 640, 3), (i * 3) % 255, dtype=np.uint8)
            for packet in video.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                out.mux(packet)
        for packet in video.encode(None):
            out.mux(packet)
    return mp4


# ------------------------------------------------------------------------------------------------------------ the validation
def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--app", type=Path, default=DEFAULT_APP)
    parser.add_argument("--dmg", type=Path, required=False, help="the DMG as downloaded (its quarantine flag is tested); its SHA-256 goes in the report")
    parser.add_argument("--attest", default="", help="comma list of checks you verified by eye: frontend,study_workspace,tts")
    parser.add_argument("--report", type=Path, default=ROOT / "release" / "validation_report.json")
    args = parser.parse_args(argv)
    if sys.platform != "darwin":
        print("REAL INTEL MAC REQUIRED - this validation only means something on the Mac itself", file=sys.stderr)
        return 2

    checks: dict[str, str] = {}
    notes: dict[str, str] = {}
    facts = machine_facts()
    off, why = network_is_off()
    report: dict[str, Any] = {"real_hardware": real_hardware(), "machine": facts["machine"], "facts": facts, "network_disconnected": off,
                              "network_note": why, "checks": checks, "notes": notes, "attested": [], "dmg_sha256": sha256(args.dmg) if args.dmg else ""}
    say(f"{facts['model']} | {facts['cpu']} | {facts['ram_gb']} GB | macOS {facts['macos']} | real hardware: {report['real_hardware']} | network off: {off} ({why})")

    def record(key: str, ok: bool, note: str = "") -> None:
        checks[key] = "PASS" if ok else "FAIL"
        notes[key] = note
        say(f"{key}: {checks[key]} {note}")

    work = Path(tempfile.mkdtemp(prefix="mlt_validate_"))
    app = args.app
    session = Session(app, work)
    try:
        # install state: the app is where a student puts it, the DMG is ejected
        record("drag_to_applications", app.is_dir() and str(app).startswith("/Applications/"), str(app))
        mounted = subprocess.run(["mount"], capture_output=True, text=True, check=False).stdout
        record("dmg_eject", intel.VOLUME_NAME not in mounted, "volume not mounted")
        # Gatekeeper on the downloaded (quarantined) disk image
        if args.dmg:
            q = work / args.dmg.name
            shutil.copyfile(args.dmg, q)
            subprocess.run(["xattr", "-w", "com.apple.quarantine", f"0083;{int(time.time()):x};Safari;", str(q)], check=False)
            mount = work / "mnt"
            mount.mkdir()
            att = subprocess.run(["hdiutil", "attach", "-nobrowse", "-readonly", "-mountpoint", str(mount), str(q)], capture_output=True, text=True, check=False)
            record("dmg_mount", att.returncode == 0 and (mount / intel.APP_FOLDER).is_dir(), (att.stderr or "")[:120])
            if att.returncode == 0:
                copy = work / "gatekeeper" / intel.APP_FOLDER
                copy.parent.mkdir()
                subprocess.run(["ditto", str(mount / intel.APP_FOLDER), str(copy)], check=True)
                subprocess.run(["xattr", "-w", "com.apple.quarantine", f"0083;{int(time.time()):x};Safari;", str(copy)], check=False)
                spctl = subprocess.run(["spctl", "--assess", "--type", "execute", "-vv", str(copy)], capture_output=True, text=True, check=False)
                record("gatekeeper_downloaded_file", spctl.returncode == 0, ((spctl.stdout or "") + (spctl.stderr or "")).strip().replace("\n", " | ")[:200])
                subprocess.run(["hdiutil", "detach", str(mount)], capture_output=True, check=False)
        else:
            checks["dmg_mount"] = checks["gatekeeper_downloaded_file"] = "not run (no --dmg)"
        record("offline_launch", off, why)

        # the deep self-test: model checksums, medical_vocab, OCR models, licences, voices
        env = {**session.env(), "MLT_SELFTEST_DEEP": "1"}
        (work / "tmp").mkdir(exist_ok=True)
        out = work / "selftest.json"
        done = subprocess.run([str(session.exe), "--selftest", str(out)], env=env, cwd=work, check=False, timeout=1200)
        st = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {"ok": False, "checks": []}
        failed = [c["name"] for c in st.get("checks", []) if not c["ok"]]
        record("launch_from_applications", done.returncode == 0 and bool(st.get("ok")), "self-test " + ("ok" if not failed else "failed: " + ", ".join(failed)))
        by_name = {c["name"]: c for c in st.get("checks", [])}
        record("whisper_large_v3", "Model discovery" in by_name and by_name["Model discovery"]["ok"] and "Whisper Large-v3: installed" in by_name["Model discovery"]["detail"],
               by_name.get("Model discovery", {}).get("detail", "")[:120])
        record("tts", by_name.get("Read aloud", {}).get("ok", False) and not by_name.get("Read aloud", {}).get("skipped", False), by_name.get("Read aloud", {}).get("detail", ""))

        # real work, on the Mac's own speech
        session.write_settings()
        speech = make_speech(work, "The glycolysis pathway converts glucose to pyruvate. Hexokinase phosphorylates glucose and ATP is produced.", "audio")
        medical = make_speech(work, MEDICAL_SENTENCE, "medical")
        audio_seconds = 0.0
        result = work / "audio_result.json"
        r = session.run(["--autorun", str(speech), "--result", str(result), "--timeout", "1500", "--language", "en"], 1600)
        data = json.loads(result.read_text(encoding="utf-8")) if result.exists() else {}
        audio_seconds = float(data.get("audio_seconds", 0) or 0)
        raw = ""
        if data.get("output_dir"):
            raw_file = Path(data["output_dir"]) / "local_raw_transcript.txt"
            raw = raw_file.read_text(encoding="utf-8").lower() if raw_file.is_file() else ""
        record("audio_stt", data.get("state") == "completed" and ("glycolysis" in raw or "hexokinase" in raw),
               f"{data.get('state')} device={data.get('device')} compute={data.get('compute_type')}")
        report["performance"] = {"audio_seconds": audio_seconds, "processing_seconds": round(r["seconds"], 1), "real_time_factor": intel.realtime_factor(r["seconds"], audio_seconds),
                                 "peak_ram_gb": round((r["peak_bytes"] or 0) / 1024 ** 3, 2), "backend": "faster-whisper (CTranslate2) on the CPU", "device": data.get("device"),
                                 "compute_type": data.get("compute_type")}
        opened = set(r["opened_files"])

        mp4 = make_video(speech, work / "short.mp4")
        before = len(events_of(session.data_dir))
        result_v = work / "video_result.json"
        rv = session.run(["--autorun", str(mp4), "--result", str(result_v), "--timeout", "1500", "--language", "en"], 1600, measure=False)
        dv = json.loads(result_v.read_text(encoding="utf-8")) if result_v.exists() else {}
        record("video_stt", dv.get("state") == "completed", f"{dv.get('state')} {dv.get('audio_seconds')} s")
        ok, note = video_progress_ok(events_of(session.data_dir)[before:])
        record("video_progress", ok, note)
        opened |= set(rv["opened_files"])

        result_q = work / "queue_result.json"
        a2 = make_speech(work, "The citric acid cycle takes place in the mitochondria.", "second")
        rq = session.run(["--autorun-queue", f"{speech}|{a2}", "--result", str(result_q), "--timeout", "1500", "--language", "en"], 1600, measure=False)
        dq = json.loads(result_q.read_text(encoding="utf-8")) if result_q.exists() else {}
        states = [i["state"] for i in dq.get("items", [])]
        record("queue", dq.get("state") == "completed" and states == ["completed", "completed"], f"items {states} (the on-screen '1 of 2' / '2 of 2' text is part of the eyes check)")

        result_m = work / "med_result.json"
        session.run(["--autorun", str(medical), "--result", str(result_m), "--timeout", "1500", "--language", "en"], 1600, measure=False)
        dm = json.loads(result_m.read_text(encoding="utf-8")) if result_m.exists() else {}
        ok, note = medical_check(Path(dm["output_dir"])) if dm.get("output_dir") else (False, "no output folder")
        record("medical_expert", ok, note)

        # documents
        from scripts.smoke_packaged import make_scan, make_text_pdf
        docs = work / "docs"
        docs.mkdir()
        make_text_pdf(docs / "embedded.pdf", ["The nephron is the functional unit of the kidney and filters blood plasma in the glomerulus."])
        make_scan(docs / "scan.pdf", "The afferent arteriole regulates glomerular filtration")
        make_scan(docs / "slide.png", "The afferent arteriole regulates glomerular filtration")
        (docs / "notes.txt").write_text("Kidney notes: the nephron filters plasma.", encoding="utf-8")
        result_d = work / "docs_result.json"
        session.run(["--autorun-queue", "|".join(str(docs / n) for n in ("embedded.pdf", "scan.pdf", "slide.png", "notes.txt")), "--result", str(result_d),
                     "--timeout", "600", "--language", "en"], 700, measure=False)
        dd = json.loads(result_d.read_text(encoding="utf-8")) if result_d.exists() else {}
        ok, note = check_documents_mac(dd.get("items", []))
        items = {i["name"]: i for i in dd.get("items", [])}
        record("pdf", items.get("embedded.pdf", {}).get("state") == "completed" and "embedded.pdf: " not in note, note)
        record("ocr", ok, note)

        # nothing downloaded, nothing copied, no development environment
        models_dir = session.data_dir / "models"
        leftovers = sorted(p.name for p in models_dir.iterdir()) if models_dir.is_dir() else []
        record("no_model_download", off and not leftovers and not (models_dir / ".partial").exists(), "no model files in the user's data folder: " + (", ".join(leftovers) or "none"))
        ok, note = no_dev_files(opened)
        record("no_dev_environment", ok, note)

        # a second launch finds everything again
        done2 = subprocess.run([str(session.exe), "--selftest", str(work / "selftest2.json")], env=session.env(), cwd=work, check=False, timeout=900)
        record("app_restart", done2.returncode == 0, "second start")
    finally:
        pass

    for key in (a.strip() for a in args.attest.split(",") if a.strip()):
        if key in ("frontend", "study_workspace", "tts"):
            checks[key] = "PASS"
            report["attested"].append(key)
    for key in intel.VALIDATION_KEYS:
        checks.setdefault(key, "not run")
    report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)
    bad = [k for k in intel.VALIDATION_KEYS if checks.get(k) != "PASS"]
    say(f"report: {args.report}")
    say("ALL CHECKS PASS" if not bad else "NOT PASSING: " + ", ".join(f"{k}={checks.get(k)}" for k in bad))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
