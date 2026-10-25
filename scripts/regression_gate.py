"""The v0.1.1 regression gate: nothing of the production release may change while 0.2.0 is developed.

    python scripts/regression_gate.py                      baseline integrity + core-source check + test suite
    python scripts/regression_gate.py --golden             ... + the same recording through the PRODUCTION 0.1.1 exe and the NEW exe,
                                                           raw transcript, JSON, SRT, VTT must be identical
    python scripts/regression_gate.py --full-smoke         ... + scripts/smoke_packaged.py on the new exe (GPU, CPU fallback, PyAV
                                                           fallback, pause/resume)
    options: --exe NEW.exe  --baseline-exe OLD.exe  --audio FILE  --skip-tests  --report FILE

Exit code 0 = PASS. Every check prints PASS / FAIL with the reason; the report (JSON) lists them all. The gate only READS the production
files (git tag, the SHA-256 manifests, the 0.1.1 exe is started from a temporary folder with a private data folder).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

BASELINE_TAG = "desktop-app-v0.1.1"
BASELINE_COMMIT = "c9f8e84"
MIN_TESTS = 712                                   # the 0.1.1 suite; the 0.2 suite is larger
BASELINE_DIR = ROOT / "docs" / "baseline"
PROD_EXE = ROOT / "dist" / "MedicalLectureTranscriber" / "MedicalLectureTranscriber.exe"

#: behaviour of 0.1.1 that must not change: these files must be byte-identical to the baseline commit
CORE_UNCHANGED = [
    "src/cleaner.py", "src/language.py",
    "src/long_audio.py", "src/checkpoint.py", "src/uncertainty.py", "src/medical_report.py", "src/gate.py", "src/guard.py",
    "src/consensus.py", "src/context_engine.py", "src/vocabulary.py", "src/budget.py", "src/entities.py",
    "src/topic.py", "src/priority.py", "src/second_opinion.py", "src/deepgram_backend.py", "src/metrics.py",
]
#: 4.1.1 (multilingual recordings): these core files gained a few ADDITIVE lines that only act when a recording changes language or sings (``language_plan`` / ``vad_override`` are
#: None by default, a language tag is written only when a transcript holds more than one language, new schema fields are empty by default, a credit in Hangul / Han / Devanagari is
#: flagged like the English ones, a new ``multilingual:`` section of the settings). Single-language behaviour is proven unchanged by tests/test_regression_4_1_0.py and the golden comparison.
CORE_MULTILINGUAL_ONLY = ["src/chunked_decode.py", "src/exporters.py", "src/schema.py", "src/quality.py", "config/transcription.yaml"]
MULTILINGUAL_KEYWORDS = ("multilingual", "language", "coverage", "recheck", "level_quiet", "leveler", "source_transcript", "ml_", "activity", "unload", "rejected", "plan", "credit",
                         "artifact", "vad_override", "sequence")
#: files that may differ, but only in hunks that are about the macOS engine (``mlx``)
CORE_MLX_ONLY = ["src/pipeline.py", "src/model_manager.py", "src/config.py"]
#: 3.0.1 (video preparation): real progress + prompt pause / cancel before the first segment. These files may differ ONLY in hunks about
#: progress reporting / stopping / the ffmpeg watchdog; what the transcription produces is proven unchanged by the golden comparison.
PROGRESS_KEYWORDS = ("progress", "stage", "should_stop", "pauserequested", "_stop_point", "extractioncancelled", "stall", "_kill",
                     "threading", "callable", "emit", "extracted", "sampled_so_far", "last_report", "last = [", "-nostats", "demux")
CORE_PROGRESS_ONLY = ["src/audio.py", "src/events.py", "src/transcriber.py"]
#: 3.1 (medical expert layer): the pipeline gains a few call lines into ``src/medical`` (soft language lock, medical corrected transcript);
#: those hunks are recognisable by these names. Raw transcript, JSON, SRT and VTT stay byte-identical for clean recordings (golden comparison).
MEDICAL_KEYWORDS = ("medical_expert", "medical_integration", "custom_medical_terms", "medical_decode_window", "medical_redecode")
#: 4.1 (Output Language): the pipeline gains a step that only ADDS translated_<code>_transcript.* files (src/translation) and ``DecodeRequest.task``
#: ("translate" is used only for that step; the default stays "transcribe"); the privacy note of the metadata names the optional transcript sharing.
TRANSLATION_KEYWORDS = ("translation", "output_language", "privacy_note", "translate")


@dataclass
class Check:
    name: str
    ok: bool = False
    detail: str = ""
    seconds: float = 0.0

    def line(self) -> str:
        return f"[{'PASS' if self.ok else 'FAIL'}] {self.name}: {self.detail}"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False).stdout


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


# ------------------------------------------------------------------------------------------------------------------------- checks
def check_tag() -> Check:
    check = Check("baseline tag")
    resolved = git("rev-parse", f"{BASELINE_TAG}^{{commit}}").strip()      # the tag is annotated: ask for the commit it points at
    expected = git("rev-parse", BASELINE_COMMIT).strip()
    check.ok = bool(resolved) and resolved == expected
    check.detail = f"{BASELINE_TAG} -> {resolved[:8] or 'MISSING'} (expected {expected[:8]})"
    return check


def check_release_files() -> Check:
    check = Check("production release files (installer, portable, full offline)")
    manifest = BASELINE_DIR / "BASELINE_v0.1.1_release.sha256"
    problems, count = [], 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, _, name = line.partition(" *")
        path = ROOT / "release" / name.strip()
        count += 1
        if not path.is_file():
            problems.append(f"{name} is missing")
        elif sha256(path) != digest:
            problems.append(f"{name} CHANGED")
    check.ok, check.detail = not problems, (f"{count} files identical to the recorded SHA-256" if not problems else "; ".join(problems))
    return check


def check_dist_tree() -> Check:
    check = Check("production unpacked app (dist/MedicalLectureTranscriber)")
    data = json.loads((BASELINE_DIR / "BASELINE_v0.1.1_dist.json").read_text(encoding="utf-8"))
    root = ROOT / "dist" / "MedicalLectureTranscriber"
    problems = []
    for rel, info in data["files"].items():
        path = root / rel
        if not path.is_file():
            problems.append(f"{rel} missing")
        elif path.stat().st_size != info["size"] or sha256(path) != info["sha256"]:
            problems.append(f"{rel} changed")
    extra = [p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p.relative_to(root).as_posix() not in data["files"]
             and "__pycache__" not in p.parts]
    if extra:
        problems.append(f"{len(extra)} new file(s), for example {extra[0]}")
    check.ok, check.detail = not problems, (f"{len(data['files'])} files identical" if not problems else "; ".join(problems[:5]))
    return check


def check_core_sources() -> Check:
    check = Check("v0.1.1 source behaviour (core files vs the baseline commit)")
    problems = []
    for rel in CORE_UNCHANGED:
        if git("diff", BASELINE_COMMIT, "--", rel).strip():
            problems.append(f"{rel} differs")
    for rel in CORE_MLX_ONLY:
        diff = git("diff", "-U0", BASELINE_COMMIT, "--", rel)
        hunks = [h for h in re.split(r"(?m)^@@ ", diff)[1:]]
        keywords = ("mlx",) + ((PROGRESS_KEYWORDS + MEDICAL_KEYWORDS + TRANSLATION_KEYWORDS) if rel == "src/pipeline.py" else ())
        keywords += TRANSLATION_KEYWORDS if rel == "src/model_manager.py" else ()
        keywords += MULTILINGUAL_KEYWORDS
        bad = [h.splitlines()[0] for h in hunks if not any(k in h.casefold() for k in keywords)]
        if bad:
            problems.append(f"{rel}: a change that is not about the macOS engine, progress reporting, the Output Language step or multilingual recordings ({bad[0]})")
    for rel in CORE_PROGRESS_ONLY:
        diff = git("diff", "-U0", BASELINE_COMMIT, "--", rel)
        hunks = [h for h in re.split(r"(?m)^@@ ", diff)[1:]]
        bad = [h.splitlines()[0] for h in hunks if not any(k in h.casefold() for k in PROGRESS_KEYWORDS + MULTILINGUAL_KEYWORDS)]
        if bad:
            problems.append(f"{rel}: a change that is not about progress reporting / stopping or multilingual recordings ({bad[0]})")
    for rel in CORE_MULTILINGUAL_ONLY:
        diff = git("diff", "-U0", BASELINE_COMMIT, "--", rel)
        hunks = [h for h in re.split(r"(?m)^@@ ", diff)[1:]]
        bad = [h.splitlines()[0] for h in hunks if not any(k in h.casefold() for k in MULTILINGUAL_KEYWORDS)]
        if bad:
            problems.append(f"{rel}: a change that is not about multilingual recordings ({bad[0]})")
    check.ok = not problems
    check.detail = (f"{len(CORE_UNCHANGED)} core files identical; {', '.join(CORE_MULTILINGUAL_ONLY)} differ only in multilingual hunks; {', '.join(CORE_MLX_ONLY)} differ only in "
                    f"macOS-engine, progress, translation and multilingual hunks; {', '.join(CORE_PROGRESS_ONLY)} only in progress / stop / multilingual hunks" if not problems else "; ".join(problems))
    return check


def check_tests() -> Check:
    check = Check("test suite")
    started = time.monotonic()
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"], cwd=ROOT, capture_output=True,
                          text=True, check=False)
    tail = (proc.stdout or "").strip().splitlines()[-1:] or [""]
    passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", tail[0])) else 0
    failed = int(m.group(1)) if (m := re.search(r"(\d+) failed", tail[0])) else 0
    check.ok = proc.returncode == 0 and failed == 0 and passed >= MIN_TESTS
    check.detail = f"{passed} passed, {failed} failed (the 0.1.1 suite had {MIN_TESTS})"
    check.seconds = time.monotonic() - started
    return check


def _transcribe(exe: Path, audio: Path, language: str) -> Optional[Path]:
    """One real transcription through the real window of ``exe`` (the app's own --autorun hook) in a private data folder."""
    import smoke_packaged as smoke

    work = Path(tempfile.mkdtemp(prefix="mlt_gate_"))
    out_dir = work / "out"
    smoke.write_settings(work, output_dir=str(out_dir), extra_model_dirs=[str(smoke.find_model_dir())], open_folder_after=False)
    result = work / "result.json"
    args = ["--autorun", str(audio), "--result", str(result), "--timeout", "1500", "--language", language]
    smoke.run_exe(exe, args, smoke.clean_env(work, {}), work, 1560)
    if not result.exists():
        return None
    data = json.loads(result.read_text(encoding="utf-8"))
    return Path(data["output_dir"]) if data.get("state") == "completed" else None


def check_golden(new_exe: Path, old_exe: Path, audio: Path, language: str = "en") -> Check:
    check = Check(f"same recording through 0.1.1 and 0.2.0 ({audio.name}, language {language})")
    started = time.monotonic()
    if not new_exe.is_file() or not old_exe.is_file():
        check.detail = f"exe missing: {new_exe if not new_exe.is_file() else old_exe}"
        return check
    old = _transcribe(old_exe, audio, language)
    new = _transcribe(new_exe, audio, language)
    check.seconds = time.monotonic() - started
    if old is None or new is None:
        check.detail = f"a run did not complete (0.1.1: {'ok' if old else 'FAILED'}, 0.2.0: {'ok' if new else 'FAILED'})"
        return check
    differences = []
    for name in ("local_raw_transcript.txt", "local_clean_transcript.txt", "local_transcript_with_timestamps.txt", "local_transcript.srt",
                 "local_transcript.vtt"):
        a, b = (old / name).read_bytes(), (new / name).read_bytes()
        if a != b:
            differences.append(f"{name} differs ({len(a)} vs {len(b)} bytes)")
    ja = json.loads((old / "local_transcript.json").read_text(encoding="utf-8"))
    jb = json.loads((new / "local_transcript.json").read_text(encoding="utf-8"))
    if [(s["start"], s["end"], s["text"]) for s in ja["segments"]] != [(s["start"], s["end"], s["text"]) for s in jb["segments"]]:
        differences.append("JSON segments differ")
    if set(ja) != set(jb) or (ja["segments"] and jb["segments"] and set(ja["segments"][0]) != set(jb["segments"][0])):
        differences.append("JSON schema differs")
    if not ja["segments"] and not jb["segments"]:
        differences.append("both runs produced NO text (identical but empty: not evidence - choose a recording with speech)")
    check.ok = not differences
    check.detail = (f"raw, clean, timestamped, SRT, VTT and JSON identical ({len(ja['segments'])} segments)" if not differences
                    else "; ".join(differences))
    return check


def check_full_smoke(new_exe: Path) -> Check:
    check = Check("packaged smoke test of the new exe (GPU, CPU fallback, PyAV fallback, pause/resume)")
    started = time.monotonic()
    report = Path(tempfile.mkdtemp(prefix="mlt_gate_smoke_")) / "smoke.json"
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "smoke_packaged.py"), "--exe", str(new_exe), "--report", str(report)],
                          cwd=ROOT, capture_output=True, text=True, check=False)
    check.seconds = time.monotonic() - started
    try:
        steps = json.loads(report.read_text(encoding="utf-8")).get("steps", [])
    except (OSError, ValueError):
        steps = []
    failed = [s["name"] for s in steps if not s.get("ok")]
    check.ok = proc.returncode == 0 and bool(steps) and not failed
    check.detail = f"{len(steps) - len(failed)}/{len(steps)} steps passed" + (f"; failed: {', '.join(failed)}" if failed else "")
    return check


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exe", type=Path, default=None, help="the new exe (default: dist/v<version>/MedicalLectureTranscriber/...)")
    parser.add_argument("--baseline-exe", type=Path, default=PROD_EXE)
    parser.add_argument("--audio", type=Path, default=ROOT / "benchmarks" / "audio" / "biochem_clean.wav")
    parser.add_argument("--language", default="en")
    parser.add_argument("--golden", action="store_true")
    parser.add_argument("--full-smoke", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--report", type=Path, default=ROOT / "logs" / "regression_gate.json")
    args = parser.parse_args(argv)

    from build_tools.layout import dist_dir, file_id
    from src import __version__

    new_exe = args.exe or (dist_dir(__version__, ROOT) / f"{file_id(__version__)}.exe")
    checks = [check_tag(), check_release_files(), check_dist_tree(), check_core_sources()]
    for check in checks:
        print(check.line(), flush=True)
    if not args.skip_tests:
        checks.append(check_tests())
        print(checks[-1].line(), flush=True)
    if args.golden:
        checks.append(check_golden(new_exe, args.baseline_exe, args.audio, args.language))
        print(checks[-1].line(), flush=True)
    if args.full_smoke:
        checks.append(check_full_smoke(new_exe))
        print(checks[-1].line(), flush=True)
    checks.append(check_release_files())                                   # still untouched after everything above ran
    checks[-1].name += " (after the run)"
    print(checks[-1].line(), flush=True)
    passed = all(c.ok for c in checks)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"version": __version__, "passed": passed, "checks": [c.__dict__ for c in checks]}, indent=2),
                           encoding="utf-8")
    print("\nREGRESSION GATE:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
