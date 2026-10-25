"""The PACKAGED / INSTALLED program on a recording that changes language, then translating it again from the saved transcript - offline.

    python scripts/packaged_multilingual_check.py --exe PATH/MELETRA.exe --model-dir MODELS --pack-dir PACK_PARENT --recording smoke_ko_hi_en.wav --out OUT_DIR [--truth smoke.truth.json]

What it proves, in this order (each step a separate run of the real program, so "after a restart" is built in):
 1. Auto Detect transcribes a Korean -> Hindi -> English recording; every language is in the saved source transcript (language accuracy per sentence against the known truth).
 2. The recording is DELETED. Nothing from now on may need it.
 3. ``--translate-lecture`` translates the saved transcript to Turkish with the offline pack: no speech model is loaded (the process does not even import it), no job, no upload.
 4. The same request again is answered from the saved translation (same creation time).
 5. ``--autorun-translate`` presses the real "Translate to another language" button of the real window for German; reopening it opens the saved translation; no job was created.
 6. All of it ran with ``HF_HUB_OFFLINE`` and proxies pointing nowhere, and the computer's TCP table was watched: no connection to an address outside this computer was seen.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "scripts"))

from offline_check import Watcher  # noqa: E402
from smoke_packaged import clean_env, run_exe, write_settings  # noqa: E402
import eval_multilingual  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--pack-dir", type=Path, required=True)
    ap.add_argument("--recording", type=Path, required=True)
    ap.add_argument("--truth", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)
    args.exe, args.model_dir, args.pack_dir, args.recording, args.out = (path.resolve() for path in (args.exe, args.model_dir, args.pack_dir, args.recording, args.out))
    args.truth = args.truth.resolve() if args.truth else None                         # the program is started in another working folder: every path it gets must be absolute
    args.out.mkdir(parents=True, exist_ok=True)
    work = args.out / "work"
    work.mkdir(exist_ok=True)
    recording = work / args.recording.name
    recording.write_bytes(args.recording.read_bytes())
    truth_path = args.truth or args.recording.with_suffix(".truth.json")
    truth = json.loads(truth_path.read_bytes().decode("utf-8")) if truth_path.is_file() else None
    checks: dict[str, bool] = {}
    detail: dict[str, object] = {"exe": str(args.exe)}
    out_dir = work / "out"
    write_settings(work, output_dir=str(out_dir), extra_model_dirs=[str(args.model_dir), str(args.pack_dir)], open_folder_after=False, output_language="en", device=args.device)
    env_extra = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HTTP_PROXY": "http://127.0.0.1:9", "HTTPS_PROXY": "http://127.0.0.1:9", "ALL_PROXY": "http://127.0.0.1:9", "NO_PROXY": ""}
    env = clean_env(work, env_extra)
    started = time.monotonic()
    with Watcher(args.exe.resolve().parent) as watch:
        # 1. Auto Detect on the mixed recording
        result = work / "autorun.json"
        code, timed_out = run_exe(args.exe, ["--autorun", str(recording), "--result", str(result), "--timeout", "1500", "--language", "auto"], env, work, 1600)
        run = json.loads(result.read_text(encoding="utf-8")) if result.is_file() else {}
        checks["1. the mixed recording was transcribed"] = run.get("state") == "completed"
        lecture = Path(run["output_dir"]) if run.get("output_dir") else None
        source = json.loads((lecture / "source_transcript.json").read_bytes().decode("utf-8")) if lecture and (lecture / "source_transcript.json").is_file() else {}
        languages = source.get("detected_languages", [])
        wanted = set(truth["languages"]) if truth else {"ko", "hi", "en"}
        checks["1. every spoken language is in the saved source transcript"] = wanted <= set(languages)
        detail["detected_languages"] = languages
        detail["transcription"] = {k: run.get(k) for k in ("device", "compute_type", "audio_seconds", "timing")}
        if truth and lecture and (lecture / "local_transcript.json").is_file():
            segments = json.loads((lecture / "local_transcript.json").read_bytes().decode("utf-8")).get("segments", [])
            score = eval_multilingual.score(truth, segments)
            detail["score"] = {k: score[k] for k in ("language_accuracy", "clips_written", "mean_error_rate_by_language")}
            checks["1. every sentence was written and its language is right"] = score["language_accuracy"] >= 0.99 and score["clips_written"] >= 0.99
        english = lecture / "translated_en_transcript.txt" if lecture else None
        checks["1. the English output was made (default Output Language)"] = bool(english and english.is_file() and english.read_text(encoding="utf-8").strip())

        if lecture is None or not lecture.is_dir():                                      # nothing to go on: report it as a failure, not as a crash
            report = {"ok": False, "checks": checks, "detail": detail, "error": "the mixed recording produced no lecture folder"}
            (args.out / "packaged_multilingual_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
            print("FAIL 1. the mixed recording was transcribed")
            print("PACKAGED_MULTILINGUAL_FAILED")
            return 1

        # 2. the recording is gone
        recording.unlink(missing_ok=True)
        checks["2. the recording is deleted"] = not recording.exists()

        # 3. headless translation from the saved transcript
        r3 = work / "translate_tr.json"
        code3, _ = run_exe(args.exe, ["--translate-lecture", str(lecture), "--to", "tr", "--result", str(r3)], env, work, 900)
        first = json.loads(r3.read_text(encoding="utf-8")) if r3.is_file() else {}
        checks["3. translation to Turkish from the saved transcript"] = code3 == 0 and first.get("ok") is True and first.get("status") in ("complete", "partial") and not first.get("from_cache")
        checks["3. the speech model was not used (not even loaded)"] = first.get("speech_recogniser_loaded") is False
        detail["translate_tr"] = {k: first.get(k) for k in ("seconds", "counts", "engine", "files")}

        # 4. the same request again: the saved translation
        r4 = work / "translate_tr_again.json"
        run_exe(args.exe, ["--translate-lecture", str(lecture), "--to", "tr", "--result", str(r4)], env, work, 300)
        second = json.loads(r4.read_text(encoding="utf-8")) if r4.is_file() else {}
        checks["4. asking again opens the saved translation (cache hit, same creation time)"] = second.get("from_cache") is True and second.get("created_at") == first.get("created_at")
        detail["translate_tr_again_seconds"] = second.get("seconds")

        # 5. the real window, the real button
        r5 = work / "autorun_translate.json"
        code5, _ = run_exe(args.exe, ["--autorun-translate", str(lecture), "--to", "de", "--result", str(r5), "--timeout", "600"], env, work, 700)
        window = json.loads(r5.read_text(encoding="utf-8")) if r5.is_file() else {}
        checks["5. the Translate to another language button is on the result screen"] = (window.get("button") or {}).get("visible") is True and (window.get("button") or {}).get("enabled") is True
        checks["5. German was translated from the window, no job was created"] = window.get("exit_code") == 0 and window.get("jobs_created") == 0 and window.get("job_running") is False
        checks["5. opening German a second time used the saved translation"] = window.get("reopened_saved_translation") is True
        checks["5. the lecture shows a tab for every saved translation"] = {"original", "de"} <= set(window.get("tabs") or [])
        detail["window"] = {k: window.get(k) for k in ("stage1", "stage2", "tabs", "jobs_created")}
        archive_state = (lecture / "translations")
        detail["saved_translations"] = sorted(p.stem for p in archive_state.glob("*.json")) if archive_state.is_dir() else []
        checks["5. the earlier translations are all still there"] = {"en", "tr", "de"} <= set(detail["saved_translations"])
    checks["6. no connection to an address outside this computer was seen"] = not watch.outside
    detail["watcher"] = {"samples": watch.samples, "processes": len(watch.seen_pids), "outside": watch.outside}
    ok = all(checks.values())
    report = {"ok": ok, "seconds": round(time.monotonic() - started), "checks": checks, "detail": detail}
    (args.out / "packaged_multilingual_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    for name, passed in checks.items():
        print(("PASS " if passed else "FAIL ") + name)
    print("PACKAGED_MULTILINGUAL_OK" if ok else "PACKAGED_MULTILINGUAL_FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
