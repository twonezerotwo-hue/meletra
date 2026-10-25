"""A user of 4.0 starts the PACKAGED MELETRA: the archive, the renamed lecture, the transcript, the Study pack, the settings and the model are still there.

    python scripts/packaged_migration_check.py --exe dist/v4.1.0/MELETRA/MELETRA.exe --out OUT_DIR

The 4.0 profile is written by the CODE OF 4.0.0 (the tag ``desktop-app-v4.0.0``, extracted with ``git archive``; ``tests/migration_profile_40.py``), not by a fixture of this
version. The packaged program is started on a copy of it (``MLT_DATA_DIR``; the student's real data is never touched) with ``--ui-check``: the Home and Files pages are read through
the accessibility tree and photographed from the program's own window. Afterwards every file of the 4.0 profile must still be there, byte for byte, except settings.json (the
program upgrades it: the choices of 4.0 stay, transcript sharing is switched OFF).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import packaged_final_check as pfc  # noqa: E402

TAG = "desktop-app-v4.0.0"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(data: Path) -> dict[str, str]:
    return {p.relative_to(data).as_posix(): digest(p) for p in sorted(data.rglob("*")) if p.is_file() and p.name != "expected.json"}


def write_profile_40(work: Path) -> Path:
    tree = work / "v40"
    tree.mkdir()
    archive = subprocess.run(["git", "archive", TAG, "src", "config", "medical_terms", "medical_vocab", "tests/study_support.py"], cwd=str(ROOT), capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(tree)
    data = work / "data"
    env = {k: v for k, v in os.environ.items() if k not in ("QT_QPA_PLATFORM", "QT_QPA_FONTDIR")}
    env.update({"PYTHONPATH": str(tree), "MLT_DATA_DIR": str(data)})
    done = subprocess.run([sys.executable, str(ROOT / "tests" / "migration_profile_40.py"), str(data)], cwd=str(tree), env=env, capture_output=True, text=True, timeout=300)
    if done.returncode != 0:
        raise SystemExit("writing the 4.0 profile failed:\n" + done.stderr[-2000:])
    return data


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="meletra_migration_"))
    data = write_profile_40(work)
    expected = json.loads((data / "expected.json").read_text(encoding="utf-8"))
    before = snapshot(data)
    settings_before = json.loads((data / "settings" / "settings.json").read_text(encoding="utf-8"))
    lecture, translated = pfc.make_lecture_folders(work)
    report = pfc.run_size(args.exe, "1366x768", args.out, work, lecture, translated, "accessibility,home,files,settings")
    after = snapshot(data)
    tree = report.get("accessibility", {})
    cards = sorted({e["name"] for page in ("home", "files") for e in tree.get(page, []) if e.get("class") == "LectureCard"})
    settings_after = json.loads((data / "settings" / "settings.json").read_text(encoding="utf-8"))
    kept = {name: digest(data / name) == h for name, h in before.items() if name != "settings/settings.json" and (data / name).exists()}
    checks = {
        "the packaged program ran to the end without errors": not report.get("errors") and not report.get("timed_out") and report.get("exit_code") == 0,
        "Home and Files pages photographed": sum(1 for s in report.get("screens", []) if "FAILED" not in s) >= 3,
        "the completed lecture of 4.0 is in the archive": any(c.startswith("Lecture Biochem_1.mp4, completed") for c in cards),
        "the lecture whose folder the student renamed is in the archive": any("My own name for the collagen lecture" in c or "Collagen" in c for c in cards),
        "the failed lecture of 4.0 is in the archive": any(c.startswith("Lecture Anatomy_thorax.mp4, failed") for c in cards),
        "no file of the 4.0 profile was lost": all(name in after for name in before),
        "no file of the 4.0 profile was changed (except settings.json)": all(kept.values()),
        "the 4.0 choices stay (language, theme, output folder)": all(settings_after.get(k) == settings_before.get(k) for k in ("language", "theme", "output_dir", "first_run_done", "hotwords", "mode")),
        "transcript sharing is OFF after the upgrade (the chip on Home says so; the file never says it is on)": any(e.get("name") == "Privacy status: Local processing" for e in tree.get("home", []))
                                                                                                    and settings_after.get("share_transcripts", False) is False,
        "the transcript of 4.0 is intact": (Path(expected["lectures"]["job-biochem"]) / "local_clean_transcript.txt").read_text(encoding="utf-8").splitlines()[0] == expected["transcript_first_line"],
        "the model of 4.0 is intact": "models/whisper-large-v3-folder/model.bin" in after and before.get("models/whisper-large-v3-folder/model.bin") == after.get("models/whisper-large-v3-folder/model.bin"),
    }
    result = {"exe": str(args.exe), "tag_of_the_profile": TAG, "profile_version": expected["version"], "lecture_cards": cards, "settings_schema_before_after": [settings_before.get("schema_version"), settings_after.get("schema_version")],
              "checks": checks, "ok": all(checks.values()), "screens": report.get("screens")}
    (args.out / "packaged_migration_report.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    for name, ok in checks.items():
        print(("PASS " if ok else "FAIL ") + name)
    print("MIGRATION_CHECK_OK" if result["ok"] else "MIGRATION_CHECK_FAILED")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
