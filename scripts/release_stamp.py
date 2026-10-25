"""Build provenance for a release.

    python scripts/release_stamp.py stamp    --tests-passed N [--tests-failed 0 --tests-skipped K] [--note TEXT]
        refuses a dirty tree; writes dist/vX/MedicalLectureTranscriber/build_info.json (version, commit, branch, date, tests)
        so the installer and the portable zip carry it (About / Diagnostics show the commit)
    python scripts/release_stamp.py finalize
        writes release/vX/Windows/release_manifest.json + SHA256SUMS.txt for the installer and the portable zip
    python scripts/release_stamp.py verify
        recomputes the hashes and compares them with SHA256SUMS.txt
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from build_tools.assemble import sha256_of                           # noqa: E402
from build_tools.layout import dist_dir, release_dir                 # noqa: E402
from build_tools.release import WINDOWS_INSTALLER, WINDOWS_PORTABLE  # noqa: E402
from src import __version__                                          # noqa: E402

DIST = dist_dir(__version__, ROOT)
RELEASE = release_dir(__version__, "Windows", ROOT)
MANIFEST = RELEASE / "release_manifest.json"
SUMS = RELEASE / "SHA256SUMS.txt"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def knowledge_info(dist: Path) -> dict:
    """What the package ships as medical knowledge: read straight from the database file inside the build (read-only)."""
    import sqlite3

    files = sorted(dist.rglob("medical_knowledge_public.db"))
    if not files:
        return {"bundled": False}
    db = sqlite3.connect(f"file:{files[0].as_posix()}?mode=ro", uri=True)
    try:
        meta = dict(db.execute("SELECT key, value FROM meta").fetchall())
        sources = db.execute("SELECT name, version, redistribution_allowed FROM source ORDER BY name").fetchall()
        return {"bundled": True, "name": meta.get("name"), "build_version": meta.get("build_version"), "profile": meta.get("profile"), "schema_version": meta.get("schema_version"),
                "concepts": db.execute("SELECT COUNT(*) FROM concept").fetchone()[0], "terms": db.execute("SELECT COUNT(*) FROM term").fetchone()[0],
                "sources": [{"name": n, "version": v, "redistributable": bool(r)} for n, v, r in sources], "sha256": sha256_of(files[0]),
                "all_sources_redistributable": all(r == 1 for _n, _v, r in sources)}
    finally:
        db.close()


def media_info(dist: Path) -> dict:
    f = dist / "media_libraries.json"
    if not f.is_file():
        return {"ffmpeg_for_pyav": "PyAV's own build (GPL-class: libx264 / libx265)"}
    m = json.loads(f.read_text(encoding="utf-8"))
    return {"ffmpeg_for_pyav": f"LGPL-only FFmpeg {m.get('version')} (no libx264, no libx265)", "archive_sha256": m.get("archive_sha256"), "source": m.get("source")}


def stamp(args: argparse.Namespace) -> int:
    dirty = git("status", "--porcelain", "--untracked-files=no")
    if dirty and not args.allow_dirty:
        print("refused: the working tree has uncommitted changes (a release is built from one tested commit):\n" + dirty)
        return 1
    info = {"version": __version__, "branch": git("branch", "--show-current"), "commit": git("rev-parse", "HEAD"),
            "build_date": datetime.now().astimezone().isoformat(timespec="seconds"),
            "tests": {"passed": args.tests_passed, "failed": args.tests_failed, "skipped": args.tests_skipped},
            "knowledge": knowledge_info(DIST), "media_libraries": media_info(DIST),
            "study_quality": {"engine": "MELETRA Automatic Study Engine (local AI model when available, otherwise the offline method; every item passes the premium gate)",
                              "languages": {"en": "validated study facts + premium gate", "other": "standard sentence-based method + premium gate"}},
            "medical_correction": {"default": "Medical Expert v3 (on)", "medical_intelligence_v4": {"default": "off", "label": "Experimental / Advanced"}}}
    if args.note:
        info["note"] = args.note
    DIST.mkdir(parents=True, exist_ok=True)
    (DIST / "build_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"stamped {DIST / 'build_info.json'}: {info['version']} {info['commit'][:7]} ({info['branch']})")
    return 0


def finalize(args: argparse.Namespace) -> int:
    info = json.loads((DIST / "build_info.json").read_text(encoding="utf-8"))
    files = {}
    for name in (WINDOWS_INSTALLER, WINDOWS_PORTABLE):
        path = RELEASE / name
        if not path.is_file():
            print(f"missing {path}")
            return 1
        files[name] = {"bytes": path.stat().st_size, "sha256": sha256_of(path)}
    manifest = {**info, "platform": "Windows", "architecture": "x64", "artifacts": files,
                "includes": {"core_v3": True, "video_progress_fix": True, "medical_expert": True, "new_frontend": True, "queue": True,
                             "pdf": True, "ocr": True, "tts": True, "model_discovery": True,
                             "study_quality": (ROOT / "src" / "study" / "factbase.py").is_file() and (ROOT / "src" / "study" / "quality.py").is_file(),
                             "medical_intelligence_v4": "opt-in layer, default off"},
                "not_included": ["macOS", "offline USB"]}
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    SUMS.write_text("".join(f"{v['sha256']} *{k}\n" for k, v in files.items()), encoding="utf-8")
    print(f"wrote {MANIFEST} and {SUMS}")
    return 0


def verify(args: argparse.Namespace) -> int:
    bad = 0
    for line in SUMS.read_text(encoding="utf-8").splitlines():
        digest, _, name = line.partition(" *")
        actual = sha256_of(RELEASE / name)
        ok = actual == digest
        bad += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {actual}")
    return 1 if bad else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("stamp")
    s.add_argument("--tests-passed", type=int, required=True)
    s.add_argument("--tests-failed", type=int, default=0)
    s.add_argument("--tests-skipped", type=int, default=0)
    s.add_argument("--note", default="")
    s.add_argument("--allow-dirty", action="store_true", help="only for dry runs; never for a release")
    sub.add_parser("finalize")
    sub.add_parser("verify")
    args = parser.parse_args()
    return {"stamp": stamp, "finalize": finalize, "verify": verify}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
