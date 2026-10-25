"""Inspect a BUILT app folder (or any folder of binaries) for GPL-class components and for private material that must not ship.

    python tools/license_audit.py dist/v4.0.0/MedicalLectureTranscriber [--json out.json]

What it reads is the binaries themselves, not the notices that describe them:
    * FFmpeg libraries print their build configuration inside the binary ("--enable-gpl", "--enable-libx264", "--enable-nonfree" ...). Any DLL / dylib / so / exe that carries such a string
      is listed with the exact flags. ``--enable-gpl`` / ``libx264`` / ``libx265`` / ``--enable-nonfree`` are the findings that decide the licence of the distribution.
    * Python packages with their licence from the dist-info metadata (``License``, ``Classifier``, ``License-Expression``): GPL / AGPL / SSPL / "non-commercial" are flagged.
    * Private material: personal paths of the build machine, lecture file names, transcripts and databases that are not the public knowledge pack.

Exit code 0 = no GPL-class component and no private material; 1 = findings (they are listed, never hidden); the verdict is a fact about the files, not a legal opinion.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

BINARY_SUFFIXES = {".dll", ".pyd", ".exe", ".dylib", ".so"}
FFMPEG_FLAGS = re.compile(rb"--enable-(?:gpl|version3|nonfree|libx264|libx265|libxvid|libfdk[-_]aac|libvmaf|libvpx|libopus|libmp3lame|libaom|libdav1d|libsvtav1)\b")
GPL_FLAGS = {b"--enable-gpl", b"--enable-nonfree", b"--enable-libx264", b"--enable-libx265", b"--enable-libxvid", b"--enable-libfdk-aac", b"--enable-libfdk_aac"}
COPYLEFT_LICENSE = re.compile(r"(?i)\b(?:AGPL|SSPL|GPL(?:v?[23])?(?! with)|GNU General Public|non-?commercial|CC[- ]BY[- ]NC)\b")
LGPL = re.compile(r"(?i)\bLGPL|Lesser General Public")
def personal_pattern(owners: tuple[str, ...]) -> "re.Pattern[bytes]":
    """The OWNER's user folder on the build machine (C:\\Users\\<name>\\, /Users/<name>/): paths that third parties baked into their own binaries (C:\\Users\\qt\\work ...) are not private."""
    names = b"|".join(re.escape(o.encode()) for o in owners if o)
    return re.compile(rb"(?i)(?:[A-Z]:[\\/]+Users[\\/]+|/Users/|/home/)(?:" + names + rb")[\\/]") if names else re.compile(rb"(?!x)x")
PRIVATE_NAMES = (b"PRIVATE_LECTURE_", b"PRIVATE_MEDIA_", b"Fixture Library")
CHUNK = 8 * 1024 * 1024


def scan_binary(path: Path, personal_re) -> dict:
    flags: set[bytes] = set()
    personal = private = False
    tail = b""
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(CHUNK)
            if not chunk:
                break
            data = tail + chunk
            flags.update(m.group(0) for m in FFMPEG_FLAGS.finditer(data))
            personal = personal or bool(personal_re.search(data))
            private = private or any(n in data for n in PRIVATE_NAMES)
            tail = data[-128:]
    return {"flags": sorted(f.decode() for f in flags), "gpl_class": sorted(f.decode() for f in flags & GPL_FLAGS), "personal_path": personal, "private_name": private}


def python_licenses(folder: Path) -> list[dict]:
    out = []
    for meta in list(folder.rglob("*.dist-info/METADATA")):
        text = meta.read_text(encoding="utf-8", errors="replace")[:20000]
        name = re.search(r"(?m)^Name: (.+)$", text)
        lic = re.search(r"(?m)^License(?:-Expression)?: (.+)$", text)
        classifiers = "; ".join(re.findall(r"(?m)^Classifier: License :: (.+)$", text))
        label = ((lic.group(1).strip() if lic else "") + " " + classifiers).strip()[:200]
        out.append({"package": name.group(1).strip() if name else meta.parent.name, "license": label,
                    "copyleft": bool(COPYLEFT_LICENSE.search(label)) and not LGPL.search(label), "lgpl": bool(LGPL.search(label))})
    return sorted(out, key=lambda r: r["package"].lower())


def audit(folder: Path, max_files: int = 0, owners: tuple = ()) -> dict:
    folder = Path(folder)
    personal_re = personal_pattern(owners or (os.environ.get("USERNAME", ""), os.environ.get("USER", "")))
    binaries, gpl, personal, private = [], [], [], []
    count = 0
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in BINARY_SUFFIXES:
            continue
        count += 1
        if max_files and count > max_files:
            break
        result = scan_binary(path, personal_re)
        rel = str(path.relative_to(folder))
        if result["flags"]:
            binaries.append({"file": rel, **result})
        if result["gpl_class"]:
            gpl.append({"file": rel, "flags": result["gpl_class"]})
        if result["personal_path"]:
            personal.append(rel)
        if result["private_name"]:
            private.append(rel)
    packages = python_licenses(folder)
    other_private = [str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file() and (re.search(r"(?i)local_(raw_|clean_)?transcript|medical_corrected_|review_verdicts", p.name)
                                                                                           or (p.suffix == ".db" and p.name != "medical_knowledge_public.db"))]
    findings = {"gpl_class_binaries": gpl, "copyleft_python_packages": [p for p in packages if p["copyleft"]], "personal_paths_in_binaries": personal, "private_names_in_binaries": private,
                "private_files": other_private}
    return {"folder": str(folder), "binaries_scanned": count, "ffmpeg_configuration": binaries, "python_packages": len(packages), "lgpl_python_packages": [p["package"] for p in packages if p["lgpl"]],
            "findings": {k: v for k, v in findings.items() if v}, "result": "NO GPL-CLASS COMPONENT AND NO PRIVATE MATERIAL FOUND" if not any(findings.values()) else "FINDINGS"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--max-files", type=int, default=0)
    ap.add_argument("--owner", action="append", default=[], help="a user name of the build machine that must not appear in a binary (default: the current user)")
    a = ap.parse_args(argv)
    report = audit(a.folder, a.max_files, tuple(a.owner))
    print(f"{report['folder']}: {report['binaries_scanned']} binaries, {report['python_packages']} python packages -> {report['result']}")
    for cfg in report["ffmpeg_configuration"][:12]:
        print("  ffmpeg build flags in", cfg["file"], ":", ", ".join(cfg["flags"]))
    for kind, items in report["findings"].items():
        print(f"  {kind}: {len(items)}")
        for item in items[:10]:
            print("     ", item)
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0 if not report["findings"] else 1


if __name__ == "__main__":
    sys.exit(main())
