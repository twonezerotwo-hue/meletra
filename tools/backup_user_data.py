"""Integrity-verified backup of a person's MELETRA data before anything is repaired.

    python tools/backup_user_data.py --dest BACKUP_DIR --appdata DIR --root DIR [--root DIR ...] [--reference-models DIR]

* every file of the application data folder (jobs, logs, settings, checkpoints - NOT models, temp, cache) and of every transcript root is copied, byte for byte, with its times;
* the SHA-256 of the original is taken BEFORE the copy and the SHA-256 of the copy AFTER it; a difference fails the backup;
* models are not duplicated: their size and SHA-256 are recorded as references;
* ``BACKUP_INVENTORY.json`` lists original path, size, SHA-256, backup path and the verification status of every file. Exit code 0 only when every file verified.

The tool only READS the originals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

SKIP_APPDATA = {"models", "temp", "cache"}


def lp(path: Path) -> Path:
    """Extended-length form of a Windows path: a lecture named after a long video file name easily passes the 260-character limit of the ordinary form."""
    text = str(path)
    prefix = "\\\\?\\"
    if sys.platform == "win32" and not text.startswith(prefix):
        text = prefix + str(Path(text).resolve())
    return Path(text)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with lp(path).open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def walk_files(source: Path) -> list[Path]:
    """Every file below ``source``, found through the extended-length form so a path past 260 characters is listed too (the ordinary ``Path.rglob`` cannot see it and the
    backup would silently leave such a file out)."""
    files: list[Path] = []
    base = lp(source)
    stack = [base]
    while stack:
        folder = stack.pop()
        try:
            entries = list(folder.iterdir())
        except OSError:
            continue
        for entry in entries:
            (stack if entry.is_dir() else files).append(entry)
    prefix = str(lp(source))
    return sorted(source / Path(str(f)[len(prefix):].lstrip("\\/")) for f in files)


def copy_tree(source: Path, target: Path, label: str, skip_top: set[str], rows: list[dict]) -> None:
    for path in walk_files(source):
        relative = path.relative_to(source)
        if relative.parts and relative.parts[0] in skip_top:
            continue
        destination = target / relative
        row = {"group": label, "original": str(path), "size": lp(path).stat().st_size, "backup": str(destination)}
        try:
            lp(destination.parent).mkdir(parents=True, exist_ok=True)
            before = sha256_of(path)
            shutil.copy2(lp(path), lp(destination))
            after = sha256_of(destination)
            row.update(sha256=before, backup_sha256=after, verified=(before == after and lp(destination).stat().st_size == row["size"]))
        except OSError as exc:
            row.update(sha256="", backup_sha256="", verified=False, error=f"{type(exc).__name__}: {exc}")
        rows.append(row)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--appdata", type=Path, required=True)
    ap.add_argument("--root", type=Path, action="append", default=[], help="a folder that holds lecture output folders (may be repeated)")
    ap.add_argument("--reference-models", type=Path, default=None, help="the models folder: recorded by size and SHA-256, not copied")
    args = ap.parse_args(argv)
    dest = args.dest
    dest.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    started = time.time()
    copy_tree(args.appdata, dest / "appdata" / args.appdata.name, "application data", SKIP_APPDATA, rows)
    for index, root in enumerate(args.root, start=1):
        if root.is_dir():
            copy_tree(root, dest / "transcript_roots" / f"{index}_{root.parent.name}_{root.name}".replace(" ", "_"), "transcripts", set(), rows)
    referenced = []
    if args.reference_models and args.reference_models.is_dir():
        for path in sorted(args.reference_models.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".bin", ".model", ".json", ".txt"} and ".partial" not in path.parts:
                referenced.append({"path": str(path), "size": path.stat().st_size, "sha256": sha256_of(path)})
    ok = all(r["verified"] for r in rows)
    inventory = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "files": len(rows), "all_verified": ok, "seconds": round(time.time() - started, 1), "entries": rows, "referenced_models": referenced}
    (dest / "BACKUP_INVENTORY.json").write_text(json.dumps(inventory, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(rows)} files, {sum(1 for r in rows if r['verified'])} verified, {len(referenced)} model files referenced -> {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
