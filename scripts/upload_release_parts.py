"""Upload big files to a GitHub release in small parts, over a slow or unreliable line.

    python scripts/upload_release_parts.py TAG FILE [FILE ...] [--repo owner/name] [--part-mb 64]

Each file is cut into ``NAME.part001``, ``NAME.part002`` ... (the whole file's SHA-256 stays in the release's SHA256SUMS.txt; the install scripts join the
parts and check it). Parts are uploaded one at a time, a failed part is retried alone, and a part that is already in the release with the right size is
skipped - so the script can be stopped and started again at any time and continues. Files are done in the order given.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = "twonezerotwo-hue/whispermedical"


def gh(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["gh", *args], capture_output=True, text=True, check=False)


def uploaded(tag: str, repo: str) -> dict[str, int]:
    done = gh("release", "view", tag, "--repo", repo, "--json", "assets")
    return {a["name"]: a["size"] for a in json.loads(done.stdout)["assets"]} if done.returncode == 0 else {}


def cut(path: Path, folder: Path, part_bytes: int) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    parts = []
    with path.open("rb") as source:
        index = 1
        while True:
            block = source.read(part_bytes)
            if not block:
                break
            target = folder / f"{path.name}.part{index:03d}"
            if not target.is_file() or target.stat().st_size != len(block):
                target.write_bytes(block)
            parts.append(target)
            index += 1
    return parts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tag")
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--part-mb", type=int, default=64)
    args = parser.parse_args(argv)
    for path in args.files:
        parts = cut(path, path.parent / "_parts", args.part_mb * 1024 * 1024)
        print(f"{path.name}: {len(parts)} parts", flush=True)
        for part in parts:
            for attempt in range(1, 31):
                if uploaded(args.tag, args.repo).get(part.name) == part.stat().st_size:
                    break
                started = time.monotonic()
                done = gh("release", "upload", args.tag, "--repo", args.repo, "--clobber", str(part))
                if done.returncode == 0:
                    print(f"  {part.name} uploaded in {time.monotonic() - started:.0f} s", flush=True)
                    break
                print(f"  {part.name} attempt {attempt} failed: {(done.stderr or '').strip()[:150]}", flush=True)
                time.sleep(min(60, 5 * attempt))
            else:
                print(f"giving up on {part.name}", flush=True)
                return 1
    print("UPLOAD_OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
