"""Windows side of the Intel-Mac release: take the DMG that was built AND validated on a real Intel Mac, prove it is the same file, put it where
the upload is prepared. The only place that may print READY TO UPLOAD.

    python scripts/finalize_mac_dmg.py --dmg D:\\transfer\\MELETRA-Intel-Mac-Offline.dmg \\
        --manifest D:\\transfer\\release_manifest.json --validation D:\\transfer\\validation_report.json

It checks, in this order: the SHA-256 of the DMG against the .sha256 written on the Mac and against the manifest; the validation report
(real hardware, x86_64, network disconnected, the same SHA-256, every check PASS); the GPL source-offer contact. Then it copies the DMG to
release/v3.1.0/macOS-Intel/ and to the Desktop, verifies both copies byte for byte (SHA-256), and writes the .sha256 files.
Without a validation report from a real Intel Mac the answer is REAL INTEL MAC REQUIRED, and nothing is copied to the Desktop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from build_tools import macos_intel as intel                          # noqa: E402
from build_tools.layout import base_version, release_dir              # noqa: E402
from src import __version__                                           # noqa: E402

TARGET = release_dir(__version__, "macOS", ROOT).parent / intel.RELEASE_FOLDER
DESKTOP = Path.home() / "Desktop"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dmg", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True, help="release_manifest.json written by the build on the Mac")
    parser.add_argument("--validation", type=Path, help="validation_report.json written by scripts/validate_macos_intel.py on the Mac")
    parser.add_argument("--desktop", type=Path, default=DESKTOP)
    args = parser.parse_args(argv)

    digest = sha256(args.dmg)
    manifest = intel.load_json(args.manifest)
    report = intel.load_json(args.validation) if args.validation and args.validation.is_file() else {}
    print(f"dmg      {args.dmg}  {args.dmg.stat().st_size / 1024 ** 3:.2f} GB\nsha256   {digest}")
    reasons: list[str] = []
    if manifest.get("dmg_sha256") != digest:
        reasons.append("the DMG is not the file the Mac build recorded (SHA-256 differs from release_manifest.json) - the transfer corrupted it or it is another build")
    sidecar = args.dmg.with_name(args.dmg.name + ".sha256")
    if sidecar.is_file() and sidecar.read_text(encoding="utf-8").split()[0] != digest:
        reasons.append(f"{sidecar.name} does not match the file")
    verdict, why = intel.readiness(report, digest, manifest)
    reasons += why
    print(f"verdict  {verdict}")
    for r in reasons:
        print(f"  - {r}")
    if verdict != "READY TO UPLOAD" or reasons:
        if verdict == "READY TO UPLOAD":
            verdict = "NEEDS WORK"
            print("verdict  NEEDS WORK")
        print("Nothing was copied to the Desktop.")
        return 1

    TARGET.mkdir(parents=True, exist_ok=True)
    for folder in (TARGET, args.desktop):
        out = folder / intel.DMG_NAME
        shutil.copyfile(args.dmg, out)
        if sha256(out) != digest:
            print(f"the copy {out} differs from the tested DMG")
            return 1
        (folder / f"{intel.DMG_NAME}.sha256").write_text(f"{digest}  {intel.DMG_NAME}\n", encoding="utf-8")
        print(f"copied   {out}  (SHA-256 verified)")
    shutil.copyfile(args.manifest, TARGET / "release_manifest.json")
    shutil.copyfile(args.validation, TARGET / "validation_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
