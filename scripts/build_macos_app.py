"""Build the macOS app on a Mac: PyInstaller -> .app -> (sign) -> DMG -> (notarize) -> checksums -> report.

    python scripts/build_macos_app.py                 build for this Mac's architecture (arm64 = Apple Silicon, x86_64 = Intel)
    python scripts/build_macos_app.py --dry-run       print every command, change nothing (also works on Windows / Linux)

Environment (all optional):
    MLT_SIGN_IDENTITY     "Developer ID Application: Name (TEAMID)" - sign the app and the DMG
    MLT_NOTARY_PROFILE    name of a stored `xcrun notarytool store-credentials` profile - notarize the DMG (needs the identity)

Without a certificate the result is UNSIGNED and NOT NOTARIZED, and the report (release/v<version>/macOS/build_report.json) says exactly
that. The same steps are listed by ``build_plan`` so they are tested on any machine; this script only runs them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from build_tools import macos as mac                                  # noqa: E402
from build_tools.layout import dist_parent, release_dir               # noqa: E402
from src import __version__                                           # noqa: E402


@dataclass
class Step:
    name: str
    argv: list[str]
    cwd: Optional[Path] = None
    env: dict[str, str] = field(default_factory=dict)
    optional: bool = False                                            # a failure is reported but does not stop the build


def paths(version: str = __version__, root: Path = ROOT) -> dict[str, Path]:
    dist = dist_parent(version, root) / "macos"
    release = release_dir(version, "macOS", root)
    return {"dist": dist, "app": dist / mac.APP_BUNDLE, "release": release, "assets": root / "build" / "assets",
            "work": root / "build" / "macos-work", "spec": root / "build_tools" / "MELETRA-macos.spec"}


def build_plan(arch: str, version: str = __version__, root: Path = ROOT, identity: Optional[str] = None,
               notary_profile: Optional[str] = None, python: str = sys.executable) -> list[Step]:
    arch = mac.normalize_arch(arch)
    p = paths(version, root)
    dmg = p["release"] / mac.dmg_name(arch)
    steps = [
        Step("assets", [python, "-c", "from build_tools.icns import write_icns; from pathlib import Path; "
                                      "write_icns(Path('build/assets/app.icns')); "
                                      "from build_tools.macos import entitlements_bytes; "
                                      "Path('build/assets/entitlements.plist').write_bytes(entitlements_bytes())"], cwd=root),
        Step("pyinstaller", [python, "-m", "PyInstaller", "--noconfirm", "--clean", "--distpath", str(p["dist"]), "--workpath",
                             str(p["work"]), str(p["spec"])], cwd=root, env={"MLT_MAC_ARCH": arch}),
    ]
    if identity:
        steps += [Step(f"codesign {i + 1}", argv) for i, argv in enumerate(mac.codesign_commands(p["app"], identity,
                                                                                                 p["assets"] / "entitlements.plist"))]
    steps += [Step(f"dmg {i + 1}", argv) for i, argv in enumerate(mac.dmg_commands(p["app"], dmg, staging=p["work"] / "dmg"))]
    if identity:
        steps.append(Step("codesign dmg", ["codesign", "--force", "--timestamp", "--sign", identity, str(dmg)]))
    if identity and notary_profile:
        steps += [Step(f"notarize {i + 1}", argv) for i, argv in enumerate(mac.notarize_commands(dmg, notary_profile))]
    return steps


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_report(arch: str, version: str, identity: Optional[str], notary_profile: Optional[str], root: Path = ROOT) -> Path:
    p = paths(version, root)
    dmg = p["release"] / mac.dmg_name(arch)
    state = mac.signing_state(identity, notary_profile)
    report = {"version": version, "architecture": arch, "app": str(p["app"]), "dmg": str(dmg) if dmg.is_file() else None,
              "dmg_sha256": sha256(dmg) if dmg.is_file() else None, "macos_built_on": platform.mac_ver()[0], **state}
    p["release"].mkdir(parents=True, exist_ok=True)
    target = p["release"] / "build_report.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if dmg.is_file():
        (p["release"] / "SHA256SUMS.txt").write_text(f"{report['dmg_sha256']}  {dmg.name}\n", encoding="utf-8")
    return target


def run(steps: list[Step]) -> int:
    for step in steps:
        print(f"[macos] {step.name}: {' '.join(step.argv)}", flush=True)
        proc = subprocess.run(step.argv, cwd=step.cwd, env={**os.environ, **step.env}, check=False)
        if proc.returncode != 0:
            print(f"[macos] {step.name} FAILED (exit {proc.returncode})", file=sys.stderr)
            if not step.optional:
                return proc.returncode
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", default=platform.machine() or "arm64", help="arm64 (Apple Silicon) or x86_64 (Intel); default: this computer")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    identity, notary = os.environ.get("MLT_SIGN_IDENTITY") or None, os.environ.get("MLT_NOTARY_PROFILE") or None
    steps = build_plan(args.arch, identity=identity, notary_profile=notary)
    if args.dry_run:
        for step in steps:
            print(f"{step.name}: {' '.join(step.argv)}")
        print(json.dumps(mac.signing_state(identity, notary)))
        return 0
    if sys.platform != "darwin":
        print("This build must run on a Mac (use --dry-run to see the steps).", file=sys.stderr)
        return 2
    arch = mac.normalize_arch(args.arch)                              # unknown architecture: fail before building
    code = run(steps)
    report = write_report(arch, __version__, identity, notary)
    print(f"[macos] report: {report}")
    print(json.dumps(mac.signing_state(identity, notary)))
    return code


if __name__ == "__main__":
    sys.exit(main())
