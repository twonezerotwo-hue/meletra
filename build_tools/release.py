"""The release folder of a version: README.txt, the all-platforms package, and a plain report of what exists.

    release/v4.1.0/Windows/MELETRA-Setup-Windows-x64.exe
                  Windows/MELETRA-Portable-Windows-x64.zip
                  macOS/MELETRA-macOS-AppleSilicon.dmg      (only when built on a Mac)
                  macOS/MELETRA.app
                  README.txt
                  MELETRA-v4.1.0-AllPlatforms.zip           (optional)

Nothing here pretends: an artifact that does not exist is listed as "not built", never described as available.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import macos as mac
from .layout import PRODUCT_ID, base_version, release_dir

WINDOWS_INSTALLER = f"{PRODUCT_ID}-Setup-Windows-x64.exe"
WINDOWS_PORTABLE = f"{PRODUCT_ID}-Portable-Windows-x64.zip"
WINDOWS_FULL_OFFLINE = f"{PRODUCT_ID}-FullOffline-x64.zip"


@dataclass
class Artifact:
    name: str
    platform: str
    path: Path

    @property
    def exists(self) -> bool:
        return self.path.exists()


def artifacts(version: str, root: Path) -> list[Artifact]:
    win, macos = release_dir(version, "Windows", root), release_dir(version, "macOS", root)
    return [
        Artifact("Windows installer", "Windows", win / WINDOWS_INSTALLER),
        Artifact("Windows portable", "Windows", win / WINDOWS_PORTABLE),
        Artifact("Windows full offline", "Windows", win / WINDOWS_FULL_OFFLINE),
        Artifact("macOS app (Apple Silicon)", "macOS", macos / mac.APP_BUNDLE),
        Artifact("macOS DMG (Apple Silicon)", "macOS", macos / mac.DMG_ARM),
        Artifact("macOS DMG (Intel)", "macOS", macos / mac.DMG_INTEL),
    ]


def readme_text(version: str, root: Path, signing: Optional[dict[str, str]] = None) -> str:
    items = artifacts(version, root)
    lines = [f"MELETRA {version}", "=" * (len(version) + 28), "",
             "Turn recordings, PDFs and pictures into text - and into speech or study material (summaries, flashcards, questions).",
             "Everything runs on your computer. Nothing is uploaded unless you choose a cloud option yourself.", "",
             "WHAT IS IN THIS RELEASE", "-----------------------"]
    for item in items:
        lines.append(f"  [{'built' if item.exists else 'not built'}] {item.name}: {item.path.name}")
    lines += ["", "WINDOWS 10 / 11 (64-bit)",
              f"  Installer: run {WINDOWS_INSTALLER}. Portable: unzip {WINDOWS_PORTABLE} anywhere and start "
              f"{PRODUCT_ID}.exe (keeps its data next to itself).",
              "  The first start downloads the speech model once (about 3 GB); after that it works without internet.",
              "  NVIDIA GPUs are used automatically; without one the CPU is used (slower).", "",
              "MACOS (Apple Silicon first, Intel second)"]
    if any(i.platform == "macOS" and i.exists for i in items):
        lines += [f"  Open {mac.DMG_ARM}, drag {mac.APP_BUNDLE} into Applications, start it, press DOWNLOAD RECOMMENDED MODEL."]
        if signing:
            lines.append(f"  Signed: {signing.get('signed', 'NO')}   Notarized: {signing.get('notarized', 'NO')}. {signing.get('note', '')}")
    else:
        lines += ["  The macOS app is NOT included in this release: it can only be built on a Mac. On a Mac run ./build_macos.sh",
                  "  (see BUILDING.md); it produces the DMG and says whether it is signed and notarized."]
    lines += ["", "FIRST START", "  Drop a lecture into the window (optionally add the lecture PDF) and press TRANSCRIBE.",
              "  When it is ready: Read Transcript, or Study Lecture (Summary, Key Concepts, Flashcards, Questions, Study Mode).", ""]
    return "\n".join(lines)


def write_readme(version: str, root: Path, signing: Optional[dict[str, str]] = None) -> Path:
    target = root / "release" / f"v{base_version(version)}" / "README.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(readme_text(version, root, signing), encoding="utf-8")
    return target


MAC_NOT_BUILT = """macOS: NOT BUILT in this release
================================

The macOS app and its DMG can only be built on a Mac, and none was available when this release was made. Nothing in this folder is a
macOS program. The macOS build is prepared (build scripts, platform layer, speech-engine backend, tests with a fake Mac) but it has never
been run on a Mac.

To build it on a Mac (Apple Silicon recommended):

    ./build_macos.sh

The result appears in this folder: MELETRA-macOS-AppleSilicon.dmg (Intel: ...-Intel.dmg), the .app, SHA256SUMS.txt and
build_report.json, which states whether the build is signed and notarized. Without an Apple Developer ID certificate it is UNSIGNED and NOT
NOTARIZED: macOS then asks for confirmation the first time (right-click the app, choose Open, then Open).
"""


def write_mac_placeholder(version: str, root: Path) -> Optional[Path]:
    """``macOS/README.txt`` saying plainly that nothing was built - only when no macOS artifact exists (never next to a real build)."""
    macos = release_dir(version, "macOS", root)
    if any(a.exists for a in artifacts(version, root) if a.platform == "macOS"):
        return None
    macos.mkdir(parents=True, exist_ok=True)
    target = macos / "README.txt"
    target.write_text(MAC_NOT_BUILT, encoding="utf-8")
    return target


def all_platforms_zip(version: str, root: Path) -> Path:
    """One zip with everything that exists, in the release layout. Skips what is missing (and the README says so)."""
    base = root / "release" / f"v{base_version(version)}"
    target = base / all_platforms_name(version)
    readme = write_readme(version, root)
    placeholder = write_mac_placeholder(version, root)
    target.unlink(missing_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_STORED, allowZip64=True) as archive:
        archive.write(readme, "README.txt")
        if placeholder is not None:
            archive.write(placeholder, "macOS/README.txt")
        for item in artifacts(version, root):
            if not item.exists:
                continue
            if item.path.is_dir():
                for child in sorted(item.path.rglob("*")):
                    if child.is_file():
                        archive.write(child, f"{item.platform}/{item.path.name}/{child.relative_to(item.path).as_posix()}")
            else:
                archive.write(item.path, f"{item.platform}/{item.path.name}")
    return target


def all_platforms_name(version: str) -> str:                        
    return mac.ALL_PLATFORMS_ZIP.format(app=PRODUCT_ID, version=base_version(version))
