"""Everything the macOS build decides, as plain functions: names, Info.plist, entitlements, and the exact commands for DMG, signing and
notarization. No macOS tool is run here, so all of it is tested on any machine; ``scripts/build_macos_app.py`` runs the commands on a Mac.

Signing and notarization need an Apple Developer ID certificate and an App Store Connect API key (or an Apple ID app password). The
repository has neither: without them the build is *architecture-ready* - it produces an UNSIGNED, NOT NOTARIZED app and DMG, and says so
(see ``signing_state``). On the student's Mac, Gatekeeper then needs "right-click -> Open" the first time.
"""

from __future__ import annotations

import plistlib
from pathlib import Path
from typing import Optional

from .layout import PRODUCT_ID, base_version

APP_NAME = "MELETRA"                                       # the .app folder name (MELETRA.app); 4.1 and newer
LEGACY_APP_NAMES = ("WhisperMedical", "Medical Lecture Transcriber")   # the folder names of earlier releases: the installers still find and replace them
LEGACY_APP_NAME = LEGACY_APP_NAMES[-1]
DISPLAY_NAME = "MELETRA"                                   # what the bundle calls itself
BUNDLE_ID = "com.medicallecturetranscriber.app"
APP_BUNDLE = f"{PRODUCT_ID}.app"
MIN_MACOS = "12.0"                                       # PySide6 6.x and mlx need a recent macOS; Monterey is the floor

DMG_ARM = f"{PRODUCT_ID}-macOS-AppleSilicon.dmg"
DMG_INTEL = f"{PRODUCT_ID}-macOS-Intel.dmg"
ALL_PLATFORMS_ZIP = "{app}-v{version}-AllPlatforms.zip"


def normalize_arch(arch: str) -> str:
    """``arm64`` | ``x86_64`` from whatever ``uname -m`` / ``platform.machine()`` says (``AMD64``, ``aarch64``, ``Intel`` ...)."""
    value = (arch or "").strip().lower()
    if value in ("arm64", "aarch64"):
        return "arm64"
    if value in ("x86_64", "amd64", "intel", "i386"):
        return "x86_64"
    raise ValueError(f"unknown macOS architecture {arch!r}")


def dmg_name(arch: str) -> str:
    """``arm64`` -> the Apple Silicon DMG (primary); ``x86_64`` -> the Intel DMG (secondary)."""
    return DMG_ARM if normalize_arch(arch) == "arm64" else DMG_INTEL


def info_plist(version: str, arch: str = "arm64") -> dict:
    """The Info.plist of the .app (PyInstaller's BUNDLE merges this into its default one)."""
    return {
        "CFBundleName": DISPLAY_NAME,
        "CFBundleDisplayName": DISPLAY_NAME,
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleVersion": base_version(version),
        "CFBundleShortVersionString": version,
        "CFBundleExecutable": PRODUCT_ID,
        "CFBundlePackageType": "APPL",
        "CFBundleIconFile": "app.icns",
        "LSMinimumSystemVersion": MIN_MACOS,
        "LSApplicationCategoryType": "public.app-category.education",
        "NSHighResolutionCapable": True,
        "NSHumanReadableCopyright": "MELETRA - Medical Learning & Transcription.",
        "NSMicrophoneUsageDescription": "MELETRA does not record; it only reads audio files you choose.",
        # audio / video / PDF files can be opened with "Open With" and by dropping them on the Dock icon
        "CFBundleDocumentTypes": [
            {"CFBundleTypeName": "Lecture recording", "CFBundleTypeRole": "Viewer", "LSHandlerRank": "Alternate",
             "LSItemContentTypes": ["public.audio", "public.movie"]},
            {"CFBundleTypeName": "Lecture slides", "CFBundleTypeRole": "Viewer", "LSHandlerRank": "Alternate",
             "LSItemContentTypes": ["com.adobe.pdf"]},
            {"CFBundleTypeName": "Picture or text", "CFBundleTypeRole": "Viewer", "LSHandlerRank": "Alternate",
             "LSItemContentTypes": ["public.image", "public.plain-text", "net.daringfireball.markdown"]},
        ],
        "LSArchitecturePriority": [arch] if arch in ("arm64", "x86_64") else ["arm64", "x86_64"],
    }


def info_plist_bytes(version: str, arch: str = "arm64") -> bytes:
    return plistlib.dumps(info_plist(version, arch))


def entitlements() -> dict:
    """Hardened-runtime entitlements the Python / Qt / Metal stack needs. Nothing network-server related; no camera / microphone."""
    return {
        "com.apple.security.cs.allow-unsigned-executable-memory": True,       # Python extension modules, numpy
        "com.apple.security.cs.allow-jit": True,                              # MLX / Metal shader compilation
        "com.apple.security.cs.disable-library-validation": True,            # bundled third-party dylibs (PySide6, ctranslate2, onnxruntime)
        "com.apple.security.cs.allow-dyld-environment-variables": False,
    }


def entitlements_bytes() -> bytes:
    return plistlib.dumps(entitlements())


# ---------------------------------------------------------------------------------------------------------------- commands
def codesign_commands(app: Path, identity: str, entitlements_file: Path) -> list[list[str]]:
    """Sign the bundle from the inside out: every Mach-O file first is done by ``--deep``; then the app itself is verified."""
    sign = ["codesign", "--force", "--deep", "--options", "runtime", "--timestamp", "--entitlements", str(entitlements_file),
            "--sign", identity, str(app)]
    return [sign, ["codesign", "--verify", "--deep", "--strict", "--verbose=2", str(app)],
            ["spctl", "--assess", "--type", "execute", "--verbose=2", str(app)]]


def dmg_commands(app: Path, dmg: Path, volume: str = DISPLAY_NAME, staging: Optional[Path] = None) -> list[list[str]]:
    """A compressed DMG with the app and an /Applications shortcut (drag-and-drop install), built from a staging folder."""
    staging = staging or dmg.parent / "dmg-staging"
    return [["rm", "-rf", str(staging)], ["mkdir", "-p", str(staging)], ["cp", "-R", str(app), str(staging / app.name)],
            ["ln", "-s", "/Applications", str(staging / "Applications")],
            ["hdiutil", "create", "-volname", volume, "-srcfolder", str(staging), "-ov", "-format", "UDZO", str(dmg)],
            ["rm", "-rf", str(staging)]]


def notarize_commands(dmg: Path, keychain_profile: str) -> list[list[str]]:
    """Submit the DMG to Apple's notary service and staple the ticket. ``keychain_profile`` is created once with
    ``xcrun notarytool store-credentials``."""
    return [["xcrun", "notarytool", "submit", str(dmg), "--keychain-profile", keychain_profile, "--wait"],
            ["xcrun", "stapler", "staple", str(dmg)], ["xcrun", "stapler", "validate", str(dmg)]]


def signing_state(identity: Optional[str], notary_profile: Optional[str]) -> dict[str, str]:
    """What the release report must say. Honest by construction: with no identity it says NO, whatever else is true."""
    signed = bool(identity)
    notarized = signed and bool(notary_profile)
    note = ("Signed with a Developer ID certificate" + (" and notarized by Apple." if notarized else ", NOT notarized."))\
        if signed else ("No Developer ID certificate was available: the app and the DMG are UNSIGNED and NOT NOTARIZED. "
                        "On first launch macOS Gatekeeper asks for confirmation: right-click the app, choose Open, then Open.")
    return {"signed": "YES" if signed else "NO", "notarized": "YES" if notarized else "NO", "note": note}
