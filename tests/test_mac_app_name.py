"""The Mac app is called MELETRA.app (4.1 and newer). The names of earlier releases, WhisperMedical.app and "Medical Lecture Transcriber.app", are still recognised so an old
image or an old install keeps working.

This is checked statically and with the lookup snippet of the installer run in bash; a build and an install on a real Mac are NOT possible on this machine (see docs/v4_1/RELEASE_NOTES.md).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from build_tools import macos, macos_intel

ROOT = Path(__file__).resolve().parent.parent
NEW = "MELETRA.app"
LEGACY = ("WhisperMedical.app", "Medical Lecture Transcriber.app")
#: files that may name an old app: the compatibility code itself, its tests, the notes about the renames and the history of the project
MAY_MENTION_OLD = {"build_tools/macos.py", "build_tools/macos_intel.py", "build_tools/installer.iss", "install/install-macos.sh", "scripts/validate_macos_intel.py",
                   "docs/MACOS_INTEL_RELEASE.md", "docs/v4_1/RELEASE_NOTES.md", "docs/v4_1/MELETRA_RELEASE.md", "docs/v4_1/FINAL_POLISH_REPORT.md",
                   "tests/test_mac_app_name.py", "tests/test_macos_intel_tools.py", "tests/test_meletra_brand.py", "DEVELOPMENT_STATUS.md", "DEVELOPMENT_STATUS_V3.md"}


def test_the_bundle_folder_is_meletra_app_and_the_old_names_are_kept_for_lookup():
    assert macos.APP_NAME == "MELETRA" and macos.LEGACY_APP_NAMES == ("WhisperMedical", "Medical Lecture Transcriber") and macos.DISPLAY_NAME == "MELETRA"
    assert macos_intel.APP_FOLDER == NEW and macos_intel.LEGACY_APP_FOLDERS == LEGACY and macos_intel.VOLUME_NAME == "MELETRA"
    assert macos.APP_BUNDLE == NEW and macos_intel.DMG_NAME == "MELETRA-Intel-Mac-Offline.dmg"
    plist = macos.info_plist("4.1.0", "x86_64")
    assert plist["CFBundleName"] == plist["CFBundleDisplayName"] == plist["CFBundleExecutable"] == "MELETRA"
    assert plist["CFBundleIdentifier"] == "com.medicallecturetranscriber.app"                  # unchanged: it is how an old copy is recognised as this program


def test_the_ci_workflows_and_scripts_use_the_new_bundle_name():
    for relative in (".github/workflows/build-macos-intel.yml", ".github/workflows/test-install-macos.yml"):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert NEW in text and not any(old in text for old in LEGACY), relative
    validate = (ROOT / "scripts" / "validate_macos_intel.py").read_text(encoding="utf-8")
    assert "intel.APP_FOLDER" in validate and "intel.LEGACY_APP_FOLDER" in validate


def test_no_other_file_still_calls_the_mac_app_by_an_old_name():
    stale = []
    skip = {".venv", "dist", "build", "release", "logs", ".git", ".pytest_tmp", "node_modules", "__pycache__", ".pytest_cache", "build_cache", "vendor"}
    for folder, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in skip and not d.startswith(".venv")]
        for name in files:
            path = Path(folder) / name
            if path.suffix not in {".py", ".sh", ".yml", ".yaml", ".md", ".txt", ".bat", ".ps1", ".iss", ".spec", ".html"}:
                continue
            relative = path.relative_to(ROOT).as_posix()
            if relative in MAY_MENTION_OLD:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if any(old in text for old in LEGACY):
                stale.append(relative)
    assert stale == [], stale


def lookup_snippet() -> str:
    """The lines of install-macos.sh that choose the app inside the mounted image (everything else needs a Mac: hdiutil, ditto, osascript)."""
    script = (ROOT / "install" / "install-macos.sh").read_text(encoding="utf-8")
    header = re.search(r'^APP_NAME=.*\nLEGACY_APP_NAMES=.*\n', script, re.M).group(0)
    choice = re.search(r'^if \[ ! -d "\$MOUNT/\$APP_NAME" \]; then.*?\nfi\n', script, re.M | re.S).group(0)
    return header + choice + 'echo "$APP_NAME"\n'


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
@pytest.mark.parametrize("present, expected", [([NEW], NEW), ([LEGACY[0]], LEGACY[0]), ([LEGACY[1]], LEGACY[1]), ([LEGACY[1], NEW], NEW),
                                               ([LEGACY[1], LEGACY[0]], LEGACY[0]), ([], NEW)])
def test_the_installer_prefers_the_new_name_and_falls_back_to_the_earlier_ones(tmp_path, present, expected):
    mount = tmp_path / "mount"
    mount.mkdir()
    for name in present:
        (mount / name).mkdir()
    done = subprocess.run(["bash", "-c", f'MOUNT="{mount.as_posix()}"\n{lookup_snippet()}'], capture_output=True, text=True, timeout=30)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == expected


def test_the_installer_replaces_an_old_copy_only_when_it_is_this_program():
    script = (ROOT / "install" / "install-macos.sh").read_text(encoding="utf-8")
    assert 'BUNDLE_ID="com.medicallecturetranscriber.app"' in script and "CFBundleIdentifier" in script            # an unrelated app with an old name is never removed
    assert 'rm -rf "$old"' in script and "OLD_COPIES" in script
    assert "MELETRA.app" in script
