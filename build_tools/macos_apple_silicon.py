"""Everything the Apple-Silicon-Mac OFFLINE release decides, as plain functions: names, the marker file inside the app, the notices, the checks of a
built bundle, the readiness rule. Nothing here runs a macOS tool, so it is tested on any computer; ``scripts/build_macos_apple_silicon_offline.py``
(on an Apple Silicon Mac) and ``scripts/finalize_mac_dmg.py`` (on Windows) use it.

The rule it encodes: a DMG is only "ready" when the build host was a real Apple Silicon Mac, the validation ran on real Apple Silicon hardware with the
network off, every check passed, the checksum of the tested file is the checksum of the file that is handed out - and the GPL source
offer exists. Missing evidence is never rounded up.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from .layout import PRODUCT_ID

from src import __version__ as _VERSION

RELEASE_LABEL = f"{_VERSION}-mac-apple-silicon"
DMG_NAME = f"{PRODUCT_ID}-Apple-Silicon-Mac-Offline.dmg"
VOLUME_NAME = "MELETRA"                             # what the mounted disk image is called
APP_FOLDER = "MELETRA.app"                          # what the student sees in Applications (PyInstaller builds MELETRA.app)
LEGACY_APP_FOLDERS = ("WhisperMedical.app", "Medical Lecture Transcriber.app")   # the names of earlier releases: installers and validators still recognise an installed copy under them
LEGACY_APP_FOLDER = LEGACY_APP_FOLDERS[-1]
OFFLINE_MARKER = "offline_release.json"
MODEL_KEY = "whisper-large-v3-mlx"
MIN_MACOS = "14.0"                      # the public GitHub arm64 runner and the pinned MLX/PySide wheels target macOS 14+
MIN_FREE_GB = 25                        # model (3 GB) + venv + PyInstaller work folder + app + DMG
RELEASE_FOLDER = "macOS-Apple-Silicon"          # release/v<version>/macOS-Apple-Silicon/

#: libraries the validation of a real Apple Silicon Mac must find in the bundle (checked by the build host after PyInstaller)
REQUIRED_BUNDLE_DIRS = ("medical_vocab", "medical_terms", "config", "data", "models/whisper-large-v3-mlx")      # data: the public medical knowledge pack
REQUIRED_RESOURCE_FILES = ("THIRD_PARTY_LICENSES.txt", "MODEL_LICENSES.txt", OFFLINE_MARKER)


def marker(version: str, model: dict[str, Any], commit: str) -> dict[str, Any]:
    """``Contents/Resources/offline_release.json``: what this build promises to carry (the self-test holds it to that)."""
    return {"release": RELEASE_LABEL, "version": version, "architecture": "arm64", "target": "macOS Apple Silicon", "min_macos": MIN_MACOS,
            "model": MODEL_KEY, "model_files": model, "source_commit": commit, "license_files": ["THIRD_PARTY_LICENSES.txt", "MODEL_LICENSES.txt"]}


# ------------------------------------------------------------------------------------------------------------ notices
GPL_COMPONENTS = ("libx264 (GPL-2.0-or-later)", "libx265 (GPL-2.0-or-later)")


def media_section(v: dict[str, str], contact_line: str, lgpl: Optional[dict] = None) -> str:
    """The paragraph about the FFmpeg libraries: the GPL warning for the stock wheels, or the LGPL-only statement for an app whose libraries were replaced (build_tools/lgpl_media_macos.py)."""
    if lgpl:
        builds = "\n".join(f"    FFmpeg {b['version']} ({b['url']}, SHA-256 {b['sha256']}) for {b['for']}" for b in lgpl.values())
        return (f"FFmpeg (LGPL-2.1+) - this macOS package carries LGPL-only FFmpeg libraries\n"
                f"  The FFmpeg libraries of PyAV {v.get('av', '')} and OpenCV {v.get('opencv-python-headless', '')} were replaced by builds of the same major versions made from the unmodified\n"
                f"  source releases below, configured with --disable-gpl --disable-nonfree --disable-autodetect: NO libx264, NO libx265, no external codec library. The program only DECODES\n"
                f"  audio and video with FFmpeg's own decoders. The libraries are separate files in the app (.dylibs folders) and can be replaced by your own build.\n{builds}\n"
                f"  Build script: scripts/build_macos_ffmpeg_lgpl.sh of the project; licence text: COPYING.LGPLv2.1 inside the source tarballs.")
    return (f"GPL - READ THIS (this macOS package differs from the Windows package, which carries an LGPL-only FFmpeg)\n"
            f"  The FFmpeg libraries inside PyAV {v.get('av', '')} and OpenCV {v.get('opencv-python-headless', '')} (libavcodec, libavformat, libavutil, libswscale, libswresample) are built\n"
            f"  with the video encoders {' and '.join(GPL_COMPONENTS)} (the OpenCV build also with --enable-gpl). As built they must be treated as GPL, not as LGPL-only. This program only\n"
            f"  DECODES audio and video (it uses FFmpeg's own decoders), but the two GPL libraries are present in the app's libraries folder and are loaded by the FFmpeg libraries.\n"
            f"  Whether this satisfies your distribution needs is a question for a lawyer: LEGAL REVIEW REQUIRED before this package is passed on. The corresponding source code:\n"
            f"    x264   https://code.videolan.org/videolan/x264\n"
            f"    x265   https://bitbucket.org/multicoreware/x265_git\n"
            f"    FFmpeg https://ffmpeg.org/download.html   (build recipes of the wheels: https://github.com/PyAV-Org/pyav-ffmpeg and\n"
            f"                                               https://github.com/opencv/opencv-python)\n"
            f"  {contact_line}")


def third_party_licenses(contact: str = "", versions: Optional[dict[str, str]] = None, media_lgpl: Optional[dict] = None) -> str:
    v = versions or {}
    offer = (f"Contact for source-code requests: {contact}" if contact else
             "Contact for source-code requests: NOT SET - the person who distributes this disk image must add it (build variable MLT_SOURCE_CONTACT).")
    return f"""MELETRA {RELEASE_LABEL} - third-party components
================================================

The program runs on your Mac only; nothing is uploaded. It carries the components below. Their licence texts are in the folder
"licenses" next to this file (Contents/Resources/licenses/<package>/).

MIT / BSD / Apache-2.0 / similar
  faster-whisper {v.get('faster-whisper', '')} (MIT) - CTranslate2 {v.get('ctranslate2', '')} (MIT) - ONNX Runtime {v.get('onnxruntime', '')} (MIT)
  Silero VAD model inside faster-whisper (MIT) - tokenizers (Apache-2.0) - NumPy (BSD-3-Clause and bundled MIT / Zlib / CC0 parts)
  PyYAML (MIT) - RapidFuzz (MIT) - pypdf (BSD-3-Clause) - Pillow (MIT-CMU) - soundfile (BSD-3-Clause) - pyclipper (MIT) - shapely (BSD-3-Clause)
  keyring (MIT) - huggingface_hub (Apache-2.0, library only: this release never contacts the network for models)
  RapidOCR {v.get('rapidocr-onnxruntime', '')} (Apache-2.0) with its PP-OCR text detection / recognition models (Apache-2.0), carried inside the app
  Python 3.11 (PSF licence) - PyInstaller bootloader (GPL-2.0-or-later with the exception that allows distributing the built program)

LGPL (the libraries are separate files in the app and can be replaced by your own build)
  Qt 6 / PySide6 / shiboken6 {v.get('PySide6-Essentials', '')} (LGPL-3.0) - libsndfile (LGPL-2.1+) - GEOS (LGPL-2.1+, inside shapely)
  libmp3lame (LGPL)

{media_section(v, offer, media_lgpl)}

Whisper large-v3 (speech model) is listed in MODEL_LICENSES.txt.
"""


def model_licenses(model_info: dict[str, Any]) -> str:
    files = "\n".join(f"    {name}  {spec['size']:,} bytes  sha256 {spec['sha256']}" for name, spec in model_info.items())
    return f"""Models carried inside MELETRA {RELEASE_LABEL}
==========================================

1. Whisper large-v3 - speech recognition
   Weights by OpenAI (https://github.com/openai/whisper), MIT licence: "Permission is hereby granted, free of charge, ... to deal in the
   Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies ..." - redistribution is allowed provided the copyright and permission notice stay with it.
   Converted for Apple MLX by the MLX Community
   (https://huggingface.co/mlx-community/whisper-large-v3-mlx), MIT licence.
   Files (also checked on every start by size, and completely by the self-test):
{files}

2. Silero VAD (voice activity detection), file silero_vad_v6.onnx inside faster-whisper - MIT licence (https://github.com/snakers4/silero-vad).

3. RapidOCR / PP-OCR text detection, direction and recognition models (ONNX), inside the rapidocr_onnxruntime package - Apache-2.0
   (https://github.com/RapidAI/RapidOCR, models from PaddleOCR https://github.com/PaddlePaddle/PaddleOCR).

4. Medical vocabulary (medical_vocab, medical_terms): written for this program; sources and licences per domain are listed in
   medical_vocab/SOURCES.json.

Nothing else is downloaded: the program has no model downloader in this release.
"""


# ------------------------------------------------------------------------------------------------------------ bundle checks
_SYSTEM_PREFIXES = ("/usr/lib/", "/System/Library/", "/Library/Apple/")
_BAD_MARKERS = ("/Users/", "/opt/homebrew", "/usr/local/", "/Library/Frameworks/Python", "/private/var/folders", "/Volumes/")


def audit_otool(path: str, output: str) -> list[str]:
    """Problems in the ``otool -L`` output of one Mach-O file: a dependency on a developer-machine location, a Homebrew library or a
    path outside the app that is not a macOS system library. ``@rpath`` / ``@loader_path`` / ``@executable_path`` references are the
    app's own and are fine (their targets are verified by the build host)."""
    problems = []
    for line in output.splitlines()[1:]:
        m = re.match(r"\s+(\S.*?) \(compatibility version", line)
        if not m:
            continue
        dep = m.group(1)
        if dep.startswith(("@rpath/", "@loader_path/", "@executable_path/")) or dep.startswith(_SYSTEM_PREFIXES):
            continue
        problems.append(f"{path}: depends on {dep}")
    return problems


def audit_rpaths(path: str, output: str) -> list[str]:
    """``otool -l`` output: LC_RPATH entries that point at a developer machine."""
    return [f"{path}: rpath {p}" for p in re.findall(r"cmd LC_RPATH\n(?:.*\n)*?\s+path (\S.*?) \(offset", output)
            if any(marker in p for marker in _BAD_MARKERS)]


def minos_of(output: str) -> Optional[tuple[int, int]]:
    """The macOS version a Mach-O file asks for, from ``otool -l`` (LC_BUILD_VERSION ``minos`` with platform 1, or LC_VERSION_MIN_MACOSX)."""
    found = []
    for platform_id, minos in re.findall(r"cmd LC_BUILD_VERSION\s+cmdsize \d+\s+platform (\d+)\s+minos ([\d.]+)", output):
        if platform_id == "1":
            found.append(minos)
    found += re.findall(r"cmd LC_VERSION_MIN_MACOSX\s+cmdsize \d+\s+version ([\d.]+)", output)
    versions = [tuple(int(x) for x in (v.split(".") + ["0"])[:2]) for v in found]
    return max(versions) if versions else None


def split_name(dmg_name: str, index: int) -> str:
    return f"{dmg_name}.part{index:03d}"


def parse_peak_memory(stderr: str) -> Optional[int]:
    """Bytes from ``/usr/bin/time -l`` ("  1234567  maximum resident set size" - bytes on macOS)."""
    m = re.search(r"^\s*(\d+)\s+maximum resident set size", stderr, re.MULTILINE)
    return int(m.group(1)) if m else None


def realtime_factor(processing_seconds: float, audio_seconds: float) -> Optional[float]:
    return round(processing_seconds / audio_seconds, 3) if audio_seconds > 0 else None


# ------------------------------------------------------------------------------------------------------------ readiness
VALIDATION_KEYS = ("dmg_mount", "drag_to_applications", "dmg_eject", "launch_from_applications", "offline_launch", "whisper_large_v3",
                   "no_model_download", "audio_stt", "video_stt", "video_progress", "queue", "medical_expert", "pdf", "ocr", "tts",
                   "frontend", "study_workspace", "app_restart", "no_dev_environment", "gatekeeper_downloaded_file")


def readiness(report: dict[str, Any], dmg_sha256: str, manifest: dict[str, Any]) -> tuple[str, list[str]]:
    """``("READY TO UPLOAD" | "NEEDS WORK" | "REAL APPLE SILICON MAC REQUIRED", reasons)`` - the only place that may say READY."""
    if not report or not report.get("real_hardware") or report.get("machine") != "arm64":
        return "REAL APPLE SILICON MAC REQUIRED", ["no validation report from a real Apple Silicon (arm64) Mac"]
    reasons: list[str] = []
    if not report.get("network_disconnected"):
        reasons.append("the offline test did not run with the network disconnected")
    if report.get("dmg_sha256") != dmg_sha256:
        reasons.append("the tested DMG is not the DMG that is being handed out (checksums differ)")
    for key in VALIDATION_KEYS:
        status = (report.get("checks") or {}).get(key)
        if status != "PASS":
            reasons.append(f"{key}: {status or 'not run'}")
    if manifest.get("gpl_source_offer") not in ("SET", "NOT NEEDED") and not (manifest.get("media_libraries") or {}).get("lgpl_only"):
        reasons.append("the GPL source-offer contact is not set (libx264 / libx265 are in the bundled PyAV / OpenCV libraries)")
    if not manifest.get("study_quality_included"):
        reasons.append("Study Quality: the validated study-facts engine is not in the build - check the manifest")
    return ("READY TO UPLOAD", []) if not reasons else ("NEEDS WORK", reasons)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------------------------ CI-tested builds
CI_KEYS = ("network_blocked_for_app_runs", "app_is_a_copy_outside_the_repo", "packaged_selftest", "medical_vocab", "bundled_large_v3", "model_checksum",
           "ocr_resources", "medical_expert_selftest", "diagnostics", "tts_english", "audio_stt", "new_frontend", "first_launch_no_downloader",
           "no_model_download", "video_stt", "video_progress", "queue", "medical_expert", "medical_intelligence_resources", "medical_intelligence_on", "media_formats", "pdf", "ocr", "study_workspace",
           "no_dev_environment", "app_restart")


def ci_readiness(report: dict[str, Any], dmg_sha256: str, manifest: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    """``(verdict, reasons, caveats)`` for a DMG built and tested on a GitHub-hosted Apple Silicon macOS runner. The verdict is about the technical tests
    only; it never says anything about a physical Mac, signing authority or notarization - those stay in ``caveats``."""
    reasons: list[str] = []
    if not report or report.get("provider") != "GitHub Actions" or report.get("machine") != "arm64":
        return "NEEDS WORK", ["no packaged-app validation report from an arm64 GitHub Actions runner"], []
    if manifest.get("dmg_sha256") != dmg_sha256:
        reasons.append("the DMG is not the file the CI build recorded (SHA-256 differs)")
    if report.get("dmg_sha256") and report["dmg_sha256"] != dmg_sha256:
        reasons.append("the packaged-app tests ran on a different DMG than the one handed out")
    for key in CI_KEYS:
        status = (report.get("checks") or {}).get(key)
        if status != "PASS":
            reasons.append(f"{key}: {status or 'not run'}")
    check = manifest.get("dmg_check") or {}
    for key in ("mount", "copy", "eject", "selftest"):
        if not check.get(key):
            reasons.append(f"DMG {key} check did not pass")
    if (manifest.get("mach_o_audit") or {}).get("problems", 1) != 0:
        reasons.append("the Mach-O / rpath / architecture audit found problems")
    if not manifest.get("study_quality_included"):
        reasons.append("Study Quality: the validated study-facts engine is not in the build")
    tests = manifest.get("tests") or {}
    if not tests.get("ran"):
        reasons.append("the macOS test suite was not run for this build")
    if tests.get("failed", 1):
        reasons.append(f"{tests.get('failed')} automated tests failed on macOS")
    caveats = ["tested on a GitHub-hosted Apple Silicon macOS VM, not on the user's physical Mac"]
    if not manifest.get("developer_id"):
        caveats.append("no Developer ID: signed ad-hoc only, so another Mac shows a Gatekeeper warning (right-click > Open)")
    if not manifest.get("notarized"):
        caveats.append("not notarized")
    if (manifest.get("media_libraries") or {}).get("lgpl_only"):
        caveats.append("media libraries are LGPL-only builds swapped in after PyInstaller (libx264 / libx265 removed)")
    elif manifest.get("gpl_source_offer") != "SET":
        caveats.append("GPL source-offer contact not set (libx264 / libx265 are inside the bundled PyAV / OpenCV libraries)")
    effective = manifest.get("min_macos_effective")
    if effective and effective != MIN_MACOS:
        caveats.append(f"the binaries really need macOS {effective} (target was {MIN_MACOS})")
    return ("CI-TESTED APPLE SILICON DMG READY" if not reasons else "NEEDS WORK"), reasons, caveats
