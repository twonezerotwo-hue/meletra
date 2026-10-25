#!/usr/bin/env bash
# ONE COMMAND, on an Intel Mac:   ./build_macos_intel.sh
# Result: release/v3.1.0/macOS-Intel/MedicalLectureTranscriber-Intel-Mac-Offline.dmg (+ .sha256, release_manifest.json)
#
# Needs: an Intel (x86_64) Mac with macOS 12 or newer, 25 GB free disk, internet during the BUILD only (pip packages and, once, the 3 GB speech
# model - skipped if MLT_MODEL_DIR points at a folder that already holds it), Python 3.11 (python.org installer is fine), Xcode command line
# tools (hdiutil, codesign, otool are part of macOS; `xcode-select --install` if a tool is missing).
#
# Optional:  MLT_MODEL_DIR=/path/to/whisper-large-v3   MLT_SOURCE_CONTACT="you@example.com"   (GPL source offer - needed for "ready")
#            MLT_SIGN_IDENTITY="Developer ID Application: Name (TEAMID)"   MLT_NOTARY_PROFILE=notary   MLT_SKIP_TESTS=1   PYTHON=python3.11
#
# After it: install the DMG on the test Mac, switch the network OFF, run scripts/validate_macos_intel.py (see docs/MACOS_INTEL_RELEASE.md).
set -euo pipefail
cd "$(dirname "$0")"

if [ "$(uname -s)" != "Darwin" ]; then
    echo "REAL INTEL MAC REQUIRED: this script builds a macOS app and must run on a Mac." >&2
    exit 2
fi
if [ "$(uname -m)" != "x86_64" ]; then
    echo "REAL INTEL MAC REQUIRED: this Mac is $(uname -m). An Intel build has to be made and tested on an Intel Mac (not under Rosetta)." >&2
    exit 2
fi

PY="${PYTHON:-python3.11}"
if ! command -v "$PY" >/dev/null 2>&1; then
    echo "Python 3.11 was not found. Install it from https://www.python.org/downloads/macos/ (3.11.x), then run this again." >&2
    exit 2
fi
VENV=".venv-macos-intel"
[ -d "$VENV" ] || "$PY" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -c "import platform, sys; assert platform.machine() == 'x86_64' and sys.version_info[:2] == (3, 11), (platform.machine(), sys.version)"

python -m pip install --upgrade pip wheel
python -m pip install -r requirements-macos-intel.txt
# text recognition: RapidOCR carries its ONNX models inside the wheel; it asks for the full OpenCV, the headless one is enough
python -m pip install --no-deps rapidocr-onnxruntime==1.4.4

python scripts/build_macos_intel_offline.py "$@"
