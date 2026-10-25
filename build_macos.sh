#!/usr/bin/env bash
# Build MELETRA for macOS: environment -> tests -> .app -> DMG (-> sign -> notarize).
#
#   ./build_macos.sh                 builds for this Mac (Apple Silicon -> ...-AppleSilicon.dmg, Intel -> ...-Intel.dmg)
#   MLT_SKIP_TESTS=1 ./build_macos.sh    skip the test run (not recommended)
#   MLT_SIGN_IDENTITY="Developer ID Application: Name (TEAMID)" MLT_NOTARY_PROFILE=notary ./build_macos.sh
#
# Result: release/v<version>/macOS/  (the DMG, the .app, SHA256SUMS.txt, build_report.json - which says Signed / Notarized YES or NO).
# Needs: macOS 12+, Python 3.11 (python3.11 on PATH or PYTHON=...), Xcode command line tools (hdiutil, codesign are part of macOS).
set -euo pipefail
cd "$(dirname "$0")"

if [ "$(uname -s)" != "Darwin" ]; then
    echo "build_macos.sh must run on a Mac. (On Windows use BUILD_WINDOWS.bat; 'python build.py --dry-run' shows the macOS steps.)" >&2
    exit 2
fi

ARCH="$(uname -m)"                         # arm64 = Apple Silicon, x86_64 = Intel
PY="${PYTHON:-python3.11}"
VENV=".venv-macos"

echo "[macos] architecture: $ARCH   python: $($PY --version 2>&1)"
[ -d "$VENV" ] || "$PY" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel >/dev/null
python -m pip install -r requirements-macos.txt
# text recognition for scanned pages and pictures: RapidOCR ships its own models but asks for the full OpenCV, the headless one is enough
python -m pip install --no-deps rapidocr-onnxruntime==1.4.4

if [ "${MLT_SKIP_TESTS:-0}" != "1" ]; then
    echo "[macos] running the tests (tests that need Windows are skipped)"
    python -m pytest -q -p no:cacheprovider tests
fi

python scripts/build_macos_app.py --arch "$ARCH"
echo
echo "[macos] done. Look in release/ for the v$(python -c 'from build_tools.layout import base_version; from src import __version__; print(base_version(__version__))')/macOS folder."
