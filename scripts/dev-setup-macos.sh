#!/bin/bash
# DEVELOPER SETUP (macOS) - for authorized collaborators of the private source repository. NOT the way to install MELETRA.
#
# End users install MELETRA from the public download page or with the one-command installer (see the README); they never need this repository.
# This script prepares a development checkout on a Mac: a Python 3.11 virtual environment .venv with the pinned dependencies of the Intel build
# (requirements-macos-intel.txt; on Apple Silicon requirements-macos.txt), then checks that the test runner works.
#
#   bash scripts/dev-setup-macos.sh              create .venv and install the dependencies
#   bash scripts/dev-setup-macos.sh --run-tests  ... and run the full test suite
set -euo pipefail

cd "$(cd "$(dirname "$0")/.." && pwd)"
echo "DEVELOPER SETUP - this is not the installation for users (see the README: \"Install MELETRA\")."
[ "$(uname -s)" = "Darwin" ] || { echo "This script is for macOS." >&2; exit 2; }

PY=""
for candidate in python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 11) else 1)' 2>/dev/null; then PY="$candidate"; break; fi
done
[ -n "$PY" ] || { echo "Python 3.11 is needed for development (https://www.python.org/downloads/macos/). Not found." >&2; exit 2; }

if [ "$(uname -m)" = "x86_64" ]; then REQ="requirements-macos-intel.txt"; else REQ="requirements-macos.txt"; fi
"$PY" -m venv .venv
.venv/bin/python -m pip install --upgrade pip wheel
.venv/bin/python -m pip install -r "$REQ"
if [ "$REQ" = "requirements-macos-intel.txt" ]; then .venv/bin/python -m pip install --no-deps rapidocr-onnxruntime==1.4.4; fi
.venv/bin/python -m pip install pytest
.venv/bin/python -c "import sys; print('Python', sys.version.split()[0])"
.venv/bin/python -m pytest --version
if [ "${1:-}" = "--run-tests" ]; then exec .venv/bin/python -m pytest -q -p no:cacheprovider; fi
echo "Done. Run the tests with:  .venv/bin/python -m pytest -q"
