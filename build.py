"""One entry point for every build:  python build.py [--all] [--full-offline] [--dry-run] [--skip-tests]

    On Windows    tests -> app -> packaged smoke test -> portable zip -> installer   (scripts/*.py, results in release/v<version>/Windows)
    On macOS      ./build_macos.sh                                                   (results in release/v<version>/macOS)
    --all         afterwards: release/v<version>/README.txt and the all-platforms zip (whatever exists; what is missing is said so)
    --dry-run     print the steps and change nothing

A build never touches the files of an earlier version: 0.2.x goes to dist/v0.2.0 and release/v0.2.0, the 0.1.1 files stay as they are.
The macOS artifacts exist only if this script (or build_macos.sh) ran on a Mac; running it on Windows does not pretend otherwise.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from build_tools import release                                       # noqa: E402
from src import __version__                                           # noqa: E402


def plan(os_name: str, *, full_offline: bool = False, skip_tests: bool = False, python: str = sys.executable) -> list[list[str]]:
    """The commands for ``os_name`` (``win32`` | ``darwin``), in order."""
    if os_name == "win32":
        steps = [] if skip_tests else [[python, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"]]
        steps += [[python, "scripts/build_app.py"], [python, "scripts/smoke_packaged.py", "--quick"], [python, "scripts/build_portable.py"],
                  [python, "scripts/build_installer.py"]]
        if full_offline:
            steps.append([python, "scripts/build_fulloffline.py"])
        return steps
    if os_name == "darwin":
        return [["./build_macos.sh"]]
    raise SystemExit(f"No build is defined for {os_name!r}: build on Windows or macOS.")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="also write README.txt and the all-platforms zip")
    parser.add_argument("--full-offline", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    steps = plan(sys.platform, full_offline=args.full_offline, skip_tests=args.skip_tests)
    for step in steps:
        print("[build]", " ".join(step), flush=True)
        if args.dry_run:
            continue
        code = subprocess.run(step, cwd=ROOT, check=False).returncode
        if code != 0:
            print(f"[build] stopped: {' '.join(step)} failed with exit code {code}", file=sys.stderr)
            return code
    if args.all:
        if args.dry_run:
            print(f"[build] would write {release.all_platforms_name(__version__)} and README.txt in release/")
        else:
            archive = release.all_platforms_zip(__version__, ROOT)
            print(f"[build] all-platforms package: {archive}")
    print("[build] done" + (" (dry run)" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
