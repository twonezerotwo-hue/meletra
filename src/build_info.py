"""Which build is this? Read from ``build_info.json`` that ``scripts/release_stamp.py`` writes next to the packaged program.

An installed build can be checked against the commit that passed the tests: About shows the commit, Diagnostics lists it. In a source
checkout there is no such file and the answer says so (no git call at run time).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import APP_DIR, RESOURCE_ROOT, __version__

FILE_NAME = "build_info.json"


def load() -> dict[str, Any]:
    for folder in (APP_DIR, RESOURCE_ROOT):
        try:
            data = json.loads((Path(folder) / FILE_NAME).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict):
            if data.get("version") not in (None, "", __version__):         # left behind by an older installation: it says nothing about THIS program
                return {"version": __version__, "commit": "", "branch": "", "build_date": "", "stale_build_file": str(data.get("version"))}
            return {"version": __version__, **data}
    return {"version": __version__, "commit": "", "branch": "", "build_date": ""}


def commit(short: bool = True) -> str:
    full = str(load().get("commit") or "")
    return (full[:7] if short else full) or "source checkout"


def line() -> str:
    """One line for About / logs: ``3.1.0 (commit 1a2b3c4)``."""
    return f"{__version__} (commit {commit()})"
