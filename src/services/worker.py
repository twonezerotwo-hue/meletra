"""``python -m src.services.worker <request.json>``: run one transcription job in this process (used by the GUI)."""

from __future__ import annotations

import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    from .transcription_service import run_worker

    if not argv:
        print("usage: python -m src.services.worker <request.json>", file=sys.stderr)
        return 2
    return run_worker(Path(argv[0]))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
