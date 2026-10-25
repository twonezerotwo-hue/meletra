"""Download / verify / list models with the same code the desktop app uses (resumable, checksum-verified).

Usage:
  python scripts/download_model.py list
  python scripts/download_model.py download whisper-large-v3
  python scripts/download_model.py verify whisper-large-v3
  python scripts/download_model.py import whisper-large-v3 "D:\\models\\large-v3"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.platform_utils import configure_utf8_console  # noqa: E402
from src.services.model_service import ModelService, human_bytes  # noqa: E402
from src.services.paths import get_paths  # noqa: E402


def main() -> int:
    configure_utf8_console()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("list", "download", "verify", "delete", "import"))
    parser.add_argument("key", nargs="?")
    parser.add_argument("folder", nargs="?")
    args = parser.parse_args()
    service = ModelService(get_paths())
    if args.action == "list":
        for state in service.list_models(include_hidden=True):
            print(f"{state.key:28} {'INSTALLED' if state.installed else 'not installed':14} {human_bytes(state.size_on_disk) if state.installed else '':>10}"
                  f"  partial {human_bytes(state.partial_bytes)}  {state.path or ''}")
        return 0
    if not args.key:
        parser.error("a model key is required")
    last = [0.0]

    def show(event: dict) -> None:
        now = time.time()
        if now - last[0] < 10 and event.get("percent", 0) < 99.99:
            return
        last[0] = now
        if event.get("message"):
            print(f"[{time.strftime('%H:%M:%S')}] {event['message']}", flush=True)
            return
        speed = event.get("speed_bps") or 0
        eta = event.get("eta_seconds")
        print(f"[{time.strftime('%H:%M:%S')}] {event.get('file', ''):28} {event['percent']:5.1f}%  "
              f"{human_bytes(event.get('bytes_done', 0))} / {human_bytes(event.get('bytes_total', 0))}  "
              f"{human_bytes(speed)}/s  ETA {int(eta // 60) if eta else '?'} min", flush=True)

    if args.action == "download":
        folder = service.download(args.key, progress=show)
        print("installed:", folder)
        result = service.verify(args.key)
        print("verified:", result.ok, result.problems)
        return 0 if result.ok else 1
    if args.action == "verify":
        result = service.verify(args.key, progress=show)
        print("verified:", result.ok, result.problems)
        return 0 if result.ok else 1
    if args.action == "delete":
        print("deleted:", service.delete(args.key))
        return 0
    folder, result = service.import_folder(args.key, Path(args.folder), progress=show)
    print("imported:", folder, "verified:", result.ok, result.problems)
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
