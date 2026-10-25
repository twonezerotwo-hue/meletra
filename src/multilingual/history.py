"""Earlier results are never lost: when a recording is transcribed again into the folder it already has results in, the finished results of the earlier run (transcript, medical
transcript, subtitles, translations, source transcript, metadata) are moved to ``previous_versions/<date-time>/`` before the new run writes anything. Study material stays where it is
(it is tied to the transcript revision and is made again when that changed)."""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

HISTORY_DIR = "previous_versions"
RESULT_MARKERS = ("local_transcript.json", "metadata.json")           # a folder with both holds a finished run
KEEP_IN_PLACE = {"study", HISTORY_DIR, "checkpoint", "checkpoints", ".job_id"}


def has_finished_results(folder: Path) -> bool:
    return all((Path(folder) / name).is_file() for name in RESULT_MARKERS)


def archive_previous(folder: Path) -> Optional[Path]:
    """Move the finished results of an earlier run into ``previous_versions/<stamp>/``; returns that folder, or None when there was nothing to keep."""
    folder = Path(folder)
    if not folder.is_dir() or not has_finished_results(folder):
        return None
    target = folder / HISTORY_DIR / time.strftime("%Y%m%d-%H%M%S")
    counter = 1
    while target.exists():
        counter += 1
        target = folder / HISTORY_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-{counter}"
    try:
        target.mkdir(parents=True)
        for item in sorted(folder.iterdir()):
            if item.name in KEEP_IN_PLACE or item.suffix == ".log":
                continue
            shutil.move(str(item), str(target / item.name))
    except OSError:
        log.warning("could not keep the earlier results of %s; the new run continues", folder, exc_info=True)
        return None
    log.info("Earlier results of this recording were kept in %s", target)
    return target
