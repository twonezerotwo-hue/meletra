"""``MELETRA.exe --translate-lecture FOLDER --to CODE``: translate the saved transcript of a lecture without a window (the same code as the Translate window)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def translate_lecture_command(folder: str, target: str, result_path: str = "") -> int:
    """Exit codes: 0 translated (or the saved translation was already there), 2 nothing to translate / pack missing / unsupported language, 1 failed."""
    from ..services.paths import get_paths
    from ..services.settings_service import SettingsService
    from .service import TranslationService

    imported_before = "faster_whisper" in sys.modules                                           # in a fresh program (the real command) this is False
    paths = get_paths().ensure()
    try:
        extra = list(SettingsService(paths.settings_file).settings.extra_model_dirs)
    except Exception:  # noqa: BLE001 - the settings are optional here
        extra = []
    service = TranslationService([paths.models, *[Path(d) for d in extra]])
    lecture = Path(folder)
    report: dict = {"folder": lecture.name, "target": target}
    availability = service.availability(lecture, target)
    report["availability"] = availability.state
    code = 0
    started = time.perf_counter()
    if availability.state in ("model_missing", "nothing_to_translate", "unsupported_target"):
        report.update(ok=False, error=availability.message)
        code = 2
    else:
        try:
            doc = service.translate(lecture, target)
            report.update(ok=True, from_cache=availability.state == "cached", status=doc.status, counts=doc.counts, engine=doc.engine_id, created_at=doc.created_at,
                          source_revision=doc.source_revision, segments=len(doc.segments), warnings=doc.warnings,
                          files=sorted(p.name for p in lecture.glob(f"translated_{target}_transcript.*")))
        except Exception as exc:  # noqa: BLE001
            report.update(ok=False, error=f"{type(exc).__name__}: {exc}")
            code = 1
    report["seconds"] = round(time.perf_counter() - started, 2)
    report["speech_recogniser_loaded"] = not imported_before and "faster_whisper" in sys.modules      # proof that THIS command did not load the speech model (a fresh program starts without it)
    report["speech_recogniser_imported_before"] = imported_before
    text = json.dumps(report, ensure_ascii=False, indent=1)
    if result_path:
        Path(result_path).parent.mkdir(parents=True, exist_ok=True)
        Path(result_path).write_text(text, encoding="utf-8")
    elif sys.stdout is not None:
        sys.stdout.write(text + "\n")
    return code
