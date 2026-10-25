"""``--download-model KEY [OUT]``: download one model without a window (used to prove, from the packaged .exe, that HTTPS download,
resume and SHA-256 verification work). Prints nothing; writes a JSON result file."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from .model_service import ModelError, ModelService


def download_command(key: str, out_file: str = "", models: Optional[ModelService] = None) -> int:
    started = time.monotonic()
    result: dict = {"ok": False, "key": key}
    code = 1
    try:
        if models is None:
            from .paths import get_paths

            models = ModelService(get_paths())
        info = models.info(key)
        last = {"bytes_total": 0}

        def progress(event: dict) -> None:
            last["bytes_total"] = event.get("bytes_total", last["bytes_total"])

        path = models.download(key, progress)
        check = models.verify(key)
        state = models.state(key)
        result.update(ok=bool(check.ok), path=str(path), verified=check.ok, problems=check.problems, size_on_disk=state.size_on_disk,
                      files=len(info.files), seconds=round(time.monotonic() - started, 1))
        code = 0 if check.ok else 1
    except ModelError as exc:
        result["error"] = str(exc)
    except Exception as exc:  # noqa: BLE001 - the result file is the only channel of a windowed exe
        result["error"] = f"{type(exc).__name__}: {exc}"
    if out_file:
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        Path(out_file).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return code
