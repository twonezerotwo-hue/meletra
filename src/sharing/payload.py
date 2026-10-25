"""What a shared transcript contains - and nothing else. The allow-list is the whole privacy contract of the upload."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from . import MAX_PAYLOAD_BYTES, SCHEMA

#: the only keys of a payload (tests fail if a key is added without being listed here and described in the notice)
ALLOWED_KEYS = ("schema", "installation_id", "app_version", "created_at", "lecture_name", "languages", "transcript", "medical_corrected_transcript",
                "translated_transcript")


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def build(folder: Path, lecture_name: str, installation: str, app_version: str, spoken: str = "", requested: str = "", output: str = "",
          now: Optional[datetime] = None) -> Optional[dict[str, Any]]:
    """The payload for the finished lecture in ``folder``, or None when there is no transcript text to share."""
    folder = Path(folder)
    transcript = _read(folder / "local_raw_transcript.txt").strip()
    if not transcript:
        return None
    corrected = _read(folder / "medical_corrected_transcript.txt").strip()
    translated = _read(folder / f"translated_{output}_transcript.txt").strip() if output else ""
    if not spoken:
        try:
            spoken = str((json.loads((folder / "metadata.json").read_text(encoding="utf-8")).get("language") or {}).get("detected") or "")
        except (OSError, ValueError):
            spoken = ""
    payload: dict[str, Any] = {
        "schema": SCHEMA, "installation_id": installation, "app_version": app_version,
        "created_at": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
        "lecture_name": " ".join((lecture_name or "").split())[:200],
        "languages": {"spoken": spoken, "requested": requested, "output": output},
        "transcript": transcript,
    }
    if corrected and corrected != transcript:
        payload["medical_corrected_transcript"] = corrected
    if translated:
        payload["translated_transcript"] = translated
    if len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) > MAX_PAYLOAD_BYTES:          # an extremely long lecture: the transcript alone, trimmed
        payload.pop("translated_transcript", None)
        payload.pop("medical_corrected_transcript", None)
        payload["transcript"] = transcript[: MAX_PAYLOAD_BYTES // 4]
    return payload
