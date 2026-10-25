"""Structured job events: how the pipeline tells the desktop app (or any other caller) what it is doing.

Every event is a plain JSON-serialisable dict ``{"type": ..., "time": ..., <payload>}``. The pipeline never depends on who
listens: with no sink events are simply dropped. A broken sink can never break a transcription.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)

#: every event type the pipeline can emit, in the order a normal job produces them
EVENT_TYPES: tuple[str, ...] = (
    "job_started", "source_validated", "audio_preparing", "audio_ready", "language_detecting", "language_detected",
    "subject_detected", "model_loading", "model_loaded", "resume_started", "chunk_started", "chunk_progress",
    "chunk_finished", "checkpoint_saved", "gap_scan_started", "gap_found", "gap_repair_started", "gap_repair_finished",
    "loop_repaired", "quality_audit_started", "quality_audit_finished", "warning", "pause_requested", "paused",
    "completed", "failed",
    "translation_started", "translation_progress", "translation_finished",     # Output Language: the translated transcript (src/translation)
    "stage",           # what the job is doing BEFORE the first transcript segment: {stage, percent (None = not measurable), done, total}
)

EventSink = Callable[[dict[str, Any]], None]


def jsonable(value: Any) -> Any:
    """Round-trip through JSON (paths, tuples, dataclasses-as-str ...) so a payload can always be serialised."""
    return json.loads(json.dumps(value, default=str))


class Emitter:
    """Builds events and hands them to the sink. ``Emitter(None)`` is a no-op."""

    def __init__(self, sink: Optional[EventSink] = None) -> None:
        self.sink = sink
        self.history: list[str] = []          # event types seen, in order (tests, diagnostics)

    def __call__(self, event_or_type: Any, **data: Any) -> None:
        """``emitter({"type": "x", ...})`` or ``emitter("x", key=value)``."""
        if isinstance(event_or_type, dict):
            payload = dict(event_or_type)
        else:
            payload = {"type": str(event_or_type), **data}
        payload.setdefault("time", round(time.time(), 3))
        self.history.append(str(payload.get("type")))
        if self.sink is None:
            return
        try:
            self.sink(jsonable(payload))
        except Exception as exc:  # noqa: BLE001 - listeners must never break a job
            log.debug("event sink failed for %s: %s", payload.get("type"), exc)
