"""Timing bookkeeping that survives pauses, crashes and resumes.

A long lecture is often transcribed in several sessions. The numbers a user cares about are therefore of two kinds:
*this run* (what just happened) and *cumulative / effective* (what the file really cost). The cumulative figures are kept in
``sessions.json`` inside the job's checkpoint folder, one entry per finished session.

Real-time factor (RTF) = seconds of work / seconds of audio. It is only defined when audio was actually decoded: a run that
found every chunk already finished reports ``None``, never a nonsense number.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

SESSIONS_FILE = "sessions.json"


def safe_rtf(work_seconds: float, audio_seconds: float) -> Optional[float]:
    """work / audio, or ``None`` when no audio was decoded."""
    if audio_seconds is None or audio_seconds <= 0.5 or work_seconds is None or work_seconds < 0:
        return None
    return round(work_seconds / audio_seconds, 4)


@dataclass
class SessionStats:
    started: str = ""
    outcome: str = "completed"          # completed | paused | failed
    wall_seconds: float = 0.0
    model_load_seconds: float = 0.0
    decode_seconds: float = 0.0         # main chunk decoding done by this session (model load excluded)
    decoded_audio_seconds: float = 0.0
    repair_seconds: float = 0.0         # loop + gap re-decoding
    repair_audio_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_sessions(directory: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads((Path(directory) / SESSIONS_FILE).read_text(encoding="utf-8"))
        return list(data) if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def record_session(directory: Path, session: SessionStats) -> None:
    """Append ``session`` to the job's history (atomic write; never raises)."""
    try:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        sessions = load_sessions(directory)
        sessions.append(session.to_dict())
        target = directory / SESSIONS_FILE
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(sessions, ensure_ascii=False), encoding="utf-8")
        tmp.replace(target)
    except OSError:
        pass


def timing_report(*, audio_total: float, this_run: SessionStats, earlier: list[dict[str, Any]],
                  chunk_cumulative: Optional[dict[str, float]] = None) -> dict[str, Any]:
    """The ``timing`` block of metadata.json (and the numbers the GUI shows).

    * ``audio_duration_total``      length of the recording
    * ``audio_decoded_this_run``    audio decoded in this session (main decode + repairs)
    * ``audio_decoded_cumulative``  audio decoded over all sessions
    * ``wall_time_this_run`` / ``wall_time_cumulative``
    * ``model_load_time``           this session
    * ``repair_time``               this session
    * ``real_time_factor_this_run`` / ``real_time_factor_effective`` (all sessions; ``None`` if nothing was decoded)
    """
    prior_wall = sum(float(s.get("wall_seconds", 0.0)) for s in earlier)
    prior_work = sum(float(s.get("decode_seconds", 0.0)) + float(s.get("repair_seconds", 0.0)) for s in earlier)
    prior_audio = sum(float(s.get("decoded_audio_seconds", 0.0)) + float(s.get("repair_audio_seconds", 0.0)) for s in earlier)
    work_now = this_run.decode_seconds + this_run.repair_seconds
    audio_now = this_run.decoded_audio_seconds + this_run.repair_audio_seconds
    cumulative_audio = prior_audio + audio_now
    cumulative_work = prior_work + work_now
    if chunk_cumulative and chunk_cumulative.get("audio_seconds", 0) > 0:
        # the chunk checkpoints know the true totals even if an earlier session's summary was lost (crash)
        cumulative_audio = max(cumulative_audio, chunk_cumulative["audio_seconds"] + this_run.repair_audio_seconds)
        cumulative_work = max(cumulative_work, chunk_cumulative["decode_seconds"] + this_run.repair_seconds)
    return {
        "audio_duration_total": round(audio_total, 2),
        "audio_decoded_this_run": round(audio_now, 2),
        "audio_decoded_cumulative": round(cumulative_audio, 2),
        "wall_time_this_run": round(this_run.wall_seconds, 2),
        "wall_time_cumulative": round(prior_wall + this_run.wall_seconds, 2),
        "model_load_time": round(this_run.model_load_seconds, 2),
        "repair_time": round(this_run.repair_seconds, 2),
        "real_time_factor_this_run": safe_rtf(work_now, audio_now),
        "real_time_factor_effective": safe_rtf(cumulative_work, cumulative_audio),
        "sessions": len(earlier) + 1,
        # legacy keys that scripts and tests already read
        "audio_seconds": round(audio_total, 2),
        "decode_seconds": round(this_run.decode_seconds, 2),
        "decoded_audio_seconds": round(this_run.decoded_audio_seconds, 2),
        "real_time_factor": safe_rtf(cumulative_work, cumulative_audio),
        "wall_seconds": round(this_run.wall_seconds, 2),
        "model_load_seconds": round(this_run.model_load_seconds, 2),
    }


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")
