"""Medical Intelligence as a LAYER on top of the v3 Medical Expert - the combined correction.

    ASR -> language / repetition / timestamp safety (v3) -> Medical Expert (v3) -> v4 Medical Intelligence -> confidence gate -> combine -> medical corrected transcript

The v3 Medical Expert keeps its decisions. v4 adds only the corrections of its HIGH tier that sit where the Medical Expert changed nothing; on every conflict the Medical Expert wins. v4's MEDIUM
tier is never applied: it is listed as suggestions. The raw transcript is never touched (this module only builds the text of the medical corrected transcript).

``mode`` (Settings > Medical Intelligence, or MLT_V4_MODE): ``off`` (default: v4 does not run), ``shadow`` (v4 runs and is compared with the Medical Expert in its own report, the transcript is
not changed), ``on`` (shadow + the combined correction). Any problem with the knowledge database - missing, corrupt, a schema this version cannot read - degrades to ``off`` for that recording,
says so in the reports and never stops the transcription.
"""

from __future__ import annotations

import logging
import os
import time
from difflib import SequenceMatcher
from typing import Any, Optional, Sequence

log = logging.getLogger(__name__)

MODES = ("off", "shadow", "on")
MODE_ENV = "MLT_V4_MODE"
DEFAULT_MODE = "off"                 # v4 did not pass the promotion gates (docs/v4/HELDOUT_V3.md): it is opt-in


def resolve_mode(setting: Optional[str] = None, legacy_shadow: Optional[bool] = None) -> str:
    """The environment wins (for tests and support), then the setting, then the old shadow flag (MLT_V4_SHADOW), then ``off``."""
    env = (os.environ.get(MODE_ENV) or "").strip().lower()
    if env:
        return env if env in MODES else DEFAULT_MODE
    if setting and setting.strip().lower() in MODES:
        return setting.strip().lower()
    from .shadow import enabled

    return "shadow" if enabled(legacy_shadow) else DEFAULT_MODE


def overlay(before: str, production: str, items: Sequence[Any]) -> tuple[str, list]:
    """``(text, applied)``: ``items`` are the v4 corrections of ONE segment (``start``, ``end``, ``replacement``; offsets in ``before``, the text the Medical Expert worked on)."""
    if not items:
        return production, []
    ops = SequenceMatcher(None, before, production, autojunk=False).get_opcodes()
    edits = []
    for c in items:
        inside = [op for op in ops if op[1] < c.end and c.start < op[2]]
        if not inside or any(op[0] != "equal" for op in inside):
            continue                                                    # the Medical Expert changed something here: it decided
        first, last = inside[0], inside[-1]
        start = first[3] + (c.start - first[1])
        end = last[3] + (c.end - last[1])
        if 0 <= start <= end <= len(production) and production[start:end].casefold() == before[c.start:c.end].casefold():
            edits.append((start, end, c))
    text, applied, last_start = production, [], None
    for start, end, c in sorted(edits, key=lambda e: -e[0]):
        if last_start is not None and end > last_start:
            continue                                                    # two corrections may not overlap
        text = text[:start] + c.replacement + text[end:]
        applied.append(c)
        last_start = start
    applied.reverse()
    return text, applied


def run_intelligence(views: Sequence[Any], current: Any, *, mode: str, subjects: Sequence[str] = (), lecture_terms: Sequence[str] = (), custom_terms: Sequence[str] = (),
                     knowledge_path: Any = None) -> tuple[Optional[dict], dict, Optional[list[str]]]:
    """``(shadow report or None, the v4 block for the correction report, new texts or None)``.

    ``current`` is the Medical Expert's result (``texts``, ``corrections``). In mode ``on`` the third value holds the combined texts (same length as ``views``); otherwise None. Never raises.
    """
    info: dict = {"mode": mode, "status": "off", "applied": [], "suggested": [], "database": None}
    if mode not in ("shadow", "on"):
        return None, info, None
    started = time.perf_counter()
    try:
        from .engine import V4Engine
        from .knowledge import open_knowledge
        from .shadow import build_report, compare

        if knowledge_path and not os.path.isfile(knowledge_path):                  # an explicit path is never silently replaced by another database
            info["status"] = f"unavailable: the knowledge database {knowledge_path} does not exist - the v3 Medical Expert ran alone"
            return None, info, None
        kb = open_knowledge(knowledge_path)
        if kb is None:
            info["status"] = "unavailable: no medical knowledge database is installed - the v3 Medical Expert ran alone"
            return None, info, None
        try:
            problems = [c for c in kb.health() if not c.ok]
            if problems:
                info["status"] = "unavailable: the knowledge database failed its health check (" + "; ".join(f"{c.name}: {c.detail}" for c in problems[:2]) + ") - the v3 Medical Expert ran alone"
                return None, info, None
            meta = kb.info()
            info["database"] = {"build_version": meta["build_version"], "profile": meta["profile"], "schema_version": meta["schema_version"], "terms": meta["counts"].get("terms")}
            v4 = V4Engine(kb).run(views, subjects, lecture_terms, custom_terms)
            report = build_report(v4, compare(current, v4), kb, time.perf_counter() - started)
        finally:
            kb.close()
        info["suggested"] = [c.to_dict() for c in v4.uncertain]
        new_texts: Optional[list[str]] = None
        if mode == "on":
            new_texts = list(current.texts)
            for k, view in enumerate(views):
                items = [c for c in v4.corrections if c.segment == k]
                if items:
                    new_texts[k], applied = overlay(view.text, current.texts[k], items)
                    info["applied"] += [c.to_dict() for c in applied]
        info["status"] = "ok"
        info["seconds"] = round(time.perf_counter() - started, 2)
        return report, info, new_texts
    except Exception as exc:                                                    # noqa: BLE001 - Medical Intelligence must never stop a transcription
        log.exception("v4 Medical Intelligence failed; the v3 Medical Expert result is used")
        info.update({"status": f"unavailable: {type(exc).__name__}: {exc} - the v3 Medical Expert ran alone", "applied": [], "suggested": []})
        return None, info, None
