"""SHADOW MODE: v4 runs next to the production Medical Expert, is compared with it, and changes NOTHING.

For every doubtful span either engine looked at, the report says what the current engine did, what v4 would do, how confident v4 is and whether they agree.
The production transcript is produced only by the existing engine; ``v4_shadow_report.json`` is an extra file that is written only when the feature flag is on
(setting "Medical Intelligence (shadow)" or the environment variable ``MLT_V4_SHADOW=1``). Any failure here is logged and swallowed: shadow mode must never
affect a transcription.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from ...medical.corrector import CorrectionResult, SegmentView
from .engine import V4Engine, V4Result
from .knowledge import MedicalKnowledge, find_database, open_knowledge

log = logging.getLogger(__name__)
FLAG_ENV = "MLT_V4_SHADOW"
REPORT_NAME = "v4_shadow_report.json"
SHADOW_VERSION = "0.1.0"


def enabled(setting: Optional[bool] = None) -> bool:
    env = os.environ.get(FLAG_ENV)
    if env is not None:
        return env.strip().lower() in ("1", "true", "yes", "on")
    return bool(setting)


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


V4_FIELDS = ("best_score", "keep_original_score", "correction_cost", "adjusted_score", "second_best", "second_best_score", "score_margin", "margin_over_original", "retrieval_source",
             "phonetic_score", "lexical_score", "context_evidence", "pdf_evidence", "domain_evidence", "evidence_families", "decision_reason", "number_form", "changed_word_validity")


def _v4(v) -> dict:
    """Everything needed to debug one proposal: the candidate, its confidence (the adjusted score), the keep-original score, best vs second best, where it came from,
    the sound / spelling / context / lecture (PDF) / field evidence and the reason for the tier."""
    d = v.details
    out = {"replacement": v.replacement, "tier": v.tier, "confidence": round(v.confidence, 3), "domain": v.domain, "concept_id": v.concept_id}
    out.update({k: d[k] for k in V4_FIELDS if k in d})
    return out


def _production(c) -> dict:
    return {"replacement": c.corrected, "tier": c.tier, "confidence": round(c.confidence, 3)}


def compare(current: Optional[CorrectionResult], v4: V4Result) -> list[dict]:
    """One row per doubtful span: the PRODUCTION decision, the V4 proposal with its full evidence, the confidence and whether they AGREE."""
    rows: list[dict] = []
    used: set[int] = set()
    cur = [c for c in (current.corrections if current else []) if c.category == "medical" and c.tier in ("high", "medium")]
    for v in v4.corrections + v4.uncertain:
        match = next((i for i, c in enumerate(cur) if c.segment == v.segment and _overlap(c.span, (v.start, v.end))), None)
        if match is not None:
            used.add(match)
            c = cur[match]
            same = c.corrected.strip().lower() == v.replacement.strip().lower()
            agreement = "agree" if same and v.tier == "high" else "agree_weak" if same else "conflict"
            rows.append({"segment": v.segment, "timestamp": round(v.timestamp, 2), "original": v.original, "production_decision": "corrected", "current": _production(c),
                         "v4": _v4(v), "agreement": agreement})
        else:
            rows.append({"segment": v.segment, "timestamp": round(v.timestamp, 2), "original": v.original, "production_decision": "none", "current": None, "v4": _v4(v),
                         "agreement": "v4_only" if v.tier == "high" else "v4_only_weak"})
    for i, c in enumerate(cur):
        if i not in used:
            rows.append({"segment": c.segment, "timestamp": None, "original": c.original, "production_decision": "corrected", "current": _production(c), "v4": None,
                         "agreement": "current_only"})
    rows.sort(key=lambda r: (r["segment"], r["original"]))
    return rows


def near_misses(v4: V4Result, limit: int = 30) -> list[dict]:
    """Proposals the gate REJECTED (tier low) that came close: why the original stayed. Debug material, not proposals."""
    out = []
    for d in v4.decisions:
        if d.tier == "low" and d.best is not None and d.best.score >= 0.6:
            a = d.assessment
            out.append({"segment": d.span.segment, "original": d.span.text, "candidate": d.best.candidate.surface, "best_score": round(d.best.score, 3),
                        "adjusted_score": round(a.adjusted, 3), "keep_original_score": round(a.keep, 3), "second_best": d.runner_up.candidate.surface if d.runner_up else "",
                        "second_best_score": round(d.runner_up.score, 3) if d.runner_up else None, "score_margin": round(d.candidate_margin, 3),
                        "retrieval_source": list(d.best.candidate.channels), "evidence_families": list(a.families), "decision_reason": d.reason})
    out.sort(key=lambda r: -r["adjusted_score"])
    return out[:limit]


def summarize(rows: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["agreement"]] = counts.get(r["agreement"], 0) + 1
    return counts


def build_report(v4: V4Result, rows: list[dict], knowledge: MedicalKnowledge, seconds: float) -> dict:
    info = knowledge.info()
    return {"version": SHADOW_VERSION, "mode": "shadow", "production_transcript_changed": False,
            "knowledge": {"build_version": info["build_version"], "profile": info["profile"], "schema_version": info["schema_version"], "terms": info["counts"].get("terms"),
                          "concepts": info["counts"].get("concepts")},
            "topic": v4.topic, "stats": {k: v for k, v in v4.stats.items() if k != "cache"}, "seconds": round(seconds, 2), "agreement_summary": summarize(rows),
            "abbreviations": v4.abbreviations, "hotwords_for_targeted_redecode": v4.hotwords, "comparison": rows, "rejected_near_misses": near_misses(v4), "v4_corrections": [c.to_dict() for c in v4.corrections], "v4_uncertain": [c.to_dict() for c in v4.uncertain]}


def run_shadow(views: Sequence[SegmentView], current: Optional[CorrectionResult], *, subjects: Sequence[str] = (), lecture_terms: Iterable[str] = (),
               custom_terms: Iterable[str] = (), knowledge: Optional[MedicalKnowledge] = None, knowledge_path: Optional[str | Path] = None) -> Optional[dict]:
    """The shadow report for one lecture, or None when no knowledge base is installed / anything fails. Never raises."""
    try:
        started = time.perf_counter()
        kb = knowledge or open_knowledge(knowledge_path)
        if kb is None:
            log.info("v4 shadow: no medical knowledge database installed; skipped")
            return None
        v4 = V4Engine(kb).run(views, subjects, lecture_terms, custom_terms)
        rows = compare(current, v4)
        return build_report(v4, rows, kb, time.perf_counter() - started)
    except Exception:                                                              # noqa: BLE001 - shadow mode must never affect production
        log.exception("v4 shadow analysis failed; the transcription is unaffected")
        return None


def write_report(report: dict, directory: Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / REPORT_NAME
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return path
