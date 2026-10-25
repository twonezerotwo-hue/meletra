"""``MedicalLectureTranscriber.exe --medical-knowledge-info [OUT]`` and the self-test check of the knowledge base.

Exit codes: 0 = a knowledge base is installed and healthy, 1 = installed but damaged / incompatible (fails loudly), 2 = none installed.
A build that is MEANT to contain the database (a ``REQUIRED`` marker next to it, or MLT_KNOWLEDGE_REQUIRED=1) treats "none installed" as a failure too.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

from .database.store import IncompatibleKnowledgeBase
from .runtime.knowledge import KnowledgeError, MedicalKnowledge, find_database

REQUIRED_ENV = "MLT_KNOWLEDGE_REQUIRED"
REQUIRED_MARKER = "REQUIRED"


class NotInstalled(Exception):
    """No database is installed and none is required: the check is skipped, not failed."""


def is_required(db_path: Optional[Path] = None) -> bool:
    if os.environ.get(REQUIRED_ENV, "").strip().lower() in ("1", "yes", "true"):
        return True
    return bool(db_path and (Path(db_path).parent / REQUIRED_MARKER).is_file())


def report(explicit: Optional[str] = None) -> dict:
    """Everything the CLI, the self-test and the Settings page show. Never raises."""
    found = find_database(explicit)
    out: dict = {"installed": found is not None, "file": str(found) if found else "", "required": is_required(found), "ok": False, "status": "", "checks": []}
    if found is None:
        out["status"] = "no medical knowledge database installed (the Medical Expert of v3 is used alone)"
        out["ok"] = not out["required"]
        if out["required"]:
            out["status"] = "REQUIRED medical knowledge database is missing"
        return out
    try:
        kb = MedicalKnowledge.open(found)
    except (IncompatibleKnowledgeBase, KnowledgeError, Exception) as exc:  # noqa: BLE001 - a damaged file is a result, not a crash
        out["status"] = f"cannot open the medical knowledge database: {type(exc).__name__}: {exc}"
        return out
    try:
        out.update(kb.info())
        checks = kb.health()
        out["checks"] = [{"name": c.name, "ok": c.ok, "detail": c.detail} for c in checks]
        failed = [c.name for c in checks if not c.ok]
        out["ok"] = not failed
        out["status"] = "ready" if not failed else "FAILED: " + ", ".join(failed)
    except Exception as exc:  # noqa: BLE001
        out["status"] = f"health check failed: {type(exc).__name__}: {exc}"
    finally:
        kb.close()
    return out


def as_text(info: dict) -> str:
    lines = ["Medical knowledge", "-" * 17]
    if not info.get("installed"):
        lines.append(info["status"])
        return "\n".join(lines) + "\n"
    c = info.get("counts", {})
    lines += [f"file       : {info.get('file')}", f"name       : {info.get('name', '')} {info.get('build_version', '')} (schema {info.get('schema_version', '?')})",
              f"profile    : {info.get('profile', '')}   built {info.get('build_date', '')}", f"size       : {info.get('size_bytes', 0) / 1e6:.1f} MB",
              f"concepts   : {c.get('concepts', 0):,}   terms: {c.get('terms', 0):,}   abbreviations: {c.get('abbreviations', 0):,}   relationships: {c.get('relationships', 0):,}",
              "languages  : " + (", ".join(f"{k} {v:,}" for k, v in info.get("languages", {}).items()) or "-"), "sources    :"]
    for s in info.get("sources", []):
        lines.append(f"  - {s['source_id']:<14} {s['name'][:44]:<44} {s['version'] or '':<12} {s['license_status']:<16} {'redistributable' if s['redistribution_allowed'] else 'private only'}")
    lines.append("checks     :")
    for k in info.get("checks", []):
        lines.append(f"  [{'ok' if k['ok'] else 'FAIL'}] {k['name']}: {k['detail']}")
    lines.append(f"status     : {info.get('status')}")
    return "\n".join(lines) + "\n"


def exit_code(info: dict) -> int:
    if not info.get("installed"):
        return 1 if info.get("required") else 2
    return 0 if info.get("ok") else 1


def run_command(out: str = "", explicit: Optional[str] = None, as_json: bool = False) -> int:
    info = report(explicit)
    text = json.dumps(info, indent=1, ensure_ascii=False) if as_json or (out and out.lower().endswith(".json")) else as_text(info)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text, encoding="utf-8")
    elif sys.stdout is not None:
        sys.stdout.write(text)
    return exit_code(info)


def selftest_detail() -> str:
    """For src/services/selftest.py: raises when an installed or required database is not healthy; a build without one is reported as such."""
    info = report()
    if not info["installed"] and not info["required"]:
        raise NotInstalled(info["status"])
    if not info["ok"]:
        bad = "; ".join(f"{k['name']}: {k['detail']}" for k in info.get("checks", []) if not k["ok"]) or info["status"]
        raise RuntimeError(f"medical knowledge database is not usable - {bad}")
    c = info.get("counts", {})
    return (f"{info['name']} {info['build_version']} ({info['profile']}): {c.get('concepts', 0):,} concepts, {c.get('terms', 0):,} terms; "
            f"{len(info['checks'])} checks passed (open, schema, indexes, search, phonetic, relationships, licence metadata)")


