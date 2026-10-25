"""The three corpus boundaries of the v4 benchmark (benchmarks/v4/corpus/splits.json).

    dev          the passages the engine was developed on (including the first held-out corpus, which stopped being clean when it was inspected)
    validation   checkpoints while hardening the decision engine; rules may be adjusted until these numbers are acceptable
    heldout_v2   written and committed BEFORE v4 ran on it; evaluated once; never used for tuning

Every metric printed by the tools names the set it was computed on.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

CORPUS = Path(__file__).resolve().parents[1] / "benchmarks" / "v4" / "corpus"
SET_NAMES = ("dev", "validation", "heldout_v2", "heldout_v3", "heldout_v4", "heldout_v5")


def splits() -> dict[str, list[str]]:
    return json.loads((CORPUS / "splits.json").read_text(encoding="utf-8"))["sets"]


def passages_of(name: Optional[str] = None) -> dict[str, dict]:
    """id -> passage (with the extra key ``set``). ``name`` None or 'all' = every set."""
    wanted = SET_NAMES if name in (None, "all") else (name,)
    out: dict[str, dict] = {}
    for set_name in wanted:
        for file in splits()[set_name]:
            for p in json.loads((CORPUS / file).read_text(encoding="utf-8"))["passages"]:
                if p["id"] in out:
                    raise ValueError(f"passage id {p['id']} is in two sets")
                out[p["id"]] = {**p, "set": set_name}
    return out


def set_of(passage_id: str) -> str:
    return passages_of().get(passage_id, {}).get("set", "")
