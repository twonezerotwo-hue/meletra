"""Parameter sweeps of the confidence gate on the DEVELOPMENT and VALIDATION sets (never on heldout_v2).

    python tools/tune_gate.py --kb data/medical_knowledge_public.db --grid high=0.60,0.66,0.72 --grid high_keep_margin=0.15,0.2,0.3 [--runs-too]

Prepares every clip once (spans, retrieval, scoring), then re-decides for each combination of the listed GateConfig fields. Prints, per combination, the HIGH tier on dev and on
validation: corrected / wrong / false and precision / recall - so a setting that only looks good on the development set is visible.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from v4_lib import load_clips, score_high  # noqa: E402

from src.medical_knowledge.runtime.engine import V4Engine  # noqa: E402
from src.medical_knowledge.runtime.gate import GateConfig  # noqa: E402
from src.medical_knowledge.runtime.knowledge import MedicalKnowledge  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "benchmarks" / "v4" / "results_local"
DEV = [("auto", RES / "corpus_runs" / "auto"), ("nohot", RES / "corpus_runs" / "nohot")]
VAL = [("auto", RES / "runs_validation" / "auto"), ("nohot", RES / "runs_validation" / "nohot")]


def fmt(s: dict) -> str:
    p = "  -  " if s["precision"] is None else f"{100 * s['precision']:5.1f}"
    r = "  -  " if s["medical_recall"] is None else f"{100 * s['medical_recall']:5.1f}"
    return f"{s['corrected']:3}/{s['wrong']}/{s['false']} P{p} R{r}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", type=Path, required=True)
    ap.add_argument("--grid", action="append", default=[], help="field=v1,v2,v3")
    ap.add_argument("--base", action="append", default=[], help="field=value fixed for all combinations")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    engine = V4Engine(MedicalKnowledge.open(a.kb))
    types = {f.name: f.type for f in dataclasses.fields(GateConfig)}

    def cast(name, value):
        return {"int": int, "float": float}.get(types[name] if isinstance(types[name], str) else types[name].__name__, float)(value)

    started = time.time()
    dev = load_clips(engine, "dev", DEV)
    val = load_clips(engine, "validation", VAL)
    print(f"prepared {len(dev)} dev and {len(val)} validation clips in {time.time() - started:.0f} s", flush=True)
    base = {k: cast(k, v) for k, v in (b.split("=", 1) for b in a.base)}
    grid = {}
    for g in a.grid:
        key, values = g.split("=", 1)
        grid[key] = [cast(key, v) for v in values.split(",")]
    rows = []
    for combo in itertools.product(*grid.values()) if grid else [()]:
        cfg = dataclasses.replace(GateConfig(), **base, **dict(zip(grid.keys(), combo)))
        d, v = score_high(dev, engine, cfg), score_high(val, engine, cfg)
        label = " ".join(f"{k}={x}" for k, x in zip(grid.keys(), combo)) or "(defaults)"
        print(f"{label:70} | dev {fmt(d)} | val {fmt(v)}", flush=True)
        rows.append({"settings": dict(zip(grid.keys(), combo)), "dev": d, "validation": v})
    if a.out:
        a.out.write_text(json.dumps({"base": base, "rows": rows}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
