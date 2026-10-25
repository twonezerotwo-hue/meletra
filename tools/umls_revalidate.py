"""After a licensed UMLS import: does the bigger vocabulary keep the engine safe?

    python tools/umls_revalidate.py --public data/medical_knowledge_public.db --full data/medical_knowledge_full.db \
        --set dev=benchmarks/v4/results_local/corpus_runs/auto --set validation=benchmarks/v4/results_local/runs_validation/auto \
        --set heldout_v2=benchmarks/v4/results_local/runs_heldout_v2/auto --set heldout_v3=benchmarks/v4/results_local/runs_heldout_v3/auto [--json out.json]

Every corpus is scored twice, with the public pack and with the full database (UMLS included), by the same code. The rule the project set before any scale-up:

    HIGH precision of the full database >= 99 %, and not lower than the public pack's by more than one correction;
    false corrections per 1000 words not higher than the public pack's;
    candidate recall@10 / @25 / @50 not lower (a bigger vocabulary may only find more).

If any corpus violates it the verdict is NOT SAFE and the answer is NOT to enable the full database globally: use domain narrowing, semantic-type filters or specialty indexes (the
runtime already narrows to the lecture's fields) and measure again. This tool only measures; it changes nothing. Without a full database it says so and exits.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MIN_PRECISION = 0.99


def verdict(public: dict, full: dict) -> dict:
    """Compare two ``v4_lib.score`` reports of the same corpus."""
    problems = []
    ph, fh = public["v4_high"], full["v4_high"]
    if fh["precision"] is not None and fh["precision"] < MIN_PRECISION:
        problems.append(f"HIGH precision {fh['precision']:.4f} is below {MIN_PRECISION}")
    if (fh["wrong"] + fh["false"]) > (ph["wrong"] + ph["false"]) + 1:
        problems.append(f"{fh['wrong'] + fh['false']} wrong / false HIGH corrections against {ph['wrong'] + ph['false']} with the public pack")
    if fh["false_per_1000_words"] > ph["false_per_1000_words"]:
        problems.append(f"false corrections per 1000 words rose from {ph['false_per_1000_words']} to {fh['false_per_1000_words']}")
    for k in ("recall@10", "recall@25", "recall@50"):
        a, b = public["retrieval"].get(k), full["retrieval"].get(k)
        if a is not None and b is not None and b + 1e-9 < a:
            problems.append(f"{k} fell from {a} to {b}")
    return {"safe": not problems, "problems": problems, "public": {"high": ph, "retrieval": public["retrieval"]}, "full": {"high": fh, "retrieval": full["retrieval"]}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--public", type=Path, required=True)
    ap.add_argument("--full", type=Path, required=True)
    ap.add_argument("--set", action="append", required=True, help="corpus=run folder, e.g. dev=benchmarks/v4/results_local/corpus_runs/auto")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    if not a.full.is_file():
        print("UMLS REAL IMPORT: BLOCKED - LICENSED UMLS DATA REQUIRED. No full database exists at", a.full, "- see docs/v4/UMLS_OWNER_ACTION.md. Nothing was measured.")
        return 3
    from v4_lib import load_clips, score

    from src.medical_knowledge.runtime.engine import V4Engine
    from src.medical_knowledge.runtime.knowledge import MedicalKnowledge

    results = {}
    for spec in a.set:
        name, folder = spec.split("=", 1)
        reports = {}
        for label, db in (("public", a.public), ("full", a.full)):
            engine = V4Engine(MedicalKnowledge.open(db))
            clips = load_clips(engine, name, [("auto", Path(folder))])
            reports[label] = score(clips, engine, label=f"{name} / {label}")
        results[name] = verdict(reports["public"], reports["full"])
        print(f"{name:12} {'SAFE' if results[name]['safe'] else 'NOT SAFE'}  " + "; ".join(results[name]["problems"]))
    overall = all(r["safe"] for r in results.values())
    print("\nVERDICT:", "SAFE to enable the full database for these corpora" if overall else "NOT SAFE - do not enable the full database globally; narrow by domain / semantic type and measure again")
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps({"overall_safe": overall, "corpora": results}, indent=1), encoding="utf-8")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
