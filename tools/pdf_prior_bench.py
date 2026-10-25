"""Does a lecture PDF / slide list help, and can it hurt?  WITH vs WITHOUT slide terms on the same transcripts.

    python tools/pdf_prior_bench.py --set validation --runs auto=... --runs nohot=... --kb data/medical_knowledge_public.db

Variants (the slide terms are given to the engine as ``lecture_terms``, exactly as the PDF reader does):
    without        no slides
    slides         the passage's own medical terms (an oracle: the best case a PDF can be)
    slides+noise   the same terms plus three times as many terms of OTHER passages (a real deck mentions much that is not said in this lecture)
A PDF only raises the retrieval priority and counts as one family of evidence; it can never force a replacement - the numbers show whether that holds.
"""

from __future__ import annotations

import argparse
import dataclasses
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from v4_lib import load_clips, score  # noqa: E402

from src.medical_knowledge.runtime.engine import V4Engine  # noqa: E402
from src.medical_knowledge.runtime.knowledge import MedicalKnowledge  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=["dev", "validation", "heldout_v2"])
    ap.add_argument("--runs", action="append", required=True)
    ap.add_argument("--kb", type=Path, required=True)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    engine = V4Engine(MedicalKnowledge.open(a.kb))
    clips = load_clips(engine, a.set, [(s.split("=", 1)[0], Path(s.split("=", 1)[1])) for s in a.runs])
    pool = sorted({t for c in clips for t in c.passage.get("terms", [])})
    rng = random.Random(7)
    out = {}
    for name in ("without", "slides", "slides+noise"):
        variant = []
        for c in clips:
            own = list(c.passage.get("terms", []))
            terms = [] if name == "without" else own + (rng.sample([t for t in pool if t not in own], min(len(pool) - len(own), 3 * max(1, len(own)))) if name == "slides+noise" and own else [])
            variant.append(dataclasses.replace(c, prepared=engine.prepare(c.segs, c.subjects, lecture_terms=terms)) if terms else c)
        rep = score(variant, engine, label=f"{a.set} / {name}")
        out[name] = rep
        h, r = rep["v4_high"], rep["retrieval"]
        print(f"{name:13} HIGH corrected {h['corrected']:3} wrong {h['wrong']} false {h['false']} precision {h['precision']} recall {h['medical_recall']} | recall@10 {r['recall@10']} @25 {r['recall@25']}")
    if a.json:
        import json
        a.json.write_text(json.dumps({k: {"v4_high": v["v4_high"], "retrieval": v["retrieval"], "v4_medium_only": v["v4_medium_only"]} for k, v in out.items()}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
