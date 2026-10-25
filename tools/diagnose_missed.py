"""Why did v4 miss (or wrongly change) a medical term of the corpus? Prints, for every missed region, the decisions v4 took on the spans that overlap the heard words.

    python tools/diagnose_missed.py --runs benchmarks/v4/results_local/corpus_runs/auto --kb data/kb_open_b.db [--clips histo1_clean,embryo4_hard] [--kind missed]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from eval_corpus import apply_tiers, load_passages, segments  # noqa: E402
from src.medical_knowledge.benchmarks.evaluate import evaluate  # noqa: E402
from src.medical_knowledge.runtime.engine import V4Engine  # noqa: E402
from src.medical_knowledge.runtime.knowledge import MedicalKnowledge  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--kb", type=Path, required=True)
    ap.add_argument("--clips", default="")
    ap.add_argument("--kind", default="missed")
    a = ap.parse_args(argv)
    kb = MedicalKnowledge.open(a.kb)
    engine = V4Engine(kb)
    passages = load_passages()
    wanted = {c for c in a.clips.split(",") if c}
    for folder in sorted(a.runs.iterdir()):
        if not (folder / "local_transcript.json").is_file() or (wanted and folder.name not in wanted):
            continue
        pid, _ = folder.name.rsplit("_", 1)
        p = passages.get(pid)
        if not p:
            continue
        segs = segments(folder)
        meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8")) if (folder / "metadata.json").is_file() else {}
        subjects = tuple(s for s in ((meta.get("subject") or {}).get("subjects") or ()) if s != "general_medical")
        res = engine.run(segs, subjects=subjects)
        raw = (folder / "local_raw_transcript.txt").read_text(encoding="utf-8")
        ev = evaluate(p["text"], raw, " ".join(apply_tiers([s.text for s in segs], res, False)), p.get("terms", []))
        for r in ev.regions:
            if r.kind != a.kind or not r.medical:
                continue
            heard = set(r.raw.lower().split())
            print(f"== {folder.name}: gold={r.gold!r} heard={r.raw!r} engine={r.engine!r} term={r.term!r}")
            shown = 0
            for d in res.decisions:
                words = set(d.span.text.lower().replace("-", " ").split())
                if heard & words and d.best is not None:
                    sig = {k: round(v, 2) for k, v in d.best.signals.items() if k in ("phonetic", "lexical", "context", "domain", "lecture", "oddness")}
                    print(f"     span {d.span.text!r:30} {d.tier:6} -> {d.best.candidate.surface!r:30} score {d.best.score:.2f} {sig} {d.blocked_by[:2]}")
                    shown += 1
                    if shown >= 4:
                        break
            if not shown:
                print("     (no span of this word was examined: not suspicious, or inside a known term)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
