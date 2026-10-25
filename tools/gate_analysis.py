"""Which gate rejections were GOOD (conservative for a reason) and which were UNNECESSARILY conservative?

    python tools/gate_analysis.py --kb data/medical_knowledge_public.db

For every decision of every clip of the DEVELOPMENT evidence (dev + validation + heldout_v2, now all development data) the replacement the engine would write is applied ALONE to the
clip and compared with the known text: it is RIGHT (the region becomes the gold text), WRONG (it changes an error into something else) or FALSE (it damages correct speech).
Decisions are then grouped by the reason the gate gave. A bucket whose proposals are right >= 99 % of the time (with enough of them) shows a rejection that was unnecessarily
conservative; a bucket with wrong / false proposals shows a rejection that protected the transcript. The gate is only loosened where a whole bucket is safe.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from v4_lib import ROOT, apply_items, load_clips  # noqa: E402

from src.medical_knowledge.benchmarks.analysis import offsets, overlaps, raw_range, region_range  # noqa: E402
from src.medical_knowledge.benchmarks.evaluate import evaluate  # noqa: E402
from src.medical_knowledge.runtime.engine import V4Correction, V4Engine  # noqa: E402
from src.medical_knowledge.runtime.gate import HIGH, LOW, MEDIUM  # noqa: E402
from src.medical_knowledge.runtime.knowledge import MedicalKnowledge  # noqa: E402

RES = ROOT / "benchmarks" / "v4" / "results_local"
SETS = {"dev": [("auto", RES / "corpus_runs" / "auto"), ("nohot", RES / "corpus_runs" / "nohot")],
        "validation": [("auto", RES / "runs_validation" / "auto"), ("nohot", RES / "runs_validation" / "nohot")],
        "heldout_v2": [("auto", RES / "runs_heldout_v2" / "auto"), ("nohot", RES / "runs_heldout_v2" / "nohot")]}


def bucket(d) -> str:
    """The reason of the tier, reduced to a small label: the guards by name, otherwise the set of HIGH conditions that failed."""
    if d.tier == HIGH:
        return "HIGH"
    reason = d.reason
    for key, label in (("plain English", "guard: plain English words"), ("sound too short", "guard: sound too short"), ("one word is not replaced", "guard: one word -> 3-word term"),
                       ("same words", "guard: same words / number"), ("incomplete mention", "guard: incomplete mention"), ("real word would be dropped", "guard: real word dropped"),
                       ("inside the word", "guard: term inside word"), ("beginning of the term", "guard: beginning of term"), ("different ending", "guard: same stem, other ending"),
                       ("contains '", "guard: contained word"), ("changed word sounds nothing", "guard: changed word"), ("already named", "guard: sibling"),
                       ("rare) dictionary", "guard: rare dictionary word"), ("ambiguous abbreviation", "guard: abbreviation"), ("not verified", "guard: unverified number form"),
                       ("malformed", "guard: malformed phrase"), ("sound 0.", "guard: weak sound"), ("spelling 0.", "guard: weak spelling")):
        if key in reason:
            return label
    parts = []
    for key, label in (("sound + spelling", "form"), ("independent evidence", "families"), ("adjusted score", "score"), ("margin over keeping", "keep-margin"), ("margin to second best", "second-best margin"),
                       ("collision", "collision"), ("gene", "gene")):
        if key in reason:
            parts.append(label)
    return "fails: " + (" + ".join(parts) if parts else "other")


def judge(clip, texts, offs, d, engine, ctx) -> str:
    """right / wrong / false for the replacement of decision ``d`` applied alone."""
    best = d.best
    seg = ctx.segments[d.span.segment].view
    sentence_start = d.span.start == 0 or seg.text[:d.span.start].rstrip().endswith((".", "!", "?"))
    surface, _status = engine.render(d.span, best.candidate.surface, seg.text, sentence_start)
    item = V4Correction(d.span.segment, seg.start, d.span.text, surface, 0.0, "x", best.candidate.concept_id, "", d.span.start, d.span.end, {}, [], [], [], "")
    ev = evaluate(clip.gold, clip.raw, " ".join(apply_items(texts, [item])), clip.passage.get("terms", []))
    rng = raw_range(d.span.segment, d.span.start, d.span.end, texts, offs)
    for r in ev.regions:
        if overlaps(rng, region_range(r)):
            return {"corrected": "right", "wrong": "wrong", "false": "false"}.get(r.kind, "other")
    return "no-op"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", type=Path, required=True)
    ap.add_argument("--sets", default="dev,validation,heldout_v2")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    engine = V4Engine(MedicalKnowledge.open(a.kb))
    table: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    seen: set[tuple] = set()
    for name in a.sets.split(","):
        for clip in load_clips(engine, name, SETS[name]):
            result = engine.finish(clip.prepared, None, extras=False)
            offs = offsets(clip.texts)
            for d in result.decisions:
                if d.best is None:
                    continue
                key = (name, clip.passage["id"], d.span.segment, d.span.start, d.span.end, d.best.candidate.surface)
                if key in seen:                                                    # the same span of the same passage in another variant of the audio: count it once
                    continue
                seen.add(key)
                truth = judge(clip, clip.texts, offs, d, engine, clip.prepared.ctx)
                table[bucket(d)][truth] += 1
    rows = []
    print(f"{'bucket':44} {'n':>5} {'right':>6} {'wrong':>6} {'false':>6} {'no-op':>6}  precision")
    for label, c in sorted(table.items(), key=lambda kv: -sum(kv[1].values())):
        n = sum(c.values())
        decided = c["right"] + c["wrong"] + c["false"]
        prec = c["right"] / decided if decided else None
        rows.append({"bucket": label, "n": n, **dict(c), "precision": prec})
        print(f"{label:44} {n:5} {c['right']:6} {c['wrong']:6} {c['false']:6} {c['no-op']:6}  {'-' if prec is None else f'{100 * prec:.1f}%'}")
    if a.json:
        a.json.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
