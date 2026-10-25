"""Score the production Medical Expert and v4 (shadow) on the transcribed v4 corpus.

    python tools/eval_corpus.py --runs benchmarks/v4/results_local/corpus_runs/auto --kb data/kb_open_a.db --curated-kb data/medical_knowledge_fixture.db --out benchmarks/v4/results/auto

For every clip: GOLD = the text that was spoken, RAW = the recogniser's text, ENGINES = the production Medical Expert output and the v4 output (HIGH tier applied; and
HIGH + MEDIUM applied as an upper bound). Regions of difference are classified corrected / wrong / false / missed (src/medical_knowledge/benchmarks/evaluate.py);
numbers are counts, nothing is judged by hand. The false-positive corpus (``neg*``: ordinary English) must show ZERO changes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.medical.corrector import SegmentView  # noqa: E402
from src.medical_knowledge.benchmarks.evaluate import Evaluation, evaluate, merge  # noqa: E402
from src.medical_knowledge.runtime.engine import V4Engine, V4Result  # noqa: E402
from src.medical_knowledge.runtime.knowledge import MedicalKnowledge  # noqa: E402

CORPUS = ROOT / "benchmarks" / "v4" / "corpus"


def load_passages(set_name: str = "all") -> dict[str, dict]:
    """id -> passage of one corpus set (dev / validation / heldout_v2) or of all of them."""
    from corpus_sets import passages_of

    return passages_of(set_name)


def segments(folder: Path) -> list[SegmentView]:
    data = json.loads((folder / "local_transcript.json").read_text(encoding="utf-8"))
    return [SegmentView(i, s["start"], s["end"], s["text"], s.get("language"), [(w["word"].strip(), w.get("probability")) for w in s.get("words", [])])
            for i, s in enumerate(data["segments"])]


def production_text(folder: Path) -> str:
    f = folder / "medical_corrected_segments.json"
    if f.is_file():
        return " ".join(s["text"] for s in json.loads(f.read_text(encoding="utf-8")))
    return (folder / "local_raw_transcript.txt").read_text(encoding="utf-8")


def apply_tiers(texts: list[str], result: V4Result, include_medium: bool) -> list[str]:
    items = list(result.corrections) + (list(result.uncertain) if include_medium else [])
    out = list(texts)
    for seg in sorted({c.segment for c in items}):
        text = out[seg]
        for c in sorted((c for c in items if c.segment == seg), key=lambda c: -c.start):
            text = text[:c.start] + c.replacement + text[c.end:]
        out[seg] = text
    return out


def coverage(term: str, curated: MedicalKnowledge, full: MedicalKnowledge) -> str:
    if curated.lookup(term, limit=1):
        return "curated"
    return "open_only" if full.lookup(term, limit=1) else "not_in_knowledge_base"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--kb", type=Path, required=True, help="knowledge base v4 uses")
    ap.add_argument("--curated-kb", type=Path, required=True, help="curated-only database (to classify term coverage)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--subjects", choices=["production", "auto"], default="production")
    ap.add_argument("--set", default="all", help="corpus set to score: dev | validation | heldout_v2 | all (every printed number names it)")
    ap.add_argument("--no-confusables-note", action="store_true")
    a = ap.parse_args(argv)
    passages = load_passages(a.set)
    kb = MedicalKnowledge.open(a.kb)
    curated = MedicalKnowledge.open(a.curated_kb)
    engines = ("production", "v4_high", "v4_high_medium")
    per_clip: list[dict] = []
    buckets: dict[tuple, list[Evaluation]] = defaultdict(list)
    cover: dict[tuple, dict] = defaultdict(lambda: {"occurrences": 0, "raw_ok": 0, "engine_ok": 0})
    region_cover: dict[tuple, dict] = defaultdict(lambda: defaultdict(int))
    timing = {"v4_seconds": 0.0, "words": 0}
    engine = V4Engine(kb)
    for folder in sorted(a.runs.iterdir()):
        if not (folder / "local_transcript.json").is_file():
            continue
        pid, cond = folder.name.rsplit("_", 1)
        p = passages.get(pid)
        if p is None:
            continue
        gold = p["text"]
        raw = (folder / "local_raw_transcript.txt").read_text(encoding="utf-8")
        segs = segments(folder)
        subjects = ()
        if a.subjects == "production":
            meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8")) if (folder / "metadata.json").is_file() else {}
            subjects = tuple((meta.get("subject") or {}).get("subjects") or ())
            subjects = tuple(s for s in subjects if s != "general_medical")
        t = time.perf_counter()
        result = engine.run(segs, subjects=subjects)
        timing["v4_seconds"] += time.perf_counter() - t
        timing["words"] += sum(len(s.text.split()) for s in segs)
        texts = {"production": production_text(folder), "v4_high": " ".join(apply_tiers([s.text for s in segs], result, False)),
                 "v4_high_medium": " ".join(apply_tiers([s.text for s in segs], result, True))}
        terms = p.get("terms", [])
        row = {"clip": folder.name, "subject": p["subject"], "condition": cond}
        for name in engines:
            ev = evaluate(gold, raw, texts[name], terms)
            buckets[(name, p["subject"], cond)].append(ev)
            buckets[(name, "ALL", cond)].append(ev)
            buckets[(name, "ALL", "ALL")].append(ev)
            row[name] = ev.summary() | {"regions": [{"kind": r.kind, "medical": r.medical, "gold": r.gold, "raw": r.raw, "engine": r.engine, "term": r.term} for r in ev.regions]}
            for d in ev.term_details:
                c = coverage(d["term"], curated, kb)
                key = (name, c)
                cover[key]["occurrences"] += 1
                cover[key]["raw_ok"] += d["raw_ok"]
                cover[key]["engine_ok"] += d["engine_ok"]
            for r in ev.regions:
                if r.medical:
                    region_cover[(name, coverage(r.term, curated, kb))][r.kind] += 1
        row["v4_corrections"] = [{"original": c.original, "replacement": c.replacement, "confidence": round(c.confidence, 3)} for c in result.corrections]
        per_clip.append(row)
    summary = {f"{n}|{s}|{c}": merge(evs).summary() for (n, s, c), evs in sorted(buckets.items())}
    out = {"corpus_set": a.set, "runs": str(a.runs), "kb": str(a.kb), "clips": len(per_clip), "v4_seconds_per_1000_words": round(1000 * timing["v4_seconds"] / max(1, timing["words"]), 3),
           "summary": summary,
           "term_coverage": {f"{n}|{c}": v | {"raw_recall": round(v["raw_ok"] / v["occurrences"], 4), "engine_recall": round(v["engine_ok"] / v["occurrences"], 4)}
                             for (n, c), v in sorted(cover.items())},
           "error_regions_by_coverage": {f"{n}|{c}": dict(v) for (n, c), v in sorted(region_cover.items())}}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    (a.out / "per_clip.json").write_text(json.dumps(per_clip, indent=1, ensure_ascii=False), encoding="utf-8")
    print_report(out)
    return 0


def print_report(out: dict) -> None:
    print(f"{out['clips']} clips; v4 {out['v4_seconds_per_1000_words']} s per 1000 words")
    print(f"{'engine':16} {'set':26} {'corr':>5} {'wrong':>5} {'false':>5} {'missed':>6} {'prec':>6} {'med.recall':>10} {'false/1k':>8} {'termR raw->eng':>16}")
    for key, s in out["summary"].items():
        n, subject, cond = key.split("|")
        if subject != "ALL":
            continue
        tr = f"{s['term_recall_raw']}->{s['term_recall_engine']}" if s["terms_total"] else "-"
        print(f"{n:16} {subject + '/' + cond:26} {s['corrected']:>5} {s['wrong']:>5} {s['false']:>5} {s['missed']:>6} {str(s['precision']):>6} {str(s['medical_recall']):>10} {s['false_per_1000_words']:>8} {tr:>16}")


if __name__ == "__main__":
    sys.exit(main())
