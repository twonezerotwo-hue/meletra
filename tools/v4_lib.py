"""Shared benchmark code of the v4 tools: load transcribed clips once (running the expensive part of the engine once), score any gate configuration quickly.

    clips = load_clips(kb, "dev", [("auto", Path(".../corpus_runs/auto")), ("nohot", Path(".../corpus_runs/nohot"))])
    report = score(clips, engine, GateConfig(...), label="dev")

``report`` has the separate metrics the promotion gate needs: production / v4 HIGH / v4 MEDIUM (advisory) precision and recall, false and missed corrections per 1,000 words,
candidate recall@10/25/50, the A/B/C/D failure breakdown and the best-vs-second margin distribution. Every report names the corpus set it was computed on.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from corpus_sets import passages_of  # noqa: E402
from eval_corpus import segments  # noqa: E402
from src.medical_knowledge.benchmarks.analysis import breakdown, diagnose, is_target, margins, recall_at, retrieval_reasons  # noqa: E402
from src.medical_knowledge.benchmarks.evaluate import evaluate, merge  # noqa: E402
from src.medical_knowledge.runtime.engine import Prepared, V4Engine  # noqa: E402
from src.medical_knowledge.runtime.gate import GateConfig  # noqa: E402
from src.medical_knowledge.runtime.knowledge import MedicalKnowledge  # noqa: E402


@dataclass
class Clip:
    name: str
    setup: str
    condition: str
    passage: dict
    gold: str
    raw: str
    segs: list
    texts: list
    prepared: Prepared
    production_text: str
    subjects: tuple = ()
    production_texts: tuple = ()                 # the production Medical Expert's text per segment (same segmentation as ``texts``)


def _production_texts(folder: Path) -> tuple:
    f = folder / "medical_corrected_segments.json"
    return tuple(s["text"] for s in json.loads(f.read_text(encoding="utf-8"))) if f.is_file() else ()


def _production_text(folder: Path, raw: str) -> str:
    f = folder / "medical_corrected_segments.json"
    if f.is_file():
        return " ".join(s["text"] for s in json.loads(f.read_text(encoding="utf-8")))
    return raw


def load_clips(engine: V4Engine, set_name: str, run_dirs: Sequence[tuple[str, Path]], limit: int = 0) -> list[Clip]:
    passages = passages_of(set_name)
    clips: list[Clip] = []
    for setup, runs in run_dirs:
        if not runs.is_dir():
            continue
        for folder in sorted(runs.iterdir()):
            if not (folder / "local_transcript.json").is_file():
                continue
            pid, cond = folder.name.rsplit("_", 1)
            passage = passages.get(pid)
            if passage is None:
                continue
            segs = segments(folder)
            meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8")) if (folder / "metadata.json").is_file() else {}
            subjects = tuple(s for s in ((meta.get("subject") or {}).get("subjects") or ()) if s != "general_medical")
            raw = (folder / "local_raw_transcript.txt").read_text(encoding="utf-8")
            prepared = engine.prepare(segs, subjects=subjects)
            clips.append(Clip(folder.name, setup, cond, passage, passage["text"], raw, segs, [s.text for s in segs], prepared, _production_text(folder, raw), subjects,
                              _production_texts(folder)))
            if limit and len(clips) >= limit:
                return clips
    return clips


def apply_items(texts: Sequence[str], items) -> list[str]:
    out = list(texts)
    for seg in sorted({c.segment for c in items}):
        text = out[seg]
        for c in sorted((c for c in items if c.segment == seg), key=lambda c: -c.start):
            text = text[:c.start] + c.replacement + text[c.end:]
        out[seg] = text
    return out


def overlay(raw_texts: Sequence[str], production_texts: Sequence[str], items) -> list[str]:
    """The COMBINED system: the production Medical Expert's text plus every v4 HIGH correction that sits where production changed nothing. On a conflict production wins.

    A v4 correction is addressed in the RAW segment (start, end); production changed some stretches of that segment. The two are aligned character by character: a v4 span that lies
    entirely inside an unchanged stretch is mapped into the production text and replaced there; a span that touches a production change is dropped (production decided there).
    """
    from difflib import SequenceMatcher

    out = list(production_texts) if len(production_texts) == len(raw_texts) else list(raw_texts)
    if len(production_texts) != len(raw_texts):
        return out
    for seg in sorted({c.segment for c in items}):
        raw, prod = raw_texts[seg], out[seg]
        ops = SequenceMatcher(None, raw, prod, autojunk=False).get_opcodes()
        edits = []
        for c in items:
            if c.segment != seg:
                continue
            inside = [op for op in ops if op[1] < c.end and c.start < op[2] or (op[1] == op[2] and c.start < op[1] < c.end)]
            if not inside or any(op[0] != "equal" for op in inside):
                continue
            first, last = inside[0], inside[-1]
            start = first[3] + (c.start - first[1])
            end = last[3] + (c.end - last[1])
            if 0 <= start <= end <= len(prod):
                edits.append((start, end, c.replacement))
        for start, end, replacement in sorted(edits, reverse=True):
            prod = prod[:start] + replacement + prod[end:]
        out[seg] = prod
    return out


def _summary(evals) -> dict:
    ev = merge(evals)
    s = ev.summary()
    words = max(1, ev.gold_words)
    s["missed_per_1000_words"] = round(1000 * ev.count("missed") / words, 3)
    s["medical_missed_per_1000_words"] = round(1000 * ev.count("missed", True) / words, 3)
    s["gold_words"] = ev.gold_words
    return s


def _wsummary(evals, weights) -> dict:
    """The same numbers as ``_summary`` with every clip counted ``weight`` times (1 / clips of its base passage): each BASE PASSAGE counts once, however many voices it was spoken in."""
    def cnt(kind, medical=None):
        return sum(w * e.count(kind, medical) for e, w in zip(evals, weights))

    tp, wrong, false, missed = cnt("corrected"), cnt("wrong"), cnt("false"), cnt("missed")
    changes = tp + wrong + false
    med = cnt("corrected", True) + cnt("wrong", True) + cnt("missed", True)
    words = max(1.0, sum(w * e.gold_words for e, w in zip(evals, weights)))
    return {"corrected": round(tp, 2), "wrong": round(wrong, 2), "false": round(false, 2), "missed": round(missed, 2),
            "precision": None if changes == 0 else round(tp / changes, 4), "medical_error_regions": round(med, 2),
            "medical_recall": None if med == 0 else round(cnt("corrected", True) / med, 4), "false_per_1000_words": round(1000 * false / words, 3),
            "missed_per_1000_words": round(1000 * missed / words, 3), "gold_words": round(words, 1)}


def _wrecall(diags, ks=(10, 25, 50)) -> dict:
    med = [d for d in diags if d.region.medical and d.region.kind in ("missed", "wrong", "corrected")]
    kb_med = [d for d in med if d.in_kb or d.region.kind == "corrected"]
    total, total_kb = sum(d.weight for d in med), sum(d.weight for d in kb_med)
    out = {"medical_error_regions": round(total, 2), "with_term_in_knowledge_base": round(total_kb, 2)}
    for k in ks:
        hit = sum(d.weight for d in med if d.region.kind == "corrected" or (d.rank is not None and d.rank <= k))
        hit_kb = sum(d.weight for d in kb_med if d.region.kind == "corrected" or (d.rank is not None and d.rank <= k))
        out[f"recall@{k}"] = None if not total else round(hit / total, 4)
        out[f"recall@{k}_in_kb"] = None if not total_kb else round(hit_kb / total_kb, 4)
    return out


def term_available(kb, term: str, gold: str = "") -> bool:
    """The knowledge base can produce the answer: the whole term is a term, or EVERY word of the heard region's gold text is a word of the vocabulary (the token channel)."""
    from src.medical_knowledge.normalize.text import normalize

    if kb.lookup(term, limit=1):
        return True
    words = [w for w in normalize(gold or term).split() if len(w) >= 4]
    return bool(words) and all(kb.token_frequency(w) > 0 for w in words)


def explain_filtered(engine, decision, region) -> str:
    """Re-run the retrieval of the span: a channel that found the target and a cut that removed it = FILTERED; no channel at all = NO_INDEX_MATCH."""
    from src.medical_knowledge.normalize.text import compact, normalize

    trace: dict = {}
    engine.retriever.retrieve(decision.span.text, [], 50, extra=decision.span.expected, trace=trace)
    kb = engine.kb
    target_comp = compact(region.term)
    tids = {r[0] for r in kb.db.execute("SELECT term_id FROM term WHERE compact_text = ?", (target_comp,))}
    words = {w for w in normalize(region.gold).split() if len(w) >= 4}
    if any(t in trace.get("channels", {}) for t in tids) or any(w in trace.get("tokens", {}) for w in words):
        return "FILTERED"
    return "NO_INDEX_MATCH"


def score_high(clips: Sequence[Clip], engine: V4Engine, gate: GateConfig) -> dict:
    """Quick scoring for parameter sweeps: only the HIGH tier (corrected / wrong / false / missed), no diagnosis."""
    evals = []
    for clip in clips:
        result = engine.finish(clip.prepared, gate, extras=False)
        evals.append(evaluate(clip.gold, clip.raw, " ".join(apply_items(clip.texts, result.corrections)), clip.passage.get("terms", [])))
    return _summary(evals)


def score(clips: Sequence[Clip], engine: V4Engine, gate: Optional[GateConfig] = None, label: str = "", term_in_kb=None, keep_details: bool = False) -> dict:
    gate = gate or engine.gate
    kb = engine.kb
    in_kb = term_in_kb or (lambda term, gold="": term_available(kb, term, gold))
    explain = lambda d, region: explain_filtered(engine, d, region)                                                                              # noqa: E731
    ev = {"production": [], "high": [], "medium": [], "near": [], "combined": []}
    diags = []
    decisions_all = []
    high_margins = {"correct": [], "wrong": []}
    n_high = n_medium = 0
    details = []
    per_passage: dict[str, int] = {}
    for clip in clips:
        per_passage[clip.passage["id"]] = per_passage.get(clip.passage["id"], 0) + 1
    weights = [1.0 / per_passage[c.passage["id"]] for c in clips]
    for clip, weight in zip(clips, weights):
        result = engine.finish(clip.prepared, gate, extras=False)
        terms = clip.passage.get("terms", [])
        e_prod = evaluate(clip.gold, clip.raw, clip.production_text, terms)
        e_high = evaluate(clip.gold, clip.raw, " ".join(apply_items(clip.texts, result.corrections)), terms)
        e_med = evaluate(clip.gold, clip.raw, " ".join(apply_items(clip.texts, result.uncertain)), terms)
        e_near = evaluate(clip.gold, clip.raw, " ".join(apply_items(clip.texts, [c for c in result.uncertain if c.details.get("near_high")])), terms)
        ev["near"].append(e_near)
        ev["combined"].append(evaluate(clip.gold, clip.raw, " ".join(overlay(clip.texts, clip.production_texts, result.corrections)), terms))
        ev["production"].append(e_prod)
        ev["high"].append(e_high)
        ev["medium"].append(e_med)
        found = diagnose(e_high.regions, result.decisions, clip.texts, in_kb, explain)
        for d in found:
            d.weight = weight
        diags += found
        decisions_all += result.decisions
        n_high += len(result.corrections)
        n_medium += len(result.uncertain)
        if keep_details:
            details.append({"clip": clip.name, "setup": clip.setup, "high": [c.to_dict() for c in result.corrections], "medium": [c.to_dict() for c in result.uncertain],
                            "regions": [{"kind": r.kind, "medical": r.medical, "gold": r.gold, "raw": r.raw, "engine": r.engine, "term": r.term} for r in e_high.regions]})
    out = {"label": label, "clips": len(clips), "gold_words": sum(len(c.gold.split()) for c in clips),
           "production": _summary(ev["production"]), "v4_high": _summary(ev["high"]), "v4_medium_only": _summary(ev["medium"]), "v4_near_high_only": _summary(ev["near"]), "combined_production_plus_v4_high": _summary(ev["combined"]),
           "proposals": {"high": n_high, "medium": n_medium},
           "retrieval": recall_at(diags), "retrieval_reasons": retrieval_reasons(diags), "failure_breakdown": breakdown(diags), "margins_all_decisions": margins(decisions_all)}
    if len(per_passage) < len(clips):
        subjects = sorted({c.passage.get("subject", "") for c in clips})
        out["base_passage"] = {"passages": len(per_passage), "clips": len(clips),
                               "production": _wsummary(ev["production"], weights), "v4_high": _wsummary(ev["high"], weights),
                               "v4_medium_only": _wsummary(ev["medium"], weights), "v4_near_high_only": _wsummary(ev["near"], weights),
                               "combined_production_plus_v4_high": _wsummary(ev["combined"], weights), "retrieval": _wrecall(diags),
                               "by_subject": {subj: {name: _wsummary([e for e, c in zip(ev[key], clips) if c.passage.get("subject", "") == subj],
                                                                     [w for w, c in zip(weights, clips) if c.passage.get("subject", "") == subj])
                                                     for name, key in (("production", "production"), ("v4_high", "high"))} for subj in subjects}}
    high_decisions = [d for d in decisions_all if d.tier == "high"]
    out["margins_high"] = margins(high_decisions)
    if keep_details:
        out["details"] = details
        out["diagnoses"] = [{"clip": None, "kind": d.region.kind, "term": d.region.term, "gold": d.region.gold, "raw": d.region.raw, "failure": d.failure, "rank": d.rank,
                             "in_kb": d.in_kb, "detail": d.detail} for d in diags if d.failure]
    return out


def line(name: str, s: dict) -> str:
    p = "-" if s["precision"] is None else f"{100 * s['precision']:.1f}%"
    r = "-" if s["medical_recall"] is None else f"{100 * s['medical_recall']:.1f}%"
    return (f"{name:16} corrected {s['corrected']:3} wrong {s['wrong']:2} false {s['false']:2} missed {s['missed']:3} | precision {p:>6} recall {r:>6} "
            f"| false/1k {s['false_per_1000_words']:.3f} missed/1k {s['missed_per_1000_words']:.2f}")


def wline(name: str, s: dict) -> str:
    p = "-" if s["precision"] is None else f"{100 * s['precision']:.1f}%"
    r = "-" if s["medical_recall"] is None else f"{100 * s['medical_recall']:.1f}%"
    return (f"{name:16} corrected {s['corrected']:6.1f} wrong {s['wrong']:5.1f} false {s['false']:5.1f} missed {s['missed']:6.1f} | precision {p:>6} recall {r:>6} "
            f"| false/1k {s['false_per_1000_words']:.3f} missed/1k {s['missed_per_1000_words']:.2f}")


def print_report(rep: dict) -> None:
    base = rep.get("base_passage")
    if base:
        print(f"== BASE-PASSAGE metrics (PRIMARY): {rep['label']}  ({base['passages']} base passages, {base['clips']} clips; every passage counts once)")
        for key, name in (("production", "current engine"), ("v4_high", "v4 HIGH"), ("combined_production_plus_v4_high", "COMBINED prod+v4"), ("v4_medium_only", "v4 MEDIUM only"),
                          ("v4_near_high_only", "  of which near-HIGH")):
            print("  " + wline(name, base[key]))
        r = base["retrieval"]
        print(f"  retrieval (base-passage weighted): recall@10 {r.get('recall@10')} @25 {r.get('recall@25')} @50 {r.get('recall@50')}   (term in the knowledge base: "
              f"@10 {r.get('recall@10_in_kb')} @25 {r.get('recall@25_in_kb')} @50 {r.get('recall@50_in_kb')})")
        print("== clip metrics (SECONDARY)")
    print(f"== corpus set: {rep['label']}  ({rep['clips']} clips, {rep['gold_words']} words; v4 proposals: {rep['proposals']['high']} HIGH, {rep['proposals']['medium']} MEDIUM)")
    for key, name in (("production", "current engine"), ("v4_high", "v4 HIGH"), ("combined_production_plus_v4_high", "COMBINED prod+v4"), ("v4_medium_only", "v4 MEDIUM only"),
                      ("v4_near_high_only", "  of which near-HIGH")):
        print("  " + line(name, rep[key]))
    r = rep["retrieval"]
    print(f"  retrieval: recall@10 {r.get('recall@10')} @25 {r.get('recall@25')} @50 {r.get('recall@50')}   (regions with the term in the knowledge base: @10 {r.get('recall@10_in_kb')} "
          f"@25 {r.get('recall@25_in_kb')} @50 {r.get('recall@50_in_kb')}; {r['medical_error_regions']} regions)")
    rr = rep["retrieval_reasons"]
    print(f"  span detection recall {rr['span_detection_recall']}; not retrieved because {rr['not_retrieved_reasons']}; found by {rr['found_by_channel']}")
    b = rep["failure_breakdown"]
    print(f"  failures {b['failures']}: {b['counts']}  share {b['share']}")
    print(f"  margins (best vs second, all decisions): {rep['margins_all_decisions']}")
