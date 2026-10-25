"""Milestone 4: which sound-alike key retrieves the true term, and which similarity ranks it first? Evaluated on REAL mishearing pairs.

    python tools/eval_phonetic.py --db data/medical_knowledge_fixture.db --pairs benchmarks/v4/phonetic_pairs.json [--json out.json]

Retrieval (per key): the true term is in the pool of terms with the SAME key / a key at most one letter apart (deletion neighbourhood); pool = how many terms came
with it (smaller is better at equal recall). Ranking: inside the union pool of the skeleton deletion index, where does the true concept rank under each similarity?
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rapidfuzz.distance import Levenshtein

from src.medical.phonetics import skeleton
from src.medical_knowledge.indexes.double_metaphone import dm_key
from src.medical_knowledge.indexes.phonetic import deletions
from src.medical_knowledge.indexes.phonetic_extra import mp_key, msx_key
from src.medical_knowledge.normalize.text import letters, normalize
from src.medical_knowledge.runtime.knowledge import MedicalKnowledge

KEYS = {"sk": skeleton, "dm": dm_key, "mp": mp_key, "msx": msx_key}


def bigrams(s: str) -> set[str]:
    return {s[i:i + 2] for i in range(len(s) - 1)} or {s}


def jaccard(a: str, b: str) -> float:
    x, y = bigrams(a), bigrams(b)
    return len(x & y) / len(x | y) if x or y else 0.0


def sim_lev(a: str, b: str) -> float:
    return Levenshtein.normalized_similarity(a, b) if a and b else 0.0


SIMS = {
    "sk": lambda h, t: sim_lev(skeleton(h), skeleton(t)),
    "dm": lambda h, t: sim_lev(dm_key(h), dm_key(t)),
    "mp": lambda h, t: sim_lev(mp_key(h), mp_key(t)),
    "mp_bigram": lambda h, t: jaccard(mp_key(h), mp_key(t)),
    "ortho": lambda h, t: sim_lev(letters(h), letters(t)),
    "current(0.7sk+0.3ortho)": lambda h, t: 0.7 * sim_lev(skeleton(h), skeleton(t)) + 0.3 * sim_lev(letters(h), letters(t)),
    "blend(.4sk+.3mp+.15dm+.15ortho)": lambda h, t: 0.4 * sim_lev(skeleton(h), skeleton(t)) + 0.3 * sim_lev(mp_key(h), mp_key(t)) + 0.15 * sim_lev(dm_key(h), dm_key(t))
    + 0.15 * sim_lev(letters(h), letters(t)),
    "blend2(.5mp+.3sk+.2ortho)": lambda h, t: 0.5 * sim_lev(mp_key(h), mp_key(t)) + 0.3 * sim_lev(skeleton(h), skeleton(t)) + 0.2 * sim_lev(letters(h), letters(t)),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    kb = MedicalKnowledge.open(a.db)
    pairs = json.loads(a.pairs.read_text(encoding="utf-8"))["pairs"]
    terms = kb.db.execute("SELECT term_id, concept_id, text, token_count FROM term WHERE language IN ('en','la')").fetchall()
    index: dict[str, dict[str, set[int]]] = {name: defaultdict(set) for name in KEYS}
    near: dict[str, dict[str, set[int]]] = {name: defaultdict(set) for name in KEYS}
    by_id = {t[0]: t for t in terms}
    for tid, cid, text, tokens in terms:
        for name, fn in KEYS.items():
            k = fn(text)
            if not k:
                continue
            index[name][k].add(tid)
            if 3 <= len(k) <= 24 and tokens <= 4:
                for d in deletions(k):
                    near[name][d].add(tid)
    out: dict = {"pairs": len(pairs), "terms": len(terms), "retrieval": {}, "ranking": {}, "by_source": {}}
    for name, fn in KEYS.items():
        hit_exact = hit_near = 0
        pools_exact, pools_near = [], []
        for p in pairs:
            hk = fn(p["heard"])
            truth_ids = {tid for tid, cid, _, _ in terms if cid in p["concepts"]} if False else None
            concepts = set(p["concepts"])
            exact = index[name].get(hk, set())
            cand = set()
            for d in deletions(hk) if len(hk) >= 3 else (hk,):
                cand |= near[name].get(d, set())
            hit_exact += any(by_id[t][1] in concepts for t in exact)
            hit_near += any(by_id[t][1] in concepts for t in cand)
            pools_exact.append(len(exact))
            pools_near.append(len(cand))
        out["retrieval"][name] = {"recall_same_key": round(hit_exact / len(pairs), 3), "recall_within_one_letter": round(hit_near / len(pairs), 3),
                                  "pool_same_key_median": int(statistics.median(pools_exact)), "pool_within_one_letter_median": int(statistics.median(pools_near)),
                                  "pool_within_one_letter_p90": sorted(pools_near)[int(len(pools_near) * 0.9)]}
    # ranking inside the skeleton deletion pool (what the engine retrieves)
    ranks: dict[str, list[int]] = {k: [] for k in SIMS}
    for p in pairs:
        hk = skeleton(p["heard"])
        cand: set[int] = set()
        for d in deletions(hk) if len(hk) >= 3 else (hk,):
            cand |= near["sk"].get(d, set())
        concepts = set(p["concepts"])
        if not any(by_id[t][1] in concepts for t in cand):
            continue                                                       # not retrieved: counted in recall, not in ranking
        for name, sim in SIMS.items():
            best: dict[str, float] = {}
            for t in cand:
                cid, text = by_id[t][1], by_id[t][2]
                best[cid] = max(best.get(cid, 0.0), sim(p["heard"], text))
            order = sorted(best, key=lambda c: -best[c])
            ranks[name].append(min(order.index(c) for c in concepts if c in best) + 1)
    for name, rs in ranks.items():
        n = len(rs)
        out["ranking"][name] = {"n": n, "top1": round(sum(r == 1 for r in rs) / n, 3), "top3": round(sum(r <= 3 for r in rs) / n, 3), "top5": round(sum(r <= 5 for r in rs) / n, 3),
                                "mrr": round(sum(1 / r for r in rs) / n, 3)}
    src_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for p in pairs:
        hk = skeleton(p["heard"])
        cand = set()
        for d in deletions(hk) if len(hk) >= 3 else (hk,):
            cand |= near["sk"].get(d, set())
        src_counts[p["source"]][0] += 1
        src_counts[p["source"]][1] += any(by_id[t][1] in set(p["concepts"]) for t in cand)
    out["by_source"] = {k: {"pairs": v[0], "skeleton_deletion_recall": round(v[1] / v[0], 3)} for k, v in src_counts.items()}
    print(json.dumps(out, indent=1))
    if a.json:
        a.json.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
