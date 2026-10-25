"""Measure candidate-retrieval structures on SYNTHETIC medical-looking terms (milestone 3): FTS5 trigram (+bm25), rare-q-gram table, deletion (SymSpell-style) index.

    python tools/bench_retrieval_structures.py --terms 1000000 --queries 300

The terms are random compositions of Greek/Latin roots (realistic length and consonant structure, NOT medical coverage). Recall is measured against a
brute-force scan with the same similarity, so it says how many of the truly closest terms a structure finds. Results are printed as JSON.
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from src.medical.phonetics import skeleton

ROOTS = ("cardi", "neur", "gastr", "hepat", "nephr", "pulmon", "derm", "oste", "myo", "arthr", "cyt", "hist", "angi", "enter", "gloss", "pneum", "thromb", "leuk",
         "erythr", "lip", "gluc", "ren", "cerebr", "spin", "vascul", "thyr", "adren", "gonad", "ocul", "ot", "rhin", "laryng", "bronch", "pancreat", "splen", "lymph",
         "immun", "bacteri", "vir", "fung", "onc", "path", "pharm", "tox", "hem", "chol", "sten", "ather", "isch", "emb", "fibr", "scler", "ectas", "plas", "troph")
LINKS = ("o", "i", "o", "o", "")
ENDS = ("itis", "osis", "oma", "pathy", "ectomy", "plasty", "scopy", "algia", "emia", "uria", "penia", "genesis", "lysis", "stomy", "tomy", "cele", "rrhea", "megaly", "ia", "ic", "al", "ase", "in", "ide", "one", "ol")


def word(rnd: random.Random) -> str:
    parts = rnd.randint(1, 3)
    w = "".join(rnd.choice(ROOTS) + rnd.choice(LINKS) for _ in range(parts))
    return w.rstrip("aeiou") + rnd.choice(ENDS)


def terms(n: int, seed: int) -> list[str]:
    rnd = random.Random(seed)
    out, seen = [], set()
    while len(out) < n:
        w = word(rnd)
        if rnd.random() < 0.35:
            w = rnd.choice(("acute", "chronic", "benign", "primary", "congenital", "left", "right", "hereditary")) + " " + w
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def grams(sk: str) -> list[str]:
    return sorted({sk[i:i + 3] for i in range(len(sk) - 2)})


def deletions(sk: str) -> set[str]:
    return {sk} | {sk[:i] + sk[i + 1:] for i in range(len(sk))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--terms", type=int, default=200000)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--edits", type=int, default=1, help="how many characters of the skeleton are damaged in a query")
    ap.add_argument("--only", default="", help="comma list of: fts,qgram,deletion")
    a = ap.parse_args()
    data = terms(a.terms, a.seed)
    sks = [skeleton(t) for t in data]
    rnd = random.Random(a.seed + 1)
    queries, source = [], []
    for _ in range(a.queries):                                  # a query = a term whose skeleton was damaged the way ASR damages it
        src_i = rnd.randrange(len(sks))
        sk = sks[src_i]
        if len(sk) < 5:
            continue
        q = sk
        for _ in range(a.edits):
            i = rnd.randrange(len(q))
            op = rnd.choice(("del", "sub", "ins"))
            q = q[:i] + q[i + 1:] if op == "del" else q[:i] + rnd.choice("ptknslrfm") + q[i + 1:] if op == "sub" else q[:i] + rnd.choice("ptknslrfm") + q[i:]
        queries.append(q)
        source.append(src_i)
    out: dict = {"terms": len(data), "queries": len(queries)}
    tmp = Path(tempfile.mkdtemp(prefix="retr_"))

    t0 = time.perf_counter()
    hits = 0
    for q, s in zip(queries[:60], source):
        res = process.extract(q, sks, scorer=Levenshtein.normalized_similarity, limit=a.k)
        hits += any(i == s for _, _, i in res)
    out["bruteforce_ms_per_query"] = round((time.perf_counter() - t0) / 60 * 1000, 1)
    out["bruteforce_source_hit"] = round(hits / 60, 3)

    def evaluate(name, build, query):
        db = sqlite3.connect(tmp / f"{name}.db")
        t = time.perf_counter()
        build(db)
        db.commit()
        build_s = time.perf_counter() - t
        db.close()
        db = sqlite3.connect(f"file:{(tmp / (name + '.db')).as_posix()}?mode=ro", uri=True)
        lat, recalls, pool = [], [], []
        for n, q in enumerate(queries):
            t = time.perf_counter()
            cand = query(db, q)
            best = process.extract(q, [sks[i] for i in cand], scorer=Levenshtein.normalized_similarity, limit=a.k) if cand else []
            lat.append((time.perf_counter() - t) * 1000)
            pool.append(len(cand))
            recalls.append(1.0 if cand and any(cand[j] == source[n] for _, _, j in best) else 0.0)
        lat.sort()
        out[name] = {"build_s": round(build_s, 1), "size_mb": round((tmp / f"{name}.db").stat().st_size / 1e6, 1), "p50_ms": round(statistics.median(lat), 2),
                     "p95_ms": round(lat[int(len(lat) * 0.95)], 2), "source_hit_in_top%d" % a.k: round(statistics.mean(recalls), 3) if recalls else None,
                     "pool_median": int(statistics.median(pool))}

    # A: FTS5 trigram over skeletons, bm25
    def build_a(db):
        db.execute("CREATE VIRTUAL TABLE f USING fts5(sk, tokenize='trigram')")
        db.executemany("INSERT INTO f(rowid, sk) VALUES (?,?)", ((i, sk) for i, sk in enumerate(sks) if len(sk) >= 3))

    def query_a(db, q):
        g = grams(q)
        return [r[0] for r in db.execute("SELECT rowid FROM f WHERE f MATCH ? ORDER BY rank LIMIT 200", (" OR ".join('"%s"' % x for x in g),))] if g else []

    # B: q-gram postings, rarest grams first
    def build_b(db):
        db.execute("CREATE TABLE g (gram TEXT, tid INTEGER, PRIMARY KEY (gram, tid)) WITHOUT ROWID")
        db.execute("CREATE TABLE df (gram TEXT PRIMARY KEY, n INTEGER) WITHOUT ROWID")
        db.executemany("INSERT OR IGNORE INTO g VALUES (?,?)", ((x, i) for i, sk in enumerate(sks) for x in grams(sk)))
        db.execute("INSERT INTO df SELECT gram, COUNT(*) FROM g GROUP BY gram")

    def query_b(db, q, rare=4):
        g = grams(q)
        if not g:
            return []
        dfs = {r[0]: r[1] for r in db.execute("SELECT gram, n FROM df WHERE gram IN (%s)" % ",".join("?" * len(g)), g)}
        use = sorted(g, key=lambda x: dfs.get(x, 0))[:rare]
        rows = db.execute("SELECT tid, COUNT(*) c FROM g WHERE gram IN (%s) GROUP BY tid ORDER BY c DESC LIMIT 300" % ",".join("?" * len(use)), use).fetchall()
        return [r[0] for r in rows]

    # C: deletion neighbourhood (edit distance 1 on the skeleton)
    def build_c(db):
        db.execute("CREATE TABLE d (key TEXT, tid INTEGER, PRIMARY KEY (key, tid)) WITHOUT ROWID")
        db.executemany("INSERT OR IGNORE INTO d VALUES (?,?)", ((k, i) for i, sk in enumerate(sks) for k in deletions(sk)))

    def query_c(db, q):
        keys = list(deletions(q))
        rows = db.execute("SELECT DISTINCT tid FROM d WHERE key IN (%s) LIMIT 400" % ",".join("?" * len(keys)), keys).fetchall()
        return [r[0] for r in rows]

    wanted = set(a.only.split(",")) if a.only else {"fts", "qgram", "deletion"}
    if "fts" in wanted:
        evaluate("fts5_trigram_bm25", build_a, query_a)
    if "qgram" in wanted:
        evaluate("qgram_rarest4", build_b, query_b)
    if "deletion" in wanted:
        evaluate("deletion_index", build_c, query_c)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
