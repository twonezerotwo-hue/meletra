"""Measure the Medical Knowledge runtime on a database: start-up, exact lookup, candidate retrieval, memory.

    python tools/bench_retrieval.py --db data/medical_knowledge_fixture.db [--queries 400] [--json out.json]

Query sets: ``exact`` = real terms of the database (every query a different term, uncached); ``fuzzy`` = the same terms with their spelling damaged the way an
ASR damages them (one letter dropped / swapped / doubled, a word split); ``miss`` = ordinary English words (what most transcript spans are).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def rss_mb() -> float:
    """Resident memory of this process in MB (psutil if installed, else the Windows / Unix system call)."""
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / 1e6
    except ImportError:
        pass
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        c = Counters()
        c.cb = ctypes.sizeof(c)
        ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return c.WorkingSetSize / 1e6
    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / (1e6 if sys.platform == "darwin" else 1e3)


def damage(text: str, rnd: random.Random) -> str:
    if len(text) < 5:
        return text
    i = rnd.randrange(1, len(text) - 1)
    op = rnd.choice(("drop", "swap", "double", "split"))
    if op == "drop":
        return text[:i] + text[i + 1:]
    if op == "swap" and text[i] != text[i + 1]:
        return text[:i] + text[i + 1] + text[i] + text[i + 2:]
    if op == "double":
        return text[:i] + text[i] + text[i:]
    return text[:i] + " " + text[i:]


COMMON = ("between signals formation body cell process central peripheral spinal during because therefore important structure function number system example "
          "different another through however together around without whether almost always second answer example question remember today lecture slide").split()


def pct(values: list[float], p: float) -> float:
    values = sorted(values)
    return round(values[min(len(values) - 1, int(len(values) * p))], 3)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, required=True)
    ap.add_argument("--queries", type=int, default=400)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    from src.medical_knowledge.runtime.knowledge import MedicalKnowledge
    from src.medical_knowledge.runtime.retrieve import CandidateRetriever

    rss0 = rss_mb()
    t0 = time.perf_counter()
    kb = MedicalKnowledge.open(a.db)
    kb.lookup("nissl body")
    startup_ms = (time.perf_counter() - t0) * 1000
    rnd = random.Random(5)
    rows = kb.db.execute("SELECT text FROM term WHERE token_count <= 3 ORDER BY term_id LIMIT 1 OFFSET ?", (0,)).fetchall()
    total = kb.db.execute("SELECT COUNT(*) FROM term").fetchone()[0]
    sample = [kb.db.execute("SELECT text FROM term WHERE term_id = ?", (rnd.randrange(1, total + 1),)).fetchone() for _ in range(a.queries * 2)]
    texts = [r[0] for r in sample if r and len(r[0]) >= 5][:a.queries]
    retr = CandidateRetriever(kb, cache_size=1)
    out: dict = {"db": str(a.db), "size_mb": round(a.db.stat().st_size / 1e6, 1), "terms": total, "startup_ms": round(startup_ms, 1)}

    def run(name, items, fn):
        lat = []
        for item in items:
            t = time.perf_counter()
            fn(item)
            lat.append((time.perf_counter() - t) * 1000)
        out[name] = {"n": len(lat), "p50_ms": pct(lat, 0.5), "p95_ms": pct(lat, 0.95), "max_ms": round(max(lat), 2)}

    run("exact_lookup", texts, lambda s: (kb._terms.clear(), kb.lookup(s)))
    run("retrieval_clean_span", texts, lambda s: retr.retrieve(s, ["anatomy"]))
    run("retrieval_damaged_span", [damage(s, rnd) for s in texts], lambda s: retr.retrieve(s, ["anatomy"]))
    run("retrieval_common_word", COMMON * (a.queries // len(COMMON) + 1), lambda s: retr.retrieve(s, ["anatomy"]))
    hits = 0
    pairs = [(s, damage(s, rnd)) for s in texts[:200]]
    for truth, heard in pairs:
        if heard == truth:
            hits += 1
            continue
        got = retr.retrieve(heard, ["anatomy"], limit=20)
        hits += any(c.surface.lower() == truth.lower() for c in got)
    out["source_term_in_top20_after_damage"] = round(hits / max(1, len(pairs)), 3)
    out["rss_growth_mb"] = round(rss_mb() - rss0, 1)
    out["rss_mb"] = round(rss_mb(), 1)
    out["cache"] = kb.cache_stats()
    print(json.dumps(out, indent=2))
    if a.json:
        a.json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
