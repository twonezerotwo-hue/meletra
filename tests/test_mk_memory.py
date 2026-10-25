"""Batch memory regression: many lectures one after another in ONE process must not grow without bound (every cache is bounded, per-lecture objects are released)."""

import gc
import json
from pathlib import Path

import pytest

from src.medical.corrector import SegmentView
from src.medical_knowledge.runtime.engine import V4Engine
from src.medical_knowledge.runtime.knowledge import MedicalKnowledge
from tools.bench_retrieval import rss_mb

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "medical_knowledge_public.db"


def lectures(count: int = 8, passages_each: int = 22):
    texts = []
    for name in ("passages.json", "heldout_v2.json", "validation.json"):
        texts += [p["text"] for p in json.loads((ROOT / "benchmarks" / "v4" / "corpus" / name).read_text(encoding="utf-8"))["passages"]]
    for k in range(count):
        chunk = [texts[(k * 7 + i * 3) % len(texts)] for i in range(passages_each)]           # different mixtures of fields
        yield [SegmentView(i, i * 6.0, i * 6.0 + 5.5, text, "en", []) for i, text in enumerate(chunk)]


@pytest.mark.skipif(not DB.is_file(), reason="data/medical_knowledge_public.db is not built")
def test_eight_sequential_lectures_stay_within_the_memory_budget():
    kb = MedicalKnowledge.open(DB)
    engine = V4Engine(kb)
    gc.collect()
    baseline = rss_mb()                                                   # the test process may already hold libraries (Qt ...): measure the GROWTH
    readings = []
    for segs in lectures():
        engine.run(segs)
        gc.collect()
        readings.append(rss_mb())
    assert max(readings) - baseline < 200.0, (baseline, readings)         # the practical batch budget: about 250 MB for the whole process
    assert readings[-1] - readings[2] < 60.0, readings                    # after the caches have filled, a lecture adds (almost) nothing
    stats = kb.cache_stats()
    assert all(s["size"] <= s["max"] for s in stats.values()) and engine.retriever._cache.stats()["size"] <= engine.retriever._cache.stats()["max"]


@pytest.mark.skipif(not DB.is_file(), reason="data/medical_knowledge_public.db is not built")
def test_a_finished_lecture_leaves_no_per_lecture_object_behind():
    import weakref

    kb = MedicalKnowledge.open(DB)
    engine = V4Engine(kb)
    segs = next(lectures(1, 6))
    result = engine.run(segs)
    ref = weakref.ref(result.decisions[0].span) if result.decisions else None
    del result
    gc.collect()
    assert ref is None or ref() is None
