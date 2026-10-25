"""v4 candidate retrieval (milestone 3): indexes, channels, domain pre-ranking, latency. Database = curated vocabulary + fixtures, built once."""

import time

import pytest

from src.medical_knowledge.indexes.phonetic import deletion_eligible, deletions
from src.medical_knowledge.ingest.builder import KnowledgeBuilder
from src.medical_knowledge.runtime.knowledge import MedicalKnowledge
from src.medical_knowledge.runtime.retrieve import CandidateRetriever
from src.medical_knowledge.sources.curated import CuratedSource
from src.medical_knowledge.sources.fixture import FixtureSource


@pytest.fixture(scope="module")
def kb(tmp_path_factory):
    path = tmp_path_factory.mktemp("kb") / "fixture.db"
    builder = KnowledgeBuilder(path, "fixture", "test")
    for source in (CuratedSource(), FixtureSource()):
        source.export(builder)
    builder.finalize()
    with MedicalKnowledge.open(path) as handle:
        yield handle


@pytest.fixture()
def retriever(kb):
    return CandidateRetriever(kb, cache_size=1)


def names(candidates, n=3):
    return [c.surface.lower() for c in candidates[:n]]


def test_the_deletion_neighbourhood_is_the_key_and_every_one_letter_shorter_key():
    assert deletions("nsl") == {"nsl", "sl", "nl", "ns"}
    assert deletion_eligible("nslpt", 2) and not deletion_eligible("ns", 1) and not deletion_eligible("x" * 30, 1) and not deletion_eligible("nslpt", 6)


@pytest.mark.parametrize("heard, domain, expected", [
    ("nisal bodies", "neuroanatomy", "nissl bodies"),
    ("crazy violet", "histology", "cresyl violet"),
    ("tanny sites", "neuroanatomy", "tanycytes"),
    ("exodendritic", "neuroanatomy", "axodendritic"),
    ("exosomatic", "neuroanatomy", "axosomatic"),
    ("rimac fiber", "neuroanatomy", "remak fiber"),
    ("gamma amino butyric ester", "neuroscience", "gamma-aminobutyric acid"),
    ("lumber puncture", "anatomy", "lumbar puncture"),
    ("sell body", "histology", "cell body"),
])
def test_known_mishearings_retrieve_the_right_term_among_the_first_candidates(retriever, heard, domain, expected):
    assert expected in names(retriever.retrieve(heard, [domain]), 4), (heard, names(retriever.retrieve(heard, [domain]), 6))


def test_a_candidate_remembers_which_channels_proposed_it_and_why_it_ranks(retriever):
    top = retriever.retrieve("exodendritic", ["neuroanatomy"])[0]
    assert top.surface == "axodendritic" and {"confusion", "phonetic", "fuzzy"} <= set(top.channels)
    assert top.known_confusion and top.sound == 1.0 and top.domain_fit == 0.6 and not top.exact and top.prior > 1.0       # axodendritic: neuroscience, related to the focus
    exact = retriever.retrieve("Nissl body", ["neuroanatomy"])[0]
    assert exact.exact and "exact" in exact.channels and exact.sound == 1.0 and exact.ortho == 1.0
    assert set(top.to_dict()) >= {"concept_id", "channels", "sound", "ortho", "domain_fit", "prior", "known_confusion"}


def test_the_field_of_study_pre_ranks_but_never_removes_candidates(retriever):
    # "nisal" sounds like Nissl (a histology / neuroanatomy eponym) and like "nasal" (anatomy)
    with_focus = retriever.retrieve("nisal", ["neuroanatomy"])
    without = retriever.retrieve("nisal", [])
    assert names(with_focus, 1) == ["nissl"] and "nasal" in names(with_focus, 5) and "nasal" in names(without, 5)
    assert with_focus[0].domain_fit == 0.5 and without[0].domain_fit == 0.0                 # Nissl is an eponym: neutral; no focus: no fit


def test_the_pool_is_a_few_dozen_candidates_not_the_vocabulary(retriever, kb):
    total = kb.counts()["terms"]
    for span in ("cell", "body", "nervous system", "spinal cord", "synapse", "protein"):
        found = retriever.retrieve(span, ["neuroanatomy"], limit=30)
        assert len(found) <= 30 < total / 100
        assert len({c.concept_id for c in found}) == len(found)                     # one entry per concept


def test_unrelated_text_retrieves_little_and_nothing_crashes(retriever):
    assert retriever.retrieve("") == [] and retriever.retrieve("a") == [] and retriever.retrieve("zzzzqqqq xxxxvvvv") == []
    assert retriever.retrieve("!!!") == []


def test_spacing_hyphens_and_accents_do_not_matter_for_the_exact_channel(retriever):
    for variant in ("gamma-aminobutyric acid", "gamma aminobutyric acid", "gammaaminobutyric acid", "Gamma-Aminobutyric  Acid"):
        assert any(c.exact for c in retriever.retrieve(variant, ["neuroscience"])), variant


def test_retrieval_is_fast_exact_under_10_ms_fuzzy_under_50_ms(retriever, kb):
    spans = ["nisal bodies", "crazy violet", "tanny sites", "exodendritic", "rimac fiber", "sell body", "lumber puncture", "spinal", "between", "signals",
             "peripheral", "formation", "tigroid", "perikaria", "gliomatosis", "acetylcolin"] * 4
    lat = []
    for s in spans:
        t = time.perf_counter()
        retriever.retrieve(s, ["neuroanatomy"])
        lat.append((time.perf_counter() - t) * 1000)
    lat.sort()
    assert lat[len(lat) // 2] < 10 and lat[int(len(lat) * 0.95)] < 50, lat[-5:]
    exact = []
    for term in ("Nissl body", "tanycyte", "saltatory conduction", "GABA", "cresyl violet"):
        kb._terms.clear()
        t = time.perf_counter()
        kb.lookup(term)
        exact.append((time.perf_counter() - t) * 1000)
    assert max(exact) < 10


def test_the_cache_answers_a_repeated_span_without_a_new_search(kb):
    r = CandidateRetriever(kb, cache_size=8)
    r.retrieve("nisal bodies", ["neuroanatomy"])
    r.retrieve("nisal bodies", ["neuroanatomy"])
    assert r._cache.stats()["hits"] == 1 and r._cache.stats()["misses"] == 1
