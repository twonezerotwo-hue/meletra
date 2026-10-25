"""v4 milestone 6: lecture vocabulary (slides / own terms), hotwords, abbreviation resolution."""

import tempfile
from pathlib import Path

import pytest

from src.medical.corrector import SegmentView
from src.medical_knowledge.ingest.builder import KnowledgeBuilder
from src.medical_knowledge.runtime.context import LectureContext
from src.medical_knowledge.runtime.engine import V4Engine
from src.medical_knowledge.runtime.knowledge import MedicalKnowledge
from src.medical_knowledge.runtime.lecture import LectureVocabulary
from src.medical_knowledge.sources.base import AbbreviationRecord, ConceptRecord, SourceInfo, TermRecord, TerminologySource
from src.medical_knowledge.sources.curated import CuratedSource
from src.medical_knowledge.sources.fixture import FixtureSource

NEURO = ["Today we will talk about neurons and glia in the nervous system.",
         "The Schwann cell wraps the axon in myelin and the node of Ranvier allows saltatory conduction.",
         "Rough endoplasmic reticulum and ribosomes form the Nissl bodies in the soma of the neuron.",
         "Astrocytes and microglia and oligodendrocytes are the glial cells of the central nervous system.",
         "The dendrites receive signals and the axon hillock starts the action potential."]


@pytest.fixture(scope="module")
def kb(tmp_path_factory):
    path = tmp_path_factory.mktemp("kb") / "fixture.db"
    builder = KnowledgeBuilder(path, "fixture", "test")
    for source in (CuratedSource(use_confusables=False), FixtureSource()):
        source.export(builder)
    builder.finalize()
    with MedicalKnowledge.open(path) as handle:
        yield handle


def views(*sentences):
    return [SegmentView(i, i * 5.0, i * 5.0 + 4.5, s, "en", []) for i, s in enumerate(sentences)]


def run(sentences, kb_, **kw):
    return V4Engine(kb_).run(views(*sentences), subjects=("neuroanatomy",), **kw)


def test_a_slide_term_the_knowledge_base_does_not_know_is_a_candidate(kb):
    text = NEURO + ["The flurbonine receptors are expressed on the dendrites."]
    without = run(text, kb)
    with_slide = run(text, kb, lecture_terms=["flurbonin receptors"])
    assert all("flurbonin" not in c.replacement.lower() for c in without.corrections + without.uncertain)
    hit = [c for c in with_slide.corrections + with_slide.uncertain if c.replacement == "flurbonin receptors"]
    assert hit and hit[0].sources == ["lecture vocabulary"]


def test_the_users_own_term_is_not_a_blank_cheque(kb):
    result = run(NEURO + ["The sentence mentions a completely different thing about weather."], kb, custom_terms=["Kreutzfeldt"])
    assert not [c for c in result.corrections + result.uncertain if c.replacement == "Kreutzfeldt"]


def test_hotwords_are_a_short_list_of_the_lectures_rare_terms_never_common_words(kb):
    result = run(NEURO, kb, lecture_terms=["tanycytes", "flurbonin receptors"], custom_terms=["Kreutzfeldt"])
    words = result.hotwords
    assert 3 <= len(words) <= 60 and len({w.lower() for w in words}) == len(words)
    assert "Kreutzfeldt" in words[:3] and "flurbonin receptors" in words and any("tanycyte" in w.lower() for w in words)
    assert not [w for w in words if w.lower() in ("cell", "body", "neuron", "axon", "soma")]


def test_hotwords_respect_a_token_budget(kb):
    ctx = LectureContext(kb, views(*NEURO), ["neuroanatomy"], ["tanycytes", "perikaryon", "saltatory conduction", "oligodendrocyte"], ["Kreutzfeldt"])
    vocab = LectureVocabulary(ctx)
    cost = lambda t: len(t) // 3 + 1                                                   # noqa: E731 - stand-in tokenizer
    out = vocab.hotwords(limit=100, token_count=cost, token_budget=24)
    assert sum(cost(w) + 1 for w in out) <= 24 and len(out) < len(vocab.hotwords(limit=100))


def test_slide_terms_known_to_the_knowledge_base_become_boosted_concepts_not_virtual_terms(kb):
    ctx = LectureContext(kb, views(*NEURO), ["neuroanatomy"], ["tanycytes", "oligodendrocyte"], [])
    assert "MLT:tanycyte" in ctx.lecture_concepts and "MLT:oligodendrocyte" in ctx.lecture_concepts
    assert ctx.unmatched_lecture_terms == []


def test_an_abbreviation_defined_in_the_lecture_is_resolved_with_certainty(kb):
    result = run(["The central nervous system (CNS) consists of the brain and the spinal cord.",
                  "Neurons of the CNS are surrounded by glia and the Schwann cell is in the peripheral nervous system."], kb)
    cns = [a for a in result.abbreviations if a["abbreviation"] == "CNS"]
    assert cns and cns[0]["expansion"].lower() == "central nervous system" and cns[0]["reason"] == "defined in the lecture" and not cns[0]["ambiguous"]


class Ambiguous(TerminologySource):
    info = SourceInfo("MLT_FIXTURE", "ambiguous abbreviations", "1")

    def parse(self):
        yield {}

    def normalize(self, raw):
        for cid, long, dom in (("A:ms", "multiple sclerosis", "neurology"), ("A:mass", "mass spectrometry", "biochemistry"), ("A:ms2", "mitral stenosis", "cardiology")):
            yield ConceptRecord(cid, long, "", (dom,))
            yield TermRecord(cid, long, "en", "preferred", True)
            yield AbbreviationRecord("MS", long, cid, dom)


def test_an_ambiguous_abbreviation_is_reported_not_guessed():
    path = Path(tempfile.mkdtemp()) / "amb.db"
    builder = KnowledgeBuilder(path, "fixture")
    Ambiguous().export(builder)
    builder.finalize()
    with MedicalKnowledge.open(path) as small:
        r = V4Engine(small).run(views("The patient has MS and we looked at it today."), subjects=[])
        assert r.abbreviations and r.abbreviations[0]["ambiguous"] and r.abbreviations[0]["expansion"] is None
        r2 = V4Engine(small).run(views("Multiple sclerosis is common.", "The patient has MS and we looked at it today."), subjects=[])
        ms = [a for a in r2.abbreviations if a["abbreviation"] == "MS"]
        assert ms and ms[0]["expansion"] == "multiple sclerosis"                     # said earlier in the lecture: that settles it
