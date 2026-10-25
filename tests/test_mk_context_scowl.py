"""v4: the lecture context (exact n-gram hits, topic, windows) and the SCOWL English word list source (real-word protection, licence notice)."""

import pytest

from src.medical.corrector import SegmentView
from src.medical_knowledge.ingest.builder import KnowledgeBuilder
from src.medical_knowledge.runtime.context import LectureContext
from src.medical_knowledge.runtime.knowledge import MedicalKnowledge
from src.medical_knowledge.sources import registry
from src.medical_knowledge.sources.curated import CuratedSource
from src.medical_knowledge.sources.fixture import FixtureSource
from src.medical_knowledge.sources.scowl import ScowlSource


@pytest.fixture(scope="module")
def kb(tmp_path_factory):
    path = tmp_path_factory.mktemp("ctx") / "fixture.db"
    builder = KnowledgeBuilder(path, "fixture", "test")
    for source in (CuratedSource(use_confusables=False), FixtureSource()):
        source.export(builder)
    builder.finalize()
    with MedicalKnowledge.open(path) as handle:
        yield handle


def views(*sentences):
    return [SegmentView(i, i * 5.0, i * 5.0 + 4.5, s, "en", []) for i, s in enumerate(sentences)]


NEURO = ["Today we will talk about neurons and glia in the nervous system.",
         "The Schwann cell wraps the axon in myelin and the node of Ranvier allows saltatory conduction.",
         "Rough endoplasmic reticulum and ribosomes form the Nissl bodies in the soma of the neuron.",
         "Then we go for lunch and the weather is nice outside.",
         "Astrocytes and microglia are the glial cells of the central nervous system."]


def test_exact_terms_are_found_and_plain_segments_have_none(kb):
    ctx = LectureContext(kb, views(*NEURO), subjects=("neuroanatomy",))
    medical = [sum(1 for h in seg.hits if not h.common) for seg in ctx.segments]
    assert medical[1] >= 2 and medical[2] >= 2 and medical[3] == 0
    assert ctx.counts and all(count >= 1 for count in ctx.counts.values())


def test_the_topic_follows_the_lecture_and_the_subject(kb):
    ctx = LectureContext(kb, views(*NEURO), subjects=("neuroanatomy",))
    assert ctx.focus and any(d in ctx.domains for d in ("neuroanatomy", "neuroscience", "histology"))
    assert ctx.topic_weight(["neuroanatomy"]) >= ctx.topic_weight(["pharmacology"])


def test_windows_see_neighbouring_segments_only(kb):
    ctx = LectureContext(kb, views(*NEURO), subjects=("neuroanatomy",), window=1)
    near, far = ctx.window_words(1), ctx.window_words(4)
    assert "myelin" in near and "myelin" not in far
    assert ctx.window_concepts(2) and ctx.window_concepts(3)


def test_slide_terms_the_database_does_not_know_are_kept_aside(kb):
    ctx = LectureContext(kb, views(*NEURO), subjects=("neuroanatomy",), lecture_terms=["Schwann cell", "zzyxqterm"], custom_terms=["Nissl bodies"])
    assert ("zzyxqterm", "lecture") in ctx.unmatched_lecture_terms
    assert any(kind == "lecture" for kind in ctx.lecture_concepts.values()) or any(kind == "custom" for kind in ctx.lecture_concepts.values())


# ------------------------------------------------------------------------------------------------------------------ SCOWL
@pytest.fixture()
def scowl_dir(tmp_path):
    final = tmp_path / "scowl" / "final"
    final.mkdir(parents=True)
    (tmp_path / "scowl" / "VERSION").write_text("2020.12.07\n", encoding="utf-8")
    (final / "english-words.10").write_text("the\nbetween\nsignal\nsignals\n", encoding="latin-1")
    (final / "english-words.70").write_text("formation\nspinal\nglycerin\n", encoding="latin-1")
    (final / "english-words.80").write_text("glycerins\nfibrocytes\n", encoding="latin-1")
    (final / "english-words.95").write_text("never-included\n", encoding="latin-1")
    (final / "american-words.10").write_text("the\ncolor\nBob\nx\n", encoding="latin-1")
    return tmp_path / "scowl"


def test_scowl_reads_the_lowest_level_of_each_word_and_leaves_names_and_the_largest_list_out(scowl_dir):
    source = ScowlSource(scowl_dir)
    assert source.verify() == [] and source.info.source_version == "2020.12.07" and source.info.redistribution_allowed
    words = {r["word"]: r["level"] for r in source.parse()}
    assert words["the"] == 10 and words["glycerins"] == 80 and words["spinal"] == 70 and words["color"] == 10
    assert "never-included" not in words and "bob" in words and "x" not in words              # 1-letter words dropped; single-case names are lower-cased words
    assert max(words.values()) <= 80


def test_scowl_missing_files_are_reported_not_raised(tmp_path):
    assert any("not found" in problem for problem in ScowlSource(tmp_path / "nowhere").verify())


def test_scowl_words_reach_the_database_as_real_words_and_carry_the_notice(scowl_dir, tmp_path):
    builder = KnowledgeBuilder(tmp_path / "w.db", "public", "test")
    ScowlSource(scowl_dir).export(builder)
    CuratedSource().export(builder)
    builder.finalize()
    with MedicalKnowledge.open(tmp_path / "w.db") as kb:
        assert kb.is_english_word("spinal", 70) and kb.is_english_word("glycerins", 80) and not kb.is_english_word("glycerins", 70)
        assert not kb.is_english_word("dentitic", 80)
        assert any(s["source_id"] == "SCOWL" and s["redistribution_allowed"] for s in kb.info()["sources"])
    assert (registry.Path(__file__).resolve().parent.parent / "src" / "medical_knowledge" / "licenses" / "SCOWL_COPYRIGHT.txt").is_file()
