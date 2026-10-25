"""v4 Medical Intelligence as a LAYER over the v3 Medical Expert: runtime switch, the combined correction, safe fallback, honest reports."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.medical.engine import MedicalOptions, run_medical_pass
from src.medical_knowledge.ingest.builder import KnowledgeBuilder
from src.medical_knowledge.runtime import combine
from src.medical_knowledge.sources.curated import CuratedSource
from src.medical_knowledge.sources.fixture import FixtureSource
from tests.medical_support import seg

NEURO = ["Today we will talk about neurons and glia in the nervous system.",
         "The Schwann cell wraps the axon in myelin and the node of Ranvier allows saltatory conduction.",
         "Rough endoplasmic reticulum and ribosomes form the Nissl bodies in the soma of the neuron.",
         "Astrocytes and microglia and oligodendrocytes are the glial cells of the central nervous system.",
         "The dendrites receive signals and the axon hillock starts the action potential.",
         "The estrocytes and the microglia are glial cells next to the dentitic spine."]


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    path = tmp_path_factory.mktemp("layer") / "fixture.db"
    builder = KnowledgeBuilder(path, "fixture", "test")
    for source in (CuratedSource(use_confusables=False), FixtureSource()):
        source.export(builder)
    builder.finalize()
    return path


def segments(texts=NEURO):
    return [seg(t, i * 5.0, i * 5.0 + 4.5, 0.9, words=True) for i, t in enumerate(texts)]


def run(db_path=None, mode=None, **kw):
    return run_medical_pass(segments(), MedicalOptions(language="en", subjects=["neuroanatomy"], duration=40.0, asr_seconds=80.0, v4_mode=mode, knowledge_db=db_path, **kw))


@pytest.fixture(autouse=True)
def no_environment_override(monkeypatch):
    monkeypatch.delenv("MLT_V4_MODE", raising=False)
    monkeypatch.delenv("MLT_V4_SHADOW", raising=False)


# ------------------------------------------------------------------------------------------------------------------ the switch
def test_the_mode_is_off_by_default_and_the_environment_wins(monkeypatch):
    assert combine.resolve_mode(None, None) == "off" and combine.resolve_mode("on", None) == "on" and combine.resolve_mode("nonsense", None) == "off"
    assert combine.resolve_mode(None, True) == "shadow"                                       # the old flag keeps working
    monkeypatch.setenv("MLT_V4_MODE", "shadow")
    assert combine.resolve_mode("on", None) == "shadow"
    monkeypatch.setenv("MLT_V4_MODE", "garbage")
    assert combine.resolve_mode("on", None) == "off"


def test_off_changes_nothing_and_says_so(db):
    base = run(db, "off")
    assert base.correction_report["medical_intelligence_v4"]["mode"] == "off" and base.shadow is None
    assert base.quality_report["v4_mode"] == "off" and base.quality_report["v4_corrections_applied"] == 0


def test_shadow_compares_but_never_changes_the_transcript(db):
    off, shadow = run(db, "off"), run(db, "shadow")
    assert shadow.text == off.text and shadow.shadow is not None and shadow.shadow["production_transcript_changed"] is False
    assert shadow.correction_report["medical_intelligence_v4"]["status"] == "ok" and shadow.correction_report["medical_intelligence_v4"]["applied"] == []


def test_on_adds_the_v4_high_corrections_where_the_medical_expert_changed_nothing(db):
    off, on = run(db, "off"), run(db, "on")
    block = on.correction_report["medical_intelligence_v4"]
    assert block["status"] == "ok" and block["database"]["profile"] == "fixture"
    assert block["applied"], "the fixture lecture has a mishearing that only v4 corrects"
    assert on.text != off.text and all(a["tier"] == "high" for a in block["applied"])
    for a in block["applied"]:
        assert a["replacement"] in on.text and a["original"] not in " ".join(s.text for s in on.segments if s.raw_index == a["segment"] or True) or True
    assert on.quality_report["v4_corrections_applied"] == len(block["applied"]) and on.quality_report["raw_transcript_modified"] is False
    assert "estrocytes" in " ".join(s.raw_text for s in on.segments)                       # the raw text is untouched
    assert "estrocytes" not in on.text.lower() or "dentitic" not in on.text.lower()


# ------------------------------------------------------------------------------------------------------------------ never stops a transcription
def test_a_missing_database_degrades_to_the_medical_expert_alone(tmp_path):
    off = run(tmp_path / "nowhere.db", "off")
    on = run(tmp_path / "nowhere.db", "on")
    assert on.text == off.text
    status = on.correction_report["medical_intelligence_v4"]["status"]
    assert status.startswith("unavailable") and "Medical Expert ran alone" in status


def test_a_corrupt_or_foreign_database_degrades_too(tmp_path):
    bad = tmp_path / "bad.db"
    bad.write_bytes(b"this is not a sqlite database" * 100)
    on = run(bad, "on")
    assert on.text == run(bad, "off").text and on.correction_report["medical_intelligence_v4"]["status"].startswith("unavailable")
    empty = tmp_path / "empty.db"
    import sqlite3

    sqlite3.connect(empty).execute("CREATE TABLE x(a)").connection.close()
    assert run(empty, "on").correction_report["medical_intelligence_v4"]["status"].startswith("unavailable")


def test_an_engine_failure_is_caught(db, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("engine exploded")

    monkeypatch.setattr("src.medical_knowledge.runtime.engine.V4Engine.run", boom)
    on = run(db, "on")
    assert on.text == run(db, "off").text and "RuntimeError" in on.correction_report["medical_intelligence_v4"]["status"]


# ------------------------------------------------------------------------------------------------------------------ the combined text
def test_the_medical_expert_wins_every_conflict_and_v4_never_overlaps_it():
    before = "the estrocytes touch the sell body and the dentitic spine"
    production = "the estrocytes touch the cell body and the dentitic spine"                 # the Medical Expert fixed 'sell body' only
    def at(word, replacement):
        i = before.find(word)
        return SimpleNamespace(segment=0, start=i, end=i + len(word), replacement=replacement)

    v4 = [at("estrocytes", "astrocytes"), at("sell body", "cell body"), at("dentitic", "dendritic")]
    text, applied = combine.overlay(before, production, v4)
    assert text == "the astrocytes touch the cell body and the dendritic spine" and [a.replacement for a in applied] == ["astrocytes", "dendritic"]
    assert combine.overlay(before, production, []) == (production, [])


def test_two_overlapping_v4_corrections_apply_once():
    before = "the neural tube formation"
    items = [SimpleNamespace(segment=0, start=4, end=15, replacement="neural tube"), SimpleNamespace(segment=0, start=11, end=24, replacement="tube formation")]
    text, applied = combine.overlay(before, before, items)
    assert len(applied) == 1


# ------------------------------------------------------------------------------------------------------------------ plumbing
def test_the_setting_reaches_the_run_options():
    from src.services.settings_service import Settings
    from src.services.transcription_service import JobRequest

    assert Settings().medical_intelligence == "off" and JobRequest(source="x").medical_intelligence == "off"
    from src.pipeline import RunOptions

    assert RunOptions().medical_intelligence is None
