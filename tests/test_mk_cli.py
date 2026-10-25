"""v4: --medical-knowledge-info, the self-test check ("fails loudly if a required database is missing or corrupt"), the Settings > Medical Intelligence page,
and the form helpers (number / case of a replacement)."""

import json

import pytest

from src.medical_knowledge import cli
from src.medical_knowledge.ingest.builder import KnowledgeBuilder
from src.medical_knowledge.normalize.forms import match_case, match_number, plural_like, pluralize, singularize, strip_qualifier, trim_neighbours
from src.medical_knowledge.sources.curated import CuratedSource
from src.medical_knowledge.sources.fixture import FixtureSource


@pytest.fixture(scope="module")
def db_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("cli") / "medical_knowledge_public.db"
    builder = KnowledgeBuilder(path, "fixture", "test")
    for source in (CuratedSource(), FixtureSource()):
        source.export(builder)
    builder.finalize()
    return path


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    monkeypatch.delenv(cli.REQUIRED_ENV, raising=False)
    monkeypatch.setenv("MLT_KNOWLEDGE_DB", str(tmp_path / "does_not_exist.db"))      # find_database falls through to the standard folders
    monkeypatch.setattr("src.medical_knowledge.runtime.knowledge.DEFAULT_NAMES", ("no_such_default.db",))


def test_info_of_a_healthy_database(db_path):
    info = cli.report(str(db_path))
    assert info["installed"] and info["ok"] and info["status"] == "ready" and info["name"] == "MedicalKnowledge"
    assert info["counts"]["concepts"] > 0 and info["counts"]["terms"] > 0 and info["sources"] and "en" in info["languages"]
    assert {c["name"] for c in info["checks"]} >= {"schema version", "integrity", "exact index", "full-text index", "phonetic index", "relationships", "license metadata"}
    assert cli.exit_code(info) == 0 and "Concepts" not in cli.as_text(info) and "concepts   :" in cli.as_text(info)


def test_the_command_writes_text_or_json(db_path, tmp_path):
    assert cli.run_command(str(tmp_path / "info.json"), str(db_path)) == 0
    assert json.loads((tmp_path / "info.json").read_text(encoding="utf-8"))["ok"] is True
    assert cli.run_command(str(tmp_path / "info.txt"), str(db_path)) == 0
    assert "status     : ready" in (tmp_path / "info.txt").read_text(encoding="utf-8")


def test_no_database_is_exit_2_unless_the_build_requires_one(monkeypatch, tmp_path):
    info = cli.report()
    assert not info["installed"] and info["ok"] and cli.exit_code(info) == 2
    monkeypatch.setenv(cli.REQUIRED_ENV, "1")
    required = cli.report()
    assert not required["ok"] and "REQUIRED" in required["status"] and cli.exit_code(required) == 1


def test_selftest_skips_when_none_is_installed_and_fails_loudly_when_required(monkeypatch):
    with pytest.raises(cli.NotInstalled):
        cli.selftest_detail()
    monkeypatch.setenv(cli.REQUIRED_ENV, "1")
    with pytest.raises(RuntimeError, match="not usable"):
        cli.selftest_detail()


def test_a_corrupt_or_foreign_file_fails_loudly(tmp_path, monkeypatch):
    bad = tmp_path / "medical_knowledge_public.db"
    bad.write_bytes(b"this is not a sqlite database" * 100)
    info = cli.report(str(bad))
    assert info["installed"] and not info["ok"] and "cannot open" in info["status"] and cli.exit_code(info) == 1
    monkeypatch.setenv("MLT_KNOWLEDGE_DB", str(bad))
    with pytest.raises(RuntimeError, match="not usable"):
        cli.selftest_detail()


def test_selftest_detail_of_a_healthy_database(db_path, monkeypatch):
    monkeypatch.setenv("MLT_KNOWLEDGE_DB", str(db_path))
    detail = cli.selftest_detail()
    assert "MedicalKnowledge" in detail and "checks passed" in detail


def test_the_required_marker_next_to_the_database(db_path, tmp_path):
    assert not cli.is_required(db_path)
    (db_path.parent / cli.REQUIRED_MARKER).write_text("1", encoding="utf-8")
    try:
        assert cli.is_required(db_path)
    finally:
        (db_path.parent / cli.REQUIRED_MARKER).unlink()


# ------------------------------------------------------------------------------------------------------------------ forms
def test_number_follows_the_heard_words():
    assert match_number("exosomatic synapses", "axosomatic synapse") == "axosomatic synapses"
    assert match_number("exosomatic synapse", "axosomatic synapses") == "axosomatic synapse"
    assert match_number("perikarya", "perikaryon") == "perikaryon"                    # a Latin plural is never invented: without a verified form the source's form stays
    assert match_number("perikarya", "perikaryon", None, lambda phrase: phrase == "perikarya") == "perikarya"          # ... and a verified one is used
    assert match_number("acunites", "aconitase") == "aconitase"                    # the "s" belongs to the misheard word
    assert match_number("papes", "Papez", {"Papez"}) == "Papez"                      # a proper name has no plural
    assert plural_like("vertebrae") and plural_like("synapses") and not plural_like("mitosis") and pluralize("axon") == "axons" and singularize("axons") == "axon"


def test_case_follows_the_sentence_and_proper_names_stay_capitalised():
    assert match_case("nerve system", "Nervous System") == "nervous system"
    assert match_case("pyre patches", "Peyer Patches", None, False, {"Peyer"}) == "Peyer patches"
    assert match_case("gaba", "GABA") == "GABA" and match_case("alzheimer disease", "Alzheimer Disease") == "Alzheimer disease"
    assert match_case("Dendritic", "dendritic", None, True) == "Dendritic"


def test_qualifiers_and_neighbour_words_are_not_written_twice():
    assert strip_qualifier("Phosphoenolpyruvate Carboxykinase (GTP)") == "Phosphoenolpyruvate Carboxykinase"
    assert trim_neighbours("Peyer patches", "the ", " patches in the ileum") == "Peyer"
    assert trim_neighbours("Nissl bodies", "the ", " are stained") == "Nissl bodies"
    assert trim_neighbours("axon", "an ", " axon") == "axon"                         # never empties the term
