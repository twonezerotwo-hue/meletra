"""v4 Medical Knowledge: schema, build, provenance, license filter, API, health (milestones 1-2)."""

import json
import sqlite3
from pathlib import Path

import pytest

from src.medical_knowledge.database.store import IncompatibleKnowledgeBase, KnowledgeError, open_readonly
from src.medical_knowledge.ingest.builder import KnowledgeBuilder
from src.medical_knowledge.licenses import policy
from src.medical_knowledge.licenses.policy import LicenseViolation
from src.medical_knowledge.runtime.cache import LRU
from src.medical_knowledge.runtime.knowledge import MedicalKnowledge
from src.medical_knowledge.sources.base import (AbbreviationRecord, ConceptRecord, ConfusionRecord, CueRecord, RelationRecord, SourceInfo, TermRecord,
                                                TerminologySource)


class Mini(TerminologySource):
    def __init__(self, source_id="MLT_FIXTURE", version="9", extra_terms=()):
        self.info = SourceInfo(source_id, f"mini {source_id}", version, "", "PROJECT_OWNED", True)
        self.extra = extra_terms

    def parse(self):
        yield {}

    def normalize(self, raw):
        yield ConceptRecord("T:remak", "Remak bundle", "", ("neuroanatomy",), "", "term", "CURATED")
        yield TermRecord("T:remak", "Remak bundle", "en", "preferred", True, "R1", "R1")
        yield TermRecord("T:remak", "Remak fibers", "en", "synonym", False, "R2", "R1")
        yield TermRecord("T:remak", "Remak-Bündel", "de", "synonym", False, "R3", "R1")
        yield ConceptRecord("T:schwann", "Schwann cell", "Cell", ("neuroanatomy", "histology"), "", "term", "ONTOLOGY_VERIFIED")
        yield TermRecord("T:schwann", "Schwann cell", "en", "preferred", True, "S1", "S1")
        yield ConceptRecord("T:pns", "peripheral nervous system", "", ("neuroanatomy",))
        yield TermRecord("T:pns", "peripheral nervous system", "en", "preferred", True)
        yield TermRecord("T:pns", "PNS", "en", "abbreviation", False)
        yield AbbreviationRecord("PNS", "peripheral nervous system", "T:pns", "neuroanatomy")
        yield RelationRecord("T:remak", "related_to", "T:schwann", 0.9)
        yield RelationRecord("T:schwann", "part_of", "T:pns", 1.0)
        yield CueRecord("T:remak", "unmyelinated")
        yield ConfusionRecord("T:remak", "Rimak")
        for i, text in enumerate(self.extra):
            yield ConceptRecord(f"T:x{i}", text, "", ("pharmacology",))
            yield TermRecord(f"T:x{i}", text, "en", "preferred", True)


def build(tmp_path, sources, profile="fixture", name="kb.db"):
    builder = KnowledgeBuilder(tmp_path / name, profile, "2026.10")
    for s in sources:
        s.export(builder)
    return builder.finalize(), tmp_path / name


def test_build_open_lookup_and_concept(tmp_path):
    manifest, path = build(tmp_path, [Mini()])
    assert manifest["counts"]["concepts"] == 3 and manifest["counts"]["terms"] == 6 and manifest["counts"]["languages"] == {"en": 5, "de": 1}
    with MedicalKnowledge.open(path) as kb:
        assert [h.concept_id for h in kb.lookup("remak bundle")] == ["T:remak"]
        assert kb.lookup("Remak-fibers")[0].text == "Remak fibers"                  # hyphen / space insensitive
        assert kb.lookup("remakbündel")[0].language == "de"                          # accents folded
        assert kb.lookup("Rimak") == []                                              # a known mishearing is not a term
        concept = kb.get_concept("T:remak")
        assert concept.preferred_name == "Remak bundle" and concept.languages == ("de", "en") and concept.sources == ("MLT_FIXTURE",)
        assert [t.text for t in concept.terms if t.is_preferred] == ["Remak bundle"]
        assert {r.preferred_name for r in kb.related("T:remak")} == {"Schwann cell"}
        both = {(r.preferred_name, r.direction) for r in kb.related("T:schwann")}
        assert both == {("peripheral nervous system", "out"), ("Remak bundle", "in")}
        v = kb.validate_term("PNS", "neuroanatomy")
        assert v.known and v.is_abbreviation and v.expansions == ("peripheral nervous system",) and v.names == ("peripheral nervous system",)
        assert not kb.validate_term("flibbertigibbet").known
        assert kb.validate_term("Schwann cell", "histology").in_domain is True
        assert kb.validate_term("Schwann cell", "pharmacology").in_domain is False
        assert [h.text for h in kb.lookup("Remak", prefix=True)][:2] == ["Remak bundle", "Remak fibers"]
        assert kb.lookup("Schwann cell", domains=["pharmacology"]) == []


def test_health_and_manifest(tmp_path):
    manifest, path = build(tmp_path, [Mini()])
    sidecar = json.loads(path.with_suffix(".db.manifest.json").read_text(encoding="utf-8"))
    assert sidecar["sha256"] == manifest["sha256"] and sidecar["build_version"] == "2026.10" and sidecar["sources"][0]["source_id"] == "MLT_FIXTURE"
    with MedicalKnowledge.open(path) as kb:
        checks = kb.health()
        assert all(c.ok for c in checks), [(c.name, c.detail) for c in checks if not c.ok]
        info = kb.info()
        assert info["schema_version"] == 1 and info["build_version"] == "2026.10" and info["languages"]["de"] == 1


def test_every_term_keeps_provenance_and_the_license_of_its_source(tmp_path):
    _, path = build(tmp_path, [Mini()])
    db = sqlite3.connect(path)
    rows = db.execute("SELECT t.text, ts.source_id, ts.source_code, ts.source_concept_id, s.license_status, s.redistribution_allowed FROM term t "
                      "JOIN term_source ts USING (term_id) JOIN source s ON s.source_id = ts.source_id ORDER BY t.text").fetchall()
    assert ("Remak fibers", "MLT_FIXTURE", "R2", "R1", "ALLOW", 1) in rows
    assert db.execute("SELECT COUNT(*) FROM term t WHERE NOT EXISTS (SELECT 1 FROM term_source ts WHERE ts.term_id = t.term_id)").fetchone()[0] == 0


@pytest.mark.parametrize("source_id", ["UMLS", "SNOMEDCT", "HPO", "SYNTHETIC_SCALE", "SOME_NEW_SOURCE"])
def test_a_public_build_refuses_sources_that_are_not_allowed(tmp_path, source_id):
    builder = KnowledgeBuilder(tmp_path / "public.db", "public")
    with pytest.raises(LicenseViolation):
        Mini(source_id).export(builder)
    assert not (tmp_path / "public.db").exists() and not list(tmp_path.glob("*.building*"))      # nothing half-written is left


def test_a_public_build_accepts_the_allowlisted_sources(tmp_path):
    manifest, path = build(tmp_path, [Mini("MLT_CURATED"), Mini("MLT_FIXTURE")], profile="public")
    assert manifest["profile"] == "public" and manifest["counts"]["private_only_terms"] == 0
    with MedicalKnowledge.open(path) as kb:
        assert [c for c in kb.health() if c.name == "public license filter"][0].ok


def test_a_full_build_keeps_private_terms_marked_not_redistributable(tmp_path):
    private = Mini("UMLS", extra_terms=("heparin sodium",))
    manifest, path = build(tmp_path, [Mini("MLT_FIXTURE"), private], profile="full")
    db = sqlite3.connect(path)
    assert db.execute("SELECT redistributable FROM term WHERE text = 'heparin sodium'").fetchone() == (0,)
    assert db.execute("SELECT redistributable FROM term WHERE text = 'Remak bundle'").fetchone() == (1,)          # also delivered by an allowed source
    assert manifest["counts"]["private_only_terms"] == 1


def test_the_public_gate_checks_the_finished_data_not_only_the_importers(tmp_path):
    builder = KnowledgeBuilder(tmp_path / "gate.db", "public")
    Mini("MLT_FIXTURE").export(builder)
    builder.db.execute("UPDATE source SET redistribution_allowed = 0")            # something slipped in
    with pytest.raises(LicenseViolation, match="PUBLIC KNOWLEDGE BUILD FAILS"):
        builder.finalize()
    assert not (tmp_path / "gate.db").exists()


def test_policy_needs_both_the_allow_status_and_the_allowlist():
    assert policy.check_public(["MLT_CURATED"]) == []
    assert any("PRIVATE_ONLY" in p for p in policy.check_public(["UMLS"]))
    assert any("REVIEW_REQUIRED" in p for p in policy.check_public(["HPO"]))
    assert any("REVIEW_REQUIRED" in p for p in policy.check_public(["NO_SUCH_SOURCE"]))
    assert policy.license_of("UMLS").requires_credentials and not policy.license_of("UMLS").redistribution_allowed


def test_an_incompatible_schema_version_is_reported_cleanly(tmp_path):
    _, path = build(tmp_path, [Mini()])
    db = sqlite3.connect(path)
    db.execute("UPDATE meta SET value = '99' WHERE key = 'schema_version'")
    db.commit()
    db.close()
    with pytest.raises(IncompatibleKnowledgeBase, match="schema version 99"):
        open_readonly(path)
    with pytest.raises(KnowledgeError):
        open_readonly(tmp_path / "missing.db")
    (tmp_path / "junk.db").write_bytes(b"not a database" * 100)
    with pytest.raises(KnowledgeError):
        open_readonly(tmp_path / "junk.db")


def test_the_database_is_opened_read_only(tmp_path):
    _, path = build(tmp_path, [Mini()])
    with MedicalKnowledge.open(path) as kb:
        with pytest.raises(sqlite3.OperationalError):
            kb.db.execute("DELETE FROM term")


def test_an_unknown_trust_level_is_rejected(tmp_path):
    builder = KnowledgeBuilder(tmp_path / "x.db", "fixture")
    builder.add_source(SourceInfo("MLT_FIXTURE", "x", "1"))
    with pytest.raises(ValueError):
        builder.add_concept("MLT_FIXTURE", ConceptRecord("c", "c", trust="TRUST_ME"))
    builder.abort()


def test_the_cache_is_bounded():
    cache = LRU(3)
    for i in range(10):
        cache.get(i, lambda i=i: i)
    assert len(cache) == 3 and cache.stats()["misses"] == 10
    assert cache.get(9, lambda: -1) == 9 and cache.stats()["hits"] == 1


def test_the_curated_vocabulary_becomes_concepts_with_domains_cues_and_confusions(tmp_path):
    from src.medical_knowledge.sources.curated import CuratedSource
    from src.medical_knowledge.sources.fixture import FixtureSource

    manifest, path = build(tmp_path, [CuratedSource(), FixtureSource()], profile="public")
    assert manifest["counts"]["concepts"] > 3500 and manifest["counts"]["abbreviations"] > 200 and manifest["counts"]["confusions"] > 300
    with MedicalKnowledge.open(path) as kb:
        remak = kb.lookup("Remak bundle")[0]
        assert "neuroanatomy" in remak.domains
        assert {"Schwann cell", "peripheral nervous system"} <= {r.preferred_name for r in kb.related(remak.concept_id)}
        assert kb.validate_term("tanycytes", "neuroanatomy").known
        assert kb.validate_term("CNS").expansions == ("central nervous system",)
        assert kb.lookup("Rimak") == []
        assert all(c.ok for c in kb.health())
        langs = kb.counts()["languages"]
        assert langs["en"] > 9000 and langs["la"] > 100


def test_the_no_confusables_option_removes_the_known_mishearings(tmp_path):
    from src.medical_knowledge.sources.curated import CuratedSource

    manifest, _ = build(tmp_path, [CuratedSource(use_confusables=False)])
    assert manifest["counts"]["confusions"] == 0


def test_the_open_sources_that_were_reviewed_are_allowed_with_their_conditions_recorded():
    for sid in ("MESH", "GO", "CHEBI", "HGNC"):
        lic = policy.license_of(sid)
        assert lic.status == "ALLOW" and lic.verified_on and "read on" in lic.evidence and sid in policy.public_allowlist()
    assert "Courtesy of the U.S. National Library of Medicine" in policy.license_of("MESH").attribution
    assert "CC BY 4.0" in policy.license_of("GO").license_name and "CC0" in policy.license_of("HGNC").license_name
    assert policy.license_of("HPO").status == "REVIEW_REQUIRED" and "HPO" not in policy.public_allowlist()


def test_a_public_build_ships_third_party_notices_and_a_private_build_says_so(tmp_path):
    from src.medical_knowledge.licenses.notices import notice_text
    from src.medical_knowledge.sources.curated import CuratedSource

    path = tmp_path / "pub.db"
    builder = KnowledgeBuilder(path, "public", "test")
    CuratedSource().export(builder)
    manifest = builder.finalize()
    notice = (tmp_path / "pub.db.NOTICE.txt").read_text(encoding="utf-8")
    assert "THIRD-PARTY NOTICES" in notice and "MLT_CURATED" in notice and manifest["sha256"] in notice
    private = notice_text({"name": "MedicalKnowledge", "profile": "full", "sources": [{"source_id": "UMLS", "name": "UMLS", "redistributable": False, "license_name": "UMLS"}]})
    assert "NOT FOR REDISTRIBUTION" in private and "UMLS" in private
    mesh_scowl = notice_text({"profile": "public", "sources": [{"source_id": "MESH", "name": "MeSH", "redistributable": True}, {"source_id": "SCOWL", "name": "SCOWL", "redistributable": True}]})
    assert "currency notice" in mesh_scowl and "Kevin Atkinson" in mesh_scowl


def test_the_health_check_of_a_shipped_public_pack_fails_on_a_restricted_term_or_link(tmp_path):
    """A pack that left the builder is checked again at self-test time (the packaged app, the CI build): a restricted source, a term not marked redistributable or a link to a
    restricted source in a database that says it is public fails the check."""
    manifest, path = build(tmp_path, [Mini("MLT_CURATED"), Mini("MLT_FIXTURE")], profile="public")

    def public_filter(file):
        db = sqlite3.connect(file)
        try:
            from src.medical_knowledge.database.store import health

            return [c for c in health(db) if c.name == "public license filter"][0]
        finally:
            db.close()

    assert public_filter(path).ok
    tampered = tmp_path / "tampered.db"
    import shutil

    for what, sql in (("a restricted term", "UPDATE term SET redistributable = 0 WHERE term_id = (SELECT MIN(term_id) FROM term)"),
                      ("a restricted source", "UPDATE source SET redistribution_allowed = 0 WHERE source_id = (SELECT MIN(source_id) FROM source)")):
        shutil.copyfile(path, tampered)
        db = sqlite3.connect(tampered)
        db.execute(sql)
        db.commit()
        db.close()
        check = public_filter(tampered)
        assert not check.ok, what
        tampered.unlink()
