"""v4 UMLS importer (milestone 2): runs on a SYNTHETIC release in RRF format. No UMLS data is involved or needed."""

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mk_umls_fixture import write_bulk, write_mini  # noqa: E402

from src.medical_knowledge.ingest.builder import KnowledgeBuilder
from src.medical_knowledge.licenses.policy import LicenseViolation, license_of
from src.medical_knowledge.runtime.knowledge import MedicalKnowledge
from src.medical_knowledge.sources.base import CredentialsRequired
from src.medical_knowledge.sources.umls import UmlsSource
from src.medical_knowledge.sources.umls_types import domains_for


def build(tmp_path, source, profile="full"):
    builder = KnowledgeBuilder(tmp_path / "umls.db", profile, "2099.01")
    stats = source.export(builder)
    return stats, builder.finalize(), tmp_path / "umls.db"


def test_the_importer_reads_concepts_terms_languages_and_source_vocabularies(tmp_path):
    meta = write_mini(tmp_path)
    src = UmlsSource(meta.parent)
    assert src.info.source_version == "2099AA" and src.verify() == []
    stats, manifest, path = build(tmp_path, src)
    assert stats.concepts == 5                                # "Between" (spatial concept) and the Spanish-only concept are not imported
    with MedicalKnowledge.open(path) as kb:
        nissl = kb.lookup("Nissl Bodies")[0]
        assert nissl.concept_id == "UMLS:C9000001" and nissl.is_preferred and nissl.trust == "ONTOLOGY_VERIFIED" and "histology" in nissl.domains
        concept = kb.get_concept("UMLS:C9000001")
        assert {t.text for t in concept.terms} == {"Nissl Bodies", "Nissl Substance", "Tigroid substance", "Nissl-Substanz"}
        assert concept.languages == ("de", "en") and concept.semantic_type == "Cell Component"
        assert kb.lookup("Obsolete Nissl Name") == [] and kb.lookup("Hidden Editor Term") == []           # suppressed atoms are skipped
        assert kb.lookup("Between") == [] and kb.lookup("Cuerpos de Nissl") == []
        assert kb.lookup("Systema nervosum periphericum")[0].language == "la"
        assert kb.validate_term("PNS").expansions == ("Peripheral Nervous System",)


def test_every_vocabulary_is_its_own_source_with_a_private_license(tmp_path):
    _, manifest, path = build(tmp_path, UmlsSource(write_mini(tmp_path).parent))
    db = sqlite3.connect(path)
    sources = {r[0]: r[1:] for r in db.execute("SELECT source_id, name, version, license_status, redistribution_allowed FROM source")}
    assert {"UMLS", "UMLS/MSH", "UMLS/MSHGER", "UMLS/MSHLAT", "UMLS/SNOMEDCT_US"} <= set(sources)
    assert sources["UMLS/SNOMEDCT_US"][2:] == ("PRIVATE_ONLY", 0) and sources["UMLS/MSH"][1] == "2099_2099"
    row = db.execute("SELECT ts.source_id, ts.source_code, ts.source_concept_id FROM term t JOIN term_source ts USING (term_id) WHERE t.text = 'Tigroid substance'").fetchone()
    assert row == ("UMLS/SNOMEDCT_US", "1234001", "C9000001") or row[:2] == ("UMLS/SNOMEDCT_US", "1234001")
    assert db.execute("SELECT COUNT(*) FROM term WHERE redistributable = 1").fetchone()[0] == 0
    assert manifest["counts"]["redistributable_terms"] == 0 and manifest["counts"]["private_only_terms"] == manifest["counts"]["terms"]


def test_umls_is_refused_by_a_public_build(tmp_path):
    builder = KnowledgeBuilder(tmp_path / "pub.db", "public")
    with pytest.raises(LicenseViolation):
        UmlsSource(write_mini(tmp_path).parent).export(builder)
    assert not (tmp_path / "pub.db").exists()
    assert license_of("UMLS/MSH").status == "PRIVATE_ONLY"            # inherited until a vocabulary is reviewed on its own


def test_relationships_are_kept_light_and_only_between_imported_concepts(tmp_path):
    _, _, path = build(tmp_path, UmlsSource(write_mini(tmp_path).parent))
    with MedicalKnowledge.open(path) as kb:
        rel = {(r.preferred_name, r.relation, r.direction) for r in kb.related("UMLS:C9000002")}
        assert ("Peripheral Nervous System", "part_of", "out") in rel and ("Tanycytes", "is_a", "in") in rel and ("Nissl Bodies", "related_to", "out") in rel
        assert kb.db.execute("SELECT COUNT(*) FROM relationship WHERE concept_b = 'UMLS:C9000099' OR concept_a = 'UMLS:C9000099'").fetchone()[0] == 0
        assert kb.db.execute("SELECT COUNT(*) FROM relationship WHERE concept_a = 'UMLS:C9000004' AND concept_b = 'UMLS:C9000006'").fetchone()[0] == 0   # suppressed row
        assert kb.db.execute("SELECT COUNT(*) FROM relationship WHERE relation_type = 'narrower' OR relation_type = 'CHD'").fetchone()[0] == 0


def test_the_importer_options_limit_languages_vocabularies_and_relations(tmp_path):
    meta = write_mini(tmp_path).parent
    stats, _, path = build(tmp_path, UmlsSource(meta, languages=("ENG", "SPA"), sabs=("MSH", "MSHSPA"), with_relations=False))
    assert stats.relations == 0 and stats.concepts == 6                      # + the Spanish-only concept
    with MedicalKnowledge.open(path) as kb:
        assert kb.lookup("Cuerpos de Nissl")[0].language == "es"
        assert kb.lookup("Nissl-Substanz") == []                           # German atom: language not requested
        assert kb.lookup("Tigroid substance") == []                        # SNOMED atom filtered out by --sabs


def test_without_the_licensed_files_the_importer_says_where_to_get_them(tmp_path):
    src = UmlsSource(tmp_path / "nothing")
    problems = src.verify()
    assert problems and "uts.nlm.nih.gov" in problems[0] and "license" in problems[0].lower()
    with pytest.raises(CredentialsRequired, match="NOT downloaded"):
        src.download(tmp_path)
    junk = tmp_path / "junk" / "META"
    junk.mkdir(parents=True)
    (junk / "MRCONSO.RRF").write_text("not|an|rrf\n")
    (junk / "MRSTY.RRF").write_text("C1|T1|x|y|z|\n")
    assert any("does not look like an RRF" in p for p in UmlsSource(junk).verify())


def test_semantic_types_map_to_fields_and_everyday_types_are_skipped():
    assert domains_for(["T026"]) == ("histology", "molecular_biology") and domains_for(["T082", "T079"]) == ()
    assert domains_for(["T047", "T191"]) == ("internal_medicine", "pathology", "oncology")


def test_a_bulk_release_streams_in_constant_memory_and_a_reasonable_time(tmp_path):
    import time
    import tracemalloc

    meta = write_bulk(tmp_path, 20000)
    tracemalloc.start()
    t0 = time.time()
    stats, manifest, _ = build(tmp_path, UmlsSource(meta.parent, with_relations=False))
    seconds = time.time() - t0
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert stats.concepts == 20000 and manifest["counts"]["terms"] == 60000
    assert seconds < 60 and peak < 120 * 1024 * 1024                       # 60,000 atoms: seconds, and no per-concept dictionaries


def test_the_public_build_command_fails_when_umls_is_offered(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools.build_medical_knowledge import main as build_tool

    meta = write_mini(tmp_path).parent
    out = tmp_path / "medical_knowledge_public.db"
    assert build_tool(["--profile", "public", "--output", str(out), "--umls", str(meta)]) == 3
    assert not out.exists()
    assert build_tool(["--profile", "public", "--output", str(out)]) == 0          # the curated vocabulary alone is fine
    assert out.exists() and out.with_suffix(".db.manifest.json").exists()
