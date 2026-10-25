"""v4 open terminology importers (milestone 8): MeSH, GO / HPO / ChEBI (OBO), HGNC - on tiny synthetic files in the real formats."""

import gzip
from pathlib import Path

import pytest

from src.medical_knowledge.ingest.builder import KnowledgeBuilder
from src.medical_knowledge.licenses.policy import LicenseViolation
from src.medical_knowledge.runtime.knowledge import MedicalKnowledge
from src.medical_knowledge.sources import registry
from src.medical_knowledge.sources.hgnc import HgncSource
from src.medical_knowledge.sources.mesh import MeshSource, tree_domains
from src.medical_knowledge.sources.obo import ChebiSource, GoSource, HpoSource

MESH_XML = """<?xml version="1.0"?>
<!DOCTYPE DescriptorRecordSet SYSTEM "https://www.nlm.nih.gov/databases/dtd/nlmdescriptorrecordset_20260101.dtd">
<DescriptorRecordSet LanguageCode = "eng">
<DescriptorRecord DescriptorClass = "1">
 <DescriptorUI>D009412</DescriptorUI><DescriptorName><String>Nervous System</String></DescriptorName>
 <TreeNumberList><TreeNumber>A08</TreeNumber></TreeNumberList>
 <ConceptList><Concept PreferredConceptYN="Y"><ConceptUI>M0014999</ConceptUI><ConceptName><String>Nervous System</String></ConceptName>
  <TermList><Term ConceptPreferredTermYN="Y" IsPermutedTermYN="N" LexicalTag="NON" RecordPreferredTermYN="Y"><TermUI>T033457</TermUI><String>Nervous System</String></Term></TermList></Concept></ConceptList>
</DescriptorRecord>
<DescriptorRecord DescriptorClass = "1">
 <DescriptorUI>D017636</DescriptorUI><DescriptorName><String>Neuroglia</String></DescriptorName>
 <TreeNumberList><TreeNumber>A08.663.360</TreeNumber><TreeNumber>A11.671</TreeNumber></TreeNumberList>
 <ConceptList><Concept PreferredConceptYN="Y"><ConceptUI>M0014600</ConceptUI><ConceptName><String>Neuroglia</String></ConceptName>
  <SemanticTypeList><SemanticType><SemanticTypeUI>T025</SemanticTypeUI><SemanticTypeName>Cell</SemanticTypeName></SemanticType></SemanticTypeList>
  <TermList>
   <Term ConceptPreferredTermYN="Y" IsPermutedTermYN="N" LexicalTag="NON" RecordPreferredTermYN="Y"><TermUI>T050801</TermUI><String>Neuroglia</String></Term>
   <Term ConceptPreferredTermYN="N" IsPermutedTermYN="N" LexicalTag="NON" RecordPreferredTermYN="N"><TermUI>T050802</TermUI><String>Glia</String></Term>
   <Term ConceptPreferredTermYN="N" IsPermutedTermYN="Y" LexicalTag="NON" RecordPreferredTermYN="N"><TermUI>T050803</TermUI><String>Cells, Glial</String></Term>
  </TermList></Concept></ConceptList>
 <SeeRelatedList><SeeRelatedDescriptor><DescriptorReferredTo><DescriptorUI>D009412</DescriptorUI></DescriptorReferredTo></SeeRelatedDescriptor></SeeRelatedList>
</DescriptorRecord>
<DescriptorRecord DescriptorClass = "1">
 <DescriptorUI>D001253</DescriptorUI><DescriptorName><String>Astrocytes</String></DescriptorName>
 <TreeNumberList><TreeNumber>A08.663.360.100</TreeNumber></TreeNumberList>
 <ConceptList><Concept PreferredConceptYN="Y"><ConceptUI>M0001</ConceptUI><ConceptName><String>Astrocytes</String></ConceptName>
  <TermList><Term ConceptPreferredTermYN="Y" IsPermutedTermYN="N" LexicalTag="NON" RecordPreferredTermYN="Y"><TermUI>T1</TermUI><String>Astrocytes</String></Term>
   <Term ConceptPreferredTermYN="N" IsPermutedTermYN="N" LexicalTag="ABB" RecordPreferredTermYN="N"><TermUI>T2</TermUI><String>Astro</String></Term></TermList></Concept></ConceptList>
</DescriptorRecord>
<DescriptorRecord DescriptorClass = "1">
 <DescriptorUI>D006801</DescriptorUI><DescriptorName><String>Humans</String></DescriptorName>
 <TreeNumberList><TreeNumber>B01.050.150.900.649.801.400.112.400.400.310.355</TreeNumber></TreeNumberList>
 <ConceptList><Concept PreferredConceptYN="Y"><ConceptUI>M1</ConceptUI><ConceptName><String>Humans</String></ConceptName>
  <TermList><Term ConceptPreferredTermYN="Y" IsPermutedTermYN="N" LexicalTag="NON" RecordPreferredTermYN="Y"><TermUI>T9</TermUI><String>Humans</String></Term></TermList></Concept></ConceptList>
</DescriptorRecord>
<DescriptorRecord DescriptorClass = "1">
 <DescriptorUI>D019999</DescriptorUI><DescriptorName><String>Publication Characteristics</String></DescriptorName>
 <TreeNumberList><TreeNumber>V02</TreeNumber></TreeNumberList>
 <ConceptList><Concept PreferredConceptYN="Y"><ConceptUI>M2</ConceptUI><ConceptName><String>Publication Characteristics</String></ConceptName>
  <TermList><Term ConceptPreferredTermYN="Y" IsPermutedTermYN="N" LexicalTag="NON" RecordPreferredTermYN="Y"><TermUI>T10</TermUI><String>Publication Characteristics</String></Term></TermList></Concept></ConceptList>
</DescriptorRecord>
</DescriptorRecordSet>
"""

GO_OBO = """format-version: 1.2
data-version: releases/2026-07-26
ontology: go

[Term]
id: GO:0005783
name: endoplasmic reticulum
namespace: cellular_component
synonym: "ER" EXACT []
synonym: "endoplasmic reticulum membrane system" BROAD []
is_a: GO:0005737 ! cytoplasm

[Term]
id: GO:0005737
name: cytoplasm
namespace: cellular_component

[Term]
id: GO:0099999
name: regulation of something rare
namespace: biological_process

[Term]
id: GO:0000001
name: old thing
namespace: cellular_component
is_obsolete: true
"""

HPO_OBO = """format-version: 1.2
data-version: hp/releases/2026-09-01
remark: Please see license of HPO at http://www.human-phenotype-ontology.org

[Term]
id: HP:0000252
name: Microcephaly
synonym: "Abnormally small skull" EXACT layperson []
synonym: "Small head" RELATED layperson []
"""

CHEBI_OBO = """format-version: 1.2
data-version: chebi/256

[Term]
id: CHEBI:7990
name: metoprolol
synonym: "Lopressor" RELATED BRAND:NAME []
synonym: "(2RS)-1-(isopropylamino)-3-[4-(2-methoxyethyl)phenoxy]propan-2-ol" EXACT IUPAC:NAME []

[Term]
id: CHEBI:99999
name: 2,2'-dichloro-1,1'-biphenyl
"""

HGNC_TSV = ("hgnc_id\tsymbol\tname\tlocus_group\tlocus_type\talias_symbol\talias_name\tprev_symbol\tprev_name\tstatus\n"
            "HGNC:4235\tGFAP\tglial fibrillary acidic protein\tprotein-coding gene\tgene with protein product\t\t\t\t\tApproved\n"
            "HGNC:1\tFAKE1\tfake rna\tnon-coding RNA\tRNA\t\t\t\t\tApproved\n"
            "HGNC:2\tOLD1\told gene\tprotein-coding gene\tgene with protein product\t\t\t\t\tWithdrawn\n"
            "HGNC:3\tAR\tandrogen receptor\tprotein-coding gene\tgene with protein product\tDHTR|NR3C4\t\t\tandrogen receptor old\tApproved\n")


def build(tmp_path, source, profile="public"):
    builder = KnowledgeBuilder(tmp_path / "os.db", profile, "t")
    stats = source.export(builder)
    manifest = builder.finalize()
    return stats, manifest, MedicalKnowledge.open(tmp_path / "os.db")


def test_mesh_descriptors_terms_trees_and_license_filter(tmp_path):
    path = tmp_path / "desc2026.gz"
    path.write_bytes(gzip.compress(MESH_XML.encode("utf-8")))
    src = MeshSource(path)
    assert src.verify() == [] and src.info.source_version == "2026"
    stats, manifest, kb = build(tmp_path, src)
    with kb:
        assert stats.concepts == 3                                                 # "Humans" (organisms) and "Publication Characteristics" are not medical vocabulary
        glia = kb.lookup("Glia")[0]
        assert glia.concept_id == "MESH:D017636" and glia.trust == "ONTOLOGY_VERIFIED" and {"neuroanatomy", "histology"} <= set(glia.domains)
        assert kb.lookup("Cells, Glial") == []                                     # permuted entry terms are skipped
        assert kb.validate_term("Astro").is_abbreviation and kb.validate_term("Astro").expansions == ("Astrocytes",)
        rel = {(r.preferred_name, r.relation, r.direction) for r in kb.related("MESH:D001253")}
        assert ("Neuroglia", "broader", "out") in rel
        assert ("Nervous System", "related_to", "out") in {(r.preferred_name, r.relation, r.direction) for r in kb.related("MESH:D017636")}
    assert manifest["profile"] == "public" and manifest["sources"][0]["license_status"] == "ALLOW" and "Courtesy of the U.S. National Library" in manifest["sources"][0]["attribution"]


def test_mesh_tree_numbers_map_to_fields_of_study():
    assert tree_domains(["A08.186"]) == ("neuroanatomy", "anatomy") and "cardiology" in tree_domains(["C14.280"]) and tree_domains(["Z01.107"]) == ()
    assert tree_domains(["C10.228", "C04.557"]) == ("neurology", "pathology", "oncology")


def test_go_keeps_cellular_terms_and_skips_regulation_families_and_obsolete_ones(tmp_path):
    path = tmp_path / "go-basic.obo"
    path.write_text(GO_OBO, encoding="utf-8")
    stats, _, kb = build(tmp_path, GoSource(path))
    with kb:
        assert stats.concepts == 2 and kb.lookup("regulation of something rare") == [] and kb.lookup("old thing") == []
        er = kb.lookup("endoplasmic reticulum")[0]
        assert er.concept_id == "GO:0005783" and {"histology", "molecular_biology"} <= set(er.domains)
        assert kb.lookup("ER")[0].concept_id == "GO:0005783" and kb.lookup("endoplasmic reticulum membrane system") == []     # BROAD synonym not kept
        assert {r.preferred_name for r in kb.related("GO:0005783") if r.relation == "is_a"} == {"cytoplasm"}


def test_hpo_is_private_only_until_its_license_is_read(tmp_path):
    path = tmp_path / "hp.obo"
    path.write_text(HPO_OBO, encoding="utf-8")
    with pytest.raises(LicenseViolation):
        build(tmp_path, HpoSource(path), profile="public")
    _, manifest, kb = build(tmp_path, HpoSource(path), profile="full")
    with kb:
        assert kb.lookup("Small head")[0].concept_id == "HP:0000252" and manifest["counts"]["private_only_terms"] == manifest["counts"]["terms"]


def test_chebi_keeps_drug_like_names_and_skips_iupac_names(tmp_path):
    path = tmp_path / "chebi_lite.obo"
    path.write_text(CHEBI_OBO, encoding="utf-8")
    stats, _, kb = build(tmp_path, ChebiSource(path))
    with kb:
        assert stats.concepts == 1 and kb.lookup("metoprolol")[0].domains == ("pharmacology", "biochemistry")
        assert kb.lookup("Lopressor")[0].concept_id == "CHEBI:7990" and kb.lookup("(2RS)-1-(isopropylamino)-3-[4-(2-methoxyethyl)phenoxy]propan-2-ol") == []


def test_hgnc_symbols_names_aliases_and_status(tmp_path):
    path = tmp_path / "hgnc_complete_set.txt"
    path.write_text(HGNC_TSV, encoding="utf-8")
    stats, _, kb = build(tmp_path, HgncSource(path))
    with kb:
        assert stats.concepts == 2                                                  # non-coding and withdrawn entries are left out
        assert kb.lookup("GFAP")[0].concept_id == "HGNC:4235" and kb.lookup("glial fibrillary acidic protein")[0].concept_id == "HGNC:4235"
        assert kb.lookup("NR3C4")[0].concept_id == "HGNC:3" and kb.lookup("androgen receptor old")[0].concept_id == "HGNC:3"
        assert kb.validate_term("GFAP").is_abbreviation and kb.get_concept("HGNC:4235").terms[0].term_type in ("gene", "preferred")


def test_the_registry_finds_files_and_explains_what_is_missing(tmp_path):
    assert registry.locate("go", tmp_path).name == "go-basic.obo"
    src = registry.create("mesh", tmp_path)
    assert src.verify() and "nlmpubs.nlm.nih.gov" in src.verify()[0]
    (tmp_path / "desc2025.gz").write_bytes(gzip.compress(b"<x/>"))
    assert registry.locate("mesh", tmp_path).name == "desc2025.gz"
    with pytest.raises(ValueError):
        registry.create("nonsense")
