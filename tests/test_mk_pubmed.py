"""v4 milestone 9: the optional literature layer - keywords only, never article text; UNVERIFIED until the promotion rule says otherwise; private builds only."""

import json

import pytest

from src.medical_knowledge.ingest.builder import KnowledgeBuilder
from src.medical_knowledge.licenses.policy import LicenseViolation
from src.medical_knowledge.runtime.knowledge import MedicalKnowledge
from src.medical_knowledge.sources.curated import CuratedSource
from src.medical_knowledge.sources.pubmed import (Evidence, LiteratureMiner, MinedTerm, PubmedSource, concept_id, confidence, fetch, parse_articles, parse_records, promote,
                                                  recognized_identifier, term_row, term_shape_ok, verification_state)

SECRET_TITLE = "A TITLE THAT MUST NEVER BE STORED"
SECRET_ABSTRACT = "AN ABSTRACT SENTENCE THAT MUST NEVER BE STORED"


def article(pmid, year, *keywords):
    kw = "".join(f"<Keyword>{k}</Keyword>" for k in keywords)
    return (f"<PubmedArticle><MedlineCitation><PMID>{pmid}</PMID><Article><Journal><JournalIssue><PubDate><Year>{year}</Year></PubDate></JournalIssue></Journal>"
            f"<ArticleTitle>{SECRET_TITLE}</ArticleTitle><Abstract><AbstractText>{SECRET_ABSTRACT}</AbstractText></Abstract></Article>"
            f"<KeywordList>{kw}</KeywordList></MedlineCitation></PubmedArticle>")


def xml(*articles):
    return "<PubmedArticleSet>" + "".join(articles) + "</PubmedArticleSet>"


def test_only_ids_years_and_keywords_are_read():
    rows = list(parse_articles(xml(article("11", 2020, "periglial lamella system", "cerebellum"))))
    assert rows == [("11", 2020, ["periglial lamella system", "cerebellum"])]


@pytest.mark.parametrize("term, ok", [("periglial lamella system", True), ("mTORC1 signalling", True), ("a", False), ("x" * 3, False), ("one two three four five", False),
                                      ("p < 0.05, n=12", False), ("2020 2021 2022", False), ("(parenthesis)", False)])
def test_keyword_shape(term, ok):
    assert term_shape_ok(term) is ok


def test_counting_known_and_plain_terms_are_not_candidates_and_no_text_is_kept(tmp_path):
    miner = LiteratureMiner(is_known=lambda t: t.lower() == "dendrite", is_plain_english=lambda t: t == "healthy")
    for i in range(6):
        miner.add_xml(xml(article(str(100 + i), 2018 + i, "periglial lamella system", "dendrite", "healthy", "periglial lamella system")))      # duplicate keyword in one record counts once
    assert miner.records == 6
    (term,) = miner.candidates()
    assert term.term == "periglial lamella system" and term.articles == 6 and len(term.pmids) == 6 and (term.first_year, term.last_year) == (2018, 2023)
    out = tmp_path / "mined.json"
    out.write_text(json.dumps(miner.to_json("q", ["neuroanatomy"])), encoding="utf-8")
    text = out.read_text(encoding="utf-8")
    assert SECRET_TITLE not in text and SECRET_ABSTRACT not in text and "no article text" in text
    assert miner.rejected["known"] == 6 and miner.rejected["plain English"] == 6


def test_publications_alone_are_multi_source_not_verified():
    assert verification_state(MinedTerm("periglial lamella system", 1, ["1"], 2020, 2020)) == "UNVERIFIED"
    assert verification_state(MinedTerm("periglial lamella system", 9, ["1"], 2020, 2020)) == "UNVERIFIED"                  # one year only
    assert verification_state(MinedTerm("periglial lamella system", 5, ["1"], 2019, 2021)) == "MULTI-SOURCE"
    assert promote(MinedTerm("periglial lamella system", 5, ["1"], 2019, 2021)) == "UNVERIFIED_CANDIDATE"                   # not enough for the knowledge base
    assert verification_state(MinedTerm("p<0.05", 50, ["1"], 2010, 2024), Evidence(ontology_match=True, in_pdf=True)) == "UNVERIFIED"   # a shape that is not a term never advances


def test_a_term_is_verified_by_publications_plus_one_independent_signal_or_by_two_signals():
    many = MinedTerm("periglial lamella system", 6, ["1"], 2018, 2023)
    one = MinedTerm("periglial lamella system", 1, ["1"], 2023, 2023)
    for ev in (Evidence(ontology_match=True), Evidence(recognized_identifier=True), Evidence(lectures=2), Evidence(in_pdf=True), Evidence(custom_vocabulary=True)):
        assert verification_state(many, ev) == "VERIFIED" and promote(many, ev) == "LITERATURE_VERIFIED", ev
    assert verification_state(many, Evidence(lectures=1)) == "MULTI-SOURCE"                                                # one lecture is not a repeat
    assert verification_state(one, Evidence(ontology_match=True)) == "ONTOLOGY_MATCHED" and promote(one, Evidence(ontology_match=True)) == "UNVERIFIED_CANDIDATE"
    assert verification_state(one, Evidence(in_pdf=True, custom_vocabulary=True)) == "VERIFIED"                            # two independent signals without publications
    assert verification_state(one, Evidence(in_pdf=True)) == "UNVERIFIED"


def test_confidence_grows_with_independent_evidence_and_is_not_a_probability():
    base = MinedTerm("periglial lamella system", 1, ["1"], 2023, 2023)
    many = MinedTerm("periglial lamella system", 6, ["1"], 2018, 2023)
    assert confidence(base) < confidence(many) < confidence(many, Evidence(ontology_match=True)) <= 0.99
    assert confidence(MinedTerm("p<0.05", 50, ["1"], 2010, 2024)) == 0.0


def test_recognised_identifiers():
    assert recognized_identifier("rs12345") and recognized_identifier("50-00-0") and recognized_identifier("EC 3.4.21.4") and not recognized_identifier("periglial")
    assert recognized_identifier("GFAP", is_gene_symbol=lambda s: s == "GFAP") and not recognized_identifier("GFAP", is_gene_symbol=lambda s: False)


def test_the_stored_row_has_the_ledger_fields_and_no_article_text():
    term = MinedTerm("periglial lamella system", 6, ["200", "201"], 2018, 2023, ["10.1000/x1"])
    row = term_row(term, Evidence(in_pdf=True))
    assert {"term", "identifier", "source_ids", "frequency", "first_seen", "last_seen", "confidence", "verification_state"} <= set(row)
    assert row["verification_state"] == "VERIFIED" and "DOI:10.1000/x1" in row["source_ids"] and "PMID:200" in row["source_ids"]
    assert SECRET_TITLE not in json.dumps(row) and row["first_seen"] == 2018 and row["last_seen"] == 2023


def test_dois_are_read_and_a_record_mined_twice_counts_once():
    doc = article("11", 2020, "periglial lamella system").replace("</MedlineCitation>", "</MedlineCitation><PubmedData><ArticleIdList><ArticleId IdType=\"doi\">10.1000/abc</ArticleId></ArticleIdList></PubmedData>")
    assert list(parse_records(xml(doc))) == [("11", 2020, ["periglial lamella system"], "10.1000/abc")]
    miner = LiteratureMiner()
    miner.add_xml(xml(doc))
    miner.add_xml(xml(doc))
    (term,) = miner.candidates()
    assert term.articles == 1 and term.dois == ["10.1000/abc"]


def mined_file(tmp_path):
    miner = LiteratureMiner()
    for i in range(6):
        miner.add_xml(xml(article(str(200 + i), 2018 + i, "periglial lamella system")))
    miner.add_xml(xml(article("300", 2024, "rare keyword term")))
    miner.evidence["periglial lamella system"] = Evidence(in_pdf=True)                             # the user's slides print it: an independent signal besides six publications
    path = tmp_path / "mined.json"
    path.write_text(json.dumps(miner.to_json("q", ["neuroanatomy"])), encoding="utf-8")
    return path


def test_the_source_builds_private_databases_with_trust_and_provenance(tmp_path):
    path = mined_file(tmp_path)
    builder = KnowledgeBuilder(tmp_path / "full.db", "full", "test")
    CuratedSource().export(builder)
    stats = PubmedSource(path).export(builder)
    builder.finalize()
    assert stats.concepts == 2 and stats.terms == 2
    with MedicalKnowledge.open(tmp_path / "full.db") as kb:
        hit = kb.lookup("periglial lamella system", limit=3)[0]
        assert hit.concept_id == concept_id("periglial lamella system") and hit.trust == "LITERATURE_VERIFIED" and not hit.redistributable
        assert kb.lookup("rare keyword term", limit=1)[0].trust == "UNVERIFIED_CANDIDATE"
        row = kb.db.execute("SELECT ts.source_id, ts.source_code, ts.source_concept_id FROM term_source ts JOIN term t USING (term_id) WHERE t.text = 'periglial lamella system'").fetchone()
        assert row[0] == "PUBMED_MINED" and row[1] == "2018-2023" and row[2].startswith("PMID:200")


def test_a_public_build_refuses_the_literature_source(tmp_path):
    path = mined_file(tmp_path)
    builder = KnowledgeBuilder(tmp_path / "pub.db", "public", "test")
    with pytest.raises(LicenseViolation):
        PubmedSource(path).export(builder)
    builder.abort()


def test_the_network_helper_batches_and_stays_polite():
    calls = []

    def fake(url, params):
        calls.append((url.rsplit("/", 1)[-1], dict(params)))
        if "esearch" in url:
            return json.dumps({"esearchresult": {"idlist": [str(i) for i in range(1, 6)]}}).encode()
        return xml(article("1", 2020, "x keyword")).encode()

    batches = list(fetch("query", retmax=5, batch=2, delay=0, getter=fake))
    assert len(batches) == 3 and [c[0] for c in calls] == ["esearch.fcgi", "efetch.fcgi", "efetch.fcgi", "efetch.fcgi"]
    assert calls[1][1]["id"] == "1,2" and calls[0][1]["retmax"] == 5 and all("tool" in c[1] for c in calls)


def test_the_mining_tool_gives_evidence_from_the_knowledge_base_and_reports_the_states(tmp_path, monkeypatch, capsys):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    import mine_pubmed

    assert mine_pubmed.variants("neural folds") == ["neural fold", "neural-folds"]
    builder = KnowledgeBuilder(tmp_path / "k.db", "full", "test")
    CuratedSource().export(builder)
    builder.finalize()
    xml_file = tmp_path / "dump.xml"
    xml_file.write_text(xml(*[article(str(500 + i), 2015 + i, "neural folds", "periglial lamella system") for i in range(6)]), encoding="utf-8")
    out = tmp_path / "terms.json"
    assert mine_pubmed.main(["--xml-file", str(xml_file), "--kb", str(tmp_path / "k.db"), "--out", str(out)]) == 0
    rows = {r["term"]: r for r in json.loads(out.read_text(encoding="utf-8"))["terms"]}
    assert rows["periglial lamella system"]["verification_state"] == "MULTI-SOURCE"                                           # only publications
    printed = capsys.readouterr().out
    assert "states" in printed and "MULTI-SOURCE" in printed
