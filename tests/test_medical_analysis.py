"""PDF context extraction, acronym engine, abbreviation disambiguation, entity categories, name report, fuzzy
suggestions, Turkish terminology - and the rule that none of them ever rewrites the transcript."""

import pytest

from src.acronyms import AcronymEngine, acronyms_report_text, disambiguate, extract_definitions
from src.entities import build_entities, classify_term, entities_summary, known_drug_names
from src.medical_report import build_report, detect_terms
from src.name_report import name_report_text, scan_name_variants
from src.pdf_context import build_context
from src.terminology_tr import build_language_index, load_pairs, localize_hotwords, with_turkish_forms
from src.vocabulary import DEFAULT_TERMS_DIR, VocabularyIndex, build_index
from tests.conftest import make_segment
from tests.test_pdf_context import make_pdf


# ------------------------------------------------------------------------------------------- PDF extraction
SLIDE = ("Cerebrospinal fluid (CSF) circulates around the brain. CSF is produced by the choroid plexus. "
         "Acetylcholinesterase degrades acetylcholine. Acetylcholinesterase inhibitors help. "
         "Patients treated with Donepezil improve. Donepezil blocks the enzyme. "
         "The Kluver-Bucy syndrome follows amygdala damage. See Kluver-Bucy syndrome in the Kluver-Bucy review.")


def test_pdf_context_extracts_terms_acronyms_rare_terms_and_categories(tmp_path):
    pdf = make_pdf(tmp_path / "lec.pdf", [SLIDE])
    vocab = build_index("all", None, DEFAULT_TERMS_DIR)
    ctx = build_context(pdf, vocab, min_frequency=2, max_terms=40)
    assert ctx.error is None and ctx.text and ctx.pages == 1
    assert ctx.acronyms.get("CSF", "").lower() == "cerebrospinal fluid"
    assert any("acetylcholinesterase" in t.lower() for t in ctx.terms)
    assert "enzyme" in ctx.categories and any("acetylcholinesterase" in t.lower() for t in ctx.categories["enzyme"])
    assert any(t.lower().startswith("kluver") for t in ctx.rare_terms + ctx.candidates + ctx.terms)
    assert "CSF" in ctx.terms                                   # a defined acronym is offered to the decoder


def test_pdf_without_text_reports_an_error_instead_of_raising(tmp_path):
    pdf = tmp_path / "bad.pdf"
    pdf.write_bytes(b"not a pdf")
    ctx = build_context(pdf, None)
    assert ctx.error and ctx.terms == [] and ctx.text == ""


# ------------------------------------------------------------------------------------------- acronym engine
def test_definitions_are_learned_in_both_directions_and_checked_against_the_letters():
    found = extract_definitions("The cerebrospinal fluid (CSF) is clear. Also MRI (magnetic resonance imaging) helps.")
    assert found["CSF"].lower() == "cerebrospinal fluid" and found["MRI"].lower() == "magnetic resonance imaging"
    assert "XYZ" not in extract_definitions("the quick brown fox (XYZ) jumped")      # letters do not fit: rejected


def test_acronym_scan_finds_written_spoken_and_spaced_forms():
    engine = AcronymEngine()
    engine.learn("cerebrospinal fluid (CSF)", "defined in slides")
    segments = [make_segment("The CSF is clear.", 1, 3), make_segment("Cerebrospinal fluid fills the ventricles.", 5, 8),
                make_segment("We sample C S F by lumbar puncture.", 10, 13)]
    hits = {h.acronym: h for h in engine.scan(segments)}
    assert "CSF" in hits and hits["CSF"].occurrences >= 1 and hits["CSF"].expansion_mentions >= 1
    assert hits["CSF"].odd_spellings == 1 and hits["CSF"].first_start == 1
    assert hits["CSF"].origin == "defined in slides"
    text = acronyms_report_text(list(hits.values()), [])
    assert "CSF" in text and "never rewritten" in text.replace("\n", " ") or "Reporting only" in text


def test_lecture_definitions_override_the_glossary_and_feed_the_decoder():
    engine = AcronymEngine()
    engine.learn("Cyclic AMP (cAMP) is a second messenger", "defined in speech")
    assert engine.learned and "cAMP" in engine.learned
    hits = engine.scan([make_segment("cAMP rises.", 0, 2)])
    assert any(item["acronym"] == "cAMP" for item in engine.for_hotwords(hits))


# ------------------------------------------------------------------------------------------- abbreviation ambiguity
def test_ambiguous_abbreviation_is_resolved_from_context_and_marked_confident():
    segs = [make_segment("Demyelination in MS is seen on MRI with periventricular plaques.", 0, 5)]
    [finding] = disambiguate(segs)
    assert finding.abbreviation == "MS" and finding.best == "multiple sclerosis" and finding.confidence == "CONFIDENT"


def test_ambiguous_abbreviation_without_evidence_is_flagged_for_review_not_guessed():
    [finding] = disambiguate([make_segment("Note the MS here.", 0, 2)])
    assert finding.confidence == "REVIEW" and finding.best == "undetermined"


def test_a_definition_in_the_lecture_wins_over_the_surrounding_words():
    learned = {"MS": ("mitral stenosis", "defined in speech")}
    segs = [make_segment("Rheumatic fever leads to MS with a diastolic murmur.", 0, 4)]
    [finding] = disambiguate(segs, learned)
    assert finding.best == "mitral stenosis" and finding.defined_in_lecture


def test_disambiguation_never_edits_the_segments():
    segs = [make_segment("Demyelination in MS.", 0, 3)]
    before = [s.text for s in segs]
    disambiguate(segs)
    assert [s.text for s in segs] == before


# ------------------------------------------------------------------------------------------- entity categories
@pytest.mark.parametrize("term,category", [
    ("corticospinal tract", "tract"), ("vagus nerve", "nerve"), ("acetylcholinesterase", "enzyme"),
    ("glucose-6-phosphate dehydrogenase", "enzyme"), ("dopamine receptor", "receptor"),
    ("propranolol", "drug"), ("BRCA1", "gene"), ("pernicious anemia", "disease"),
])
def test_term_categories(term, category):
    result = classify_term(term, known_drug_names(DEFAULT_TERMS_DIR))
    assert category in result.categories, (term, result)


def test_nerves_and_tracts_are_also_anatomy():
    assert "anatomy" in classify_term("vagus nerve").categories
    assert "anatomy" in classify_term("spinothalamic tract").categories


def test_entities_are_built_from_detected_terms_with_counts():
    index = build_index("all", None, DEFAULT_TERMS_DIR)
    segs = [make_segment("The corticospinal tract and the spinothalamic tract. The corticospinal tract again.", 2, 9)]
    entities = build_entities(detect_terms(segs, index), DEFAULT_TERMS_DIR)
    by_term = {e.term: e for e in entities}
    assert by_term["corticospinal tract"].count == 2 and "tract" in by_term["corticospinal tract"].categories
    summary = entities_summary(entities)
    assert summary["tract"] == 3


# ------------------------------------------------------------------------------------------- drug / gene / protein names
def test_name_variants_are_reported_and_the_text_is_untouched():
    segs = [make_segment("TNF alpha and TNF-alpha levels rise, and il6 too.", 0, 4),
            make_segment("Acetyl cholinesterase breaks down acetylcholine.", 5, 8)]
    before = [s.text for s in segs]
    findings = {f.canonical: f for f in scan_name_variants(segs)}
    assert findings["TNF-alpha"].status == "SPELLING_VARIATION"
    assert findings["acetylcholinesterase"].status == "NON_CANONICAL"
    assert findings["IL-6"].status == "NON_CANONICAL"
    text = name_report_text(list(findings.values()), [], [])
    assert "Reporting only" in text and "acetylcholinesterase" in text
    assert [s.text for s in segs] == before


# ------------------------------------------------------------------------------------------- fuzzy suggestion
def test_fuzzy_suggestion_is_a_report_entry_with_evidence_never_a_rewrite():
    index = build_index("all", None, DEFAULT_TERMS_DIR)
    seg = make_segment("The cortical spinal tract carries motor fibers.", 3725, 3730)
    pdf_text = "The corticospinal tract is the main motor pathway."
    report = build_report([seg], index, pdf_terms=["corticospinal tract"], pdf_text=pdf_text)
    assert report.errors, "a near-miss of a dictionary term must be reported"
    err = report.errors[0]
    assert err.suggestion.lower() == "corticospinal tract" and err.original.lower() == "cortical spinal tract"
    assert "pdf" in err.pdf_evidence.lower() or err.pdf_evidence
    assert seg.text == "The cortical spinal tract carries motor fibers."          # raw text unchanged
    assert err.confidence_label in {"high", "medium", "low"}


def test_exact_dictionary_terms_are_not_reported_as_errors():
    index = build_index("all", None, DEFAULT_TERMS_DIR)
    report = build_report([make_segment("The corticospinal tract and the substantia nigra.", 0, 4)], index)
    assert report.errors == []


# ------------------------------------------------------------------------------------------- Turkish terminology
def test_turkish_pairs_are_loaded_and_bidirectional():
    pairs = load_pairs(DEFAULT_TERMS_DIR)
    assert len(pairs) >= 100
    assert pairs.to_turkish("corticospinal tract") == "kortikospinal trakt"
    assert pairs.to_english("kortikospinal trakt") == "corticospinal tract"


def test_turkish_forms_are_recognised_in_turkish_and_auto_runs_but_not_forced_english():
    tr_index, _ = build_language_index("tr", DEFAULT_TERMS_DIR)
    auto_index, _ = build_language_index(None, DEFAULT_TERMS_DIR)
    en_index, _ = build_language_index("en", DEFAULT_TERMS_DIR)
    text = "Kortikospinal trakt motor liflerini taşır."
    assert tr_index.find_terms(text) and auto_index.find_terms(text)
    assert not en_index.find_terms(text)


def test_hotwords_are_localised_only_for_turkish_runs():
    pairs = load_pairs(DEFAULT_TERMS_DIR)
    terms = ["corticospinal tract", "propranolol"]
    assert localize_hotwords(terms, "tr", pairs)[0] == "kortikospinal trakt"
    assert localize_hotwords(terms, "en", pairs) == terms and localize_hotwords(terms, None, pairs) == terms
    assert with_turkish_forms(["corticospinal tract"], pairs) == ["corticospinal tract", "kortikospinal trakt"]


def test_turkish_term_in_a_turkish_transcript_is_a_known_term_not_a_candidate():
    index, _ = build_language_index("tr", DEFAULT_TERMS_DIR)
    stats = detect_terms([make_segment("Spinotalamik trakt ağrıyı taşır.", 0, 3, language="tr")], index)
    assert any("spinotalamik" in s.term.lower() for s in stats)


@pytest.mark.parametrize("term", ["absorption", "cholesterol", "withdrawal", "excretion", "half life"])
def test_general_pharmacology_words_and_sterols_are_not_labelled_as_drugs(term):
    assert "drug" not in classify_term(term, known_drug_names(DEFAULT_TERMS_DIR)).categories


def test_real_drug_names_are_still_drugs():
    drugs = known_drug_names(DEFAULT_TERMS_DIR)
    for term in ("propranolol", "formoterol", "losartan", "atorvastatin", "gentamicin"):
        assert "drug" in classify_term(term, drugs).categories, term
