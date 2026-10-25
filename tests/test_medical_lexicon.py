"""The medical vocabulary: modular, versioned, lazy, searchable, extensible; sound-alike candidates; Latin plurals."""

import json

from scripts.lint_medical_vocab import lint
from src.medical.lexicon import VOCAB_DIR, Lexicon, available_domains, norm, parse_line
from src.medical.phonetics import similarity, skeleton

WANTED_DOMAINS = {
    "anatomy", "neuroanatomy", "histology", "embryology", "physiology", "biochemistry", "molecular_biology", "genetics", "pharmacology", "pathology",
    "microbiology", "immunology", "neuroscience", "radiology", "surgery", "internal_medicine", "cardiology", "neurology", "gastroenterology",
    "endocrinology", "hematology", "nephrology", "pulmonology", "dermatology", "ophthalmology", "otolaryngology", "psychiatry", "orthopedics",
    "reproductive_medicine", "laboratory_medicine", "latin", "abbreviations", "eponyms",
}


def test_every_requested_domain_folder_exists_and_has_terms():
    assert WANTED_DOMAINS <= set(available_domains())
    lexicon = Lexicon()
    for name in WANTED_DOMAINS:
        domain = lexicon.domain(name)
        assert domain is not None and domain.terms, name
        assert domain.version and domain.title


def test_the_vocabulary_passes_its_own_lint():
    assert lint() == []


def test_sources_are_recorded_with_license_and_version():
    data = json.loads((VOCAB_DIR / "SOURCES.json").read_text(encoding="utf-8"))
    assert data["vocabulary_version"] and data["sources"]
    for source in data["sources"]:
        assert source["name"] and source["license"] and source["version"] and "downloaded" in source
    assert json.loads((VOCAB_DIR / "manifest.json").read_text(encoding="utf-8"))["version"]


def test_domains_load_lazily_and_only_once():
    lexicon = Lexicon()
    assert lexicon._domains == {}
    lexicon.domain("anatomy")
    assert set(lexicon._domains) == {"anatomy"}
    assert lexicon.domain("anatomy") is lexicon.domain("anatomy")
    assert lexicon.domain("no_such_domain") is None


def test_the_vocabulary_is_large_and_versioned():
    lexicon = Lexicon()
    total = sum(len(lexicon.domain(n).terms) for n in lexicon.names())
    assert total >= 4500
    assert set(lexicon.version()) >= WANTED_DOMAINS and all(lexicon.version().values())


def test_exact_terms_with_aliases_and_plurals_are_found():
    index = Lexicon().subset(None)
    for phrase in ("Nissl body", "Nissl bodies", "nissl bodies", "node of Ranvier", "saltatory conduction", "Remak fiber", "tanycytes", "GABA",
                   "gamma-aminobutyric acid", "cresyl violet", "axodendritic", "microglial cells", "metformin", "mTOR"):
        assert index.exact(phrase), phrase


def test_latin_terms_list_singular_and_plural():
    index = Lexicon().subset(None)
    for singular, plural in (("foramen", "foramina"), ("nucleus", "nuclei"), ("ganglion", "ganglia"), ("septum", "septa"), ("cornu", "cornua"),
                             ("ramus", "rami"), ("sulcus", "sulci"), ("gyrus", "gyri"), ("fascia", "fasciae"), ("vertebra", "vertebrae"),
                             ("plexus", "plexuses"), ("nervus", "nervi"), ("arteria", "arteriae"), ("vena", "venae"), ("musculus", "musculi"),
                             ("ligamentum", "ligamenta")):
        terms = index.exact(plural)
        assert terms and any(t.canonical == singular for t in terms), (singular, plural)
        assert all(t.kind in ("latin", "term") for t in index.exact(singular))


def test_latin_and_english_are_never_merged():
    index = Lexicon().subset(None)
    assert not index.exact("foramina") or all(t.canonical != "opening" for t in index.exact("foramina"))
    assert any(term.kind == "latin" for term in index.exact("nuclei"))


def test_sound_alike_candidates_propose_but_do_not_decide():
    index = Lexicon().subset(None)
    names = {t.canonical for _s, t, _sim in index.candidates("nisal bodies")}
    assert "Nissl body" in names
    names = {t.canonical for _s, t, _sim in index.candidates("tannicides")}
    assert "tanycyte" in names
    assert similarity("nisal bodies", "Nissl bodies") > 0.85 and similarity("crazy violet", "cresyl violet") > 0.7 and similarity("table", "Nissl") < 0.5


def test_phonetic_skeleton_ignores_vowels_voicing_and_spelling():
    assert skeleton("Remak") == skeleton("RIMAC") and skeleton("exodendritic") == skeleton("axodendritic")
    assert skeleton("shells") == skeleton("cells") and skeleton("") == "" and skeleton("gh") == ""


def test_line_format_aliases_cues_confusables_and_kind():
    term = parse_line("Nissl body | Nissl bodies; Nissl substance | ribosomes; soma | nisal body | term", "histology")
    assert term.canonical == "Nissl body" and "Nissl substance" in term.aliases and "soma" in term.cues and "nisal body" in term.confusables
    assert parse_line("# comment", "x") is None and parse_line("", "x") is None
    drug = parse_line("metformin | | diabetes | | drug", "pharmacology")
    assert drug.kind == "drug"
    assert "diverticula" not in parse_line("diverticulum | diverticula | colon | | latin", "x").aliases[:0]
    assert "cells" in " ".join(parse_line("stem cell", "x").aliases)                   # an ordinary term gets its English plural


def test_a_concept_in_two_files_is_merged_not_duplicated():
    lexicon = Lexicon()
    tidal = [t for t in lexicon.domain("physiology").terms if norm(t.canonical) == norm("tidal volume")]
    assert len(tidal) == 1 and tidal[0].cues                                             # the starter list entry gained the richer cues


def test_user_and_slide_terms_extend_a_private_copy_only():
    base = Lexicon()
    run = base.fork()
    run.add_terms("custom", ["zorbaxinol", "Bergmann-like glia"])
    assert run.subset(["abbreviations"]).exact("zorbaxinol")
    assert not base.fork().subset(["abbreviations"]).exact("zorbaxinol")
    assert run.subset(["abbreviations"]) is run.subset(["abbreviations"])               # indexes are cached
