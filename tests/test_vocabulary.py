from pathlib import Path

import pytest

from src.vocabulary import (
    SUBJECTS, UnknownSubjectError, VocabularyIndex, available_subjects, build_hotwords, load_terms,
    normalize_key, parse_subjects, read_term_file,
)


def test_all_required_subject_files_exist_and_have_terms():
    assert set(SUBJECTS) <= set(available_subjects())
    for subject in SUBJECTS:
        assert len(load_terms(subject)) >= 20, subject


def test_neuroscience_contains_the_reference_terms():
    terms = {t.casefold() for t in load_terms("neuroscience")}
    for expected in ("corticospinal tract", "spinothalamic tract", "medial lemniscus", "substantia nigra",
                     "subthalamic nucleus", "periaqueductal gray", "broca area", "wernicke area",
                     "nucleus ambiguus", "tractus solitarius", "spinocerebellar tract"):
        assert expected in terms


def test_subject_selection():
    assert parse_subjects("neuroscience") == ["neuroscience"]
    assert parse_subjects("Neuroscience, anatomy") == ["neuroscience", "anatomy"]
    assert parse_subjects("all") == available_subjects()
    assert parse_subjects(None) == []
    with pytest.raises(UnknownSubjectError, match="astrology"):
        parse_subjects("neuroscience,astrology")


def test_multiple_subjects_are_merged_without_duplicates():
    merged = load_terms(["neuroscience", "anatomy"])
    assert len(merged) == len({normalize_key(t) for t in merged})
    assert "thalamus" in {t.casefold() for t in merged}


def test_language_prefixes_filter_terms(tmp_path: Path):
    f = tmp_path / "demo.txt"
    f.write_text("# comment\n\nuniversal term\ntr:omurilik\nen:spinal cord\n", encoding="utf-8")
    assert read_term_file(f, "tr") == ["universal term", "omurilik"]
    assert read_term_file(f, "en") == ["universal term", "spinal cord"]
    assert read_term_file(f, None) == ["universal term", "omurilik", "spinal cord"]


def test_turkish_characters_survive_roundtrip(tmp_path: Path):
    f = tmp_path / "t.txt"
    f.write_text("tr:şişman bağırsak\ntr:İleum\n", encoding="utf-8")
    assert read_term_file(f, "tr") == ["şişman bağırsak", "İleum"]


def test_normalize_key_ignores_case_spacing_and_diacritics():
    assert normalize_key("Spino-Thalamic  Tract") == normalize_key("spinothalamic tract")
    assert normalize_key("İleum") == normalize_key("ileum")
    assert normalize_key("ILIK") == normalize_key("ılık")


def test_build_hotwords_priority_dedup_and_limits():
    result = build_hotwords(["alpha", "beta", "gamma"], priority_terms=["pdf term", "Alpha"], max_terms=3)
    assert result == ["pdf term", "Alpha", "beta"]
    assert build_hotwords(["x" * 50, "y" * 50], max_chars=60) == ["x" * 50]


def test_index_matches_split_hyphenated_and_cased_variants():
    vocab = VocabularyIndex(load_terms("neuroscience"))
    assert vocab.canonical("spino thalamic tract") == "spinothalamic tract"
    assert vocab.canonical("Substantia Nigra") == "substantia nigra"
    found = vocab.find_terms("The corticospinal tract and the substantia nigra differ.")
    assert found == ["corticospinal tract", "substantia nigra"]


def test_inflections_are_not_near_misses():
    from src.vocabulary import is_inflection

    for a, b in (("receptors", "receptor"), ("antagonists", "antagonist"), ("bradycardic", "bradycardia"),
                 ("transcriptional", "transcription"), ("thyroidal", "thyroid")):
        assert is_inflection(a, b), (a, b)
    for a, b in (("spinothalmic", "spinothalamic"), ("propanolol", "propranolol"), ("corticospinel", "corticospinal")):
        assert not is_inflection(a, b), (a, b)
    vocab = VocabularyIndex(["beta receptor", "mitochondria"])
    assert vocab.near_miss("receptors") is None
    assert vocab.near_miss("mitocondria") == "mitochondria"


def test_near_miss_finds_misheard_words_but_not_valid_ones():
    vocab = VocabularyIndex(load_terms("neuroscience"))
    assert vocab.near_miss("spinothalmic") == "spinothalamic"
    assert vocab.near_miss("spinothalamic") is None   # valid word
    assert vocab.near_miss("table") is None           # too short / unrelated


def test_derived_english_forms_are_inflections_but_real_mishearings_are_not():
    from src.vocabulary import is_inflection

    for word, term in (("relatively", "relative"), ("negatively", "negative"), ("transporter", "transport"),
                       ("exchange", "exchanger"), ("dominate", "dominant"), ("positively", "positive"),
                       ("selectively", "selective")):
        assert is_inflection(word, term), (word, term)
    for word, term in (("lemoniscus", "lemniscus"), ("serious", "serous"), ("advertise", "adverse"),
                       ("chemotrypsinogen", "chymotrypsinogen"), ("asperate", "aspartate")):
        assert not is_inflection(word, term), (word, term)
