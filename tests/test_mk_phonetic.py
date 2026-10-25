"""v4 phonetic engine (milestone 4): Double Metaphone, medical phoneme key, Soundex-like key, sound similarity."""

import pytest

from src.medical_knowledge.indexes.double_metaphone import dm_key, double_metaphone
from src.medical_knowledge.indexes.phonetic import KEYS, deletion_eligible, deletions
from src.medical_knowledge.indexes.phonetic_extra import mp_key, msx_key
from src.medical_knowledge.runtime.retrieve import sound_similarity

# reference values of the published algorithm
DM = {"Smith": ("SM0", "XMT"), "Schmidt": ("XMT", "SMT"), "Xavier": ("SF", "SFR"), "Jose": ("HS", "HS"), "Knight": ("NT", "NT"), "Wright": ("RT", "RT"),
      "Caesar": ("SSR", "SSR"), "Michael": ("MKL", "MXL"), "Cheney": ("XN", "XN"), "Gnome": ("NM", "NM"), "Albert": ("ALPRT", "ALPRT"), "Bartlett": ("PRTLT", "PRTLT"),
      "Schwarz": ("XRS", "XFRTS"), "Peter": ("PTR", "PTR"), "Catherine": ("K0RN", "KTRN"), "Stephen": ("STFN", "STFN"), "Phillips": ("FLPS", "FLPS"),
      "Brian": ("PRN", "PRN"), "Jennifer": ("JNFR", "ANFR"), "Gandhi": ("KNT", "KNT"), "Zhang": ("JNK", "JNK")}


@pytest.mark.parametrize("word, expected", list(DM.items()))
def test_double_metaphone_matches_the_published_reference(word, expected):
    assert double_metaphone(word) == expected


def test_double_metaphone_has_no_four_letter_cut_off_and_ignores_case_and_punctuation():
    assert len(dm_key("axodendritic")) == 9 and double_metaphone("NISSL") == double_metaphone("nissl") == ("NSL", "NSL")
    assert dm_key("gamma-aminobutyric acid") == dm_key("gamma aminobutyric acid") and double_metaphone("") == ("", "") and double_metaphone("123") == ("", "")


@pytest.mark.parametrize("heard, term", [("nisal", "Nissl"), ("exodendritic", "axodendritic"), ("rimac", "Remak"), ("tanny sites", "tanycytes"), ("crazy violet", "cresyl violet"),
                                         ("sell body", "cell body"), ("lumber puncture", "lumbar puncture"), ("metaprolol", "metoprolol"), ("wernick", "Wernicke")])
def test_real_mishearings_sound_alike_under_at_least_one_key(heard, term):
    assert any(KEYS[k](heard) == KEYS[k](term) for k in ("sk", "dm", "mp")) or sound_similarity(heard, term) > 0.85, (heard, term, sound_similarity(heard, term))


def test_the_phoneme_key_keeps_the_syllables_the_skeleton_drops():
    assert KEYS["sk"]("Nissl") == KEYS["sk"]("nasal") == "nsl"                      # the skeleton cannot tell them apart
    assert mp_key("Nissl") != mp_key("nasal") and mp_key("nisal") == mp_key("nasal")  # ... the phoneme key can ("nisal" is "nasal" to an ear)
    assert mp_key("tanny sites") == mp_key("tanycytes")                             # word boundaries do not matter
    assert mp_key("photo") == mp_key("foto") and mp_key("psychiatry")[0] == "s" and mp_key("pneumonia")[0] == "n"
    assert mp_key("oedema") == mp_key("edema") and mp_key("gynecology")[0] == "s"   # medical spellings (ae/oe, soft g) before folding


def test_the_soundex_like_key_is_coarser_than_the_phoneme_key():
    assert msx_key("Nissl") == msx_key("nasal") == msx_key("nisal")                 # vowels are gone: all three collapse
    assert msx_key("") == "" and msx_key("tanycytes") == msx_key("tanny sites") and msx_key("Nissl") != msx_key("Remak")


def test_sound_similarity_ranks_the_true_term_above_unrelated_ones():
    heard = "exodendritic"
    assert sound_similarity(heard, "axodendritic") > 0.9 > sound_similarity(heard, "exocytosis") > sound_similarity(heard, "hypothalamus")
    assert sound_similarity("crazy violet", "cresyl violet") > sound_similarity("crazy violet", "gentian violet") > 0.0
    assert sound_similarity("", "x") == 0.0 and sound_similarity("same", "same") == pytest.approx(1.0)


def test_the_deletion_index_keys_are_the_double_metaphone_keys_of_short_enough_terms():
    assert "NSL" in deletions(dm_key("Nissl")) and "NS" in deletions(dm_key("Nissl"))
    assert deletion_eligible(dm_key("Nissl"), 1) and not deletion_eligible(dm_key("a"), 1)


def test_the_phonetic_evaluation_data_is_present_and_documented():
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "benchmarks" / "v4" / "phonetic_pairs.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data["pairs"]) > 300 and {"curated_confusable", "lecture_known", "tts_benchmark"} <= set(data["counts"])
    assert all(len(p["heard"].split()) <= 4 for p in data["pairs"])                 # short spans only: no transcript text is stored
