"""'Did you mean ...?' suggestions need evidence, not just similar spelling (false positives seen on the real lecture)."""

import copy

import pytest

from src.common_words import common_words, is_common_word
from src.medical_report import (
    HIGH_SCORE, MEDIUM_SCORE, build_report, label_for, phonetic_key, phonetic_similarity, scan_possible_errors,
    score_suggestion,
)
from src.vocabulary import VocabularyIndex, load_terms, normalize_key
from tests.conftest import make_segment


@pytest.fixture(scope="module")
def vocab():
    return VocabularyIndex(load_terms("all", "en"))


def scan(vocab, *sentences, **kw):
    segs = [make_segment(text, i * 10.0, i * 10.0 + 8.0) for i, text in enumerate(sentences)]
    return scan_possible_errors(segs, vocab, **kw)


# ------------------------------------------------------------------------------------------- the four named false positives
@pytest.mark.parametrize("sentence", [
    "What else can I say to advertise the lectures?",
    "But it raises concerns regarding the maintenance of the nitrogen balance.",
    "So muscle uses the branched chain amino acids so the candidate will be serious.",
    "It is the relation of the protein that is taken to the composition of the human proteins.",
    "As you see, these amino acids have relatively large non-polar side chains and positively charged groups.",
    "Here we see one transporter, again with an abbreviation, and the exchange type of transport.",
])
def test_ordinary_english_is_never_corrected(vocab, sentence):
    assert scan(vocab, sentence) == []


def test_the_real_lecture_noise_is_gone_but_real_mishearings_are_kept(vocab):
    errors = scan(
        vocab,
        "So we should advertise the vegetarian diet because regarding the serious relation it matters.",
        "The proenzymes are chemotrypsinogen, proelastase and procalboxypeptidase and they have no activity.",
        "It is the function of the gomaglutamyl cycle that I have to explain.",
    )
    pairs = {(e.original.lower(), e.suggestion.lower()) for e in errors}
    assert ("chemotrypsinogen", "chymotrypsinogen") in pairs
    assert ("procalboxypeptidase", "procarboxypeptidase") in pairs
    assert ("gomaglutamyl cycle", "gamma-glutamyl cycle") in pairs
    assert not {"advertise", "regarding", "serious", "relation"} & {e.original.lower() for e in errors}


def test_function_words_at_the_edge_of_a_phrase_are_not_part_of_a_term(vocab):
    assert scan(vocab, "This is the regulation of blood pressure, and hepatocytes in the liver.") == []


# ------------------------------------------------------------------------------------------- the evidence score
def test_pdf_evidence_and_subject_raise_the_score_and_the_label(vocab):
    sentence = "Damage to the cortical spinal tract causes weakness."
    plain = scan(vocab, sentence)[0]
    backed = scan(vocab, sentence, pdf_terms=["corticospinal tract"], pdf_text="the corticospinal tract descends",
                  active_subjects=["neuroscience"])[0]
    assert backed.score > plain.score
    assert backed.components["pdf_evidence"] == 1.0 and plain.components["pdf_evidence"] == 0.0
    assert backed.confidence_label in {"medium", "high"} and plain.confidence_label in {"low", "medium"}
    assert "appears in the slides" in backed.pdf_evidence


def test_labels_follow_the_score_thresholds():
    assert label_for(HIGH_SCORE) == "high" and label_for(MEDIUM_SCORE) == "medium" and label_for(MEDIUM_SCORE - 0.01) == "low"


def test_components_are_reported_and_weighted_to_at_most_one():
    score, parts = score_suggestion(heard="asperate", suggestion="aspartate", ratio=0.9, subject_files=["biochemistry"],
                                    active_subjects=["biochemistry"], in_pdf=True, local_terms=3, neighbor_medical=2,
                                    heard_is_nonword=True)
    assert 0.0 < score <= 1.0
    for name in ("phonetic_similarity", "medical_specificity", "subject_relevance", "local_context", "pdf_evidence",
                 "neighboring_medical_words"):
        assert 0.0 <= parts[name] <= 1.0, name


def test_subject_relevance_depends_on_the_lectures_subject():
    on = score_suggestion(heard="x", suggestion="aspartate", ratio=0.9, subject_files=["biochemistry"],
                          active_subjects=["biochemistry"], in_pdf=False, local_terms=0, neighbor_medical=0)[1]
    off = score_suggestion(heard="x", suggestion="aspartate", ratio=0.9, subject_files=["pharmacology"],
                           active_subjects=["biochemistry"], in_pdf=False, local_terms=0, neighbor_medical=0)[1]
    assert on["subject_relevance"] == 1.0 and off["subject_relevance"] < 0.2


def test_neighbouring_medical_words_and_local_context_add_evidence(vocab):
    lonely = scan(vocab, "A spinothalmic pathway.")
    rich = scan(vocab, "The thalamus and the substantia nigra and the spinothalmic tract carry sensory pathways.")
    assert rich and rich[0].components["local_context"] > 0
    assert not lonely or rich[0].score >= lonely[0].score


def test_a_minimum_score_can_silence_weak_suggestions(vocab):
    sentence = "Damage to the cortical spinal tract causes weakness."
    assert scan(vocab, sentence, min_score=0.95) == []
    assert scan(vocab, sentence, min_score=0.1)


def test_an_ordinary_word_is_only_offered_as_the_fix_when_the_slides_use_it(vocab):
    sentence = "The change was relativly small in that group."          # a typo of an everyday word
    assert scan(vocab, sentence) == []
    backed = scan(vocab, sentence, pdf_terms=["relative"], pdf_text="relative risk and relative humidity")
    assert all(e.suggestion != "relative" or e.pdf_evidence.startswith("yes") for e in backed)


def test_suggestions_never_touch_the_transcript(vocab):
    segs = [make_segment(" The cortical spinal tract and a medial lemoniscus.", 0, 5)]
    before = copy.deepcopy(segs)
    report = build_report(segs, vocab)
    assert segs == before and report.errors


# ------------------------------------------------------------------------------------------- phonetics, common words, vocabulary
def test_phonetic_similarity_rewards_sound_alikes():
    assert phonetic_key("phosphate") == phonetic_key("fosfate")
    assert phonetic_similarity("asperate", "aspartate") > phonetic_similarity("asperate", "aspirin")
    assert phonetic_similarity("jejunum", "jejunum") == pytest.approx(1.0)


def test_common_word_list_ships_with_the_app():
    words = common_words()
    assert len(words) > 15000
    for word in ("advertise", "regarding", "serious", "relation", "relatively", "transport"):
        assert is_common_word(word), word
    for word in ("jejunum", "chymotrypsinogen", "asperate", "gomaglutamyl", "enteropeptidase"):
        assert not is_common_word(word), word


def test_requested_canonical_terms_are_in_the_vocabulary():
    keys = {normalize_key(t) for t in load_terms("all", "en")}
    for term in ("cysteine", "methionine", "arginine", "carboxypeptidase", "enteropeptidase", "tripeptide", "gamma-glutamyl",
                 "glutathione", "HMG-CoA reductase", "essential amino acids", "jejunum", "ileum", "aspartate", "protonated",
                 "procarboxypeptidase", "chymotrypsinogen", "ammonium", "ACE2", "B0AT1", "proline"):
        assert normalize_key(term) in keys, term


def test_correctly_heard_canonical_terms_are_not_reported(vocab):
    text = ("Cysteine and methionine, arginine and proline; carboxypeptidase and enteropeptidase cleave the tripeptide; "
            "gamma-glutamyl transpeptidase and glutathione; aspartate is protonated; chymotrypsinogen, ammonium, ACE2, B0AT1.")
    assert scan(vocab, text) == []


# ----------------------------------------------------------------------------------------------- valid terms of their own (found on a real lecture)
from src.medical_report import is_distinct_entity   # noqa: E402


KNOWN = {"proteasome", "phosphorylation", "phosphorylase", "transcription", "factor", "kinase", "protein", "inositol"}


def test_a_different_one_character_label_is_a_different_entity():
    assert is_distinct_entity(["protein", "kinase", "B"], "protein kinase A", KNOWN)
    assert is_distinct_entity(["factor", "VIII"], "factor IX", KNOWN | {"factor"}) is False        # labels longer than 2 characters: checked normally
    assert is_distinct_entity(["vitamin", "B"], "vitamin C", KNOWN)


def test_a_term_with_a_number_attached_is_not_a_misspelling():
    assert is_distinct_entity(["inositol-3"], "inositol", KNOWN)
    assert not is_distinct_entity(["inositols"], "inositol", KNOWN)


def test_a_member_of_the_same_word_family_is_not_a_misspelling():
    assert is_distinct_entity(["phosphorylated"], "phosphorylase", KNOWN)             # 'phosphorylation' is known
    assert is_distinct_entity(["transcriptional", "factor"], "transcription factor", KNOWN)


def test_real_mishearings_are_still_candidates():
    assert not is_distinct_entity(["proteosome"], "proteasome", KNOWN)
    assert not is_distinct_entity(["asperate"], "aspartate", KNOWN)
    assert not is_distinct_entity(["cortical", "spinal", "tract"], "corticospinal tract", KNOWN)
    assert not is_distinct_entity(["spinothalmic"], "spinothalamic", KNOWN | {"spinothalamic"})


def test_the_scan_no_longer_offers_those_valid_terms(cfg):
    from src.medical_report import scan_possible_errors
    from src.schema import Segment
    from src.vocabulary import build_index

    vocab = build_index("all", "en")
    texts = ["Another target of the protein kinase B is the factor FOXO.", "And if they are phosphorylated, this allows the autophagy.",
             "Remember the phosphatidyl inositol-3 kinase signalling.", "The transcriptional factor moves to the nucleus."]
    segs = [Segment(i * 5.0, i * 5.0 + 4.0, " " + t, "en", "m", "b") for i, t in enumerate(texts)]
    found = {e.original.lower() for e in scan_possible_errors(segs, vocab, min_score=0.0)}
    assert not found & {"protein kinase b", "phosphorylated", "inositol-3", "transcriptional factor"}
    segs2 = [Segment(0.0, 4.0, " The proteosome degrades the ubiquitinated protein.", "en", "m", "b")]
    assert any(e.original.lower() == "proteosome" for e in scan_possible_errors(segs2, vocab, min_score=0.0))
