"""19-area rule-based subject classifier: filename + early transcript + slides + --subject -> 0..1 scores."""

import pytest

from src.classifier import SubjectClassifier, filename_areas, vocab_file_scores
from src.knowledge_subjects import AREA_VOCAB_FILES, AREAS
from src.vocabulary import DEFAULT_TERMS_DIR, available_subjects

NEURO = ("The corticospinal tract descends through the internal capsule. The medial lemniscus carries touch, and the "
         "spinothalamic tract carries pain. Lesions of the substantia nigra affect the basal ganglia.")
BIOCHEM = ("In glycolysis, hexokinase and phosphofructokinase are regulated. Pyruvate dehydrogenase links glycolysis to "
           "the citric acid cycle, producing NADH and FADH2 for the electron transport chain.")
PHARM = ("Propranolol is a beta blocker. The pharmacokinetics include absorption, volume of distribution and clearance. "
         "ACE inhibitors such as lisinopril lower blood pressure.")


@pytest.fixture(scope="module")
def clf():
    return SubjectClassifier(DEFAULT_TERMS_DIR)


def test_there_are_nineteen_areas_and_every_one_maps_to_real_vocabulary_files():
    assert len(AREAS) == 19 and len(set(AREAS)) == 19
    files = set(available_subjects(DEFAULT_TERMS_DIR))
    for area in AREAS:
        assert AREA_VOCAB_FILES[area] and set(AREA_VOCAB_FILES[area]) <= files, area
        assert len(AREA_VOCAB_FILES[area]) <= 3


@pytest.mark.parametrize("text,expected", [(NEURO, {"neuroscience", "neuroanatomy"}), (BIOCHEM, {"biochemistry"}),
                                           (PHARM, {"pharmacology"})])
def test_early_transcript_picks_the_right_area(clf, text, expected):
    scores = clf.score(early_text=text)
    assert scores[0].area in expected
    selected = clf.select(scores)
    assert 1 <= len(selected) <= 3 and selected[0].area in expected


def test_scores_are_between_zero_and_one_and_sorted(clf):
    scores = clf.score(early_text=NEURO, pdf_text=NEURO, filename="neuro_week2.mp4", user_subjects=["neuroscience"])
    assert all(0.0 <= s.score <= 1.0 for s in scores)
    assert [s.score for s in scores] == sorted((s.score for s in scores), reverse=True)
    assert scores[0].score > 0.5                                    # four independent kinds of evidence agree


def test_no_evidence_means_no_selected_area(clf):
    scores = clf.score(early_text="Welcome everybody, let us begin. Please sit down and open your notebooks.")
    assert clf.select(scores) == []
    assert all(s.score == 0.0 for s in scores)


def test_file_name_alone_is_a_hint_but_not_overconfident(clf):
    scores = clf.score(filename="Biochem_lecture3_glycolysis.mp4")
    assert scores[0].area == "biochemistry"
    assert 0 < scores[0].score < 0.6
    assert "biochemistry" in filename_areas("Biochem lecture 3.mp4")
    assert filename_areas("WhatsApp Audio 2026-10-10 at 11.14.04.mp4") == []        # nothing to learn from this name


def test_user_subject_accepts_areas_and_vocabulary_files(clf):
    by_area = clf.score(user_subjects=["cardiology"])
    assert by_area[0].area == "cardiology"
    by_file = clf.score(user_subjects=["pharmacology"])
    assert by_file[0].area == "pharmacology"


def test_evidence_is_explained(clf):
    top = clf.score(early_text=BIOCHEM)[0]
    assert top.evidence and any("glycolysis" in e or "hexokinase" in e for e in top.evidence)
    assert "early_transcript" in top.by_source


def test_selection_keeps_only_areas_close_to_the_best_one(clf):
    scores = clf.score(early_text=BIOCHEM)
    selected = clf.select(scores, max_areas=3, relative_cutoff=0.5)
    assert all(s.score >= 0.5 * scores[0].score for s in selected)


def test_selected_areas_translate_to_vocabulary_file_weights(clf):
    selected = clf.select(clf.score(early_text=NEURO))
    weights = vocab_file_scores(selected)
    assert weights["neuroscience"] > 0 and all(0 < w <= 1 for w in weights.values())
    assert "pharmacology" not in weights
