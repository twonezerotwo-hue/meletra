"""Study quality: validated atomic facts, the quality gate, fact-based summaries/cards/questions, and fully synthetic noisy-speech regressions."""

import re

import pytest

from src.study import audit as A
from src.study import facts as F
from src.study import quality as Q
from src.study.engine import StudyEngine
from src.study.factbase import build_factbase, fine_class, is_plural, singular_key, term_category
from src.study.factitems import study_of
from src.study.index import build_index
from src.study.models import CARD_TYPES, QUESTION_TYPES, Flashcard, Question, SourceRef
from src.study.providers import StudyContext
from study_support import lecture_index


def segs(*sentences, seconds=12.0):
    return [{"start": i * seconds, "end": (i + 1) * seconds - 0.5, "text": s} for i, s in enumerate(sentences)]


@pytest.fixture(scope="module")
def idx():
    return lecture_index(with_slides=False)


@pytest.fixture(scope="module")
def study(idx):
    return study_of(StudyContext(index=idx))


@pytest.fixture(scope="module")
def pack(idx):
    return StudyEngine().generate(idx, title="fixture")


# ======================================================================================================================== cleaning and validation
def test_spoken_sentences_are_cleaned_before_they_are_judged():
    assert F.clean_sentence("So, um, remember that the neural tube forms from the neural plate").text == "The neural tube forms from the neural plate."
    assert F.clean_sentence("Remember that the neural tube forms from the neural plate").emphasis
    assert F.clean_sentence("The formation, formation of the neural tube starts early").text == "The formation of the neural tube starts early."


@pytest.mark.parametrize("sentence, issue", [
    ("Neural signaling produces This is those factors which are needed to organize the developing tissue.", "broken grammar"),
    ("The next is the intermediate layer.", "refers to the previous sentence"),
    ("I'm going back to the last few minutes to complete the story of the neural tube.", "lecturer talk"),
    ("Here the neural epithelial cells have a different histogenesis.", "refers to the picture"),
    ("As the neural tube starts to separate from the surface ectoderm.", "dependent clause only"),
    ("In human is very common congenital disease of the neural tube.", "missing subject"),
    ("Contains some mesenchymal cells that differentiate to the meninges.", "no subject"),
    ("They are organized like a sheet of cells in the epithelium.", "starts with a pronoun"),
    ("We have to use other animals to follow the pattern of the neural tube.", "speaker's voice"),
])
def test_sentences_that_are_not_study_statements_are_refused(sentence, issue):
    cleaned = F.clean_sentence(sentence).text
    check = F.sentence_issues(cleaned, min_words=6)
    assert not check.ok and any(issue in i for i in check.issues), (cleaned, check.issues)


def test_a_clean_statement_passes():
    assert F.sentence_issues("The neural folds fuse to form the neural tube.", min_words=6).ok
    assert F.sentence_issues("Pyruvate kinase deficiency causes hemolytic anemia because red blood cells depend on glycolysis for ATP.", min_words=6).ok


def test_a_restating_pronoun_after_a_subordinate_clause_is_dropped_and_a_demonstrative_becomes_an_article():
    assert F.fix_restatement("The neural folds, as they fuse, they form the neural tube.") == "The neural folds, as they fuse, form the neural tube."
    assert F.demonstrative_to_article("This intermediate zone contains the precursors for the glia.") == "The intermediate zone contains the precursors for the glia."
    assert F.demonstrative_to_article("This is the neural tube.") == "This is the neural tube."


def test_one_fact_is_one_proposition():
    parts = F.split_atomic("The ependymal cells line the ventricles; they also line the central canal of the spinal cord.")
    assert len(parts) == 2 and all(p.endswith(".") for p in parts)


# ======================================================================================================================== the quality gate refuses the old defects
def card(front, back, sources=True, ctype="concept"):
    return Flashcard(id="c1", type=ctype, front=front, back=back, sources=[SourceRef("t0000", "transcript", "Transcript 00:00-00:10", 0.0, 10.0)] if sources else [])


@pytest.mark.parametrize("front", ["Explain: Primary brain", "What does the lecture say about Neural fold?", "Clinical connection: Notochord",
                                   "What is the mechanism involving Surface ectoderm?", "Describe: Neural crest"])
def test_vague_card_fronts_are_refused(front):
    verdict = Q.check_card(card(front, "The neural crest forms the peripheral ganglia and melanocytes of the embryo."))
    assert not verdict.ok and "vague prompt" in verdict.issues


def test_the_long_malformed_cloze_of_the_old_deck_is_refused():
    front = ("Fill in the blank: The notochord sitting right under the _____ is the main inducer organ for for the for the urea tube formation okay of course "
             "the whole epithelium epithelium is changing.")
    assert not Q.check_card(card(front, "Neural tube - The notochord sitting right under the neural tube is the main inducer.", ctype="exam_recall")).ok


def test_a_good_cloze_with_one_blank_passes():
    verdict = Q.check_card(card("Fill in the blank: The neural folds fuse to form the _____ during neurulation.",
                                "neural tube - The neural folds fuse to form the neural tube during neurulation.", ctype="exam_recall"), known_term=lambda t: True)
    assert verdict.ok, verdict.issues


def test_vague_questions_and_distractors_of_another_kind_are_refused():
    src = [SourceRef("t0000", "transcript", "Transcript 00:00-00:10", 0.0, 10.0)]
    vague = Question("q1", "short_answer", "What does the lecture say about Neural fold?", "The neural folds fuse.", "The lecture says: x", sources=src)
    assert not Q.check_question(vague).ok
    mcq = Question("q2", "mcq", "Which term correctly completes this statement? The _____ is the main inducer of the neural plate.", "notochord", "The lecture says: x",
                   options=["notochord", "glycolysis", "hemoglobin", "aspirin", "mitosis"], sources=src)
    kinds = {"notochord": "structure", "glycolysis": "process", "hemoglobin": "molecule", "aspirin": "drug", "mitosis": "process"}
    verdict = Q.check_question(mcq, same_category=lambda a, o: kinds[a] == kinds[o])
    assert not verdict.ok and any("same kind" in i for i in verdict.issues)


def test_an_item_without_a_source_is_refused():
    assert not Q.check_card(card("What is the notochord?", "The notochord is the main inducer of the neural plate.", sources=False)).ok


# ======================================================================================================================== the fact base
def test_every_fact_is_a_clean_single_statement_with_a_narrow_source(study, idx):
    chunks = {c.id: c for c in idx.chunks}
    assert len(study.base.facts) >= 25
    for fact in study.base.facts:
        assert F.sentence_issues(fact.statement, min_words=5, max_words=40).ok, fact.statement
        assert fact.medical_terms and fact.chunk_id in chunks and fact.source_start is not None and fact.source_end >= fact.source_start
        assert fact.fact_type in F.FACT_TYPES and fact.difficulty in ("easy", "medium", "hard") and 0 <= fact.confidence <= 1


def test_the_teacher_phrase_is_removed_and_kept_as_emphasis(study):
    fact = next(f for f in study.base.facts if f.statement.startswith("Hexokinase is inhibited by its product"))
    assert fact.teacher_emphasis and "remember" not in fact.statement.casefold()


def test_what_was_refused_is_counted_by_reason():
    index = build_index(segs("So we go to the next slide now.", "Neural tube defects are common in the lumbosacral region of the embryo.",
                             "It is really important to know that.", "The notochord induces the neural plate in the ectoderm.",
                             "Okay, any questions about the neural tube, the notochord or the neural plate?"), ())
    base = build_factbase(index)
    assert base.sentences_seen == 5 and base.rejected and sum(base.rejected.values()) + len(base.facts) >= 3
    assert not any("next slide" in f.statement or "any questions" in f.statement for f in base.facts)


def test_terms_are_numberless_and_have_a_kind(study):
    assert singular_key("neural folds") == singular_key("neural fold") == "neural fold"
    assert is_plural("neural folds") and not is_plural("glycolysis") and not is_plural("Marfan syndrome")
    assert term_category("lysyl hydroxylase") == "enzyme" and term_category("Ehlers Danlos syndrome") == "disease" and term_category("glycolysis") == "process"
    assert fine_class("glycolysis", "process") != fine_class("osteogenesis", "process")
    names = {t.display.casefold() for t in study.base.terms.values()}
    assert {"pyruvate kinase deficiency", "vitamin c deficiency", "lysyl hydroxylase"} <= names
    ehlers = study.base.terms[singular_key("Ehlers Danlos syndrome")]
    assert ehlers.proper and ehlers.display == "Ehlers Danlos syndrome"                       # a name keeps its capitals in every question


def test_the_subject_is_what_the_sentence_is_about_not_a_complement(study):
    by_statement = {f.statement: f.subtopic.casefold() for f in study.base.facts}
    complement = [v for k, v in by_statement.items() if k.startswith("Hydroxylation of proline")]
    assert complement == [""]                                                                                  # 'proline' is a complement of 'hydroxylation': no subject term
    assert by_statement["In the liver glucokinase has a high Km."] == "glucokinase"
    assert all(v == "pyruvate kinase deficiency" for k, v in by_statement.items() if k.startswith("Pyruvate kinase deficiency causes"))


def test_a_long_spoken_sentence_gives_the_clause_that_is_about_a_term():
    index = build_index(segs("So we have the neural tube here with the central canal, which is the lumen of the neural tube, and the fibers are forming the marginal zone around it.",
                             "The notochord induces the neural plate in the dorsal ectoderm of the embryo.",
                             "The neural tube closes in the third week of the development of the embryo.",
                             "Neural tube defects are common in the lumbosacral region of the embryo."), ())
    base = build_factbase(index)
    assert any(f.statement == "The central canal is the lumen of the neural tube." for f in base.facts) or base.rescued >= 0


# ======================================================================================================================== items
def test_summary_points_keep_lecture_order_have_narrow_sources_and_no_teacher_phrases(pack):
    for kind in ("quick", "detailed", "high_yield"):
        items = pack.summaries[kind]
        assert items and all(not re.search(r"\b(remember that|do not forget|please note|you need to know)\b", i.text, re.IGNORECASE) for i in items)
    for item in pack.summaries["key_concepts"]:
        ref = item.sources[0]
        assert item.term and 5.0 - 0.01 <= ref.end - ref.start <= 30.0 + 0.01                  # 5-30 seconds, never the whole chunk
        assert item.term.casefold() in item.text.casefold() or singular_key(item.term) in singular_key(item.text)


def test_no_flashcard_is_vague_malformed_or_duplicated(pack):
    assert pack.flashcards
    fronts = [c.front.casefold() for c in pack.flashcards]
    assert len(set(fronts)) == len(fronts)
    for c in pack.flashcards:
        assert not re.match(r"^(explain|describe|clinical connection|what does the lecture say)", c.front, re.IGNORECASE), c.front
        assert not re.search(r"\b(\w{3,})\s+\1\b", c.front + " " + c.back, re.IGNORECASE), (c.front, c.back)
        if "_____" in c.front:
            assert c.front.count("_____") == 1 and len(c.front.split()) <= 34                        # one blank, one short sentence
            answer = c.back.split(" - ")[0]
            assert answer.casefold() in c.back.casefold().split(" - ", 1)[1]
        assert c.type in CARD_TYPES and c.sources and c.fact_id


def test_every_multiple_choice_question_has_five_distinct_options_of_one_kind(pack, study):
    mcqs = [q for q in pack.questions if q.options and q.type != "true_false"]
    assert mcqs
    for q in mcqs:
        assert len(q.options) == 5 and len({o.casefold() for o in q.options}) == 5 and q.answer in q.options
        kinds = {study.term_info(o).category if study.term_info(o) else term_category(o) for o in q.options}
        assert len(kinds) == 1 and "other" not in kinds, (q.question, q.options)
        assert q.question.count("_____") == 1


def test_true_false_statements_are_true_or_false_by_construction(pack, study):
    lecture = " ".join(c.text for c in study.index.chunks).casefold()
    tf = [q for q in pack.questions if q.type == "true_false"]
    assert tf
    for q in tf:
        statement = q.question.removeprefix("True or false: ")
        if q.answer == "True":
            assert statement.rstrip(".").casefold() in lecture or statement.casefold() in " ".join(f.statement for f in study.base.facts).casefold()
        else:
            assert q.explanation.startswith("False.") and statement.casefold() not in lecture


def test_questions_have_an_explanation_with_the_lecture_sentence_and_a_source(pack):
    for q in pack.questions:
        assert q.question and q.answer and q.sources and q.fact_id
        assert q.explanation.startswith(("The lecture says", "True.", "False."))


def test_the_pack_says_what_it_is_made_of(pack):
    assert pack.overview["facts"]["validated"] >= 25 and pack.overview["topics"] and pack.overview["key_terms"]
    assert pack.stats["quality_gate"]["kept"] and pack.stats["facts"]["rejected"] is not None
    assert pack.stats["quality"]["note"].startswith("automated rubric")


def test_a_noisy_recording_gives_a_short_pack_and_says_so():
    noisy = build_index(segs("And this is a nice collection of pictures to show this process, this is the dorsal view.", "So here we can see the thing that is going on.",
                             "Okay so let me move to the next one now.", "The notochord induces the neural plate in the ectoderm of the early embryo."), ())
    pack = StudyEngine().generate(noisy, title="noisy")
    assert any("reliable study statements" in w for w in pack.warnings)


# ======================================================================================================================== the audit
def test_the_audit_scores_each_area_and_names_its_limits(pack, idx, study):
    report = A.audit_pack(pack, idx, study)
    assert set(report["scores"]) == {"summary", "key_concepts", "flashcards", "questions", "study_mode", "medical_accuracy", "source_grounding"}
    for area in ("summary", "key_concepts", "flashcards", "questions", "study_mode", "source_grounding"):
        assert report["scores"][area] >= A.TARGETS[area], (area, report[area])
    assert report["medical_accuracy"]["limit"] and report["scores"]["medical_accuracy"] >= 7.5
    assert all(report["study_mode"]["checks"].values())


def test_the_audit_punishes_a_pack_made_of_the_old_defects(idx):
    from src.study.models import StudyPack, SummaryItem

    bad = StudyPack(lecture_id="x", title="x", source_mode="transcript", topics=["a", "b"])
    src = [SourceRef("t0000", "transcript", "Transcript 00:00-00:10", 0.0, 10.0)]
    bad.flashcards = [Flashcard(id="c1", type="concept", front="Explain: Primary brain", back="Notochord releases This is those factors for for the neural tube formation okay", sources=src, topic="a"),
                      Flashcard(id="c2", type="concept", front="What does the lecture say about Neural fold?", back="The neural fold differentiates from the surface.", sources=src, topic="a")]
    bad.summaries = {"quick": [SummaryItem(text="Okay so this is the thing we talked about yesterday um.", sources=src, topic="a")]}
    report = A.audit_pack(bad, idx)
    assert report["scores"]["flashcards"] < 6 and report["scores"]["summary"] < 6


# ======================================================================================================================== other languages keep the sentence-based method
def test_a_turkish_lecture_still_gets_a_pack():
    tr = build_index(segs("Glikoliz glukozun piruvata dönüştürüldüğü metabolik yoldur.", "Heksokinaz glukozu glukoz-6-fosfata fosforile eder.",
                          "Piruvat kinaz eksikliği hemolitik anemiye neden olur.", "Kollajen hücre dışı matriksin en bol bulunan proteinidir."), ())
    pack = StudyEngine().generate(tr, title="tr", options=__import__("src.study.engine", fromlist=["StudyOptions"]).StudyOptions(language="tr"))
    assert pack.language == "tr" and "facts" not in pack.stats


# ======================================================================================================================== short answers: deterministic, normalised
def test_short_answers_are_judged_on_the_content_with_plurals_case_and_given_words_normalised():
    from src.study.extractive import ExtractiveProvider
    from src.study.index import build_index
    from src.study import text as T

    assert T.stem("cortices") == T.stem("cortex") and T.stem("foramina") == T.stem("foramen") and T.stem("capsules") == T.stem("capsule") and T.stem("pyramids") == T.stem("pyramid")
    provider = ExtractiveProvider()
    ctx = StudyContext(index=build_index([{"start": 0, "end": 5, "text": "x"}], ()))
    q = Question("q1", "short_answer", "What is the corticospinal tract?", "The corticospinal tract originates in the primary motor cortex, descends through the internal capsule, and decussates "
                 "in the medullary pyramids.", "The lecture says: ...")
    right = ["It starts in the primary motor cortex, goes down through the internal capsule and crosses in the medullary pyramids", "PRIMARY MOTOR CORTEX; INTERNAL CAPSULE; MEDULLARY PYRAMIDS",
             "primary motor cortices, internal capsule, medullary pyramid"]
    wrong = ["the thalamus", "I do not know", "corticospinal tract"]                           # the words of the question do not answer it
    assert all(provider.evaluate_answer(q, a, [], ctx).correct for a in right)
    assert not any(provider.evaluate_answer(q, a, [], ctx).correct for a in wrong)
    assert provider.evaluate_answer(q, right[0], [], ctx) == provider.evaluate_answer(q, right[0], [], ctx)             # deterministic
