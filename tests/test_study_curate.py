"""FEATURE-06: the premium gate. Every card and question, whichever provider wrote it, must be a real question with a real answer that the lecture supports."""

from __future__ import annotations

import pytest

from src.study import curate
from src.study.models import Flashcard, Question, SourceRef
from tests.study_support import lecture_index


@pytest.fixture(scope="module")
def idx():
    return lecture_index()


def ref(idx, text):
    chunk = next(c for c in idx.chunks if text in c.text)
    return SourceRef.of(chunk)


def card(idx, front, back, anchor="Pyruvate kinase catalyzes", **kw):
    return Flashcard(id="c-" + str(abs(hash((front, back)))), type=kw.pop("type", "concept"), front=front, back=back, sources=[ref(idx, anchor)], **kw)


def mcq(idx, stem, answer, options, anchor="Pyruvate kinase catalyzes", explanation="Pyruvate kinase catalyzes the last step of glycolysis and produces pyruvate and ATP.", **kw):
    return Question(id="q-" + str(abs(hash(stem))), type=kw.pop("type", "mcq"), question=stem, answer=answer, options=options, explanation=explanation, sources=[ref(idx, anchor)], **kw)


# --------------------------------------------------------------------------------------------------------------------- flashcards
def test_a_good_card_is_kept_with_its_evidence_difficulty_and_moment(idx):
    good = card(idx, "Which enzyme catalyzes the last step of glycolysis and what does it produce?", "Pyruvate kinase; it produces pyruvate and ATP.")
    result = curate.curate_cards([good], idx)
    assert result.kept == [good] and not result.rejected
    assert "Pyruvate kinase catalyzes the last step" in good.evidence and good.generation == curate.GENERATION_VERSION
    assert good.difficulty in ("easy", "medium", "hard") and good.sources[0].start is not None                  # the moment of the lecture it comes from


@pytest.mark.parametrize("front,back,reason", [
    ("Fill in the blank: _____ catalyzes the last step of glycolysis.", "Pyruvate kinase catalyzes the last step.", "fill-in-the-blank"),
    ("What does the lecture say about pyruvate kinase?", "It catalyzes the last step of glycolysis.", "vague prompt"),
    ("What is the result of in muscle during intense exercise lactate?", "A drop in pH.", "garbled question"),
    ("It catalyzes which step of glycolysis in the cytoplasm?", "The last step of glycolysis.", "not self-contained"),
    ("Which enzyme?", "Pyruvate kinase.", "trivial"),
    ("What is the last step of glycolysis catalyzed by pyruvate kinase?", "The last step of glycolysis catalyzed by pyruvate kinase.", "answer in question"),
])
def test_bad_cards_are_refused_for_the_right_reason(idx, front, back, reason):
    result = curate.curate_cards([card(idx, front, back)], idx)
    assert not result.kept and reason in result.rejected[0].reasons


def test_a_statement_the_lecture_does_not_make_is_refused(idx):
    invented = card(idx, "Which enzyme catalyzes the last step of glycolysis and how fast is it?", "Pyruvate kinase, at 450 reactions per second in every cell.")
    result = curate.curate_cards([invented], idx)
    assert not result.kept and result.rejected[0].reasons[0] in ("unsupported", "evidence not in the lecture")


def test_quoted_evidence_must_really_be_in_the_cited_passage(idx):
    forged = card(idx, "Which enzyme catalyzes the last step of glycolysis and what does it produce?", "Pyruvate kinase; it produces pyruvate and ATP.",
                  evidence="Pyruvate kinase is destroyed by insulin during the fasting state of the liver.")
    assert curate.curate_cards([forged], idx).rejected[0].reasons == ["evidence not in the lecture"]


def test_a_what_does_x_do_card_must_say_what_it_does(idx):
    mismatch = card(idx, "What does glycolysis do?", "Glycolysis happens in the cytoplasm and not in the mitochondria.", anchor="Do not forget that glycolysis happens")
    assert "answer does not say what it does" in curate.curate_cards([mismatch], idx).rejected[0].reasons


def test_a_definition_must_define_the_term_that_was_asked(idx):
    wrong = card(idx, "What is collagen?", "Collagen type one is the main collagen of bone, skin and tendon.", anchor="Collagen type one is the main collagen")
    assert "answer is about a different term" in curate.curate_cards([wrong], idx).rejected[0].reasons


def test_near_identical_cards_are_kept_once(idx):
    a = card(idx, "Which enzyme catalyzes the last step of glycolysis and what does it produce?", "Pyruvate kinase; it produces pyruvate and ATP.")
    b = card(idx, "Which enzyme catalyzes the last step of glycolysis, and what does it produce?", "Pyruvate kinase, which produces pyruvate and ATP.")
    result = curate.curate_cards([a, b], idx)
    assert result.kept == [a] and result.rejected[0].reasons == ["near-identical to another card"]


def test_difficulty_follows_what_answering_takes_not_what_the_writer_claimed(idx):
    recall = card(idx, "Which enzyme catalyzes the last step of glycolysis and what does it produce?", "Pyruvate kinase; it produces pyruvate and ATP.", type="definition", difficulty="hard")
    curate.curate_cards([recall], idx)
    assert recall.difficulty == "easy"
    joined = card(idx, "How does the lack of pyruvate kinase explain the anemia of red blood cells that lack mitochondria?",
                  "Red blood cells have no mitochondria, so they depend on glycolysis for ATP; without pyruvate kinase the last step fails and they break down.", type="integration",
                  anchor="Pyruvate kinase deficiency causes hemolytic anemia")
    joined.sources.append(ref(idx, "red blood cells have no mitochondria"))
    curate.curate_cards([joined], idx)
    assert joined.difficulty == "hard"


# --------------------------------------------------------------------------------------------------------------------- questions
def test_a_good_multiple_choice_question_is_kept(idx):
    q = mcq(idx, "Which enzyme catalyzes the last step of glycolysis?", "Pyruvate kinase", ["Pyruvate kinase", "Hexokinase", "Enolase", "Glucokinase"])
    result = curate.curate_questions([q], idx)
    assert result.kept == [q] and q.evidence and q.generation == curate.GENERATION_VERSION


@pytest.mark.parametrize("stem,options,reason", [
    ("Which term correctly completes this statement? _____ catalyzes the last step of glycolysis.", ["Pyruvate kinase", "Hexokinase", "Enolase", "Glucokinase"], "fill-in-the-blank"),
    ("Which enzyme catalyzes the last step of glycolysis?", ["Pyruvate kinase", "Hexokinase", "Enolase", "All of the above"], "all/none of the above"),
    ("Which enzyme catalyzes the last step of glycolysis?", ["Pyruvate kinase", "Pyruvate kinase", "Enolase", "Glucokinase"], "duplicate options"),
    ("Which enzyme catalyzes the last step of glycolysis?", ["Pyruvate kinase", "Hexokinase", "Enolase"], "needs four or five options"),
    ("Which enzyme catalyzes the last step of glycolysis?", ["Pyruvate kinase", "Pyruvate", "Enolase", "Glucokinase"], "a distractor contains the answer"),
    ("Which enzyme catalyzes the last step of glycolysis?", ["Pyruvate kinase", "A very long answer that has nothing in common with the others in size or in kind at all, truly", "Enolase", "Glucokinase"],
     "options differ a lot in length"),
])
def test_bad_multiple_choice_questions_are_refused(idx, stem, options, reason):
    q = mcq(idx, stem, "Pyruvate kinase", options)
    result = curate.curate_questions([q], idx)
    assert not result.kept and reason in result.rejected[0].reasons


def test_a_distractor_that_the_evidence_also_makes_correct_is_refused(idx):
    """"Which enzyme ... produces pyruvate and ATP?" - if ATP-producing phosphoglycerate kinase is among the options AND in the same evidence, there is no single best answer."""
    q = mcq(idx, "Which molecule is produced in the last step of glycolysis?", "ATP", ["ATP", "Pyruvate", "Glucose", "NADH"],
            explanation="Pyruvate kinase produces pyruvate and ATP in the last step of glycolysis.")
    assert "a distractor is also in the evidence" in curate.curate_questions([q], idx).rejected[0].reasons


def test_the_answer_may_not_be_in_the_question(idx):
    q = mcq(idx, "Pyruvate kinase catalyzes the last step of glycolysis - what is the name of this enzyme: pyruvate kinase?", "Pyruvate kinase",
            ["Pyruvate kinase", "Hexokinase", "Enolase", "Glucokinase"])
    assert "answer in question" in curate.curate_questions([q], idx).rejected[0].reasons


def test_a_question_without_an_explanation_is_refused(idx):
    q = mcq(idx, "Which enzyme catalyzes the last step of glycolysis?", "Pyruvate kinase", ["Pyruvate kinase", "Hexokinase", "Enolase", "Glucokinase"], explanation="")
    assert "no explanation" in curate.curate_questions([q], idx).rejected[0].reasons


def test_a_true_false_flood_is_capped(idx):
    sentences = ["Pyruvate kinase catalyzes the last step of glycolysis and produces pyruvate and ATP.", "Net glycolysis gives two ATP and two NADH per glucose molecule.",
                 "GLUT4 is the insulin-dependent glucose transporter of muscle and adipose tissue.", "Collagen is the most abundant structural protein of the extracellular matrix.",
                 "Glycine is found at every third position of the collagen chain because only glycine fits inside the triple helix.", "Fibroblasts are the cells that synthesize most of the collagen in connective tissue."]
    items = [Question(id=f"t{i}", type="true_false", question=f"True or false: {s}", answer="True", options=["True", "False"], explanation=f"True. The lecture says: {s}",
                      sources=[ref(idx, s[:30])]) for i, s in enumerate(sentences)]
    result = curate.curate_questions(items, idx)
    assert 1 <= len(result.kept) < len(items) and "too many true/false questions" in result.rejected[0].reasons


def test_a_question_that_repeats_a_flashcard_is_refused_but_a_different_one_about_the_same_enzyme_is_kept(idx):
    fronts = ["Which enzyme catalyzes the last step of glycolysis?"]
    options = ["Pyruvate kinase", "Hexokinase", "Enolase", "Glucokinase"]
    same = mcq(idx, "Which of the following enzymes catalyzes the last step of glycolysis?", "Pyruvate kinase", options)
    other = mcq(idx, "Why does pyruvate kinase deficiency make red blood cells fragile?", "They cannot make enough ATP without glycolysis", [
        "They cannot make enough ATP without glycolysis", "They accumulate too much pyruvate", "They lose their haemoglobin", "They cannot take up oxygen"],
        anchor="Pyruvate kinase deficiency causes hemolytic anemia", explanation="Pyruvate kinase deficiency causes hemolytic anemia because red blood cells depend on glycolysis for ATP.")
    result = curate.curate_questions([same, other], idx, unlike=fronts)
    assert [q.id for q in result.kept] == [other.id] and result.rejected[0].reasons == ["repeats a flashcard"]


def test_short_definitions_of_different_terms_are_different_asks_but_the_same_term_twice_is_one():
    ask = curate._ask
    assert not curate._alike(ask("What is systolic heart failure?"), ask("What is heart failure?"), curate.NEAR_DUPLICATE)      # one more word: a subtype, another concept
    assert not curate._alike(ask("What is the nephron loop?"), ask("What is the nephron?"), curate.NEAR_DUPLICATE)
    assert curate._alike(ask("What is heart failure?"), ask("What is the heart failure?"), curate.NEAR_DUPLICATE)
    long_a, long_b = "Which enzyme catalyzes the last step of glycolysis?", "Which enzyme catalyzes the last step of glycolysis, and what does it produce?"
    assert curate._alike(ask(long_a), ask(long_b), curate.NEAR_DUPLICATE)                                                    # everything else is judged on the words of the whole front
    assert not curate._alike(ask(long_a), ask("Why does pyruvate kinase deficiency cause hemolytic anemia?"), curate.NEAR_DUPLICATE)


def test_a_what_happens_card_must_say_what_happens(idx):
    asks = "What happens when pyruvate kinase is missing from red blood cells?"
    restates = card(idx, asks, "Pyruvate kinase catalyzes the last step of glycolysis.")
    says = card(idx, asks, "Pyruvate kinase deficiency causes hemolytic anemia because red blood cells depend on glycolysis for ATP.", anchor="Pyruvate kinase deficiency causes hemolytic anemia")
    assert "answer does not say what follows" in curate.review_card(restates, idx).reasons
    assert "answer does not say what follows" not in curate.review_card(says, idx).reasons


def test_exam_phrasing_and_hyphens_do_not_make_two_fronts_different():
    ask = curate._ask
    assert curate._alike(ask("Which of the following is a common symptom of left-sided heart failure?"), ask("What are the typical symptoms of left sided heart failure?"), curate.CARD_REPEAT)
    assert not curate._alike(ask("Which of the following is a characteristic of diastolic heart failure?"), ask("How do systolic and diastolic heart failure differ in terms of ejection fraction?"),
                             curate.CARD_REPEAT)


def test_two_questions_about_the_same_sentence_are_one_question_asked_twice(idx):
    options = ["Pyruvate kinase", "Hexokinase", "Enolase", "Glucokinase"]
    first = mcq(idx, "Which enzyme catalyzes the last step of glycolysis?", "Pyruvate kinase", options)
    again = mcq(idx, "In the final reaction of the glycolytic pathway, which protein converts phosphoenolpyruvate into the product?", "Pyruvate kinase", [
        "Pyruvate kinase", "Phosphofructokinase", "Aldolase", "Lactate dehydrogenase"])
    result = curate.curate_questions([first, again], idx)
    assert result.kept == [first] and result.rejected[0].reasons == ["same fact as another question"]


def test_two_vignettes_that_only_change_the_first_sentence_are_one_question(idx):
    options = ["Pyruvate kinase", "Hexokinase", "Enolase", "Glucokinase"]
    stem = "A six year old child is brought to the clinic with pallor, jaundice and fatigue, and the blood tests show hemolytic anemia. Which enzyme deficiency is the most likely cause of the anemia in this child?"
    kw = dict(type="clinical_vignette", anchor="Pyruvate kinase deficiency causes hemolytic anemia",
              explanation="Pyruvate kinase deficiency causes hemolytic anemia because red blood cells depend on glycolysis for ATP.")
    a = mcq(idx, stem, "Pyruvate kinase", options, **kw)
    b = mcq(idx, "A seven year old girl is brought to the clinic with pallor, jaundice and fatigue, and the blood tests show hemolytic anemia. Which enzyme deficiency is the most likely cause of the anemia in this girl?",
            "Pyruvate kinase", options[::-1], **kw)
    result = curate.curate_questions([a, b], idx)
    assert result.kept == [a] and result.rejected[0].reasons[0] in ("near-identical to another question", "same fact as another question")


def test_a_true_false_statement_of_a_few_words_is_refused(idx):
    short = Question(id="t-short", type="true_false", question="Pyruvate kinase is a glycolytic enzyme.", answer="True", options=["True", "False"],
                     explanation="True. Pyruvate kinase catalyzes the last step of glycolysis and produces pyruvate and ATP.", sources=[ref(idx, "Pyruvate kinase catalyzes")])
    assert "statement too short" in curate.review_question(short, idx).reasons


def test_a_clinical_vignette_needs_a_real_scenario(idx):
    short = mcq(idx, "Which enzyme is deficient?", "Pyruvate kinase", ["Pyruvate kinase", "Hexokinase", "Enolase", "Glucokinase"], type="clinical_vignette",
                anchor="Pyruvate kinase deficiency causes hemolytic anemia", explanation="Pyruvate kinase deficiency causes hemolytic anemia because red blood cells depend on glycolysis for ATP.")
    assert "vignette too short" in curate.curate_questions([short], idx).rejected[0].reasons


# --------------------------------------------------------------------------------------------------------------------- what a model adds on its own
def test_an_answer_that_adds_more_than_the_lecture_says_is_refused(idx):
    """The answer keeps some of the lecture's words and adds a consequence the lecture never states - 4.1.2 requires 75 percent of the answer to be in the passage."""
    padded = card(idx, "What happens to the pathway when pyruvate kinase is missing in a red blood cell?",
                  "Pyruvate kinase deficiency causes hemolytic anemia, leading to jaundice, splenomegaly, gallstones and a lifelong need for blood transfusions.",
                  anchor="Pyruvate kinase deficiency causes hemolytic anemia")
    result = curate.curate_cards([padded], idx)
    assert not result.kept and result.rejected[0].reasons[0] in ("unsupported", "evidence not in the lecture")


def test_a_word_the_medical_pass_doubted_cannot_carry_a_card_or_a_question(idx):
    idx.unresolved = ["nuclides"]
    try:
        misheard = card(idx, "Which nuclides of the glycolytic pathway produce pyruvate and ATP in the last step?", "Pyruvate kinase produces pyruvate and ATP in the last step.")
        assert "rests on a misheard word" in curate.curate_cards([misheard], idx).rejected[0].reasons
        q = mcq(idx, "Which of the nuclides catalyzes the last step of glycolysis?", "Pyruvate kinase", ["Pyruvate kinase", "Hexokinase", "Enolase", "Glucokinase"])
        assert "rests on a misheard word" in curate.curate_questions([q], idx).rejected[0].reasons
        fine = card(idx, "Which enzyme catalyzes the last step of glycolysis and what does it produce?", "Pyruvate kinase; it produces pyruvate and ATP.")
        assert curate.curate_cards([fine], idx).kept == [fine]
    finally:
        idx.unresolved = []


def test_a_pack_does_not_open_ten_items_with_the_same_words(idx):
    fronts = ["What is the clinical significance of pyruvate kinase deficiency in red blood cells?", "What is the clinical significance of vitamin C deficiency for collagen?",
              "What is the clinical significance of copper deficiency for lysyl oxidase?"]
    backs = [("Pyruvate kinase deficiency causes hemolytic anemia because red blood cells depend on glycolysis for ATP.", "Pyruvate kinase deficiency causes hemolytic anemia"),
             ("Vitamin C deficiency causes scurvy because collagen without hydroxyproline is unstable.", "Vitamin C deficiency causes scurvy"),
             ("Copper deficiency therefore reduces the strength of collagen.", "Copper deficiency therefore reduces")]
    cards = [card(idx, f, b, anchor=a) for f, (b, a) in zip(fronts, backs)]
    result = curate.curate_cards(cards, idx)
    assert len(result.kept) == curate.SAME_OPENING_MAX and result.rejected[0].reasons == ["same opening as other cards"]


def test_evidence_quoted_with_a_masked_word_shows_the_lectures_own_sentence(idx):
    masked = card(idx, "Which enzyme catalyzes the last step of glycolysis and what does it produce?", "Pyruvate kinase; it produces pyruvate and ATP.",
                  evidence="Pyruvate [unclear] catalyzes the last step of glycolysis and produces pyruvate and ATP.")
    result = curate.curate_cards([masked], idx)
    assert result.kept == [masked]
    assert "[unclear]" not in masked.evidence.lower() and "Pyruvate kinase catalyzes the last step" in masked.evidence        # what the lecture says, not what the model was shown
    assert masked.sources[0].chunk_id == "t0001" or masked.sources[0].start is not None


@pytest.mark.parametrize("front,vague", [
    ("What is the role of the forebrain in the regulation of the pathway?", True),
    ("What is the effect of the damage in the system?", True),
    ("What is the role of the forebrain in the regulation of the visceromotor pathway?", False),
    ("What is the sequence of events in the renin-angiotensin-aldosterone system?", False),
    ("Which enzyme catalyzes the last step of the pathway of glycolysis?", False),
])
def test_a_question_that_ends_on_an_unnamed_pathway_or_system_is_not_self_contained(idx, front, vague):
    reasons = curate.review_card(card(idx, front, "Pyruvate kinase catalyzes the last step of glycolysis, and it produces pyruvate and ATP."), idx).reasons
    assert ("not self-contained" in reasons) is vague


def _why(idx, stem, answer, explanation, anchor):
    return Question(id="q-" + str(abs(hash(stem))), type="explain_why", question=stem, answer=answer, explanation=explanation, options=[], sources=[ref(idx, anchor)])


def test_a_why_question_needs_a_reason_in_the_lecture_not_one_the_model_made_up(idx):
    invented = _why(idx, "Why is pyruvate kinase needed in glycolysis?", "It catalyzes the last step of glycolysis and produces pyruvate and ATP.",
                    "Pyruvate kinase catalyzes the last step of glycolysis and produces pyruvate and ATP.", "Pyruvate kinase catalyzes")
    stated = _why(idx, "Why does pyruvate kinase deficiency cause hemolytic anemia?", "Because red blood cells depend on glycolysis for ATP.",
                  "Pyruvate kinase deficiency causes hemolytic anemia because red blood cells depend on glycolysis for ATP.", "Pyruvate kinase deficiency causes hemolytic anemia")
    assert "the lecture gives no reason for this" in curate.review_question(invented, idx).reasons
    assert "the lecture gives no reason for this" not in curate.review_question(stated, idx).reasons


def test_an_answer_that_says_because_needs_a_because_in_the_sentence_it_rests_on(idx):
    made_up = card(idx, "What does pyruvate kinase do at the end of glycolysis?", "It produces pyruvate and ATP, because it is the last enzyme of the pathway.")
    quoted = card(idx, "What does a lack of pyruvate kinase do to red blood cells?", "It causes hemolytic anemia, because red blood cells depend on glycolysis for ATP.",
                  anchor="Pyruvate kinase deficiency causes hemolytic anemia")
    assert "the lecture gives no reason for this" in curate.review_card(made_up, idx).reasons
    assert "the lecture gives no reason for this" not in curate.review_card(quoted, idx).reasons


def test_a_cause_in_the_neighbouring_sentence_of_a_quote_does_not_excuse_an_invented_reason(idx):
    two = ("Pyruvate kinase catalyzes the last step of glycolysis and produces pyruvate and ATP. "
           "Pyruvate kinase deficiency causes hemolytic anemia because red blood cells depend on glycolysis for ATP.")
    q = _why(idx, "Why does pyruvate kinase produce pyruvate and ATP?", "Because it catalyzes the last step of glycolysis.",
             "Pyruvate kinase catalyzes the last step of glycolysis and produces pyruvate and ATP.", "Pyruvate kinase catalyzes")
    q.evidence = two
    assert "the lecture gives no reason for this" in curate.review_question(q, idx).reasons


def test_a_long_masked_quote_is_matched_to_the_lectures_words_and_never_left_with_the_mask(idx):
    chunk = next(c for c in idx.chunks if "Pyruvate kinase catalyzes" in c.text)
    start = chunk.text.index("Pyruvate kinase catalyzes")
    stretch = " ".join(chunk.text[start:].split())[:230]                                                              # two sentences of the lecture, in a row
    words = stretch.split()
    masked_quote = " ".join(words[:1] + ["[unclear]"] + words[2:])                                                    # "kinase" hidden from the model
    q = card(idx, "Which enzyme catalyzes the last step of glycolysis and what does it produce?", "Pyruvate kinase; it produces pyruvate and ATP.", evidence=masked_quote)
    curate.attach_evidence(q, idx)
    assert "unclear" not in q.evidence.lower() and q.evidence.startswith("Pyruvate kinase catalyzes the last step of glycolysis")      # the lecture's own words, wildcard for the masked one
    nowhere = card(idx, "Which enzyme catalyzes the last step of glycolysis and what does it produce?", "Pyruvate kinase; it produces pyruvate and ATP.",
                   evidence="Something [unclear] the lecture never said at all in any place.")
    curate.attach_evidence(nowhere, idx)
    assert "[unclear]" not in nowhere.evidence and "..." in nowhere.evidence                                          # no match anywhere: the gap is shown as a gap
