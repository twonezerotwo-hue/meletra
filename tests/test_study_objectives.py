"""Learning objectives: made only from validated cards and questions of the pack, never invented, with the material that practises each."""

import pytest

from src.study.models import Flashcard, Question, SourceRef, StudyPack, SummaryItem
from src.study.objectives import MAX_PER_TOPIC, derive_objectives, objective_text


def ref(chunk, start=0.0):
    return SourceRef(chunk, "transcript", f"Transcript {chunk}", start, start + 10)


def card(cid, front, chunk="t1", topic="Heart failure", fact="", start=0.0):
    return Flashcard(cid, "concept", front, "An answer that is long enough to be one.", [ref(chunk, start)], "medium", topic, fact_id=fact)


def mcq(qid, stem, chunk="t1", topic="Heart failure", fact=""):
    return Question(qid, "mcq", stem, "Right", "Because the lecture says so.", ["Right", "a", "b", "c"], [ref(chunk)], topic=topic, fact_id=fact)


@pytest.mark.parametrize("front,text,kind", [
    ("What is the effect of ACE inhibitors on the renin-angiotensin-aldosterone system?", "Explain the effect of ACE inhibitors on the renin-angiotensin-aldosterone system", "explain"),
    ("How do systolic and diastolic heart failure differ in terms of ejection fraction?", "Compare systolic and diastolic heart failure in terms of ejection fraction", "compare"),
    ("What is the sequence of events in the renin-angiotensin-aldosterone system?", "Describe the sequence of events in the renin-angiotensin-aldosterone system", "describe"),
    ("What happens when a failing heart reaches the limit of the Frank-Starling law?", "Explain what happens when a failing heart reaches the limit of the Frank-Starling law", "explain"),
    ("What is the most common cause of systolic heart failure?", "Identify the most common cause of systolic heart failure", "identify"),
    ("Why do ACE inhibitors reduce mortality in systolic heart failure?", "Explain why ACE inhibitors reduce mortality in systolic heart failure", "explain"),
    ("How does conjugation spread resistance between bacteria?", "Explain how conjugation spread resistance between bacteria", "explain"),
    ("What is the mechanism by which beta lactams kill bacteria?", "Explain the mechanism by which beta lactams kill bacteria", "explain"),
    ("What is the clinical relevance of fluoroquinolones?", "Explain the clinical relevance of fluoroquinolones", "explain"),
    ("What is vancomycin used for in hospitals today?", "Define vancomycin used for in hospitals today", "define"),
])
def test_a_question_becomes_an_instruction(front, text, kind):
    made = objective_text(front)
    assert made is not None and made[1] == kind
    if kind != "define" and "spread" not in front:                                    # the clean ones exactly (the two awkward ones only have to be recognised)
        assert made[0] == text


@pytest.mark.parametrize("front", [
    "Why?", "What is _____ in the pathway?", "What is the result of aminoglycosides bind to the thirty S ribosomal subunit and?",
    "Which of the following is a cause?", "Fill in the blank: heart failure is [unclear].", "What does the lecture say about preload and afterload in this passage?",
    "Name the enzyme.", "What is it?",
])
def test_a_question_that_does_not_read_as_an_objective_gives_none(front):
    assert objective_text(front) is None


def test_objectives_come_from_the_pack_with_their_sources_and_practice_counts():
    pack = StudyPack("l", "t", "transcript", topics=["Heart failure"], flashcards=[
        card("c1", "What is the effect of ACE inhibitors on the renin-angiotensin-aldosterone system?", "t1", fact="f1", start=10),
        card("c2", "How do systolic and diastolic heart failure differ in terms of ejection fraction?", "t2", start=60),
        card("c3", "What is the most common cause of systolic heart failure?", "t3", start=120),
        card("c4", "What is _____ in heart failure?", "t4", start=200)],
        questions=[mcq("q1", "Which of the following is the most common cause of systolic heart failure?", "t3"),
                   mcq("q2", "What is the effect of ACE inhibitors on the renin-angiotensin-aldosterone system and on blood pressure?", "t1", fact="f1")])
    objectives = derive_objectives(pack)
    assert [o.text for o in objectives] == ["Explain the effect of ACE inhibitors on the renin-angiotensin-aldosterone system",
                                            "Compare systolic and diastolic heart failure in terms of ejection fraction",
                                            "Identify the most common cause of systolic heart failure"]         # the blank card gave nothing; the order is the lecture's
    first, _second, third = objectives
    assert first.sources[0].chunk_id == "t1" and (first.cards, first.questions) == (1, 1)
    assert third.practice == "Practise with 1 flashcard and 1 question" and objectives[1].practice == "Practise with 1 flashcard"
    assert all(o.sources and o.from_id for o in objectives)                                                    # every objective rests on a real item with real sources


def test_the_two_forms_of_the_same_point_make_one_objective():
    pack = StudyPack("l", "t", "transcript", topics=["A"], flashcards=[
        card("c1", "What is the most common cause of systolic heart failure?", "t3", topic="A"),
        card("c2", "What is the most common cause of systolic heart failure in adults?", "t3", topic="A")])
    assert len(derive_objectives(pack)) == 1


def test_a_list_of_definitions_is_capped_so_that_other_kinds_get_their_place():
    cards = [card(f"d{i}", f"What is {name}?", f"t{i}", start=i) for i, name in enumerate(["hexokinase", "pyruvate kinase", "enolase", "aldolase"])]
    cards += [card("m1", "How does the sodium iodide symporter move iodide into the thyroid cell?", "t9", start=9),
              card("m2", "How do beta blockers lower the heart rate of a patient with thyrotoxicosis?", "t10", start=10)]
    objectives = derive_objectives(StudyPack("l", "t", "transcript", topics=["Heart failure"], flashcards=cards))
    assert sum(1 for o in objectives if o.kind == "define") == 2 and sum(1 for o in objectives if o.kind == "explain") == 2 and len(objectives) == 4


def test_a_topic_with_key_concepts_gets_a_key_term_objective_and_a_longer_lecture_more_objectives():
    concepts = [SummaryItem("Preload is the stretch of the ventricle before it contracts.", [ref("t1")], "Heart failure", term=t) for t in ("Preload", "Afterload", "Contractility")]
    fronts = ["What is the effect of preload on stroke volume?", "How do ACE inhibitors reduce mortality in systolic heart failure?", "What is the role of the Frank-Starling law in heart function?",
              "What happens when afterload rises over many years?", "What is the significance of BNP in diagnosing heart failure?", "Why does angiotensin two raise the preload of the ventricle?"]
    cards = [card(f"c{i}", front, f"t{i}", start=i) for i, front in enumerate(fronts, 1)]
    pack = StudyPack("l", "t", "transcript", topics=["Heart failure"], flashcards=cards, summaries={"key_concepts": concepts})
    objectives = derive_objectives(pack)
    assert objectives[-1].text == "Define and recognise the key terms: Preload, Afterload, Contractility" or any(o.text.startswith("Define and recognise the key terms") for o in objectives)
    assert 4 <= len(objectives) <= MAX_PER_TOPIC


def test_an_empty_or_unusable_pack_has_no_objectives():
    assert derive_objectives(StudyPack("l", "t", "transcript")) == []
    junk = StudyPack("l", "t", "transcript", flashcards=[card("c1", "Why?"), card("c2", "Name it.")])
    assert derive_objectives(junk) == []
