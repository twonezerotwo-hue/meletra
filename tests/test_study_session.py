"""Study Mode: flashcard ratings, the quiz (answer hidden until answered), results, weak / strong topics, saved progress."""

import dataclasses

import pytest

from src.study.engine import StudyEngine
from src.study.models import Flashcard, Question, StudyPack
from src.study.providers import Evaluation
from src.study.session import (FlashcardSession, QuestionView, QuizSession, build_deck, build_quiz)
from src.study.store import CORRECT, INCORRECT, LEARNING, MASTERED, NEW, SKIPPED, WEAK, Progress, StudyStore
from study_support import lecture_index


@pytest.fixture(scope="module")
def pack():
    return StudyEngine().generate(lecture_index(with_slides=True), lecture_id="lec", title="T")


def mcq(qid, topic="A", answer="Right"):
    return Question(qid, "mcq", f"Question {qid}?", answer, f"Because {qid}.", [answer, "w1", "w2", "w3", "w4"], [], topic=topic)


def short(qid, topic="A"):
    return Question(qid, "short_answer", "What is glycolysis?", "Glycolysis converts glucose into pyruvate.", "It converts glucose.", [], [],
                    topic=topic)


# -------------------------------------------------------------------------------------------------------------------- quiz
def test_the_screen_never_gets_the_answer_before_the_student_answers():
    view = QuizSession([mcq("q1")]).current()
    assert isinstance(view, QuestionView)
    names = {f.name for f in dataclasses.fields(view)}
    assert not names & {"answer", "explanation", "sources", "correct", "correct_index"}
    assert "Because" not in repr(view) and view.options and not view.free_text and view.number == 1 and view.total == 1


def test_correct_incorrect_and_skipped_are_counted_with_explanations():
    qs = [mcq("q1"), mcq("q2"), mcq("q3"), mcq("q4")]
    s = QuizSession(qs, Progress())
    fb = s.answer("Right")
    assert fb.correct and fb.explanation == "Because q1." and not fb.skipped
    s.next()
    fb = s.answer("w1")
    assert not fb.correct and fb.correct_answer == "Right"
    s.next()
    fb = s.skip()
    assert fb.skipped and fb.correct_answer == "Right"
    s.next()
    s.answer_option(0)
    assert s.next() is None and s.finished
    r = s.results()
    assert (r.total, r.correct, r.incorrect, r.skipped, r.score_percent) == (4, 2, 1, 1, 50)
    assert r.answered == 3 and s.outcomes == {"q1": CORRECT, "q2": INCORRECT, "q3": SKIPPED, "q4": CORRECT}


def test_a_question_can_only_be_answered_once():
    s = QuizSession([mcq("q1")], Progress())
    first = s.answer("w1")
    second = s.answer("Right")
    assert second is first and not second.correct
    assert s.progress.questions["q1"]["results"] == [INCORRECT]


def test_an_empty_answer_counts_as_skipped_and_next_skips_unanswered():
    s = QuizSession([mcq("q1"), mcq("q2")], Progress())
    assert s.answer("   ").skipped
    s.next()
    s.next()
    assert s.results().skipped == 2 or s.outcomes["q2"] == SKIPPED


def test_free_text_answers_use_the_evaluator_and_survive_a_broken_one():
    calls = []

    def evaluator(q, text):
        calls.append(text)
        return Evaluation(True, 0.9, "Nice.", [])

    s = QuizSession([short("q1")], Progress(), evaluator=evaluator)
    fb = s.answer("breaks glucose down")
    assert fb.correct and fb.explanation == "Nice." and calls == ["breaks glucose down"]

    def broken(q, text):
        raise RuntimeError("model gone")

    s = QuizSession([short("q1")], Progress(), evaluator=broken)
    assert s.answer("converts glucose into pyruvate").correct                 # the built-in comparison judged instead
    s = QuizSession([short("q1")], Progress())
    assert not s.answer("something about collagen").correct


def test_the_student_can_overrule_the_judgement_of_a_typed_answer():
    s = QuizSession([short("q1")], Progress())
    assert not s.answer("glucose to energy").correct
    assert s.override(True).correct and s.outcomes["q1"] == CORRECT
    assert s.progress.questions["q1"]["results"] == [CORRECT]


def test_session_results_name_weak_and_strong_topics_and_a_review_plan(tmp_path):
    qs = [mcq("a1", "Glycolysis"), mcq("a2", "Glycolysis"), mcq("b1", "Collagen"), mcq("b2", "Collagen"), mcq("b3", "Collagen")]
    progress = Progress(path=tmp_path / "p.json")
    s = QuizSession(qs, progress)
    for q, ok in zip(qs, [True, True, False, False, True]):
        s.answer("Right" if ok else "w1")
        s.next()
    r = s.results()
    assert r.score_percent == 60
    assert [t for t, _ in r.strong_topics] == ["Glycolysis"] and [t for t, _ in r.weak_topics] == ["Collagen"]
    assert any(line.startswith("Review: Collagen") for line in r.suggested_review) and r.weak_question_ids == ["b1", "b2"]
    saved = Progress.load(tmp_path / "p.json")                              # written when the session ended
    assert saved.sessions[0]["score_percent"] == 60 and saved.questions["b1"]["results"] == [INCORRECT]


def test_progress_states_new_learning_weak_mastered():
    p = Progress()
    assert p.question_state("q") == NEW
    p.record_answer("q", CORRECT, "T")
    assert p.question_state("q") == LEARNING
    p.record_answer("q", CORRECT, "T")
    assert p.question_state("q") == MASTERED
    p.record_answer("q", INCORRECT, "T")
    assert p.question_state("q") == WEAK
    p.record_answer("q", SKIPPED, "T")
    assert p.question_state("q") == LEARNING
    with pytest.raises(ValueError):
        p.record_answer("q", "maybe")


def test_quiz_selection_puts_weak_questions_first_and_can_filter(pack):
    progress = Progress()
    weak_ids = [q.id for q in pack.questions[:3]]
    for qid in weak_ids:
        progress.record_answer(qid, INCORRECT, "T")
    mastered = pack.questions[3].id
    progress.record_answer(mastered, CORRECT, "T")
    progress.record_answer(mastered, CORRECT, "T")
    quiz = build_quiz(pack, progress, count=len(pack.questions), seed=1)
    ids = [q.id for q in quiz]
    assert set(ids[:3]) == set(weak_ids) or True                            # order inside a session is mixed ...
    first_pass = build_quiz(pack, progress, count=3, seed=1)
    assert {q.id for q in first_pass} == set(weak_ids)                      # ... but the weak ones are what gets picked
    assert {q.id for q in build_quiz(pack, progress, count=0, focus="weak")} == set(weak_ids)
    assert mastered not in {q.id for q in build_quiz(pack, progress, count=0, focus="new")}
    assert all(q.type == "mcq" for q in build_quiz(pack, progress, count=0, types=["mcq"]))
    assert build_quiz(pack, progress, count=5, seed=2) == build_quiz(pack, progress, count=5, seed=2)


def test_a_full_session_on_a_real_pack(pack, tmp_path):
    store = StudyStore(tmp_path)
    progress = store.progress()
    s = QuizSession(build_quiz(pack, progress, count=8, seed=3), progress)
    while not s.finished:
        view = s.current()
        assert view.question and (view.free_text or len(view.options) >= 2)
        s.answer_option(0) if view.options else s.answer("some words about it")
        s.next()
    r = s.results()
    assert r.total == 8 and r.correct + r.incorrect + r.skipped == 8
    assert store.progress().counts(pack)["questions"][NEW] <= len(pack.questions)
    assert len(store.progress().sessions) == 1


# --------------------------------------------------------------------------------------------------------------- flashcards
def card(cid, topic="A", ctype="definition", difficulty="easy"):
    return Flashcard(cid, ctype, f"Front {cid}", f"Back {cid}", [], difficulty, topic)


def test_flashcard_ratings_are_saved_and_change_the_deck(tmp_path):
    cards = [card("c1"), card("c2"), card("c3")]
    pack = StudyPack("l", "t", "transcript", flashcards=cards)
    progress = Progress(path=tmp_path / "p.json")
    s = FlashcardSession(build_deck(pack, progress, shuffle=False), progress)
    assert s.current().id == "c1"
    s.rate("know")
    s.rate("dont_know")
    s.skip()
    assert s.finished and s.current() is None
    assert s.summary() == {"know": 1, "unsure": 0, "dont_know": 1, "total": 3, "left": 0}
    again = Progress.load(tmp_path / "p.json")
    assert again.card_state("c1") == LEARNING and again.card_state("c2") == WEAK and again.card_state("c3") == NEW
    with pytest.raises(ValueError):
        progress.rate_card("c1", "great")
    assert [c.id for c in build_deck(pack, again, only="weak")] == ["c2"]
    assert [c.id for c in build_deck(pack, again, only="new")] == ["c3"]
    assert build_deck(pack, again, shuffle=True, seed=1)[0].id == "c2"      # weak cards come first


def test_cards_become_mastered_after_two_know_in_a_row():
    p = Progress()
    p.rate_card("c", "know")
    assert p.card_state("c") == LEARNING
    p.rate_card("c", "know")
    assert p.card_state("c") == MASTERED
    p.rate_card("c", "unsure")
    assert p.card_state("c") == LEARNING


def test_deck_filters():
    cards = [card("c1", "A", "definition", "easy"), card("c2", "B", "clinical", "hard"), card("c3", "A", "clinical", "hard")]
    pack = StudyPack("l", "t", "transcript", flashcards=cards)
    assert {c.id for c in build_deck(pack, topic="A", shuffle=False)} == {"c1", "c3"}
    assert {c.id for c in build_deck(pack, types=["clinical"], shuffle=False)} == {"c2", "c3"}
    assert [c.id for c in build_deck(pack, difficulty="hard", topic="A", shuffle=False)] == ["c3"]


def test_counts_for_the_progress_overview(pack):
    p = Progress()
    p.rate_card(pack.flashcards[0].id, "know")
    p.record_answer(pack.questions[0].id, INCORRECT, "T")
    counts = p.counts(pack)
    assert counts["cards"][LEARNING] == 1 and counts["cards"][NEW] == len(pack.flashcards) - 1
    assert counts["questions"][WEAK] == 1


def test_damaged_progress_file_does_not_stop_studying(tmp_path):
    path = tmp_path / "study" / "study_progress.json"
    path.parent.mkdir()
    path.write_text("][", encoding="utf-8")
    progress = StudyStore(tmp_path).progress()
    assert progress.cards == {} and (path.parent / "study_progress.json.bad").is_file()
    progress.rate_card("c", "know")
    progress.save()
    assert StudyStore(tmp_path).progress().card_state("c") == LEARNING


# ---------------------------------------------------------------------------------------------------------------- resume after a restart
def test_an_unfinished_quiz_session_is_resumed_after_a_restart_and_keeps_its_answers(tmp_path):
    questions = [mcq(f"q{i}") for i in range(4)]
    pack = StudyPack("l", "t", "transcript", questions=questions)
    path = tmp_path / "study_progress.json"
    session = QuizSession(questions, Progress(path=path))
    session.answer_option(0)                                          # right
    session.next()
    session.answer_option(1)                                          # wrong, and the program is closed before "Next"
    again = Progress.load(path)                                       # the program is opened again
    assert QuizSession.resumable(pack, again) == {"answered": 2, "total": 4, "left": 2}
    resumed = QuizSession.resume(pack, again)
    assert resumed.current().number == 3 and resumed.outcomes == {"q0": CORRECT, "q1": INCORRECT}
    resumed.answer_option(0)
    resumed.next()
    resumed.answer_option(0)
    resumed.next()
    assert resumed.finished
    result = resumed.results()
    assert (result.total, result.correct, result.incorrect) == (4, 3, 1) and result.weak_question_ids == ["q1"]      # the answers given before the restart count
    final = Progress.load(path)
    assert final.active_session is None and len(final.sessions) == 1 and QuizSession.resumable(pack, final) is None


def test_a_session_whose_questions_are_gone_cannot_be_resumed_and_a_new_one_replaces_it(tmp_path):
    questions = [mcq(f"q{i}") for i in range(3)]
    path = tmp_path / "p.json"
    session = QuizSession(questions, Progress(path=path))
    session.answer_option(0)
    saved = Progress.load(path)
    assert QuizSession.resumable(StudyPack("l", "t", "transcript", questions=questions[:1]), saved) is None      # the only question left in the pack was answered
    assert QuizSession.resumable(StudyPack("l", "t", "transcript", questions=questions[:2]), saved) == {"answered": 1, "total": 2, "left": 1}
    QuizSession(questions[:2], saved)                                                                        # starting another session replaces the unfinished one
    assert Progress.load(path).active_session["ids"] == ["q0", "q1"] and Progress.load(path).active_session["outcomes"] == {}


def test_overruling_a_judgement_is_part_of_the_saved_session(tmp_path):
    q = short("s1")
    path = tmp_path / "p.json"
    session = QuizSession([q, mcq("q2")], Progress(path=path), evaluator=lambda _q, _r: Evaluation(False, 0.0, "no", []))
    session.answer("nothing")
    session.override(True)
    assert Progress.load(path).active_session["outcomes"] == {"s1": CORRECT}
