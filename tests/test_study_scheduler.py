"""Spaced repetition and the flashcard progress: the SM-2 schedule, due / difficult decks, the review log, older progress files, and progress that survives new study material."""

import json
from datetime import date, timedelta

import pytest

from src.study.models import Flashcard, Question, StudyPack
from src.study.scheduler import MAX_INTERVAL, MIN_EASE, START_EASE, Schedule, days_overdue, is_due, next_schedule, replay, when
from src.study.session import FlashcardSession, build_deck
from src.study.store import LEARNING, MASTERED, NEW, REVIEW_LOG, WEAK, Progress, StudyStore

DAY = date(2026, 3, 2)


def card(cid, front=None, topic="A"):
    return Flashcard(cid, "concept", front or f"Why does the concept {cid} behave the way it does in the cell?", f"Because of {cid}.", [], "medium", topic)


def pack_of(*cards, questions=()):
    return StudyPack("l", "t", "transcript", flashcards=list(cards), questions=list(questions))


# ------------------------------------------------------------------------------------------------------------------ the algorithm
def test_sm2_intervals_for_a_card_that_is_always_known():
    s, days, today = Schedule(), [], DAY
    for _ in range(5):
        s = next_schedule(s, "know", today)
        days.append(s.interval)
        assert s.due == (today + timedelta(days=s.interval)).isoformat()
        today += timedelta(days=s.interval)
    assert days[:3] == [1, 6, 17]                                    # 1 day, 6 days, then 6 x ease (2.5 + 3 x 0.1 = 2.8)
    assert days == sorted(days) and days[3] > days[2]               # the gaps only grow while the card is known
    assert s.ease == pytest.approx(START_EASE + 0.5) and s.reps == 5 and s.lapses == 0


def test_unsure_is_a_pass_that_costs_ease_and_know_is_a_pass_that_adds_ease():
    start = Schedule(ease=2.5, interval=6, reps=2, due=DAY.isoformat())
    assert next_schedule(start, "unsure", DAY).ease == pytest.approx(2.5 + 0.1 - 2 * (0.08 + 2 * 0.02))      # SM-2 quality 3: -0.14
    assert next_schedule(start, "know", DAY).ease == pytest.approx(2.6)                                       # quality 5: +0.1
    assert next_schedule(start, "unsure", DAY).interval < next_schedule(start, "know", DAY).interval
    assert next_schedule(start, "unsure", DAY).reps == 3


def test_dont_know_is_a_lapse_the_card_is_due_today_and_starts_over():
    start = Schedule(ease=2.5, interval=17, reps=3, lapses=0, due=DAY.isoformat())
    lapsed = next_schedule(start, "dont_know", DAY)
    assert (lapsed.interval, lapsed.reps, lapsed.lapses, lapsed.due) == (0, 0, 1, DAY.isoformat())
    assert lapsed.ease == pytest.approx(2.3)
    assert is_due(lapsed, DAY)
    again = next_schedule(lapsed, "know", DAY)
    assert again.interval == 1 and again.reps == 1                    # after the lapse the climb begins again: 1 day, 6 days ...


def test_ease_never_falls_below_the_floor_and_the_interval_is_capped():
    s = Schedule()
    for _ in range(20):
        s = next_schedule(s, "dont_know", DAY)
    assert s.ease == MIN_EASE and s.lapses == 20
    big = Schedule(ease=2.9, interval=300, reps=9)
    assert next_schedule(big, "know", DAY).interval == MAX_INTERVAL


def test_an_unknown_rating_is_refused():
    with pytest.raises(ValueError):
        next_schedule(Schedule(), "great", DAY)


def test_due_and_overdue():
    s = Schedule(interval=1, reps=1, due="2026-03-01")
    assert is_due(s, DAY) and days_overdue(s, DAY) == 1
    assert not is_due(s, date(2026, 2, 28)) and days_overdue(s, date(2026, 2, 28)) == 0
    assert not is_due(Schedule(), DAY)                              # never reviewed: new, not due


def test_a_damaged_schedule_is_read_as_a_fresh_one():
    assert Schedule.from_dict({"ease": "x", "interval": None}) == Schedule()
    assert Schedule.from_dict(None) == Schedule()
    assert Schedule.from_dict({"ease": 0.2}).ease == MIN_EASE


def test_intervals_in_words():
    assert [when(n) for n in (0, 1, 6, 21, 90)] == ["today", "tomorrow", "in 6 days", "in 3 weeks", "in 3 months"]


# ------------------------------------------------------------------------------------------------------------------ progress on disk
def test_rating_a_card_sets_its_schedule_and_review_log_and_survives_a_restart(tmp_path):
    path = tmp_path / "study" / "study_progress.json"
    p = Progress(path=path)
    p.rate_card("c1", "know", "A", DAY)
    p.rate_card("c1", "know", "A", DAY + timedelta(days=1))
    p.rate_card("c2", "dont_know", "A", DAY)
    p.save()
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["version"] == 2 and raw["cards"]["c1"]["sr"]["interval"] == 6 and raw["cards"]["c1"]["reviews"] == 2
    again = Progress.load(path)                                       # "closing and reopening the program"
    assert again.schedule("c1").due == (DAY + timedelta(days=7)).isoformat()
    assert again.review_history("c1") == [{"on": "2026-03-02", "rating": "know"}, {"on": "2026-03-03", "rating": "know"}]
    assert again.card_state("c1") == MASTERED and again.card_state("c2") == WEAK
    assert not again.is_due("c1", DAY + timedelta(days=6)) and again.is_due("c1", DAY + timedelta(days=7))
    assert again.is_due("c2", DAY)                                    # the lapse is due the same day


def test_the_review_log_is_kept_in_full_up_to_its_limit_while_the_state_uses_the_last_ratings(tmp_path):
    p = Progress()
    for n in range(REVIEW_LOG + 15):
        p.rate_card("c", "know" if n % 2 else "unsure", "A", DAY)
    assert len(p.review_history("c")) == REVIEW_LOG and p.cards["c"]["reviews"] == REVIEW_LOG + 15 and len(p.cards["c"]["ratings"]) == 6


def test_a_progress_file_from_before_spaced_repetition_is_read_and_upgraded_when_rated(tmp_path):
    path = tmp_path / "study_progress.json"
    path.write_text(json.dumps({"version": 1, "cards": {"old": {"ratings": ["know", "know", "know"], "topic": "A", "last": "2026-02-20T10:00:00Z"},
                                                          "weak": {"ratings": ["dont_know"], "topic": "A", "last": "2026-02-20T10:00:00Z"}},
                                "questions": {"q1": {"results": ["correct"], "topic": "A"}}, "sessions": []}), encoding="utf-8")
    p = Progress.load(path)
    assert p.card_state("old") == MASTERED and p.card_state("weak") == WEAK       # nothing the student did is lost
    assert p.schedule("old", DAY).due == "2026-03-09"                             # three passes in a row, the last on 20 Feb: 1 day, 6 days, then 17 days after the last review
    assert p.is_due("weak", DAY) and not p.is_due("never", DAY)
    p.rate_card("old", "know", "A", DAY)
    assert p.cards["old"]["reviews"] == 4 and "sr" in p.cards["old"] and p.questions["q1"]["results"] == ["correct"]


def test_decks_for_due_and_difficult_cards(tmp_path):
    cards = [card("c1"), card("c2"), card("c3"), card("c4")]
    pack = pack_of(*cards)
    p = Progress(path=tmp_path / "p.json")
    p.rate_card("c1", "know", "A", DAY - timedelta(days=10))          # due 9 days ago
    p.rate_card("c2", "know", "A", DAY - timedelta(days=2))           # due yesterday
    p.rate_card("c3", "know", "A", DAY)                               # due tomorrow
    p.set_difficult("c4", True, "A")
    p.set_difficult("c3", True, "A")
    assert p.due_cards(pack, DAY) == ["c1", "c2"]                     # the longest forgotten first
    assert [c.id for c in build_deck(pack, p, only="due", shuffle=True, seed=3, today=DAY)] == ["c1", "c2"]
    assert {c.id for c in build_deck(pack, p, only="difficult", shuffle=False)} == {"c3", "c4"}
    assert p.study_plan(pack, DAY) == {"due": 2, "new": 1, "difficult": 2, "reviews_today": 1, "reviews": 3}
    assert p.next_due(pack, DAY) == DAY - timedelta(days=9)           # the earliest day a reviewed card comes due (c4 was never reviewed)
    assert Progress().next_due(pack) is None


def test_the_difficult_mark_is_the_students_own_and_changes_nothing_else(tmp_path):
    p = Progress(path=tmp_path / "p.json")
    p.set_difficult("ghost", False)
    assert "ghost" not in p.cards                                     # un-marking a card that was never touched leaves no entry
    p.rate_card("c", "know", "A", DAY)
    before = p.schedule("c")
    p.set_difficult("c", True, "A")
    assert p.is_difficult("c") and p.schedule("c") == before and p.card_state("c") == LEARNING
    p.save()
    assert Progress.load(tmp_path / "p.json").is_difficult("c")
    p.set_difficult("c", False)
    assert not p.is_difficult("c")


def test_marking_a_new_card_difficult_does_not_make_it_reviewed(tmp_path):
    p = Progress()
    p.set_difficult("c", True, "A")
    assert p.card_state("c") == NEW and not p.is_due("c", DAY) and p.schedule("c") == Schedule()


# ------------------------------------------------------------------------------------------------------------------ the session
def test_a_flashcard_session_goes_back_marks_difficult_and_lists_the_missed(tmp_path):
    cards = [card("c1"), card("c2"), card("c3")]
    p = Progress(path=tmp_path / "p.json")
    s = FlashcardSession(cards, p)
    assert not s.back()                                               # nothing before the first card
    s.rate("know", DAY)
    s.rate("dont_know", DAY)
    assert s.back() and s.current().id == "c2"
    assert s.toggle_difficult() is True and p.is_difficult("c2")
    assert Progress.load(tmp_path / "p.json").is_difficult("c2")      # kept on disk at once
    assert s.toggle_difficult() is False
    s.rate("dont_know", DAY)                                          # rated again: a second review
    assert p.cards["c2"]["reviews"] == 2 and [c.id for c in s.missed()] == ["c2"]
    s.rate("unsure", DAY)
    assert s.finished and s.toggle_difficult() is None


# ------------------------------------------------------------------------------------------------------------------ new material
def test_new_wording_of_a_studied_card_inherits_its_progress_and_nothing_is_deleted():
    old = pack_of(card("old1", "What does hexokinase do to glucose at the start of glycolysis?"), card("old2", "Why is the heart muscle rich in mitochondria?"),
                  card("old3", "Which vitamin does collagen synthesis depend on?"), questions=[
                      Question("oq1", "mcq", "Which enzyme phosphorylates glucose first?", "Hexokinase", "x", ["Hexokinase", "a", "b", "c"], [], topic="A")])
    p = Progress()
    p.rate_card("old1", "know", "A", DAY)
    p.rate_card("old1", "know", "A", DAY + timedelta(days=1))
    p.rate_card("old2", "dont_know", "A", DAY)
    p.set_difficult("old3", True, "A")                                # marked difficult, never rated
    p.record_answer("oq1", "incorrect", "A")
    new = pack_of(card("new1", "What does hexokinase do to glucose at the start of glycolysis"),                   # the same card, punctuation changed
                  card("new2", "How do the cells of the heart muscle obtain most of their ATP?"),                 # a different card
                  card("new3", "Which vitamin does collagen synthesis depend on?"), questions=[
                      Question("nq1", "mcq", "Which enzyme phosphorylates glucose first?", "Hexokinase", "x", ["Hexokinase", "a", "b", "c"], [], topic="A")])
    assert p.carry_over(old, new) == {"cards": 2, "questions": 1}
    assert p.card_state("new1") == MASTERED and p.cards["new1"]["sr"] == p.cards["old1"]["sr"] and p.cards["new1"]["carried_from"] == "old1"
    assert p.is_difficult("new3")
    assert "new2" not in p.cards                                      # a card about something else starts as a new card
    assert p.question_state("nq1") == WEAK
    assert all(key in p.cards for key in ("old1", "old2", "old3")) and "oq1" in p.questions   # the earlier material's progress is still in the file
    assert p.carry_over(old, new) == {"cards": 0, "questions": 0}     # asking again changes nothing: the new cards have a history now


def test_two_new_cards_never_inherit_from_the_same_old_card():
    old = pack_of(card("o", "What does hexokinase do to glucose?"))
    p = Progress()
    p.rate_card("o", "know", "A", DAY)
    new = pack_of(card("n1", "What does hexokinase do to glucose?"), card("n2", "What does hexokinase do to glucose ?"))
    assert p.carry_over(old, new)["cards"] == 1
    assert [cid for cid in ("n1", "n2") if cid in p.cards] == ["n1"]


def test_a_card_with_its_own_history_is_never_overwritten_by_an_older_one():
    old = pack_of(card("o", "What does hexokinase do to glucose?"))
    p = Progress()
    p.rate_card("o", "know", "A", DAY)
    p.rate_card("n", "dont_know", "A", DAY)
    new = pack_of(card("n", "What does hexokinase do to glucose?"))
    assert p.carry_over(old, new)["cards"] == 0 and p.card_state("n") == WEAK


def test_no_earlier_material_means_nothing_to_carry():
    assert Progress().carry_over(None, pack_of(card("a"))) == {"cards": 0, "questions": 0}


def test_progress_survives_the_study_store_round_trip(tmp_path):
    store = StudyStore(tmp_path)
    p = store.progress()
    p.rate_card("c", "unsure", "A", DAY)
    p.set_difficult("c", True, "A")
    p.save()
    again = StudyStore(tmp_path).progress()
    assert again.schedule("c").reps == 1 and again.is_difficult("c") and again.review_history("c")[0]["rating"] == "unsure"


def test_replay_matches_using_the_scheduler_all_along():
    ratings = ["know", "unsure", "dont_know", "know", "know"]
    s, day = Schedule(), DAY
    for n, rating in enumerate(ratings):
        s = next_schedule(s, rating, DAY if n == len(ratings) - 1 else DAY - timedelta(days=len(ratings) - n))
    assert replay(ratings, DAY) == s
