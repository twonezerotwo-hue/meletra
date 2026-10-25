"""Interactive study sessions: the flashcard deck and the question quiz ("Study Mode").

The quiz hands the interface a :class:`QuestionView` that contains **no answer, explanation or source**; those come back only in the
:class:`Feedback` returned by ``answer()`` / ``skip()``. A question can be answered once. Results (correct / incorrect / skipped,
weak and strong topics, what to review) are computed from the session and written to the progress file when it ends.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Callable, Optional, Sequence

from .models import RATINGS, Flashcard, Question, SourceRef, StudyPack
from .providers import Evaluation
from .store import CORRECT, INCORRECT, LEARNING, MASTERED, NEW, SKIPPED, WEAK, Progress

def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


Evaluator = Callable[[Question, str], Evaluation]
FOCUS_ALL, FOCUS_WEAK, FOCUS_NEW = "all", "weak", "new"
_PRIORITY = {WEAK: 0, NEW: 1, LEARNING: 2, MASTERED: 3}


# --------------------------------------------------------------------------------------------------------------------- flashcards
def build_deck(pack: StudyPack, progress: Optional[Progress] = None, *, topic: str = "", types: Sequence[str] = (),
               difficulty: str = "", only: str = FOCUS_ALL, shuffle: bool = True, seed: Optional[int] = None, today: Optional[date] = None) -> list[Flashcard]:
    """The cards to study. ``only``: all | new | weak | unmastered | due | difficult. Weak cards come first when shuffled; the due cards of a "due" deck are always
    ordered by how overdue they are (the longest forgotten first), because that is the point of a schedule."""
    progress = progress or Progress()
    due_order = {cid: place for place, cid in enumerate(progress.due_cards(pack, today))} if only == "due" else {}
    deck = []
    for card in pack.flashcards:
        if topic and card.topic != topic:
            continue
        if types and card.type not in types:
            continue
        if difficulty and card.difficulty != difficulty:
            continue
        state = progress.card_state(card.id)
        if only == "new" and state != NEW:
            continue
        if only == "weak" and state != WEAK:
            continue
        if only == "unmastered" and state == MASTERED:
            continue
        if only == "due" and card.id not in due_order:
            continue
        if only == "difficult" and not progress.is_difficult(card.id):
            continue
        deck.append((state, card))
    if only == "due":
        deck.sort(key=lambda sc: due_order[sc[1].id])
    elif shuffle:
        rnd = random.Random(seed)
        rnd.shuffle(deck)
        deck.sort(key=lambda sc: _PRIORITY[sc[0]])                            # stable: weak, new, learning, mastered
    return [c for _s, c in deck]


class FlashcardSession:
    def __init__(self, deck: Sequence[Flashcard], progress: Optional[Progress] = None) -> None:
        self.deck = list(deck)
        self.progress = progress or Progress()
        self.position = 0
        self.ratings: dict[str, str] = {}

    @property
    def finished(self) -> bool:
        return self.position >= len(self.deck)

    def current(self) -> Optional[Flashcard]:
        return None if self.finished else self.deck[self.position]

    def rate(self, rating: str, today: Optional[date] = None) -> None:
        """Know / Unsure / Don't know for the current card (its next review day is set by the scheduler), then on to the next."""
        card = self.current()
        if card is None:
            return
        self.progress.rate_card(card.id, rating, card.topic, today)
        self.ratings[card.id] = rating
        self.position += 1
        self.progress.save()                                                  # every rating is kept even if the app is closed now

    def skip(self) -> None:
        if not self.finished:
            self.position += 1

    def back(self) -> bool:
        """Go to the previous card (it can be looked at or rated again: a second rating is a second review). False at the first card."""
        if self.position <= 0:
            return False
        self.position -= 1
        return True

    def toggle_difficult(self) -> Optional[bool]:
        """Mark / unmark the current card as difficult (kept on disk at once). Returns the new mark, None when there is no current card."""
        card = self.current()
        if card is None:
            return None
        flag = not self.progress.is_difficult(card.id)
        self.progress.set_difficult(card.id, flag, card.topic)
        self.progress.save()
        return flag

    def missed(self) -> list[Flashcard]:
        """The cards of this session that were rated Don't know, in the order they were shown."""
        return [c for c in self.deck if self.ratings.get(c.id) == "dont_know"]

    def summary(self) -> dict[str, int]:
        out = {r: 0 for r in RATINGS}
        for r in self.ratings.values():
            out[r] += 1
        out["total"] = len(self.deck)
        out["left"] = max(len(self.deck) - self.position, 0)
        return out


# ------------------------------------------------------------------------------------------------------------------------- quiz
@dataclass
class QuestionView:
    """What the screen may show before the student has answered. Deliberately without answer, explanation and sources."""

    id: str
    type: str
    question: str
    options: list[str]
    number: int
    total: int
    topic: str
    difficulty: str

    @property
    def free_text(self) -> bool:
        return not self.options


@dataclass
class Feedback:
    correct: bool
    skipped: bool
    correct_answer: str
    explanation: str
    sources: list[SourceRef] = field(default_factory=list)
    score: float = 0.0
    missing: list[str] = field(default_factory=list)
    your_answer: str = ""


@dataclass
class SessionResult:
    total: int = 0
    correct: int = 0
    incorrect: int = 0
    skipped: int = 0
    score_percent: int = 0
    weak_topics: list[tuple[str, float]] = field(default_factory=list)
    strong_topics: list[tuple[str, float]] = field(default_factory=list)
    suggested_review: list[str] = field(default_factory=list)
    weak_question_ids: list[str] = field(default_factory=list)
    mastered: int = 0

    @property
    def answered(self) -> int:
        return self.correct + self.incorrect

    def to_dict(self) -> dict:
        return {"total": self.total, "correct": self.correct, "incorrect": self.incorrect, "skipped": self.skipped,
                "score_percent": self.score_percent, "weak_topics": self.weak_topics, "strong_topics": self.strong_topics,
                "suggested_review": self.suggested_review, "mastered": self.mastered}


def build_quiz(pack: StudyPack, progress: Optional[Progress] = None, *, count: int = 10, topic: str = "", types: Sequence[str] = (),
               difficulty: str = "", focus: str = FOCUS_ALL, seed: Optional[int] = None) -> list[Question]:
    """The questions of one session: weak ones first, then new, then the ones the student is still learning; mastered last."""
    progress = progress or Progress()
    pool = []
    for q in pack.questions:
        if topic and q.topic != topic:
            continue
        if types and q.type not in types:
            continue
        if difficulty and q.difficulty != difficulty:
            continue
        state = progress.question_state(q.id)
        if focus == FOCUS_WEAK and state != WEAK:
            continue
        if focus == FOCUS_NEW and state != NEW:
            continue
        pool.append((state, q))
    rnd = random.Random(seed)
    rnd.shuffle(pool)
    pool.sort(key=lambda sq: _PRIORITY[sq[0]])
    chosen = [q for _s, q in pool[:count]] if count else [q for _s, q in pool]
    rnd.shuffle(chosen)                                                       # the order inside a session is mixed
    return chosen


class QuizSession:
    def __init__(self, questions: Sequence[Question], progress: Optional[Progress] = None, *, evaluator: Optional[Evaluator] = None) -> None:
        self.questions = list(questions)
        self.progress = progress or Progress()
        self.evaluator = evaluator
        self.position = 0
        self.outcomes: dict[str, str] = {}
        self._feedback: dict[str, Feedback] = {}
        self._saved = False
        self._started = _stamp()
        self._checkpoint()                                                     # a session that is started can be picked up again after a restart

    # ---- resume after a restart
    @classmethod
    def resumable(cls, pack: StudyPack, progress: Progress) -> Optional[dict[str, int]]:
        """What an unfinished session has left: {"answered", "total", "left"}, or None when there is nothing to resume (none saved, or none of its questions is left in this pack)."""
        saved = progress.active_session
        if not saved:
            return None
        known = {q.id for q in pack.questions}
        ids = [i for i in saved.get("ids", []) if i in known]
        outcomes = {k: v for k, v in (saved.get("outcomes") or {}).items() if k in known}
        left = [i for i in ids if i not in outcomes]
        return {"answered": len(outcomes), "total": len(ids), "left": len(left)} if left else None

    @classmethod
    def resume(cls, pack: StudyPack, progress: Progress, *, evaluator: Optional[Evaluator] = None) -> Optional["QuizSession"]:
        """The unfinished session, on the first question that was not answered. The answers already given stay in its results."""
        if cls.resumable(pack, progress) is None:
            return None
        saved = progress.active_session or {}
        by_id = {q.id: q for q in pack.questions}
        questions = [by_id[i] for i in saved.get("ids", []) if i in by_id]
        session = cls.__new__(cls)
        session.questions, session.progress, session.evaluator = questions, progress, evaluator
        session.outcomes = {k: v for k, v in (saved.get("outcomes") or {}).items() if k in by_id}
        session._feedback, session._saved = {}, False
        session._started = str(saved.get("started") or _stamp())
        session.position = next(n for n, q in enumerate(questions) if q.id not in session.outcomes)
        return session

    def _checkpoint(self) -> None:
        self.progress.active_session = {"ids": [q.id for q in self.questions], "outcomes": dict(self.outcomes), "started": self._started}
        self.progress.save()

    @property
    def finished(self) -> bool:
        return self.position >= len(self.questions)

    def current(self) -> Optional[QuestionView]:
        if self.finished:
            return None
        q = self.questions[self.position]
        return QuestionView(id=q.id, type=q.type, question=q.question, options=list(q.options), number=self.position + 1,
                            total=len(self.questions), topic=q.topic, difficulty=q.difficulty)

    @property
    def answered_current(self) -> bool:
        return not self.finished and self.questions[self.position].id in self.outcomes

    def answer(self, response: str) -> Feedback:
        """Check the student's answer (option text, "True"/"False", or free text). Answering twice returns the first result."""
        if self.finished:
            raise RuntimeError("the session is finished")
        q = self.questions[self.position]
        if q.id in self._feedback:
            return self._feedback[q.id]
        response = (response or "").strip()
        if not response:
            return self.skip()
        if q.options:
            correct = response.casefold() == q.answer.casefold()
            feedback = Feedback(correct, False, q.answer, q.explanation, list(q.sources), 1.0 if correct else 0.0, [], response)
        else:
            evaluation = self._evaluate(q, response)
            feedback = Feedback(evaluation.correct, False, q.answer, evaluation.feedback or q.explanation, list(q.sources),
                                evaluation.score, evaluation.missing, response)
        return self._record(q, feedback, CORRECT if feedback.correct else INCORRECT)

    def answer_option(self, number: int) -> Feedback:
        """Answer with the option at ``number`` (0-based)."""
        q = self.questions[self.position]
        if not 0 <= number < len(q.options):
            raise IndexError("no such option")
        return self.answer(q.options[number])

    def skip(self) -> Feedback:
        q = self.questions[self.position]
        if q.id in self._feedback:
            return self._feedback[q.id]
        return self._record(q, Feedback(False, True, q.answer, q.explanation, list(q.sources)), SKIPPED)

    def override(self, correct: bool) -> Feedback:
        """The student disagrees with the automatic judgement of a typed answer."""
        q = self.questions[self.position]
        if q.id not in self._feedback:
            raise RuntimeError("answer the question first")
        feedback = self._feedback[q.id]
        feedback.correct, feedback.score = correct, 1.0 if correct else feedback.score
        self.outcomes[q.id] = CORRECT if correct else INCORRECT
        self.progress.questions[q.id]["results"][-1] = self.outcomes[q.id]
        self._checkpoint()
        return feedback

    def next(self) -> Optional[QuestionView]:
        if not self.finished:
            if self.questions[self.position].id not in self.outcomes:
                self.skip()
            self.position += 1
        if self.finished:
            self._finish()
        return self.current()

    def _evaluate(self, q: Question, response: str) -> Evaluation:
        if self.evaluator is None:
            from .extractive import ExtractiveProvider
            from .providers import StudyContext
            from .index import DocumentIndex

            return ExtractiveProvider().evaluate_answer(q, response, [], StudyContext(index=DocumentIndex([])))
        try:
            return self.evaluator(q, response)
        except Exception:                                                     # noqa: BLE001 - a quiz never stops on the evaluator
            self.evaluator = None
            return self._evaluate(q, response)

    def _record(self, q: Question, feedback: Feedback, outcome: str) -> Feedback:
        self._feedback[q.id] = feedback
        self.outcomes[q.id] = outcome
        self.progress.record_answer(q.id, outcome, q.topic)
        self._checkpoint()                                                    # the answer and the place in the session are on disk together
        return feedback

    def _finish(self) -> None:
        if not self._saved:
            self._saved = True
            self.progress.add_session(self.results().to_dict())
            self.progress.active_session = None
            self.progress.save()

    def results(self) -> SessionResult:
        r = SessionResult(total=len(self.questions))
        by_topic: dict[str, list[int]] = {}
        for q in self.questions:
            outcome = self.outcomes.get(q.id)
            if outcome is None:
                continue
            if outcome == CORRECT:
                r.correct += 1
            elif outcome == INCORRECT:
                r.incorrect += 1
                r.weak_question_ids.append(q.id)
            else:
                r.skipped += 1
                r.weak_question_ids.append(q.id)
            if outcome != SKIPPED:
                row = by_topic.setdefault(q.topic or "", [0, 0])
                row[1] += 1
                row[0] += outcome == CORRECT
            if self.progress.question_state(q.id) == MASTERED:
                r.mastered += 1
        r.score_percent = round(100 * r.correct / r.total) if r.total else 0
        rows = [(t, c / n) for t, (c, n) in by_topic.items() if t and n]
        r.weak_topics = sorted([row for row in rows if row[1] < 0.6], key=lambda row: row[1])
        r.strong_topics = sorted([row for row in rows if row[1] >= 0.8], key=lambda row: -row[1])
        for topic, accuracy in r.weak_topics:
            r.suggested_review.append(f"Review: {topic} ({round(accuracy * 100)}% correct)")
        if r.skipped:
            r.suggested_review.append(f"{r.skipped} skipped question{'s' if r.skipped != 1 else ''} to come back to")
        if not r.suggested_review and r.total:
            r.suggested_review.append("Nothing to review - try a longer session or the Hard questions.")
        return r
