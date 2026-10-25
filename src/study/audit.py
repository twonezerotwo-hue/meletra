"""The quality audit of a study pack: one definition of "how good is this pack", used by the engine (it is stored in the pack), by the tests and by the benchmark script.

It is an AUTOMATED RUBRIC, not a human rating. Every number is computed from the delivered items by the same checks that decide whether an item is delivered at all
(``quality.py``) plus checks that need the whole pack (coverage, redundancy, answer fairness) and the source (grounding). What the rubric cannot see - a fluent but medically
wrong statement, a transcript whose recogniser errors passed every filter - is not measured, and the report says so.

    summary / key concepts   item quality (0-10) - 2 x missing topic coverage - 2 x near-duplicates
    flashcards               item quality - coverage - duplicates - 0.5 for a deck of a single card type
    questions                item quality - coverage - duplicates - 0.5 when true/false is one-sided - 3 for every multiple-choice question without a same-kind distractor set
    study mode               share of the checks of a simulated session that pass (no leaked answers, answerable, consistent evaluation, no duplicates, mixed difficulty)
    medical accuracy         share of delivered items built only from words the vocabulary / slides / the lecture itself know, none of them doubted by the medical pass or the recogniser
    source grounding         share of items whose statement is carried by the passage they cite (>= 90 % of its content words) with a reference of 5-30 seconds
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Optional, Sequence

from . import facts as F
from . import quality as Q
from . import text as T
from .models import SOURCE_PDF, Flashcard, Question, StudyPack, SummaryItem

TARGETS = {"summary": 9.2, "key_concepts": 9.3, "flashcards": 9.3, "questions": 9.2, "study_mode": 9.3, "medical_accuracy": 9.5, "source_grounding": 9.5}
NARROW_MIN, NARROW_MAX = 5.0, 30.0


def _mean(values: Sequence[float], default: float = 0.0) -> float:
    return sum(values) / len(values) if values else default


def _duplicates(texts: Sequence[str]) -> float:
    """Share of items that repeat an earlier one (same content words)."""
    seen: list[str] = []
    dup = 0
    for text in texts:
        if any(F.near_duplicate(text, other) for other in seen):
            dup += 1
        else:
            seen.append(text)
    return dup / len(texts) if texts else 0.0


def _coverage(topics: Sequence[str], used: Sequence[str]) -> float:
    names = [t for t in topics if t]
    if len(names) <= 1:
        return 1.0
    return len({u for u in used if u} & set(names)) / len(names)


def _clip(x: float) -> float:
    return round(max(0.0, min(10.0, x)), 2)


def grounded(statement: str, source_text: str) -> bool:
    """The statement's content words are (almost) all in the cited passage."""
    words = {T.stem(w) for w in T.content_words(statement)}
    if not words:
        return False
    have = {T.stem(w) for w in T.content_words(source_text)}
    return len(words & have) / len(words) >= 0.9


class PackAudit:
    def __init__(self, pack: StudyPack, index, study=None) -> None:
        self.pack, self.index, self.study = pack, index, study
        self.chunks = {c.id: c for c in index.chunks}
        unknown = study.words.unknown if study is not None else None
        self.unknown = unknown
        self.unresolved = list(getattr(index, "unresolved", []))
        self.known_term = study.known_term if study is not None else None
        # the topics the deck CAN cover: those with at least one usable fact. A part of the recording that yields no reliable statement is a fact about the recording (reported as
        # "uncovered_topics" and as a warning to the student), not a defect of the delivered items - the rubric scores the items (see the note in run()).
        if study is not None:
            usable_topics = {f.topic for f in study.base.facts if f.topic and study.usable(f)}
            self.coverable = [t for t in pack.topics if t in usable_topics] or list(pack.topics)
        else:
            self.coverable = list(pack.topics)
        self.uncovered = [t for t in pack.topics if t not in self.coverable]

    # ---------------------------------------------------------------------------------------------------------------- per area
    def summary(self) -> dict:
        items = [i for kind in ("quick", "detailed", "high_yield") for i in self.pack.summaries.get(kind, [])]
        verdicts = [Q.check_summary_point(i.text, unknown=self.unknown, unresolved=self.unresolved) for i in items]
        quality = _mean([v.score for v in verdicts], 0.0)
        cov, cov_all = _coverage(self.coverable, [i.topic for i in items]), _coverage(self.pack.topics, [i.topic for i in items])
        dup = _duplicates([i.text for i in self.pack.summaries.get("detailed", [])])
        return {"items": len(items), "item_quality": round(quality, 2), "topic_coverage": round(cov, 2), "topic_coverage_all_topics": round(cov_all, 2), "duplicate_share": round(dup, 2),
                "score": _clip(quality - 2 * (1 - cov) - 2 * dup) if items else 0.0}

    def key_concepts(self) -> dict:
        items = self.pack.summaries.get("key_concepts", [])
        verdicts = [Q.check_concept(i.term, i.text, known_term=self.known_term, unknown=self.unknown, unresolved=self.unresolved) for i in items]
        quality = _mean([v.score for v in verdicts], 0.0)
        keys = [" ".join(T.stem(w) for w in T.words(i.term)) for i in items]
        dup = 1.0 - len(set(keys)) / len(keys) if keys else 0.0
        spans = [(s.end - s.start) for i in items for s in i.sources[:1] if s.start is not None and s.end is not None and s.kind != SOURCE_PDF]
        narrow = _mean([1.0 if NARROW_MIN - 0.01 <= x <= NARROW_MAX + 0.01 else 0.0 for x in spans], 1.0)
        return {"items": len(items), "item_quality": round(quality, 2), "duplicate_share": round(dup, 2), "narrow_source_share": round(narrow, 2),
                "score": _clip(quality - 2 * dup - 2 * (1 - narrow)) if items else 0.0}

    def flashcards(self) -> dict:
        cards = self.pack.flashcards
        verdicts = [Q.check_card(c, unknown=self.unknown, unresolved=self.unresolved, known_term=self.known_term) for c in cards]
        quality = _mean([v.score for v in verdicts], 0.0)
        cov, cov_all = _coverage(self.coverable, [c.topic for c in cards]), _coverage(self.pack.topics, [c.topic for c in cards])
        dup = _duplicates([c.front for c in cards])
        types = len({c.type for c in cards})
        one_type = 0.5 if len(cards) >= 10 and types < 3 else 0.0
        return {"items": len(cards), "item_quality": round(quality, 2), "topic_coverage": round(cov, 2), "topic_coverage_all_topics": round(cov_all, 2), "duplicate_share": round(dup, 2), "card_types": types,
                "score": _clip(quality - 2 * (1 - cov) - 2 * dup - one_type) if cards else 0.0}

    def questions(self) -> dict:
        qs = self.pack.questions
        same = self.study._same_category(None) if self.study is not None else None
        verdicts = [Q.check_question(q, unknown=self.unknown, unresolved=self.unresolved, same_category=same) for q in qs]
        quality = _mean([v.score for v in verdicts], 0.0)
        cov, cov_all = _coverage(self.coverable, [q.topic for q in qs]), _coverage(self.pack.topics, [q.topic for q in qs])
        dup = _duplicates([q.question for q in qs])
        tf = [q for q in qs if q.type == "true_false"]
        one_sided = 0.5 if len(tf) >= 6 and len({q.answer for q in tf}) < 2 else 0.0
        return {"items": len(qs), "item_quality": round(quality, 2), "topic_coverage": round(cov, 2), "topic_coverage_all_topics": round(cov_all, 2), "duplicate_share": round(dup, 2),
                "question_types": len({q.type for q in qs}), "score": _clip(quality - 2 * (1 - cov) - 2 * dup - one_sided) if qs else 0.0}

    def study_mode(self) -> dict:
        """A simulated Study Mode session on this pack: nothing leaks before the answer, every question can be answered, the student's right answer is accepted and a wrong one is not."""
        from .extractive import ExtractiveProvider
        from .providers import StudyContext
        from .session import QuizSession, build_quiz

        qs = self.pack.questions
        if not qs:
            return {"checks": {}, "score": 0.0}
        provider = ExtractiveProvider()
        ctx = StudyContext(index=self.index)

        def evaluator(q: Question, answer: str):
            chunks = [c for c in (self.chunks.get(r.chunk_id) for r in q.sources) if c]
            return provider.evaluate_answer(q, answer, chunks, ctx)

        quiz = build_quiz(self.pack, None, count=min(10, len(qs)), seed=1)
        session = QuizSession(quiz, evaluator=evaluator)
        view = session.current()
        checks: dict[str, float] = {}
        checks["no answer, explanation or source before the answer"] = 0.0 if view is None or any(hasattr(view, n) for n in ("answer", "explanation", "sources")) else 1.0
        checks["every question can be answered"] = _mean([1.0 if ((q.answer in q.options) if q.options else bool(q.answer.strip())) else 0.0 for q in qs])
        checks["no repeated question in a quiz"] = 1.0 - _duplicates([q.question for q in quiz])
        right = wrong = 0
        for q in quiz:
            right += int(evaluator(q, q.answer).correct)
            if q.options:
                bad = next((o for o in q.options if o != q.answer), "")
            else:
                other = next((x.answer for x in qs if x.id != q.id and not (set(T.content_stems(x.answer)) & set(T.content_stems(q.answer)))), "I do not know.")
                bad = other
            wrong += int(not evaluator(q, bad).correct)
        checks["the right answer is accepted"] = right / len(quiz)
        checks["a wrong answer is refused"] = wrong / len(quiz)
        checks["easy, medium and hard mixed"] = 1.0 if len({q.difficulty for q in qs}) >= 2 or len(qs) < 6 else 0.5
        return {"checks": {k: round(v, 2) for k, v in checks.items()}, "score": _clip(10.0 * _mean(list(checks.values())))}

    # ---------------------------------------------------------------------------------------------------------------- whole pack
    def _items(self) -> list[tuple[str, str, object]]:
        out: list[tuple[str, str, object]] = []
        for kind, items in self.pack.summaries.items():
            out += [("summary:" + kind, i.text, i) for i in items]
        out += [("flashcard", (c.back.split(" - ", 1)[-1] if " - " in c.back and "_____" in c.front else c.back), c) for c in self.pack.flashcards]
        for q in self.pack.questions:
            m = re.search(r'The lecture says: "(.*)"\s*$', q.explanation, re.DOTALL)
            out.append(("question", m.group(1) if m else q.answer, q))
        return out

    def _v4_flags(self, statements: Sequence[str]):
        from .factbase import medical_validation

        return medical_validation(statements)

    def medical_accuracy(self) -> dict:
        items = self._items()
        statements = [st for _a, st, _i in items]
        bad = set()
        for k, statement in enumerate(statements):
            low = statement.casefold()
            doubtful = any(u.casefold() in low for u in self.unresolved if u)
            odd = [w for w in T.words(statement) if len(w) >= 5 and self.unknown is not None and self.unknown(w)]
            if doubtful or odd:
                bad.add(k)
        flagged = self._v4_flags(statements)
        if flagged is not None:
            bad |= flagged
        share = 1.0 - len(bad) / len(items) if items else 0.0
        note = (self.index.meta or {}).get("source_note", "")
        return {"items_checked": len(items), "with_a_doubtful_word": len(bad) if flagged is None else len(bad - flagged), "questioned_by_v4": None if flagged is None else len(flagged),
                "score": _clip(10 * share), "transcript": note or "unknown",
                "limit": ("unknown / doubted words and the v4 medical engine's check (the same check removes such statements before delivery, so a high score here is expected); " if flagged is not None else
                          "unknown / doubted words only (no knowledge base installed for the second opinion); ") + "a fluent but wrong statement is not detected"}

    def source_grounding(self) -> dict:
        items = self._items()
        ok = 0
        for _area, statement, item in items:
            refs = getattr(item, "sources", [])
            if not refs:
                continue
            ref = refs[0]
            chunk = self.chunks.get(ref.chunk_id)
            if chunk is None or not grounded(statement, chunk.text):
                continue
            if chunk.source != SOURCE_PDF and ref.start is not None and ref.end is not None and not (NARROW_MIN - 0.01 <= ref.end - ref.start <= NARROW_MAX + 0.01):
                continue
            ok += 1
        return {"items_checked": len(items), "grounded": ok, "score": _clip(10 * ok / len(items)) if items else 0.0}

    def run(self) -> dict:
        report = {"summary": self.summary(), "key_concepts": self.key_concepts(), "flashcards": self.flashcards(), "questions": self.questions(), "study_mode": self.study_mode(),
                  "medical_accuracy": self.medical_accuracy(), "source_grounding": self.source_grounding()}
        scores = {k: v["score"] for k, v in report.items()}
        report["scores"] = scores
        report["targets"] = TARGETS
        report["meets_targets"] = {k: scores[k] >= TARGETS[k] for k in TARGETS}
        report["uncovered_topics"] = list(self.uncovered)
        report["note"] = ("automated rubric (see audit.py), not a human rating; the numbers describe the delivered items, not the quality of the recording: topic coverage is "
                          "measured over the topics that yielded at least one reliable statement (topic_coverage_all_topics and uncovered_topics show the rest)")
        return report


def audit_pack(pack: StudyPack, index, study=None) -> dict:
    return PackAudit(pack, index, study).run()
