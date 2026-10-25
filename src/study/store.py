"""Saving the study pack and the student's progress next to the lecture, on this computer only.

Layout (inside the lecture's output folder, so the study material travels with the transcript)::

    <output folder>/study/study_pack.json      summaries, flashcards, questions, sources, provider used
    <output folder>/study/study_index.json     chunks + topics + slide links (the local retrieval index)
    <output folder>/study/study_progress.json  ratings, answers, session results

Every write is atomic (temp file, then rename) and a damaged file is set aside as ``.bad`` instead of crashing the study view.
"""

from __future__ import annotations

import json
import logging
import os
import copy
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .index import DocumentIndex
from .models import RATINGS, StudyPack
from .scheduler import Schedule, days_overdue, is_due, next_schedule, parse_day, replay

log = logging.getLogger(__name__)

STUDY_DIR = "study"
PACK_FILE = "study_pack.json"
INDEX_FILE = "study_index.json"
PROGRESS_FILE = "study_progress.json"
PROGRESS_VERSION = 2                           # 2: spaced-repetition schedule, "difficult" mark and the review log per card (version 1 files are read as they are and upgraded when a card is rated)
HISTORY = 6                                    # answers remembered per item (the state of an item is judged on these)
REVIEW_LOG = 200                               # reviews remembered per flashcard (the review history the student can look back on)
CARRY_OVER_MATCH = 90                          # how alike (0-100) the wording of a new item must be to a studied item of the earlier material to inherit its progress

NEW, LEARNING, WEAK, MASTERED = "new", "learning", "weak", "mastered"
CORRECT, INCORRECT, SKIPPED = "correct", "incorrect", "skipped"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def _read_json(path: Path) -> Optional[Any]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        log.warning("damaged study file %s was set aside", path)
        try:
            os.replace(path, path.with_suffix(path.suffix + ".bad"))
        except OSError:
            pass
        return None


@dataclass
class Progress:
    """What the student did. Item states: new / learning / weak / mastered."""

    cards: dict[str, dict[str, Any]] = field(default_factory=dict)         # id -> {"ratings": [...], "last": iso}
    questions: dict[str, dict[str, Any]] = field(default_factory=dict)     # id -> {"results": [...], "topic": str, "last": iso}
    sessions: list[dict[str, Any]] = field(default_factory=list)
    active_session: Optional[dict[str, Any]] = None                        # a Study Mode session that was started and not finished: {"ids", "outcomes", "started"} (resumed after a restart)
    path: Optional[Path] = None

    # ---- flashcards
    def rate_card(self, card_id: str, rating: str, topic: str = "", today: Optional[date] = None) -> None:
        if rating not in RATINGS:
            raise ValueError(f"unknown rating {rating!r}")
        day = today or date.today()
        entry = self.cards.setdefault(card_id, {"ratings": [], "topic": topic})
        entry["sr"] = next_schedule(self.schedule(card_id, day), rating, day).to_dict()          # before the rating is added: an old entry is upgraded from what it had
        reviews = int(entry.get("reviews", len(entry.get("ratings", []))))                       # an old entry counts the ratings it still has
        entry["ratings"] = (entry.get("ratings", []) + [rating])[-HISTORY:]
        entry["topic"] = topic or entry.get("topic", "")
        entry["last"] = _now()
        entry["log"] = (list(entry.get("log") or []) + [{"on": day.isoformat(), "rating": rating}])[-REVIEW_LOG:]
        entry["reviews"] = reviews + 1

    def card_state(self, card_id: str) -> str:
        ratings = self.cards.get(card_id, {}).get("ratings", [])
        if not ratings:
            return NEW
        if ratings[-1] == "dont_know":
            return WEAK
        if ratings[-2:] == ["know", "know"]:
            return MASTERED
        return LEARNING

    def schedule(self, card_id: str, today: Optional[date] = None) -> Schedule:
        """When the card comes back. A card from a progress file of an earlier version has only its ratings: its schedule is worked out from them (and stored when it is rated again)."""
        entry = self.cards.get(card_id) or {}
        if isinstance(entry.get("sr"), dict):
            return Schedule.from_dict(entry["sr"])
        ratings = [r for r in entry.get("ratings", []) if r in RATINGS]
        if not ratings:
            return Schedule()
        return replay(ratings, parse_day(str(entry.get("last", ""))) or today or date.today())

    def is_due(self, card_id: str, today: Optional[date] = None) -> bool:
        return is_due(self.schedule(card_id, today), today or date.today())

    def due_cards(self, pack: StudyPack, today: Optional[date] = None) -> list[str]:
        """Ids of the cards of this pack whose day has come, the most overdue first."""
        day = today or date.today()
        due = [(days_overdue(self.schedule(c.id, day), day), c.id) for c in pack.flashcards if self.is_due(c.id, day)]
        return [cid for _late, cid in sorted(due, key=lambda row: -row[0])]

    def next_due(self, pack: StudyPack, today: Optional[date] = None) -> Optional[date]:
        """The earliest day a reviewed card of this pack comes due (None when nothing has been reviewed)."""
        days = [parse_day(self.schedule(c.id, today).due) for c in pack.flashcards if self.cards.get(c.id, {}).get("ratings")]
        days = [d for d in days if d is not None]
        return min(days) if days else None

    def set_difficult(self, card_id: str, flag: bool, topic: str = "") -> None:
        """The student's own mark, "this one is hard for me". It does not change the schedule."""
        if not flag and card_id not in self.cards:
            return
        entry = self.cards.setdefault(card_id, {"ratings": [], "topic": topic})
        entry["difficult"] = bool(flag)
        entry["topic"] = topic or entry.get("topic", "")

    def is_difficult(self, card_id: str) -> bool:
        return bool(self.cards.get(card_id, {}).get("difficult"))

    def review_history(self, card_id: str) -> list[dict[str, str]]:
        """Every remembered review of the card, oldest first: {"on": day, "rating": ...}."""
        return [dict(row) for row in self.cards.get(card_id, {}).get("log", []) if isinstance(row, dict)]

    def study_plan(self, pack: StudyPack, today: Optional[date] = None) -> dict[str, int]:
        """What is waiting: cards due today, new cards, cards marked difficult, and the reviews done today and in all (of the cards of this pack)."""
        day = today or date.today()
        entries = [self.cards.get(c.id) or {} for c in pack.flashcards]
        return {"due": len(self.due_cards(pack, day)), "new": sum(1 for e in entries if not e.get("ratings")), "difficult": sum(1 for e in entries if e.get("difficult")),
                "reviews_today": sum(1 for e in entries for row in e.get("log", []) if isinstance(row, dict) and row.get("on") == day.isoformat()),
                "reviews": sum(int(e.get("reviews", len(e.get("ratings", [])))) for e in entries)}

    # ---- questions
    def record_answer(self, question_id: str, outcome: str, topic: str = "") -> None:
        if outcome not in (CORRECT, INCORRECT, SKIPPED):
            raise ValueError(f"unknown outcome {outcome!r}")
        entry = self.questions.setdefault(question_id, {"results": [], "topic": topic})
        entry["results"] = (entry["results"] + [outcome])[-HISTORY:]
        entry["topic"] = topic or entry.get("topic", "")
        entry["last"] = _now()

    def question_state(self, question_id: str) -> str:
        results = self.questions.get(question_id, {}).get("results", [])
        if not results:
            return NEW
        if results[-1] == INCORRECT:
            return WEAK
        if results[-2:] == [CORRECT, CORRECT]:
            return MASTERED
        return LEARNING

    # ---- topics
    def topic_accuracy(self) -> dict[str, tuple[int, int]]:
        """topic -> (correct, answered) over the remembered answers (skips are not counted as answers)."""
        out: dict[str, list[int]] = {}
        for entry in self.questions.values():
            topic = entry.get("topic") or ""
            for result in entry.get("results", []):
                if result == SKIPPED:
                    continue
                row = out.setdefault(topic, [0, 0])
                row[1] += 1
                row[0] += result == CORRECT
        return {t: (c, n) for t, (c, n) in out.items()}

    def weak_topics(self, minimum: int = 2) -> list[tuple[str, float]]:
        rows = [(t, c / n) for t, (c, n) in self.topic_accuracy().items() if t and n >= minimum and c / n < 0.6]
        return sorted(rows, key=lambda r: r[1])

    def strong_topics(self, minimum: int = 2) -> list[tuple[str, float]]:
        rows = [(t, c / n) for t, (c, n) in self.topic_accuracy().items() if t and n >= minimum and c / n >= 0.8]
        return sorted(rows, key=lambda r: -r[1])

    def counts(self, pack: StudyPack) -> dict[str, dict[str, int]]:
        cards = {NEW: 0, LEARNING: 0, WEAK: 0, MASTERED: 0}
        for c in pack.flashcards:
            cards[self.card_state(c.id)] += 1
        questions = {NEW: 0, LEARNING: 0, WEAK: 0, MASTERED: 0}
        for q in pack.questions:
            questions[self.question_state(q.id)] += 1
        return {"cards": cards, "questions": questions}

    # ---- persistence
    def add_session(self, result: dict[str, Any]) -> None:
        self.sessions = (self.sessions + [{**result, "at": _now()}])[-50:]

    # ---- new material for the same lecture
    def carry_over(self, old: Optional[StudyPack], new: StudyPack) -> dict[str, int]:
        """Regenerating the study material must not throw away what the student has done. A card or question of the new pack that has no history of its own inherits the
        history (schedule and difficult mark too) of the studied item of the earlier pack whose wording it almost repeats. Nothing is ever removed: the entries of items that
        are no longer in the pack stay in the file. Returns how many cards and questions inherited."""
        if old is None:
            return {"cards": 0, "questions": 0}
        return {"cards": self._carry(self.cards, [(c.id, c.front) for c in old.flashcards], [(c.id, c.front) for c in new.flashcards]),
                "questions": self._carry(self.questions, [(q.id, q.question) for q in old.questions], [(q.id, q.question) for q in new.questions])}

    @staticmethod
    def _carry(store: dict[str, dict[str, Any]], previous: list[tuple[str, str]], current: list[tuple[str, str]]) -> int:
        from rapidfuzz import fuzz

        def plain(text: str) -> str:
            return " ".join(re.sub(r"[^\w\s]", " ", text.casefold()).split())

        def studied(entry: Optional[dict[str, Any]]) -> bool:
            return bool(entry and (entry.get("ratings") or entry.get("results") or entry.get("difficult")))

        candidates = [(item_id, plain(text)) for item_id, text in previous if studied(store.get(item_id))]
        taken: set[str] = set()
        count = 0
        for item_id, text in current:
            if studied(store.get(item_id)):
                continue                                                      # it has a history of its own (the same id is the same item)
            wanted = plain(text)
            best, best_score = "", 0.0
            for old_id, old_text in candidates:
                if old_id in taken or old_id == item_id:
                    continue
                score = fuzz.ratio(wanted, old_text)
                if score > best_score:
                    best, best_score = old_id, score
            if best and best_score >= CARRY_OVER_MATCH:
                store[item_id] = {**copy.deepcopy(store[best]), "carried_from": best}
                taken.add(best)
                count += 1
        return count

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"version": PROGRESS_VERSION, "cards": self.cards, "questions": self.questions, "sessions": self.sessions}
        if self.active_session:
            out["active_session"] = self.active_session
        return out

    def save(self) -> None:
        if self.path:
            _write_json(self.path, self.to_dict())

    @classmethod
    def load(cls, path: Path) -> "Progress":
        data = _read_json(path)
        if not isinstance(data, dict):
            return cls(path=path)
        active = data.get("active_session")
        return cls(cards=dict(data.get("cards", {})), questions=dict(data.get("questions", {})), sessions=list(data.get("sessions", [])),
                   active_session=active if isinstance(active, dict) else None, path=path)


class StudyStore:
    """The study files of one lecture."""

    def __init__(self, lecture_folder: Path, language: str = "") -> None:
        """``language``: the language the material is written in. The first language studied for a lecture keeps the original layout (``study/study_pack.json`` ...); another
        language gets its own folder (``study/<code>/``), so material in one language never overwrites material in another."""
        self.base = Path(lecture_folder) / STUDY_DIR
        self.language = language
        self.folder = self._resolve(self.base, language)

    @staticmethod
    def stored_language(folder: Path) -> Optional[str]:
        """The language of the material stored directly in ``folder`` (pack first, then index); None when nothing is stored there."""
        pack = _read_json(Path(folder) / PACK_FILE)
        if isinstance(pack, dict) and pack.get("language"):
            return str(pack["language"])
        index = _read_json(Path(folder) / INDEX_FILE)
        if isinstance(index, dict):
            language = (index.get("meta") or {}).get("language")
            return str(language) if language else None
        return None

    @classmethod
    def _resolve(cls, base: Path, language: str) -> Path:
        if not language:
            return base
        sub = base / language
        if (sub / PACK_FILE).is_file() or (sub / INDEX_FILE).is_file():
            return sub
        owner = cls.stored_language(base)
        return base if owner is None or owner == language else sub

    @classmethod
    def languages_on_disk(cls, lecture_folder: Path) -> list[str]:
        """Every language this lecture has study material in."""
        base = Path(lecture_folder) / STUDY_DIR
        found = []
        owner = cls.stored_language(base)
        if owner:
            found.append(owner)
        if base.is_dir():
            for child in sorted(base.iterdir()):
                if child.is_dir() and ((child / PACK_FILE).is_file() or (child / INDEX_FILE).is_file()) and child.name not in found:
                    found.append(child.name)
        return found

    @property
    def pack_path(self) -> Path:
        return self.folder / PACK_FILE

    @property
    def index_path(self) -> Path:
        return self.folder / INDEX_FILE

    @property
    def progress_path(self) -> Path:
        return self.folder / PROGRESS_FILE

    def save_pack(self, pack: StudyPack) -> Path:
        _write_json(self.pack_path, pack.to_dict())
        return self.pack_path

    def load_pack(self) -> Optional[StudyPack]:
        data = _read_json(self.pack_path)
        if not isinstance(data, dict):
            return None
        try:
            return StudyPack.from_dict(data)
        except (KeyError, TypeError, ValueError):
            log.warning("unreadable study pack %s", self.pack_path)
            return None

    def save_index(self, index: DocumentIndex) -> Path:
        _write_json(self.index_path, index.to_dict())
        return self.index_path

    def load_index(self) -> Optional[DocumentIndex]:
        data = _read_json(self.index_path)
        if not isinstance(data, dict):
            return None
        try:
            return DocumentIndex.from_dict(data)
        except (KeyError, TypeError, ValueError):
            return None

    def progress(self) -> Progress:
        return Progress.load(self.progress_path)

    def has_pack(self) -> bool:
        return self.pack_path.is_file()
