"""Spaced repetition for the flashcards: when a card should come back.

This is the SM-2 algorithm (P. Wozniak, SuperMemo 2) with the three buttons of the cards mapped onto its 0-5 quality scale:

    Know       -> 5   (recalled without trouble)
    Unsure     -> 3   (recalled with effort: still a pass, but the ease drops and the gap grows less)
    Don't know -> 1   (a lapse: the card starts again and is due today)

A pass at repetition 0 comes back after 1 day, at repetition 1 after 6 days, then after ``previous interval x ease`` days. The ease factor
(start 2.5, never below 1.3) moves with the answer as in SM-2: ``ease += 0.1 - (5 - q) * (0.08 + (5 - q) * 0.02)``. A lapse resets the repetition count,
costs 0.2 ease and is counted. Everything is a pure function of (state, rating, today), so it is tested without a clock.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Optional

QUALITY = {"know": 5, "unsure": 3, "dont_know": 1}
START_EASE = 2.5
MIN_EASE = 1.3
FIRST_INTERVAL, SECOND_INTERVAL = 1, 6
MAX_INTERVAL = 365                                   # a lecture is revised within a year; longer gaps only hide forgetting


@dataclass(frozen=True)
class Schedule:
    ease: float = START_EASE
    interval: int = 0                                # days until the next review (0 = due again today)
    reps: int = 0                                    # passes in a row
    lapses: int = 0                                  # times it was forgotten
    due: str = ""                                    # ISO date of the next review ("" = never reviewed)

    def to_dict(self) -> dict[str, Any]:
        return {"ease": round(self.ease, 3), "interval": self.interval, "reps": self.reps, "lapses": self.lapses, "due": self.due}

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> "Schedule":
        data = data if isinstance(data, dict) else {}
        try:
            return cls(ease=max(MIN_EASE, float(data.get("ease", START_EASE))), interval=max(0, int(data.get("interval", 0))), reps=max(0, int(data.get("reps", 0))),
                       lapses=max(0, int(data.get("lapses", 0))), due=str(data.get("due", "") or ""))
        except (TypeError, ValueError):
            return cls()


def next_schedule(current: Schedule, rating: str, today: date) -> Schedule:
    """The schedule after answering ``rating`` on ``today``."""
    if rating not in QUALITY:
        raise ValueError(f"unknown rating {rating!r}")
    q = QUALITY[rating]
    if q < 3:
        return Schedule(ease=max(MIN_EASE, current.ease - 0.2), interval=0, reps=0, lapses=current.lapses + 1, due=today.isoformat())
    ease = max(MIN_EASE, current.ease + 0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
    if current.reps == 0:
        interval = FIRST_INTERVAL
    elif current.reps == 1:
        interval = SECOND_INTERVAL
    else:
        interval = round(max(current.interval, 1) * ease)
    interval = max(1, min(MAX_INTERVAL, interval))
    return Schedule(ease=ease, interval=interval, reps=current.reps + 1, lapses=current.lapses, due=(today + timedelta(days=interval)).isoformat())


def when(days: int) -> str:
    """An interval in words, for the buttons: today, tomorrow, in 6 days, in 3 weeks, in 2 months."""
    if days <= 0:
        return "today"
    if days == 1:
        return "tomorrow"
    if days < 14:
        return f"in {days} days"
    if days < 60:
        return f"in {round(days / 7)} weeks"
    return f"in {round(days / 30)} months"


def parse_day(text: str) -> Optional[date]:
    try:
        return date.fromisoformat(text[:10]) if text else None
    except ValueError:
        return None


def is_due(schedule: Schedule, today: date) -> bool:
    """A card that was reviewed before and whose day has come (or passed)."""
    day = parse_day(schedule.due)
    return day is not None and day <= today


def days_overdue(schedule: Schedule, today: date) -> int:
    day = parse_day(schedule.due)
    return 0 if day is None else max((today - day).days, 0)


def replay(ratings: list[str], last_day: date) -> Schedule:
    """The schedule of a card that only has its old list of ratings (progress files from before spaced repetition): the ratings are replayed in order and the last one
    counts as given on ``last_day``. Only the last review's day reaches the result (``due`` = that day + the interval), so the dates of the earlier ones do not matter."""
    schedule = Schedule()
    for position, rating in enumerate(ratings):
        schedule = next_schedule(schedule, rating, last_day if position == len(ratings) - 1 else last_day - timedelta(days=len(ratings) - position))
    return schedule
