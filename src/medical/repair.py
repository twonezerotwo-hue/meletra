"""Evidence-based repetition and timestamp repair for the MEDICAL transcript (the raw transcript is never changed).

Duplicates are removed only when the evidence says "decoder artefact", not "lecturer said it twice": the two copies overlap in time or follow
each other within a second, they are (nearly) the same words, and there are enough words for that to mean something. A sentence repeated after
a real pause is the lecturer's emphasis and stays.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

from rapidfuzz import fuzz

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WORDS = re.compile(r"\S+")
_STOPWORD_REPEAT = {"the", "a", "an", "of", "and", "to", "in", "is", "are", "it", "that", "this", "we", "so", "but", "or", "for", "with", "as", "at", "on"}


@dataclass
class Item:
    """A segment as the repair sees it. ``index`` is the position in the RAW transcript (kept for the report)."""

    index: int
    start: float
    end: float
    text: str
    confidence: Optional[float] = None
    language: Optional[str] = None
    source: int = -1                      # raw segment this text came from (for the report / word probabilities)
    notes: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        return re.sub(r"\s+", " ", _PUNCT.sub("", self.text.lower())).strip()


def word_count(text: str) -> int:
    return len(_WORDS.findall(text))


# ====================================================================== repeated phrases inside one segment
def collapse_repeats(text: str) -> tuple[str, int]:
    """"together are together are embedded" -> "together are embedded"; "formation, formation" -> "formation". Returns (text, collapsed).

    Repeated units of two or more words (and any unit repeated three times or more) are collapsed; a single repeated word only when it is
    a function word or is separated by a comma - emphasis such as "very very important" is the speaker's.
    """
    words = text.split()
    if len(words) < 2:
        return text, 0
    collapsed = 0
    changed = True
    while changed:
        changed = False
        norm = [re.sub(r"[^\w'’-]", "", w.lower()) for w in words]
        for n in range(8, 0, -1):
            i = 0
            while i + 2 * n <= len(words):
                unit, nxt = norm[i:i + n], norm[i + n:i + 2 * n]
                if all(unit) and unit == nxt:
                    repeats = 2
                    while i + (repeats + 1) * n <= len(words) and norm[i + repeats * n:i + (repeats + 1) * n] == unit:
                        repeats += 1
                    comma = any(w.endswith(",") for w in words[i + n - 1:i + n])
                    single_ok = n > 1 or repeats >= 3 or unit[0] in _STOPWORD_REPEAT or comma
                    if single_ok:
                        tail_punct = words[i + repeats * n - 1]
                        end_mark = tail_punct[-1] if tail_punct and tail_punct[-1] in ",.;:!?" else ""
                        keep = words[i:i + n]
                        keep[-1] = keep[-1].rstrip(",.;:!?") + end_mark                      # the punctuation of the LAST copy continues the sentence
                        words[i:i + repeats * n] = keep
                        norm[i:i + repeats * n] = unit
                        collapsed += repeats - 1
                        changed = True
                        continue
                i += 1
    return " ".join(words), collapsed


# ====================================================================== duplicate segments
@dataclass
class DuplicateRemoval:
    index: int
    start: float
    text: str
    kept_index: int
    reason: str

    def to_dict(self) -> dict:
        h, rest = divmod(int(self.start), 3600)
        m, s = divmod(rest, 60)
        return {"timestamp": f"{h:02d}:{m:02d}:{s:02d}", "segment": self.index, "text": self.text, "kept_segment": self.kept_index, "reason": self.reason}


def remove_duplicates(items: Sequence[Item], look_back: int = 3, max_gap: float = 1.0, min_words: int = 4,
                      similarity: float = 92.0) -> tuple[list[Item], list[DuplicateRemoval]]:
    """Drop decoder-artefact duplicates; trim a chunk-edge overlap; keep a lecturer's deliberate repetition."""
    kept: list[Item] = []
    removed: list[DuplicateRemoval] = []
    for item in items:
        drop_reason = None
        partner = None
        for prev in kept[-look_back:]:
            gap = item.start - prev.end
            ka, kb = prev.key, item.key
            if not ka or not kb:
                continue
            words_b = word_count(item.text)
            close = gap <= max_gap
            if ka == kb and word_count(item.text) >= 2 and close:
                drop_reason, partner = "exact duplicate of the previous text" + (" (overlapping in time)" if gap < 0 else f" ({gap:.1f}s later)"), prev
            elif words_b >= min_words and close and fuzz.ratio(ka, kb) >= similarity:
                drop_reason, partner = f"near duplicate ({fuzz.ratio(ka, kb):.0f}% same words)", prev
            elif gap < 0 and words_b >= min_words and (kb in ka or ka in kb) and min(len(ka), len(kb)) / max(len(ka), len(kb)) >= 0.6:
                drop_reason, partner = "overlapping copy of the same sentence", prev
            if drop_reason:
                break
        if drop_reason and partner is not None:
            # keep the more confident copy, but at the earlier position
            if (item.confidence or 0.0) > (partner.confidence or 0.0) + 0.1 and word_count(item.text) >= word_count(partner.text):
                partner.text, partner.confidence = item.text, item.confidence
                partner.end = max(partner.end, item.end)
                partner.notes.append(f"duplicate segment {item.index} had the better reading")
            elif item.start < partner.end and item.end > partner.end:                # an overlapping copy that runs on: cover its time
                partner.end = item.end
            removed.append(DuplicateRemoval(item.index, item.start, item.text, partner.index, drop_reason))
            continue
        trimmed = _trim_overlap(kept[-1], item) if kept else None
        if trimmed is not None:
            item = trimmed
            if not item.text.strip():
                removed.append(DuplicateRemoval(item.index, item.start, "", kept[-1].index, "chunk-edge overlap, nothing new"))
                continue
        kept.append(replace(item, notes=list(item.notes)))
    return kept, removed


def _trim_overlap(prev: Item, item: Item, min_words: int = 3) -> Optional[Item]:
    """The first words of ``item`` repeat the last words of ``prev`` while the two overlap in time: cut the repeated head."""
    if item.start - prev.end >= -0.2:
        return None
    a, b = prev.text.split(), item.text.split()
    for n in range(min(len(a), len(b), 12), min_words - 1, -1):
        if [re.sub(r"\W", "", w.lower()) for w in a[-n:]] == [re.sub(r"\W", "", w.lower()) for w in b[:n]]:
            rest = " ".join(b[n:])
            return replace(item, text=rest, notes=item.notes + [f"{n} repeated word(s) at the chunk edge removed"])
    return None


# ====================================================================== timestamps
@dataclass
class TimestampRepair:
    index: int
    kind: str
    before: tuple[float, float]
    after: tuple[float, float]

    def to_dict(self) -> dict:
        return {"segment": self.index, "kind": self.kind, "before": [round(self.before[0], 2), round(self.before[1], 2)],
                "after": [round(self.after[0], 2), round(self.after[1], 2)]}


def repair_timestamps(items: Sequence[Item], duration: Optional[float] = None, max_trim: float = 1.5,
                      words_per_second: float = 2.8) -> tuple[list[Item], list[TimestampRepair]]:
    """start >= previous start, preferably >= previous end; no negative or reversed duration; nothing outside the recording."""
    out: list[Item] = []
    repairs: list[TimestampRepair] = []
    prev_start = prev_end = 0.0
    for item in items:
        start, end = item.start, item.end
        before = (start, end)
        kind = []
        if start < 0:
            start = 0.0
            kind.append("negative start")
        if end < start:
            kind.append("reversed or negative duration")
            end = start + max(0.4, word_count(item.text) / words_per_second)
        if start < prev_start:
            kind.append("start before the previous start")
            shift = prev_start - start
            start, end = prev_start, max(end + shift, prev_start + 0.4)
        overlap = prev_end - start
        if overlap > 0.05 and out:
            if overlap <= max_trim:
                last = out[-1]
                if start > last.start + 0.25:
                    kind.append("overlap with the previous segment")
                    repairs.append(TimestampRepair(last.index, "end trimmed to the next start", (last.start, last.end), (last.start, start)))
                    last.end = start
                else:
                    kind.append("overlap with the previous segment")
                    start = prev_end
                    end = max(end, start + 0.4)
            else:
                kind.append("long overlap with the previous segment")
                start = min(prev_end, end - 0.4)
        if duration and end > duration + 0.5:
            kind.append("beyond the end of the recording")
            end = max(start + 0.05, min(end, duration))
        if kind:
            repairs.append(TimestampRepair(item.index, ", ".join(kind), before, (start, end)))
        out.append(replace(item, start=start, end=end, notes=list(item.notes)))
        prev_end = max(prev_end, end)
        prev_start = start
    return out, repairs


def count_overlaps(starts_ends: Sequence[tuple[float, float]], tolerance: float = 0.05) -> int:
    """Pairs of consecutive segments that overlap (the measure used for the before / after numbers)."""
    n = 0
    prev_end = None
    for start, end in starts_ends:
        if prev_end is not None and start < prev_end - tolerance:
            n += 1
        prev_end = end if prev_end is None else max(prev_end, end)
    return n


def count_reversed(starts_ends: Sequence[tuple[float, float]]) -> int:
    return sum(1 for s, e in starts_ends if e < s)
