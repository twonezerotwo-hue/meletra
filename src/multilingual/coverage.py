"""Speech coverage: does the transcript cover what is audibly there?

The voice-activity filter that cuts silence before decoding can also cut singing or speech over music (it kept 24 s of a 12-minute song compilation). The audit that compares the transcript
with that same filter's output therefore says "97 % covered" about a recording of which 4 % was ever looked at. This module measures against the AUDIO instead: the loudness of every
second, compared with the loudness of the seconds that DID become words. Loud stretches without words are decoded again (bounded, without the filter, in the language of the surrounding
run); what comes back is kept only if it passes the same checks as every other unfiltered stretch (no stock phrase, no loop, not a music tag). What cannot be recovered is REPORTED as a
stretch the program could not read - it is never filled with invented text and never dropped silently.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from .plan import Activity

log = logging.getLogger(__name__)

MIN_GAP_SECONDS = 5.0           # a silent-to-the-transcript stretch shorter than this is a pause or a breath
LEVEL_RATIO = 0.4               # audible = at least this share of the loudness of the seconds that became words
LEVEL_FLOOR = 0.004             # ... and never below this absolute loudness (room noise)
QUIET_FLOOR = 0.0012            # speech that is much quieter than the rest (a speaker far from the microphone) is still looked for above this loudness (about -58 dBFS) ...
QUIET_MIN_SECONDS = 6.0         # ... but only when it lasts this long (steady room noise would otherwise be decoded again and again)
BRIDGE_SECONDS = 2.0            # audible stretches closer than this are one stretch
EDGE_PAD = 0.4                  # seconds around a transcribed segment that count as covered (word timestamps are not exact)
MAX_RECOVERY_SHARE = 0.25       # at most this share of the recording is decoded again in one run
MAX_RECOVERY_SECONDS_GPU, MAX_RECOVERY_SECONDS_CPU = 1800.0, 600.0
MAX_PIECE_SECONDS = 90.0        # one gap is decoded in pieces of at most this length (long unfiltered decodes drift into loops)


@dataclass
class Gap:
    start: float
    end: float
    level: float = 0.0
    status: str = "found"         # found | recovered | unreadable | skipped
    kind: str = "audible"         # audible | quiet (much quieter than the rest: decoded again with the gain raised)
    reason: str = ""
    words: int = 0

    @property
    def seconds(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        return {"start": round(self.start, 1), "end": round(self.end, 1), "seconds": round(self.seconds, 1), "kind": self.kind, "status": self.status, "reason": self.reason,
                "words": self.words}


def speech_level(activity: Activity, segments: Sequence[Any]) -> float:
    """The typical loudness of the seconds that became words (median over those seconds); 0 when nothing was transcribed."""
    covered: list[float] = []
    frame = activity.frame or 1.0
    for seg in segments:
        a, b = max(int(seg.start / frame), 0), min(int(seg.end / frame) + 1, len(activity.rms))
        covered.extend(activity.rms[a:b])
    return statistics.median(covered) if covered else 0.0


def audible_threshold(activity: Activity, segments: Sequence[Any]) -> float:
    level = speech_level(activity, segments)
    return max(LEVEL_FLOOR, level * LEVEL_RATIO) if level > 0 else max(LEVEL_FLOOR, 0.02)


def _covered_mask(activity: Activity, segments: Sequence[Any], pad: float) -> list[bool]:
    frame = activity.frame or 1.0
    mask = [False] * len(activity.rms)
    for seg in segments:
        a, b = max(int((seg.start - pad) / frame), 0), min(int((seg.end + pad) / frame) + 1, len(mask))
        for i in range(a, b):
            mask[i] = True
    return mask


def find_gaps(activity: Activity, segments: Sequence[Any], *, min_seconds: float = MIN_GAP_SECONDS, bridge: float = BRIDGE_SECONDS, pad: float = EDGE_PAD,
              quiet: bool = True) -> list[Gap]:
    """Stretches (by loudness) that no transcribed segment covers, in time order: audible ones (as loud as the words that exist) and, with ``quiet``, much quieter ones that are
    still above room-noise level for a while."""
    if not activity.rms:
        return []
    threshold = audible_threshold(activity, segments)
    mask = _covered_mask(activity, segments, pad)
    frame = activity.frame or 1.0
    gaps: list[list[Any]] = []
    for i, level in enumerate(activity.rms):
        if mask[i] or level < threshold:
            continue
        t0, t1 = i * frame, (i + 1) * frame
        if gaps and t0 - gaps[-1][1] <= bridge:
            gaps[-1][1] = t1
            gaps[-1][2].append(level)
        else:
            gaps.append([t0, t1, [level]])
    found = [Gap(a, b, sum(levels) / len(levels)) for a, b, levels in gaps if b - a >= min_seconds]
    if quiet:
        quiet_gaps: list[list[Any]] = []
        for i, level in enumerate(activity.rms):
            if mask[i] or level >= threshold or level < QUIET_FLOOR:
                continue
            t0, t1 = i * frame, (i + 1) * frame
            if quiet_gaps and t0 - quiet_gaps[-1][1] <= bridge:
                quiet_gaps[-1][1] = t1
                quiet_gaps[-1][2].append(level)
            else:
                quiet_gaps.append([t0, t1, [level]])
        for a, b, levels in quiet_gaps:
            if b - a >= QUIET_MIN_SECONDS and not any(g.start < b and a < g.end for g in found):
                found.append(Gap(a, b, sum(levels) / len(levels), kind="quiet"))
        found.sort(key=lambda g: g.start)
    return found


def coverage_numbers(activity: Activity, segments: Sequence[Any]) -> dict[str, float]:
    """Seconds that are audible, and the share of them that carry words."""
    if not activity.rms:
        return {"audible_seconds": 0.0, "covered_seconds": 0.0, "share": 1.0}
    threshold = audible_threshold(activity, segments)
    mask = _covered_mask(activity, segments, EDGE_PAD)
    frame = activity.frame or 1.0
    audible = sum(1 for level in activity.rms if level >= threshold)
    covered = sum(1 for level, m in zip(activity.rms, mask) if level >= threshold and m)
    return {"audible_seconds": round(audible * frame, 1), "covered_seconds": round(covered * frame, 1), "share": round(covered / audible, 3) if audible else 1.0}


def budget_seconds(duration: float, on_gpu: bool, share: float = MAX_RECOVERY_SHARE) -> float:
    return min(duration * share, MAX_RECOVERY_SECONDS_GPU if on_gpu else MAX_RECOVERY_SECONDS_CPU)


Decoder = Callable[[float, float, bool], list]   # (start, end, raise_gain) -> the segments (language-labelled, filtered) decoded from that stretch


def recover_gaps(gaps: Sequence[Gap], decode: Decoder, *, budget: float, should_stop: Optional[Callable[[], bool]] = None, piece: float = MAX_PIECE_SECONDS,
                 progress: Optional[Callable[[int, int], None]] = None) -> list[Any]:
    """Decode the gaps again, longest first, until ``budget`` seconds of audio have been spent. Returns the new segments; every gap gets a ``status`` (recovered / unreadable / skipped)."""
    new: list[Any] = []
    spent = 0.0
    ordered = sorted(gaps, key=lambda g: -g.seconds)
    for index, gap in enumerate(ordered):
        if should_stop is not None and should_stop():
            gap.status, gap.reason = "skipped", "stopped"
            continue
        if spent + min(gap.seconds, piece) > budget:
            gap.status, gap.reason = "skipped", "the re-decode budget for this recording was used up"
            continue
        found: list[Any] = []
        errors = 0
        position = gap.start
        while position < gap.end - 0.5:
            end = min(position + piece, gap.end)
            spent += end - position
            try:
                found.extend(decode(position, end, gap.kind == "quiet"))
            except Exception as exc:  # noqa: BLE001 - a failed re-decode leaves the gap reported, never fails the recording
                errors += 1
                log.warning("re-decode of %.0f-%.0f s failed: %s", position, end, exc)
            position = end
        words = sum(len(s.text.split()) for s in found)
        gap.words = words
        if found:
            gap.status = "recovered"
            for seg in found:
                meta = dict(getattr(seg, "language_meta", None) or {})
                meta["status"] = "repaired" if "status" not in meta else meta["status"]
                seg.language_meta = meta
            new.extend(found)
        else:
            gap.status = "unreadable"
            gap.reason = "re-decoding failed" if errors else ("very quiet: no speech could be recognised even with the volume raised" if gap.kind == "quiet" else
                                                              "no speech could be recognised (instrumental, singing too masked to read, or noise)")
        if progress is not None:
            progress(index + 1, len(ordered))
    return new


def summary(before: dict[str, float], after: dict[str, float], gaps: Sequence[Gap], recovered: int, budget: float) -> dict[str, Any]:
    """The coverage record stored with the lecture. Numbers are measured on the audio, not on the voice-activity filter; they say how much AUDIBLE time carries words, not how
    accurate the words are."""
    return {"method": "loudness of the audio compared with the loudness of the transcribed seconds (independent of the voice-activity filter)",
            "audible_seconds": after["audible_seconds"], "covered_share_before": before["share"], "covered_share_after": after["share"],
            "gaps_found": len(gaps), "quiet_gaps_found": sum(1 for g in gaps if g.kind == "quiet"), "gaps_recovered": sum(1 for g in gaps if g.status == "recovered"),
            "segments_recovered": recovered,
            "gaps_unreadable": sum(1 for g in gaps if g.status == "unreadable"), "gaps_skipped": sum(1 for g in gaps if g.status == "skipped"),
            "recovery_budget_seconds": round(budget, 1),
            "unrecovered": [g.to_dict() for g in gaps if g.status in ("unreadable", "skipped")][:60]}
