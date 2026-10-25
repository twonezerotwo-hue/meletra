"""Long-recording production layer: chunk planning, ownership merge, per-chunk checkpoints.

A 90-100 minute lecture is decoded in ``chunk_minutes`` pieces (default 15) whose decode windows overlap by
``overlap_seconds`` (default 3). Cut points are moved onto the nearest real pause (found by the VAD pass) so a
sentence is rarely split. Each chunk *owns* a range of the timeline; a segment is kept by the chunk whose range
contains its midpoint, so the overlap never produces duplicated text. Finished chunks are saved immediately and are
never decoded twice; a half-finished chunk resumes after its last saved segment.

Timestamps stay absolute: the backend decodes the slice ``[start, end]`` and adds ``start`` back.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Optional, Sequence

from .checkpoint import SegmentCheckpoint
from .schema import Segment
from .vocabulary import normalize_key

log = logging.getLogger(__name__)


@dataclass
class Chunk:
    index: int
    own_start: float            # the timeline range this chunk is responsible for
    own_end: float
    start: float                # decode window = own range padded by the overlap
    end: float

    @property
    def window(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict[str, float]:
        return {"index": self.index, "own_start": round(self.own_start, 2), "own_end": round(self.own_end, 2),
                "start": round(self.start, 2), "end": round(self.end, 2)}


def _pauses(speech: Sequence[tuple[float, float]], min_gap: float) -> list[tuple[float, float]]:
    """(start, end) of the silences between consecutive speech regions that last at least ``min_gap`` seconds."""
    gaps = []
    for (_, prev_end), (next_start, _) in zip(speech, speech[1:]):
        if next_start - prev_end >= min_gap:
            gaps.append((prev_end, next_start))
    return gaps


def _snap(nominal: float, pauses: Sequence[tuple[float, float]], window: float) -> float:
    """Middle of the longest pause within ``window`` seconds of ``nominal`` (or ``nominal`` if none)."""
    near = [(e - s, (s + e) / 2) for s, e in pauses if abs((s + e) / 2 - nominal) <= window]
    return max(near)[1] if near else nominal


def plan_chunks(
    duration: float, chunk_minutes: float = 15.0, overlap_seconds: float = 3.0,
    speech: Optional[Sequence[tuple[float, float]]] = None, snap_window_s: float = 60.0,
    min_chunk_seconds: float = 90.0, min_pause_s: float = 0.6,
) -> list[Chunk]:
    """Split ``duration`` seconds into chunks. Short files give a single chunk covering everything."""
    chunk_s = max(60.0, chunk_minutes * 60.0)
    overlap = max(0.0, min(overlap_seconds, chunk_s / 4))
    if duration <= chunk_s * 1.25:
        return [Chunk(0, 0.0, duration, 0.0, duration)]
    pauses = _pauses(speech, min_pause_s) if speech else []
    cuts: list[float] = []
    nominal = chunk_s
    while nominal < duration - min_chunk_seconds:
        cut = _snap(nominal, pauses, snap_window_s)
        if cuts and cut <= cuts[-1] + min_chunk_seconds:
            cut = nominal
        cuts.append(round(cut, 3))
        nominal = cuts[-1] + chunk_s
    bounds = [0.0, *cuts, duration]
    return [Chunk(i, bounds[i], bounds[i + 1], max(0.0, bounds[i] - (overlap if i else 0.0)),
                  min(duration, bounds[i + 1] + (overlap if i < len(bounds) - 2 else 0.0)))
            for i in range(len(bounds) - 1)]


def own_segments(chunk: Chunk, segments: Sequence[Segment], last: bool) -> list[Segment]:
    """Segments this chunk is responsible for: midpoint inside [own_start, own_end) (last chunk: up to the end)."""
    kept = []
    for seg in segments:
        mid = (seg.start + seg.end) / 2.0
        if chunk.own_start <= mid < chunk.own_end or (last and mid >= chunk.own_start):
            kept.append(seg)
    return kept


def _is_cut_fragment(short_key: str, long_key: str) -> bool:
    """The shorter text is the beginning of, or sits inside, the longer one (a sentence cut off at the window edge)."""
    return len(short_key) >= 6 and len(short_key) < len(long_key) and (long_key.startswith(short_key) or short_key in long_key)


def stitch(parts: Sequence[tuple[Chunk, Sequence[Segment]]], dedupe_window_s: float = 6.0,
           dedupe_ratio: float = 0.92) -> tuple[list[Segment], int]:
    """Concatenate chunk results in time order. A segment that repeats the last kept segments of the PREVIOUS chunk
    (the same sentence decoded on both sides of a cut) is dropped. Two cases:

    * the same sentence twice (near-identical text): the later one is dropped;
    * the sentence was cut off at the end of the previous window ("That's why when we have...") and appears complete at the
      start of the next one: the cut-off fragment is dropped and the complete sentence kept.

    Repeats inside one chunk are never touched: they may be a hallucination loop, which must stay visible and gets flagged.
    Text is never trimmed or edited. Returns (segments, dropped)."""
    out: list[Segment] = []
    origin: list[int] = []                     # chunk index of every kept segment
    dropped = 0
    last_index = max((c.index for c, _ in parts), default=0)
    for chunk, segs in sorted(parts, key=lambda p: p[0].index):
        for seg in own_segments(chunk, segs, last=chunk.index == last_index):
            if out and origin[-1] != chunk.index and seg.start - out[-1].end < dedupe_window_s:
                a, b = normalize_key(out[-1].text), normalize_key(seg.text)
                if a and b and (a == b or (min(len(a), len(b)) >= 25 and SequenceMatcher(None, a, b).ratio() >= dedupe_ratio)):
                    dropped += 1
                    continue
                # cut-off fragments of the previous chunk that overlap this segment in time (look back over its last two)
                key = normalize_key(seg.text)
                removed = False
                for back in (1, 2):
                    if len(out) < back or origin[-back] == chunk.index:
                        break
                    prev = out[-back]
                    overlap = min(prev.end, seg.end) - max(prev.start, seg.start)
                    if overlap > 0.3 * max(min(prev.end - prev.start, seg.end - seg.start), 0.1) \
                            and _is_cut_fragment(normalize_key(prev.text), key):
                        del out[-back]
                        del origin[-back]
                        dropped += 1
                        removed = True
                        break
                if not removed and len(out) and origin[-1] != chunk.index:
                    prev_key = normalize_key(out[-1].text)
                    overlap = min(out[-1].end, seg.end) - max(out[-1].start, seg.start)
                    if overlap > 0.3 * max(min(out[-1].end - out[-1].start, seg.end - seg.start), 0.1) \
                            and _is_cut_fragment(key, prev_key):
                        dropped += 1                  # the new segment is only a piece of what was already kept
                        continue
            out.append(seg)
            origin.append(chunk.index)
    order = sorted(range(len(out)), key=lambda i: (out[i].start, out[i].end))
    return [out[i] for i in order], dropped


class ChunkStore:
    """One :class:`SegmentCheckpoint` per chunk below ``<output>/checkpoint/chunks``."""

    def __init__(self, root: Path, base_fingerprint: dict[str, Any]) -> None:
        self.root = Path(root)
        self.base = base_fingerprint

    def checkpoint(self, chunk: Chunk) -> SegmentCheckpoint:
        fingerprint = {**self.base, "chunk": chunk.to_dict()}
        return SegmentCheckpoint(self.root / f"chunk_{chunk.index:04d}", fingerprint)

    def status(self, chunk: Chunk) -> tuple[str, list[Segment]]:
        """('completed' | 'partial' | 'none', saved segments) for a chunk."""
        ck = self.checkpoint(chunk)
        segments, _ = ck.load()
        if not ck.meta_path.is_file() or ck.read_status() is None:
            return "none", []
        return ("completed" if ck.read_status() == "completed" else "partial"), segments


def correct_offsets(chunk: Chunk, segments: Sequence[Segment], slack: float = 1.0) -> bool:
    """Make chunk timestamps absolute if a backend returned window-relative ones. Returns True if it shifted them.

    A segment can never start before the window it was decoded from. If the earliest segment starts before the window
    does (while every segment still fits inside the window length), the times are relative to the window start.
    """
    if chunk.start <= slack or not segments:
        return False
    earliest = min(seg.start for seg in segments)
    latest = max(seg.end for seg in segments)
    if earliest >= chunk.start - slack or latest > chunk.window + slack:
        return False                      # already absolute (or not clearly relative): leave it to the timestamp audit
    for seg in segments:
        seg.start += chunk.start
        seg.end += chunk.start
        for word in seg.words:
            word.start += chunk.start
            word.end += chunk.start
    return True
