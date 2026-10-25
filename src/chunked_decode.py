"""Decode a long recording chunk by chunk.

* every chunk has its own checkpoint: a finished chunk is never decoded again, a half-finished one resumes after its
  last saved segment (timestamps stay absolute);
* a chunk that fails is retried (the model is reloaded in between); when it finally fails, everything finished so far
  stays on disk and the SAME command resumes there ("partial recovery");
* the user can ask to stop: the segment being written is saved, the checkpoint is closed and the run reports
  "paused"; the same file resumes later ("safe cancel");
* after each chunk the medical context adapts a little (terms heard, acronyms defined) for the next one;
* the hotword safeguard (:mod:`src.guard`) runs for every chunk request;
* two repairs re-decode only the damaged window, with hotwords off: decoder loops (``_repair_loops``) and stretches of real
  speech that came back without any text (``repair_gaps``).
"""

from __future__ import annotations

import json
import logging
import re
import time
from difflib import SequenceMatcher
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from .acronyms import AcronymEngine
from .budget import TokenBudgets
from .context_engine import MedicalContextEngine
from .diagnostics import CudaUnavailableError
from .guard import prepare_decode_profile
from . import quality as quality_mod
from .long_audio import Chunk, ChunkStore, correct_offsets, own_segments
from .model_manager import DecodeRequest, UnknownModelError
from .schema import Segment, TranscriptionResult
from .terminology_tr import TermPairs, localize_hotwords
from .transcriber import ProgressReporter, Transcriber, format_clock
from .uncertainty import find_repetition
from .vocabulary import normalize_key, tokenize

log = logging.getLogger(__name__)


class PauseRequested(Exception):
    """The user asked to stop. Raised after the current segment was saved to the checkpoint."""


NON_RETRYABLE = (CudaUnavailableError, UnknownModelError, PauseRequested, KeyboardInterrupt, SystemExit)

ECHO_HOTWORD_FRACTION = 0.6      # share of a segment's words that come from the hotword list ...
ECHO_MAX_AVG_LOGPROB = -0.9      # ... while the decoder was unsure of it: the list is being echoed, not speech
GAP_NO_SPEECH_MAX = 0.6          # a re-decoded segment the model itself calls "no speech" is not inserted
GAP_MIN_AVG_LOGPROB = -1.0


MAX_WHOLE_WINDOW_SECONDS = 120.0       # a looped stretch longer than this is not re-decoded as one block


def phrase_loops(text: str, min_repeats: int = 3, min_size: int = 3, max_size: int = 6) -> bool:
    """A phrase of at least ``min_size`` words said ``min_repeats`` times in a row ("acetyl CoA carboxylase" x3).
    Real speech does not do that; a single repeated word ("no, no, no") is left to the stricter 4-repeat rule."""
    words = [normalize_key(w) for w in tokenize(text)]
    words = [w for w in words if w]
    for size in range(min_size, max_size + 1):
        for start in range(0, len(words) - size * min_repeats + 1):
            gram = words[start:start + size]
            if all(words[start + k * size:start + (k + 1) * size] == gram for k in range(1, min_repeats)):
                return True
    return False


def is_decoder_loop(seg: Segment, hotword_vocabulary: set[str], min_repeats: int, max_ngram: int) -> bool:
    """A looped segment: repeated words/phrases, or a low-confidence echo made mostly of hotword vocabulary."""
    if find_repetition(seg.text, min_repeats, max_ngram) or phrase_loops(seg.text):
        return True
    words = [w.casefold() for w in tokenize(seg.text)]
    if len(words) >= 2 and seg.avg_logprob is not None and seg.avg_logprob < ECHO_MAX_AVG_LOGPROB:
        return sum(w in hotword_vocabulary for w in words) / len(words) >= ECHO_HOTWORD_FRACTION
    return False


def _loop_words(text: str) -> frozenset[str]:
    """Words of a segment for comparing repeats: digits are kept ("table 1" and "table 2" are different sentences)."""
    return frozenset(re.findall(r"\w+", text.casefold()))


def repeated_runs(segments: Sequence[Segment], min_run: int = 3, long_chars: int = 60, similarity: float = 0.85,
                  min_chars: int = 8) -> list[tuple[int, int]]:
    """Index ranges ``(first, last)`` of consecutive segments that repeat each other: a loop that the decoder spread over many
    segments instead of one ("This is the lysosome." twenty times in a row; one 30 s sentence coming back for eleven minutes).

    A run needs ``min_run`` segments whose word sets are (near) identical to the one before (Jaccard >= ``similarity``), or just
    two when the repeated text is long (``long_chars``): saying a short phrase twice is natural, a 100-character sentence twice in
    a row is not.
    """
    texts = [s.text.strip() for s in segments]
    words = [_loop_words(t) for t in texts]
    runs: list[tuple[int, int]] = []
    start = 0
    for index in range(1, len(segments) + 1):
        same = False
        if index < len(segments) and len(texts[index]) >= min_chars and len(texts[index - 1]) >= min_chars:
            a, b = words[index], words[index - 1]
            same = bool(a and b) and len(a & b) / len(a | b) >= similarity
        if same:
            continue
        length = index - start
        if length >= min_run or (length >= 2 and len(texts[start]) >= long_chars):
            runs.append((start, index - 1))
        start = index
    return runs


def uncovered_speech(speech: Sequence[tuple[float, float]], segments: Sequence[Segment], min_seconds: float
                     ) -> list[tuple[float, float]]:
    """Stretches of VAD speech of at least ``min_seconds`` that no segment covers (text that went missing).

    Uncovered pieces on both sides of a VAD cut count as one stretch (see :data:`quality.GAP_BRIDGE_SECONDS`).
    """
    covered = sorted((s.start, s.end) for s in segments if s.text.strip() and s.end > s.start)
    merged: list[tuple[float, float]] = []
    for start, end in covered:
        if merged and start <= merged[-1][1] + 0.05:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    pieces = quality_mod.uncovered_pieces(speech, merged)
    return [(a, b) for a, b in quality_mod.bridge_pieces(pieces, merged) if b - a >= min_seconds]


def clamp_words(seg: Segment) -> Segment:
    """Keep the word times inside the segment (after a segment was clipped to a window its words must not stick out)."""
    for w in seg.words:
        w.start = min(max(w.start, seg.start), seg.end)
        w.end = min(max(w.end, w.start), seg.end)
    return seg


def overlap_fraction(seg: Segment, others: Sequence[Segment]) -> float:
    """Share of ``seg`` that is already covered by the text-carrying ``others`` (0..1)."""
    duration = max(seg.end - seg.start, 1e-6)
    spans = sorted((max(o.start, seg.start), min(o.end, seg.end)) for o in others if o.text.strip() and o.end > seg.start and o.start < seg.end)
    covered, cursor = 0.0, seg.start
    for a, b in spans:
        a = max(a, cursor)
        if b > a:
            covered += b - a
            cursor = b
    return min(covered / duration, 1.0)


def trim_to_gap(seg: Segment, a: float, b: float, tolerance: float = 0.15) -> Optional[Segment]:
    """Cut a recovered segment down to the words that really lie inside the gap ``[a, b]``.

    The re-decode starts a little before the gap (context), so its first words often repeat the end of the neighbouring
    segment ("... generates NADPH, and" heard twice). With word timestamps the words whose middle lies outside the gap are
    dropped and the text and times are rebuilt; without them the segment is returned unchanged.
    """
    if not seg.words:
        return seg
    inside = [w for w in seg.words if a - tolerance <= (w.start + w.end) / 2 <= b + tolerance]
    if len(inside) == len(seg.words):
        return seg
    if not inside:
        return None
    seg.words = inside
    seg.text = " " + "".join(w.word if w.word.startswith(" ") else " " + w.word for w in inside).strip()
    seg.start, seg.end = inside[0].start, inside[-1].end
    return seg


class ChunkFailedError(RuntimeError):
    def __init__(self, chunk: Chunk, attempts: int, cause: BaseException) -> None:
        super().__init__(f"chunk {chunk.index + 1} ({format_clock(chunk.start)} - {format_clock(chunk.end)}) failed "
                         f"after {attempts} attempt(s): {type(cause).__name__}: {cause}")
        self.chunk, self.attempts, self.cause = chunk, attempts, cause


@dataclass
class DecodeOutcome:
    parts: list[tuple[Chunk, list[Segment]]] = field(default_factory=list)
    chunk_log: list[dict[str, Any]] = field(default_factory=list)
    result: Optional[TranscriptionResult] = None          # the last real decode result (language, model, ...)
    load_seconds: float = 0.0
    resumed_segments: int = 0
    hotwords_applied: list[str] = field(default_factory=list)
    hotword_tokens: Optional[int] = None
    notes: list[str] = field(default_factory=list)
    capabilities_used: dict[str, bool] = field(default_factory=dict)
    fallback_used: bool = False
    conditioning_used: Optional[bool] = None   # condition_on_previous_text of the requests that carried hotwords
    loop_repairs: list[dict[str, Any]] = field(default_factory=list)   # looped windows re-decoded without hotwords
    gap_repairs: list[dict[str, Any]] = field(default_factory=list)    # silent stretches of real speech re-decoded
    decode_seconds: float = 0.0                # decoding done by THIS run (model load excluded)
    decoded_audio_seconds: float = 0.0         # ... and how much audio that was (0 when everything came from checkpoints)
    repair_seconds: float = 0.0                # loop / gap re-decoding done by THIS run
    repair_audio_seconds: float = 0.0


class ChunkedDecoder:
    def __init__(
        self, *, transcriber: Transcriber, wav: Path, duration: float, chunks: Sequence[Chunk], store: ChunkStore,
        plan_dict: Callable[[], dict[str, Any]], cfg: dict[str, Any], base_profile: dict[str, Any],
        budgets: TokenBudgets, language: Optional[str], hotword_source: Callable[[], list[str]],
        progress: Optional[ProgressReporter] = None, multilingual: bool = False, resume: bool = True, retries: int = 2,
        language_plan: Optional[list[tuple[float, float, str, str]]] = None,
        engine: Optional[MedicalContextEngine] = None, acronyms: Optional[AcronymEngine] = None,
        pairs: Optional[TermPairs] = None, sleep: Callable[[float], None] = time.sleep,
        events: Optional[Callable[[dict[str, Any]], None]] = None, should_stop: Optional[Callable[[], bool]] = None,
    ) -> None:
        self.transcriber, self.wav, self.duration = transcriber, Path(wav), duration
        self.chunks, self.store, self.plan_dict = list(chunks), store, plan_dict
        self.cfg, self.base_profile, self.budgets = cfg, base_profile, budgets
        self.language, self.hotword_source = language, hotword_source
        self.progress, self.multilingual, self.resume, self.retries = progress, multilingual, resume, max(0, retries)
        self.language_plan = language_plan                       # Auto Detect over time: (start, end, language, vad) runs decoded one by one
        self.engine, self.acronyms, self.pairs = engine, acronyms, pairs
        self._sleep = sleep
        self._events, self._should_stop = events, should_stop
        self.outcome = DecodeOutcome()
        self._current: list[Segment] = []
        self._pause_announced = False

    # ------------------------------------------------------------------ public
    def pending_start(self) -> float:
        """Where this run's own decoding begins (everything before it comes from checkpoints)."""
        for chunk in self.chunks:
            status, saved = self._status(chunk)
            if status != "completed":
                return max(chunk.start, saved[-1].end) if saved else chunk.start
        return self.duration

    def partial_segments(self) -> list[Segment]:
        """Everything finished so far (completed chunks + the chunk that was running), in time order."""
        done = [seg for chunk, segs in self.outcome.parts for seg in own_segments(chunk, segs, last=False)]
        return sorted([*done, *self._current], key=lambda s: (s.start, s.end))

    def run(self) -> DecodeOutcome:
        total = len(self.chunks)
        for chunk in self.chunks:
            self._check_stop()
            status, saved = self._status(chunk)
            label = f"CHUNK {chunk.index + 1}/{total} ({format_clock(chunk.start)} - {format_clock(chunk.end)})"
            if self.progress is not None:
                self.progress.chunk_index, self.progress.chunk_count = chunk.index, total
            if status == "completed":
                log.info("%s: already finished in an earlier run (%d segments) - skipped", label, len(saved))
                self.outcome.resumed_segments += len(saved)
                self._emit(type="chunk_finished", index=chunk.index + 1, chunks=total, skipped=True,
                           segments=len(saved))
                saved = self._repair_loops(chunk, saved, label)
                self._finish_chunk(chunk, saved, resumed=True, attempts=0, seconds=0.0, applied=0, tokens=None)
                continue
            if saved:
                log.info("%s: resuming after %d saved segments (up to %s)", label, len(saved), format_clock(saved[-1].end))
                self.outcome.resumed_segments += len(saved)
            else:
                log.info("%s: decoding", label)
            self._emit(type="chunk_started", index=chunk.index + 1, chunks=total, start=chunk.start, end=chunk.end,
                       resumed_segments=len(saved))
            self._decode(chunk, saved if status == "partial" else [], label)
            self._emit(type="chunk_finished", index=chunk.index + 1, chunks=total, skipped=False)
            self._emit(type="checkpoint_saved", chunk=chunk.index + 1, chunks=total)
        return self.outcome

    def resume_summary(self) -> dict[str, int]:
        """How much of this file is already on disk: finished chunks and half-finished ones."""
        completed = partial = 0
        for chunk in self.chunks:
            status, saved = self._status(chunk)
            if status == "completed":
                completed += 1
            elif saved:
                partial += 1
        return {"completed": completed, "partial": partial, "chunks": len(self.chunks)}

    def cumulative_stats(self) -> dict[str, float]:
        """Decode seconds / audio seconds over ALL sessions that worked on this file (read from the chunk checkpoints)."""
        seconds = audio = 0.0
        for chunk in self.chunks:
            stats = self.store.checkpoint(chunk).read_stats()
            seconds += float(stats.get("decode_seconds", 0.0))
            audio += float(stats.get("audio_seconds", 0.0))
        return {"decode_seconds": seconds, "audio_seconds": audio,
                "rtf": (seconds / audio) if audio > 0 else 0.0}

    # ------------------------------------------------------------------ internals
    def _emit(self, **event: Any) -> None:
        if self._events is not None:
            self._events(event)

    def _check_stop(self) -> None:
        if self._should_stop is not None and self._should_stop():
            if not self._pause_announced:
                self._pause_announced = True
                self._emit(type="pause_requested")
            raise PauseRequested()

    def _status(self, chunk: Chunk) -> tuple[str, list[Segment]]:
        return self.store.status(chunk) if self.resume else ("none", [])

    def _decode(self, chunk: Chunk, base: list[Segment], label: str) -> None:
        checkpoint = self.store.checkpoint(chunk)
        prior = checkpoint.read_stats() if base else {}
        attempts, started = 0, time.perf_counter()
        first_offset: Optional[float] = None
        session_seconds = 0.0

        def stats(offset_now: float) -> dict[str, Any]:
            audio_done = max(offset_now - (first_offset if first_offset is not None else offset_now), 0.0)
            return {"decode_seconds": round(float(prior.get("decode_seconds", 0.0)) + session_seconds, 2),
                    "audio_seconds": round(float(prior.get("audio_seconds", 0.0)) + audio_done, 2)}

        while True:
            attempts += 1
            offset = chunk.start if not base else max(chunk.start, base[-1].end)
            first_offset = offset if first_offset is None else first_offset
            if base and chunk.end - offset <= 0.5:
                checkpoint.mark(self.plan_dict(), "completed", stats(chunk.end))
                checkpoint.close()
                self._finish_chunk(chunk, base, resumed=True, attempts=attempts, seconds=0.0, applied=0, tokens=None)
                return
            hotwords = self._hotwords()
            profile, budget, notes = prepare_decode_profile(self.base_profile, hotwords, self.cfg, self.budgets)
            for note in notes:
                if note not in self.outcome.notes:
                    self.outcome.notes.append(note)
                    log.warning(note)
            if hotwords:
                self.outcome.conditioning_used = bool(profile.get("condition_on_previous_text"))
            collected = list(base)
            self._current = collected
            checkpoint.begin(self.plan_dict(), base)

            def on_segment(segment: Segment, _collected: list[Segment] = collected) -> None:
                _collected.append(segment)
                checkpoint.add(segment)
                self._check_stop()                         # the segment is already on disk: stopping here loses nothing

            def on_restart(_collected: list[Segment] = collected, _base: list[Segment] = base) -> None:
                _collected[:] = _base
                checkpoint.rollback()

            whole_rest = chunk.end >= self.duration - 0.01
            request = DecodeRequest(
                audio_path=self.wav, language=self.language, profile=profile, decoding=self.cfg["decoding"],
                vad_parameters=self.cfg["vad"], hotwords=hotwords, multilingual=self.multilingual and not self.language_plan, language_plan=self.language_plan,
                progress=self.progress, hotword_token_budget=budget, start_offset=offset,
                max_seconds=None if whole_rest else chunk.end - offset, total_duration=self.duration,
                on_segment=on_segment, on_restart=on_restart,
            )
            attempt_started = time.perf_counter()
            try:
                result = self.transcriber.transcribe(request)
            except PauseRequested:
                session_seconds += time.perf_counter() - attempt_started
                checkpoint.mark(self.plan_dict(), "running", stats(collected[-1].end if collected else offset))
                checkpoint.close()
                raise
            except NON_RETRYABLE:
                checkpoint.close()
                raise
            except Exception as exc:  # noqa: BLE001 - retried, then reported as a failed chunk
                session_seconds += time.perf_counter() - attempt_started
                checkpoint.close()
                base = list(collected)                       # what reached the disk before the failure is kept
                if attempts > self.retries:
                    raise ChunkFailedError(chunk, attempts, exc) from exc
                log.warning("%s failed (%s: %s); retry %d/%d - reloading the model, keeping %d saved segments",
                            label, type(exc).__name__, str(exc).splitlines()[0][:160], attempts, self.retries, len(base))
                self.transcriber.close()
                self._sleep(min(5.0 * attempts, 20.0))
                continue

            session_seconds += max(time.perf_counter() - attempt_started - result.load_seconds, 0.0)
            fresh = list(result.segments)
            if correct_offsets(chunk, fresh):
                log.warning("%s: backend returned window-relative timestamps; offset %.1fs added", label, chunk.start)
            segments = [*base, *fresh]
            for seg in segments[checkpoint.saved_count:]:    # adapters that do not stream segments
                checkpoint.add(seg)
            run_stats = stats(chunk.end)
            checkpoint.mark(self.plan_dict(), "completed", run_stats)
            checkpoint.close()
            self.outcome.decode_seconds += session_seconds
            self.outcome.decoded_audio_seconds += max(chunk.end - first_offset, 0.0)
            segments = self._repair_loops(chunk, segments, label, hotwords_were_sent=bool(hotwords))
            self._record_result(result)
            self._finish_chunk(chunk, segments, resumed=False, attempts=attempts, seconds=time.perf_counter() - started,
                               applied=len(result.hotwords_applied), tokens=result.hotword_tokens)
            return

    # ------------------------------------------------------------------ repairs
    def _redecode(self, start: float, end: float, vocabulary: set[str], min_repeats: int, max_ngram: int
                  ) -> tuple[list[Segment], Optional[str]]:
        """Decode ``[start, end]`` again with the plain profile (no hotwords, no conditioning). Returns (segments, problem)."""
        profile = dict(self.base_profile)
        profile["condition_on_previous_text"] = False
        request = DecodeRequest(
            audio_path=self.wav, language=self.language, profile=profile, decoding=self.cfg["decoding"],
            vad_parameters=self.cfg["vad"], hotwords=[], multilingual=self.multilingual and not self.language_plan, language_plan=self.language_plan, start_offset=max(start, 0.0),
            max_seconds=max(end - start, 1.0), total_duration=self.duration)
        began = time.perf_counter()
        try:
            result = self.transcriber.transcribe(request)
        except PauseRequested:
            raise
        except Exception as exc:  # noqa: BLE001 - the original (flagged) text simply stays
            return [], f"re-decode failed: {type(exc).__name__}"
        self.outcome.repair_seconds += max(time.perf_counter() - began - result.load_seconds, 0.0)
        self.outcome.repair_audio_seconds += max(end - start, 0.0)
        self.outcome.load_seconds += result.load_seconds
        fresh = [s for s in result.segments if s.text.strip() and not quality_mod.is_stock_phrase(s.text)]
        if any(is_decoder_loop(s, vocabulary, min_repeats, max_ngram) for s in fresh):
            return [], "the hotword-free re-decode looped as well"
        return fresh, None

    @staticmethod
    def _is_better(fresh: Sequence[Segment], original: Sequence[Segment]) -> bool:
        """A repair replaces the original only if it is not itself a pile of repeats and says something new.

        Words that occur in the repair count only once per distinct segment text; a repair whose segments repeat each other as much as
        the original does is no improvement (the original is kept, flagged)."""
        if not fresh:
            return False
        fresh_runs = sum(b - a + 1 for a, b in repeated_runs(list(fresh)))
        original_runs = sum(b - a + 1 for a, b in repeated_runs(list(original)))
        distinct_fresh = len({normalize_key(s.text) for s in fresh})
        return fresh_runs < max(original_runs, 1) and distinct_fresh >= 1

    def _redecode_in_pieces(self, start: float, end: float, vocabulary: set[str], min_repeats: int, max_ngram: int,
                            neighbours: Sequence[Segment] = ()) -> tuple[list[Segment], Optional[str]]:
        """Decode ``[start, end]`` again in short windows (padding on both sides, same plain settings: no hotwords, no conditioning,
        temperature 0, beam 5, word timestamps).

        A 30 s window that loops often decodes cleanly when it is cut into 8 s pieces: the decoder never gets a long, uncertain
        context to repeat. Every piece is judged on its own (no loop, enough speech density, no stock phrase, confident enough);
        the result is used only if the accepted pieces cover at least half of the window, otherwise the caller keeps the original.
        """
        long_cfg = self.cfg.get("long_audio", {})
        piece = max(3.0, float(long_cfg.get("loop_piece_seconds", 8.0)))
        padding = max(0.0, float(long_cfg.get("loop_piece_padding_seconds", 1.0)))
        accepted: list[Segment] = []
        covered = 0.0
        cursor = start
        while cursor < end - 0.25:
            self._check_stop()
            stop = min(cursor + piece, end)
            fresh, problem = self._redecode(max(cursor - padding, 0.0), min(stop + padding, self.duration), vocabulary, min_repeats,
                                            max_ngram)
            keep = self._accept_gap_segments(fresh, cursor, stop, [*accepted, *neighbours]) if fresh and not problem else []
            for seg in keep:
                seg.start, seg.end = max(seg.start, cursor), min(seg.end, stop)
                clamp_words(seg)
                if seg.end > seg.start:
                    covered += seg.end - seg.start
                    accepted.append(seg)
            cursor = stop
        if covered < 0.5 * max(end - start, 1.0):
            return [], "re-decoding in short pieces did not give enough clean speech either"
        return accepted, None

    def _repair_loops(self, chunk: Chunk, segments: list[Segment], label: str,
                      hotwords_were_sent: Optional[bool] = None) -> list[Segment]:
        """Re-decode ONLY the windows where the model looped, with hotwords and conditioning off.

        Measured on the real 87-minute lecture: with hotwords on, large-v3 occasionally replaced 15-30 s of real speech by a
        loop ("glycogen storage disease, glycogen storage disease, ..." = an echo of the hotword list). The window is decoded
        again with the plain, hotword-free profile; the new text replaces the loop only if it is not itself a loop,
        otherwise the original stays (and stays flagged). Everything that was replaced is recorded in the metadata.
        """
        if not self.cfg.get("long_audio", {}).get("loop_repair", True):
            return segments
        if self.language_plan and any(vad == "off" for _s, _e, _l, vad in self.language_plan):
            return segments                                  # singing / speech over music: a repeated line is a chorus, not a decoder loop (loops of filler are rejected by the music guard)
        unc = self.cfg.get("uncertainty", {})
        min_repeats, max_ngram = int(unc.get("repetition_min_repeats", 4)), int(unc.get("repetition_max_ngram", 4))
        if hotwords_were_sent is None:                       # a chunk that was finished in an earlier run
            hotwords_were_sent = bool(self._hotwords())
        vocabulary = {w.casefold() for term in self._hotwords() for w in tokenize(term)}
        spans = sorted({(i, i) for i, seg in enumerate(segments) if is_decoder_loop(seg, vocabulary, min_repeats, max_ngram)}
                       | set(repeated_runs(segments)))
        merged: list[list[int]] = []
        for first, last in spans:                            # overlapping / touching spans become one window
            if merged and first <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], last)
            else:
                merged.append([first, last])
        if not merged:
            return segments
        repaired: list[Segment] = []
        changed = False
        previous = 0
        for first, last in merged:
            self._check_stop()
            members = segments[first:last + 1]
            lo, hi = members[0].start, members[-1].end
            repaired += segments[previous:first]
            previous = last + 1
            replaced = " ".join(s.text.strip() for s in members)
            record: dict[str, Any] = {"chunk": chunk.index, "start": round(lo, 2), "end": round(hi, 2), "segments": len(members),
                                      "replaced_text": replaced[:240], "repaired": False}
            fresh, problem = [], None
            if hotwords_were_sent and hi - lo <= MAX_WHOLE_WINDOW_SECONDS:   # first try: the same window without the hotword bias
                fresh, problem = self._redecode(lo, hi, vocabulary, min_repeats, max_ngram)
            if not fresh:                                    # still looping (or nothing to switch off): decode it in short pieces
                outside = [s for i, s in enumerate(segments) if not first <= i <= last and s.end >= lo - 3.0 and s.start <= hi + 3.0]
                fresh, piece_problem = self._redecode_in_pieces(lo, hi, vocabulary, min_repeats, max_ngram, neighbours=outside)
                problem = piece_problem if not fresh else None
            for new in fresh:
                new.start, new.end = max(new.start, lo), min(new.end, hi)
                clamp_words(new)
            if fresh and self._is_better(fresh, members):
                repaired += fresh
                record.update(repaired=True, new_text=" ".join(s.text.strip() for s in fresh)[:240])
                changed = True
                log.warning("%s: LOOP REPAIR %s-%s (%d segment(s)): '%s ...' -> '%s ...'", label, format_clock(lo), format_clock(hi),
                            len(members), record["replaced_text"][:50], record["new_text"][:50])
            else:
                repaired += members
                record["note"] = problem or ("the re-decode was not better than the original; the original text was kept" if fresh
                                            else "the hotword-free re-decode was empty; the original text was kept")
                log.warning("%s: loop at %s-%s could not be repaired (%s)", label, format_clock(lo), format_clock(hi), record["note"])
            self.outcome.loop_repairs.append(record)
            self._emit(type="loop_repaired", start=lo, end=hi, repaired=record["repaired"],
                       replaced_text=record["replaced_text"][:80], new_text=record.get("new_text", "")[:80])
        repaired += segments[previous:]
        if changed:                                          # persist, so a later resume does not repeat the repair
            checkpoint = self.store.checkpoint(chunk)
            stats = checkpoint.read_stats()
            checkpoint.begin(self.plan_dict(), repaired)
            checkpoint.mark(self.plan_dict(), "completed", stats or None)
            checkpoint.close()
        return repaired

    def repair_gaps(self, segments: list[Segment], speech: Sequence[tuple[float, float]]) -> list[Segment]:
        """Recover stretches of real speech (per the independent VAD pass) that came back without any text.

        Measured on the real lecture: four such stretches (44 s) were silently skipped by the hotword-biased decode;
        decoded again without hotwords they held ordinary sentences. Each stretch is re-decoded ONCE (with ``padding``
        seconds of context on both sides; hotwords off, conditioning off, beam 5, temperature 0, word timestamps on - the
        profile's own settings), and the verdict is kept in ``gap_repairs.json`` so a resume never repeats it. The text
        is inserted only if the model itself says it is speech, it is not a loop, it is not a copy of a neighbouring
        segment and it is dense enough to be speech. Nothing existing is changed and the full recording is never decoded again.
        """
        long_cfg = self.cfg.get("long_audio", {})
        if not long_cfg.get("gap_repair", True) or not speech:
            return segments
        min_gap = float(long_cfg.get("gap_repair_min_seconds", 5.0))
        padding = max(0.0, float(long_cfg.get("gap_repair_padding_seconds", 2.0)))
        self._emit(type="gap_scan_started", minimum_seconds=min_gap)
        gaps = uncovered_speech(speech, segments, min_gap)
        known = self._load_gap_repairs()
        unc = self.cfg.get("uncertainty", {})
        min_repeats, max_ngram = int(unc.get("repetition_min_repeats", 4)), int(unc.get("repetition_max_ngram", 4))
        vocabulary = {w.casefold() for term in self._hotwords() for w in tokenize(term)}
        inserted: list[Segment] = []
        for number, (a, b) in enumerate(gaps, start=1):
            key = f"{round(a, 1)}-{round(b, 1)}"
            self._emit(type="gap_found", index=number, gaps=len(gaps), start=a, end=b, seconds=round(b - a, 1))
            record = known.get(key)
            if record is None:
                self._check_stop()
                self._emit(type="gap_repair_started", index=number, gaps=len(gaps), start=a, end=b)
                lo, hi = max(a - padding, 0.0), min(b + padding, self.duration)
                fresh, problem = self._redecode(lo, hi, vocabulary, min_repeats, max_ngram)
                if not fresh:                                    # looped or empty as one block: try again in short pieces
                    fresh, piece_problem = self._redecode_in_pieces(a, b, vocabulary, min_repeats, max_ngram)
                    problem = None if fresh else (problem or piece_problem)
                keep = self._accept_gap_segments(fresh, a, b, segments)
                record = {"start": round(a, 2), "end": round(b, 2), "seconds": round(b - a, 1), "inserted": len(keep),
                          "text": " ".join(s.text.strip() for s in keep)[:300], "problem": problem,
                          "segments": [s.to_checkpoint() for s in keep]}
                known[key] = record
                self._save_gap_repairs(known)
                if keep:
                    log.warning("GAP REPAIR %s-%s (%.0f s of speech without text): '%s ...'", format_clock(a),
                                format_clock(b), b - a, record["text"][:70])
                self._emit(type="gap_repair_finished", index=number, gaps=len(gaps), start=a, end=b,
                           repaired=bool(keep), text=record["text"][:120], reason=problem or ("" if keep else "no accepted text"))
            self.outcome.gap_repairs.append({k: v for k, v in record.items() if k != "segments"})
            inserted += [Segment.from_checkpoint(d) for d in record.get("segments", [])]
        if not inserted:
            return segments
        return sorted([*segments, *inserted], key=lambda s: (s.start, s.end))

    @staticmethod
    def _accept_gap_segments(fresh: list[Segment], a: float, b: float, existing: Sequence[Segment]) -> list[Segment]:
        """Quality gate for text recovered inside the gap ``[a, b]``."""
        keep: list[Segment] = []
        for seg in fresh:
            middle = (seg.start + seg.end) / 2
            if not (a - 0.5 <= middle <= b + 0.5):
                continue                                              # belongs to the padding, i.e. to a neighbour
            seg = trim_to_gap(seg, a, b)
            if seg is None:
                continue
            if seg.no_speech_prob is not None and seg.no_speech_prob >= GAP_NO_SPEECH_MAX:
                continue
            if seg.avg_logprob is not None and seg.avg_logprob <= GAP_MIN_AVG_LOGPROB:
                continue
            words = len(tokenize(seg.text))
            if words < 2 or words / max(seg.end - seg.start, 0.5) > 8.0:     # a lone word / impossible speaking rate
                continue
            if quality_mod.is_stock_phrase(seg.text):
                continue                                              # "Thanks for watching" invented on silence
            key = normalize_key(seg.text)
            near = [e for e in existing if e.end >= seg.start - 3.0 and e.start <= seg.end + 3.0]
            if any(SequenceMatcher(None, key, normalize_key(e.text)).ratio() >= 0.85 for e in near):
                continue                                              # the padding re-heard a neighbour's sentence
            seg.start, seg.end = max(seg.start, a - 0.5), min(seg.end, b + 0.5)
            clamp_words(seg)
            if overlap_fraction(seg, existing) > 0.5:
                continue                                              # the same stretch already has text (a neighbour's)
            keep.append(seg)
        return keep

    def _gap_file(self) -> Path:
        return self.store.root.parent / "gap_repairs.json"

    def _load_gap_repairs(self) -> dict[str, Any]:
        try:
            data = json.loads(self._gap_file().read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not self.resume or data.get("fingerprint") != json.loads(json.dumps(self.store.base, default=str)):
            return {}
        return dict(data.get("gaps", {}))

    def _save_gap_repairs(self, known: dict[str, Any]) -> None:
        path = self._gap_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"fingerprint": json.loads(json.dumps(self.store.base, default=str)), "gaps": known}
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    # ------------------------------------------------------------------ bookkeeping
    def _hotwords(self) -> list[str]:
        words = self.hotword_source()
        if self.pairs is not None:
            words = localize_hotwords(words, self.language, self.pairs)
        return words

    def _record_result(self, result: TranscriptionResult) -> None:
        out = self.outcome
        out.result = result
        out.load_seconds += result.load_seconds
        out.fallback_used = out.fallback_used or result.fallback_used
        out.capabilities_used = {**out.capabilities_used, **result.capabilities_used}
        out.notes += [n for n in result.notes if n not in out.notes]
        for term in result.hotwords_applied:
            if term not in out.hotwords_applied:
                out.hotwords_applied.append(term)
        if result.hotword_tokens is not None:
            out.hotword_tokens = max(out.hotword_tokens or 0, result.hotword_tokens)

    def _finish_chunk(self, chunk: Chunk, segments: list[Segment], *, resumed: bool, attempts: int, seconds: float,
                      applied: int, tokens: Optional[int]) -> None:
        last = chunk.index == len(self.chunks) - 1
        self.outcome.parts.append((chunk, segments))
        self._current = []
        owned = own_segments(chunk, segments, last)
        text = " ".join(s.text.strip() for s in owned)
        adaptation: Optional[dict[str, Any]] = None
        if self.engine is not None and not last:
            adaptation = self.engine.update_with_heard(text, chunk.index)
        if self.acronyms is not None and text:
            learned = self.acronyms.learn(text, "defined in speech")
            if learned and self.engine is not None:
                self.engine.add_acronyms([{"acronym": a, "expansion": e, "occurrences": 1}
                                          for a, (e, _) in self.acronyms.learned.items()])
        self.outcome.chunk_log.append({
            **chunk.to_dict(), "segments": len(segments), "owned_segments": len(owned), "resumed": resumed,
            "attempts": attempts, "seconds": round(seconds, 1), "hotwords_applied": applied, "hotword_tokens": tokens,
            "context_update": adaptation,
        })
