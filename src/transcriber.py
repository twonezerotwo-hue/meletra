"""Model lifecycle + compute-type fallback around an :class:`ASRAdapter`."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional

from .model_manager import (
    ASRAdapter, ComputeTypeUnsupportedError, DecodeRequest, GpuOutOfMemoryError, ModelSpec, create_adapter,
)
from .schema import TranscriptionResult

log = logging.getLogger(__name__)

VRAM_ADVICE = (
    "Not enough free GPU memory. Close other GPU programs (for example `ollama stop <model>`; check "
    "`nvidia-smi`), then retry. If it still fails use --mode fast (int8_float16, ~2x smaller)."
)


def format_clock(seconds: float) -> str:
    seconds = int(max(0, seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


GpuStats = Callable[[], dict[str, Optional[float]]]


class ProgressReporter:
    """Log a progress line every ``step`` percent *or* every ``interval_s`` seconds, whichever is first.

    A line looks like::

        PROGRESS  37.2% | 32:31 / 1:27:21 | elapsed 4:51 | ETA 8:12 | RTF 0.149 | GPU dedicated 4259 MiB | shared 78 MiB

    ``start_offset`` is the part of the file already finished in an earlier run (resume), so RTF and
    ETA describe only the audio decoded in this run.
    """

    def __init__(
        self, step: float = 5.0, interval_s: float = 30.0, gpu_stats: Optional[GpuStats] = None,
        start_offset: float = 0.0, sticky: bool = False, on_update: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> None:
        self.on_update = on_update      # structured progress for the desktop app (called on every segment, throttled)
        self.chunk_index = 0
        self.chunk_count = 1
        self._last_event = 0.0
        self.step = step
        self.interval_s = interval_s
        self.gpu_stats = gpu_stats
        self.start_offset = start_offset
        self.sticky = sticky          # chunked runs: one clock for the whole file, not one per chunk
        self._clock_started = False
        self.reset()

    def reset(self, force: bool = False) -> None:
        """Restart the clock (called once the model is loaded so load time is not counted).

        A ``sticky`` reporter keeps its clock after the first start so RTF / ETA describe the whole file,
        not just the chunk being decoded.
        """
        if self.sticky and self._clock_started and not force:
            return
        self._clock_started = True
        self._next = self.step
        self._started = self._last_print = time.perf_counter()

    def line(self, processed: float, total: float) -> str:
        elapsed = time.perf_counter() - self._started
        done = max(processed - self.start_offset, 0.0)
        rtf = elapsed / done if done > 0 else 0.0
        eta = rtf * max(total - processed, 0.0)
        percent = 100.0 * processed / total if total else 0.0
        text = (f"PROGRESS {percent:5.1f}% | {format_clock(processed)} / {format_clock(total)} | "
                f"elapsed {format_clock(elapsed)} | ETA {format_clock(eta)} | RTF {rtf:.3f}")
        stats = self.gpu_stats() if self.gpu_stats else {}
        if stats and stats.get("dedicated_mib") is not None:
            text += f" | GPU dedicated {stats['dedicated_mib']:.0f} MiB | shared {(stats.get('shared_mib') or 0):.0f} MiB"
        return text

    def snapshot(self, processed: float, total: float) -> dict[str, Any]:
        """The numbers behind a PROGRESS line, as data."""
        elapsed = time.perf_counter() - self._started
        done = max(processed - self.start_offset, 0.0)
        rtf = elapsed / done if done > 0 else 0.0
        stats = (self.gpu_stats() if self.gpu_stats else {}) or {}
        return {"type": "chunk_progress", "percent": 100.0 * processed / total if total else 0.0, "processed": processed,
                "total": total, "elapsed": elapsed, "eta": rtf * max(total - processed, 0.0), "rtf": rtf,
                "chunk": self.chunk_index + 1, "chunks": self.chunk_count,
                "gpu_dedicated_mib": stats.get("dedicated_mib"), "gpu_shared_mib": stats.get("shared_mib")}

    def __call__(self, processed: float, total: float) -> None:
        processed = min(processed, total)
        percent = 100.0 * processed / total if total else 0.0
        now = time.perf_counter()
        if self.on_update is not None and now - self._last_event >= 0.5:
            self._last_event = now
            self.on_update(self.snapshot(processed, total))
        if percent < self._next and now - self._last_print < self.interval_s:
            return
        log.info(self.line(processed, total))
        self._last_print = now
        while self._next <= percent:
            self._next += self.step


class Transcriber:
    """Owns one adapter and applies the quality -> fallback compute-type policy.

    The policy is *sticky*: after float16 fails once, later files in a batch go straight to
    the fallback so the model is not reloaded and re-crashed for every file. Never falls back
    to CPU - a GPU failure surfaces as an error.
    """

    def __init__(
        self, spec: ModelSpec, profile: dict[str, Any], device: str = "cuda", device_index: int = 0,
        offline: bool = False,
    ) -> None:
        self.spec = spec
        self.profile = profile
        self.adapter: ASRAdapter = create_adapter(spec, device, device_index, offline)
        self.compute_types: list[str] = self._plan_compute_types()
        self._index = 0
        self._loaded = False
        self.events: Optional[Callable[..., None]] = None      # the desktop app listens here (model_loading / model_loaded)

    def _plan_compute_types(self) -> list[str]:
        primary = str(self.profile["compute_type"])
        fallback = self.profile.get("fallback_compute_type")
        if not self.adapter.capabilities.compute_types:
            return [primary]  # e.g. Transformers: dtype comes from models.yaml
        if self.adapter.device == "cpu":
            # float16 does not exist on CPU; int8 is the CPU type that keeps Whisper's accuracy at a usable speed
            log.info("CPU run: using compute type int8 instead of %s.", primary)
            return ["int8"]
        # GPU ladder: float16 (quality) -> int8_float16 -> int8; each step only if the one before did not fit / is unsupported
        ladder = [primary, str(fallback) if fallback else "", "int8_float16", "int8"]
        out: list[str] = []
        for item in ladder:
            if item and item not in out and (not out or item != "float16"):
                out.append(item)
        return out

    @property
    def current_compute_type(self) -> str:
        return self.compute_types[self._index]

    def unload(self) -> None:
        """Let go of the speech model (its GPU memory) while another model needs the computer; the next decode loads it again."""
        try:
            self.adapter.close()
        finally:
            self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        compute_type = self.compute_types[self._index]
        started = time.perf_counter()
        if self.events is not None:
            self.events("model_loading", model=str(self.spec.model), key=self.spec.key, device=self.adapter.device,
                        compute_type=compute_type)
        self.adapter.load(compute_type)
        self._loaded = True
        if self.events is not None:
            self.events("model_loaded", seconds=round(time.perf_counter() - started, 2), device=self.adapter.device,
                        compute_type=compute_type)

    def detect_languages(self, wav: Any, starts: list[float], seconds: float = 30.0,
                         progress: Optional[Callable[[int, int], None]] = None,
                         should_stop: Optional[Callable[[], bool]] = None) -> list[tuple[float, str, float]]:
        """Whisper's language detection on several windows of the recording: ``[(start, language, probability), ...]``.

        Empty when the backend cannot detect languages or the model cannot be loaded yet (the main decode then reports the
        real problem through its own fallback ladder)."""
        detector = getattr(self.adapter, "detect_language_at", None)
        if detector is None or not starts:
            return []
        try:
            self._ensure_loaded()
        except Exception as exc:  # noqa: BLE001 - advisory only
            log.warning("Language sampling skipped: the model could not be loaded here (%s: %s)", type(exc).__name__,
                        str(exc).splitlines()[0][:120] if str(exc) else "")
            return []
        found: list[tuple[float, str, float]] = []
        for done, start in enumerate(starts):
            if should_stop is not None and should_stop():
                from .chunked_decode import PauseRequested
                raise PauseRequested()
            if progress is not None:
                progress(done, len(starts))
            try:
                language, probability = detector(wav, start, seconds)
            except Exception:  # noqa: BLE001 - one unreadable window must not lose the others
                log.debug("language detection failed at %.0f s", start, exc_info=True)
                continue
            found.append((float(start), language, probability))
        return found

    def detect_language_windows(self, wav: Any, starts: list[float], seconds: float = 30.0, progress: Optional[Callable[[int, int], None]] = None,
                                should_stop: Optional[Callable[[], bool]] = None) -> list[tuple[float, dict[str, float]]]:
        """The detector's top languages with their probabilities for each window: ``[(start, {language: probability}), ...]``. Empty when the backend cannot do it."""
        detector = getattr(self.adapter, "detect_language_probs", None)
        if detector is None or not starts:
            return []
        try:
            self._ensure_loaded()
        except Exception as exc:  # noqa: BLE001 - advisory only: the main decode reports the real problem through its own fallback ladder
            log.warning("Language scan skipped: the model could not be loaded here (%s)", type(exc).__name__)
            return []
        found: list[tuple[float, dict[str, float]]] = []
        for done, start in enumerate(starts):
            if should_stop is not None and should_stop():
                from .chunked_decode import PauseRequested
                raise PauseRequested()
            if progress is not None:
                progress(done, len(starts))
            try:
                found.append((float(start), dict(detector(wav, start, seconds))))
            except Exception:  # noqa: BLE001 - one unreadable window must not lose the others
                log.debug("language detection failed at %.0f s", start, exc_info=True)
        return found

    def score_languages(self, wav: Any, start: float, seconds: float, languages: list[str], max_tokens: int = 32) -> dict[str, dict[str, Any]]:
        """Likelihood of each candidate language on one window (see ``FasterWhisperAdapter.score_languages``); empty when the backend cannot."""
        scorer = getattr(self.adapter, "score_languages", None)
        if scorer is None or not languages:
            return {}
        self._ensure_loaded()
        return scorer(wav, start, seconds, languages, max_tokens)

    def transcribe(self, request: DecodeRequest) -> TranscriptionResult:
        if request.language_plan:                                      # Auto Detect over time: decode run by run, each in its own language
            from .multilingual.decode import transcribe_runs

            return transcribe_runs(self.transcribe, request, getattr(self, "language_runs_meta", None))
        started_index = self._index
        load_seconds = 0.0
        while True:
            compute_type = self.compute_types[self._index]
            try:
                if not self._loaded:
                    load_started = time.perf_counter()
                    if self.events is not None:
                        self.events("model_loading", model=str(self.spec.model), key=self.spec.key,
                                    device=self.adapter.device, compute_type=compute_type)
                    self.adapter.load(compute_type)
                    load_seconds += time.perf_counter() - load_started
                    self._loaded = True
                    if self.events is not None:
                        self.events("model_loaded", seconds=round(load_seconds, 2), device=self.adapter.device,
                                    compute_type=compute_type)
                if isinstance(request.progress, ProgressReporter):
                    request.progress.reset()
                result = self.adapter.transcribe(request)
            except (GpuOutOfMemoryError, ComputeTypeUnsupportedError) as exc:
                self.adapter.close()
                self._loaded = False
                reason = "out of GPU memory" if isinstance(exc, GpuOutOfMemoryError) else "compute type unsupported"
                if self._index + 1 >= len(self.compute_types):
                    raise GpuOutOfMemoryError(f"{reason} with {compute_type}. {VRAM_ADVICE}\nDetail: {exc}") from exc
                self._index += 1
                log.warning("FALLBACK: %s with compute_type=%s; retrying on the GPU with %s. (CPU is never used.)",
                            reason, compute_type, self.compute_types[self._index])
                if request.on_restart is not None:
                    request.on_restart()        # e.g. drop the checkpoint segments of the failed attempt
                if isinstance(request.progress, ProgressReporter):
                    request.progress.reset(force=True)
                continue
            result.compute_type = compute_type
            result.fallback_used = self._index > 0
            result.load_seconds = load_seconds
            if self._index != started_index:
                result.notes.append(
                    f"compute_type fell back from {self.compute_types[started_index]} to {compute_type}")
            return result

    def close(self) -> None:
        self.adapter.close()
        self._loaded = False
