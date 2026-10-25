"""The macOS-optimized speech backend: Whisper on Apple's MLX (Metal GPU / unified memory) through the ``mlx-whisper`` package.

It implements the same ``ASRAdapter`` interface as the faster-whisper backend and returns the SAME result objects (``Segment`` /
``Word`` / ``TranscriptionResult``), so everything downstream - cleaning, medical analysis, SRT / VTT / JSON, raw and clean
transcripts - is identical on both platforms. What differs is only what the engine itself can do:

* decoding is greedy with temperature fallback (``mlx-whisper`` has no beam search), so a few words may differ from the Windows
  output; this is stated in ``TranscriptionResult.notes``;
* speech windows come from the same Silero VAD as the Transformers backend (<= 29 s each), because silence decoded without a
  VAD makes Whisper hallucinate;
* per-window language switching (``multilingual``) is not offered; language sampling over the whole file still works through
  ``detect_language_at`` (when the installed ``mlx-whisper`` exposes it - otherwise the pipeline falls back to one language).

Not verified on real hardware in this repository's CI (it runs on Windows): the module is exercised with a fake ``mlx_whisper``
package; the first real run happens on a Mac (``scripts/benchmark_asr_backend.py`` is the check to run there).
"""

from __future__ import annotations

import gc
import logging
import time
from typing import Any, Optional

from ..model_manager import (MAX_HOTWORD_TOKENS, ASRAdapter, Capabilities, DecodeRequest, GpuOutOfMemoryError, TransformersAdapter,
                             read_wav_slice)
from ..platform import current
from ..schema import Segment, TranscriptionResult, Word
from ..vocabulary import fit_token_budget

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
WINDOW_SECONDS = 29.0


class BackendUnavailable(RuntimeError):
    """The optimized macOS engine cannot run here; the message says why and what to do."""


def _looks_like_metal_oom(message: str) -> bool:
    lowered = message.lower()
    return any(tag in lowered for tag in ("insufficient memory", "outofmemory", "out of memory", "kiogpucommandbuffercallbackerroroutofmemory"))


class MlxWhisperBackend(ASRAdapter):
    backend_name = "mlx-whisper"
    capabilities = Capabilities(hotwords=True, word_timestamps=True, vad=True, segment_stats=True, language_detection=True,
                                multilingual=False, compute_types=False)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._mlx: Any = None                    # the imported mlx_whisper module
        self._path: str = ""
        self._model_info: dict[str, Any] = {}

    # ---------------------------------------------------------------------------------------------------------------------- load
    def load(self, compute_type: str) -> None:
        if not current().is_apple_silicon:
            raise BackendUnavailable("The optimized speech engine needs a Mac with Apple Silicon (M1 or newer). On this computer the "
                                     "standard engine is used (choose the CTranslate2 Large-v3 model).")
        try:
            import mlx_whisper                    # type: ignore[import-not-found]
        except ImportError as exc:
            raise BackendUnavailable("The optimized macOS speech engine (mlx-whisper) is not installed in this build.") from exc
        self.close()
        self._mlx = mlx_whisper
        self._path = str(self.spec.model)
        log.info("Loading %s (%s, Metal) ...", self._path, self.backend_name)
        started = time.perf_counter()
        try:                                       # a one-second silent decode loads the weights into mlx-whisper's model cache
            self._mlx.transcribe(self._silence(1.0), path_or_hf_repo=self._path, verbose=None, language="en", fp16=True)
        except Exception as exc:                   # noqa: BLE001 - translate the common failures, re-raise the rest
            if _looks_like_metal_oom(str(exc)):
                raise GpuOutOfMemoryError(str(exc)) from exc
            raise
        self.compute_type = "float16"
        log.info("Model loaded in %.1fs", time.perf_counter() - started)

    @staticmethod
    def _silence(seconds: float) -> Any:
        import numpy as np

        return np.zeros(int(seconds * SAMPLE_RATE), dtype="float32")

    # ---------------------------------------------------------------------------------------------------------- language detection
    def detect_language_at(self, wav: Any, start: float, seconds: float = 30.0) -> tuple[str, float]:
        """Whisper's language detection on ``seconds`` of audio from ``start``: ``(code, probability)``."""
        if self._mlx is None:
            raise RuntimeError("Model is not loaded; call load() first")
        return self._detect(read_wav_slice(wav, start, seconds))

    def _detect(self, audio: Any) -> tuple[str, float]:
        import mlx.core as mx                      # type: ignore[import-not-found]
        from mlx_whisper.audio import N_FRAMES, N_SAMPLES, log_mel_spectrogram, pad_or_trim      # type: ignore[import-not-found]
        from mlx_whisper.transcribe import ModelHolder                                           # type: ignore[import-not-found]

        model = ModelHolder.get_model(self._path, mx.float16)
        mel = log_mel_spectrogram(audio, n_mels=model.dims.n_mels, padding=N_SAMPLES)
        segment = pad_or_trim(mel, N_FRAMES, axis=-2).astype(mx.float16)
        _tokens, probabilities = model.detect_language(segment)
        language = max(probabilities, key=probabilities.get)
        return str(language), float(probabilities[language])

    # ------------------------------------------------------------------------------------------------------------------- hotwords
    def _count_tokens(self, text: str) -> int:
        try:
            from mlx_whisper.tokenizer import get_tokenizer      # type: ignore[import-not-found]

            vocab = int(self._model_info.get("n_vocab", 51866))
            tokenizer = get_tokenizer(True, num_languages=100 if vocab >= 51866 else 99, language=None, task="transcribe")
            return len(tokenizer.encode(" " + text.strip()))
        except Exception:                                         # noqa: BLE001 - a cautious character estimate keeps the budget safe
            return max(1, int(len(text) / 2.5))

    def _fit_hotwords(self, request: DecodeRequest) -> tuple[list[str], Optional[int]]:
        if not request.hotwords:
            return [], None
        budget = min(request.hotword_token_budget if request.hotword_token_budget is not None else MAX_HOTWORD_TOKENS, MAX_HOTWORD_TOKENS)
        kept, used = fit_token_budget(request.hotwords, self._count_tokens, budget)
        return kept, used

    # ------------------------------------------------------------------------------------------------------------------ transcribe
    def transcribe(self, request: DecodeRequest) -> TranscriptionResult:
        if self._mlx is None:
            raise RuntimeError("Model is not loaded; call load() first")
        profile, decoding = request.profile, request.decoding
        offset = float(request.start_offset)
        audio = read_wav_slice(request.audio_path, request.start_offset, request.max_seconds)
        duration = len(audio) / SAMPLE_RATE
        total = request.total_duration or (offset + duration)

        applied, hotword_tokens = self._fit_hotwords(request)
        options: dict[str, Any] = {
            "path_or_hf_repo": self._path, "verbose": None, "task": request.task if request.task in ("transcribe", "translate") else "transcribe", "fp16": True,
            "temperature": tuple(profile["temperature"]) if isinstance(profile["temperature"], (list, tuple)) else (float(profile["temperature"]),),
            "condition_on_previous_text": False,                   # windows are decoded one by one; nothing to condition on
            "word_timestamps": bool(profile["word_timestamps"]),
        }
        if applied:
            options["initial_prompt"] = ", ".join(applied)
        for ours, theirs in (("compression_ratio_threshold", "compression_ratio_threshold"), ("log_prob_threshold", "logprob_threshold"),
                             ("no_speech_threshold", "no_speech_threshold")):
            if decoding.get(ours) is not None:
                options[theirs] = decoding[ours]
        if decoding.get("hallucination_silence_threshold") is not None and options["word_timestamps"]:
            options["hallucination_silence_threshold"] = decoding["hallucination_silence_threshold"]

        use_vad = bool(profile.get("vad_filter"))
        windows = (TransformersAdapter.speech_windows(audio, SAMPLE_RATE, request.vad_parameters, WINDOW_SECONDS) if use_vad
                   else [(0, len(audio))])
        language = request.language
        probability: Optional[float] = None
        if language is None and windows:
            try:
                language, probability = self._detect(audio[windows[0][0]:windows[0][1]])
            except Exception:                                      # noqa: BLE001 - mlx-whisper then detects on every window itself
                log.debug("language detection through mlx-whisper internals failed", exc_info=True)
        if language:
            options["language"] = language

        started = time.perf_counter()
        segments: list[Segment] = []
        for first, last in windows:
            window = audio[first:last]
            base = offset + first / SAMPLE_RATE
            window_seconds = len(window) / SAMPLE_RATE
            try:
                result = self._mlx.transcribe(window, **options)
            except Exception as exc:                               # noqa: BLE001
                if _looks_like_metal_oom(str(exc)):
                    raise GpuOutOfMemoryError(str(exc)) from exc
                raise
            if language is None:
                language = result.get("language")
            for raw in result.get("segments", []):
                segment = self._segment(raw, base, window_seconds, language)
                if segment is None:
                    continue
                segments.append(segment)
                if request.on_segment is not None:
                    request.on_segment(segment)
            if request.progress is not None:
                request.progress(base + window_seconds, total)
        log.info("Decoded %d segments in %.1fs", len(segments), time.perf_counter() - started)
        return TranscriptionResult(
            segments=segments, language=language, language_probability=probability, duration=offset + duration,
            model=str(self.spec.model), backend=self.backend_name, compute_type=self.compute_type,
            capabilities_used={"hotwords": bool(applied), "word_timestamps": options["word_timestamps"], "vad": use_vad,
                               "multilingual": False},
            notes=["mlx-whisper: greedy decoding with temperature fallback (no beam search), speech windows from Silero VAD; "
                   "the text can differ slightly from the faster-whisper output of the Windows version."],
            hotwords_applied=applied, hotword_tokens=hotword_tokens,
        )

    def _segment(self, raw: dict[str, Any], base: float, window_seconds: float, language: Optional[str]) -> Optional[Segment]:
        """One mlx-whisper segment (a dict like OpenAI's Whisper) -> the shared ``Segment``; timestamps made absolute and clamped."""
        text = str(raw.get("text", ""))
        if not text.strip():
            return None
        start = min(max(float(raw.get("start", 0.0)), 0.0), window_seconds) + base
        end = min(max(float(raw.get("end", window_seconds)), 0.0), window_seconds) + base
        words = [Word(min(max(float(w["start"]), 0.0), window_seconds) + base, min(max(float(w["end"]), 0.0), window_seconds) + base,
                      str(w["word"]), None if w.get("probability") is None else float(w["probability"]))
                 for w in raw.get("words", []) or [] if "start" in w and "end" in w and "word" in w]
        return Segment(
            start=start, end=max(end, start), text=text, language=language, model=str(self.spec.model), backend=self.backend_name,
            avg_logprob=_optional_float(raw.get("avg_logprob")), no_speech_prob=_optional_float(raw.get("no_speech_prob")),
            compression_ratio=_optional_float(raw.get("compression_ratio")), words=words)

    def close(self) -> None:
        if self._mlx is not None:
            self._mlx = None
            gc.collect()
            try:
                import mlx.core as mx              # type: ignore[import-not-found]

                clear = getattr(mx, "clear_cache", None) or getattr(getattr(mx, "metal", None), "clear_cache", None)
                if clear is not None:
                    clear()
            except Exception:                      # noqa: BLE001 - freeing the cache is a courtesy
                pass


def _optional_float(value: Any) -> Optional[float]:
    return None if value is None else float(value)
