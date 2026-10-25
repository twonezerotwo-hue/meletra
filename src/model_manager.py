"""Model registry and ASR backend adapters.

Adapters expose a *capabilities* record so the pipeline can gracefully disable features a
backend cannot do (hotwords, word timestamps, VAD, decode statistics) instead of failing.
Third-party Hugging Face checkpoints are never assumed to be faster-whisper compatible: they
run through the Transformers pipeline, or through an explicit CTranslate2 conversion.
"""

from __future__ import annotations

import gc
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from .config import load_models_config
from .diagnostics import CudaUnavailableError, require_cuda
from .platform_utils import register_cuda_dll_directories
from .schema import Segment, TranscriptionResult, Word
from .vocabulary import fit_token_budget

log = logging.getLogger(__name__)

# faster-whisper truncates the hotword prompt to max_length // 2 - 1 tokens (max_length = 448).
MAX_HOTWORD_TOKENS = 223

ProgressCallback = Callable[[float, float], None]  # (seconds processed, total seconds)


class ModelNotConfiguredError(RuntimeError):
    """A 'configurable' model entry has no model set."""


class UnknownModelError(KeyError):
    """The requested model key is not in config/models.yaml."""


class GpuOutOfMemoryError(RuntimeError):
    """CUDA ran out of memory while loading or running a model."""


class ComputeTypeUnsupportedError(RuntimeError):
    """The GPU / backend rejected the requested compute type."""


@dataclass(frozen=True)
class ModelSpec:
    key: str
    engine: str
    model: Optional[str]
    description: str = ""
    default_language: Optional[str] = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Capabilities:
    hotwords: bool = False           # biases decoding with a term list
    word_timestamps: bool = False
    vad: bool = False
    segment_stats: bool = False      # avg_logprob / no_speech_prob / compression_ratio per segment
    language_detection: bool = False
    multilingual: bool = False       # per-window language re-detection
    compute_types: bool = False      # honours float16 / int8_float16 selection


@dataclass
class DecodeRequest:
    """Everything an adapter needs for one file (already validated by the pipeline)."""

    audio_path: Path
    language: Optional[str]
    profile: dict[str, Any]
    decoding: dict[str, Any]
    vad_parameters: dict[str, Any]
    hotwords: list[str] = field(default_factory=list)       # priority-ordered candidates
    multilingual: bool = False
    progress: Optional[ProgressCallback] = None
    # Whisper-token budget for the hotword string. Whole terms only; None = no budget (old behaviour).
    hotword_token_budget: Optional[int] = None
    # Decode only audio from ``start_offset`` seconds on (resume) and, if set, only ``max_seconds`` of it
    # (topic probe). Segment/word timestamps are returned ABSOLUTE (offset already added back).
    start_offset: float = 0.0
    max_seconds: Optional[float] = None
    total_duration: Optional[float] = None                  # whole-file length, for progress
    on_segment: Optional[Callable[[Segment], None]] = None  # called with every finished segment (checkpoint)
    on_restart: Optional[Callable[[], None]] = None         # called before a decode attempt is repeated
    task: str = "transcribe"                                # "translate" = Whisper writes English whatever language is spoken (Output Language English)
    # Auto Detect over time: ``(start, end, language, vad)`` stretches (absolute seconds) that are decoded one after the other, each in its own language
    # (``vad`` is "on" or "off": off = decode the whole stretch, for singing and speech over music). Set => ``language`` / ``multilingual`` are ignored.
    language_plan: Optional[list[tuple[float, float, str, str]]] = None
    vad_override: Optional[bool] = None                     # False = decode without the voice-activity filter whatever the profile says


def load_model_specs(path: Optional[Path] = None) -> dict[str, ModelSpec]:
    raw = load_models_config(path)
    return {
        key: ModelSpec(
            key=key, engine=entry["engine"], model=entry.get("model"),
            description=entry.get("description", ""), default_language=entry.get("default_language"),
            options=dict(entry.get("options") or {}),
        )
        for key, entry in raw.items()
    }


def get_model_spec(key: str, specs: Optional[dict[str, ModelSpec]] = None) -> ModelSpec:
    specs = specs or load_model_specs()
    if key not in specs:
        raise UnknownModelError(f"Unknown model '{key}'. Available: {', '.join(specs)}")
    return specs[key]


def _looks_like_oom(message: str) -> bool:
    lowered = message.lower()
    return "out of memory" in lowered or "cudamalloc" in lowered or "cuda failed with error out of memory" in lowered


def _looks_like_missing_cuda_lib(message: str) -> bool:
    lowered = message.lower()
    return any(tag in lowered for tag in ("cublas", "cudnn", "cudart", "cannot be loaded", "not found or cannot"))


def read_wav_slice(path: Path, start: float, seconds: Optional[float], rate: int = 16000):
    """``seconds`` of audio from ``start`` as mono float32, reading only that slice of a 16 kHz WAV (the pipeline's own format).

    The whole-file decode this replaces cost 330 MB of memory and a few seconds for every 8-30 s re-decode of an 87 minute lecture.
    Other formats fall back to the full decode."""
    try:
        import soundfile as sf

        info = sf.info(str(path))
        if info.samplerate == rate:
            frames = -1 if not seconds else int(seconds * rate)
            data, _ = sf.read(str(path), start=int(max(start, 0.0) * rate), frames=frames, dtype="float32", always_2d=True)
            return data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    except Exception:  # noqa: BLE001 - the full decode below is always correct
        log.debug("slice read failed, decoding the whole file", exc_info=True)
    from faster_whisper.audio import decode_audio

    audio = decode_audio(str(path), sampling_rate=rate)
    first = int(max(start, 0.0) * rate)
    return audio[first:first + int(seconds * rate)] if seconds else audio[first:]


class ASRAdapter(ABC):
    """Common interface of every ASR backend."""

    backend_name: str = "unknown"
    capabilities: Capabilities = Capabilities()

    def __init__(self, spec: ModelSpec, device: str = "cuda", device_index: int = 0, offline: bool = False) -> None:
        self.spec = spec
        self.device = device
        self.device_index = device_index
        self.offline = offline
        self.compute_type: Optional[str] = None

    @abstractmethod
    def load(self, compute_type: str) -> None: ...

    @abstractmethod
    def transcribe(self, request: DecodeRequest) -> TranscriptionResult: ...

    @abstractmethod
    def close(self) -> None: ...

    def _require_gpu(self) -> None:
        if self.device == "cuda":
            require_cuda(self.device_index)
        else:
            log.warning("Running on the %s: this is MUCH slower than an NVIDIA GPU (Large-v3 can take several times "
                        "the length of the recording).", self.device.upper())


class FasterWhisperAdapter(ASRAdapter):
    """faster-whisper / CTranslate2 backend (primary)."""

    backend_name = "faster-whisper"
    capabilities = Capabilities(
        hotwords=True, word_timestamps=True, vad=True, segment_stats=True,
        language_detection=True, multilingual=True, compute_types=True,
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._model: Any = None

    def load(self, compute_type: str) -> None:
        self._require_gpu()
        register_cuda_dll_directories()
        from faster_whisper import WhisperModel

        self.close()
        log.info("Loading %s (%s) on %s with compute_type=%s ...",
                 self.spec.model, self.backend_name, self.device, compute_type)
        started = time.perf_counter()
        try:
            extra: dict[str, Any] = {}
            threads = int(self.spec.options.get("cpu_threads") or 0)
            if self.device == "cpu" and threads > 0:
                extra["cpu_threads"] = threads
            self._model = WhisperModel(
                str(self.spec.model), device=self.device, device_index=self.device_index,
                compute_type=compute_type, local_files_only=self.offline, **extra,
            )
        except ValueError as exc:
            if "compute type" in str(exc).lower():
                raise ComputeTypeUnsupportedError(str(exc)) from exc
            raise
        except RuntimeError as exc:
            self._translate_runtime_error(exc)
            raise
        self.compute_type = compute_type
        log.info("Model loaded in %.1fs", time.perf_counter() - started)

    @staticmethod
    def _translate_runtime_error(exc: RuntimeError) -> None:
        message = str(exc)
        if _looks_like_oom(message):
            raise GpuOutOfMemoryError(message) from exc
        if _looks_like_missing_cuda_lib(message):
            raise CudaUnavailableError(
                f"CUDA runtime library problem: {message}\n"
                "cuBLAS 12 / cuDNN 9 DLLs must be reachable. Install the cu126 PyTorch wheel "
                "(it bundles them) or `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12`."
            ) from exc

    def transcribe(self, request: DecodeRequest) -> TranscriptionResult:
        if self._model is None:
            raise RuntimeError("Model is not loaded; call load() first")
        profile, decoding = request.profile, request.decoding
        kwargs: dict[str, Any] = dict(
            language=request.language,
            task=request.task if request.task in ("transcribe", "translate") else "transcribe",
            beam_size=int(profile["beam_size"]),
            temperature=profile["temperature"],
            condition_on_previous_text=bool(profile["condition_on_previous_text"]),
            word_timestamps=bool(profile["word_timestamps"]),
            repetition_penalty=float(profile.get("repetition_penalty", 1.0)),
            no_repeat_ngram_size=int(profile.get("no_repeat_ngram_size", 0)),
            vad_filter=bool(profile["vad_filter"]) and request.vad_override is not False,
            multilingual=request.multilingual,
        )
        if kwargs["vad_filter"]:
            kwargs["vad_parameters"] = dict(request.vad_parameters)
        for key in ("compression_ratio_threshold", "log_prob_threshold", "no_speech_threshold",
                    "hallucination_silence_threshold"):
            if key in decoding:
                kwargs[key] = decoding[key]
        if kwargs.get("hallucination_silence_threshold") is not None and not kwargs["word_timestamps"]:
            kwargs["hallucination_silence_threshold"] = None  # needs word timestamps
        applied, hotword_tokens = self._fit_hotwords(request)
        hotwords_used = bool(applied)
        if applied:
            kwargs["hotwords"] = ", ".join(applied)
        audio_input, offset = self._audio_input(request)

        started = time.perf_counter()
        try:
            segment_iter, info = self._model.transcribe(audio_input, **kwargs)
            segments = self._collect(segment_iter, info, request, offset)
        except RuntimeError as exc:
            self._translate_runtime_error(exc)
            raise
        log.info("Decoded %d segments in %.1fs", len(segments), time.perf_counter() - started)
        return TranscriptionResult(
            segments=segments, language=info.language, language_probability=info.language_probability,
            duration=offset + float(info.duration), model=str(self.spec.model), backend=self.backend_name,
            compute_type=self.compute_type,
            capabilities_used={
                "hotwords": hotwords_used, "word_timestamps": kwargs["word_timestamps"],
                "vad": kwargs["vad_filter"], "multilingual": request.multilingual,
            },
            hotwords_applied=applied, hotword_tokens=hotword_tokens,
        )

    def _count_tokens(self, text: str) -> int:
        """Whisper tokens of ``text`` exactly as faster-whisper encodes a hotword string."""
        return len(self._model.hf_tokenizer.encode(" " + text.strip(), add_special_tokens=False).ids)

    def _fit_hotwords(self, request: DecodeRequest) -> tuple[list[str], Optional[int]]:
        """Apply the token budget: keep whole terms in priority order (never a half-cut word)."""
        if not request.hotwords:
            return [], None
        budget = request.hotword_token_budget
        if budget is None:
            return list(request.hotwords), self._count_tokens(", ".join(request.hotwords))
        if budget > MAX_HOTWORD_TOKENS:
            log.warning("hotword_token_budget %d exceeds faster-whisper's own cap of %d; using the cap",
                        budget, MAX_HOTWORD_TOKENS)
            budget = MAX_HOTWORD_TOKENS
        kept, used = fit_token_budget(request.hotwords, self._count_tokens, budget)
        dropped = len(request.hotwords) - len(kept)
        log.info("Hotwords: %d of %d terms fit the %d-token budget (%d tokens used%s)",
                 len(kept), len(request.hotwords), budget, used, f", {dropped} lower-priority dropped" if dropped else "")
        return kept, used

    @staticmethod
    def _audio_input(request: DecodeRequest) -> tuple[Any, float]:
        """The file path, or a sliced waveform when resuming (start_offset) or probing (max_seconds)."""
        if request.start_offset <= 0 and not request.max_seconds:
            return str(request.audio_path), 0.0
        return read_wav_slice(request.audio_path, request.start_offset, request.max_seconds), float(request.start_offset)

    def detect_language_at(self, wav: Path, start: float, seconds: float = 30.0) -> tuple[str, float]:
        """Whisper's own language detection on ``seconds`` of audio from ``start``: ``(code, probability)``."""
        if self._model is None:
            raise RuntimeError("Model is not loaded; call load() first")
        language, probability, _all = self._model.detect_language(read_wav_slice(wav, start, seconds))
        return str(language), float(probability)

    def detect_language_probs(self, wav: Path, start: float, seconds: float = 30.0, top: int = 5) -> list[tuple[str, float]]:
        """Whisper's language detection on ``seconds`` of audio from ``start``: the detector's own top ``top`` ``(code, probability)`` pairs, best first."""
        if self._model is None:
            raise RuntimeError("Model is not loaded; call load() first")
        _language, _probability, all_probs = self._model.detect_language(read_wav_slice(wav, start, seconds))
        ranked = sorted(((str(code), float(p)) for code, p in all_probs), key=lambda item: -item[1])
        return ranked[:top]

    def score_languages(self, wav: Path, start: float, seconds: float, languages: Sequence[str], max_tokens: int = 32) -> dict[str, dict[str, Any]]:
        """How well each candidate language explains ONE window of audio: the window is encoded once, then the decoder writes its first ``max_tokens`` tokens forced into each
        language; the average log-probability of what it wrote (and the text) is the evidence. A language the audio is not in tends to give a lower likelihood, a shorter text or a
        loop. This is a measurement of the model's own likelihood, not a detector: the caller compares languages with each other and keeps the uncertainty."""
        if self._model is None:
            raise RuntimeError("Model is not loaded; call load() first")
        from faster_whisper.audio import pad_or_trim
        from faster_whisper.tokenizer import Tokenizer

        model = self._model
        features = model.feature_extractor(read_wav_slice(wav, start, seconds))
        encoded = model.encode(pad_or_trim(features[..., : model.feature_extractor.nb_max_frames]))
        out: dict[str, dict[str, Any]] = {}
        for language in languages:
            tokenizer = Tokenizer(model.hf_tokenizer, model.model.is_multilingual, task="transcribe", language=language)
            prompt = [*tokenizer.sot_sequence, tokenizer.no_timestamps]
            result = model.model.generate(encoded, [prompt], beam_size=1, length_penalty=1.0, max_length=len(prompt) + max_tokens, return_scores=True,
                                          return_no_speech_prob=True, suppress_blank=True, suppress_tokens=[-1])[0]
            tokens = [t for t in result.sequences_ids[0] if t < tokenizer.eot]
            out[language] = {"avg_logprob": float(result.scores[0]), "no_speech_prob": float(getattr(result, "no_speech_prob", 0.0) or 0.0),
                             "text": tokenizer.decode(tokens).strip(), "tokens": len(tokens)}
        return out

    def _collect(self, segment_iter: Any, info: Any, request: DecodeRequest, offset: float = 0.0) -> list[Segment]:
        segments: list[Segment] = []
        slice_total = float(info.duration) or 1.0
        full_total = request.total_duration or (offset + slice_total)
        for raw in segment_iter:
            # Whisper can emit timestamps beyond the audio when it hallucinates; never export those.
            start = min(float(raw.start), slice_total) + offset
            end = min(float(raw.end), slice_total) + offset
            words = [
                Word(min(w.start, slice_total) + offset, min(w.end, slice_total) + offset, w.word, w.probability)
                for w in (raw.words or [])
            ]
            segment = Segment(
                start=start, end=end, text=raw.text, language=info.language,
                model=str(self.spec.model), backend=self.backend_name,
                avg_logprob=float(raw.avg_logprob), no_speech_prob=float(raw.no_speech_prob),
                compression_ratio=float(raw.compression_ratio), words=words,
            )
            segments.append(segment)
            if request.on_segment is not None:
                request.on_segment(segment)
            if request.progress is not None:
                request.progress(end, full_total)
        return segments

    def close(self) -> None:
        if self._model is not None:
            self._model = None
            gc.collect()


class TransformersAdapter(ASRAdapter):
    """Hugging Face Transformers ``automatic-speech-recognition`` pipeline backend.

    Works for any Whisper checkpoint in Transformers format (e.g. third-party medical
    fine-tunes). It exposes no per-segment decoding statistics. The pipeline has no VAD of its
    own, so speech regions are found with the Silero VAD that ships with faster-whisper and only
    those windows are decoded (silence/music otherwise triggers hallucination loops such as
    "abone ol abone ol ...", measured on real Turkish speech).
    """

    backend_name = "transformers"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._pipe: Any = None
        self._torch: Any = None
        # Prompt biasing (prompt_ids) makes Whisper stop emitting timestamp tokens in our tests
        # (one giant segment), so it is opt-in per model via options.prompt_hotwords.
        self.capabilities = Capabilities(
            hotwords=bool(self.spec.options.get("prompt_hotwords", False)), word_timestamps=False, vad=True,
            segment_stats=False, language_detection=True, multilingual=False, compute_types=False,
        )

    def load(self, compute_type: str) -> None:
        self._require_gpu()
        register_cuda_dll_directories()
        import torch
        from transformers import pipeline

        self.close()
        self._torch = torch
        dtype_name = self.spec.options.get("dtype", "float16")
        dtype = getattr(torch, dtype_name)
        target = f"cuda:{self.device_index}" if self.device == "cuda" else "cpu"
        log.info("Loading %s (transformers, %s) on %s ...", self.spec.model, dtype_name, target)
        started = time.perf_counter()
        try:
            self._pipe = pipeline(
                "automatic-speech-recognition", model=str(self.spec.model), dtype=dtype, device=target,
                model_kwargs={"local_files_only": True} if self.offline else None,
            )
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            if _looks_like_oom(str(exc)) or isinstance(exc, torch.cuda.OutOfMemoryError):
                raise GpuOutOfMemoryError(str(exc)) from exc
            raise
        self.compute_type = dtype_name
        log.info("Model loaded in %.1fs", time.perf_counter() - started)

    def transcribe(self, request: DecodeRequest) -> TranscriptionResult:
        if self._pipe is None:
            raise RuntimeError("Model is not loaded; call load() first")
        import numpy as np
        import soundfile as sf

        audio, rate = sf.read(str(request.audio_path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        full_duration = len(audio) / float(rate)
        base = int(request.start_offset * rate)                       # resume / probe: decode a slice,
        stop = base + int(request.max_seconds * rate) if request.max_seconds else len(audio)
        audio = audio[base:stop]                                      # timestamps get `base_offset` added back
        base_offset = base / float(rate)
        duration = len(audio) / float(rate)
        progress_total = request.total_duration or full_duration
        opts = self.spec.options
        generate_kwargs: dict[str, Any] = {
            "task": "transcribe",
            "num_beams": int(opts.get("num_beams", request.profile["beam_size"])),
        }
        if request.language:
            generate_kwargs["language"] = request.language
        hotwords_used = False
        applied: list[str] = []
        if request.hotwords:
            try:
                applied = list(request.hotwords)
                if request.hotword_token_budget is not None:
                    applied, _ = fit_token_budget(
                        applied, lambda s: len(self._pipe.tokenizer.encode(" " + s, add_special_tokens=False)),
                        min(request.hotword_token_budget, MAX_HOTWORD_TOKENS))
                ids = self._pipe.tokenizer.get_prompt_ids(", ".join(applied), return_tensors="pt")
                generate_kwargs["prompt_ids"] = ids.to(self._pipe.device)
                hotwords_used = bool(applied)
            except Exception as exc:  # noqa: BLE001 - prompt support varies; never fail the run
                applied = []
                log.warning("Prompt/hotword context disabled for this backend: %s", exc)

        use_vad = bool(request.profile.get("vad_filter"))
        windows = self.speech_windows(audio, rate, request.vad_parameters) if use_vad else [(0, len(audio))]
        if use_vad:
            speech_s = sum(e - s for s, e in windows) / rate
            log.info("VAD kept %d speech window(s), %.1fs of %.1fs", len(windows), speech_s, duration)
        batch_size = max(1, int(opts.get("batch_size", 1)))

        started = time.perf_counter()
        segments: list[Segment] = []
        for first in range(0, len(windows), batch_size if use_vad else 1):
            batch = windows[first:first + (batch_size if use_vad else 1)]
            try:
                outputs = self._run_batch(audio, rate, batch, use_vad, opts, generate_kwargs)
            except Exception as exc:  # noqa: BLE001
                if hotwords_used:
                    log.warning("Decoding with hotword prompt failed (%s); retrying without it.", exc)
                    generate_kwargs.pop("prompt_ids", None)
                    hotwords_used = False
                    outputs = self._run_batch(audio, rate, batch, use_vad, opts, generate_kwargs)
                elif _looks_like_oom(str(exc)) or isinstance(exc, self._torch.cuda.OutOfMemoryError):
                    raise GpuOutOfMemoryError(str(exc)) from exc
                else:
                    raise
            for (start_sample, end_sample), output in zip(batch, outputs):
                offset = start_sample / rate
                window_duration = (end_sample - start_sample) / rate
                window_segments = self._chunks_to_segments(output.get("chunks", []), window_duration, request)
                if not window_segments and output.get("text", "").strip():
                    window_segments = [Segment(0.0, window_duration, output["text"], request.language,
                                               str(self.spec.model), self.backend_name)]
                for seg in window_segments:
                    seg.start = min(seg.start + offset, duration) + base_offset
                    seg.end = min(seg.end + offset, duration) + base_offset
                    if request.on_segment is not None:
                        request.on_segment(seg)
                segments.extend(window_segments)
            if request.progress is not None:
                request.progress(base_offset + batch[-1][1] / rate, progress_total)
        log.info("Decoded %d segments in %.1fs", len(segments), time.perf_counter() - started)
        return TranscriptionResult(
            segments=segments, language=request.language, language_probability=None,
            duration=base_offset + duration, model=str(self.spec.model), backend=self.backend_name,
            compute_type=self.compute_type,
            capabilities_used={"hotwords": hotwords_used, "word_timestamps": False, "vad": use_vad,
                               "multilingual": False},
            notes=["Transformers pipeline: no per-segment decode statistics, so confidence is unavailable; "
                   + ("speech windows come from Silero VAD (faster-whisper)." if use_vad
                      else "VAD was off: silence/music can trigger hallucinated text.")],
            hotwords_applied=applied if hotwords_used else [],
        )

    @staticmethod
    def speech_windows(
        audio: Any, rate: int, vad_parameters: dict[str, Any], max_window_s: float = 29.0, max_gap_s: float = 2.0,
    ) -> list[tuple[int, int]]:
        """Silero VAD speech regions grouped into <=29 s windows (Whisper's 30 s context).

        Pauses shorter than ``max_gap_s`` stay inside a window, so timestamps remain exact
        (``window start + position inside the window``) without any remapping.
        """
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        fields = {k: v for k, v in vad_parameters.items() if k in VadOptions.__dataclass_fields__}
        options = VadOptions(**{**fields, "max_speech_duration_s": max_window_s - 1.0})
        windows: list[tuple[int, int]] = []
        for chunk in get_speech_timestamps(audio, options, sampling_rate=rate):
            start, end = int(chunk["start"]), int(chunk["end"])
            if windows and (end - windows[-1][0]) / rate <= max_window_s and (start - windows[-1][1]) / rate <= max_gap_s:
                windows[-1] = (windows[-1][0], end)
            else:
                windows.append((start, end))
        return windows

    def _run_batch(
        self, audio: Any, rate: int, batch: list[tuple[int, int]], use_vad: bool, opts: dict[str, Any],
        generate_kwargs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Decode one batch of windows (or, with VAD off, the whole file as one sequential long-form run)."""
        inputs = [{"raw": audio[s:e], "sampling_rate": rate} for s, e in batch]
        kwargs: dict[str, Any] = dict(
            batch_size=len(inputs), return_timestamps=True, generate_kwargs=generate_kwargs,
        )
        if not use_vad and opts.get("chunk_length_s"):
            # null (default) = Whisper's own sequential long-form decoding, which the Transformers docs
            # recommend over the "very experimental" pipeline chunking.
            kwargs["chunk_length_s"] = int(opts["chunk_length_s"])
        result = self._pipe(inputs if len(inputs) > 1 else inputs[0], **kwargs)
        return result if isinstance(result, list) else [result]

    def _chunks_to_segments(
        self, chunks: list[dict[str, Any]], duration: float, request: DecodeRequest,
    ) -> list[Segment]:
        """Pipeline chunks -> segments. Whisper sometimes omits an end time or emits end < start;
        repair those from the next chunk's start (or the audio end) and clamp to the audio."""
        kept = [c for c in chunks if c.get("text", "").strip()]
        segments: list[Segment] = []
        for index, chunk in enumerate(kept):
            start, end = chunk["timestamp"]
            start = min(max(float(start or 0.0), 0.0), duration)
            if end is None or float(end) < start:
                next_start = kept[index + 1]["timestamp"][0] if index + 1 < len(kept) else None
                end = next_start if next_start is not None and float(next_start) >= start else duration
            segments.append(Segment(
                start=start, end=min(float(end), duration), text=chunk["text"], language=request.language,
                model=str(self.spec.model), backend=self.backend_name,
            ))
        return segments

    def close(self) -> None:
        if self._pipe is not None:
            self._pipe = None
            gc.collect()
            if self._torch is not None and self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()


def _resolve_engine(spec: ModelSpec) -> str:
    """Turn a 'configurable' entry into a concrete engine (or refuse clearly)."""
    if spec.engine != "configurable":
        return spec.engine
    if not spec.model:
        raise ModelNotConfiguredError(
            f"Model entry '{spec.key}' is a placeholder: set 'model' in config/models.yaml "
            "(a Hugging Face Whisper repo, or a local CTranslate2 folder) to use it."
        )
    candidate = Path(str(spec.model))
    if candidate.is_dir() and (candidate / "model.bin").is_file():
        return "faster_whisper"
    return "transformers"


def create_adapter(
    spec: ModelSpec, device: str = "cuda", device_index: int = 0, offline: bool = False,
) -> ASRAdapter:
    engine = _resolve_engine(spec)
    if engine == "mlx_whisper":                    # macOS / Apple Silicon: imported only when such a model is used
        from .asr.mac import MlxWhisperBackend

        return MlxWhisperBackend(spec, device="metal", device_index=0, offline=offline)
    adapter_cls = {"faster_whisper": FasterWhisperAdapter, "transformers": TransformersAdapter}[engine]
    return adapter_cls(spec, device=device, device_index=device_index, offline=offline)


def convert_to_ctranslate2(
    spec: ModelSpec, output_dir: Path, quantization: str = "float16", force: bool = False,
) -> Path:
    """Convert a Transformers Whisper checkpoint to CTranslate2 so faster-whisper can run it.

    Experimental for third-party fine-tunes: the converter supports standard Whisper
    architectures, but always verify the result with ``scripts/benchmark.py`` against the
    Transformers backend before trusting it.
    """
    register_cuda_dll_directories()
    from ctranslate2.converters import TransformersConverter

    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"{output_dir} is not empty; pass force=True / --force to overwrite")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    log.info("Converting %s -> %s (%s)", spec.model, output_dir, quantization)
    converter = TransformersConverter(str(spec.model), copy_files=["preprocessor_config.json"])
    converter.convert(str(output_dir), quantization=quantization, force=force)

    tokenizer_json = output_dir / "tokenizer.json"
    if not tokenizer_json.is_file():
        # faster-whisper needs tokenizer.json; many fine-tune repos only ship vocab.json + merges.txt
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(spec.model))
        backend = getattr(tokenizer, "backend_tokenizer", None)
        if backend is None:
            raise RuntimeError("Could not build tokenizer.json (no fast tokenizer available)")
        backend.save(str(tokenizer_json))
        log.info("Generated tokenizer.json")
    return output_dir
