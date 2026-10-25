"""Offline translation with M2M100 (Facebook, MIT licence) converted for CTranslate2 - the engine behind every Output Language except English.

The model is NOT part of the program: Settings -> Models installs it once (about 0.5 GB, SHA-256 checked). Until then ``find_model`` returns None and the
transcript is kept in the spoken language. Tokenisation uses the model's own SentencePiece file; nothing else (no transformers, no torch) is needed.
"""

from __future__ import annotations

import gc
import logging
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

from ..schema import Segment

log = logging.getLogger(__name__)

MODEL_FOLDER = "translation-m2m100"
REQUIRED_FILES = ("model.bin", "config.json", "sentencepiece.bpe.model")
VOCAB_FILES = ("shared_vocabulary.json", "shared_vocabulary.txt", "vocabulary.json", "vocabulary.txt")
BATCH = 16


class TranslationUnavailable(RuntimeError):
    """The engine cannot run (model files missing or damaged, library absent)."""


def find_model(*parents: Optional[Path]) -> Optional[Path]:
    """The first ``<parent>/translation-m2m100`` folder that holds a complete converted model."""
    for parent in parents:
        if parent is None:
            continue
        folder = Path(parent) / MODEL_FOLDER
        if all((folder / name).is_file() for name in REQUIRED_FILES) and any((folder / name).is_file() for name in VOCAB_FILES):
            return folder
    return None


def lang_token(code: str) -> str:
    return f"__{code}__"


def groups(items: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


class M2M100Engine:
    """``translate(texts, source, target)`` for lists of sentences. ``translator`` / ``tokenizer`` can be injected (tests)."""

    def __init__(self, model_dir: Optional[Path], device: str = "auto", translator: Any = None, tokenizer: Any = None) -> None:
        self.model_dir = Path(model_dir) if model_dir else None
        self.device = device
        self._translator = translator
        self._sp = tokenizer
        self._active_device: Optional[str] = None

    def load(self) -> None:
        if self._translator is not None and self._sp is not None:
            return
        if self.model_dir is None:
            raise TranslationUnavailable("the translation model is not installed")
        try:
            import ctranslate2
            import sentencepiece
        except ImportError as exc:
            raise TranslationUnavailable(f"a translation library is missing ({exc.name})") from exc
        try:
            device = self.device if self.device in ("cpu", "cuda") else ("cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu")
            self._translator = ctranslate2.Translator(str(self.model_dir), device=device, compute_type="auto")
            self._active_device = device
            self._sp = sentencepiece.SentencePieceProcessor()
            if not self._sp.load(str(self.model_dir / "sentencepiece.bpe.model")):
                raise TranslationUnavailable("the SentencePiece file of the translation model could not be read")
        except TranslationUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - damaged model, no CUDA runtime ...
            raise TranslationUnavailable(f"the translation model could not be loaded ({type(exc).__name__}: {exc})") from exc

    def _translate_batch(self, batch: Sequence[Sequence[str]], target: str) -> Any:
        kwargs = {"target_prefix": [[lang_token(target)]] * len(batch), "beam_size": 4, "max_decoding_length": 256}
        try:
            return self._translator.translate_batch(batch, **kwargs)
        except Exception as first:  # noqa: BLE001 - CUDA can fail lazily on the first inference
            if self.device != "auto" or self._active_device != "cuda" or self.model_dir is None:
                raise TranslationUnavailable(f"translation failed ({type(first).__name__}: {first})") from first
            log.warning("M2M100 CUDA inference failed; retrying this translation on the CPU: %s", first)
            self._translator = None
            gc.collect()
            try:
                import ctranslate2

                self._translator = ctranslate2.Translator(str(self.model_dir), device="cpu", compute_type="auto")
                self._active_device = "cpu"
                return self._translator.translate_batch(batch, **kwargs)
            except Exception as second:  # noqa: BLE001 - report both failed execution paths as one actionable error
                raise TranslationUnavailable(
                    f"translation failed on CUDA ({type(first).__name__}: {first}) and CPU ({type(second).__name__}: {second})"
                ) from second

    def translate(self, texts: Sequence[str], source: str, target: str) -> list[str]:
        self.load()
        sp, translator = self._sp, self._translator
        tokens = [[lang_token(source)] + list(sp.encode(t.strip(), out_type=str)) + ["</s>"] if t.strip() else [] for t in texts]
        live = [i for i, t in enumerate(tokens) if t]
        results = []
        if live:
            results = self._translate_batch([tokens[i] for i in live], target)
        out = [""] * len(texts)
        for slot, result in zip(live, results):
            hypothesis = list(result.hypotheses[0])
            if hypothesis and hypothesis[0] == lang_token(target):
                hypothesis = hypothesis[1:]
            out[slot] = sp.decode(hypothesis).strip()
        return out


def translate_segments(engine: M2M100Engine, segments: Sequence[Segment], source: str, target: str, progress: Optional[Callable[[float], None]] = None,
                       should_stop: Optional[Callable[[], None]] = None) -> list[Segment]:
    """The same segments (same times) with the translated text. ``should_stop`` is called between batches and may raise the pipeline's pause exception."""
    started = time.perf_counter()
    texts = [s.text for s in segments]
    translated: list[str] = []
    for batch in groups(texts, BATCH):
        translated.extend(engine.translate(batch, source, target))
        if progress is not None:
            progress(min(len(translated) / max(len(texts), 1), 1.0))
        if should_stop is not None:
            should_stop()
    log.info("M2M100 translated %d segments %s -> %s in %.0fs", len(translated), source, target, time.perf_counter() - started)
    return [Segment(start=s.start, end=s.end, text=t, language=target, model="m2m100", backend="ctranslate2") for s, t in zip(segments, translated)]
