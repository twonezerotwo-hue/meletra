"""Speech-recognition backends behind ONE interface.

    ASRBackend              the interface (the abstract ``ASRAdapter``): ``load`` / ``transcribe`` / ``close``, a capabilities record,
                            and one result type (``Segment`` / ``Word`` / ``TranscriptionResult``) for every backend
    FasterWhisperBackend    faster-whisper / CTranslate2 - Windows (CUDA, CPU fallback), also macOS Intel and Linux. UNCHANGED.
    TransformersBackend     Hugging Face Transformers pipeline (the optional Turkish medical model). UNCHANGED.
    MacOptimizedBackend     Whisper on Apple's MLX (Metal) for Apple Silicon Macs

``create_backend`` picks the backend a model entry (``config/models.yaml``: ``engine``) asks for. The Windows code path is exactly the
one of version 0.1.1; ``mlx_whisper`` is imported only when a model with ``engine: mlx_whisper`` is used.
"""

from __future__ import annotations

from typing import Optional

from ..model_manager import ASRAdapter as ASRBackend
from ..model_manager import Capabilities, DecodeRequest, FasterWhisperAdapter as FasterWhisperBackend, ModelSpec, TransformersAdapter as TransformersBackend
from ..model_manager import create_adapter as create_backend
from .mac import BackendUnavailable, MlxWhisperBackend as MacOptimizedBackend

#: ``engine`` values of config/models.yaml -> the backend class (the mac backend is imported lazily by create_adapter)
ENGINES = {"faster_whisper": FasterWhisperBackend, "transformers": TransformersBackend, "mlx_whisper": MacOptimizedBackend}

__all__ = ["ASRBackend", "BackendUnavailable", "Capabilities", "DecodeRequest", "ENGINES", "FasterWhisperBackend", "MacOptimizedBackend",
           "ModelSpec", "TransformersBackend", "create_backend", "default_engine", "describe_engines"]


def default_engine(platform=None) -> str:
    """The engine a computer should use for the standard Large-v3 model: MLX on Apple Silicon, CTranslate2 everywhere else."""
    from ..platform import current

    platform = platform or current()
    return "mlx_whisper" if platform.name == "macos" and platform.is_apple_silicon else "faster_whisper"


def describe_engines(platform=None) -> list[dict[str, str]]:
    """For diagnostics: which engines exist and whether they can run on this computer (never raises)."""
    import importlib.util

    from ..platform import current

    platform = platform or current()
    rows = []
    for name, cls in ENGINES.items():
        if name == "mlx_whisper":
            ok = platform.name == "macos" and platform.is_apple_silicon and importlib.util.find_spec("mlx_whisper") is not None
            note = "ready" if ok else ("needs a Mac with Apple Silicon" if not (platform.name == "macos" and platform.is_apple_silicon)
                                       else "mlx-whisper is not installed")
        elif name == "transformers":
            ok = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None
            note = "ready" if ok else "optional PyTorch + Transformers runtime not installed"
        else:
            ok = importlib.util.find_spec("faster_whisper") is not None
            note = "ready" if ok else "faster-whisper is not installed"
        rows.append({"engine": name, "backend": cls.backend_name, "ready": "yes" if ok else "no", "note": note})
    return rows
