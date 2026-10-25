"""Optional WhisperX word alignment and speaker diarization (isolated in its own venv).

WhisperX is NOT a dependency of the main environment: it pins torch~=2.8 and huggingface-hub<1.0,
which clash with the pinned main stack. Install it into ``.venv-whisperx`` (setup_windows.ps1
-WithWhisperX) and this module drives it through ``scripts/whisperx_worker.py``. Any failure
here is reported but never aborts the main transcription.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Sequence

from . import PROJECT_ROOT
from .schema import Segment, Word

log = logging.getLogger(__name__)

WORKER = PROJECT_ROOT / "scripts" / "whisperx_worker.py"
DEFAULT_PYTHON = PROJECT_ROOT / ".venv-whisperx" / "Scripts" / "python.exe"


class WhisperXUnavailableError(RuntimeError):
    """The optional WhisperX environment is missing or failed."""


def find_whisperx_python(explicit: Optional[Path] = None) -> Optional[Path]:
    for candidate in (explicit, DEFAULT_PYTHON, PROJECT_ROOT / ".venv-whisperx" / "bin" / "python"):
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    return None


def check_diarization_prerequisites(token: Optional[str]) -> None:
    if not token:
        raise WhisperXUnavailableError(
            "Speaker diarization needs a Hugging Face access token: create one at "
            "https://huggingface.co/settings/tokens, accept the terms of the pyannote diarization model, "
            "then set it (this session only):  $env:HF_TOKEN='hf_...'   Diarization is OFF by default."
        )


def refine_segments(
    segments: Sequence[Segment],
    wav_path: Path,
    language: Optional[str],
    *,
    diarize: bool = False,
    device: str = "cuda",
    python: Optional[Path] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    timeout: int = 3600,
) -> list[str]:
    """Replace word timings (and add speakers) in place. Segment text is never touched.

    Returns notes describing what happened. Raises :class:`WhisperXUnavailableError` on failure.
    """
    token = os.environ.get("HF_TOKEN")
    if diarize:
        check_diarization_prerequisites(token)
    interpreter = find_whisperx_python(python)
    if interpreter is None:
        raise WhisperXUnavailableError(
            "WhisperX environment not found (.venv-whisperx). Install it with: "
            "powershell -ExecutionPolicy Bypass -File setup_windows.ps1 -WithWhisperX"
        )
    if not language:
        raise WhisperXUnavailableError("alignment needs a known language; pass --language en/tr")

    with tempfile.TemporaryDirectory(prefix="mlt_wx_") as tmp:
        request_path, result_path = Path(tmp) / "request.json", Path(tmp) / "result.json"
        request_path.write_text(json.dumps({
            "audio_path": str(wav_path), "language": language, "device": device, "diarize": diarize,
            "min_speakers": min_speakers, "max_speakers": max_speakers,
            "segments": [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments],
        }, ensure_ascii=False), encoding="utf-8")
        env = dict(os.environ)
        env["PYTHONUTF8"] = "1"
        log.info("Running WhisperX %s in %s ...", "alignment + diarization" if diarize else "alignment", interpreter)
        try:
            proc = subprocess.run(
                [str(interpreter), "-I", str(WORKER), str(request_path), str(result_path)],
                capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise WhisperXUnavailableError(f"WhisperX timed out after {timeout}s") from exc
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
    if proc.returncode != 0 or "error" in result:
        detail = result.get("error") or (proc.stderr or proc.stdout)[-800:]
        raise WhisperXUnavailableError(f"WhisperX failed: {detail}")
    return _merge(segments, result)


def _merge(segments: Sequence[Segment], result: dict) -> list[str]:
    notes: list[str] = []
    refined = result.get("segments", [])
    if len(refined) != len(segments):
        notes.append(f"WhisperX returned {len(refined)} segments for {len(segments)}; refinement skipped")
        log.warning(notes[-1])
        return notes
    for original, new in zip(segments, refined):
        words = [
            Word(float(w["start"]), float(w["end"]), w["word"], w.get("score"))
            for w in new.get("words", []) if w.get("start") is not None and w.get("end") is not None
        ]
        if words:
            original.words = words
        if new.get("speaker"):
            original.speaker = new["speaker"]
    notes.append("word timestamps refined by WhisperX forced alignment" if result.get("aligned") else "")
    if result.get("diarized"):
        notes.append("speaker labels added by pyannote diarization (via WhisperX)")
    return [n for n in notes if n]
