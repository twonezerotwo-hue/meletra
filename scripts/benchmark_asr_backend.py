"""Compare speech-recognition backends on ONE recording: speed (real-time factor), load time, memory, segments, and - when a reference
transcript is given - the word error rate. Meant to be run on a Mac to choose / verify the macOS backend, and on Windows to see the
CTranslate2 baseline; the numbers of one machine are never copied to another.

    python scripts/benchmark_asr_backend.py --audio lecture.wav --reference lecture.txt \
        --faster-whisper-model ~/Library/Application\\ Support/MedicalLectureTranscriber/models/whisper-large-v3 \
        --mlx-model ~/Library/Application\\ Support/MedicalLectureTranscriber/models/whisper-large-v3-mlx \
        --whispercpp-bin /opt/homebrew/bin/whisper-cli --whispercpp-model ~/models/ggml-large-v3.bin \
        --seconds 300 --out benchmark_mac.md

Backends:
  faster_whisper  the app's Windows / Intel backend (CTranslate2); on a Mac it runs on the CPU (int8)
  mlx_whisper     the app's Apple Silicon backend (MLX, Metal)
  whispercpp      NOT part of the app: a command-line shim only for the comparison (whisper.cpp, optionally with a Core ML encoder)

A backend whose runtime or model is missing is reported as "skipped" with the reason; it never stops the others.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@dataclass
class Result:
    backend: str
    status: str = "ok"                      # ok | skipped | failed
    note: str = ""
    load_seconds: float = 0.0
    decode_seconds: float = 0.0
    audio_seconds: float = 0.0
    rtf: float = 0.0                        # decode time / audio time (smaller is faster)
    peak_memory_mb: Optional[float] = None
    segments: int = 0
    words: int = 0
    wer: Optional[float] = None
    monotonic: bool = True                  # segment timestamps never go backwards and stay inside the audio
    text: str = field(default="", repr=False)


# ------------------------------------------------------------------------------------------------------------- pure helpers
def normalize(text: str) -> list[str]:
    """Lower-case words without punctuation (the usual basis of a word error rate)."""
    return re.findall(r"[\w'-]+", text.casefold())


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Word-level edit distance divided by the length of the reference."""
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        current = [i]
        for j, h in enumerate(hyp, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (r != h)))
        previous = current
    return previous[-1] / len(ref)


def timestamps_ok(segments: list[Any], audio_seconds: float, slack: float = 1.0) -> bool:
    last = 0.0
    for seg in segments:
        if seg.start < last - 1e-6 or seg.end < seg.start - 1e-6 or seg.end > audio_seconds + slack:
            return False
        last = seg.start
    return True


def render_markdown(results: list[Result], audio: Path, seconds: float) -> str:
    lines = ["# ASR backend benchmark", "", f"Audio: `{audio.name}` ({seconds:.0f} s decoded)", "",
             "| Backend | Status | RTF | Load s | Decode s | Segments | Words | WER | Peak MB | Timestamps | Note |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        if r.status != "ok":
            lines.append(f"| {r.backend} | {r.status} | - | - | - | - | - | - | - | - | {r.note} |")
            continue
        lines.append(f"| {r.backend} | ok | {r.rtf:.3f} | {r.load_seconds:.1f} | {r.decode_seconds:.1f} | {r.segments} | {r.words} | "
                     f"{'-' if r.wer is None else f'{r.wer * 100:.1f} %'} | {'-' if r.peak_memory_mb is None else f'{r.peak_memory_mb:.0f}'} | "
                     f"{'ok' if r.monotonic else 'PROBLEM'} | {r.note} |")
    lines += ["", "RTF = decode time / audio time (0.1 means ten times faster than real time). WER needs `--reference`.", ""]
    return "\n".join(lines)


def audio_seconds_of(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as wav:
            return wav.getnframes() / float(wav.getframerate())
    except (wave.Error, EOFError):
        from src.audio import probe_media_pyav

        return float(probe_media_pyav(path).get("duration_s") or 0.0)


def memory_mb() -> Optional[float]:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1024 ** 2
    except Exception:                                         # noqa: BLE001
        return None


# --------------------------------------------------------------------------------------------------------------------- backends
def prepare_wav(audio: Path, seconds: float, workdir: Path) -> tuple[Path, float]:
    """A 16 kHz mono WAV of the first ``seconds`` seconds (0 = everything), made with the app's own audio code (PyAV, no ffmpeg needed)."""
    from src.audio import extract_audio_pyav
    from src.config import load_transcription_config

    target = workdir / "bench.wav"
    extract_audio_pyav(audio, target, load_transcription_config()["audio"])
    total = audio_seconds_of(target)
    if seconds and seconds < total:
        import soundfile as sf

        data, rate = sf.read(str(target), dtype="float32")
        sf.write(str(target), data[: int(seconds * rate)], rate)
        total = seconds
    return target, total


def run_adapter(name: str, spec_engine: str, model: str, wav: Path, total: float, language: Optional[str], reference: Optional[str]) -> Result:
    from src.config import load_transcription_config
    from src.model_manager import DecodeRequest, ModelSpec, create_adapter

    result = Result(backend=name, audio_seconds=total)
    if not model or not Path(model).exists():
        result.status, result.note = "skipped", "model folder not given or not found"
        return result
    try:
        cfg = load_transcription_config()
        profile = cfg["quality"]
        spec = ModelSpec(name, spec_engine, str(model))
        device = "metal" if spec_engine == "mlx_whisper" else "cpu"
        adapter = create_adapter(spec, device, 0, True)
        started = time.perf_counter()
        adapter.load("int8" if spec_engine == "faster_whisper" else str(profile["compute_type"]))
        result.load_seconds = time.perf_counter() - started
        request = DecodeRequest(audio_path=wav, language=language, profile=profile, decoding=cfg["decoding"], vad_parameters=cfg["vad"],
                                total_duration=total)
        started = time.perf_counter()
        decoded = adapter.transcribe(request)
        result.decode_seconds = time.perf_counter() - started
        result.peak_memory_mb = memory_mb()
        adapter.close()
        result.segments = len(decoded.segments)
        result.text = " ".join(s.text.strip() for s in decoded.segments)
        result.words = len(normalize(result.text))
        result.monotonic = timestamps_ok(decoded.segments, total)
        result.note = f"{decoded.backend}, language {decoded.language}"
    except Exception as exc:                                  # noqa: BLE001 - one backend failing must not stop the comparison
        result.status, result.note = ("skipped" if exc.__class__.__name__ == "BackendUnavailable" else "failed"), f"{exc.__class__.__name__}: {exc}"
        return result
    finish(result, reference)
    return result


def run_whispercpp(binary: str, model: str, wav: Path, total: float, language: Optional[str], reference: Optional[str]) -> Result:
    result = Result(backend="whispercpp", audio_seconds=total)
    if not binary or not Path(binary).exists() or not model or not Path(model).exists():
        result.status, result.note = "skipped", "whisper.cpp binary or model not given"
        return result
    with tempfile.TemporaryDirectory() as tmp:
        prefix = Path(tmp) / "out"
        command = [binary, "-m", model, "-f", str(wav), "-oj", "-of", str(prefix)] + (["-l", language] if language else [])
        started = time.perf_counter()
        try:
            proc = subprocess.run(command, capture_output=True, text=True, timeout=max(600, total * 4), check=False)
        except (OSError, subprocess.SubprocessError) as exc:
            result.status, result.note = "failed", f"{exc.__class__.__name__}: {exc}"
            return result
        result.decode_seconds = time.perf_counter() - started
        data_file = Path(str(prefix) + ".json")
        if proc.returncode != 0 or not data_file.is_file():
            result.status, result.note = "failed", (proc.stderr or "no JSON written")[-200:]
            return result
        data = json.loads(data_file.read_text(encoding="utf-8"))
        parts = [str(item.get("text", "")).strip() for item in data.get("transcription", [])]
        result.segments, result.text = len(parts), " ".join(parts)
        result.words = len(normalize(result.text))
        result.note = "command line (whisper.cpp), load time is inside decode time"
    finish(result, reference)
    return result


def finish(result: Result, reference: Optional[str]) -> None:
    result.rtf = result.decode_seconds / result.audio_seconds if result.audio_seconds else 0.0
    if reference is not None:
        result.wer = word_error_rate(reference, result.text)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--reference", type=Path, help="a text file with the correct transcript (enables the WER column)")
    parser.add_argument("--language", default=None, help="en, tr ... (default: let the backend detect)")
    parser.add_argument("--seconds", type=float, default=300.0, help="decode only the first N seconds (0 = the whole file)")
    parser.add_argument("--faster-whisper-model", default="")
    parser.add_argument("--mlx-model", default="")
    parser.add_argument("--whispercpp-bin", default="")
    parser.add_argument("--whispercpp-model", default="")
    parser.add_argument("--out", type=Path, help="write the markdown table here (also <out>.json)")
    args = parser.parse_args(argv)

    reference = args.reference.read_text(encoding="utf-8") if args.reference else None
    with tempfile.TemporaryDirectory() as tmp:
        wav, total = prepare_wav(args.audio, args.seconds, Path(tmp))
        results = [
            run_adapter("faster_whisper", "faster_whisper", args.faster_whisper_model, wav, total, args.language, reference),
            run_adapter("mlx_whisper", "mlx_whisper", args.mlx_model, wav, total, args.language, reference),
            run_whispercpp(args.whispercpp_bin, args.whispercpp_model, wav, total, args.language, reference),
        ]
    table = render_markdown(results, args.audio, total)
    print(table)
    if args.out:
        args.out.write_text(table, encoding="utf-8")
        args.out.with_suffix(".json").write_text(json.dumps([{k: v for k, v in asdict(r).items() if k != "text"} for r in results], indent=2),
                                                 encoding="utf-8")
    return 0 if any(r.status == "ok" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
