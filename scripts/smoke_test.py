"""Real end-to-end smoke test (needs the GPU and downloads/uses the Whisper model).

Generates a short English medical lecture with the built-in Windows voice, wraps it in an MP4,
runs the real CLI in the chosen mode(s) and checks outputs, WER and medical-term recall.
This is NOT part of the pytest suite because it needs a GPU and model weights.

Usage:  python scripts/smoke_test.py [--modes quality,fast] [--model default] [--keep]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_transcription_config  # noqa: E402
from src.metrics import medical_term_scores, word_error_rate  # noqa: E402
from src.platform_utils import configure_utf8_console, find_executable  # noqa: E402
from src.vocabulary import build_index  # noqa: E402

SCRIPT = (
    "The corticospinal tract descends through the internal capsule. "
    "Lesions of the substantia nigra cause Parkinson disease. "
    "The spinothalamic tract carries pain and temperature sensation. "
    "Broca area is responsible for speech production."
)
EXPECTED_FILES = ("raw_transcript.txt", "clean_transcript.txt", "transcript.srt", "transcript.json",
                  "uncertain_segments.json", "metadata.json", "processing.log")


def synthesize(text: str, wav: Path) -> None:
    txt = wav.with_suffix(".txt")
    txt.write_text(text, encoding="utf-8")
    command = (
        "Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.SetOutputToWaveFile('{wav}'); $s.Speak([IO.File]::ReadAllText('{txt}', [Text.Encoding]::UTF8)); $s.Dispose()"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", command], check=True, capture_output=True)


def make_mp4(ffmpeg: Path, wav: Path, mp4: Path) -> None:
    subprocess.run(
        [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=320x240:r=5",
         "-i", str(wav), "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(mp4)],
        check=True, capture_output=True)


def run_mode(mode: str, media: Path, out_root: Path, model: str, vocab) -> bool:
    cmd = [sys.executable, str(ROOT / "app.py"), "transcribe", str(media), "--mode", mode, "--language", "en",
           "--subject", "neuroscience", "--model", model, "--output-dir", str(out_root / mode)]
    started = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=ROOT)
    elapsed = time.perf_counter() - started
    print(f"\n--- {mode.upper()} mode: exit {proc.returncode}, {elapsed:.1f}s wall (includes model load) ---")
    if proc.returncode != 0:
        print(proc.stderr[-2500:] or proc.stdout[-2500:])
        return False
    out = out_root / mode / media.stem
    missing = [name for name in EXPECTED_FILES if not (out / name).is_file()]
    if missing:
        print(f"FAIL missing outputs: {missing}")
        return False
    raw = (out / "raw_transcript.txt").read_text(encoding="utf-8")
    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    wer = word_error_rate(SCRIPT, raw, "en")
    terms = medical_term_scores(SCRIPT, raw, vocab)
    timing, gpu = meta["timing"], meta["gpu"]
    print(f"transcript : {raw.strip()}")
    print(f"WER        : {100 * wer:.1f}%")
    print(f"term recall: {terms['recall']:.2f}  precision: {terms['precision']:.2f}  missed: {terms['missed']}")
    print(f"compute    : {meta['model']['compute_type_used']} (fallback used: {meta['model']['fallback_used']})")
    print(f"timing     : audio {timing['audio_seconds']}s | load {timing['model_load_seconds']}s | "
          f"decode {timing['decode_seconds']}s | RTF {timing['real_time_factor']}")
    print(f"GPU memory : baseline {gpu.get('baseline_mib')} MiB -> peak {gpu.get('peak_mib')} MiB "
          f"(+{gpu.get('peak_delta_mib')} MiB)")
    ok = wer < 0.35 and (terms["recall"] or 0) >= 0.5
    print("RESULT     :", "PASS" if ok else "FAIL (accuracy below smoke threshold)")
    return ok


def main() -> int:
    configure_utf8_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", default="quality,fast")
    parser.add_argument("--model", default="default")
    parser.add_argument("--keep", action="store_true", help="keep generated files in outputs/smoke_test")
    args = parser.parse_args()

    ffmpeg = find_executable("ffmpeg")
    if ffmpeg is None:
        print("FAIL: FFmpeg not found")
        return 1
    cfg = load_transcription_config()
    vocab = build_index("all", "en", Path(cfg["paths"]["medical_terms_abs"]))
    work = ROOT / "outputs" / "smoke_test" if args.keep else Path(tempfile.mkdtemp(prefix="mlt_smoke_"))
    work.mkdir(parents=True, exist_ok=True)
    wav, mp4 = work / "smoke_lecture.wav", work / "smoke_lecture.mp4"
    synthesize(SCRIPT, wav)
    make_mp4(ffmpeg, wav, mp4)
    print(f"Test media: {mp4}")

    results = {mode: run_mode(mode, mp4, work, args.model, vocab) for mode in args.modes.split(",")}
    print("\nSMOKE TEST:", ", ".join(f"{m}={'PASS' if ok else 'FAIL'}" for m, ok in results.items()))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
