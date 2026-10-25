"""Compare ASR models on the same audio: speed, VRAM, WER and medical-term recall/precision.

Examples
  python scripts/benchmark.py input/sample.wav --models default,turkish_medical --language tr --reference ref.txt
  python scripts/benchmark.py input/sample.mp4 --models default --mode fast

Any key from config/models.yaml works, so a new English medical model only needs a new
entry there. Results go to outputs/benchmarks/<timestamp>/ (JSON + Markdown).
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import audio as audio_mod  # noqa: E402
from src import language as lang_mod  # noqa: E402
from src.config import load_transcription_config  # noqa: E402
from src.diagnostics import GpuMemoryMonitor  # noqa: E402
from src.exporters import write_json, write_text  # noqa: E402
from src.metrics import medical_term_scores, word_error_rate  # noqa: E402
from src.model_manager import DecodeRequest, ModelNotConfiguredError, get_model_spec, load_model_specs  # noqa: E402
from src.pipeline import apply_hotword_guard  # noqa: E402
from src.platform_utils import configure_utf8_console  # noqa: E402
from src.transcriber import Transcriber  # noqa: E402
from src.vocabulary import build_hotwords, build_index, load_terms, parse_subjects  # noqa: E402

log = logging.getLogger("benchmark")


def free_gpu_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def run_one(key: str, wav: Path, args: argparse.Namespace, cfg: dict[str, Any], reference: Optional[str],
            vocab: Any, hotwords: list[str]) -> dict[str, Any]:
    entry: dict[str, Any] = {"model_key": key}
    try:
        spec = get_model_spec(key, load_model_specs())
        entry.update(engine=spec.engine, model=spec.model)
        transcriber = Transcriber(spec, cfg[args.mode], args.device, 0, args.offline)
    except ModelNotConfiguredError as exc:
        entry.update(status="skipped", error=str(exc))
        return entry
    except Exception as exc:  # noqa: BLE001
        entry.update(status="error", error=f"{type(exc).__name__}: {exc}")
        return entry

    decision = lang_mod.decide(args.language, spec.default_language)
    caps = transcriber.adapter.capabilities
    profile = dict(cfg[args.mode])
    profile["vad_filter"] = profile["vad_filter"] and caps.vad
    profile["word_timestamps"] = profile["word_timestamps"] and caps.word_timestamps
    used_hotwords = hotwords if (args.hotwords and caps.hotwords) else []
    guard_note = apply_hotword_guard(profile, used_hotwords, cfg)   # same safety rule as the real pipeline
    if guard_note:
        entry["profile_note"] = guard_note
    request = DecodeRequest(
        audio_path=wav, language=decision.forced, profile=profile, decoding=cfg["decoding"],
        vad_parameters=cfg["vad"], hotwords=used_hotwords,
    )
    monitor = GpuMemoryMonitor().start()
    started = time.perf_counter()
    try:
        result = transcriber.transcribe(request)
        status, error = "ok", None
    except Exception as exc:  # noqa: BLE001 - keep benchmarking the other models
        result, status, error = None, "error", f"{type(exc).__name__}: {exc}"
    wall = time.perf_counter() - started
    memory = monitor.stop()
    transcriber.close()
    free_gpu_memory()

    entry.update(status=status, error=error, wall_seconds=round(wall, 2), gpu_memory=memory)
    if result is not None:
        text = " ".join(seg.text.strip() for seg in result.segments)
        decode = max(1e-6, wall - result.load_seconds)
        entry.update(
            backend=result.backend, compute_type=result.compute_type, fallback_used=result.fallback_used,
            audio_seconds=round(result.duration, 2), model_load_seconds=round(result.load_seconds, 2),
            decode_seconds=round(decode, 2), real_time_factor=round(decode / result.duration, 4),
            language=result.language, segments=len(result.segments), transcript=text,
            hotwords_used=result.capabilities_used.get("hotwords", False),
        )
        if reference:
            entry["wer"] = round(word_error_rate(reference, text, decision.forced or result.language), 4)
            entry["medical_terms"] = medical_term_scores(reference, text, vocab)
    return entry


def to_markdown(rows: list[dict[str, Any]], audio_name: str, has_reference: bool) -> str:
    def pct(value: Optional[float]) -> str:
        return "n/a" if value is None else f"{100 * value:.1f}%"

    lines = [f"# Benchmark - {audio_name}", "",
             "| model | backend | compute | decode s | RTF | process VRAM MiB (dedicated / spilled) | WER | term recall | term precision | status |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        mem = r.get("gpu_memory") or {}
        mt = r.get("medical_terms") or {}
        lines.append(
            f"| {r['model_key']} | {r.get('backend', '-')} | {r.get('compute_type', '-')} | {r.get('decode_seconds', '-')} | "
            f"{r.get('real_time_factor', '-')} | {mem.get('process_peak_dedicated_mib', '-')} / "
            f"{mem.get('process_peak_shared_mib', '-')} | "
            f"{pct(r.get('wer')) if has_reference else 'no ref'} | {pct(mt.get('recall')) if has_reference else 'no ref'} | "
            f"{pct(mt.get('precision')) if has_reference else 'no ref'} | {r['status']}{': ' + r['error'][:60] if r.get('error') else ''} |")
    lines += ["", "RTF = decode time / audio time (lower is faster, model load excluded). "
              "VRAM = this process's own GPU memory from the Windows 'GPU Process Memory' counters: "
              "'dedicated' is real VRAM, 'spilled' is memory the driver moved to system RAM "
              "(a large spilled value means the GPU was too crowded and the run is slower).", ""]
    for r in rows:
        if r.get("transcript"):
            lines += [f"## {r['model_key']} transcript", "", r["transcript"], ""]
    return "\n".join(lines)


def main() -> int:
    configure_utf8_console()
    cfg = load_transcription_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", type=Path, help="short audio/video file")
    parser.add_argument("--models", default="default", help="comma separated keys from config/models.yaml")
    parser.add_argument("--reference", type=Path, help="reference transcript (UTF-8 text) for WER / term scores")
    parser.add_argument("--mode", choices=("quality", "fast"), default="quality")
    parser.add_argument("--language", default="auto")
    parser.add_argument("--subject", default=cfg["defaults"]["subject"], help="subject(s) for hotwords")
    parser.add_argument("--no-hotwords", dest="hotwords", action="store_false", help="disable hotwords for all models")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path(cfg["paths"]["outputs_abs"]) / "benchmarks")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    reference = args.reference.read_text(encoding="utf-8-sig") if args.reference else None
    terms_dir = Path(cfg["paths"]["medical_terms_abs"])
    forced = lang_mod.normalize_language(args.language)
    vocab = build_index("all", forced, terms_dir)
    hotwords = build_hotwords(load_terms(parse_subjects(args.subject, terms_dir), forced, terms_dir),
                              max_terms=cfg["context"]["max_hotwords"], max_chars=cfg["context"]["max_hotword_chars"])

    keys = [k.strip() for k in args.models.split(",") if k.strip()]
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="mlt_bench_") as tmp:
        wav = audio_mod.extract_audio(args.audio, Path(tmp) / "audio.wav", cfg["audio"])
        for key in keys:
            log.info("=== %s ===", key)
            rows.append(run_one(key, wav, args, cfg, reference, vocab, hotwords))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir / stamp
    write_json(out_dir / "benchmark.json", {"audio": args.audio.name, "mode": args.mode, "language": args.language,
                                           "results": rows})
    write_text(out_dir / "benchmark.md", to_markdown(rows, args.audio.name, reference is not None))
    print(to_markdown(rows, args.audio.name, reference is not None))
    print(f"\nSaved to {out_dir}")
    return 0 if all(r["status"] in ("ok", "skipped") for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
