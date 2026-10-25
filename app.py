"""medical-lecture-transcriber command line interface.

Examples
  python app.py transcribe "input/lecture.mp4" --mode quality --language en --subject neuroscience
  python app.py batch --mode quality --language auto
  python app.py models
  python app.py convert-model --model turkish_medical
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Privacy: never let Hugging Face libraries phone home with usage telemetry.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from src import PROJECT_ROOT, __version__  # noqa: E402
from src.audio import (  # noqa: E402
    AudioExtractionError, FFmpegNotFoundError, UnsupportedMediaError, list_media_files, validate_media_file,
)
from src.config import ConfigError, load_transcription_config, resolve_path  # noqa: E402
from src.diagnostics import CudaUnavailableError  # noqa: E402
from src.language import LanguageError, normalize_language  # noqa: E402
from src.model_manager import (  # noqa: E402
    GpuOutOfMemoryError, ModelNotConfiguredError, UnknownModelError, convert_to_ctranslate2, get_model_spec,
    load_model_specs,
)
from src.pipeline import ProcessingSummary, RunOptions, process_files, resolve_subjects  # noqa: E402
from src.platform_utils import configure_utf8_console  # noqa: E402
from src.transcriber import format_clock  # noqa: E402
from src.vocabulary import UnknownSubjectError, available_subjects  # noqa: E402

EXIT_OK, EXIT_ERROR, EXIT_USAGE = 0, 1, 2
log = logging.getLogger("app")


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S", stream=sys.stderr,
    )
    for noisy in ("httpx", "httpcore", "httpx2", "httpcore2", "urllib3", "filelock", "numba", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)  # hides the "unauthenticated request" notice


def ensure_directories(cfg: dict) -> None:
    for key in ("input_abs", "outputs_abs"):
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)


def add_run_options(parser: argparse.ArgumentParser, cfg: dict) -> None:
    d = cfg["defaults"]
    parser.add_argument("--mode", choices=("quality", "fast"), default=d["mode"],
                        help="quality = large-v3 float16 beam 5; fast = int8_float16 beam 3 (default: %(default)s)")
    parser.add_argument("--language", default=d["language"],
                        help="auto | en | tr | other Whisper code (default: %(default)s)")
    parser.add_argument("--subject", default=d["subject"],
                        help="vocabulary file(s) or medical area(s) (e.g. neuroscience, cardiology, neuroanatomy), "
                             "comma separated, or 'all'. Optional: the subject is also detected automatically "
                             "(default: %(default)s)")
    parser.add_argument("--model", default=d["model"], help="model key from config/models.yaml (default: %(default)s)")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default=d["device"],
                        help="auto (default): CUDA when it works, otherwise the CPU. cuda never falls back silently")
    parser.add_argument("--device-index", type=int, default=d.get("device_index", 0))
    parser.add_argument("--preprocessing", choices=("none", "light"), default=d.get("preprocessing", "none"),
                        help="audio preprocessing; denoise is never applied (default: %(default)s)")
    parser.add_argument("--multilingual", action="store_true",
                        help="re-detect the language for every 30 s window (faster-whisper backend only)")
    parser.add_argument("--no-hotwords", action="store_true",
                        help="do not bias the model with subject/PDF terms (for A/B comparisons)")
    parser.add_argument("--auto-topic", action="store_true",
                        help="force the subject probe on (default: config topic.auto, which is ON): transcribe the first "
                             "minutes without hotwords to detect language and subject; reuses a saved `probe` plan")
    parser.add_argument("--no-auto-topic", action="store_true",
                        help="skip the probe: context comes from the file name, the PDF and --subject only")
    parser.add_argument("--probe-minutes", type=float, help="length of the topic probe (default: config topic.probe_minutes)")
    parser.add_argument("--context-engine", choices=("expert", "legacy"),
                        help="expert = scored, subject-focused medical context (default); legacy = the earlier word-list order")
    parser.add_argument("--chunk-minutes", type=float,
                        help="long-audio chunk length (default: config long_audio.chunk_minutes = 15)")
    parser.add_argument("--backend", choices=("local", "deepgram", "compare"), default="local",
                        help="local = Whisper on this PC (default, nothing leaves the computer). deepgram / compare "
                             "UPLOAD THE AUDIO to Deepgram (needs DEEPGRAM_API_KEY in .env)")
    parser.add_argument("--deepgram-if-suspect", action="store_true",
                        help="after the local run, ask Deepgram for a second opinion ONLY if the local transcript looks doubtful "
                             "(uploads the audio in that case)")
    parser.add_argument("--reuse-audio", type=Path,
                        help="an already extracted 16 kHz mono WAV of the same file (skips the FFmpeg step)")
    parser.add_argument("--no-resume", action="store_true", help="ignore an existing checkpoint and start from zero")
    parser.add_argument("--keep-checkpoint", action="store_true", help="keep checkpoint/ after a successful run")
    parser.add_argument("--no-quality-audit", action="store_true", help="skip the independent VAD audit (quality_report.json)")
    parser.add_argument("--no-pdf", action="store_true", help="ignore a matching lecture PDF")
    parser.add_argument("--pdf", type=Path, help="explicit lecture PDF to use for terminology context")
    parser.add_argument("--align", action="store_true",
                        help="optional WhisperX word alignment (needs .venv-whisperx, see README)")
    parser.add_argument("--diarize", action="store_true",
                        help="optional speaker diarization via WhisperX/pyannote; needs HF_TOKEN. OFF by default")
    parser.add_argument("--whisperx-python", type=Path, help="python.exe of the WhisperX environment")
    parser.add_argument("--keep-audio", action="store_true", help="keep the extracted 16 kHz WAV in the output folder")
    parser.add_argument("--offline", action="store_true", help="never touch the network (models must be cached)")
    parser.add_argument("--output-dir", type=Path, help="override the outputs folder")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")


def build_parser(cfg: dict) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="app.py", description="Local medical lecture transcription (GPU).")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("transcribe", help="transcribe one audio/video file")
    t.add_argument("file", type=Path)
    add_run_options(t, cfg)

    p = sub.add_parser("probe", help="detect language + subject from the first minutes and save the plan "
                                     "(the next `transcribe --auto-topic` run reuses it)")
    p.add_argument("file", type=Path)
    add_run_options(p, cfg)

    b = sub.add_parser("batch", help="transcribe every supported file in the input folder")
    b.add_argument("--input-dir", type=Path, help="folder to scan (default: input/)")
    add_run_options(b, cfg)

    sub.add_parser("models", help="list models from config/models.yaml")
    sub.add_parser("subjects", help="list medical vocabulary subjects")

    c = sub.add_parser("convert-model", help="convert a Transformers Whisper model to CTranslate2")
    c.add_argument("--model", required=True, help="model key from config/models.yaml")
    c.add_argument("--quantization", default="float16", choices=("float16", "int8_float16", "int8", "float32"))
    c.add_argument("--output", type=Path, help="output folder (default: models/ct2/<key>)")
    c.add_argument("--force", action="store_true")
    c.add_argument("-v", "--verbose", action="store_true")
    return parser


def options_from_args(args: argparse.Namespace, cfg: dict) -> RunOptions:
    language = normalize_language(args.language)
    return RunOptions(
        mode=args.mode, language=language, subjects=resolve_subjects(args.subject, cfg), model_key=args.model,
        device=args.device, device_index=args.device_index, preprocessing=args.preprocessing,
        multilingual=args.multilingual, use_pdf=not args.no_pdf, pdf_path=args.pdf, keep_audio=args.keep_audio,
        offline=args.offline, outputs_root=resolve_path(args.output_dir) if args.output_dir else None,
        align=args.align, diarize=args.diarize, whisperx_python=args.whisperx_python,
        use_hotwords=not args.no_hotwords,
        auto_topic=True if (args.auto_topic or getattr(args, "command", "") == "probe") else (False if args.no_auto_topic else None),
        probe_minutes=args.probe_minutes, probe_only=getattr(args, "command", "") == "probe",
        resume=not args.no_resume, keep_checkpoint=args.keep_checkpoint, quality_audit=not args.no_quality_audit,
        context_engine=args.context_engine, chunk_minutes=args.chunk_minutes, backend=args.backend,
        deepgram_if_suspect=args.deepgram_if_suspect, reuse_audio=args.reuse_audio,
    )


def print_summary(summaries: list[ProcessingSummary]) -> None:
    print("\nSUMMARY")
    print("-------")
    for s in summaries:
        if s.paused:
            print(f"PAUSED {s.media.name}: {s.segments} segments saved; run the same command again to resume\n      -> {s.output_dir}")
        elif s.ok:
            fb = " (fallback compute type)" if s.fallback_used else ""
            rtf = f"RTF {s.real_time_factor:.2f}" if s.real_time_factor else "RTF n/a"
            vram = f", peak GPU mem {s.peak_vram_mib} MiB" if s.peak_vram_mib else ""
            print(f"OK    {s.media.name}: {s.segments} segments, {s.uncertain} uncertain, "
                  f"{format_clock(s.audio_seconds)} audio in {format_clock(s.wall_seconds)} | {rtf} | "
                  f"{s.compute_type}{fb}{vram}\n      -> {s.output_dir}")
        else:
            print(f"FAIL  {s.media.name}: {s.error}")


def run_transcription(files: list[Path], args: argparse.Namespace, cfg: dict) -> int:
    options = options_from_args(args, cfg)
    log.info("Mode=%s language=%s subjects=%s model=%s device=%s backend=%s",
             options.mode, args.language, ",".join(options.subjects), options.model_key, options.device, options.backend)
    if options.backend != "local" or options.deepgram_if_suspect:
        log.warning("Online second opinion requested: the audio may be uploaded to Deepgram (third party).")
    summaries = process_files(files, options, cfg)
    print_summary(summaries)
    return EXIT_OK if all(s.ok for s in summaries) else EXIT_ERROR


def cmd_transcribe(args: argparse.Namespace, cfg: dict) -> int:
    media = validate_media_file(args.file, cfg["audio"])
    return run_transcription([media], args, cfg)


def cmd_probe(args: argparse.Namespace, cfg: dict) -> int:
    media = validate_media_file(args.file, cfg["audio"])
    return run_transcription([media], args, cfg)


def cmd_batch(args: argparse.Namespace, cfg: dict) -> int:
    folder = resolve_path(args.input_dir) if args.input_dir else Path(cfg["paths"]["input_abs"])
    folder.mkdir(parents=True, exist_ok=True)
    files = list_media_files(folder, cfg["audio"])
    if not files:
        print(f"No supported media files in {folder}.\nCopy your MP4/MP3/M4A/WAV files there and run again.")
        return EXIT_OK
    print(f"Found {len(files)} file(s) in {folder}:")
    for f in files:
        print(f"  - {f.name}")
    return run_transcription(files, args, cfg)


def cmd_models(_: argparse.Namespace, __: dict) -> int:
    for key, spec in load_model_specs().items():
        status = "ready" if spec.model else "NOT CONFIGURED (model: null)"
        print(f"{key:<18} engine={spec.engine:<15} model={spec.model}  [{status}]")
        if spec.description:
            print(f"{'':<18} {spec.description}")
    return EXIT_OK


def cmd_subjects(_: argparse.Namespace, cfg: dict) -> int:
    from src.vocabulary import load_terms

    for name in available_subjects(Path(cfg["paths"]["medical_terms_abs"])):
        count = len(load_terms(name, None, Path(cfg["paths"]["medical_terms_abs"])))
        print(f"{name:<20} {count} terms")
    return EXIT_OK


def cmd_convert(args: argparse.Namespace, cfg: dict) -> int:
    spec = get_model_spec(args.model, load_model_specs())
    if spec.engine == "faster_whisper":
        print(f"'{args.model}' already uses faster-whisper ({spec.model}); nothing to convert.")
        return EXIT_OK
    output = resolve_path(args.output) if args.output else PROJECT_ROOT / "models" / "ct2" / args.model
    result = convert_to_ctranslate2(spec, output, args.quantization, args.force)
    print(f"Converted to {result}\nTo use it, add to config/models.yaml:\n"
          f"  {args.model}_ct2:\n    engine: faster_whisper\n    model: {result}")
    return EXIT_OK


COMMANDS = {
    "transcribe": cmd_transcribe, "probe": cmd_probe, "batch": cmd_batch, "models": cmd_models,
    "subjects": cmd_subjects, "convert-model": cmd_convert,
}


def main(argv: Optional[list[str]] = None) -> int:
    configure_utf8_console()
    try:
        cfg = load_transcription_config()
        ensure_directories(cfg)
    except ConfigError as exc:
        print(f"CONFIG ERROR: {exc}", file=sys.stderr)
        return EXIT_ERROR
    parser = build_parser(cfg)
    args = parser.parse_args(argv)
    setup_logging(getattr(args, "verbose", False))
    try:
        return COMMANDS[args.command](args, cfg)
    except (UnsupportedMediaError, FileNotFoundError, LanguageError, UnknownSubjectError, UnknownModelError,
            ModelNotConfiguredError, ConfigError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except (FFmpegNotFoundError, AudioExtractionError, CudaUnavailableError, GpuOutOfMemoryError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
