"""Check Python, packages, FFmpeg, config and folders (no GPU / model download).

Usage:  python scripts/check_dependencies.py
Exit code 0 when everything required is present.
"""

from __future__ import annotations

import sys
from importlib import metadata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import PROJECT_ROOT  # noqa: E402
from src.config import ConfigError, load_models_config, load_transcription_config  # noqa: E402
from src.diagnostics import check_ffmpeg  # noqa: E402
from src.platform_utils import configure_utf8_console  # noqa: E402
from src.vocabulary import SUBJECTS, available_subjects  # noqa: E402

REQUIRED = ("faster-whisper", "ctranslate2", "numpy", "pyyaml", "av", "tokenizers", "huggingface-hub")
RECOMMENDED = ("torch", "transformers", "soundfile", "pypdf", "jiwer")
OPTIONAL = ("whisperx",)


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def line(status: str, name: str, detail: str = "") -> None:
    print(f"{status:<5} {name:<26} {detail}")


def main() -> int:
    configure_utf8_console()
    failures = 0
    print("DEPENDENCY CHECK\n----------------")

    major, minor = sys.version_info[:2]
    if (major, minor) == (3, 11):
        line("PASS", "Python", sys.version.split()[0])
    elif (major, minor) >= (3, 9):
        line("WARN", "Python", f"{sys.version.split()[0]} (project targets 3.11)")
    else:
        line("FAIL", "Python", f"{sys.version.split()[0]} is too old (need >= 3.9, target 3.11)")
        failures += 1
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    line("PASS" if in_venv else "WARN", "virtual environment", "active" if in_venv else "not active (use .venv)")

    for name in REQUIRED:
        version = package_version(name)
        if version:
            line("PASS", name, version)
        else:
            line("FAIL", name, "missing: pip install -r requirements.txt")
            failures += 1
    for name in RECOMMENDED:
        version = package_version(name)
        line("PASS" if version else "WARN", name, version or "missing (needed for some features)")
    for name in OPTIONAL:
        version = package_version(name)
        line("PASS" if version else "SKIP", name, version or "optional (requirements-whisperx.txt)")

    torch_version = package_version("torch")
    if torch_version and "+cpu" in torch_version:
        line("WARN", "torch build", f"{torch_version} is CPU-only; GPU needs the +cu126 build")

    ffmpeg = check_ffmpeg()
    line(ffmpeg.status, "FFmpeg", ffmpeg.detail)
    failures += ffmpeg.status == "FAIL"

    try:
        load_transcription_config()
        line("PASS", "config/transcription.yaml", "valid")
        models = load_models_config()
        line("PASS", "config/models.yaml", ", ".join(models))
    except ConfigError as exc:
        line("FAIL", "config", str(exc))
        failures += 1

    missing = [s for s in SUBJECTS if s not in available_subjects(PROJECT_ROOT / "medical_terms")]
    line("FAIL" if missing else "PASS", "medical_terms/", f"missing: {', '.join(missing)}" if missing else f"{len(SUBJECTS)} subjects")
    failures += bool(missing)

    for folder in ("input", "outputs"):
        (PROJECT_ROOT / folder).mkdir(exist_ok=True)
        line("PASS", f"{folder}/", "ready")

    print(f"\nRESULT: {'OK' if failures == 0 else f'{failures} required item(s) missing'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
