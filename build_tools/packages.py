"""Portable and Full Offline packages: zip the built app folder, plus portable.flag (data next to the program) and, for the
Full Offline package, the Large-v3 model folder."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Callable, Iterable, Optional

from .assemble import PRODUCT_ID

PORTABLE_FLAG = "portable.flag"

PORTABLE_README = """MELETRA - portable package
================================================

1. Unzip this folder anywhere (a USB drive works). Do NOT run it from inside the zip.
2. Double-click {exe}.
3. First start: press DOWNLOAD RECOMMENDED MODEL (about 3 GB, once), or use Settings -> Models -> Import Model Folder.

Everything the program writes (settings, models you download, logs, unfinished work) stays in the "data" folder next to
{exe}. Nothing is written to the registry or to your user profile. Delete the folder to remove the program completely.

The program runs on this computer only. No audio or text is uploaded unless you switch on the optional Deepgram comparison.
Windows 10/11 64-bit. An NVIDIA GPU makes it much faster; without one the CPU is used automatically.

If Windows SmartScreen says "Windows protected your PC": the program is not code-signed yet. Choose "More info" -> "Run anyway".
"""

OFFLINE_README = """MELETRA - Full Offline package
===================================================

This package already contains the Whisper large-v3 speech model (MIT licence) in the "models" folder, so the program works
immediately, with no internet connection.

1. Unzip this folder anywhere (a USB drive works). Do NOT run it from inside the zip.
2. Double-click {exe}. The model is found automatically.

Settings, logs and unfinished work stay in the "data" folder next to {exe}.
Windows 10/11 64-bit. An NVIDIA GPU makes it much faster; without one the CPU is used automatically.

If Windows SmartScreen says "Windows protected your PC": the program is not code-signed yet. Choose "More info" -> "Run anyway".
"""

MODEL_LICENSE = """Whisper large-v3 (CTranslate2 conversion: Systran/faster-whisper-large-v3)
Original model: OpenAI Whisper large-v3, released under the MIT licence (https://github.com/openai/whisper).
The MIT licence allows redistribution of the model weights with this program.
"""


def iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(Path(root).rglob("*")):
        if path.is_file():
            yield path


def write_zip(destination: Path, entries: Iterable[tuple[Path, str]], compresslevel: int = 6,
              progress: Optional[Callable[[int, int], None]] = None) -> Path:
    """Zip ``(file, archive name)`` pairs. Already-compressed model weights use level 1 to stay fast."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    entries = list(entries)
    tmp = destination.with_suffix(".zip.part")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=compresslevel, allowZip64=True) as z:
        for i, (path, name) in enumerate(entries, 1):
            level = 1 if path.suffix.lower() in (".bin", ".onnx", ".pyd") or path.stat().st_size > 400_000_000 else compresslevel
            z.write(path, name, compress_type=zipfile.ZIP_DEFLATED, compresslevel=level)
            if progress:
                progress(i, len(entries))
    tmp.replace(destination)
    return destination


def portable_entries(dist_dir: Path, root_name: str = PRODUCT_ID, readme: str = PORTABLE_README) -> list[tuple[Path, str]]:
    dist_dir = Path(dist_dir)
    return [(p, f"{root_name}/{p.relative_to(dist_dir).as_posix()}") for p in iter_files(dist_dir)]


def extra_text_files(staging: Path, root_name: str, files: dict[str, str]) -> list[tuple[Path, str]]:
    staging = Path(staging)
    staging.mkdir(parents=True, exist_ok=True)
    out = []
    for name, text in files.items():
        path = staging / name.replace("/", "__")
        path.write_text(text, encoding="utf-8")
        out.append((path, f"{root_name}/{name}"))
    return out


def model_entries(model_dir: Path, folder_name: str, root_name: str = PRODUCT_ID) -> list[tuple[Path, str]]:
    """Files of one model folder, stored as ``<root>/models/<folder_name>/<file>`` (skips our verification manifest? no: keep it)."""
    model_dir = Path(model_dir)
    return [(p, f"{root_name}/models/{folder_name}/{p.relative_to(model_dir).as_posix()}") for p in iter_files(model_dir)]


def zip_names(archive: Path) -> list[str]:
    with zipfile.ZipFile(archive) as z:
        return z.namelist()
