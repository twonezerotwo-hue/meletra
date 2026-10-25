"""Post-build assembly of ``dist/MELETRA``: CUDA runtime, FFmpeg, notices, and a completeness check."""

from __future__ import annotations

import fnmatch
import hashlib
import shutil
import zipfile
from pathlib import Path
from typing import Iterable, Optional

PRODUCT_ID = "MELETRA"
EXE_NAME = f"{PRODUCT_ID}.exe"

#: What CTranslate2 4.8 (CUDA 12) really loads for Whisper: cuBLAS 12 (+ cuBLASLt) and the cuDNN 9 loader. MEASURED, not assumed:
#: listing the modules of a process after real large-v3 transcriptions (float16, int8_float16, int8) shows only
#: cublas64_12, cublasLt64_12 and cudnn64_9 - none of the 1.1 GB of cuDNN sub-libraries is ever loaded, and removing all of them
#: changes neither the text nor the speed (scripts/cuda_runtime_probe.py). cudart is linked statically; its 0.5 MB DLL is kept as a
#: harmless safety. If a future CTranslate2 needs more, add the file names here and rebuild.
CUDA_DLL_PATTERNS = ("cublas64_12.dll", "cublasLt64_12.dll", "cudart64_12.dll", "cudnn64_9.dll")
CUDA_REQUIRED = ("cublas64_12.dll", "cublasLt64_12.dll", "cudart64_12.dll", "cudnn64_9.dll")


def find_cuda_source(extra: Optional[Iterable[Path]] = None) -> Optional[Path]:
    """A folder that contains the cuBLAS 12 DLL: explicit candidates, torch/lib, nvidia-*-cu12 wheels."""
    import importlib.util

    candidates = list(extra or [])
    for package in ("torch", "nvidia"):
        try:
            spec = importlib.util.find_spec(package)
        except (ImportError, ValueError):
            spec = None
        if spec and spec.submodule_search_locations:
            root = Path(list(spec.submodule_search_locations)[0])
            candidates += [root / "lib"] if package == "torch" else sorted(root.glob("*/bin"))
    for folder in candidates:
        if (Path(folder) / "cublas64_12.dll").is_file():
            return Path(folder)
    return None


def select_cuda_dlls(source: Path, patterns: Iterable[str] = CUDA_DLL_PATTERNS) -> list[Path]:
    source = Path(source)
    chosen = [p for p in sorted(source.glob("*.dll")) if any(fnmatch.fnmatch(p.name, pat) for pat in patterns)]
    return chosen


def missing_cuda_dlls(chosen: Iterable[Path], required: Iterable[str] = CUDA_REQUIRED) -> list[str]:
    names = {p.name for p in chosen}
    return [r for r in required if r not in names]


def copy_cuda(dist_dir: Path, source: Optional[Path] = None, patterns: Iterable[str] = CUDA_DLL_PATTERNS) -> list[Path]:
    source = source or find_cuda_source()
    if source is None:
        raise FileNotFoundError("No CUDA 12 runtime found (torch/lib or nvidia-*-cu12 wheels); build a CPU-only package instead.")
    chosen = select_cuda_dlls(source, patterns)
    missing = missing_cuda_dlls(chosen)
    if missing:
        raise FileNotFoundError(f"CUDA runtime incomplete in {source}: missing {', '.join(missing)}")
    target = Path(dist_dir) / "libs" / "cuda"
    target.mkdir(parents=True, exist_ok=True)
    wanted = {p.name for p in chosen}
    for stale in target.glob("*.dll"):                       # an older build may have left libraries that are no longer shipped
        if stale.name not in wanted:
            stale.unlink()
    copied = []
    for dll in chosen:
        destination = target / dll.name
        if not destination.exists() or destination.stat().st_size != dll.stat().st_size:
            shutil.copy2(dll, destination)
        copied.append(destination)
    return copied


def find_ffmpeg_pair(root: Path) -> tuple[Optional[Path], Optional[Path]]:
    """``ffmpeg.exe`` and ``ffprobe.exe`` inside an extracted build (``<root>/**/bin/``)."""
    root = Path(root)
    ffmpeg = next(iter(sorted(root.rglob("ffmpeg.exe"))), None)
    ffprobe = next(iter(sorted(root.rglob("ffprobe.exe"))), None)
    return ffmpeg, ffprobe


def extract_ffmpeg_zip(archive: Path, target: Path) -> tuple[Path, Path]:
    """Unpack only ``bin/ffmpeg.exe``, ``bin/ffprobe.exe`` and the licence files of a BtbN/Gyan zip (no path traversal)."""
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    wanted = ("/bin/ffmpeg.exe", "/bin/ffprobe.exe", "license", "licence", "readme", "copying")
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            name = info.filename.replace("\\", "/")
            lowered = name.lower()
            if info.is_dir() or ".." in name.split("/"):
                continue
            if any(lowered.endswith(w) or (w.isalpha() and w in lowered.rsplit("/", 1)[-1]) for w in wanted):
                destination = target / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src, open(destination, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    ffmpeg, ffprobe = find_ffmpeg_pair(target)
    if ffmpeg is None or ffprobe is None:
        raise FileNotFoundError(f"{archive} does not contain bin/ffmpeg.exe and bin/ffprobe.exe")
    return ffmpeg, ffprobe


def copy_ffmpeg(dist_dir: Path, ffmpeg: Path, ffprobe: Path) -> list[Path]:
    target = Path(dist_dir) / "bin"
    target.mkdir(parents=True, exist_ok=True)
    out = []
    for exe in (ffmpeg, ffprobe):
        destination = target / Path(exe).name
        shutil.copy2(exe, destination)
        out.append(destination)
    return out


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dir_size(path: Path) -> int:
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def human_size(nbytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024 or unit == "GB":
            return f"{nbytes:.0f} {unit}" if unit == "B" else f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} GB"


NOTICES = """MELETRA - third-party components
=====================================================

This program runs entirely on your computer. It bundles the following open-source and redistributable components.

OpenAI Whisper large-v3 (model weights, MIT licence) - https://github.com/openai/whisper
    Distributed in the "Full Offline" package, or downloaded on first run (Systran/faster-whisper-large-v3, MIT).
faster-whisper (MIT), CTranslate2 (MIT), Silero VAD (MIT), ONNX Runtime (MIT), tokenizers (Apache-2.0), NumPy (BSD), PyYAML (MIT),
RapidFuzz (MIT), pypdf (BSD-3-Clause), soundfile / libsndfile (BSD / LGPL-2.1).
SentencePiece (Apache-2.0, https://github.com/google/sentencepiece): the tokenizer of the optional translation model; licence text in licenses\APACHE-2.0.txt.
Optional offline translation pack (not part of this folder: the installer, Settings -> Models or the Translate window downloads it on request): Meta M2M100 418M
(https://huggingface.co/facebook/m2m100_418M), MIT licence, Copyright (c) Facebook, Inc. and its affiliates, converted to the CTranslate2 int8 format;
licence text in licenses\LICENSE-M2M100.txt (it is also stored with the pack).
PyAV (BSD-3-Clause), the Python binding to FFmpeg, used as the decoder inside the program.
Qt for Python / PySide6 (LGPL-3.0): the user interface. You may replace the Qt libraries in the program folder with your own build.
{media_section}{ffmpeg_section}{cuda_section}
The licence texts of the packages above are included in the program folder (look for LICENSE / COPYING files next to each library).
"""

MEDIA_GPL_SECTION = """GPL components - READ THIS
    The FFmpeg libraries that ship inside the PyAV wheel (_internal\\av.libs: libavcodec, libavformat, libavutil, libswscale, libswresample, libavfilter,
    libavdevice) are built with the video encoders libx264 and libx265, both GPL-2.0-or-later. They must be treated as GPL, not as LGPL-only.
    This program only DECODES audio and video; it never encodes video. Corresponding source code and the recipes that build these libraries:
      PyAV wheels (FFmpeg, x264, x265, codecs)  https://github.com/PyAV-Org/pyav-ffmpeg  and  https://github.com/PyAV-Org/PyAV
      FFmpeg   https://ffmpeg.org/download.html
      x264     https://code.videolan.org/videolan/x264
      x265     https://bitbucket.org/multicoreware/x265_git
    To ask for the source code of any component as shipped: https://github.com/twonezerotwo-hue/whispermedical-downloads/issues
"""

MEDIA_LGPL_SECTION = """FFmpeg libraries used through PyAV (_internal\\av.libs: avcodec-62, avformat-62, avutil-60, swscale-9, swresample-6, avfilter-11, avdevice-62):
    an LGPL-only build of FFmpeg {version} (configured without --enable-gpl and --enable-nonfree; no libx264, no libx265) from https://github.com/BtbN/FFmpeg-Builds,
    licence text in licenses\\FFMPEG_LGPL_LICENSE.txt. The program only decodes audio and video. PyAV (BSD-3-Clause) was built against FFmpeg 8.1; its extension
    modules load these DLLs by their plain names, so they can be replaced with any other build of the same FFmpeg major version.
    Source: {source}. To ask for the source code of any component as shipped: https://github.com/twonezerotwo-hue/whispermedical-downloads/issues
"""

FFMPEG_SECTION = """
FFmpeg command-line tools (bin/ffmpeg.exe, bin/ffprobe.exe): LGPL-2.1+ build (no GPL components) from
https://github.com/BtbN/FFmpeg-Builds. Source code: https://ffmpeg.org/download.html and the build scripts of that project.
The program works without these two files (it then decodes audio with PyAV).
"""

CUDA_SECTION = """
NVIDIA CUDA runtime libraries (libs/cuda: cuBLAS, cuDNN, CUDA runtime) are redistributed unmodified under the NVIDIA CUDA Toolkit /
cuDNN licence agreements (redistributable attachments). They are only used when an NVIDIA GPU is present; without one the program
uses the CPU.
"""


def notices_text(has_ffmpeg: bool, has_cuda: bool, media_lgpl: dict | None = None) -> str:
    """``media_lgpl``: the PINNED record of the LGPL FFmpeg build that replaced PyAV's GPL libraries (None = PyAV's own libraries ship, GPL-class)."""
    media = MEDIA_LGPL_SECTION.format(version=media_lgpl["version"], source=media_lgpl["source"]) if media_lgpl else MEDIA_GPL_SECTION
    return NOTICES.format(media_section=media, ffmpeg_section=FFMPEG_SECTION if has_ffmpeg else "", cuda_section=CUDA_SECTION if has_cuda else "")


def write_extra_licenses(dist_dir: Path) -> list[str]:
    """The licence texts of components whose wheels carry none (SentencePiece: Apache-2.0) and of the optional translation model (M2M100: MIT) go to ``licenses/``."""
    target = Path(dist_dir) / "licenses"
    target.mkdir(parents=True, exist_ok=True)
    written = []
    for source in sorted((Path(__file__).resolve().parent / "licenses").glob("*.txt")):
        (target / source.name).write_bytes(source.read_bytes())
        written.append(source.name)
    return written


def write_notices(dist_dir: Path, media_lgpl: dict | None = None) -> Path:
    dist_dir = Path(dist_dir)
    path = dist_dir / "THIRD_PARTY_NOTICES.txt"
    path.write_text(notices_text((dist_dir / "bin" / "ffmpeg.exe").is_file(), (dist_dir / "libs" / "cuda").is_dir(), media_lgpl), encoding="utf-8")
    return path


def verify_dist(dist_dir: Path, need_cuda: bool = False, need_ffmpeg: bool = False) -> list[str]:
    """Problems that make a build unusable (empty list = complete). Cheap: only looks at the file tree."""
    dist_dir = Path(dist_dir)
    problems: list[str] = []
    exe = dist_dir / EXE_NAME
    if not exe.is_file():
        return [f"{EXE_NAME} is missing in {dist_dir}"]
    if exe.stat().st_size < 100_000:
        problems.append(f"{EXE_NAME} is suspiciously small")
    internal = dist_dir / "_internal"
    if not internal.is_dir():
        problems.append("the _internal folder (libraries) is missing")
    for rel in ("config/transcription.yaml", "config/app_models.json", "config/models.yaml", "config/common_english_words.txt"):
        if not (internal / rel).is_file():
            problems.append(f"resource {rel} is missing")
    if not (internal / "medical_terms").is_dir() or not any((internal / "medical_terms").glob("*.txt")):
        problems.append("the medical vocabulary (medical_terms) is missing")
    if len(list((internal / "medical_vocab").glob("*/*.txt"))) < 20:
        problems.append("the Medical Expert vocabulary (medical_vocab) is missing")
    for pattern in ("ctranslate2/*.dll", "av.libs/*.dll", "onnxruntime/capi/*.dll", "PySide6/Qt6Widgets.dll",
                    "faster_whisper/assets/silero_vad_v6.onnx"):
        if not list(internal.glob(pattern)):
            problems.append(f"{pattern} is missing")
    if need_cuda:
        have = {p.name for p in (dist_dir / "libs" / "cuda").glob("*.dll")}
        problems += [f"CUDA runtime {name} is missing" for name in CUDA_REQUIRED if name not in have]
    if need_ffmpeg:
        for name in ("ffmpeg.exe", "ffprobe.exe"):
            if not (dist_dir / "bin" / name).is_file():
                problems.append(f"bin/{name} is missing")
    return problems
