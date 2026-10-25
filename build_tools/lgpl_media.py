"""Replace the GPL-built FFmpeg libraries inside the PyAV wheel with an LGPL-only build of the same FFmpeg release.

The PyAV wheel (``av.libs``) ships FFmpeg configured with ``--enable-gpl`` and linked to libx264 / libx265: the whole FFmpeg build is GPL then, although the program only DECODES media
(audio extraction, resampling) and never encodes video. An LGPL-only build of the same release (BtbN ``n<major.minor>-latest-win64-lgpl-shared``: configured without ``--enable-gpl`` /
``--enable-nonfree``, no x264 / x265) keeps every decoder the program needs - H.264, HEVC, AAC, MP3, PCM, Opus, Vorbis, FLAC - in the native FFmpeg code.

How: the PyAV extension modules import the FFmpeg DLLs under delvewheel's mangled names (``avcodec-62-<hash>.dll``). Their import tables are rewritten to the plain names
(``avcodec-62.dll``, shorter, so the string fits in place), the GPL ``av.libs`` folder is emptied, and the seven LGPL DLLs are copied in. The FFmpeg ABI is stable within a major
version and the LGPL build is checked to be the same major.minor release as the one PyAV was built against; the build then decodes a test file through the patched PyAV and refuses
any library whose build configuration is GPL-class.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import zipfile
from pathlib import Path
from typing import Iterable

FFMPEG_LIBS = ("avcodec", "avdevice", "avfilter", "avformat", "avutil", "swresample", "swscale")
MANGLED = re.compile(r"^(?P<lib>" + "|".join(FFMPEG_LIBS) + r")-(?P<major>\d+)-[0-9a-f]{32}\.dll$", re.IGNORECASE)
PLAIN = re.compile(r"^(?P<lib>" + "|".join(FFMPEG_LIBS) + r")-(?P<major>\d+)\.dll$", re.IGNORECASE)
GPL_FLAGS = (b"--enable-gpl", b"--enable-nonfree", b"--enable-libx264", b"--enable-libx265", b"--enable-libxvid", b"--enable-libfdk-aac")

#: the LGPL build the release uses: a dated BtbN autobuild of the same FFmpeg release PyAV 18.1 was built against (8.1), pinned by its SHA-256
PINNED = {
    "name": "ffmpeg-n8.1.3-18-g46d899d7a0-win64-lgpl-shared-8.1.zip",
    "url": "https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-10-15-13-03/ffmpeg-n8.1.3-18-g46d899d7a0-win64-lgpl-shared-8.1.zip",
    "sha256": "4acd82323c2cad9c7ffea5b8e335ebcda39beae83a3138d729545e5a7666a3fd",
    "version": "n8.1.3-18-g46d899d7a0",
    "source": "https://github.com/FFmpeg/FFmpeg/tree/46d899d7a0 (FFmpeg release/8.1 branch), build scripts: https://github.com/BtbN/FFmpeg-Builds",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pinned_archive(cache: Path) -> Path:
    """The pinned LGPL archive in ``cache`` (downloaded once, verified every time)."""
    target = Path(cache) / PINNED["name"]
    if not target.is_file():
        import urllib.request

        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_suffix(".part")
        with urllib.request.urlopen(PINNED["url"], timeout=120) as response, open(part, "wb") as out:
            shutil.copyfileobj(response, out, 1 << 20)
        part.replace(target)
    digest = sha256_file(target)
    if digest != PINNED["sha256"]:
        raise ValueError(f"{target.name}: SHA-256 {digest} is not the pinned {PINNED['sha256']}")
    return target


def plain_name(mangled: str) -> str:
    """``avcodec-62-<32 hex>.dll`` -> ``avcodec-62.dll`` ("" for anything else)."""
    m = MANGLED.match(mangled)
    return f"{m.group('lib').lower()}-{m.group('major')}.dll" if m else ""


def gpl_flags(data: bytes) -> list[str]:
    """The GPL-class options in a library's embedded FFmpeg build configuration."""
    return sorted({f.decode() for f in GPL_FLAGS if f in data})


def lgpl_dlls_from_zip(archive: Path) -> dict[str, bytes]:
    """The seven FFmpeg DLLs (``bin/<lib>-<major>.dll``) of an LGPL shared build, refused if any of them carries a GPL-class build option."""
    out: dict[str, bytes] = {}
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            name = info.filename.replace("\\", "/").rsplit("/", 1)[-1]
            if "/bin/" in "/" + info.filename.replace("\\", "/") and PLAIN.match(name):
                out[name.lower()] = z.read(info)
    missing = [lib for lib in FFMPEG_LIBS if not any(n.startswith(lib + "-") for n in out)]
    if missing:
        raise FileNotFoundError(f"{archive.name} lacks {', '.join(missing)}")
    for name, data in out.items():
        bad = gpl_flags(data)
        if bad:
            raise ValueError(f"{name} in {archive.name} is a GPL-class build ({', '.join(bad)})")
    return out


def license_text_from_zip(archive: Path) -> str:
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            if info.filename.replace("\\", "/").rsplit("/", 1)[-1].upper() in ("LICENSE.TXT", "LICENSE"):
                return z.read(info).decode("utf-8", "replace")
    return ""


def patch_imports(path: Path, mapping: dict[str, str]) -> list[tuple[str, str]]:
    """Rewrite the import table of a PE file: every imported DLL name in ``mapping`` (case-insensitive) becomes its value, written in place (the new name must not be longer)."""
    import pefile

    pe = pefile.PE(str(path), fast_load=True)
    pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"], pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT"]])
    lower = {k.lower(): v for k, v in mapping.items()}
    changed: list[tuple[str, str]] = []
    for entry in list(getattr(pe, "DIRECTORY_ENTRY_IMPORT", [])) + list(getattr(pe, "DIRECTORY_ENTRY_DELAY_IMPORT", [])):
        old = entry.dll.decode()
        new = lower.get(old.lower())
        if not new:
            continue
        if len(new) > len(old):
            raise ValueError(f"{new} is longer than {old}: it cannot be written in place")
        rva = entry.struct.Name if hasattr(entry.struct, "Name") else entry.struct.szName
        offset = pe.get_offset_from_rva(rva)
        pe.set_bytes_at_offset(offset, new.encode("ascii") + b"\0" * (len(old) - len(new) + 1))
        changed.append((old, new))
    if changed:
        data = pe.write()
        pe.close()
        Path(path).write_bytes(data)
    else:
        pe.close()
    return changed


def imported_dlls(path: Path) -> list[str]:
    import pefile

    pe = pefile.PE(str(path), fast_load=True)
    pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]])
    names = [e.dll.decode() for e in getattr(pe, "DIRECTORY_ENTRY_IMPORT", [])]
    pe.close()
    return names


def swap_pyav(av_dir: Path, libs_dir: Path, archive: Path) -> dict:
    """Patch every PyAV extension module under ``av_dir`` to the plain FFmpeg names, empty ``libs_dir`` (``av.libs``) and put the LGPL DLLs of ``archive`` there."""
    av_dir, libs_dir = Path(av_dir), Path(libs_dir)
    lgpl = lgpl_dlls_from_zip(archive)
    gpl_files = sorted(p.name for p in libs_dir.glob("*.dll"))
    mapping = {name: plain_name(name) for name in gpl_files if plain_name(name)}
    for mangled, plain in mapping.items():
        if plain not in lgpl:
            raise ValueError(f"PyAV needs {plain} (it was built against {mangled}); the LGPL build has {', '.join(sorted(lgpl))}")
    patched = {}
    for pyd in sorted(av_dir.rglob("*.pyd")):
        changes = patch_imports(pyd, mapping)
        if changes:
            patched[str(pyd.relative_to(av_dir))] = len(changes)
    leftovers = [str(p.relative_to(av_dir)) for p in av_dir.rglob("*.pyd") if any(MANGLED.match(n) for n in imported_dlls(p))]
    if leftovers:
        raise RuntimeError(f"still importing mangled FFmpeg names: {leftovers[:3]}")
    for old in libs_dir.glob("*"):
        if old.is_file():
            old.unlink()
    for name, data in lgpl.items():
        (libs_dir / name).write_bytes(data)
    return {"removed": gpl_files, "installed": sorted(lgpl), "patched_modules": len(patched), "archive": Path(archive).name,
            "archive_sha256": hashlib.sha256(Path(archive).read_bytes()).hexdigest()}


def gpl_class_files(folder: Path, suffixes: Iterable[str] = (".dll", ".pyd", ".exe")) -> list[str]:
    """Files under ``folder`` whose embedded FFmpeg configuration is GPL-class, or that are x264 / x265 themselves."""
    out = []
    for p in Path(folder).rglob("*"):
        if p.is_file() and p.suffix.lower() in suffixes:
            low = p.name.lower()
            if low.startswith(("libx264", "libx265", "x264", "x265")) or gpl_flags(p.read_bytes()):
                out.append(str(p.relative_to(folder)))
    return sorted(out)
