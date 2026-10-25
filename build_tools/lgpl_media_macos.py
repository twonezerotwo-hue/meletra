"""macOS counterpart of lgpl_media.py: replace the GPL-built FFmpeg libraries of the PyAV and OpenCV wheels inside the BUILT .app by LGPL-only builds of the same major versions.

STATUS: written WITHOUT access to a Mac. The pure parts (name parsing, otool output parsing, the swap plan, the reachability pruning, the audit rules, the build script's flags) are
unit-tested everywhere; the Mach-O editing (otool / install_name_tool) has NOT run on a Mac yet - GitHub Actions was blocked by billing when this was written. It is therefore OFF
unless the build is given ``MLT_LGPL_FFMPEG`` (see below) and the Mac package of 4.0.0 is not published until a run on the Intel runner has passed the packaged test matrix.

Why: the macOS wheels of PyAV 18.1.0 (FFmpeg 8.1, libavcodec 62) and opencv-python-headless 4.10.0.84 (FFmpeg 7.1, libavcodec 61) carry libx264 / libx265 and an FFmpeg configured with
``--enable-gpl`` (checked on the wheels themselves). The program only DECODES media, and every decoder it needs (AAC, MP3, PCM, Opus, Vorbis, FLAC, H.264, HEVC) is native FFmpeg code.

How (the same idea as on Windows, but the libraries keep their FILE NAMES so no extension module has to be relinked):
  1. scripts/build_macos_ffmpeg_lgpl.sh builds FFmpeg 8.1.3 and 7.1.5 from the pinned source tarballs with ``--disable-gpl --disable-nonfree --disable-autodetect`` (no x264, no x265,
     no external library at all);
  2. every bundled ``libav*/libsw*`` dylib of the app is overwritten by the build of the same major version, keeping the old file name; the new library's references to its siblings are
     rewritten to those names (install_name_tool);
  3. every dylib in a ``.dylibs`` folder that no remaining Mach-O refers to any more (libx264, libx265, the codec libraries and network libraries only the old FFmpeg needed) is deleted;
  4. the audit refuses any Mach-O file that still carries a GPL-class FFmpeg option or is x264 / x265 itself. The build's own Mach-O audit then checks that every dependency exists.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Iterable, Optional

from build_tools import lgpl_media

ENV = "MLT_LGPL_FFMPEG"                     # os.pathsep-separated "lib" folders of the LGPL builds (one per FFmpeg major line)
FAMILIES = ("libavcodec", "libavdevice", "libavfilter", "libavformat", "libavutil", "libswresample", "libswscale")
LIB = re.compile(r"^(?P<family>" + "|".join(FAMILIES) + r")\.(?P<major>\d+)(?:\.\d+)*\.dylib$")
GPL_NAMES = ("libx264", "libx265", "libxvid", "libxavs", "libdavs2", "libvidstab", "libpostproc", "libfdk")
MACHO_MAGIC = (b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe", b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca")

#: FFmpeg releases built from source for the app (stable tarballs from ffmpeg.org, pinned by SHA-256)
SOURCES = {
    "62": {"version": "8.1.3", "url": "https://ffmpeg.org/releases/ffmpeg-8.1.3.tar.xz", "sha256": "7138d28c96d9d3e3af4ee3d8cad72741f8ffb40da90c1112235dea3ecd3178a3",
           "for": "PyAV 18.1.0 (libavcodec 62, libavformat 62, libavutil 60, libavfilter 11, libswscale 9, libswresample 6, libavdevice 62)"},
    "61": {"version": "7.1.5", "url": "https://ffmpeg.org/releases/ffmpeg-7.1.5.tar.xz", "sha256": "de668509caf9e35e3cd162473441fdb29538c6d96ed080292b3cf9e6fc5d558f",
           "for": "opencv-python-headless 4.10.0.84 (libavcodec 61, libavformat 61, libavutil 59, libswscale 8, libswresample 5)"},
}


# ------------------------------------------------------------------------------------------------------------------------ pure parts
def family_major(name: str) -> Optional[tuple[str, int]]:
    """("libavcodec", 62) for libavcodec.62.28.102.dylib or libavcodec.62.dylib, else None."""
    m = LIB.match(name)
    return (m["family"], int(m["major"])) if m else None


def parse_otool_deps(text: str, own_name: str = "") -> list[str]:
    """The dependency paths of ``otool -L`` output (first line is the file itself; a dylib's own install name is left out)."""
    deps = []
    for line in text.splitlines()[1:]:
        line = line.strip()
        if not line:
            continue
        path = line.split(" (compatibility", 1)[0].strip()
        if own_name and os.path.basename(path) == own_name:
            continue
        deps.append(path)
    return deps


def is_system(path: str) -> bool:
    return path.startswith(("/usr/lib/", "/System/Library/"))


def is_macho(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(4) in MACHO_MAGIC
    except OSError:
        return False


def bundled_ffmpeg_libs(files: Iterable[Path]) -> list[Path]:
    return [f for f in files if not f.is_symlink() and family_major(f.name)]


def plan_swaps(bundled: Iterable[Path], replacement_dirs: Iterable[Path]) -> tuple[list[tuple[Path, Path]], list[str]]:
    """[(bundled library, LGPL library of the same family and major)] and the libraries no replacement exists for."""
    dirs = [Path(d) for d in replacement_dirs]
    swaps, missing = [], []
    for target in bundled:
        family, major = family_major(target.name)
        source = next((d / f"{family}.{major}.dylib" for d in dirs if (d / f"{family}.{major}.dylib").exists()), None)
        if source is None:
            missing.append(target.name)
        else:
            swaps.append((target, source.resolve()))
    return swaps, missing


def retarget_commands(target: Path, source_deps: Iterable[str], bundled_names: dict[tuple[str, int], str]) -> tuple[list[list[str]], list[str]]:
    """install_name_tool calls that make a copied LGPL library refer to its siblings by the bundled FILE NAMES, plus the dependencies that cannot be satisfied
    (anything that is neither a macOS system library nor another FFmpeg library of the app)."""
    commands, problems = [], []
    for dep in source_deps:
        if is_system(dep):
            continue
        key = family_major(os.path.basename(dep))
        if key and key in bundled_names:
            commands.append(["install_name_tool", "-change", dep, "@loader_path/" + bundled_names[key], str(target)])
        else:
            problems.append(f"{target.name}: depends on {dep}")
    return commands, problems


def unreferenced(graph: dict[str, set[str]], roots: Iterable[str], candidates: Iterable[str]) -> list[str]:
    """Candidates (file basenames) that no root reaches through ``graph`` (basename -> basenames it depends on): they can be deleted."""
    reached, todo = set(), list(roots)
    while todo:
        name = todo.pop()
        if name in reached:
            continue
        reached.add(name)
        todo.extend(graph.get(name, ()))
    return sorted(c for c in candidates if c not in reached)


def gpl_class_files(app: Path, files: Optional[Iterable[Path]] = None) -> list[str]:
    """Mach-O files under ``app`` that are x264 / x265 themselves or carry a GPL-class FFmpeg build option."""
    out = []
    for p in (files if files is not None else sorted(Path(app).rglob("*"))):
        if p.is_symlink() or not p.is_file() or not is_macho(p):
            continue
        if p.name.lower().startswith(GPL_NAMES) or lgpl_media.gpl_flags(p.read_bytes()):
            out.append(str(p.relative_to(app)))
    return sorted(out)


def replacement_dirs_from_env(value: str) -> list[Path]:
    return [Path(p) for p in value.split(os.pathsep) if p.strip()]


def media_record(swapped: list[str], removed: list[str]) -> dict:
    """What the release manifest and the notices state about the media libraries of a swapped app."""
    return {"lgpl_only": True, "libraries": {k: v for k, v in SOURCES.items()}, "replaced": sorted(swapped), "removed": sorted(removed),
            "configure": "--disable-gpl --disable-nonfree --disable-autodetect (no libx264, no libx265, no external library)"}


# ------------------------------------------------------------------------------------------------------------------------ the Mac part
def _run(argv: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, check=False)


def _must(argv: list[str], what: str) -> str:
    done = _run(argv)
    if done.returncode != 0:
        raise SystemExit(f"{what} failed ({' '.join(argv[:3])} ...): {(done.stderr or done.stdout)[-400:]}")
    return done.stdout


def swap_app(app: Path, replacement_dirs: Iterable[Path], run: Callable[[list[str]], str] = lambda argv: _must(argv, "tool")) -> dict:
    """Do steps 2-4 on the built app (macOS only). Returns the record for the manifest; raises SystemExit when anything GPL-class is left or a library has no replacement."""
    app = Path(app)
    files = [p for p in sorted(app.rglob("*")) if not p.is_symlink() and p.is_file() and is_macho(p)]
    bundled = bundled_ffmpeg_libs(files)
    if not bundled:
        raise SystemExit("no FFmpeg library found in the app: nothing to replace (is PyAV bundled?)")
    swaps, missing = plan_swaps(bundled, replacement_dirs)
    if missing:
        raise SystemExit("no LGPL build for: " + ", ".join(missing))
    names: dict[tuple[str, int], list[str]] = {}
    for target, _ in swaps:
        names.setdefault(family_major(target.name), []).append(target.name)
    # one bundled name per (family, major) is expected; PyAV and OpenCV use different majors
    chosen = {key: sorted(v)[0] for key, v in names.items()}
    problems: list[str] = []
    for target, source in swaps:
        shutil.copyfile(source, target)
        os.chmod(target, 0o755)
        run(["install_name_tool", "-id", "@loader_path/" + target.name, str(target)])
        deps = parse_otool_deps(run(["otool", "-L", str(target)]), own_name=target.name)
        commands, bad = retarget_commands(target, deps, chosen)
        problems += bad
        for argv in commands:
            run(argv)
    if problems:
        raise SystemExit("the LGPL libraries need more than macOS and the other FFmpeg libraries:\n  " + "\n  ".join(problems))
    # step 3: delete what nothing refers to any more (only inside .dylibs folders, never the app's own libraries)
    graph: dict[str, set[str]] = {}
    where: dict[str, list[Path]] = {}
    for path in [p for p in sorted(app.rglob("*")) if not p.is_symlink() and p.is_file() and is_macho(p)]:
        deps = parse_otool_deps(run(["otool", "-L", str(path)]), own_name=path.name)
        graph.setdefault(path.name, set()).update(os.path.basename(d) for d in deps if not is_system(d))
        where.setdefault(path.name, []).append(path)
    candidates = {n for n, paths in where.items() if all(".dylibs" in p.parts for p in paths) and n.endswith(".dylib")}
    roots = [n for n in where if n not in candidates]
    gone = unreferenced(graph, roots, candidates)
    removed = []
    for name in gone:
        for path in where[name]:
            path.unlink()
            removed.append(str(path.relative_to(app)))
    left = gpl_class_files(app)
    if left:
        raise SystemExit("GPL-class media libraries are still in the app: " + ", ".join(left))
    return media_record([str(t.relative_to(app)) for t, _ in swaps], removed)


def main(argv: Optional[list[str]] = None) -> int:
    """``python -m build_tools.lgpl_media_macos audit <app>``: exit 1 when a GPL-class Mach-O file is in the app (used by the CI as a blocking step with media=lgpl)."""
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2 or args[0] != "audit":
        print(main.__doc__)
        return 2
    found = gpl_class_files(Path(args[1]))
    print("GPL-class media libraries in the app:" if found else "no GPL-class media library in the app", *found, sep="\n  ")
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
