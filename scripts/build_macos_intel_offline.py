"""Build MELETRA-Intel-Mac-Offline.dmg on a REAL Intel Mac (macOS x86_64). Run it through ./build_macos_intel.sh.

    python scripts/build_macos_intel_offline.py            build, audit, sign, make the DMG, check it, write the manifest
    python scripts/build_macos_intel_offline.py --dry-run  print the plan (works anywhere; builds nothing)
    python scripts/build_macos_intel_offline.py --audit-only "/path/MELETRA.app"

Environment (all optional):
    MLT_MODEL_DIR        a folder that already holds the verified Whisper large-v3 files (else they are downloaded ONCE, here, on the
                         build Mac, and every file is checked against the SHA-256 pinned in config/app_models.json)
    MLT_SIGN_IDENTITY    "Developer ID Application: Name (TEAMID)" - sign for distribution (else ad-hoc: the bundle is intact but macOS warns)
    MLT_NOTARY_PROFILE   `xcrun notarytool store-credentials` profile - notarize + staple (needs the identity)
    MLT_SOURCE_CONTACT   e-mail / URL printed in THIRD_PARTY_LICENSES.txt as the contact for the GPL source offer (libx264 / libx265 are
                         inside the bundled PyAV and OpenCV libraries) - the release is not "ready" without it
    MLT_SKIP_TESTS=1     skip the unit tests (not recommended)

It refuses to run anywhere but an Intel Mac: an Apple-Silicon or Windows machine cannot produce a tested Intel build.
Nothing in this script was executed on a Mac when it was written (the development computer is Windows); its pure parts are unit-tested
(tests/test_macos_intel_release.py) and --dry-run prints the exact commands.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import plistlib
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from build_tools import macos as mac                                  # noqa: E402
from build_tools import macos_intel as intel                          # noqa: E402
from build_tools import lgpl_media_macos as lgpl_macos                # noqa: E402
from build_tools.layout import base_version, release_dir              # noqa: E402
from src import PRODUCT_ID, __version__                                   # noqa: E402

RELEASE = release_dir(__version__, "macOS", ROOT).parent / intel.RELEASE_FOLDER
WORK = ROOT / "build" / "macos-intel"
DIST = ROOT / "dist" / f"v{base_version(__version__)}" / "macos-intel"
BUILT = DIST / mac.APP_BUNDLE
APP = DIST / intel.APP_FOLDER
LICENSE_PACKAGES = ["faster-whisper", "ctranslate2", "onnxruntime", "av", "numpy", "soundfile", "PyYAML", "pypdf", "RapidFuzz", "PySide6-Essentials",
                    "shiboken6", "keyring", "Pillow", "opencv-python-headless", "pyclipper", "shapely", "six", "rapidocr-onnxruntime", "tokenizers",
                    "huggingface_hub", "pyinstaller"]


def say(message: str) -> None:
    print(f"[mac-intel] {message}", flush=True)


def run(argv: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(argv, check=False, text=True, capture_output=kw.pop("capture", True), **kw)


def must(argv: list[str], what: str, **kw) -> subprocess.CompletedProcess:
    say(f"{what}: {' '.join(argv)[:200]}")
    done = subprocess.run(argv, check=False, text=True, **kw)
    if done.returncode != 0:
        raise SystemExit(f"FAILED: {what} (exit {done.returncode})")
    return done


# --------------------------------------------------------------------------------------------------------------- preflight
def host_facts() -> dict:
    return {"system": platform.system(), "machine": platform.machine(), "macos": platform.mac_ver()[0], "python": platform.python_version(),
            "model": (run(["sysctl", "-n", "hw.model"]).stdout.strip() if sys.platform == "darwin" else ""),
            "cpu": (run(["sysctl", "-n", "machdep.cpu.brand_string"]).stdout.strip() if sys.platform == "darwin" else ""),
            "ram_gb": (round(int(run(["sysctl", "-n", "hw.memsize"]).stdout.strip() or 0) / 1024 ** 3, 1) if sys.platform == "darwin" else 0)}


def preflight() -> dict:
    facts = host_facts()
    problems = []
    if facts["system"] != "Darwin":
        problems.append("this must run on a Mac (this computer is " + facts["system"] + ")")
    elif facts["machine"] != "x86_64":
        problems.append(f"this Mac is {facts['machine']}: an Intel build must be made, and tested, on an Intel (x86_64) Mac - not under Rosetta")
    if sys.version_info[:2] != (3, 11):
        problems.append(f"Python 3.11 is required (this is {facts['python']})")
    if facts["macos"] and tuple(int(x) for x in facts["macos"].split(".")[:2]) < tuple(int(x) for x in intel.MIN_MACOS.split(".")):
        problems.append(f"macOS {intel.MIN_MACOS} or newer is required (this is {facts['macos']})")
    free = shutil.disk_usage(ROOT).free / 1024 ** 3
    if free < intel.MIN_FREE_GB:
        problems.append(f"{intel.MIN_FREE_GB} GB of free disk space are needed (there are {free:.0f} GB)")
    if problems:
        raise SystemExit("REAL INTEL MAC REQUIRED - " + "; ".join(problems))
    return facts


def source_state() -> dict:
    git = shutil.which("git")
    if git and (ROOT / ".git").exists():
        commit = run([git, "rev-parse", "HEAD"], cwd=ROOT).stdout.strip()
        branch = run([git, "branch", "--show-current"], cwd=ROOT).stdout.strip()
        dirty = bool(run([git, "status", "--porcelain", "--untracked-files=no"], cwd=ROOT).stdout.strip())
        return {"commit": commit, "branch": branch, "dirty": dirty}
    note = ROOT / "SOURCE_COMMIT.txt"                         # a source kit made on Windows carries its commit
    if note.is_file():
        data = dict(line.split("=", 1) for line in note.read_text(encoding="utf-8").splitlines() if "=" in line)
        return {"commit": data.get("commit", "unknown"), "branch": data.get("branch", "unknown"), "dirty": False}
    return {"commit": "unknown", "branch": "unknown", "dirty": False}


# --------------------------------------------------------------------------------------------------------------- the model
def stage_model() -> tuple[Path, dict]:
    """The verified Whisper large-v3 folder + its file list. Downloaded once on this (build) Mac or taken from MLT_MODEL_DIR; every file is
    checked against the pinned SHA-256. The finished app never downloads anything."""
    from src.services.model_service import ModelService
    from src.services.paths import get_paths

    stage = WORK / "model-stage"
    paths = get_paths({"MLT_DATA_DIR": str(stage)}, create=True)
    service = ModelService(paths)
    info = service.info(intel.MODEL_KEY)
    target = paths.models / info.folder
    given = os.environ.get("MLT_MODEL_DIR")
    if given:
        source = Path(given).expanduser()
        source = source / info.folder if (source / info.folder).is_dir() else source
        say(f"copying the model from {source}")
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
    else:
        say("downloading Whisper large-v3 (about 3 GB, checked against pinned SHA-256 values) ...")
        service.download(intel.MODEL_KEY, progress=lambda e: None)
    result = service.verify(intel.MODEL_KEY)
    if not result.ok:
        raise SystemExit("the speech model is not intact: " + "; ".join(result.problems))
    folder = service.locate(intel.MODEL_KEY)
    return folder, {f.path: {"size": f.size, "sha256": f.sha256} for f in info.files}


# --------------------------------------------------------------------------------------------------------------- tests
def run_tests() -> dict:
    if os.environ.get("MLT_SKIP_TESTS") == "1":
        counts_file = os.environ.get("MLT_TEST_COUNTS_FILE")                  # CI runs the suite in its own step and hands the counts over
        if counts_file and Path(counts_file).is_file():
            counts = json.loads(Path(counts_file).read_text(encoding="utf-8"))
            say(f"test counts taken from {counts_file}: {counts}")
            return {**counts, "ran": True, "ran_in": "the CI test step"}
        say("tests skipped (MLT_SKIP_TESTS=1)")
        return {"passed": 0, "failed": 0, "skipped": 0, "ran": False}
    say("running the test suite (tests that need Windows are skipped)")
    done = run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"], cwd=ROOT)
    tail = (done.stdout or "").strip().splitlines()[-1:] or [""]
    counts = {k: int(n) for n, k in re.findall(r"(\d+) (passed|failed|skipped|error|errors)", tail[0])}
    if done.returncode != 0:
        print(done.stdout[-3000:])
        raise SystemExit(f"the test suite failed: {tail[0]}")
    return {"passed": counts.get("passed", 0), "failed": counts.get("failed", 0), "skipped": counts.get("skipped", 0), "ran": True}


# --------------------------------------------------------------------------------------------------------------- bundle
def pyinstaller() -> None:
    from scripts import build_macos_app as builder

    assets = next(s for s in builder.build_plan("x86_64", root=ROOT, python=sys.executable) if s.name == "assets")
    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir(parents=True)
    must(assets.argv, "icon + entitlements", cwd=ROOT)
    spec_argv = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--distpath", str(DIST), "--workpath", str(WORK / "pyinstaller"),
                 str(ROOT / "build_tools" / "MELETRA-macos.spec")]
    must(spec_argv, "PyInstaller (x86_64)", cwd=ROOT, env={**os.environ, "MLT_MAC_ARCH": "x86_64"})
    if not BUILT.is_dir():
        raise SystemExit(f"PyInstaller did not produce {BUILT}")
    BUILT.rename(APP)                                                     # "MELETRA.app": before anything is signed
    shutil.rmtree(DIST / PRODUCT_ID, ignore_errors=True)                      # PyInstaller's second copy (the one-folder output) - not needed
    shutil.rmtree(WORK / "pyinstaller", ignore_errors=True)


def versions() -> dict[str, str]:
    out = {}
    for name in LICENSE_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = ""
    return out


def collect_licenses(resources: Path) -> int:
    count = 0
    for name in LICENSE_PACKAGES:
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        for f in dist.files or []:
            if re.search(r"(LICEN[CS]E|COPYING|NOTICE|AUTHORS)", f.name, re.I):
                target = resources / "licenses" / name / f.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(dist.locate_file(f), target)
                count += 1
    return count


def clone(src: Path, dst: Path) -> None:
    """Copy with APFS clones (no extra disk for the 3 GB model or the app); a normal copy when the volume cannot clone."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    done = subprocess.run(["cp", "-cR", str(src), str(dst)], capture_output=True, text=True, check=False)
    if done.returncode != 0:
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst, symlinks=True)


def free_gb() -> float:
    return shutil.disk_usage(ROOT).free / 1024 ** 3


def disk(stage: str) -> None:
    say(f"disk after {stage}: {free_gb():.0f} GB free")
    if free_gb() < 6:
        raise SystemExit(f"only {free_gb():.1f} GB of disk are left after '{stage}' - the DMG needs about 5 GB more; stopping with a clear message instead of a half-written file")


def assemble(model_folder: Path, model_files: dict, commit: str) -> None:
    resources = APP / "Contents" / "Resources"
    target = resources / "models" / intel.MODEL_KEY
    say(f"putting the speech model into the app: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if WORK in model_folder.parents:                                  # staged by this build: move it, never keep two copies
        shutil.move(str(model_folder), str(target))
    else:
        clone(model_folder, target)
    for junk in target.rglob(".DS_Store"):
        junk.unlink()
    contact = os.environ.get("MLT_SOURCE_CONTACT", "").strip()
    vs = versions()
    lgpl = lgpl_macos.SOURCES if os.environ.get(lgpl_macos.ENV, "").strip() else None
    (resources / "THIRD_PARTY_LICENSES.txt").write_text(intel.third_party_licenses(contact, vs, lgpl), encoding="utf-8")
    (resources / "MODEL_LICENSES.txt").write_text(intel.model_licenses(model_files), encoding="utf-8")
    n = collect_licenses(resources)
    say(f"{n} licence files copied from the installed packages")
    (resources / intel.OFFLINE_MARKER).write_text(json.dumps(intel.marker(__version__, model_files, commit), indent=2), encoding="utf-8")
    for folder in intel.REQUIRED_BUNDLE_DIRS:                         # PyInstaller places data in Resources/ (linked from Frameworks/)
        found = [p for p in (resources / folder, APP / "Contents" / "Frameworks" / folder, APP / "Contents" / "MacOS" / folder) if p.exists()]
        if not found:
            raise SystemExit(f"the bundle lacks {folder}")
    disk("assembling the app")


# --------------------------------------------------------------------------------------------------------------- audit
def swap_lgpl_media() -> Optional[dict]:
    """With MLT_LGPL_FFMPEG set (the folders of the LGPL FFmpeg builds): replace the GPL-built FFmpeg libraries of PyAV / OpenCV inside the app. None when not asked for."""
    dirs = lgpl_macos.replacement_dirs_from_env(os.environ.get(lgpl_macos.ENV, ""))
    if not dirs:
        return None
    say("replacing the GPL-built FFmpeg libraries of PyAV / OpenCV by the LGPL-only builds: " + ", ".join(str(d) for d in dirs))
    record = lgpl_macos.swap_app(APP, dirs)
    (APP / "Contents" / "Resources" / "media_libraries.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    say(f"{len(record['replaced'])} FFmpeg libraries replaced, {len(record['removed'])} libraries nothing needs any more removed")
    return record


def is_macho(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            magic = handle.read(4)
    except OSError:
        return False
    return magic in (b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe", b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca")


def macho_files(app: Path) -> list[Path]:
    return [p for p in sorted(app.rglob("*")) if not p.is_symlink() and p.is_file() and is_macho(p)]


def audit_bundle(app: Path) -> tuple[list[str], dict]:
    """Every Mach-O file: has an x86_64 slice, depends only on the app's own libraries or macOS system libraries, has no developer-machine
    rpath, every @loader_path / @executable_path dependency exists - and the highest macOS version any of them asks for (the TRUE minimum
    system of the app, whatever the build setting says)."""
    problems: list[str] = []
    files = macho_files(app)
    highest = (0, 0)
    offenders: list[tuple[tuple[int, int], str]] = []
    for path in files:
        rel = str(path.relative_to(app))
        archs = run(["lipo", "-archs", str(path)]).stdout.split()
        if "x86_64" not in archs:
            problems.append(f"{rel}: no x86_64 slice (has {archs})")
            continue
        out = run(["otool", "-arch", "x86_64", "-L", str(path)]).stdout
        problems += intel.audit_otool(rel, out)
        listing = run(["otool", "-arch", "x86_64", "-l", str(path)]).stdout
        problems += intel.audit_rpaths(rel, listing)
        minos = intel.minos_of(listing)
        if minos:
            highest = max(highest, minos)
            if minos > tuple(int(x) for x in intel.MIN_MACOS.split(".")):
                offenders.append((minos, rel))
        for line in out.splitlines()[1:]:
            m = re.match(r"\s+(@(?:loader_path|executable_path)/\S+) ", line)
            if m:
                base = path.parent if m.group(1).startswith("@loader_path") else app / "Contents" / "MacOS"
                if not (base / m.group(1).split("/", 1)[1]).exists():
                    problems.append(f"{rel}: {m.group(1)} does not exist inside the app")
    offenders.sort(reverse=True)
    info = {"mach_o_files": len(files), "highest_minos": f"{highest[0]}.{highest[1]}", "above_target": [f"{v[0]}.{v[1]} {r}" for v, r in offenders[:15]],
            "problems": len(problems)}
    say(f"audited {len(files)} Mach-O files: {len(problems)} problem(s); highest minimum macOS asked for by any file: {info['highest_minos']}")
    return problems, info


def set_min_system(app: Path, highest: str) -> str:
    """Info.plist LSMinimumSystemVersion = the larger of the target and what the bundled binaries really need (never claim more than is true)."""
    plist = app / "Contents" / "Info.plist"
    data = plistlib.loads(plist.read_bytes())
    wanted = max(tuple(int(x) for x in intel.MIN_MACOS.split(".")), tuple(int(x) for x in highest.split(".")))
    data["LSMinimumSystemVersion"] = f"{wanted[0]}.{wanted[1]}"
    plist.write_bytes(plistlib.dumps(data))
    return data["LSMinimumSystemVersion"]


# --------------------------------------------------------------------------------------------------------------- sign / dmg
def sign(app: Path, identity: Optional[str]) -> dict:
    if identity:
        for argv in mac.codesign_commands(app, identity, ROOT / "build" / "assets" / "entitlements.plist"):
            must(argv, "codesign/verify")
        return {"signed": True, "developer_id": True, "identity": identity, "nested": 0}
    files = sorted(macho_files(app), key=lambda p: len(p.parts), reverse=True)           # deepest first: libraries before what loads them
    failed = []
    for path in files:
        done = run(["codesign", "--force", "--sign", "-", "--timestamp=none", str(path)])
        if done.returncode != 0:
            failed.append(f"{path.relative_to(app)}: {(done.stderr or '').strip()[:120]}")
    if failed:
        raise SystemExit(f"ad-hoc signing failed for {len(failed)} nested file(s): " + "; ".join(failed[:5]))
    say(f"ad-hoc signed {len(files)} nested Mach-O files (inside-out)")
    must(["codesign", "--force", "--sign", "-", "--timestamp=none", str(app)], "ad-hoc signature of the app (bundle integrity only - NOT a Developer ID)")
    verify = run(["codesign", "--verify", "--deep", "--strict", "--verbose=2", str(app)])
    say((verify.stderr or verify.stdout or "").strip()[-300:])
    if verify.returncode != 0:
        raise SystemExit("codesign --verify --deep --strict failed: " + (verify.stderr or "")[-400:])
    return {"signed": True, "developer_id": False, "identity": "ad-hoc", "nested": len(files)}


def make_dmg(app: Path, identity: Optional[str], notary: Optional[str]) -> Path:
    dmg = RELEASE / intel.DMG_NAME
    RELEASE.mkdir(parents=True, exist_ok=True)
    dmg.unlink(missing_ok=True)
    staging = WORK / "dmg"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    clone(app, staging / app.name)                                    # an APFS clone: no second 5 GB copy on the build disk
    (staging / "Applications").symlink_to("/Applications")
    must(["hdiutil", "create", "-volname", intel.VOLUME_NAME, "-srcfolder", str(staging), "-ov", "-format", "UDZO", str(dmg)], "hdiutil create")
    shutil.rmtree(staging, ignore_errors=True)
    if identity:
        must(["codesign", "--force", "--timestamp", "--sign", identity, str(dmg)], "sign the DMG")
    if identity and notary:
        for argv in mac.notarize_commands(dmg, notary):
            must(argv, "notarize/staple")
    disk("creating the DMG")
    return dmg


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def split_dmg(dmg: Path, part_mb: int) -> list[dict]:
    """Pieces below the 2 GiB limit of a release asset; the whole file's SHA-256 is recorded BEFORE the split and checked after joining."""
    parts_dir = RELEASE / "parts"
    shutil.rmtree(parts_dir, ignore_errors=True)
    parts_dir.mkdir(parents=True)
    size = part_mb * 1024 * 1024
    parts = []
    with dmg.open("rb") as source:
        index = 1
        while True:
            block = source.read(size)
            if not block:
                break
            target = parts_dir / intel.split_name(dmg.name, index)
            target.write_bytes(block)
            parts.append({"name": target.name, "bytes": len(block), "sha256": hashlib.sha256(block).hexdigest()})
            index += 1
    say(f"split into {len(parts)} part(s) of at most {part_mb} MiB")
    return parts


# --------------------------------------------------------------------------------------------------------------- post checks
def clean_env(work: Path) -> dict[str, str]:
    home = work / "home"
    home.mkdir(parents=True, exist_ok=True)
    return {"HOME": str(home), "PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "TMPDIR": str(work / "tmp"), "MLT_DATA_DIR": str(work / "data"),
            "MLT_ALLOW_MULTIPLE": "1", "MLT_SELFTEST_DEEP": "1", "LANG": "en_US.UTF-8", "QT_QPA_PLATFORM": "offscreen"}


def check_installed_copy(dmg: Path, keep: Optional[Path] = None) -> dict:
    """Mount the DMG, drag the app out (copy) into a fresh Applications-like folder, EJECT, then run the app's own deep self-test from the copy
    with a bare environment - the copy must not depend on the mounted image."""
    result = {"mount": False, "copy": False, "eject": False, "selftest": False, "detail": "", "installed_app": ""}
    work = Path(tempfile.mkdtemp(prefix="mlt_dmg_check_"))
    mount = work / "mount"
    mount.mkdir()
    try:
        must(["hdiutil", "attach", "-nobrowse", "-readonly", "-mountpoint", str(mount), str(dmg)], "mount the DMG")
        result["mount"] = True
        if not (mount / intel.APP_FOLDER).is_dir() or not (mount / "Applications").is_symlink():
            raise SystemExit("the DMG does not show the app and the Applications shortcut")
        installed = (keep or work) / "Applications"
        shutil.rmtree(installed, ignore_errors=True)
        installed.mkdir(parents=True)
        must(["ditto", str(mount / intel.APP_FOLDER), str(installed / intel.APP_FOLDER)], "drag the app out of the DMG")
        result["copy"] = True
        must(["hdiutil", "detach", str(mount)], "eject the DMG")
        result["eject"] = not mount.is_mount() if hasattr(mount, "is_mount") else True
        env = clean_env(work)
        (work / "tmp").mkdir(exist_ok=True)
        out = work / "selftest.json"
        exe = installed / intel.APP_FOLDER / "Contents" / "MacOS" / PRODUCT_ID
        done = subprocess.run([str(exe), "--selftest", str(out)], env=env, cwd=work, check=False, timeout=1500)
        report = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {"ok": False}
        result["selftest"] = bool(report.get("ok")) and done.returncode == 0
        result["detail"] = "; ".join(f"{c['name']}: {c['detail'][:70]}" for c in report.get("checks", []) if not c["ok"])
        result["installed_app"] = str(installed / intel.APP_FOLDER)
        return result
    finally:
        subprocess.run(["hdiutil", "detach", "-force", str(mount)], capture_output=True, check=False)
        if keep is None:
            shutil.rmtree(work, ignore_errors=True)


# --------------------------------------------------------------------------------------------------------------- main
def plan_text() -> str:
    return "\n".join([
        "1  preflight: Darwin + x86_64 + Python 3.11 + macOS >= " + intel.MIN_MACOS + f" + {intel.MIN_FREE_GB} GB free (refuses anything else)",
        "2  stage the Whisper large-v3 folder (MLT_MODEL_DIR or one download here) and verify every SHA-256",
        "3  pytest tests (skipped with MLT_SKIP_TESTS=1; CI runs them in their own step and passes the counts in)",
        "4  PyInstaller x86_64 with build_tools/MELETRA-macos.spec (medical_vocab, medical_terms, config, RapidOCR models)",
        "5  put the model, THIRD_PARTY_LICENSES.txt, MODEL_LICENSES.txt, licenses/ and offline_release.json into Contents/Resources",
        "6  audit every Mach-O file (lipo x86_64, otool -L / -l: no Homebrew, no developer paths, no missing @loader_path files, highest minimum macOS)",
        "7  Info.plist minimum system = what the binaries really need; codesign inside-out (Developer ID with hardened runtime if MLT_SIGN_IDENTITY, else ad-hoc) and verify",
        f"8  hdiutil create {intel.DMG_NAME} (app + Applications shortcut), sign, notarize + staple if MLT_NOTARY_PROFILE",
        "9  mount the DMG, copy the app out, eject, run --selftest (deep: every model checksum) from the copy with a bare environment",
        "10 SHA-256 file, release_manifest.json, split into parts for transfer; then ci_validate_packaged.py (CI) or validate_macos_intel.py (a Mac you sit at)",
    ])


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--audit-only", type=Path)
    parser.add_argument("--keep-installed", type=Path, help="leave the copy that was taken out of the DMG in this folder (for the packaged-app tests)")
    parser.add_argument("--split-mb", type=int, default=0, help="also write the DMG in parts of this many MiB (release assets are limited to 2 GiB)")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(plan_text())
        return 0
    facts = preflight()
    if args.audit_only:
        problems, info = audit_bundle(args.audit_only)
        print("\n".join(problems) or "no problems", json.dumps(info, indent=1), sep="\n")
        return 1 if problems else 0
    started = time.time()
    identity, notary = os.environ.get("MLT_SIGN_IDENTITY") or None, os.environ.get("MLT_NOTARY_PROFILE") or None
    source = source_state()
    if source["dirty"]:
        say("WARNING: the source tree has uncommitted changes - the recorded commit does not describe this build")
    disk("start")
    model_folder, model_files = stage_model()
    disk("staging the model")
    tests = run_tests()
    pyinstaller()
    disk("PyInstaller")
    assemble(model_folder, model_files, source["commit"])
    media = swap_lgpl_media()
    problems, audit = audit_bundle(APP)
    if problems:
        print("\n".join(problems[:60]))
        raise SystemExit(f"the bundle audit found {len(problems)} problem(s) - not building the DMG")
    effective_min = set_min_system(APP, audit["highest_minos"])
    say(f"minimum macOS written to Info.plist: {effective_min}" + ("" if effective_min == intel.MIN_MACOS else f"  (above the {intel.MIN_MACOS} target: {audit['above_target'][:3]})"))
    state = sign(APP, identity)
    dmg = make_dmg(APP, identity, notary)
    digest = sha256(dmg)
    (RELEASE / f"{intel.DMG_NAME}.sha256").write_text(f"{digest}  {intel.DMG_NAME}\n", encoding="utf-8")
    shutil.rmtree(DIST / PRODUCT_ID, ignore_errors=True)
    check = check_installed_copy(dmg, args.keep_installed)
    if not (check["mount"] and check["copy"] and check["eject"] and check["selftest"]):
        raise SystemExit(f"the DMG check failed: {check}")
    parts = split_dmg(dmg, args.split_mb) if args.split_mb else []
    notarized = bool(identity and notary)
    ci = {k: os.environ.get(v, "") for k, v in (("provider", "GITHUB_ACTIONS"), ("repository", "GITHUB_REPOSITORY"), ("run_id", "GITHUB_RUN_ID"), ("sha", "GITHUB_SHA"),
                                                ("runner_name", "RUNNER_NAME"), ("image_version", "ImageVersion"), ("runner_label", "MLT_CI_RUNNER_LABEL"))}
    if ci["provider"]:
        ci["provider"] = "GitHub Actions"
    manifest = {"release": intel.RELEASE_LABEL, "version": __version__, "source_branch": source["branch"], "source_commit": source["commit"],
                "source_tree_dirty": source["dirty"], "architecture": "x86_64", "target": "macOS Intel", "min_macos_target": intel.MIN_MACOS,
                "min_macos_effective": effective_min, "mach_o_audit": audit, "build_host": facts, "ci": ci if ci["provider"] else None,
                "physical_2019_mac_tested": False, "build_date": datetime.now().astimezone().isoformat(timespec="seconds"),
                "bundled_model": intel.MODEL_KEY, "model_checksum_model_bin": model_files["model.bin"]["sha256"], "model_files": model_files,
                "tests": tests, "study_quality_included": (ROOT / "src" / "study" / "factbase.py").is_file() and (ROOT / "src" / "study" / "quality.py").is_file(), "signed": state["signed"], "developer_id": state["developer_id"],
                "signing_identity": state["identity"], "nested_files_signed": state["nested"], "notarized": notarized, "stapled": notarized,
                "dmg": intel.DMG_NAME, "dmg_bytes": dmg.stat().st_size, "dmg_sha256": digest, "parts": parts,
                "media_libraries": media or {"lgpl_only": False, "note": "the PyAV and OpenCV wheels carry libx264 / libx265 (GPL-class)"},
                "gpl_source_offer": "NOT NEEDED" if media else ("SET" if os.environ.get("MLT_SOURCE_CONTACT", "").strip() else "MISSING"), "dmg_check": check,
                "seconds": round(time.time() - started)}
    (RELEASE / "release_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    say(f"DONE: {dmg}  ({dmg.stat().st_size / 1024 ** 3:.2f} GB)  sha256 {digest}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(main())
