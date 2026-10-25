"""Windows side, BEFORE the Mac: make the "build kit" you carry to the Intel Mac - the committed source, the commit it came from, previews of the
licence files, the model checksums, and the instructions. It builds nothing for macOS.

    python scripts/prepare_macos_intel_kit.py

Refuses an uncommitted working tree (the kit must describe a real commit). Output: release/v3.1.0/macOS-Intel/
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from build_tools import macos_intel as intel                          # noqa: E402
from build_tools.layout import release_dir                            # noqa: E402
from src import __version__                                           # noqa: E402

OUT = release_dir(__version__, "macOS", ROOT).parent / intel.RELEASE_FOLDER
KIT = "MedicalLectureTranscriber-macOS-Intel-build-kit.zip"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def main() -> int:
    if git("status", "--porcelain", "--untracked-files=no"):
        print("refused: uncommitted changes - commit first, the kit records the commit")
        return 1
    commit, branch = git("rev-parse", "HEAD"), git("branch", "--show-current")
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT / "config" / "app_models.json").read_text(encoding="utf-8"))
    model = next(m for m in cfg["models"] if m["key"] == intel.MODEL_KEY)
    files = {f["path"]: {"size": f["size"], "sha256": f["sha256"]} for f in model["files"]}
    (OUT / "MODEL_SHA256SUMS.txt").write_text("".join(f"{v['sha256']}  {k}\n" for k, v in files.items()), encoding="utf-8")
    (OUT / "THIRD_PARTY_LICENSES.preview.txt").write_text(intel.third_party_licenses("", {}), encoding="utf-8")
    (OUT / "MODEL_LICENSES.preview.txt").write_text(intel.model_licenses(files), encoding="utf-8")

    archive = OUT / "source.zip"
    subprocess.run(["git", "archive", "--format=zip", "--prefix=medical-lecture-transcriber/", "-o", str(archive), "HEAD"], cwd=ROOT, check=True)
    kit = OUT / KIT
    with zipfile.ZipFile(archive) as src, zipfile.ZipFile(kit, "w", zipfile.ZIP_DEFLATED) as out:
        for item in src.infolist():
            if item.filename == "medical-lecture-transcriber/build_macos_intel.sh":
                continue                                                       # written again below with LF line ends and the run bit
            data = src.read(item.filename)
            info = zipfile.ZipInfo(item.filename, date_time=item.date_time)
            info.external_attr = item.external_attr
            info.compress_type = zipfile.ZIP_DEFLATED
            out.writestr(info, data)
        base = "medical-lecture-transcriber/"
        out.writestr(base + "SOURCE_COMMIT.txt", f"commit={commit}\nbranch={branch}\nversion={__version__}\nrelease={intel.RELEASE_LABEL}\n")
        script = zipfile.ZipInfo(base + "build_macos_intel.sh")
        script.external_attr = 0o755 << 16                                    # runnable after unzip on the Mac
        script.compress_type = zipfile.ZIP_DEFLATED
        out.writestr(script, (ROOT / "build_macos_intel.sh").read_bytes().replace(b"\r\n", b"\n"))
    archive.unlink()
    digest = hashlib.sha256(kit.read_bytes()).hexdigest()
    (OUT / f"{KIT}.sha256").write_text(f"{digest}  {KIT}\n", encoding="utf-8")
    (OUT / "STATUS.txt").write_text(
        "REAL INTEL DMG NOT BUILT - INTEL MAC BUILD HOST REQUIRED.\n\n"
        f"This folder holds the build kit for commit {commit} ({branch}). No macOS build, signing or test has happened.\n"
        "Next: unzip the kit on an Intel Mac and run ./build_macos_intel.sh (see docs/MACOS_INTEL_RELEASE.md inside the kit).\n"
        "The speech model on this PC that matches MODEL_SHA256SUMS.txt:\n"
        r"  %LOCALAPPDATA%\MedicalLectureTranscriber\models\whisper-large-v3" "\n"
        "(copy that folder to the Mac and set MLT_MODEL_DIR to it, or let the build download it once).\n", encoding="utf-8")
    print(f"kit      {kit}  ({kit.stat().st_size / 1e6:.1f} MB)\nsha256   {digest}\ncommit   {commit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
