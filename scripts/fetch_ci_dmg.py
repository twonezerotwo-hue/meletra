"""Windows side of the CI build: fetch the DMG from the private release, rebuild the EXACT file, verify it, put it on the Desktop.

    python scripts/fetch_ci_dmg.py [--tag mac-intel-<run id>] [--repo owner/name] [--desktop PATH] [--keep-parts]

Without --tag the newest ``mac-intel-*`` release is used. It downloads the parts (a release asset is limited to 2 GiB), checks every part against the
SHA-256 the CI recorded, joins them, checks the SHA-256 of the whole file against the checksum made BEFORE the split (and against the manifest), then
applies ``build_tools.macos_intel.ci_readiness`` to the packaged-app validation report. Only a DMG that passes all of that is copied - byte-identical,
SHA-256 verified again - to ``release/v3.1.0/macOS-Intel/`` and the Desktop. A DMG that fails stays in the download folder and nothing reaches the Desktop.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from build_tools import macos_intel as intel                          # noqa: E402
from build_tools.layout import release_dir                            # noqa: E402
from src import __version__                                           # noqa: E402

TARGET = release_dir(__version__, "macOS", ROOT).parent / intel.RELEASE_FOLDER
DESKTOP = Path.home() / "Desktop"
REPO = "twonezerotwo-hue/whispermedical"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def join_parts(folder: Path, parts: list[dict], out: Path) -> str:
    """Verify each part against its recorded SHA-256, then write the joined file; returns the SHA-256 of the joined file."""
    whole = hashlib.sha256()
    with out.open("wb") as target:
        for part in parts:
            path = folder / part["name"]
            if not path.is_file():
                raise SystemExit(f"missing part {part['name']}")
            data = path.read_bytes()
            if len(data) != part["bytes"] or hashlib.sha256(data).hexdigest() != part["sha256"]:
                raise SystemExit(f"part {part['name']} is damaged (size or SHA-256 differs from what the CI recorded)")
            whole.update(data)
            target.write(data)
    return whole.hexdigest()


def gh(*args: str) -> str:
    done = subprocess.run(["gh", *args], capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise SystemExit(f"gh {' '.join(args[:3])} failed: {(done.stderr or done.stdout).strip()[:300]}")
    return done.stdout


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):           # the redirect target is read by hand: it must not receive the GitHub token
        return None


def resolve_asset_url(repo: str, asset_id: int, token: str) -> str:
    """The short-lived storage URL of a release asset (the API answers 302 with it; the token stays with the API request)."""
    request = urllib.request.Request(f"https://api.github.com/repos/{repo}/releases/assets/{asset_id}",
                                     headers={"Authorization": f"Bearer {token}", "Accept": "application/octet-stream", "User-Agent": "mlt-fetch"})
    try:
        urllib.request.build_opener(_NoRedirect).open(request, timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code in (301, 302, 303, 307, 308) and exc.headers.get("Location"):
            return exc.headers["Location"]
        raise
    raise SystemExit("the asset answered without a redirect")


def download_resumable(get_url: Callable[[], str], dest: Path, size: int, attempts: int = 400, pause: Callable[[float], None] = time.sleep,
                       log: Callable[[str], None] = print) -> None:
    """Fill ``dest`` to ``size`` bytes with HTTP Range requests. A reset connection (common on a slow line) continues where it stopped; the URL is
    looked up again every time because storage URLs expire. A partial file already on disk is continued."""
    have = dest.stat().st_size if dest.is_file() else 0
    if have > size:
        dest.unlink()
        have = 0
    failures = 0
    while have < size:
        try:
            request = urllib.request.Request(get_url(), headers={"Range": f"bytes={have}-", "User-Agent": "mlt-fetch"})
            with urllib.request.urlopen(request, timeout=90) as response:
                if response.status == 200 and have:                       # the server ignored the range: start over
                    have = 0
                mode = "wb" if have == 0 else "ab"
                with dest.open(mode) as out:
                    while block := response.read(1 << 20):
                        out.write(block)
                        have += len(block)
            failures = 0
        except (OSError, http.client.HTTPException, urllib.error.URLError) as exc:
            failures += 1
            have = dest.stat().st_size if dest.is_file() else 0
            log(f"  {dest.name}: {have / 1024 ** 2:.0f} of {size / 1024 ** 2:.0f} MB - connection problem ({type(exc).__name__}), continuing")
            if failures > attempts:
                raise SystemExit(f"giving up on {dest.name} after {failures} connection problems in a row")
            pause(min(30.0, 1.0 + failures))
    if have != size:
        raise SystemExit(f"{dest.name} has {have} bytes, expected {size}")


def download_release(repo: str, tag: str, folder: Path) -> None:
    token = gh("auth", "token").strip()
    release = json.loads(gh("api", f"repos/{repo}/releases/tags/{tag}"))
    for asset in release["assets"]:
        dest = folder / asset["name"]
        print(f"download {asset['name']}  {asset['size'] / 1024 ** 2:.0f} MB")
        download_resumable(lambda a=asset: resolve_asset_url(repo, a["id"], token), dest, int(asset["size"]))


def newest_tag(repo: str) -> str:
    releases = json.loads(gh("release", "list", "--repo", repo, "--json", "tagName,createdAt", "--limit", "30"))
    tags = sorted((r for r in releases if r["tagName"].startswith("mac-intel-")), key=lambda r: r["createdAt"], reverse=True)
    if not tags:
        raise SystemExit("no mac-intel-* release exists yet - the CI build has not published a DMG")
    return tags[0]["tagName"]


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag")
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--desktop", type=Path, default=DESKTOP)
    parser.add_argument("--keep-parts", action="store_true")
    args = parser.parse_args(argv)

    tag = args.tag or newest_tag(args.repo)
    work = TARGET / "_download" / tag
    work.mkdir(parents=True, exist_ok=True)                                  # a partial download from an earlier try is continued
    print(f"release  {tag}  ->  {work}")
    download_release(args.repo, tag, work)
    manifest = intel.load_json(work / "release_manifest.json")
    report_file = work / "packaged_validation.json"
    report = intel.load_json(report_file) if report_file.is_file() else {}
    parts = manifest.get("parts") or []
    if not parts:
        raise SystemExit("the manifest lists no parts")
    joined = work / intel.DMG_NAME
    digest = join_parts(work, parts, joined)
    sidecar = work / f"{intel.DMG_NAME}.sha256"
    recorded = sidecar.read_text(encoding="utf-8").split()[0] if sidecar.is_file() else ""
    print(f"size     {joined.stat().st_size / 1024 ** 3:.2f} GB\nsha256   {digest}\nmanifest {manifest.get('dmg_sha256')}\nsidecar  {recorded}")
    problems = []
    if digest != manifest.get("dmg_sha256"):
        problems.append("the joined file's SHA-256 differs from the manifest")
    if recorded and digest != recorded:
        problems.append("the joined file's SHA-256 differs from the .sha256 file made before the split")
    verdict, reasons, caveats = intel.ci_readiness(report, digest, manifest)
    reasons = problems + reasons
    if reasons:
        verdict = "NEEDS WORK"
    print(f"verdict  {verdict}")
    for r in reasons:
        print(f"  - {r}")
    for c in caveats:
        print(f"  * {c}")
    if verdict != "CI-TESTED INTEL MAC DMG READY":
        print(f"The reconstructed file stays in {work}; nothing was copied to the Desktop.")
        return 1

    TARGET.mkdir(parents=True, exist_ok=True)
    for folder in (TARGET, args.desktop):
        out = folder / intel.DMG_NAME
        shutil.copyfile(joined, out)
        if sha256(out) != digest:
            print(f"the copy {out} differs from the tested DMG")
            return 1
        (folder / f"{intel.DMG_NAME}.sha256").write_text(f"{digest}  {intel.DMG_NAME}\n", encoding="utf-8")
        print(f"copied   {out}  (SHA-256 verified: {digest})")
    for name in ("release_manifest.json", "packaged_validation.json", "build.log", "test_counts.json"):
        if (work / name).is_file():
            shutil.copyfile(work / name, TARGET / name)
    if not args.keep_parts:
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
