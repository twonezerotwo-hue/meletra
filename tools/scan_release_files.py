"""Scan the FILES of a build (the unpacked program, an unzipped package, the public repository's working tree) for credentials and for private material.

    python tools/scan_release_files.py PATH [PATH ...] [--private-name TEXT ...] [--json out.json]

tools/secret_scan.py reads the history of a git repository; this reads what is actually shipped: every file below PATH, as bytes, in blocks. It looks for the credential
patterns of secret_scan.py (GitHub / AWS / Slack / Telegram / OpenAI / Anthropic tokens, private key blocks, bearer tokens, URLs with credentials), for the SMTP / mail-server /
master-secret names the sharing backend uses, for the absolute path of the build machine's user folder, and for every ``--private-name`` (a lecture file name, a user name).
Only patterns are reported, never the secret itself. Exit code 0 = nothing found.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import secret_scan  # noqa: E402

BLOCK = 8 * 1024 * 1024
OVERLAP = 512
CREDENTIALS = {name: pattern for name, pattern in secret_scan.CREDENTIAL_PATTERNS.items() if name != "password assignment"}      # that one is for source text; library data files are full of "password = ..." in docs
MAIL_AND_SERVER = {
    "smtp password name": re.compile(rb"(?i)SMTP_(?:PASSWORD|PASS|USER|HOST)\s*[=:]\s*[\"']?[A-Za-z0-9._%+@-]{3,}"),
    "backend master secret": re.compile(rb"(?i)(?:MASTER|SIGNING|INBOX)_(?:SECRET|KEY)\s*[=:]\s*[\"']?[A-Za-z0-9._%+/=-]{8,}"),
    "gmail password": re.compile(rb"(?i)gmail[^\n]{0,40}(?:password|app password)\s*[=:]\s*[\"']?[A-Za-z0-9 ]{8,}"),
}
PERSONAL = {"build machine user folder": re.compile(rb"(?i)[A-Z]:[\\/]+Users[\\/]+(?!Public\b|Default\b|runneradmin\b|<|\{|%)[A-Za-z0-9._-]{2,}")}
SKIP_SUFFIXES = {".png", ".ico", ".icns", ".bmp", ".jpg", ".jpeg"}


def as_bytes(pattern: re.Pattern) -> re.Pattern:
    return pattern if isinstance(pattern.pattern, bytes) else re.compile(pattern.pattern.encode("latin-1"), pattern.flags & ~re.UNICODE)


GROUPS = [{name: as_bytes(pattern) for name, pattern in group.items()} for group in (CREDENTIALS, MAIL_AND_SERVER, PERSONAL)]
BINARY_SUFFIXES = {".dll", ".pyd", ".exe", ".lib", ".so", ".dylib", ".node"}      # other people's compiled libraries: their code mentions "-----BEGIN PRIVATE KEY" and paths of THEIR build machines


def scan_file(path: Path, private: list[bytes]) -> dict[str, int]:
    found: dict[str, int] = {}
    third_party = path.suffix.lower() in BINARY_SUFFIXES
    groups = [{n: pat for n, pat in GROUPS[0].items() if n != "private key block"}, GROUPS[1]] if third_party else GROUPS
    tail = b""
    with path.open("rb") as handle:
        while True:
            block = handle.read(BLOCK)
            if not block:
                break
            data = tail + block
            for group in groups:
                for name, pattern in group.items():
                    if pattern.search(data):
                        found[name] = found.get(name, 0) + 1
            low = data.lower()
            for name in private:
                if name in low:
                    found[f"private name {name.decode('latin-1')}"] = found.get(f"private name {name.decode('latin-1')}", 0) + 1
            tail = block[-OVERLAP:]
    return found


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--private-name", action="append", default=[])
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    private = [name.lower().encode("utf-8") for name in args.private_name if name]
    findings: dict[str, dict[str, int]] = {}
    files = 0
    size = 0
    for base in args.paths:
        for path in ([base] if base.is_file() else sorted(p for p in base.rglob("*") if p.is_file())):
            if path.suffix.lower() in SKIP_SUFFIXES:
                continue
            files += 1
            size += path.stat().st_size
            hit = scan_file(path, private)
            if hit:
                findings[str(path)] = hit
    summary = {"files": files, "bytes": size, "findings": findings}
    if args.json:
        args.json.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    kinds: dict[str, int] = {}
    for hit in findings.values():
        for name, count in hit.items():
            kinds[name] = kinds.get(name, 0) + 1
    print(f"scanned {files} files, {size / 1e9:.2f} GB: {'CLEAN' if not findings else str(len(findings)) + ' file(s) with findings'}")
    for name, count in sorted(kinds.items()):
        print(f"  {name}: {count} file(s)")
    for path, hit in list(findings.items())[:25]:
        print(f"    {path}: {', '.join(sorted(hit))}")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
