"""CI helper for the Intel-Mac build (works on any OS; prints facts, builds nothing).

    python scripts/ci_macos_facts.py                    import the native libraries and list the wheel deployment tags of the installed packages
    python scripts/ci_macos_facts.py --max-macos 12.0   same, and fail when a wheel asks for a newer macOS
    python scripts/ci_macos_facts.py --counts junit.xml JSON test counts from a JUnit file
    python scripts/ci_macos_facts.py --require-green counts.json   exit 1 unless the test counts have no failed test
    python scripts/ci_macos_facts.py --probe-video      decode a generated 60 s video with the app's own extractor and print durations / progress
"""

from __future__ import annotations

import importlib
import importlib.metadata as metadata
import json
import platform
import re
import sys
import xml.etree.ElementTree as ET

ARM64_ONLY = ["mlx.core", "mlx_whisper"]                                  # the Apple Silicon speech engine (Metal); not installable on Intel
NATIVE = ["ctranslate2", "faster_whisper", "onnxruntime", "av", "numpy", "soundfile", "PIL", "cv2", "shapely", "pyclipper", "rapidfuzz",
          "tokenizers", "yaml", "pypdf", "PySide6.QtCore", "PySide6.QtGui", "PySide6.QtWidgets", "rapidocr_onnxruntime", "keyring"]


def counts(junit: str) -> dict:
    root = ET.parse(junit).getroot()
    suites = [root] if root.tag == "testsuite" else list(root)
    total = sum(int(s.get("tests", 0)) for s in suites)
    failed = sum(int(s.get("failures", 0)) for s in suites)
    errors = sum(int(s.get("errors", 0)) for s in suites)
    skipped = sum(int(s.get("skipped", 0)) for s in suites)
    return {"passed": total - failed - errors - skipped, "failed": failed + errors, "skipped": skipped, "total": total}


def wheel_tags() -> list[tuple[str, tuple[int, int], str]]:
    """(package, highest macOS version its wheel tags ask for, tags) for every installed distribution with a platform wheel."""
    rows = []
    for dist in metadata.distributions():
        wheel = next((f for f in (dist.files or []) if f.name == "WHEEL"), None)
        if wheel is None:
            continue
        text = dist.locate_file(wheel).read_text(encoding="utf-8", errors="replace")
        tags = re.findall(r"^Tag: (.+)$", text, re.MULTILINE)
        macos = [tuple(int(x) for x in m.groups()) for t in tags for m in [re.search(r"macosx_(\d+)_(\d+)_", t)] if m]
        if macos:
            rows.append((dist.metadata["Name"], max(macos), ", ".join(tags)[:90]))
    return sorted(rows, key=lambda r: (-r[1][0], -r[1][1], r[0].lower()))


def probe_video() -> int:
    """What the app's PyAV extractor does with a generated 60 s mp4 (aac + mpeg4): stream durations, decoded length, last progress report."""
    import tempfile
    from pathlib import Path

    import av

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.audio import extract_audio_pyav
    from tests.test_ingest_router import encode

    with tempfile.TemporaryDirectory() as tmp:
        video = encode(Path(tmp) / "v.mp4", "mp4", "aac", "mpeg4", seconds=60.0)
        with av.open(str(video)) as c:
            print("container duration s:", (c.duration or 0) / 1_000_000)
            for s in c.streams:
                span = float(s.duration * s.time_base) if s.duration else None
                print(f"  stream {s.type:6} {s.codec_context.name:8} duration {span}")
        seen = []
        out = extract_audio_pyav(video, Path(tmp) / "a.wav", {"sample_rate": 16000}, progress=seen.append)
        import wave
        with wave.open(str(out)) as w:
            print("decoded wav s:", w.getnframes() / w.getframerate())
        print("progress reports:", [round(x, 1) for x in seen[-5:]], "count", len(seen))
    return 0


def main(argv: list[str]) -> int:
    if argv[:1] == ["--probe-video"]:
        return probe_video()
    if len(argv) == 2 and argv[0] == "--require-green":
        data = json.loads(open(argv[1], encoding="utf-8").read())
        print(f"tests on macOS: {data}")
        if data.get("failed", 1):
            print(f"::error::{data.get('failed')} test(s) failed on macOS - the build stops here instead of making a DMG from a red suite")
        return 1 if data.get("failed", 1) else 0
    if len(argv) == 2 and argv[0] == "--counts":
        print(json.dumps(counts(argv[1])))
        return 0
    limit = None
    if len(argv) == 2 and argv[0] == "--max-macos":
        limit = tuple(int(x) for x in argv[1].split("."))
    print(f"python {platform.python_version()} {platform.machine()} {platform.platform()}")
    bad = 0
    for name in NATIVE + (ARM64_ONLY if platform.machine() == "arm64" else []):
        try:
            module = importlib.import_module(name)
            print(f"ok    {name:22} {getattr(module, '__version__', '')}")
        except Exception as exc:  # noqa: BLE001 - this IS the report
            bad += 1
            print(f"FAIL  {name:22} {type(exc).__name__}: {str(exc)[:160]}")
    print("\nwheel deployment tags (highest macOS each package asks for):")
    rows = wheel_tags()
    for name, version, tags in rows[:12]:
        print(f"  macOS {version[0]}.{version[1]:<2} {name:26} {tags}")
    top = rows[0][1] if rows else (0, 0)
    print(f"\nHIGHEST wheel requirement: macOS {top[0]}.{top[1]}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
