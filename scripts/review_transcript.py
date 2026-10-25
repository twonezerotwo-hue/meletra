"""Sanity review of a finished transcript folder: beginning / middle / end windows and a suspicious-term scan.

Reads outputs/<lecture>/ (local_transcript.json, quality_report.json when present) and prints, per 10-minute window:
segments, words, words per minute, flagged segments, repeated sentences, tiny timestamps, and the text itself so a
human can check it. Nothing is modified.

Usage:  python scripts/review_transcript.py "outputs/<lecture>" [--minutes 10] [--show 14]
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.platform_utils import configure_utf8_console  # noqa: E402
from src.vocabulary import build_index, normalize_key, tokenize  # noqa: E402

# Terms the user asked to look out for, plus common biochemistry / neuroscience ones (correct spelling).
WATCH = [
    "corticospinal tract", "spinothalamic tract", "substantia nigra", "periaqueductal gray", "vestibulocochlear nerve",
    "acetylcholinesterase", "phosphofructokinase", "medial lemniscus", "glycolysis", "gluconeogenesis", "pyruvate dehydrogenase",
    "glucose-6-phosphate dehydrogenase", "citric acid cycle", "electron transport chain", "oxidative phosphorylation",
    "hexokinase", "glucokinase", "pyruvate kinase", "lactate dehydrogenase", "glycogen phosphorylase", "glycogen synthase",
    "fructose bisphosphate", "NADH", "FADH2", "acetyl-CoA", "oxaloacetate", "isocitrate dehydrogenase", "succinate dehydrogenase",
    "cytochrome", "ATP synthase", "insulin", "glucagon", "HMG-CoA reductase", "carnitine", "ketone bodies", "urea cycle",
]


def fmt(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def window_report(segments: list[dict], start: float, end: float, show: int, label: str) -> None:
    inside = [s for s in segments if s["start"] >= start and s["start"] < end]
    words = sum(len(tokenize(s["text"])) for s in inside)
    minutes = max((end - start) / 60.0, 1e-9)
    flagged = [s for s in inside if s.get("uncertain")]
    keys = Counter(normalize_key(s["text"]) for s in inside if len(tokenize(s["text"])) >= 4)
    repeated = sum(n - 1 for n in keys.values() if n > 1)
    tiny = [s for s in inside if s["end"] - s["start"] < 0.2 and len(tokenize(s["text"])) >= 2]
    gaps = [b["start"] - a["end"] for a, b in zip(inside, inside[1:]) if b["start"] - a["end"] > 20]
    print(f"\n=== {label}: {fmt(start)} - {fmt(end)} | segments {len(inside)} | words {words} ({words / minutes:.0f}/min) | "
          f"flagged {len(flagged)} | repeated sentences {repeated} | tiny-timestamp {len(tiny)} | gaps>20s {len(gaps)}")
    for seg in inside[:show]:
        mark = " [!]" if seg.get("uncertain") else ""
        print(f"  [{fmt(seg['start'])}]{mark} {seg['text'].strip()}")


def scan(text: str, vocab) -> None:
    lowered = normalize_key(text)
    words = sorted({w.lower() for w in tokenize(text)}, key=str)
    print("\n=== suspicious-term scan (correct spelling present / near-miss spellings seen)")
    for term in WATCH:
        key = normalize_key(term)
        present = lowered.count(key)
        near = []
        if len(key) >= 8:
            for word in words:
                wk = normalize_key(word)
                if wk != key and abs(len(wk) - len(key)) <= 3 and difflib.SequenceMatcher(None, wk, key).ratio() >= 0.84:
                    near.append(word)
        if present or near:
            print(f"  {term:<34} correct x{present:<3}" + (f" | near-miss words: {', '.join(near[:6])}" if near else ""))
    unknown = Counter(w for w in re.findall(r"[A-Za-z][A-Za-z\-']{6,}", text.lower()) if not vocab.has_word(w))
    print("\n=== frequent long words that are not in the medical dictionary (candidates to eyeball)")
    print("  " + ", ".join(f"{w} x{n}" for w, n in unknown.most_common(60)))


def main() -> int:
    configure_utf8_console()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path)
    parser.add_argument("--minutes", type=float, default=10.0)
    parser.add_argument("--show", type=int, default=14, help="segments to print per window")
    parser.add_argument("--no-scan", action="store_true")
    args = parser.parse_args()
    data = json.loads((args.folder / "local_transcript.json").read_text(encoding="utf-8"))
    segments = [s for s in data["segments"] if s["text"].strip()]
    uncertain_file = args.folder / "uncertain_segments.json"
    flagged_starts = {round(e["start"], 2) for e in json.loads(uncertain_file.read_text(encoding="utf-8"))} if uncertain_file.is_file() else set()
    for seg in segments:
        seg["uncertain"] = seg.get("uncertain") or round(seg["start"], 2) in flagged_starts
    duration = float(data["duration"])
    span = args.minutes * 60.0
    first_speech = segments[0]["start"] if segments else 0.0
    window_report(segments, first_speech, first_speech + span, args.show, "BEGINNING (from first speech)")
    middle = duration / 2 - span / 2
    window_report(segments, middle, middle + span, args.show, "MIDDLE")
    window_report(segments, max(duration - span, 0), duration, args.show, "END")
    if not args.no_scan:
        scan(" ".join(s["text"] for s in segments), build_index("all", None))
    return 0


if __name__ == "__main__":
    sys.exit(main())
