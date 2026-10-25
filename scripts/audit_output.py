"""Audit a finished transcription folder (read-only):  ``python scripts/audit_output.py "<Medical Transcription>\\<name>"``

Reports: beginning / middle / end windows, missing-speech gaps and VAD coverage (from quality_report.json), hallucination and repetition
flags, hotword echo, timestamp anomalies, overlong subtitle cues, duplicate segments, low-confidence segments and the suspicious
medical ASR errors the pipeline listed. Never writes into the folder; prints a JSON document when ``--json`` is given.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.vocabulary import normalize_key, tokenize          # noqa: E402

MAX_CUE_SECONDS = 8.0
MAX_CUE_LINES = 2
MAX_CUE_CHARS_PER_LINE = 48
WINDOW_SECONDS = 600.0


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def clock(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600:d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def window_report(segments: list[dict], lo: float, hi: float) -> dict[str, Any]:
    inside = [s for s in segments if lo <= (s["start"] + s["end"]) / 2 < hi]
    words = sum(len(tokenize(s["text"])) for s in inside)
    span = max(hi - lo, 1e-9)
    speech = sum(max(0.0, min(s["end"], hi) - max(s["start"], lo)) for s in inside)
    return {"from": clock(lo), "to": clock(hi), "segments": len(inside), "words": words, "words_per_minute": round(words / span * 60, 1),
            "covered_seconds": round(speech, 1)}


def parse_srt(text: str) -> list[dict[str, Any]]:
    cues = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = block.splitlines()
        if len(lines) < 3 or "-->" not in lines[1]:
            continue
        a, b = (t.strip() for t in lines[1].split("-->"))

        def sec(t: str) -> float:
            h, m, rest = t.replace(",", ".").split(":")
            return int(h) * 3600 + int(m) * 60 + float(rest)

        cues.append({"index": lines[0], "start": sec(a), "end": sec(b), "lines": lines[2:]})
    return cues


def hotword_echo(segments: list[dict], hotwords: list[str]) -> list[dict[str, Any]]:
    """Segments that read like the hotword list itself: 3+ different hotwords in one segment, or the same hotword 3+ times."""
    keys = [(h, normalize_key(h)) for h in hotwords if len(h) > 3]
    flagged = []
    for i, seg in enumerate(segments):
        text = normalize_key(seg["text"])
        found = [h for h, k in keys if k and k in text]
        repeats = max((text.count(k) for _h, k in keys if k), default=0)
        if len(found) >= 4 or repeats >= 3:
            flagged.append({"segment": i, "start": clock(seg["start"]), "hotwords": found[:6], "max_repeats": repeats, "text": seg["text"].strip()[:140]})
    return flagged


def audit(folder: Path) -> dict[str, Any]:
    data = read_json(folder / "local_transcript.json")
    segments = data["segments"]
    meta = read_json(folder / "metadata.json")
    quality = read_json(folder / "quality_report.json")
    uncertain = read_json(folder / "uncertain_segments.json")
    duration = float(meta["timing"].get("audio_duration_total") or quality["summary"].get("audio_seconds") or 0.0)
    middle = max(0.0, duration / 2 - WINDOW_SECONDS / 2)
    windows = {"beginning": window_report(segments, 0.0, min(WINDOW_SECONDS, duration)),
               "middle": window_report(segments, middle, min(middle + WINDOW_SECONDS, duration)),
               "end": window_report(segments, max(0.0, duration - WINDOW_SECONDS), duration + 1)}
    cues = parse_srt((folder / "local_transcript.srt").read_text(encoding="utf-8"))
    overlong = [{"index": c["index"], "start": clock(c["start"]), "seconds": round(c["end"] - c["start"], 1), "lines": len(c["lines"]),
                 "longest_line": max((len(line) for line in c["lines"]), default=0)} for c in cues
                if c["end"] - c["start"] > MAX_CUE_SECONDS or len(c["lines"]) > MAX_CUE_LINES
                or any(len(line) > MAX_CUE_CHARS_PER_LINE for line in c["lines"])]
    keys = [normalize_key(s["text"]) for s in segments]
    duplicates = [{"segment": i, "start": clock(segments[i]["start"]), "text": segments[i]["text"].strip()[:100]}
                  for i in range(1, len(segments)) if keys[i] and len(tokenize(segments[i]["text"])) >= 4 and keys[i] == keys[i - 1]]
    ordering = [{"segment": i, "start": segments[i]["start"], "previous_end": segments[i - 1]["end"]} for i in range(1, len(segments))
                if segments[i]["start"] < segments[i - 1]["end"] - 0.6]
    flag_codes = Counter(f["code"] for f in uncertain.get("segments", uncertain) if isinstance(f, dict) and "code" in f) \
        if isinstance(uncertain, dict) else Counter()
    possible = (folder / "possible_medical_errors.txt").read_text(encoding="utf-8")
    hotwords = (meta.get("subject") or {}).get("hotwords_applied") or []
    summary = quality["summary"]
    return {
        "windows": windows, "summary": summary, "missing_speech": quality.get("missing_speech", []),
        "gap_repairs": (meta.get("long_audio") or {}).get("gap_repairs", []), "loop_repairs": (meta.get("long_audio") or {}).get("loop_repairs", []),
        "hallucination_flags": quality.get("hallucination_flags", []), "repetition_flags": quality.get("repetition_flags", []),
        "timestamp_issues": quality.get("timestamp_issues", []), "hotwords_applied": hotwords,
        "hotword_echo": hotword_echo(segments, hotwords), "overlong_cues": overlong, "cue_count": len(cues),
        "duplicate_segments": duplicates, "out_of_order_segments": ordering, "uncertainty": meta.get("uncertainty", {}),
        "uncertain_flag_codes": dict(flag_codes), "possible_medical_error_entries": possible.count("Original phrase:"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = audit(args.folder)
    if args.json:
        print(json.dumps(report, indent=1, ensure_ascii=False))
    else:
        for name, w in report["windows"].items():
            print(f"{name:<10} {w['from']} - {w['to']}: {w['segments']} segments, {w['words']} words ({w['words_per_minute']} wpm)")
        print("summary:", report["summary"])
        for key in ("missing_speech", "gap_repairs", "loop_repairs", "hallucination_flags", "repetition_flags", "timestamp_issues",
                    "hotword_echo", "overlong_cues", "duplicate_segments", "out_of_order_segments"):
            print(f"{key}: {len(report[key])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
