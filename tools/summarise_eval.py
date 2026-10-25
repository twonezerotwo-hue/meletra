"""Turn the result files of ``tools/eval_multilingual.py`` into the tables of docs/v4_1_1/MULTILINGUAL_VALIDATION.md (Markdown on stdout).

    python tools/summarise_eval.py benchmarks/multilingual/results_final2 [--stale benchmarks/multilingual/results_stale] [--video benchmarks/multilingual/results_final_video2]

Only numbers that are in the result files are printed (the code version each was produced with is a column of its own, so a stale file is visible as such). A recording without a human
reference has no error rate and no language accuracy: the columns say "-", never an invented number.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional


def load(folder: Path) -> dict[str, dict[str, Any]]:
    out = {}
    for path in sorted(folder.glob("*.json")):
        data = json.loads(path.read_bytes().decode("utf-8"))
        out[data["name"]] = data
    return out


def pct(value: Optional[float]) -> str:
    return "-" if value is None else f"{100 * value:.1f} %"


def rates(score: Optional[dict[str, Any]]) -> str:
    if not score:
        return "-"
    return ", ".join(f"{lang} {100 * rate:.1f} %" for lang, rate in sorted(score.get("mean_error_rate_by_language", {}).items())) or "-"


def row(data: dict[str, Any], stale: Optional[dict[str, Any]]) -> str:
    score = data.get("score") or {}
    truth = ", ".join(data.get("truth_languages") or []) or "-"
    found = ", ".join(data.get("languages_in_transcript") or []) or "-"
    cov = data.get("coverage") or {}
    before = pct(stale["score"]["language_accuracy"]) if stale and stale.get("score") else "-"
    return (f"| {data['name']} | {truth} | {found} | {pct(score.get('language_accuracy'))} | {before} | {pct(score.get('clips_written'))} | {rates(score)} | "
            f"{pct(cov.get('covered_share_after'))} | {data['audio_seconds'] or '-'} | {data['wall_seconds']:.0f} | {data['peak_working_set_mib_so_far']:.0f} | `{data['code_version']}` |")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("--stale", type=Path)
    ap.add_argument("--video", type=Path)
    args = ap.parse_args(argv)
    fresh = load(args.results)
    stale = load(args.stale) if args.stale and args.stale.is_dir() else {}
    print("| recording | spoken | found | language accuracy (per sentence) | before (earlier code) | sentences written | WER / CER by language | audible time with words | audio s | run s (CPU) | peak MiB | code |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, data in fresh.items():
        print(row(data, stale.get(name)))
    if args.video and args.video.is_dir():
        for name, data in load(args.video).items():
            cov, plan = data.get("coverage") or {}, data.get("plan") or {}
            print(f"\nreal video `{name}` (no reference text: no error rate): languages {data.get('languages_in_transcript')}, plan {plan.get('mode')} {plan.get('languages')}, segments {data.get('segments')}, "
                  f"audible time with words {pct(cov.get('covered_share_before'))} -> {pct(cov.get('covered_share_after'))}, gaps recovered {cov.get('gaps_recovered')}, unreadable {cov.get('gaps_unreadable')}, "
                  f"skipped {cov.get('gaps_skipped')}, run {data['wall_seconds']:.0f} s, peak {data['peak_working_set_mib_so_far']:.0f} MiB, code `{data['code_version']}`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
