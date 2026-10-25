"""A/B benchmark: previous behaviour (A) vs Medical Expert Edition (B) on the same short medical audio.

Setups (all use the same large-v3 weights, the same GPU settings, ONE loaded model):
  baseline   no hotwords at all (reference point)
  A_default  previous default : legacy word-list context, --subject general_medical, no probe, 128-token hotword budget
  A_subject  previous best    : legacy context with the correct --subject, 128-token budget
  B_auto     expert default   : file name + 40 % probe -> 19-area classifier -> term priority engine, no --subject
  B_subject  expert + subject : same, with the correct --subject

Metrics per run: WER, medical-term recall / precision (dictionary terms), hallucination + repetition flags,
decode time, peak dedicated VRAM. Then the quality gate (src/gate.py) judges A_default -> B_auto and A_subject -> B_subject.

Usage
  python scripts/make_benchmark_audio.py          # once
  python scripts/ab_benchmark.py [--samples neuro,biochem,pharma] [--conditions clean,hard] [--setups ...]
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_transcription_config  # noqa: E402
from src.exporters import write_json, write_text  # noqa: E402
from src.gate import aggregate, evaluate_gate  # noqa: E402
from src.metrics import medical_term_scores, word_error_rate  # noqa: E402
from src.model_manager import get_model_spec, load_model_specs  # noqa: E402
from src.pipeline import RunOptions, process_media  # noqa: E402
from src.platform_utils import configure_utf8_console  # noqa: E402
from src.transcriber import Transcriber  # noqa: E402
from src.vocabulary import build_index  # noqa: E402

AUDIO = ROOT / "benchmarks" / "audio"
NEUTRAL_NAME = {"neuro": "lecture1", "biochem": "lecture2", "pharma": "lecture3"}     # the file name must not give the subject away
SUBJECT = {"neuro": "neuroscience", "biochem": "biochemistry", "pharma": "pharmacology"}
LEGACY_HOTWORD_TOKENS = 128                                                        # the previous config value
SETUPS: dict[str, dict[str, Any]] = {
    "baseline": dict(engine="expert", subject=False, auto_topic=False, hotwords=False),
    "A_default": dict(engine="legacy", subject=False, auto_topic=False, hotwords=True),
    "A_subject": dict(engine="legacy", subject=True, auto_topic=False, hotwords=True),
    "B_auto": dict(engine="expert", subject=False, auto_topic=True, hotwords=True),
    "B_subject": dict(engine="expert", subject=True, auto_topic=True, hotwords=True),
}
PAIRS = (("A_default", "B_auto"), ("A_subject", "B_subject"))
log = logging.getLogger("ab_benchmark")


def sample_meta(sample: str) -> dict[str, Any]:
    """Subject and neutral file name of a sample: the fixed passages above, or ``<sample>.meta.json`` (random-vocabulary passages)."""
    meta_file = AUDIO / f"{sample}.meta.json"
    if meta_file.is_file():
        return json.loads(meta_file.read_text(encoding="utf-8"))
    return {"subject": SUBJECT[sample], "neutral": NEUTRAL_NAME[sample]}


def setup_config(cfg: dict[str, Any], setup: dict[str, Any]) -> dict[str, Any]:
    variant = copy.deepcopy(cfg)
    if setup["engine"] == "legacy":                       # reproduce the previous budget: one 128-token hotword list
        variant["context"]["hotwords_token_budget"] = LEGACY_HOTWORD_TOKENS
        variant["context"]["prompt_token_budget"] = LEGACY_HOTWORD_TOKENS
    return variant


def run_one(name: str, sample: str, condition: str, cfg: dict[str, Any], transcriber: Transcriber, spec: Any,
            args: argparse.Namespace, vocab: Any, reference: str, runs_dir: Path) -> dict[str, Any]:
    setup = SETUPS[name]
    wav = AUDIO / f"{sample}_{condition}.wav"
    media_dir = runs_dir / "_media"
    media_dir.mkdir(parents=True, exist_ok=True)
    media = media_dir / f"{sample_meta(sample)['neutral']}_{condition}.wav"
    shutil.copyfile(wav, media)
    options = RunOptions(
        mode=args.mode, language="en", subjects=[sample_meta(sample)["subject"]] if setup["subject"] else ["general_medical"],
        outputs_root=runs_dir, use_hotwords=setup["hotwords"], auto_topic=setup["auto_topic"],
        context_engine=setup["engine"], reuse_audio=wav, resume=False, quality_audit=True, use_pdf=False,
        device=args.device)
    started = time.perf_counter()
    summary = process_media(media, options, setup_config(cfg, setup), transcriber, spec, output_name=f"{name}__{sample}_{condition}")
    out = runs_dir / f"{name}__{sample}_{condition}"
    raw = (out / "local_raw_transcript.txt").read_text(encoding="utf-8")
    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    terms = medical_term_scores(reference, raw, vocab)
    row = {
        "setup": name, "sample": sample, "condition": condition, "wer": word_error_rate(reference, raw, "en"),
        "recall": terms["recall"], "precision": terms["precision"], "matched_terms": terms["matched_terms"],
        "reference_terms": terms["reference_terms"], "hypothesis_terms": terms["hypothesis_terms"],
        "missed": terms["missed"], "extra": terms["extra"],
        "hallucinations": summary.hallucination_flags, "repetitions": summary.repetition_flags,
        "uncertain": summary.uncertain, "decode_seconds": meta["timing"]["decode_seconds"],
        "wall_seconds": round(time.perf_counter() - started, 1), "peak_dedicated_mib": summary.peak_dedicated_mib,
        "hotwords_applied": len(meta["subject"].get("hotwords_applied") or []),
        "areas": meta["subject"].get("subjects"), "transcript": raw.strip(),
    }
    log.info("%-9s %-7s %-5s WER %5.1f%% | recall %s | precision %s | flags %d | %.0fs", name, sample, condition,
             100 * row["wer"], "n/a" if row["recall"] is None else f"{row['recall']:.2f}",
             "n/a" if row["precision"] is None else f"{row['precision']:.2f}", row["hallucinations"] + row["repetitions"],
             row["decode_seconds"])
    return row


def summarise(rows: list[dict[str, Any]], setups: list[str], thresholds: dict[str, float]) -> dict[str, Any]:
    result: dict[str, Any] = {"per_setup": {}, "gate": {}}
    for name in setups:
        mine = [r for r in rows if r["setup"] == name]
        if mine:
            total = aggregate(mine)
            total.update(runs=len(mine), decode_seconds=round(sum(r["decode_seconds"] for r in mine), 1),
                         peak_dedicated_mib=max((r["peak_dedicated_mib"] or 0) for r in mine))
            result["per_setup"][name] = total
    for before, after in PAIRS:
        if before in result["per_setup"] and after in result["per_setup"]:
            for scope in ("all", "clean", "hard"):
                pick = (lambda n: [r for r in rows if r["setup"] == n and (scope == "all" or r["condition"] == scope)])
                if pick(before) and pick(after):
                    verdict = evaluate_gate(aggregate(pick(before)), aggregate(pick(after)), thresholds)
                    result["gate"][f"{before}->{after} [{scope}]"] = verdict.to_dict()
    return result


def to_markdown(rows: list[dict[str, Any]], summary: dict[str, Any], setups: list[str]) -> str:
    out = ["# A/B benchmark - previous (A) vs Medical Expert Edition (B)", "",
           "TTS speech (Windows voice), English, large-v3 float16 beam 5. 'hard' = faster speech + noise (SNR 8 dB).", "",
           "## Totals per setup", "", "| setup | WER | term recall | term precision | halluc./loops | decode s | peak VRAM MiB |",
           "|---|---|---|---|---|---|---|"]
    for name in setups:
        s = summary["per_setup"].get(name)
        if s:
            out.append(f"| {name} | {100 * s['wer']:.2f}% | {s['recall']:.3f} | {s['precision']:.3f} | "
                       f"{s['hallucinations'] + s['repetitions']} | {s['decode_seconds']} | {s['peak_dedicated_mib']} |")
    out += ["", "## Quality gate", "", "| comparison | verdict | rule | WER delta (pts) | recall delta (pts) |", "|---|---|---|---|---|"]
    for key, g in summary["gate"].items():
        out.append(f"| {key} | **{g['verdict']}** | {g['rule']} | {g['wer_delta_points']:+.2f} | {g['recall_delta_points']:+.1f} |")
    out += ["", "## Every run", "", "| setup | sample | cond | WER | recall | precision | flags | missed terms |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        rec = "n/a" if r["recall"] is None else f"{r['recall']:.2f}"
        pre = "n/a" if r["precision"] is None else f"{r['precision']:.2f}"
        out.append(f"| {r['setup']} | {r['sample']} | {r['condition']} | {100 * r['wer']:.1f}% | {rec} | {pre} | "
                   f"{r['hallucinations'] + r['repetitions']} | {', '.join(r['missed']) or '-'} |")
    return "\n".join(out) + "\n"


def main() -> int:
    configure_utf8_console()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--samples", default="neuro,biochem,pharma")
    parser.add_argument("--conditions", default="clean,hard")
    parser.add_argument("--setups", default=",".join(SETUPS))
    parser.add_argument("--mode", default="quality", choices=("quality", "fast"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpcore", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("src").setLevel(logging.WARNING)            # only this script's own lines on the console

    samples, conditions = args.samples.split(","), args.conditions.split(",")
    setups = [s for s in args.setups.split(",") if s in SETUPS]
    missing = [f"{s}_{c}.wav" for s in samples for c in conditions if not (AUDIO / f"{s}_{c}.wav").is_file()]
    if missing:
        print(f"Missing audio: {missing}. Run: python scripts/make_benchmark_audio.py", file=sys.stderr)
        return 2
    cfg = load_transcription_config()
    spec = get_model_spec("default", load_model_specs())
    transcriber = Transcriber(spec, cfg[args.mode], args.device, 0, False)
    vocab = build_index("all", "en")
    out_dir = args.out or ROOT / "benchmarks" / "results" / datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    try:
        for sample in samples:
            reference = (AUDIO / f"{sample}.txt").read_text(encoding="utf-8")
            for condition in conditions:
                for name in setups:
                    rows.append(run_one(name, sample, condition, cfg, transcriber, spec, args, vocab, reference, runs_dir))
                    write_json(out_dir / "results_partial.json", rows)
    finally:
        transcriber.close()
    summary = summarise(rows, setups, cfg["quality_gate"])
    write_json(out_dir / "results.json", {"rows": rows, "summary": summary, "thresholds": cfg["quality_gate"]})
    write_text(out_dir / "results.md", to_markdown(rows, summary, setups))
    print(to_markdown(rows, summary, setups))
    print(f"Saved to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
