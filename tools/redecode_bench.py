"""Targeted re-decode WITH vs WITHOUT, on one corpus set: precision gain, recall gain and latency cost.

    python tools/redecode_bench.py --set validation --runs auto=.../runs_validation/auto --runs nohot=.../runs_validation/nohot --kb data/medical_knowledge_public.db --out out.json

Only the MEDIUM proposals of the current gate are re-decoded (the HIGH tier is not touched, the LOW tier never is). The speech model is loaded once; for each MEDIUM proposal only
the audio window around the span is decoded again, with the proposed term as the only hotword. Compared on the same clips:

    high             v4 HIGH only (the shipping candidate)
    high+redecode    v4 HIGH + the MEDIUM proposals the audio confirmed
    high+medium      v4 HIGH + every MEDIUM proposal (an upper bound on recall, what re-decode must not become)

Re-decode must not be used to hide a weak scorer: the numbers below are the effect on MEDIUM proposals only, and the cost is reported as wall-clock seconds.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from v4_lib import ROOT, apply_items, load_clips, _summary  # noqa: E402

from src.config import load_transcription_config  # noqa: E402
from src.medical_knowledge.benchmarks.evaluate import evaluate  # noqa: E402
from src.medical_knowledge.runtime.engine import V4Engine  # noqa: E402
from src.medical_knowledge.runtime.knowledge import MedicalKnowledge  # noqa: E402
from src.medical_knowledge.runtime.redecode import TargetedRedecoder, whisper_window_decoder  # noqa: E402
from src.model_manager import get_model_spec, load_model_specs  # noqa: E402
from src.transcriber import Transcriber  # noqa: E402

AUDIO = ROOT / "benchmarks" / "v4" / "corpus" / "audio"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=["dev", "validation", "heldout_v2"])
    ap.add_argument("--runs", action="append", required=True)
    ap.add_argument("--kb", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    cfg = load_transcription_config()
    spec = get_model_spec("default", load_model_specs())
    transcriber = Transcriber(spec, cfg["quality"], a.device, 0, False)
    engine = V4Engine(MedicalKnowledge.open(a.kb))
    clips = load_clips(engine, a.set, [(s.split("=", 1)[0], Path(s.split("=", 1)[1])) for s in a.runs])
    evals = defaultdict(list)
    outcomes = []
    windows = 0
    decode_seconds = 0.0
    audio_seconds = 0.0
    words = 0
    load_seconds = 0.0
    try:
        for clip in clips:
            result = engine.finish(clip.prepared, None, extras=False)
            wav = AUDIO / f"{clip.name}.wav"
            promoted = []
            if result.uncertain and wav.is_file():
                started = time.perf_counter()
                decode = whisper_window_decoder(transcriber, wav, cfg["quality"], cfg["decoding"], cfg["vad"])
                report = TargetedRedecoder(decode).run(result, clip.segs)
                spent = time.perf_counter() - started
                if windows == 0:
                    load_seconds = spent                                   # the first call includes loading the model: reported separately, not as per-window latency
                else:
                    decode_seconds += spent
                windows += len(report.outcomes)
                audio_seconds += report.seconds_used
                promoted = report.promoted()
                outcomes += [o.to_dict() | {"clip": clip.name} for o in report.outcomes]
            words += len(clip.gold.split())
            terms = clip.passage.get("terms", [])
            for name, items in (("high", result.corrections), ("high+redecode", list(result.corrections) + promoted), ("high+medium", list(result.corrections) + list(result.uncertain))):
                evals[name].append(evaluate(clip.gold, clip.raw, " ".join(apply_items(clip.texts, items)), terms))
            print(f"{clip.name}: {len(result.uncertain)} medium, {len(promoted)} confirmed", flush=True)
    finally:
        transcriber.close()
    verdicts: dict[str, int] = defaultdict(int)
    for o in outcomes:
        verdicts[o["verdict"]] += 1
    summary = {name: _summary(e) for name, e in evals.items()}
    per_window = decode_seconds / max(1, windows - 1)
    latency = {"windows": windows, "audio_seconds_decoded": round(audio_seconds, 1), "first_call_seconds_including_model_load": round(load_seconds, 1),
               "seconds_per_window": round(per_window, 2), "extra_seconds_per_1000_words": round(1000 * windows * per_window / max(1, words), 1), "gold_words": words}
    out = {"set": a.set, "summary": summary, "verdicts": dict(verdicts), "latency": latency, "outcomes": [{k: o[k] for k in ("clip", "original", "replacement", "verdict", "why")} for o in outcomes]}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n== re-decode on corpus set: {a.set}")
    for name, s in summary.items():
        p = "-" if s["precision"] is None else f"{100 * s['precision']:.1f}%"
        r = "-" if s["medical_recall"] is None else f"{100 * s['medical_recall']:.1f}%"
        print(f"  {name:14} corrected {s['corrected']:3} wrong {s['wrong']:2} false {s['false']:2} missed {s['missed']:3} | precision {p:>6} recall {r:>6}")
    print("  verdicts:", dict(verdicts), "| latency:", latency)
    return 0


if __name__ == "__main__":
    sys.exit(main())
