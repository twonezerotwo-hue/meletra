"""Transcribe the v4 benchmark corpus with the REAL pipeline (Whisper large-v3 + context engine + production Medical Expert) - one loaded model, many clips.

    python tools/run_v4_corpus.py [--setups auto,nohot] [--conditions clean,hard] [--only neuro1,neg3] [--out DIR]

Setups: ``auto``  production default (file-name + early-transcript context, hotwords, auto topic - the "B_auto" setup of scripts/ab_benchmark.py)
        ``nohot`` the same without hotwords (more recogniser errors: a stress test for the correction layers)
Each run folder holds local_raw_transcript.txt, local_transcript.json (with word probabilities) and the production medical_corrected_* files.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from corpus_sets import passages_of  # noqa: E402
from src.config import load_transcription_config  # noqa: E402
from src.model_manager import get_model_spec, load_model_specs  # noqa: E402
from src.pipeline import RunOptions, process_media  # noqa: E402
from src.platform_utils import configure_utf8_console  # noqa: E402
from src.transcriber import Transcriber  # noqa: E402

AUDIO = ROOT / "benchmarks" / "v4" / "corpus" / "audio"
CORPUS = ROOT / "benchmarks" / "v4" / "corpus"
#: heldout_v3 / v4 / v5: three voices / speeds / codecs per base passage (<set>_plan.json)
PLANNED = {name: (CORPUS / f"audio_{name[-2:]}", CORPUS / f"{name}_plan.json") for name in ("heldout_v3", "heldout_v4", "heldout_v5")}
SETUPS = {"auto": dict(hotwords=True, auto_topic=True), "nohot": dict(hotwords=False, auto_topic=False)}


def main(argv=None) -> int:
    configure_utf8_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--setups", default="auto")
    ap.add_argument("--conditions", default="clean,hard")
    ap.add_argument("--only", default="")
    ap.add_argument("--set", default="all", help="transcribe only the clips of one corpus set: dev | validation | heldout_v2 | heldout_v3 | heldout_v4 | heldout_v5 | all")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=ROOT / "benchmarks" / "v4" / "results_local" / "corpus_runs")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    cfg = load_transcription_config()
    spec = get_model_spec("default", load_model_specs())
    transcriber = Transcriber(spec, cfg["quality"], a.device, 0, False)
    wanted = {x for x in a.only.split(",") if x}
    in_set = set(passages_of(a.set))
    if a.set in PLANNED:
        audio_dir, plan_file = PLANNED[a.set]
        plan = json.loads(plan_file.read_text(encoding="utf-8"))["clips"]
        jobs = [(pid, variant) for pid in sorted(plan) for variant in plan[pid]]
    else:
        audio_dir = AUDIO
        ids = sorted({p.stem.rsplit("_", 1)[0] for p in AUDIO.glob("*_clean.wav")} & in_set)
        jobs = [(pid, cond) for pid in ids for cond in a.conditions.split(",")]
    started = time.time()
    done = 0
    try:
        for setup in a.setups.split(","):
            runs = a.out / setup
            media_dir = runs / "_media"
            media_dir.mkdir(parents=True, exist_ok=True)
            for pid, cond in jobs:
                if wanted and pid not in wanted:
                    continue
                name = f"{pid}_{cond}"
                if (runs / name / "local_transcript.json").is_file():
                    continue
                media = media_dir / f"lecture_{abs(hash(name)) % 10**6}.wav"                 # a neutral file name: it must not give the subject away
                shutil.copyfile(audio_dir / f"{name}.wav", media)
                options = RunOptions(mode="quality", language="en", subjects=["general_medical"], outputs_root=runs, use_hotwords=SETUPS[setup]["hotwords"],
                                     auto_topic=SETUPS[setup]["auto_topic"], context_engine="expert", reuse_audio=audio_dir / f"{name}.wav", resume=False,
                                     quality_audit=True, use_pdf=False, device=a.device)
                process_media(media, options, copy.deepcopy(cfg), transcriber, spec, output_name=name)
                done += 1
                print(f"[{setup}] {name} done ({done}, {time.time() - started:.0f} s)", flush=True)
    finally:
        transcriber.close()
    print("CORPUS_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
