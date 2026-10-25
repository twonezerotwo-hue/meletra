"""Collect the model's own evidence for candidate languages on audio windows, to calibrate / check the language verification score offline.

    python tools/collect_language_evidence.py --out FILE.json  fixtures      windows of the authorized human-read recordings (the language is KNOWN)
    python tools/collect_language_evidence.py --out FILE.json  video WAV     windows every 40 s of a recording (the language is unknown: statistics only)

For every window and every candidate language the forced-language decode (24 tokens) is recorded: text, average log-probability, no-speech probability, token count. The files stay in
the git-ignored benchmarks folder; text of a private recording is never printed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

POOL = ["en", "ko", "hi", "zh", "tr", "de"]
FIXTURES = {"ko_only": "ko", "hi_only": "hi", "en_only": "en", "tr_only": "tr", "de_only": "de", "zh_only": "zh", "english_with_brief_ko_hi": None, "en_ko_en": None}


def load_transcriber(device: str):
    import os

    from src.model_manager import get_model_spec, load_model_specs
    from src.services.model_service import ModelService
    from src.services.paths import get_paths
    from src.transcriber import Transcriber
    from dataclasses import replace

    paths = get_paths()
    models = ModelService(paths)
    spec = models.model_spec("whisper-large-v3")
    from src.config import load_transcription_config

    cfg = load_transcription_config()
    transcriber = Transcriber(spec, cfg["quality"], device, 0, False)
    transcriber._ensure_loaded()
    return transcriber


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["fixtures", "video"])
    ap.add_argument("wav", nargs="?", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--step", type=float, default=40.0)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--languages", default="en,ko,hi,zh")
    args = ap.parse_args(argv)
    transcriber = load_transcriber(args.device)
    rows = []
    if args.mode == "fixtures":
        folder = ROOT / "benchmarks" / "multilingual" / "recordings"
        for name in FIXTURES:
            truth = json.loads((folder / f"{name}.truth.json").read_bytes().decode("utf-8"))
            import numpy as np
            import soundfile as sf

            x, rate = sf.read(str(folder / f"{name}.wav"), dtype="float32")
            for clip in truth["timeline"]:
                seg = x[int(clip["start"] * rate):int(clip["end"] * rate)]
                level = float(np.sqrt((seg ** 2).mean()))
                if level < 0.01 or clip["end"] - clip["start"] < 8.0:                     # only audible clips (the quiet ones are the leveler's business)
                    continue
                start = clip["start"]
                evidence = transcriber.score_languages(folder / f"{name}.wav", start, min(args.seconds, clip["end"] - clip["start"]), POOL, 24)
                rows.append({"recording": name, "start": start, "truth": clip["language"], "evidence": evidence})
                print(name, round(start, 1), clip["language"], {l: round(e["avg_logprob"], 2) for l, e in evidence.items()}, flush=True)
                if sum(1 for r in rows if r["recording"] == name) >= 3:
                    break
    else:
        import soundfile as sf

        info = sf.info(str(args.wav))
        start = 0.0
        while start + 5 < info.duration:
            evidence = transcriber.score_languages(args.wav, start, min(args.seconds, info.duration - start), args.languages.split(","), 24)
            rows.append({"recording": args.wav.name, "start": start, "truth": None, "evidence": evidence})
            print(round(start), {l: (round(e["avg_logprob"], 2), e["tokens"]) for l, e in evidence.items()}, flush=True)
            start += args.step
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
