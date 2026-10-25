"""Render the v4 benchmark corpus (benchmarks/v4/corpus/*.json) to speech with the built-in Windows voice.

    python tools/make_v4_corpus.py [--force] [--only neuro1,neg3]

Two conditions per passage (like benchmarks/audio): ``clean`` (normal rate) and ``hard`` (faster speech + white noise, SNR 8 dB, seeded).
Writes ``benchmarks/v4/corpus/audio/<id>_<condition>.wav`` (git-ignored) and the reference text ``<id>.txt`` next to it.
SYNTHETIC speech: it measures what the engines do with real recogniser errors on KNOWN text; it is never reported as a real-lecture score.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "benchmarks" / "v4" / "corpus"
AUDIO = CORPUS / "audio"


def load_tts():
    spec = importlib.util.spec_from_file_location("make_benchmark_audio", ROOT / "scripts" / "make_benchmark_audio.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def passages() -> list[dict]:
    """Every passage of every set, in a stable order (the noise seed of a clip depends on its position)."""
    out = []
    for name in ("passages.json", "negative.json", "heldout.json", "validation.json", "heldout_v2.json"):
        out += json.loads((CORPUS / name).read_text(encoding="utf-8"))["passages"]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default="")
    a = ap.parse_args(argv)
    tts = load_tts()
    AUDIO.mkdir(parents=True, exist_ok=True)
    wanted = {x for x in a.only.split(",") if x}
    for index, p in enumerate(passages()):
        if wanted and p["id"] not in wanted:
            continue
        (AUDIO / f"{p['id']}.txt").write_text(p["text"] + "\n", encoding="utf-8", newline="\n")
        for condition, rate, snr in (("clean", 0, None), ("hard", 3, 8.0)):
            wav = AUDIO / f"{p['id']}_{condition}.wav"
            if wav.exists() and not a.force:
                continue
            tts.speak(p["text"].encode("ascii", "replace").decode("ascii"), wav, rate)
            tts.to_16k_mono(wav)
            if snr is not None:
                tts.add_noise(wav, snr, seed=5000 + index)
            print("created", wav.name, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
