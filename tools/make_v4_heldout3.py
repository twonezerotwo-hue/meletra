"""Render the v4 heldout_v3 corpus to speech: several synthetic voices x speeds x codecs x noise conditions.

    python tools/make_v4_heldout3.py --plan            # write the variant plan (benchmarks/v4/corpus/heldout_v3_plan.json) - deterministic from the seed, committed BEFORE any audio exists
    python tools/make_v4_heldout3.py [--only id1,id2] [--force]

Every base passage gets THREE clips: one Microsoft Zira (SAPI) clip, one Piper US-English clip and one Piper UK-English clip, each with its own speed / codec / noise
condition. The plan is a pure function of ``SEED`` and the passage ids, so it can be reproduced and audited. All speech is SYNTHETIC: heldout_v3 measures what the engines do with
real recogniser errors on KNOWN text; it is never reported as a real-lecture score (the real recordings are a separate, human-reviewed benchmark).
Writes ``benchmarks/v4/corpus/audio_v3/<id>_<variant>.wav`` (git-ignored) and the reference text ``<id>.txt`` next to it.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "benchmarks" / "v4" / "corpus"
AUDIO = CORPUS / "audio_v3"
VOICES = ROOT / "data" / "tts_voices"
PLAN = CORPUS / "heldout_v3_plan.json"
SEED = 20261014
SR = 16000

#: variant -> (family, description)
VARIANTS = {
    "zc": ("zira", "Microsoft Zira (SAPI), normal rate, clean"),
    "zn": ("zira", "Microsoft Zira (SAPI), faster rate, white noise at 8 dB SNR"),
    "lc": ("us", "Piper en_US lessac, normal rate, clean"),
    "rs": ("us", "Piper en_US ryan, slow (length scale 1.25)"),
    "lp": ("us", "Piper en_US lessac, +6 % speed and pitch, narrow-band AMR-NB telephone codec (8 kHz)"),
    "rr": ("us", "Piper en_US ryan, synthetic room reverberation (RT60 about 0.4 s)"),
    "af": ("gb", "Piper en_GB alan, fast (length scale 0.85), white noise at 14 dB SNR"),
    "nm": ("gb", "Piper en_GB northern_english_male, MP3 24 kbit/s round trip"),
}
VOICE_OF = {"lc": "en_US-lessac-medium", "rs": "en_US-ryan-medium", "lp": "en_US-lessac-medium", "rr": "en_US-ryan-medium", "af": "en_GB-alan-medium",
            "nm": "en_GB-northern_english_male-medium"}


def passages() -> list[dict]:
    return json.loads((CORPUS / "heldout_v3.json").read_text(encoding="utf-8"))["passages"]


def make_plan() -> dict:
    rng = random.Random(SEED)
    clips = {}
    for p in sorted(passages(), key=lambda q: q["id"]):
        clips[p["id"]] = [rng.choice([v for v, (f, _d) in VARIANTS.items() if f == family]) for family in ("zira", "us", "gb")]
    return {"seed": SEED, "variants": {v: d for v, (_f, d) in VARIANTS.items()}, "clips": clips,
            "note": "synthetic speech: three clips per base passage (Zira, Piper US, Piper UK); base-passage metrics are primary, clip metrics secondary"}


# ------------------------------------------------------------------------------------------------------------------ signal helpers (numpy only)
def resample(x: np.ndarray, source: int, target: int) -> np.ndarray:
    if source == target:
        return x.astype(np.float32)
    n = int(round(len(x) * target / source))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def band_limit(x: np.ndarray, rate: int, low: float, high: float) -> np.ndarray:
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1.0 / rate)
    spectrum[(freqs < low) | (freqs > high)] = 0
    return np.fft.irfft(spectrum, len(x)).astype(np.float32)


def add_noise(x: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    power = float(np.mean(x ** 2)) or 1e-8
    return np.clip(x + rng.normal(0.0, np.sqrt(power / (10 ** (snr_db / 10))), size=x.shape), -1.0, 1.0).astype(np.float32)


def reverb(x: np.ndarray, rate: int, seed: int, rt60: float = 0.4, wet: float = 0.35) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(rate * rt60)
    ir = rng.normal(0.0, 1.0, n) * np.exp(-6.9 * np.arange(n) / n)                 # -60 dB at rt60
    ir /= np.sqrt(np.sum(ir ** 2))
    size = 1 << int(np.ceil(np.log2(len(x) + n)))
    wet_signal = np.fft.irfft(np.fft.rfft(x, size) * np.fft.rfft(ir, size), size)[:len(x)]
    out = (1 - wet) * x + wet * wet_signal * (np.std(x) / (np.std(wet_signal) or 1.0))
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def codec_round_trip(x: np.ndarray, rate: int, codec: str, ext: str, bitrate: int = 0) -> np.ndarray:
    """Encode with PyAV (libmp3lame / libopencore_amrnb ...) and decode again at 16 kHz: the artefacts of a real lossy file."""
    import av
    from faster_whisper.audio import decode_audio

    path = os.path.join(tempfile.mkdtemp(prefix="mlt_codec_"), "clip." + ext)
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with av.open(path, "w") as out:
        stream = out.add_stream(codec, rate=rate)
        stream.layout = "mono"
        if bitrate:
            stream.bit_rate = bitrate
        size = stream.codec_context.frame_size or 1024
        pos = 0
        while pos < len(pcm):
            chunk = pcm[pos:pos + size]
            if len(chunk) < size:
                chunk = np.pad(chunk, (0, size - len(chunk)))
            frame = av.AudioFrame.from_ndarray(chunk.reshape(1, -1), format="s16", layout="mono")
            frame.sample_rate = rate
            frame.pts = pos
            for packet in stream.encode(frame):
                out.mux(packet)
            pos += len(chunk)
        for packet in stream.encode(None):
            out.mux(packet)
    return decode_audio(path, sampling_rate=SR)


# ------------------------------------------------------------------------------------------------------------------ synthesis
_VOICES: dict = {}


def piper_samples(voice_name: str, text: str, length_scale: float) -> np.ndarray:
    from piper import PiperVoice, SynthesisConfig

    voice = _VOICES.get(voice_name)
    if voice is None:
        voice = _VOICES[voice_name] = PiperVoice.load(str(VOICES / f"{voice_name}.onnx"))
    with tempfile.TemporaryDirectory(prefix="mlt_piper_") as tmp:
        wav = Path(tmp) / "p.wav"
        with wave.open(str(wav), "wb") as handle:
            voice.synthesize_wav(text, handle, syn_config=SynthesisConfig(length_scale=length_scale))
        data, rate = sf.read(str(wav), dtype="float32")
    return resample(data if data.ndim == 1 else data.mean(axis=1), rate, SR)


def zira_samples(text: str, rate: int, tts) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="mlt_sapi_") as tmp:
        wav = Path(tmp) / "z.wav"
        tts.speak(text, wav, rate)
        tts.to_16k_mono(wav)
        data, _ = sf.read(str(wav), dtype="float32")
    return data


def render(variant: str, text: str, index: int, tts) -> np.ndarray:
    seed = SEED + index
    if variant == "zc":
        return zira_samples(text, 0, tts)
    if variant == "zn":
        return add_noise(zira_samples(text, 3, tts), 8.0, seed)
    name = VOICE_OF[variant]
    if variant == "lc":
        return piper_samples(name, text, 1.0)
    if variant == "rs":
        return piper_samples(name, text, 1.25)
    if variant == "rr":
        return reverb(piper_samples(name, text, 1.0), SR, seed)
    if variant == "af":
        return add_noise(piper_samples(name, text, 0.85), 14.0, seed)
    if variant == "nm":
        return codec_round_trip(piper_samples(name, text, 1.0), SR, "libmp3lame", "mp3", 24000)
    if variant == "lp":
        fast = resample(piper_samples(name, text, 1.0), SR, int(SR * 1.06))                  # 6 % faster and higher (resampled and played back at 16 kHz)
        narrow = resample(band_limit(fast, SR, 300.0, 3400.0), SR, 8000)
        return codec_round_trip(narrow, 8000, "libopencore_amrnb", "amr")
    raise ValueError(variant)


def load_tts():
    spec = importlib.util.spec_from_file_location("make_benchmark_audio", ROOT / "scripts" / "make_benchmark_audio.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true", help="only (re)write the variant plan")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default="")
    a = ap.parse_args(argv)
    if a.plan:
        PLAN.write_text(json.dumps(make_plan(), indent=1), encoding="utf-8")
        print(f"plan written: {PLAN}")
        return 0
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    assert plan == make_plan(), "the committed plan is not the plan of this seed / passage list"
    tts = load_tts()
    AUDIO.mkdir(parents=True, exist_ok=True)
    wanted = {x for x in a.only.split(",") if x}
    by_id = {p["id"]: p for p in passages()}
    for index, (pid, variants) in enumerate(plan["clips"].items()):
        if wanted and pid not in wanted:
            continue
        text = by_id[pid]["text"].encode("ascii", "replace").decode("ascii")
        (AUDIO / f"{pid}.txt").write_text(by_id[pid]["text"] + "\n", encoding="utf-8", newline="\n")
        for k, variant in enumerate(variants):
            wav = AUDIO / f"{pid}_{variant}.wav"
            if wav.exists() and not a.force:
                continue
            samples = render(variant, text, index * 3 + k, tts)
            sf.write(str(wav), np.clip(samples, -1.0, 1.0), SR, subtype="PCM_16")
            print("created", wav.name, f"{len(samples) / SR:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
