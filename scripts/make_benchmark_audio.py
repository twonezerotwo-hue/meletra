"""Create the A/B benchmark audio: three short English medical passages spoken by the built-in Windows voice.

For each passage two conditions are written to benchmarks/audio/:
  <name>_clean.wav   normal speaking rate
  <name>_hard.wav    faster speech + white noise (SNR ~8 dB, seeded), where biasing the decoder matters more
plus <name>.txt (the reference = exactly what was spoken).

Windows SAPI is driven through cscript (VBScript), so no extra Python packages and no PowerShell are needed.
Usage:  python scripts/make_benchmark_audio.py [--force]
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "benchmarks" / "audio"
SAMPLE_RATE = 16000

PASSAGES = {
    "neuro": (
        "Today we review the descending and ascending pathways of the central nervous system. "
        "The corticospinal tract originates in the primary motor cortex, descends through the internal capsule, "
        "and decussates in the medullary pyramids. "
        "Lesions of the substantia nigra pars compacta reduce dopamine in the striatum and cause Parkinson disease. "
        "The spinothalamic tract carries pain and temperature, while the medial lemniscus carries fine touch and "
        "proprioception. "
        "The periaqueductal gray modulates pain through the nucleus raphe magnus. "
        "The vestibulocochlear nerve carries hearing and balance, and the nucleus ambiguus supplies motor fibers to "
        "the vagus nerve. "
        "Damage to Broca area produces nonfluent aphasia, whereas damage to Wernicke area produces fluent aphasia."),
    "biochem": (
        "In glycolysis, glucose is converted to pyruvate. "
        "Hexokinase traps glucose inside the cell, and phosphofructokinase is the main rate limiting enzyme, "
        "inhibited by citrate and activated by fructose bisphosphate. "
        "Pyruvate dehydrogenase links glycolysis to the citric acid cycle, and its deficiency causes lactic acidosis. "
        "In the pentose phosphate pathway, glucose phosphate dehydrogenase generates NADPH, and its deficiency "
        "leads to hemolytic anemia after oxidative stress. "
        "Cystathionine beta synthase deficiency causes homocystinuria. "
        "HMG CoA reductase is the rate limiting step of cholesterol synthesis and the target of statins. "
        "Glycogen phosphorylase and glycogen synthase are reciprocally regulated by glucagon and insulin."),
    "pharma": (
        "Pharmacokinetics describes absorption, distribution, metabolism, and excretion. "
        "Propranolol is a nonselective beta blocker used for hypertension and migraine prophylaxis, whereas "
        "metoprolol is beta one selective. "
        "Angiotensin receptor blockers such as losartan lower blood pressure without causing the cough associated "
        "with ACE inhibitors. "
        "Warfarin inhibits vitamin K epoxide reductase and is monitored with the INR. "
        "Acetylcholinesterase inhibitors like donepezil are used in Alzheimer disease, and atropine is the antidote "
        "for organophosphate poisoning. "
        "Amiodarone prolongs the action potential by blocking potassium channels. "
        "Aminoglycosides such as gentamicin cause nephrotoxicity and ototoxicity."),
}

VBS = r'''
Set fso = CreateObject("Scripting.FileSystemObject")
Set f = fso.OpenTextFile(WScript.Arguments(0), 1)
text = f.ReadAll
f.Close
Set voice = CreateObject("SAPI.SpVoice")
voice.Rate = CInt(WScript.Arguments(2))
Set fmt = CreateObject("SAPI.SpAudioFormat")
fmt.Type = 22
Set stream = CreateObject("SAPI.SpFileStream")
Set stream.Format = fmt
stream.Open WScript.Arguments(1), 3, False
Set voice.AudioOutputStream = stream
voice.Speak text, 0
stream.Close
'''


def speak(text: str, wav: Path, rate: int) -> None:
    with tempfile.TemporaryDirectory(prefix="mlt_tts_") as tmp:
        script, txt = Path(tmp) / "speak.vbs", Path(tmp) / "text.txt"
        script.write_text(VBS, encoding="ascii")
        txt.write_text(text, encoding="ascii")
        proc = subprocess.run(["cscript", "//nologo", str(script), str(txt), str(wav), str(rate)],
                              capture_output=True, timeout=300)
        if not wav.is_file():          # cscript exits 0 even when the script fails; the message is in the console codepage
            detail = (proc.stdout + proc.stderr).decode("cp1254", "replace").encode("ascii", "replace").decode()
            raise RuntimeError(f"SAPI synthesis failed: {detail.strip()}")


def to_16k_mono(wav: Path) -> None:
    """SAPI voices speak at 22.05 kHz; the pipeline's own format is 16 kHz mono PCM (decoded in-process, no FFmpeg)."""
    from faster_whisper.audio import decode_audio

    sf.write(str(wav), decode_audio(str(wav), sampling_rate=SAMPLE_RATE), SAMPLE_RATE, subtype="PCM_16")


def add_noise(wav: Path, snr_db: float, seed: int) -> None:
    data, rate = sf.read(str(wav))
    rng = np.random.default_rng(seed)
    signal_power = float(np.mean(data ** 2)) or 1e-8
    noise = rng.normal(0.0, np.sqrt(signal_power / (10 ** (snr_db / 10))), size=data.shape)
    sf.write(str(wav), np.clip(data + noise, -1.0, 1.0), rate, subtype="PCM_16")


RANDOM_SUBJECTS = {"neuro": "neuroscience", "biochem": "biochemistry", "pharma": "pharmacology"}


def vocabulary_terms(subject_file: str) -> list[str]:
    path = ROOT / "medical_terms" / f"{subject_file}.txt"
    terms = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith(("#", "tr:")):
            terms.append(line.removeprefix("en:").strip())
    return terms


def random_passage(subject_file: str, seed: int, n_terms: int = 6) -> tuple[str, list[str]]:
    """A lecture-like passage about ``n_terms`` terms drawn uniformly from the subject vocabulary (NOT the editorial favourites).

    Like a real lecture it announces the topics first, explains each one in the middle and recaps at the end, so every term is
    spoken three times - the situation the probe-then-bias design is meant for.
    """
    rng = random.Random(seed)
    pool = [t for t in vocabulary_terms(subject_file) if 4 <= len(t) <= 34 and not any(c.isdigit() for c in t)]
    terms = rng.sample(pool, n_terms)
    pairs = [(terms[i], terms[(i + 1) % n_terms]) for i in range(0, n_terms, 2)]
    intro = f"Today we will cover {', '.join(terms[:-1])}, and {terms[-1]}."
    middle = " ".join(f"{a} is clinically important, and it is closely related to {b}. Remember that {b} is often tested in exams."
                      for a, b in pairs)
    recap = f"To summarize, {terms[0]} and {terms[1]} come first, then {terms[2]} and {terms[3]}, and finally {terms[4]} with {terms[-1]}."
    return " ".join([intro, middle, recap]), terms


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="regenerate files that already exist")
    parser.add_argument("--random-seed", type=int, default=None,
                        help="instead of the fixed passages write three lecture-like passages with random vocabulary terms")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    passages = dict(PASSAGES)
    if args.random_seed is not None:
        passages = {}
        for short, subject in RANDOM_SUBJECTS.items():
            name = f"r{args.random_seed}{short}"
            text, terms = random_passage(subject, args.random_seed * 100 + len(passages))
            passages[name] = text
            (OUT / f"{name}.meta.json").write_text(json.dumps({"subject": subject, "neutral": f"lecture{len(passages)}", "terms": terms}),
                                                  encoding="utf-8")
    for index, (name, text) in enumerate(passages.items()):
        (OUT / f"{name}.txt").write_text(text + "\n", encoding="utf-8", newline="\n")
        for condition, rate, snr in (("clean", 0, None), ("hard", 3, 8.0)):
            wav = OUT / f"{name}_{condition}.wav"
            if wav.exists() and not args.force:
                print(f"exists  {wav.name}")
                continue
            speak(text, wav, rate)
            to_16k_mono(wav)
            if snr is not None:
                add_noise(wav, snr, seed=1000 + index)
            info = sf.info(str(wav))
            print(f"created {wav.name}: {info.frames / info.samplerate:.1f} s, {info.samplerate} Hz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
