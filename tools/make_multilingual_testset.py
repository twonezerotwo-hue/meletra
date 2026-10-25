"""Build the reproducible Korean / Hindi / English test recordings from REAL human speech with reference transcripts.

    python tools/make_multilingual_testset.py [--out benchmarks/multilingual] [--clips 6]

Source: FLEURS (Google, CC BY 4.0, https://huggingface.co/datasets/google/fleurs): sentences read aloud by human speakers, each with a human-checked
transcription. The clips are fetched from the Hugging Face dataset repository (no login; only the first part of each test archive), cached under ``<out>/fleurs`` and never committed (``benchmarks/multilingual``
is git-ignored); only this script and the clip ids it picks (deterministic) are part of the repository. Attribution: "FLEURS: Few-shot Learning Evaluation of Universal
Representations of Speech", Conneau et al., 2022 (CC BY 4.0).

Recordings written to ``<out>/recordings`` (16 kHz mono WAV) with ``<name>.truth.json``: the exact timeline (start, end, language, reference text of every clip):

    ko_only, hi_only, en_only            one language, about a minute
    sections_ko_hi_en_ko                 Korean, Hindi, English, Korean - whole sections (clear language switches at pauses)
    dense_ko_hi_en                       single sentences alternate every few seconds with 0.25 s gaps: the language changes INSIDE one 30-second window
    english_with_brief_ko_hi             an English lecture-like recording with ONE short Korean and ONE short Hindi passage (the case a "dominant language" rule loses)
"""

from __future__ import annotations

import argparse
import json
import urllib.request
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = {"ko": "ko_kr", "hi": "hi_in", "en": "en_us", "tr": "tr_tr", "de": "de_de", "hu": "hu_hu", "zh": "cmn_hans_cn", "nl": "nl_nl", "fr": "fr_fr", "es": "es_419"}
OTHERS = ("tr", "de", "hu", "zh", "nl", "fr", "es")
RATE = 16000


def get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "meletra-testset/1"})
    last = None
    for _ in range(4):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except Exception as exc:  # noqa: BLE001 - retried, then reported
            last = exc
    raise SystemExit(f"could not fetch {url}: {last}")


BASE = "https://huggingface.co/datasets/google/fleurs/resolve/main/data/{config}"
HEAD_MEMBERS = 70                      # how many clips are taken from the start of each test archive (about 25 MB instead of 250 MB)


def fetch_head(config: str, folder: Path) -> dict[str, dict]:
    """The first ``HEAD_MEMBERS`` clips of the language's test archive (streamed, the rest of the archive is never downloaded) with their rows from ``test.tsv``.
    Cached in ``folder``; returns ``{file name: row}`` for the clips that are on disk."""
    import tarfile

    index_file = folder / "index.json"
    if index_file.is_file():
        return json.loads(index_file.read_text(encoding="utf-8"))
    tsv = get(BASE.format(config=config) + "/test.tsv").decode("utf-8")
    meta: dict[str, dict] = {}
    for line in tsv.splitlines():
        parts = line.split("\t")
        if len(parts) >= 7:
            meta[parts[1]] = {"id": parts[0] + "_" + parts[1].removesuffix(".wav")[-6:], "text": parts[2], "normalized": parts[3], "seconds": int(parts[5]) / RATE,
                              "gender": 0 if parts[6].strip().upper().startswith("FEMALE") else 1}
    request = urllib.request.Request(BASE.format(config=config) + "/audio/test.tar.gz", headers={"User-Agent": "meletra-testset/1"})
    rows: dict[str, dict] = {}
    with urllib.request.urlopen(request, timeout=120) as response, tarfile.open(fileobj=response, mode="r|gz") as archive:
        for member in archive:
            name = Path(member.name).name
            if not member.isfile() or name not in meta:
                continue
            data = archive.extractfile(member).read()
            (folder / name).write_bytes(data)
            rows[name] = {**meta[name], "file": name}
            if len(rows) >= HEAD_MEMBERS:
                break
    index_file.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    return rows


def pick_clips(language: str, count: int, cache: Path, min_s: float = 6.0, max_s: float = 13.0, offset_rows: int = 0) -> list[dict]:
    """``count`` test clips of 6-13 s, alternating the two speaker genders where the data allows; the first part of the language's test archive is downloaded once."""
    config = CONFIGS[language]
    folder = cache / config
    folder.mkdir(parents=True, exist_ok=True)
    head = fetch_head(config, folder)
    rows = sorted((r for r in head.values() if min_s <= r["seconds"] <= max_s), key=lambda r: r["id"])
    chosen: list[dict] = []
    wanted_gender = 0
    pool = rows[offset_rows:]
    while len(chosen) < count and pool:
        for i, row in enumerate(pool):
            if row["gender"] == wanted_gender or i == len(pool) - 1:
                chosen.append(dict(pool.pop(i)))
                break
        wanted_gender = 1 - wanted_gender
    if len(chosen) < count:
        raise SystemExit(f"{config}: only {len(chosen)} suitable clips in the first {HEAD_MEMBERS} of the archive")
    for row in chosen:
        row["path"] = str(folder / row["file"])
        row["language"] = language
    return chosen


def read_wav(path: str) -> np.ndarray:
    import soundfile as sf

    data, rate = sf.read(path, dtype="float32", always_2d=True)
    assert rate == RATE, f"{path}: expected 16 kHz, got {rate}"
    return data.mean(axis=1)


def write_wav(path: Path, samples: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes((np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes())


def music_bed(seconds: float, level: float = 0.08, seed: int = 7) -> np.ndarray:
    """A synthetic, licence-free 'song' bed (chords in a slow progression with a soft beat and a little noise): background music for the speech-over-music fixtures."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * RATE)) / RATE
    chords = [(220.0, 277.18, 329.63), (196.0, 246.94, 293.66), (174.61, 220.0, 261.63), (196.0, 246.94, 392.0)]
    bed = np.zeros_like(t)
    for i, chord in enumerate(chords):
        mask = ((t // 3.0).astype(int) % len(chords)) == i
        for f in chord:
            bed += np.where(mask, np.sin(2 * np.pi * f * t) + 0.4 * np.sin(2 * np.pi * 2 * f * t), 0.0)
    beat = 0.5 + 0.5 * np.sin(2 * np.pi * 2.0 * t) ** 8
    bed = bed * beat + 0.05 * rng.standard_normal(len(t))
    return (bed / max(np.abs(bed).max(), 1e-6) * level).astype(np.float32)


def build(name: str, clips: list[dict], gaps: list[float] | float, out: Path, lead: float = 1.0, music: float = 0.0, gain: float = 1.0) -> dict:
    """Concatenate clips with silent gaps; write the WAV and the truth timeline."""
    pieces = [np.zeros(int(lead * RATE), dtype=np.float32)]
    timeline = []
    cursor = lead
    for index, clip in enumerate(clips):
        audio = read_wav(clip["path"])
        pieces.append(audio)
        timeline.append({"start": round(cursor, 2), "end": round(cursor + len(audio) / RATE, 2), "language": clip["language"], "id": clip["id"], "text": clip["text"],
                         "normalized": clip["normalized"]})
        cursor += len(audio) / RATE
        gap = gaps[index] if isinstance(gaps, list) else gaps
        pieces.append(np.zeros(int(gap * RATE), dtype=np.float32))
        cursor += gap
    audio_all = np.concatenate(pieces) * gain
    if music:
        audio_all = audio_all + music_bed(len(audio_all) / RATE, level=music)
    write_wav(out / "recordings" / f"{name}.wav", audio_all)
    truth = {"name": name, "seconds": round(len(audio_all) / RATE, 2), "languages": sorted({c["language"] for c in clips}), "timeline": timeline,
             "source": "FLEURS (Google), CC BY 4.0, human-read sentences with human-checked transcriptions",
             "music_level": music or None, "gain": gain}
    (out / "recordings" / f"{name}.truth.json").write_text(json.dumps(truth, ensure_ascii=False, indent=1), encoding="utf-8")
    return truth


def loud_clip(clips: list[dict], minimum_rms: float = 0.02) -> dict:
    """The first clip that is clearly audible (some human-read clips are recorded very quietly)."""
    for clip in clips:
        audio = read_wav(clip["path"])
        if float(np.sqrt((audio ** 2).mean())) >= minimum_rms:
            return clip
    raise SystemExit("no audible clip among the candidates")


def build_smoke(out: Path) -> dict:
    """A short Korean -> Hindi -> English recording (three audible sentences) for the checks of the installed program: about 35 seconds, known truth."""
    cache = out / "fleurs"
    clips = [loud_clip(pick_clips(code, 14, cache)[3:]) for code in ("ko", "hi", "en")]
    return build("smoke_ko_hi_en", clips, 0.8, out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "benchmarks" / "multilingual")
    parser.add_argument("--clips", type=int, default=6, help="sentences per single-language recording")
    parser.add_argument("--smoke-only", action="store_true", help="only the short Korean / Hindi / English recording for the installed-program checks")
    args = parser.parse_args(argv)
    if args.smoke_only:
        truth = build_smoke(args.out)
        print(f"smoke_ko_hi_en {truth['seconds']:.1f} s languages {','.join(truth['languages'])}")
        return 0
    cache = args.out / "fleurs"
    ko = pick_clips("ko", 14, cache)
    hi = pick_clips("hi", 14, cache)
    en = pick_clips("en", 18, cache)
    results = {}
    results["ko_only"] = build("ko_only", ko[: args.clips], 0.7, args.out)
    results["hi_only"] = build("hi_only", hi[: args.clips], 0.7, args.out)
    results["en_only"] = build("en_only", en[: args.clips], 0.7, args.out)
    sections = ko[6:10] + hi[6:10] + en[6:10] + ko[10:13]
    section_gaps = [0.5, 0.5, 0.5, 2.0] * 3 + [0.5, 0.5, 0.5]                      # a longer pause at each change of section
    results["sections_ko_hi_en_ko"] = build("sections_ko_hi_en_ko", sections, section_gaps, args.out)
    dense = [ko[13], hi[10], en[10], hi[11], ko[0], en[11], ko[1], hi[12], en[12], ko[2]]
    results["dense_ko_hi_en"] = build("dense_ko_hi_en", dense, 0.25, args.out)
    brief = en[12:16] + [ko[3]] + en[16:18] + [hi[13]] + en[0:3]
    results["english_with_brief_ko_hi"] = build("english_with_brief_ko_hi", brief, 0.6, args.out)
    results["hi_en_hi"] = build("hi_en_hi", hi[0:3] + en[0:3] + hi[3:6], 0.6, args.out)
    results["en_ko_en"] = build("en_ko_en", en[3:6] + ko[3:6] + en[6:9], 0.6, args.out)
    results["sections_with_music"] = build("sections_with_music", sections, section_gaps, args.out, music=0.05)
    results["quiet_ko_hi_en"] = build("quiet_ko_hi_en", ko[0:2] + hi[0:2] + en[0:2], 0.6, args.out, gain=0.03)
    for code in OTHERS:                                                    # one short recording per other Output Language: Auto Detect must name it
        results[f"{code}_only"] = build(f"{code}_only", pick_clips(code, 3, cache), 0.7, args.out)
    for name, truth in results.items():
        print(f"{name:28s} {truth['seconds']:6.1f} s  languages {','.join(truth['languages'])}  clips {len(truth['timeline'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
