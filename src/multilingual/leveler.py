"""Quiet speech: raise the volume of speech that is far quieter than normal before the voice filter and the recogniser look at it.

A speaker far from the microphone can be 40 dB quieter than the rest of a recording. The voice-activity filter is not level-blind: speech at -60 dBFS looks like silence to it and is
cut out before the recogniser ever hears it (measured on authorized test recordings: whole sentences were never written). This step lifts only seconds that are
* speech-like (the loudness changes the way syllables and pauses do - steady room noise does not), and
* quieter than ``QUIET_BELOW`` (normal speech is left exactly as it is),
towards ``TARGET_RMS``, by at most ``MAX_GAIN_DB``, with a smooth gain curve and a soft limiter. It works on the program's own 16 kHz working copy of the audio, never on the
student's recording, and it reports what it did.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

RATE = 16000
FRAME_SECONDS = 0.02
FLOOR_RMS = 0.00025              # below this a second is silence / noise floor (about -68 dBFS): never raised
QUIET_BELOW = 0.015              # seconds at least this loud are normal speech and are left untouched (about -36 dBFS)
TARGET_RMS = 0.05                # quiet speech is lifted towards this loudness (about -26 dBFS)
MAX_GAIN_DB = 40.0
SPEECH_RANGE_DB = 8.0            # a second is speech-like when its loud 20 ms frames are this much above its quiet ones
MIN_SECONDS = 3                  # an isolated quiet second is not worth touching
SMOOTH_SECONDS = 1.0


@dataclass
class Leveling:
    path: Path
    applied: bool = False
    seconds_raised: int = 0
    max_gain_db: float = 0.0
    regions: list[tuple[float, float, float]] = field(default_factory=list)       # (start, end, gain in dB)
    reason: str = ""

    def record(self) -> dict[str, Any]:
        return {"applied": self.applied, "seconds_raised": self.seconds_raised, "max_gain_db": round(self.max_gain_db, 1),
                "regions": [[round(a, 1), round(b, 1), round(g, 1)] for a, b, g in self.regions[:40]], "reason": self.reason,
                "note": "quiet speech was made louder in the program's working copy of the audio; the recording itself is untouched"}


def second_stats(path: Path) -> list[tuple[float, float]]:
    """``[(rms, speech range in dB)]`` for every second of the WAV (streamed, a few seconds in memory at a time)."""
    import numpy as np
    import soundfile as sf

    stats: list[tuple[float, float]] = []
    with sf.SoundFile(str(path)) as handle:
        rate = handle.samplerate
        frame = max(int(FRAME_SECONDS * rate), 1)
        block_seconds = 30
        while True:
            block = handle.read(rate * block_seconds, dtype="float32", always_2d=True)
            if not len(block):
                break
            mono = block.mean(axis=1)
            for start in range(0, len(mono), rate):
                piece = mono[start:start + rate]
                if len(piece) < rate // 2:
                    stats.append((float(np.sqrt((piece ** 2).mean())) if len(piece) else 0.0, 0.0))
                    continue
                usable = len(piece) // frame * frame
                energies = np.sqrt((piece[:usable].reshape(-1, frame) ** 2).mean(axis=1)) + 1e-9
                loud, quiet = float(np.percentile(energies, 90)), float(np.percentile(energies, 20))
                stats.append((float(np.sqrt((piece ** 2).mean())), 20.0 * float(np.log10(loud / quiet))))
    return stats


def gain_curve(stats: list[tuple[float, float]]) -> list[float]:
    """The linear gain for every second: 1.0 for normal speech, silence and noise; more for quiet speech-like seconds, smoothed."""
    import numpy as np

    wanted = np.ones(len(stats))
    candidate = np.array([FLOOR_RMS <= rms < QUIET_BELOW and spread >= SPEECH_RANGE_DB for rms, spread in stats], dtype=bool)
    # a quiet speech-like second counts only inside a stretch of at least MIN_SECONDS (a click in the noise is not speech)
    i = 0
    while i < len(candidate):
        if not candidate[i]:
            i += 1
            continue
        j = i
        while j < len(candidate) and candidate[j]:
            j += 1
        if j - i >= MIN_SECONDS:
            for k in range(i, j):
                wanted[k] = min(TARGET_RMS / max(stats[k][0], FLOOR_RMS), 10 ** (MAX_GAIN_DB / 20.0))
        i = j
    if not (wanted > 1.0).any():
        return list(wanted)
    smooth = max(int(SMOOTH_SECONDS * 2) | 1, 3)
    db = 20.0 * np.log10(wanted)
    kernel = np.ones(smooth) / smooth
    smoothed = np.convolve(np.pad(db, (smooth // 2, smooth // 2), mode="edge"), kernel, mode="valid")
    smoothed = np.maximum(smoothed, 0.0)
    smoothed[db <= 0.0] = np.minimum(smoothed[db <= 0.0], 6.0)                      # next to a raised stretch, the neighbours are touched only a little (cross-fade)
    return list(10 ** (smoothed / 20.0))


def level_quiet_speech(wav: Path, work_dir: Path, in_place: bool = True) -> Leveling:
    """Return the WAV to use from here on: ``wav`` itself when nothing needed raising, else a leveled copy (``in_place``: written over ``wav``, which must be the program's own
    working copy; otherwise next to it in ``work_dir``). Never raises: on any problem the original WAV is used and the reason says why."""
    result = Leveling(path=Path(wav))
    try:
        import numpy as np
        import soundfile as sf

        stats = second_stats(Path(wav))
        gains = gain_curve(stats)
        if not stats or not any(g > 1.05 for g in gains):
            result.reason = "no quiet speech found"
            return result
        info = sf.info(str(wav))
        rate = info.samplerate
        target = Path(wav) if in_place else Path(work_dir) / (Path(wav).stem + "_leveled.wav")
        temp = target.with_name(target.stem + ".tmp.wav")
        centres = (np.arange(len(gains)) + 0.5) * rate
        gains_arr = np.asarray(gains, dtype="float64")
        with sf.SoundFile(str(wav)) as src, sf.SoundFile(str(temp), "w", samplerate=rate, channels=1, subtype="PCM_16") as dst:
            position = 0
            while True:
                block = src.read(rate * 30, dtype="float32", always_2d=True)
                if not len(block):
                    break
                mono = block.mean(axis=1)
                index = np.arange(position, position + len(mono))
                factor = np.interp(index, centres, gains_arr)
                dst.write(np.tanh(mono * factor).astype("float32"))               # soft limiter: a raised peak never clips
                position += len(mono)
        os.replace(temp, target)
        raised = [i for i, g in enumerate(gains) if g > 1.05]
        result.path, result.applied, result.seconds_raised = target, True, len(raised)
        result.max_gain_db = float(20.0 * np.log10(max(gains)))
        start = raised[0]
        previous = raised[0]
        for i in raised[1:] + [None]:
            if i is None or i != previous + 1:
                result.regions.append((float(start), float(previous + 1), float(20.0 * np.log10(max(gains[start:previous + 1])))))
                start = i if i is not None else start
            previous = i if i is not None else previous
        result.reason = "speech much quieter than normal was made louder"
        log.info("QUIET SPEECH: %d s of speech-like audio below %.0f dBFS were made louder (up to +%.0f dB)", result.seconds_raised, 20 * np.log10(QUIET_BELOW), result.max_gain_db)
    except Exception as exc:  # noqa: BLE001 - an aid on top of the recording; the original working copy is used
        log.warning("quiet-speech leveling skipped (%s: %s)", type(exc).__name__, exc)
        result.reason = f"skipped: {type(exc).__name__}"
        result.path = Path(wav)
    return result
