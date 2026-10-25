"""Quiet speech is lifted before the voice filter sees it; normal speech, silence and steady noise are left alone; the original working copy is replaced atomically."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from src.multilingual import leveler as lv

RATE = 16000


def speechlike(seconds: float, rms: float, seed: int = 1) -> np.ndarray:
    """Syllable-like bursts with pauses (loudness changes the way speech does), scaled to the given overall rms."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * RATE)) / RATE
    envelope = np.clip(np.sin(2 * np.pi * 3.0 * t), 0, None) ** 0.6
    voice = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) / (k + 1) for k, f in enumerate((140, 280, 420, 700, 1100, 1800)))
    x = envelope * voice
    return (x / np.sqrt((x ** 2).mean()) * rms).astype("float32")


def noise(seconds: float, rms: float) -> np.ndarray:
    return (np.random.default_rng(3).standard_normal(int(seconds * RATE)) * rms).astype("float32")


def write(path, parts):
    sf.write(str(path), np.concatenate(parts), RATE, subtype="PCM_16")


def level_of(path, start, end):
    x, _r = sf.read(str(path), dtype="float32")
    seg = x[int(start * RATE):int(end * RATE)]
    return float(np.sqrt((seg ** 2).mean()))


def test_quiet_speech_is_lifted_and_normal_speech_noise_and_silence_are_not(tmp_path):
    path = tmp_path / "a.wav"
    write(path, [speechlike(10, 0.08), np.zeros(RATE * 2, dtype="float32"), speechlike(10, 0.0008, 2), noise(10, 0.0006), speechlike(10, 0.06, 4)])
    before_loud = level_of(path, 0, 10)
    result = lv.level_quiet_speech(path, tmp_path)
    assert result.applied and result.path == path and result.seconds_raised >= 8 and result.max_gain_db > 25
    assert level_of(path, 14, 20) == pytest.approx(0.05, rel=0.5)                                             # the quiet sentence is now about as loud as ordinary speech
    assert level_of(path, 0, 10) == pytest.approx(before_loud, rel=0.15)                                      # the loud passage is unchanged
    assert level_of(path, 24, 32) < 0.002                                                                     # steady noise at -64 dBFS stays where it was
    assert level_of(path, 10.3, 11.7) == pytest.approx(0.0, abs=1e-4)                                         # digital silence stays silence
    assert len(sf.read(str(path), dtype="float32")[0]) == RATE * 42 and not list(tmp_path.glob("*.tmp.wav"))
    record = result.record()
    assert record["applied"] and record["regions"] and "recording itself is untouched" in record["note"]


def test_a_recording_with_nothing_quiet_is_not_touched(tmp_path):
    path = tmp_path / "a.wav"
    write(path, [speechlike(20, 0.07), noise(5, 0.002)])
    before = path.read_bytes()
    result = lv.level_quiet_speech(path, tmp_path)
    assert not result.applied and result.path == path and path.read_bytes() == before and result.reason == "no quiet speech found"


def test_a_whole_recording_that_is_quiet_is_lifted(tmp_path):
    path = tmp_path / "quiet.wav"
    write(path, [speechlike(20, 0.003, 5)])
    result = lv.level_quiet_speech(path, tmp_path)
    assert result.applied and 0.02 < level_of(path, 2, 18) < 0.12


def test_the_gain_is_limited_and_never_clips(tmp_path):
    path = tmp_path / "quiet.wav"
    write(path, [speechlike(12, 0.0005, 6)])
    result = lv.level_quiet_speech(path, tmp_path)
    x, _r = sf.read(str(path), dtype="float32")
    assert result.max_gain_db <= lv.MAX_GAIN_DB + 0.1 and float(np.abs(x).max()) < 1.0


def test_a_copy_can_be_made_instead_of_changing_the_file(tmp_path):
    path = tmp_path / "users.wav"
    write(path, [speechlike(12, 0.001, 7)])
    original = path.read_bytes()
    result = lv.level_quiet_speech(path, tmp_path, in_place=False)
    assert result.applied and result.path != path and path.read_bytes() == original and result.path.is_file()


def test_a_damaged_file_never_stops_the_run(tmp_path):
    path = tmp_path / "broken.wav"
    path.write_bytes(b"not a wav")
    result = lv.level_quiet_speech(path, tmp_path)
    assert not result.applied and result.path == path and result.reason.startswith("skipped")
