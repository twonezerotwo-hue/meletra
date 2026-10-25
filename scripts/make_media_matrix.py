"""Make the media-format matrix used by the LGPL decoder check and the packaged media test: one spoken sentence in every container / codec the program must read.

    python scripts/make_media_matrix.py OUT_DIR [--speech WAV]

Encoding needs encoders the program itself never ships (H.264 / HEVC are encoded here with the development environment's PyAV); the program only has to DECODE the results.
Written: speech.wav (PCM), speech.mp3, speech.m4a (AAC in MP4), speech.aac (ADTS), video_h264.mp4 (H.264 + AAC), video_h264.mov (H.264 + AAC in QuickTime),
video_hevc.mp4 (HEVC + AAC) - each a few seconds, 16 kHz mono audio. A file whose encoder is missing in this environment is skipped and reported.
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

RATE = 16000


def tone_speech(seconds: float = 4.0) -> np.ndarray:
    t = np.arange(int(RATE * seconds)) / RATE
    return (0.3 * np.sin(2 * np.pi * 220 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)


def load_speech(path: Path | None, seconds: float = 0.0) -> np.ndarray:
    if path is None:
        return tone_speech()
    import soundfile as sf

    data, rate = sf.read(str(path), dtype="float32")
    data = data if data.ndim == 1 else data.mean(axis=1)
    if rate != RATE:
        data = np.interp(np.linspace(0, len(data) - 1, int(len(data) * RATE / rate)), np.arange(len(data)), data).astype(np.float32)
    return data[:int(seconds * RATE)] if seconds else data


def write_audio(path: Path, samples: np.ndarray, codec: str, fmt: str | None = None, rate: int = RATE) -> None:
    import av

    with av.open(str(path), "w", format=fmt) as out:
        stream = out.add_stream(codec, rate=rate)
        stream.layout = "mono"
        _encode_audio(out, stream, samples, rate)


def _encode_audio(out, stream, samples: np.ndarray, rate: int) -> None:
    import av

    size = stream.codec_context.frame_size or 1024
    fmt = stream.codec_context.format.name if stream.codec_context.format else "fltp"
    pos = 0
    while pos < len(samples):
        chunk = samples[pos:pos + size]
        if len(chunk) < size:
            chunk = np.pad(chunk, (0, size - len(chunk)))
        if fmt in ("s16", "s16p"):
            data, layout_fmt = (np.clip(chunk, -1, 1) * 32767).astype(np.int16).reshape(1, -1), "s16p" if fmt == "s16p" else "s16"
        else:
            data, layout_fmt = chunk.reshape(1, -1).astype(np.float32), "fltp" if fmt == "fltp" else "flt"
        frame = av.AudioFrame.from_ndarray(data, format=layout_fmt, layout="mono")
        frame.sample_rate = rate
        frame.pts = pos
        for packet in stream.encode(frame):
            out.mux(packet)
        pos += size
    for packet in stream.encode(None):
        out.mux(packet)


def write_video(path: Path, samples: np.ndarray, vcodec: str, fmt: str | None = None) -> None:
    import av

    seconds = len(samples) / RATE
    with av.open(str(path), "w", format=fmt) as out:
        video = out.add_stream(vcodec, rate=10)
        video.width, video.height, video.pix_fmt = 320, 240, "yuv420p"
        audio = out.add_stream("aac", rate=RATE)
        audio.layout = "mono"
        for i in range(int(seconds * 10)):
            img = np.full((240, 320, 3), (i * 7) % 255, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            frame.pts = i
            frame.time_base = Fraction(1, 10)
            for packet in video.encode(frame):
                out.mux(packet)
        for packet in video.encode(None):
            out.mux(packet)
        _encode_audio(out, audio, samples, RATE)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--speech", type=Path)
    ap.add_argument("--seconds", type=float, default=0.0, help="use only the first N seconds of the speech")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    samples = load_speech(a.speech, a.seconds)
    import soundfile as sf

    made, skipped = [], {}
    sf.write(str(a.out / "speech.wav"), samples, RATE, subtype="PCM_16")
    made.append("speech.wav")
    jobs = [("speech.mp3", lambda p: write_audio(p, samples, "libmp3lame")), ("speech.m4a", lambda p: write_audio(p, samples, "aac", "mp4")),
            ("speech.aac", lambda p: write_audio(p, samples, "aac", "adts")), ("video_h264.mp4", lambda p: write_video(p, samples, "libx264", "mp4")),
            ("video_h264.mov", lambda p: write_video(p, samples, "libx264", "mov")), ("video_hevc.mp4", lambda p: write_video(p, samples, "libx265", "mp4"))]
    for name, job in jobs:
        try:
            job(a.out / name)
            made.append(name)
        except Exception as exc:                       # noqa: BLE001 - an encoder this environment lacks
            skipped[name] = f"{type(exc).__name__}: {exc}"[:160]
    print(json.dumps({"made": made, "skipped": skipped, "seconds": round(len(samples) / RATE, 2)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
