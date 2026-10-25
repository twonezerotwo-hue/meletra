"""Preparing the audio of a video: audio only, real progress, a prompt cancel, an FFmpeg that hangs is abandoned for PyAV, clear errors."""

import stat
import sys
import textwrap
import time
import wave
from pathlib import Path

import pytest

from src import audio as audio_mod
from src.audio import AudioExtractionError, ExtractionCancelled, extract_audio, extract_audio_pyav
from tests.test_ingest_router import encode


def seconds_of(path: Path) -> float:
    with wave.open(str(path)) as handle:
        return handle.getnframes() / float(handle.getframerate())


@pytest.fixture(scope="module")
def video(tmp_path_factory) -> Path:
    """A 20 s MP4 with a video track and an AAC audio track."""
    return encode(tmp_path_factory.mktemp("vid") / "lecture video.mp4", "mp4", "aac", "mpeg4", seconds=20.0)


def fake_ffmpeg(tmp_path: Path, body: str) -> Path:
    """An 'ffmpeg' that is a tiny Python script (so a hang or a slow run can be staged exactly)."""
    script = tmp_path / "fake_ffmpeg.py"
    script.write_text("import sys, time, os\nargs = sys.argv[1:]\nout = args[-1]\n" + textwrap.dedent(body), encoding="utf-8")
    if sys.platform == "win32":
        launcher = tmp_path / "ffmpeg.cmd"
        launcher.write_text(f'@echo off\r\n"{sys.executable}" "{script}" %*\r\n', encoding="utf-8")
    else:
        launcher = tmp_path / "ffmpeg"
        launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8")
        launcher.chmod(launcher.stat().st_mode | stat.S_IEXEC)
    return launcher


# ----------------------------------------------------------------------------------------------- progress is real
def test_ffmpeg_path_reports_the_media_time_written_and_only_takes_the_audio(video, tmp_path, cfg, ffmpeg_path):
    ticks = []
    out = extract_audio(video, tmp_path / "a.wav", cfg["audio"], ffmpeg=ffmpeg_path, allow_fallback=True, progress=ticks.append,
                        should_stop=lambda: False)
    assert seconds_of(out) == pytest.approx(20.0, abs=0.5)
    assert ticks and ticks == sorted(ticks) and ticks[-1] == pytest.approx(20.0, abs=1.0)          # measured, increasing, ends at the duration


def test_the_command_takes_audio_only(video, tmp_path, cfg):
    seen = tmp_path / "argv.txt"
    ffmpeg = fake_ffmpeg(tmp_path, f"open({str(seen)!r}, 'w').write(' '.join(args))\nopen(out, 'wb').write(b'x')\nsys.exit(1)\n")
    with pytest.raises(AudioExtractionError):
        extract_audio(video, tmp_path / "b.wav", cfg["audio"], ffmpeg=ffmpeg, progress=lambda s: None)
    argv = seen.read_text(encoding="utf-8")
    assert "-map 0:a:0" in argv and "-vn" in argv and "-sn" in argv and "-progress pipe:1" in argv


def test_pyav_path_reports_progress_and_never_decodes_the_video(video, tmp_path, cfg, monkeypatch):
    import av
    real_open, demuxed = av.open, []

    class Spy:                                                  # the C container's attributes are read-only: wrap it
        def __init__(self, container):
            self.container = container

        def __enter__(self):
            self.container.__enter__()
            return self

        def __exit__(self, *exc):
            return self.container.__exit__(*exc)

        def __getattr__(self, name):
            return getattr(self.container, name)

        def demux(self, *streams):
            demuxed.append([s.type for s in streams])
            return self.container.demux(*streams)

    monkeypatch.setattr(av, "open", lambda *a, **k: Spy(real_open(*a, **k)))
    ticks = []
    out = extract_audio_pyav(video, tmp_path / "p.wav", cfg["audio"], progress=ticks.append)
    assert seconds_of(out) == pytest.approx(20.0, abs=0.5) and ticks
    assert demuxed == [["audio"]]                                                           # only the audio stream was asked for


# ----------------------------------------------------------------------------------------------- cancel
def test_cancel_during_pyav_decoding_stops_and_removes_the_partial_file(video, tmp_path, cfg):
    calls = []

    def stop():
        calls.append(1)
        return len(calls) > 3
    destination = tmp_path / "c.wav"
    with pytest.raises(ExtractionCancelled):
        extract_audio_pyav(video, destination, cfg["audio"], should_stop=stop)
    assert not destination.exists()


def test_cancel_during_an_ffmpeg_extraction_kills_only_that_process(tmp_path, cfg, video):
    ffmpeg = fake_ffmpeg(tmp_path, "open(out, 'wb').write(b'x')\nprint('out_time_us=100000', flush=True)\ntime.sleep(60)\n")
    stop = [False]
    started = time.monotonic()

    def should_stop():
        stop[0] = time.monotonic() - started > 1.0
        return stop[0]
    with pytest.raises(ExtractionCancelled):
        extract_audio(video, tmp_path / "d.wav", cfg["audio"], ffmpeg=ffmpeg, allow_fallback=True, should_stop=should_stop)
    assert time.monotonic() - started < 15 and not (tmp_path / "d.wav").exists()


# ----------------------------------------------------------------------------------------------- an FFmpeg that hangs is abandoned
def test_an_ffmpeg_that_stops_making_progress_falls_back_to_pyav(video, tmp_path, cfg, monkeypatch):
    monkeypatch.setattr(audio_mod, "STALL_SECONDS", 1.0)
    ffmpeg = fake_ffmpeg(tmp_path, "open(out, 'wb').write(b'x')\nprint('out_time_us=200000', flush=True)\ntime.sleep(120)\n")
    started = time.monotonic()
    ticks = []
    out = extract_audio(video, tmp_path / "e.wav", cfg["audio"], ffmpeg=ffmpeg, allow_fallback=True, progress=ticks.append)
    assert time.monotonic() - started < 60                                                 # not minutes: the stuck process was abandoned
    assert seconds_of(out) == pytest.approx(20.0, abs=0.5) and ticks[-1] > 5


def test_an_ffmpeg_that_never_starts_falls_back_to_pyav(video, tmp_path, cfg, monkeypatch):
    monkeypatch.setattr(audio_mod, "STALL_SECONDS", 1.0)
    ffmpeg = fake_ffmpeg(tmp_path, "time.sleep(120)\n")                                    # no output file, no progress: stalled at once
    started = time.monotonic()
    original = audio_mod._run_extraction
    monkeypatch.setattr(audio_mod, "_run_extraction", lambda cmd, dest, **kw: original(cmd, dest, start_deadline=1.0, **kw))
    out = extract_audio(video, tmp_path / "f.wav", cfg["audio"], ffmpeg=ffmpeg, allow_fallback=True, progress=lambda s: None)
    assert time.monotonic() - started < 60 and seconds_of(out) == pytest.approx(20.0, abs=0.5)


def test_pyav_failing_too_is_a_clear_error_not_a_silent_wait(tmp_path, cfg):
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"garbage" * 200)
    with pytest.raises(AudioExtractionError, match="Cannot open|No audio|contains no audio"):
        extract_audio(broken, tmp_path / "g.wav", cfg["audio"], ffmpeg=None, allow_fallback=True, progress=lambda s: None)


# ----------------------------------------------------------------------------------------------- other containers, names, audio kinds
@pytest.mark.parametrize("name, fmt, acodec", [("clip.mov", "mov", "aac"), ("clip.mkv", "matroska", "aac"), ("clip.mp4", "mp4", "libmp3lame"),
                                               ("clip.mkv", "matroska", "libmp3lame")])
def test_other_containers_and_audio_codecs_pyav_and_ffmpeg(tmp_path, cfg, ffmpeg_path, name, fmt, acodec):
    source = encode(tmp_path / name, fmt, acodec, "mpeg4", seconds=6.0)
    for ffmpeg in (None, ffmpeg_path):
        out = extract_audio(source, tmp_path / f"o_{bool(ffmpeg)}.wav", cfg["audio"], ffmpeg=ffmpeg, allow_fallback=True, progress=lambda s: None)
        assert seconds_of(out) == pytest.approx(6.0, abs=0.5)


def test_unicode_and_spaces_in_the_path(tmp_path, cfg, ffmpeg_path):
    folder = tmp_path / "Ders Kayıtları çşğ"
    folder.mkdir()
    source = encode(folder / "Fizyoloji Dersi öğle 1.mp4", "mp4", "aac", "mpeg4", seconds=5.0)
    for ffmpeg in (None, ffmpeg_path):
        out = extract_audio(source, folder / f"çıktı {bool(ffmpeg)}.wav", cfg["audio"], ffmpeg=ffmpeg, allow_fallback=True,
                            progress=lambda s: None)
        assert seconds_of(out) == pytest.approx(5.0, abs=0.5)


def test_a_video_without_sound_is_a_clear_error_on_both_paths(tmp_path, cfg, ffmpeg_path):
    silent = encode(tmp_path / "silent.mp4", "mp4", "", "mpeg4", seconds=3.0)
    for ffmpeg in (None, ffmpeg_path):
        with pytest.raises(AudioExtractionError, match="no audio"):
            extract_audio(silent, tmp_path / "s.wav", cfg["audio"], ffmpeg=ffmpeg, allow_fallback=True, progress=lambda s: None)


def test_the_source_video_is_never_changed(video, tmp_path, cfg, ffmpeg_path):
    before = (video.stat().st_size, video.stat().st_mtime_ns, video.read_bytes()[:4096])
    extract_audio(video, tmp_path / "x.wav", cfg["audio"], ffmpeg=ffmpeg_path, allow_fallback=True, progress=lambda s: None)
    extract_audio_pyav(video, tmp_path / "y.wav", cfg["audio"], progress=lambda s: None)
    assert (video.stat().st_size, video.stat().st_mtime_ns, video.read_bytes()[:4096]) == before
