import subprocess
import wave
from pathlib import Path

import pytest

from src import audio
from src.audio import (
    AudioExtractionError, FFmpegNotFoundError, UnsupportedMediaError, extract_audio, list_media_files,
    locate_ffmpeg, supported_extensions, validate_media_file,
)


def test_supported_extensions_cover_the_required_formats(cfg):
    exts = supported_extensions(cfg["audio"])
    assert {".mp4", ".mp3", ".m4a", ".wav"} <= exts


def test_unsupported_extension_is_rejected_with_a_helpful_message(tmp_path, cfg):
    bad = tmp_path / "notes.docx"
    bad.write_text("not media")
    with pytest.raises(UnsupportedMediaError, match=r"Unsupported file type '\.docx'.*\.mp4"):
        validate_media_file(bad, cfg["audio"])


def test_missing_file_is_reported(tmp_path, cfg):
    with pytest.raises(FileNotFoundError):
        validate_media_file(tmp_path / "ghost.mp4", cfg["audio"])


def test_extension_check_is_case_insensitive(tmp_path, cfg):
    f = tmp_path / "LECTURE.MP4"
    f.write_bytes(b"x")
    assert validate_media_file(f, cfg["audio"]) == f


def test_list_media_files_filters_and_sorts(tmp_path, cfg):
    for name in ("b.mp3", "a.MP4", "slides.pdf", "notes.txt", "c.wav"):
        (tmp_path / name).write_bytes(b"x")
    names = [p.name for p in list_media_files(tmp_path, cfg["audio"])]
    assert names == ["a.MP4", "b.mp3", "c.wav"]


def test_missing_ffmpeg_gives_install_instructions(monkeypatch, cfg):
    monkeypatch.setattr(audio, "find_executable", lambda *a, **k: None)
    with pytest.raises(FFmpegNotFoundError, match="winget install Gyan.FFmpeg"):
        locate_ffmpeg(cfg["audio"])


def test_extract_audio_without_ffmpeg_raises_before_running(monkeypatch, tmp_path, cfg):
    media = tmp_path / "a.wav"
    media.write_bytes(b"RIFF")
    monkeypatch.setattr(audio, "find_executable", lambda *a, **k: None)
    with pytest.raises(FFmpegNotFoundError):
        extract_audio(media, tmp_path / "out.wav", cfg["audio"])


def test_unknown_preprocessing_profile_rejected(tmp_path, cfg, ffmpeg_path):
    media = tmp_path / "a.wav"
    media.write_bytes(b"RIFF")
    with pytest.raises(ValueError, match="preprocessing"):
        extract_audio(media, tmp_path / "o.wav", cfg["audio"], preprocessing="denoise-max", ffmpeg=ffmpeg_path)


def _make_video(ffmpeg: Path, target: Path, with_audio: bool = True) -> Path:
    """A 1 s test video. The native mpeg4 encoder exists in every FFmpeg build (the bundled LGPL one has no libx264)."""
    cmd = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
           "-f", "lavfi", "-i", "color=c=black:s=64x64:d=1:r=10"]
    if with_audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:duration=1:sample_rate=44100"]
    cmd += ["-c:v", "mpeg4", "-pix_fmt", "yuv420p"] + (["-c:a", "aac", "-ac", "2"] if with_audio else []) + [str(target)]
    subprocess.run(cmd, check=True, capture_output=True)
    return target


@pytest.mark.ffmpeg
def test_video_audio_is_extracted_as_16k_mono_pcm(tmp_path, cfg, ffmpeg_path):
    video = _make_video(ffmpeg_path, tmp_path / "lecture.mp4")
    wav = extract_audio(video, tmp_path / "out" / "audio.wav", cfg["audio"], ffmpeg=ffmpeg_path)
    with wave.open(str(wav)) as handle:
        assert handle.getframerate() == 16000
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert 0.8 < handle.getnframes() / 16000 < 1.3


@pytest.mark.ffmpeg
def test_light_preprocessing_runs(tmp_path, cfg, ffmpeg_path):
    video = _make_video(ffmpeg_path, tmp_path / "lecture.mp4")
    wav = extract_audio(video, tmp_path / "audio.wav", cfg["audio"], "light", ffmpeg=ffmpeg_path)
    assert wav.stat().st_size > 1000


@pytest.mark.ffmpeg
def test_video_without_audio_track_is_a_clear_error(tmp_path, cfg, ffmpeg_path):
    silent = _make_video(ffmpeg_path, tmp_path / "silent.mp4", with_audio=False)
    with pytest.raises(AudioExtractionError, match="no audio track"):
        extract_audio(silent, tmp_path / "audio.wav", cfg["audio"], ffmpeg=ffmpeg_path)


@pytest.mark.ffmpeg
def test_corrupt_media_is_a_clear_error(tmp_path, cfg, ffmpeg_path):
    broken = tmp_path / "broken.mp3"
    broken.write_bytes(b"this is not an mp3 file at all")
    with pytest.raises(AudioExtractionError):
        extract_audio(broken, tmp_path / "audio.wav", cfg["audio"], ffmpeg=ffmpeg_path)
