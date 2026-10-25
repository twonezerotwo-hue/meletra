"""v3 input router: the kind comes from the file's content, not its extension; the 10-file limit is an error, never a silent cut."""

import struct
import zlib
from pathlib import Path

import av
import numpy as np
import pytest

from src.ingest import router
from src.ingest.router import MAX_QUEUE, classify, detect
from tests.conftest import write_silent_wav


def encode(path: Path, fmt: str, acodec: str = "", vcodec: str = "", seconds: float = 1.0, rate: int = 44100) -> Path:
    """A tiny real media file written with PyAV (the same library the app reads with)."""
    with av.open(str(path), "w", format=fmt) as out:
        audio = out.add_stream(acodec, rate=rate) if acodec else None
        video = out.add_stream(vcodec, rate=10) if vcodec else None
        if video is not None:
            video.width, video.height, video.pix_fmt = 64, 64, "yuv420p"
        if audio is not None:
            resampler = av.AudioResampler(format=audio.format.name, layout=audio.layout.name, rate=audio.rate)
            tone = (np.sin(np.arange(int(rate * seconds)) / 20.0) * 3000).astype(np.int16)[None, :]
            frame = av.AudioFrame.from_ndarray(tone, format="s16", layout="mono")
            frame.sample_rate = rate
            for piece in resampler.resample(frame):
                out.mux(audio.encode(piece))
            out.mux(audio.encode(None))
        if video is not None:
            for i in range(int(10 * seconds)):
                image = av.VideoFrame.from_ndarray(np.full((64, 64, 3), i * 20 % 255, dtype=np.uint8), format="rgb24")
                out.mux(video.encode(image.reformat(format="yuv420p")))
            out.mux(video.encode(None))
    return path


def png(path: Path) -> Path:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    raw = zlib.compress(b"\x00\xff\xff\xff" * 4)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 4, 8, 2, 0, 0, 0)) + chunk(b"IDAT", raw) + chunk(b"IEND", b""))
    return path


def pdf(path: Path, pages: int = 3, password: str = "") -> Path:
    from pypdf import PdfWriter
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(200, 200)
    if password:
        writer.encrypt(password)
    with path.open("wb") as handle:
        writer.write(handle)
    return path


@pytest.mark.parametrize("name, fmt, codec", [("a.mp3", "mp3", "libmp3lame"), ("a.m4a", "ipod", "aac"), ("a.flac", "flac", "flac"),
                                              ("a.opus", "opus", "libopus"), ("a.aac", "adts", "aac"), ("a.mp4", "mp4", "aac")])
def test_audio_files_are_recognised_by_content(tmp_path, name, fmt, codec):
    found = detect(encode(tmp_path / name, fmt, codec, rate=48000 if codec == "libopus" else 44100))
    assert found.kind == "audio" and found.ok and not found.ext_mismatch and not found.needs_wav
    assert found.duration_s == pytest.approx(1.0, abs=0.2) and found.label.startswith("Audio")


def test_wav_is_audio(tmp_path):
    assert detect(write_silent_wav(tmp_path / "a.wav", 1.0)).kind == "audio"


@pytest.mark.parametrize("name, fmt, acodec, vcodec", [("v.mp4", "mp4", "aac", "mpeg4"), ("v.mov", "mov", "aac", "mpeg4"),
                                                       ("v.mkv", "matroska", "aac", "mpeg4"), ("v.avi", "avi", "libmp3lame", "mpeg4")])
def test_video_files_are_recognised(tmp_path, name, fmt, acodec, vcodec):
    found = detect(encode(tmp_path / name, fmt, acodec, vcodec))
    assert found.kind == "video" and found.ok and found.label.startswith("Video")


def test_m4v_is_accepted_as_it_is(tmp_path):
    found = detect(encode(tmp_path / "v.m4v", "mp4", "aac", "mpeg4"))
    assert found.kind == "video" and found.ok and not found.needs_wav


def test_a_video_without_sound_is_refused_in_plain_words(tmp_path):
    found = detect(encode(tmp_path / "silent.mp4", "mp4", "", "mpeg4"))
    assert found.kind == "video" and not found.ok and "sound" in found.problem


def test_content_beats_the_extension(tmp_path):
    mp3 = encode(tmp_path / "x.mp3", "mp3", "libmp3lame")
    as_pdf = tmp_path / "really_audio.pdf"
    as_pdf.write_bytes(mp3.read_bytes())
    found = detect(as_pdf)
    assert found.kind == "audio" and found.ext_mismatch and found.needs_wav
    fake_mp3 = tmp_path / "really_pdf.mp3"
    fake_mp3.write_bytes(pdf(tmp_path / "p.pdf").read_bytes())
    found = detect(fake_mp3)
    assert found.kind == "pdf" and found.ext_mismatch and found.pages == 3
    wrong_image = tmp_path / "really_png.jpg"
    wrong_image.write_bytes(png(tmp_path / "i.png").read_bytes())
    found = detect(wrong_image)
    assert found.kind == "image" and found.format == "PNG" and found.ext_mismatch


def test_pdf_page_count_encryption_and_damage(tmp_path):
    assert detect(pdf(tmp_path / "a.pdf", pages=23)).label == "PDF • 23 pages"
    assert detect(pdf(tmp_path / "one.pdf", pages=1)).label == "PDF • 1 page"
    locked = detect(pdf(tmp_path / "locked.pdf", password="secret"))
    assert not locked.ok and "password" in locked.problem
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"%PDF-1.7\nnot really a pdf\n")
    assert detect(broken).kind == "pdf" and not detect(broken).ok


def test_images_by_signature(tmp_path):
    assert detect(png(tmp_path / "a.png")).kind == "image"
    for name, head in [("a.jpg", b"\xff\xd8\xff\xe0" + b"\0" * 20), ("a.webp", b"RIFF\x24\0\0\0WEBPVP8 " + b"\0" * 20),
                       ("a.tiff", b"II*\x00" + b"\0" * 20), ("a.bmp", b"BM" + b"\0" * 40)]:
        (tmp_path / name).write_bytes(head)
        assert detect(tmp_path / name).kind == "image", name


def test_text_files(tmp_path):
    (tmp_path / "n.txt").write_text("Lecture notes about the nephron.\nSecond line.", encoding="utf-8")
    (tmp_path / "n.md").write_text("# Heading\n\nTürkçe metin", encoding="utf-8")
    assert detect(tmp_path / "n.txt").kind == "text" and detect(tmp_path / "n.md").kind == "text"
    (tmp_path / "binary.txt").write_bytes(bytes(range(256)) * 4)
    assert not detect(tmp_path / "binary.txt").ok
    (tmp_path / "n.xyz").write_text("plain text but not a supported type")
    assert not detect(tmp_path / "n.xyz").ok and "not a supported" in detect(tmp_path / "n.xyz").problem


def test_text_with_a_media_extension_is_not_media(tmp_path):
    fake = tmp_path / "fake.m4a"
    fake.write_text("this is only text")
    assert not detect(fake).ok


def test_missing_and_empty_files(tmp_path):
    assert "not found" in detect(tmp_path / "nope.mp3").problem
    (tmp_path / "empty.pdf").write_bytes(b"")
    assert "empty" in detect(tmp_path / "empty.pdf").problem


def test_selection_drops_duplicates_keeps_order_and_reports_problems(tmp_path):
    a = write_silent_wav(tmp_path / "b.wav", 0.5)
    b = pdf(tmp_path / "a.pdf")
    bad = tmp_path / "junk.xyz"
    bad.write_bytes(b"\x00\x01\x02")
    selection = classify([a, b, a, bad])
    assert [d.path.name for d in selection.items] == ["b.wav", "a.pdf"]
    assert len(selection.problems) == 1 and "junk.xyz" in selection.problems[0]


def test_ten_audio_files_are_fine_and_eleven_are_an_error(tmp_path):
    files = [write_silent_wav(tmp_path / f"l{i:02d}.wav", 0.2) for i in range(MAX_QUEUE + 1)]
    ok = classify(files[:MAX_QUEUE])
    assert not ok.problems and not ok.too_many and len(ok.media) == MAX_QUEUE
    too_many = classify(files)
    assert too_many.too_many and any("limit is 10" in p for p in too_many.problems)
    assert len(too_many.items) == MAX_QUEUE + 1                                     # nothing was silently dropped


def test_documents_do_not_count_towards_the_audio_limit(tmp_path):
    files = [write_silent_wav(tmp_path / f"l{i}.wav", 0.2) for i in range(MAX_QUEUE)] + [pdf(tmp_path / "s.pdf")]
    assert not classify(files).too_many


def test_the_file_dialog_filter_lists_every_supported_type():
    text = router.file_dialog_filter()
    for ext in (".mp3", ".m4v", ".pdf", ".md", ".webp", ".tiff"):
        assert "*" + ext in text
