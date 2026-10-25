"""v3 input router: what is this file really, and where does it go?

The kind is decided from the file's CONTENT (magic bytes, then the media container through PyAV), never from the extension alone:
a ``.pdf`` that is a PNG is an image, an ``.mp3`` that is an MP4 with video is a video. The router only classifies; the work stays in
the proven code: audio and video -> the existing transcription pipeline, PDF -> ``src.ingest.documents`` (pypdf, OCR only for pages
without text), images -> OCR, txt/md -> read as they are.
"""

from __future__ import annotations

import codecs
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

log = logging.getLogger(__name__)

MAX_QUEUE = 10                                   # audio / video files in one batch

AUDIO_EXT = (".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus")
VIDEO_EXT = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v")
TEXT_EXT = (".txt", ".md", ".markdown")
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp")
DOCUMENT_EXT = (".pdf",) + TEXT_EXT
ALL_EXT = AUDIO_EXT + VIDEO_EXT + DOCUMENT_EXT + IMAGE_EXT
KINDS = ("audio", "video", "pdf", "text", "image")

_IMAGE_EXT = {"png": (".png",), "jpeg": (".jpg", ".jpeg"), "webp": (".webp",), "tiff": (".tif", ".tiff"), "bmp": (".bmp",)}


def file_dialog_filter() -> str:
    """Qt file-dialog filter for "SELECT FILES"."""
    patterns = " ".join("*" + e for e in ALL_EXT)
    return f"Lectures and documents ({patterns});;All files (*)"


@dataclass
class Detected:
    path: Path
    kind: str = "unsupported"                    # audio | video | pdf | text | image | unsupported
    format: str = ""                             # what the content really is: "PDF", "PNG", "MP4" ...
    duration_s: Optional[float] = None
    pages: Optional[int] = None
    problem: str = ""                            # why it cannot be used (plain words)
    ext_mismatch: bool = False                   # the extension says something else than the content
    needs_wav: bool = False                      # media whose extension the transcription pipeline refuses: convert first

    @property
    def ok(self) -> bool:
        return self.kind in KINDS and not self.problem

    @property
    def is_media(self) -> bool:
        return self.kind in ("audio", "video")

    @property
    def label(self) -> str:
        if self.kind in ("audio", "video"):
            return f"{self.kind.title()} • {_clock(self.duration_s)}" if self.duration_s else self.kind.title()
        if self.kind == "pdf":
            return f"PDF • {self.pages} page{'s' if self.pages != 1 else ''}" if self.pages else "PDF"
        return {"text": "Text", "image": "Image"}.get(self.kind, "Unsupported")


def _clock(seconds: Optional[float]) -> str:
    total = int(seconds or 0)
    hours, rest = divmod(total, 3600)
    return f"{hours}:{rest // 60:02d}:{rest % 60:02d}" if hours else f"{rest // 60}:{rest % 60:02d}"


# ------------------------------------------------------------------------------------------------ content signatures
def signature(head: bytes) -> str:
    """"pdf", an image name, "media" (a known audio/video container) or "" (not recognised)."""
    if b"%PDF-" in head[:1024]:
        return "pdf"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if head.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return "tiff"
    if head[:2] == b"BM" and len(head) > 14:
        return "bmp"
    if (head[:4] == b"RIFF" and head[8:12] in (b"WAVE", b"AVI ")) or head[:3] == b"ID3" or head[:4] in (b"fLaC", b"OggS", b"\x1aE\xdf\xa3") \
            or head[4:8] == b"ftyp" or head[:4] == b"\x30\x26\xb2\x75" or (head[:2] in (b"\xff\xf1", b"\xff\xf9")) \
            or (len(head) > 1 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0 and head[:2] != codecs.BOM_UTF16_LE):   # FF FE: UTF-16 text
        return "media"
    return ""


def _looks_like_text(head: bytes) -> bool:
    if head.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return True
    if b"\x00" in head:
        return False
    try:
        codecs.getincrementaldecoder("utf-8")().decode(head, final=False)
        return True
    except UnicodeDecodeError:
        controls = sum(1 for b in head if b < 9 or 13 < b < 32)           # an old 8-bit code page is still text
        return controls <= max(2, len(head) // 100)


def _probe_media(path: Path) -> tuple[bool, bool, Optional[float], str]:
    """(has_audio, has_video, seconds, container) with PyAV; a cover picture inside an audio file does not make it a video."""
    try:
        import av

        with av.open(str(path)) as container:
            audio = any(s.type == "audio" for s in container.streams)
            video = any(s.type == "video" and not (s.disposition & av.stream.Disposition.attached_pic) for s in container.streams)
            seconds = float(container.duration) / 1_000_000.0 if container.duration else None
            return audio, video, seconds, container.format.name.split(",")[0]
    except Exception as exc:                      # noqa: BLE001 - any unreadable file is simply "not media"
        log.debug("PyAV could not open %s: %s", path.name, exc)
        return False, False, None, ""


def _pipeline_extensions() -> set[str]:
    from ..audio import supported_extensions
    from ..config import load_transcription_config
    return supported_extensions(load_transcription_config()["audio"])


def _pdf_pages(path: Path) -> tuple[Optional[int], str]:
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        if getattr(reader, "is_encrypted", False) and not reader.decrypt(""):      # NOT_DECRYPTED == 0
            return None, "This PDF is password-protected."
        return len(reader.pages), ""
    except Exception as exc:                      # noqa: BLE001
        return None, f"The PDF could not be read ({exc.__class__.__name__})."


# ------------------------------------------------------------------------------------------------ detection
def detect(path: Path | str, accepted_media_ext: Optional[set[str]] = None) -> Detected:
    path = Path(path)
    out = Detected(path)
    try:
        if not path.is_file():
            out.problem = "The file was not found."
            return out
        if path.stat().st_size == 0:
            out.problem = "The file is empty."
            return out
        with path.open("rb") as handle:
            head = handle.read(8192)
    except OSError as exc:
        out.problem = f"The file could not be opened ({exc.__class__.__name__})."
        return out
    ext, sig = path.suffix.lower(), signature(head)

    if sig == "pdf":
        out.kind, out.format, out.ext_mismatch = "pdf", "PDF", ext != ".pdf"
        out.pages, out.problem = _pdf_pages(path)
    elif sig in _IMAGE_EXT:
        out.kind, out.format, out.ext_mismatch = "image", sig.upper(), ext not in _IMAGE_EXT[sig]
    else:
        has_audio, has_video, seconds, container = _probe_media(path) if (sig == "media" or ext in AUDIO_EXT + VIDEO_EXT) \
            else (False, False, None, "")
        if has_audio or has_video:
            out.kind, out.format, out.duration_s = ("video" if has_video else "audio"), container.upper(), seconds
            out.ext_mismatch = ext not in AUDIO_EXT + VIDEO_EXT
            if not has_audio:
                out.kind, out.problem = "video", "This video has no sound track to transcribe."
            allowed = accepted_media_ext if accepted_media_ext is not None else _pipeline_extensions()
            out.needs_wav = ext not in allowed
        elif sig == "media":
            out.problem = "This looks like a media file but it could not be read (damaged or unsupported)."
        elif ext in TEXT_EXT and _looks_like_text(head):
            out.kind, out.format = "text", "Text"
        elif ext in TEXT_EXT:
            out.problem = "This file is not readable text."
        else:
            out.problem = f"'{path.name}' is not a supported file type."
    return out


@dataclass
class Selection:
    items: list[Detected]
    problems: list[str]

    @property
    def media(self) -> list[Detected]:
        return [d for d in self.items if d.is_media]

    @property
    def too_many(self) -> bool:
        return len(self.media) > MAX_QUEUE


def classify(paths: Iterable[Path | str]) -> Selection:
    """Detect every selected file (order kept, duplicates dropped). Unusable files are listed in ``problems``; more than
    :data:`MAX_QUEUE` audio/video files is an error for the caller to show - nothing is silently cut."""
    seen: set[str] = set()
    items: list[Detected] = []
    problems: list[str] = []
    allowed = _pipeline_extensions()
    for raw in paths:
        key = str(Path(raw).resolve()).lower()
        if key in seen:
            continue
        seen.add(key)
        found = detect(raw, allowed)
        if found.ok:
            items.append(found)
        else:
            problems.append(f"{Path(raw).name}: {found.problem}")
    if sum(1 for d in items if d.is_media) > MAX_QUEUE:
        problems.append(f"You selected {sum(1 for d in items if d.is_media)} audio/video files. The limit is {MAX_QUEUE} at a time.")
    return Selection(items, problems)
