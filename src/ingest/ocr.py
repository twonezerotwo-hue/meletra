"""Text recognition for scanned pages and pictures: a tiny adapter over what the operating system / one optional package offers.

* Windows: the text recognition built into Windows 10/11 (no download, any installed language, no confidence value).
* Anywhere else: RapidOCR when it is installed (real per-line confidence). It is never loaded on Windows: importing it after the
  Windows runtime crashes the process.

Nothing here is imported at start-up; ``create_ocr()`` is only called when a page really has no text.
"""

from __future__ import annotations

import asyncio
import io
import logging
import sys
from dataclasses import dataclass, field
from typing import Optional, Protocol

log = logging.getLogger(__name__)

MIN_TEXT = 8                                           # fewer letters than this: try the picture turned around


class OcrUnavailable(RuntimeError):
    """No text recognition engine can run on this computer (the message says what to do in plain words)."""


@dataclass
class OcrResult:
    text: str
    confidence: Optional[float]                        # 0..1 mean of the real per-line values; None when the engine reports none
    engine: str
    lines: int = 0
    notes: list[str] = field(default_factory=list)


class OcrEngine(Protocol):
    name: str

    def recognize(self, image) -> OcrResult: ...       # a PIL image


def prepare(image, limit: int = 4000):
    """Upright (EXIF), RGB, not larger than the engine accepts. The original file is never touched."""
    from PIL import Image, ImageOps

    image = ImageOps.exif_transpose(image)
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    if max(image.size) > limit:
        image = image.copy()
        image.thumbnail((limit, limit), Image.LANCZOS)
    return image


def _mean_line(result: OcrResult) -> float:
    return len(result.text.replace("\n", "")) / max(result.lines, 1)


def best_orientation(engine: OcrEngine, image) -> OcrResult:
    """Recognise; when almost nothing comes back, or the text is cut into many tiny lines (a sideways scan), try the picture
    turned a quarter and keep the better reading. (An upside-down page is not detected.)"""
    best = engine.recognize(image)
    if len(best.text.strip()) >= MIN_TEXT and not (best.lines >= 6 and _mean_line(best) < 6):
        return best
    for angle in (90, 270):
        trial = engine.recognize(image.rotate(angle, expand=True))
        if _mean_line(trial) > _mean_line(best):
            best = trial
    return best


class WindowsOcr:
    name = "Windows text recognition"

    def __init__(self, language: str = "auto") -> None:
        from winrt.windows.globalization import Language
        from winrt.windows.media.ocr import OcrEngine as Native

        self._native = Native
        engine = None
        if language and language != "auto":
            tags = {l.language_tag.lower(): l for l in Native.available_recognizer_languages}
            match = tags.get(language.lower()) or next((l for t, l in tags.items() if t.split("-")[0] == language.lower()), None)
            engine = Native.try_create_from_language(Language(match.language_tag)) if match else None
        self.engine = engine or Native.try_create_from_user_profile_languages()
        if self.engine is None:
            raise OcrUnavailable("Windows has no text recognition language installed (Settings > Time & language > Language).")
        self.max_side = int(Native.max_image_dimension)

    def recognize(self, image) -> OcrResult:
        image = prepare(image, min(self.max_side, 4000))
        buffer = io.BytesIO()
        image.save(buffer, "PNG")
        text = asyncio.run(self._read(buffer.getvalue()))
        return OcrResult(text, None, self.name, lines=text.count("\n") + (1 if text else 0))

    async def _read(self, png: bytes) -> str:
        from winrt.windows.graphics.imaging import BitmapDecoder
        from winrt.windows.storage.streams import DataWriter, InMemoryRandomAccessStream

        stream = InMemoryRandomAccessStream()
        writer = DataWriter(stream)
        writer.write_bytes(png)
        await writer.store_async()
        await writer.flush_async()
        writer.detach_stream()
        stream.seek(0)
        decoder = await BitmapDecoder.create_async(stream)
        bitmap = await decoder.get_software_bitmap_async()
        result = await self.engine.recognize_async(bitmap)
        return "\n".join(line.text for line in result.lines)


class RapidOcr:
    name = "RapidOCR"

    def __init__(self, language: str = "auto") -> None:
        from rapidocr_onnxruntime import RapidOCR

        self.engine = RapidOCR()

    def recognize(self, image) -> OcrResult:
        import numpy as np

        found, _ = self.engine(np.asarray(prepare(image).convert("RGB")))
        rows = found or []
        text = "\n".join(str(r[1]) for r in rows)
        scores = [float(r[2]) for r in rows]
        return OcrResult(text, sum(scores) / len(scores) if scores else None, self.name, lines=len(rows))


def engines_for(platform_name: str) -> list:
    """The engines to try on an operating system (``sys.platform``). One kind per process: both together crash (winrt + OpenCV DLLs)."""
    return [WindowsOcr] if platform_name == "win32" else [RapidOcr]


def create_ocr(language: str = "auto", platform_name: Optional[str] = None) -> OcrEngine:
    """The first engine that works here. Raises :class:`OcrUnavailable` with a plain explanation when none does."""
    tried: list[str] = []
    candidates = engines_for(platform_name or sys.platform)
    for factory in candidates:
        try:
            return factory(language)
        except OcrUnavailable as exc:
            tried.append(str(exc))
        except Exception as exc:                       # noqa: BLE001 - a missing package or a Windows API problem
            log.info("%s is not available: %s", factory.__name__, exc)
            tried.append(f"{factory.__name__}: {exc.__class__.__name__}")
    raise OcrUnavailable("Text recognition is not available on this computer. " + " ".join(tried))
