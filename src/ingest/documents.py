"""Documents -> text, page by page, with an honest record of where each page's text came from.

PDF page      embedded text (pypdf, as in 0.2) when the page has any; a page without text is read through OCR (its pictures),
              a sparse page gets its pictures read too and the two texts are merged without repeating lines.
Picture       OCR (every frame of a multi-page TIFF is a page).
txt / md      read as they are.

Nothing is invented: embedded text carries no confidence value, OCR carries the engine's real one (or none when the engine
reports none). Pages are processed one at a time and the finished ones are remembered in a small file, so a stopped job resumes
where it was. The result feeds the existing study code (``LectureSources.pdf_pages`` / transcript text) - no second data model.
"""

from __future__ import annotations

import codecs
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .ocr import OcrEngine, OcrUnavailable, best_orientation, create_ocr
from .router import Detected, detect

log = logging.getLogger(__name__)

MIN_PAGE_CHARS = 40                                    # a page with less embedded text than this also gets its pictures read
MIN_PICTURE_SIDE = 120                                 # smaller pictures are icons / bullets, not scanned text
MAX_PICTURES = 3                                       # per page, largest first

EMBEDDED, OCR, MIXED, FILE, EMPTY = "embedded_text", "ocr", "embedded_text+ocr", "text_file", "empty"
LABELS = {EMBEDDED: "Embedded PDF Text", OCR: "OCR", MIXED: "Embedded PDF Text + OCR", FILE: "Text file", EMPTY: "No text found"}

Progress = Callable[[dict[str, Any]], None]


class DocumentCancelled(RuntimeError):
    pass


class DocumentError(RuntimeError):
    """The document cannot be read (plain-words message)."""


@dataclass
class PageText:
    number: int
    text: str
    source: str                                        # embedded_text | ocr | embedded_text+ocr | text_file | empty
    confidence: Optional[float] = None                 # real OCR confidence (0..1) or None
    engine: str = ""


@dataclass
class Document:
    source: Path
    kind: str
    pages: list[PageText] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text.strip())

    @property
    def page_texts(self) -> list[str]:
        return [p.text for p in self.pages]

    @property
    def text_source(self) -> str:
        kinds = {p.source for p in self.pages if p.text.strip()}
        if not kinds:
            return EMPTY
        return next(iter(kinds)) if len(kinds) == 1 else MIXED

    @property
    def source_label(self) -> str:
        label = LABELS[self.text_source]
        engines = sorted({p.engine for p in self.pages if p.engine})
        return f"{label} ({', '.join(engines)})" if engines and self.text_source in (OCR, MIXED) else label

    @property
    def confidence(self) -> Optional[float]:
        values = [p.confidence for p in self.pages if p.confidence is not None]
        return sum(values) / len(values) if values else None

    def as_dict(self) -> dict[str, Any]:
        return {"source_file": self.source.name, "kind": self.kind, "text_source": self.text_source, "label": self.source_label,
                "confidence": self.confidence, "pages": [asdict(p) for p in self.pages], "warnings": self.warnings}


# ------------------------------------------------------------------------------------------------ helpers
def _norm(line: str) -> str:
    return re.sub(r"\W+", "", line.casefold())


def merge_texts(native: str, recognized: str) -> str:
    """Embedded text first, then the lines OCR found that the embedded text does not already contain."""
    have = _norm(native)
    extra = [ln for ln in recognized.splitlines() if ln.strip() and _norm(ln) and _norm(ln) not in have]
    return "\n".join(part for part in (native.strip(), "\n".join(extra)) if part)


def read_text_file(path: Path) -> str:
    data = Path(path).read_bytes()
    if data[:2] in (codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE):
        return data.decode("utf-16")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")                 # an old Windows code page


def _pictures(page) -> list:
    found = []
    for item in page.images:
        picture = item.image
        if picture is not None and min(picture.size) >= MIN_PICTURE_SIDE:
            found.append(picture)
    return sorted(found, key=lambda im: im.size[0] * im.size[1], reverse=True)[:MAX_PICTURES]


class PageCache:
    """Finished pages of one document, one JSON line each; ignored when the file changed. Lets a stopped job resume."""

    def __init__(self, path: Optional[Path], source: Path) -> None:
        self.path = Path(path) if path else None
        stat = Path(source).stat()
        self.signature = f"{stat.st_size}-{stat.st_mtime_ns}"
        self.pages: dict[int, PageText] = {}
        if self.path and self.path.is_file():
            try:
                lines = self.path.read_text(encoding="utf-8").splitlines()
                if lines and json.loads(lines[0]).get("signature") == self.signature:
                    for line in lines[1:]:
                        page = PageText(**json.loads(line))
                        self.pages[page.number] = page
            except (OSError, ValueError, TypeError):
                self.pages = {}
        if self.path and not self.pages:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"signature": self.signature}) + "\n", encoding="utf-8")

    def add(self, page: PageText) -> None:
        self.pages[page.number] = page
        if self.path:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(asdict(page), ensure_ascii=False) + "\n")


class _Ocr:
    """Creates the engine the first time a page needs it (starting the app, reading a text PDF or transcribing never loads it)."""

    def __init__(self, factory: Callable[[], OcrEngine]) -> None:
        self.factory, self.engine, self.error = factory, None, ""

    def get(self) -> Optional[OcrEngine]:
        if self.engine is None and not self.error:
            try:
                self.engine = self.factory()
            except OcrUnavailable as exc:
                self.error = str(exc)
        return self.engine


# ------------------------------------------------------------------------------------------------ readers
def read_document(source: Path | Detected, *, language: str = "auto", progress: Optional[Progress] = None,
                  should_stop: Optional[Callable[[], bool]] = None, cache_path: Optional[Path] = None,
                  ocr_factory: Optional[Callable[[], OcrEngine]] = None) -> Document:
    found = source if isinstance(source, Detected) else detect(source)
    if not found.ok or found.kind not in ("pdf", "image", "text"):
        raise DocumentError(found.problem or "This file is not a document.")
    emit = progress or (lambda event: None)
    stop = should_stop or (lambda: False)
    ocr = _Ocr(ocr_factory or (lambda: create_ocr(language)))
    document = Document(found.path, found.kind)
    if found.kind == "text":
        document.pages = [PageText(1, read_text_file(found.path), FILE)]
        emit({"type": "document_progress", "stage": "reading", "page": 1, "pages": 1, "percent": 100.0})
    elif found.kind == "pdf":
        _read_pdf(found, document, ocr, emit, stop, PageCache(cache_path, found.path))
    else:
        _read_image(found, document, ocr, emit, stop, PageCache(cache_path, found.path))
    if ocr.error and ocr.engine is None:
        document.warnings.append(ocr.error)
    if not document.text.strip():
        document.warnings.append("No text could be found in this file.")
    return document


def _read_pdf(found: Detected, document: Document, ocr: _Ocr, emit: Progress, stop: Callable[[], bool], cache: PageCache) -> None:
    from pypdf import PdfReader

    try:
        reader = PdfReader(str(found.path))
        if getattr(reader, "is_encrypted", False) and not reader.decrypt(""):
            raise DocumentError("This PDF is password-protected.")
        total = len(reader.pages)
    except DocumentError:
        raise
    except Exception as exc:                           # noqa: BLE001 - damaged files raise very different errors
        raise DocumentError(f"The PDF could not be read ({exc.__class__.__name__}).") from exc
    for index in range(total):
        number = index + 1
        if stop():
            raise DocumentCancelled("cancelled")
        if number in cache.pages:
            document.pages.append(cache.pages[number])
            continue
        emit({"type": "document_progress", "stage": "reading", "page": number, "pages": total, "percent": 100.0 * index / total})
        page = reader.pages[index]
        try:
            native = (page.extract_text() or "").strip()
        except Exception:                              # noqa: BLE001 - one bad page must not stop the document
            native = ""
            document.warnings.append(f"Page {number}: the text could not be read.")
        result = PageText(number, native, EMBEDDED if native else EMPTY)
        if len(native) < MIN_PAGE_CHARS:
            try:
                pictures = _pictures(page)
            except Exception:                          # noqa: BLE001 - an unsupported picture format
                pictures = []
                document.warnings.append(f"Page {number}: its pictures could not be opened.")
            engine = ocr.get() if pictures else None
            if pictures and engine is None:
                document.warnings.append(f"Page {number} has no text and text recognition is not available.")
            elif engine is not None:
                emit({"type": "document_progress", "stage": "recognizing", "page": number, "pages": total, "percent": 100.0 * index / total})
                results = [best_orientation(engine, picture) for picture in pictures]
                recognized = "\n".join(r.text for r in results if r.text.strip())
                scores = [r.confidence for r in results if r.confidence is not None and r.text.strip()]
                if recognized.strip():
                    merged = merge_texts(native, recognized)
                    result = PageText(number, merged, MIXED if native else OCR, sum(scores) / len(scores) if scores else None,
                                      engine.name)
        cache.add(result)
        document.pages.append(result)
    emit({"type": "document_progress", "stage": "reading", "page": total, "pages": total, "percent": 100.0})


def _read_image(found: Detected, document: Document, ocr: _Ocr, emit: Progress, stop: Callable[[], bool], cache: PageCache) -> None:
    from PIL import Image, ImageSequence

    try:
        with Image.open(found.path) as image:
            frames = [frame.copy() for frame in ImageSequence.Iterator(image)] if getattr(image, "n_frames", 1) > 1 else [image.copy()]
    except Exception as exc:                           # noqa: BLE001
        raise DocumentError(f"The picture could not be opened ({exc.__class__.__name__}).") from exc
    engine = ocr.get()
    if engine is None:
        raise DocumentError(ocr.error or "Text recognition is not available on this computer.")
    for index, frame in enumerate(frames):
        number = index + 1
        if stop():
            raise DocumentCancelled("cancelled")
        if number in cache.pages:
            document.pages.append(cache.pages[number])
            continue
        emit({"type": "document_progress", "stage": "recognizing", "page": number, "pages": len(frames), "percent": 100.0 * index / len(frames)})
        recognized = best_orientation(engine, frame)
        page = PageText(number, recognized.text.strip(), OCR if recognized.text.strip() else EMPTY, recognized.confidence, engine.name)
        cache.add(page)
        document.pages.append(page)
    emit({"type": "document_progress", "stage": "recognizing", "page": len(frames), "pages": len(frames), "percent": 100.0})


# ------------------------------------------------------------------------------------------------ output
def write_document(document: Document, output_root: Path) -> dict[str, Path]:
    """``<output_root>/<name>/<name>_text.txt`` and ``<name>.json`` (the same folder naming and collision rules as lectures)."""
    from ..exporters import safe_name
    from ..pipeline import mark_output_owner
    from ..services.transcription_service import resolve_output_name

    name = resolve_output_name(Path(output_root), document.source)
    folder = Path(output_root) / name
    folder.mkdir(parents=True, exist_ok=True)
    mark_output_owner(folder, document.source)
    stem = safe_name(document.source.stem)
    multi = len(document.pages) > 1
    body = "\n\n".join((f"[Page {p.number}]\n{p.text}" if multi else p.text) for p in document.pages if p.text.strip())
    paths = {"text": folder / f"{stem}_text.txt", "json": folder / f"{stem}.json"}
    paths["text"].write_text(body + "\n", encoding="utf-8")
    paths["json"].write_text(json.dumps(document.as_dict(), indent=1, ensure_ascii=False), encoding="utf-8")
    return paths
