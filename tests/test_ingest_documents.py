"""v3 documents: embedded PDF text first, OCR only for pages without text, honest sources/confidence, resume, collisions."""

import json
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from src.ingest import documents as docs
from src.ingest.documents import DocumentCancelled, DocumentError, read_document, write_document
from src.ingest.ocr import OcrResult, OcrUnavailable


def text_pdf(path: Path, pages: list[str]) -> Path:
    """A small valid PDF whose pages carry real embedded text (one line per page)."""
    objects: dict[int, bytes] = {1: b"<< /Type /Catalog /Pages 2 0 R >>", 3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"}
    kids = []
    for i, text in enumerate(pages):
        page_no = 4 + 2 * i
        kids.append(f"{page_no} 0 R")
        stream = f"BT /F1 14 Tf 40 700 Td ({text}) Tj ET".encode("latin-1") if text else b""
        objects[page_no] = (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {page_no + 1} 0 R "
                            f"/Resources << /Font << /F1 3 0 R >> >> >>").encode()
        objects[page_no + 1] = b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"
    objects[2] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>".encode()
    out, offsets = bytearray(b"%PDF-1.4\n"), {}
    for number in sorted(objects):
        offsets[number] = len(out)
        out += b"%d 0 obj\n" % number + objects[number] + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for number in sorted(objects):
        out += b"%010d 00000 n \n" % offsets[number]
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    path.write_bytes(bytes(out))
    return path


def picture(text: str = "scanned slide") -> Image.Image:
    image = Image.new("RGB", (900, 260), "white")
    ImageDraw.Draw(image).text((20, 100), text, fill="black", font=ImageFont.load_default(size=44))
    return image


def scanned_pdf(path: Path, count: int = 1) -> Path:
    images = [picture(f"scanned page {i + 1}") for i in range(count)]
    images[0].save(path, "PDF", save_all=True, append_images=images[1:])
    return path


def mixed_pdf(path: Path, text: str) -> Path:
    """One page that has both embedded text and a picture."""
    from pypdf import PdfReader, PdfWriter
    scan, base = path.with_name("m_scan.pdf"), path.with_name("m_text.pdf")
    scanned_pdf(scan)
    text_pdf(base, [text])
    page = PdfReader(str(scan)).pages[0]
    page.merge_page(PdfReader(str(base)).pages[0])
    writer = PdfWriter()
    writer.add_page(page)
    with path.open("wb") as handle:
        writer.write(handle)
    return path


class FakeOcr:
    name = "FakeOCR"

    def __init__(self, texts, confidence=0.91):
        self.texts, self.confidence, self.calls = list(texts), confidence, 0

    def recognize(self, image):
        self.calls += 1
        text = self.texts.pop(0) if len(self.texts) > 1 else self.texts[0]
        return OcrResult(text, self.confidence, self.name, lines=text.count("\n") + 1)


def factory(engine):
    created = []

    def make():
        created.append(1)
        return engine
    make.created = created
    return make


# ----------------------------------------------------------------------------------------------- PDF routing
def test_text_pdf_uses_embedded_text_only_and_never_creates_the_ocr_engine(tmp_path):
    pdf = text_pdf(tmp_path / "t.pdf", [f"Page {i} describes the glomerulus and the nephron loop in detail." for i in range(1, 4)])
    make = factory(FakeOcr(["never used"]))
    document = read_document(pdf, ocr_factory=make)
    assert [p.source for p in document.pages] == [docs.EMBEDDED] * 3 and not make.created
    assert document.source_label == "Embedded PDF Text" and document.confidence is None
    assert all(p.confidence is None for p in document.pages)                            # no invented percentage
    assert "glomerulus" in document.pages[1].text


def test_scanned_pdf_is_read_with_ocr_and_keeps_the_real_confidence(tmp_path):
    engine = FakeOcr(["Scanned slide about cardiac output", "Second scanned slide about stroke volume"], confidence=0.87)
    document = read_document(scanned_pdf(tmp_path / "s.pdf", 2), ocr_factory=factory(engine))
    assert [p.source for p in document.pages] == [docs.OCR, docs.OCR] and engine.calls == 2
    assert document.confidence == pytest.approx(0.87) and "FakeOCR" in document.source_label
    assert "stroke volume" in document.text


def test_mixed_document_text_and_scan_pages(tmp_path):
    from pypdf import PdfReader, PdfWriter
    a, b = text_pdf(tmp_path / "a.pdf", ["A page of real embedded text about the loop of Henle and urine concentration."]), \
        scanned_pdf(tmp_path / "b.pdf")
    writer = PdfWriter()
    for path in (a, b):
        writer.append(str(path))
    with (tmp_path / "ab.pdf").open("wb") as handle:
        writer.write(handle)
    engine = FakeOcr(["Scanned text on the second page"])
    document = read_document(tmp_path / "ab.pdf", ocr_factory=factory(engine))
    assert [p.source for p in document.pages] == [docs.EMBEDDED, docs.OCR] and engine.calls == 1
    assert document.text_source == docs.MIXED and "Embedded PDF Text + OCR" in document.source_label
    assert document.confidence == pytest.approx(0.91)                                   # only the OCR page has one


def test_a_page_with_text_and_a_picture_merges_without_repeating_lines(tmp_path):
    pdf = mixed_pdf(tmp_path / "m.pdf", "Short title")
    engine = FakeOcr(["Short title\nA caption only the picture contains"])
    page = read_document(pdf, ocr_factory=factory(engine)).pages[0]
    assert page.source == docs.MIXED and page.text.lower().count("short title") == 1 and "caption only the picture" in page.text


def test_missing_ocr_is_a_warning_not_a_crash(tmp_path):
    def unavailable():
        raise OcrUnavailable("Windows has no text recognition language installed.")
    pdf = tmp_path / "mix.pdf"
    from pypdf import PdfWriter
    writer = PdfWriter()
    writer.append(str(text_pdf(tmp_path / "a.pdf", ["Readable embedded text on the first page of this lecture handout."])))
    writer.append(str(scanned_pdf(tmp_path / "b.pdf")))
    with pdf.open("wb") as handle:
        writer.write(handle)
    document = read_document(pdf, ocr_factory=unavailable)
    assert "Readable embedded text" in document.text and document.pages[1].source == docs.EMPTY
    assert any("not available" in w for w in document.warnings)


def test_progress_events_name_the_stage_and_page(tmp_path):
    events = []
    read_document(scanned_pdf(tmp_path / "s.pdf", 2), ocr_factory=factory(FakeOcr(["text one here", "text two here"])), progress=events.append)
    assert {"reading", "recognizing"} <= {e["stage"] for e in events} and events[-1]["page"] == events[-1]["pages"] == 2


def test_stopping_and_resuming_does_not_read_finished_pages_again(tmp_path):
    pdf, cache = scanned_pdf(tmp_path / "s.pdf", 4), tmp_path / "work" / "s.pages.jsonl"
    first = FakeOcr(["page one text here", "page two text here"])
    done = []
    with pytest.raises(DocumentCancelled):
        read_document(pdf, ocr_factory=factory(first), cache_path=cache, should_stop=lambda: first.calls >= 2 or done.append(1))
    second = FakeOcr(["page three text here", "page four text here"])
    document = read_document(pdf, ocr_factory=factory(second), cache_path=cache)
    assert second.calls == 2 and [p.number for p in document.pages] == [1, 2, 3, 4]
    assert "page one text here" in document.pages[0].text


def test_a_changed_file_invalidates_the_cache(tmp_path):
    pdf, cache = scanned_pdf(tmp_path / "s.pdf", 1), tmp_path / "s.pages.jsonl"
    read_document(pdf, ocr_factory=factory(FakeOcr(["old words here"])), cache_path=cache)
    scanned_pdf(pdf, 2)
    engine = FakeOcr(["new words here"])
    document = read_document(pdf, ocr_factory=factory(engine), cache_path=cache)
    assert engine.calls == 2 and len(document.pages) == 2


def test_password_protected_and_damaged_pdfs_are_plain_errors(tmp_path):
    from pypdf import PdfWriter
    writer = PdfWriter()
    writer.add_blank_page(100, 100)
    writer.encrypt("secret")
    with (tmp_path / "locked.pdf").open("wb") as handle:
        writer.write(handle)
    with pytest.raises(DocumentError, match="password"):
        read_document(tmp_path / "locked.pdf")
    (tmp_path / "bad.pdf").write_bytes(b"%PDF-1.7\nnope")
    with pytest.raises(DocumentError):
        read_document(tmp_path / "bad.pdf")


# ----------------------------------------------------------------------------------------------- pictures and text files
def test_image_is_recognised_and_the_original_is_untouched(tmp_path):
    path = tmp_path / "slide.png"
    picture().save(path)
    before = path.read_bytes()
    document = read_document(path, ocr_factory=factory(FakeOcr(["Slide text about the cardiac cycle"], 0.9)))
    assert document.kind == "image" and document.pages[0].source == docs.OCR and "cardiac cycle" in document.text
    assert path.read_bytes() == before


def test_multi_page_tiff_is_one_page_per_frame(tmp_path):
    path = tmp_path / "scan.tiff"
    frames = [picture(f"frame {i}") for i in range(3)]
    frames[0].save(path, save_all=True, append_images=frames[1:])
    engine = FakeOcr(["frame one text here", "frame two text here", "frame three text here"])
    assert len(read_document(path, ocr_factory=factory(engine)).pages) == 3


def test_image_without_an_ocr_engine_is_a_clear_error(tmp_path):
    path = tmp_path / "slide.png"
    picture().save(path)

    def unavailable():
        raise OcrUnavailable("Text recognition is not available on this computer.")
    with pytest.raises(DocumentError, match="not available"):
        read_document(path, ocr_factory=unavailable)


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16", "cp1252"])
def test_text_files_in_common_encodings(tmp_path, encoding):
    path = tmp_path / "n.txt"
    path.write_bytes("Glömeruläre Filtration".encode(encoding))
    document = read_document(path)
    assert document.pages[0].source == docs.FILE and "Filtration" in document.text and document.confidence is None


# ----------------------------------------------------------------------------------------------- output
def test_outputs_are_named_after_the_source_and_never_overwrite_another_file(tmp_path):
    one, two = tmp_path / "a" / "notes.txt", tmp_path / "b" / "notes.txt"
    for path, text in ((one, "first document"), (two, "second document")):
        path.parent.mkdir()
        path.write_text(text, encoding="utf-8")
    out = tmp_path / "out"
    first = write_document(read_document(one), out)
    second = write_document(read_document(two), out)
    again = write_document(read_document(one), out)
    assert first["text"].name == "notes_text.txt" and first["json"].name == "notes.json"
    assert second["text"].parent.name == "notes (2)" and again["text"].parent == first["text"].parent
    assert first["text"].read_text(encoding="utf-8").strip() == "first document"
    data = json.loads(first["json"].read_text(encoding="utf-8"))
    assert data["text_source"] == "text_file" and data["confidence"] is None


def test_multi_page_text_carries_page_markers(tmp_path):
    pdf = text_pdf(tmp_path / "t.pdf", ["First page text is long enough to count as real text.", "Second page text is long enough as well."])
    paths = write_document(read_document(pdf, ocr_factory=factory(FakeOcr(["x"]))), tmp_path / "out")
    text = paths["text"].read_text(encoding="utf-8")
    assert "[Page 1]" in text and "[Page 2]" in text


# ----------------------------------------------------------------------------------------------- the real Windows engine
@pytest.mark.windows_only
@pytest.mark.skipif(sys.platform != "win32", reason="Windows text recognition")
def test_real_windows_ocr_reads_a_scanned_pdf(tmp_path):
    from src.ingest.ocr import create_ocr
    try:
        create_ocr("auto")
    except OcrUnavailable:
        pytest.skip("no Windows OCR language installed")
    image = Image.new("RGB", (1200, 220), "white")
    ImageDraw.Draw(image).text((30, 80), "The afferent arteriole regulates glomerular filtration", fill="black", font=ImageFont.load_default(size=48))
    image.save(tmp_path / "scan.pdf", "PDF")
    document = read_document(tmp_path / "scan.pdf")
    assert document.pages[0].source == docs.OCR and "afferent" in document.text.lower() and document.confidence is None


def test_rapidocr_adapter_reports_a_real_confidence(tmp_path):
    """The cross-platform engine (used on the Mac). Run in its own process: it must never share one with the Windows runtime."""
    import subprocess
    import importlib.util
    if importlib.util.find_spec("rapidocr_onnxruntime") is None:          # not imported here: see the docstring
        pytest.skip("RapidOCR is not installed")
    image = Image.new("RGB", (1200, 220), "white")
    ImageDraw.Draw(image).text((30, 80), "The afferent arteriole regulates glomerular filtration", fill="black", font=ImageFont.load_default(size=48))
    image.save(tmp_path / "line.png")
    code = ("import sys, json; from PIL import Image; from src.ingest.ocr import RapidOcr; "
            "r = RapidOcr().recognize(Image.open(sys.argv[1])); print(json.dumps([r.text, r.confidence]))")
    out = subprocess.run([sys.executable, "-c", code, str(tmp_path / "line.png")], capture_output=True, text=True, timeout=300,
                         cwd=Path(__file__).resolve().parent.parent)
    text, confidence = json.loads(out.stdout.strip().splitlines()[-1])
    assert "afferent" in text.lower() and 0.5 < confidence <= 1.0


def test_embedded_text_matches_the_v2_pdf_reader(tmp_path):
    """Regression: for a text PDF v3 yields exactly the pages the 0.2 study reader yields."""
    from src.study.sources import pdf_pages
    pdf = text_pdf(tmp_path / "t.pdf", ["Nephron anatomy: the glomerulus filters plasma into Bowman capsule.",
                                        "The proximal tubule reabsorbs most of the filtered sodium and water."])
    assert read_document(pdf, ocr_factory=factory(FakeOcr(["x"]))).page_texts == [p.strip() for p in pdf_pages(pdf)]


def test_text_recognition_engine_by_operating_system(monkeypatch):
    from src.ingest import ocr
    assert ocr.engines_for("win32") == [ocr.WindowsOcr]                       # never together with RapidOCR in one process
    assert ocr.engines_for("darwin") == [ocr.RapidOcr] and ocr.engines_for("linux") == [ocr.RapidOcr]
    monkeypatch.setattr(ocr.RapidOcr, "__init__", lambda self, language="auto": (_ for _ in ()).throw(ImportError("no rapidocr")))
    with pytest.raises(ocr.OcrUnavailable, match="not available on this computer"):
        ocr.create_ocr("auto", platform_name="darwin")                         # a Mac without the engine: a plain message, no crash
