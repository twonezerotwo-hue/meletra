from pathlib import Path

import pytest

from src import pdf_context
from src.pdf_context import (
    PdfContextProvider, build_context, extract_candidate_terms, find_companion_pdf,
)
from src.vocabulary import VocabularyIndex, load_terms


def make_pdf(path: Path, pages: list[str]) -> Path:
    """Write a tiny but valid text PDF (no extra dependency needed)."""
    objects: list[bytes] = []
    n = len(pages)
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n))
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode())
    for i, text in enumerate(pages):
        content_id = 4 + 2 * i
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_id} 0 R "
            f"/Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> >>".encode()
        )
        stream = f"BT /F1 12 Tf 50 700 Td ({text}) Tj ET".encode()
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for idx, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{idx} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))
    return path


@pytest.fixture(scope="module")
def vocab():
    return VocabularyIndex(load_terms("all"))


def test_companion_pdf_detected_case_insensitively(tmp_path):
    media = tmp_path / "Lecture.mp4"
    media.write_bytes(b"x")
    assert find_companion_pdf(media) is None
    pdf = make_pdf(tmp_path / "lecture.PDF", ["hello"])
    assert find_companion_pdf(media) == pdf
    other = tmp_path / "other.mp4"
    other.write_bytes(b"x")
    assert find_companion_pdf(other) is None


def test_terms_extracted_from_slides(tmp_path, vocab):
    pdf = make_pdf(tmp_path / "ders.pdf", [
        "The substantia nigra and the thalamus. Parkinson disease affects the substantia nigra.",
        "Hashimoto thyroiditis and Hashimoto thyroiditis. Reed-Sternberg cell.",
    ])
    context = build_context(pdf, vocab, min_frequency=2, max_terms=10)
    assert context.error is None and context.pages == 2
    assert context.dictionary_hits[0] == "substantia nigra"           # most frequent first
    assert "thalamus" in context.dictionary_hits
    assert any("thyroiditis" in c.lower() for c in context.candidates)  # regex candidate (freq 2)
    assert context.terms[:2] == context.dictionary_hits[:2]
    assert len(context.terms) <= 10


def test_provider_interface(tmp_path, vocab):
    pdf = make_pdf(tmp_path / "x.pdf", ["Broca area and Wernicke area."])
    provider = PdfContextProvider(pdf, vocab)
    assert "Broca area" in provider.get_terms()


def test_candidate_frequency_filter():
    text = "Graves disease appears once. Hashimoto thyroiditis appears. Hashimoto thyroiditis again."
    result = extract_candidate_terms(text, min_frequency=2)
    assert any("thyroiditis" in r.lower() for r in result)
    assert not any("Graves" in r for r in result)


def test_corrupt_pdf_never_raises(tmp_path, vocab):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF-1.4 definitely not a pdf")
    context = build_context(bad, vocab)
    assert context.terms == [] and context.error


def test_pdf_without_text_reports_ocr_hint(tmp_path, vocab):
    empty = make_pdf(tmp_path / "scan.pdf", [""])
    context = build_context(empty, vocab)
    assert context.terms == [] and "no extractable text" in context.error


def test_missing_pypdf_disables_feature_gracefully(tmp_path, vocab, monkeypatch):
    pdf = make_pdf(tmp_path / "x.pdf", ["thalamus"])

    def boom(_path):
        raise ImportError("no pypdf")

    monkeypatch.setattr(pdf_context, "extract_pdf_text", boom)
    context = build_context(pdf, vocab)
    assert context.terms == [] and "pypdf" in context.error
