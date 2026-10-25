"""Chunks with timestamps / pages, topics, slide links and the local BM25 index (no model, no network)."""

import json

import pytest

from src.study import index as ix
from src.study.index import DocumentIndex, build_index, pdf_chunks, transcript_chunks
from src.study.models import SOURCE_PDF, SOURCE_TRANSCRIPT, Chunk, SourceRef
from src.study.sources import (SourceError, find_transcript, load_sources, pdf_pages, segments_from_json, segments_from_text)
from study_support import GLYCOLYSIS, SLIDES, lecture_index, segments


def test_transcript_chunks_keep_start_and_end_time_and_order():
    chunks = transcript_chunks(segments())
    assert chunks and all(c.source == SOURCE_TRANSCRIPT for c in chunks)
    assert chunks[0].start == 0.0
    assert all(a.end <= b.start + 0.6 for a, b in zip(chunks, chunks[1:]))
    assert all(60 <= len(c.text.split()) <= ix.TRANSCRIPT_CHUNK_MAX_WORDS + 30 for c in chunks[:-1])
    assert [c.id for c in chunks] == [f"t{i:04d}" for i in range(len(chunks))]


def test_long_pause_closes_a_chunk():
    segs = [{"start": 0, "end": 5, "text": "Insulin lowers blood glucose."}, {"start": 60, "end": 65, "text": "Glucagon raises it again."}]
    chunks = transcript_chunks(segs)
    assert len(chunks) == 2 and chunks[1].start == 60.0


def test_plain_text_without_times_makes_chunks_without_times():
    chunks = transcript_chunks(segments_from_text("Insulin lowers blood glucose.\n\nGlucagon raises blood glucose."))
    assert chunks and all(c.start is None and c.end is None for c in chunks)
    assert "Transcript" in chunks[0].label()


def test_text_transcript_with_times_is_parsed():
    segs = segments_from_text("00:10 Hello class\ncontinued here\n01:05:02 Second part")
    assert segs[0]["start"] == 10.0 and "continued here" in segs[0]["text"]
    assert segs[1]["start"] == 3902.0 and segs[0]["end"] == 3902.0


def test_pdf_chunks_keep_lines_pages_and_titles():
    chunks = pdf_chunks(SLIDES + ["", "   "])
    assert [c.page for c in chunks] == [1, 2]
    assert chunks[0].title == "Glycolysis" and "\n" in chunks[0].text
    assert chunks[0].label() == "Slide 1"


def test_long_slide_page_is_split_but_keeps_its_page_number():
    page = "Title\n" + "\n".join(f"Bullet point number {i} about the pathway and its enzymes" for i in range(40))
    chunks = pdf_chunks([page])
    assert len(chunks) > 1 and {c.page for c in chunks} == {1}
    assert chunks[0].id == "p001" and chunks[1].id == "p001-2"


def test_topics_split_where_the_subject_changes():
    idx = lecture_index()
    assert len(idx.topics) == 2
    first, second = idx.topics
    assert "glyc" in first.name.casefold() or "gluc" in first.name.casefold()
    assert "collagen" in second.name.casefold()
    assert first.start == 0.0 and second.start > first.start
    covered = [i for t in idx.topics for i in t.chunk_ids]
    assert covered == [c.id for c in idx.of_source(SOURCE_TRANSCRIPT)]            # every chunk in exactly one topic, in order
    assert all(c.topic for c in idx.of_source(SOURCE_TRANSCRIPT))


def test_slides_are_linked_to_the_passages_that_talk_about_them():
    idx = lecture_index(with_slides=True)
    assert idx.mode == "pdf+transcript"
    glyc, coll = idx.topics
    assert glyc.pages == [1] and coll.pages == [2]
    first = idx.chunks[0]
    assert [s.page for s in idx.linked_slides(first)] == [1]
    assert {c.page for c in idx.chunks_of_topic(coll.name) if c.source == SOURCE_PDF} == {2}


def test_pdf_only_topics_follow_slide_titles():
    idx = build_index([], SLIDES)
    assert idx.mode == "pdf" and idx.topic_names() == ["Glycolysis", "Collagen synthesis"]


def test_search_finds_the_relevant_passage_first():
    idx = lecture_index()
    hit, score = idx.search("which vitamin does proline hydroxylase need", k=3)[0]
    assert "vitamin" in hit.text.casefold() and score > 0
    assert idx.search("zzzz qqqq") == []
    only_slides = lecture_index(with_slides=True).search("hexokinase", k=5, source=SOURCE_PDF)
    assert only_slides and all(c.source == SOURCE_PDF for c, _ in only_slides)


def test_index_survives_a_json_round_trip():
    idx = lecture_index(with_slides=True)
    again = DocumentIndex.from_dict(json.loads(json.dumps(idx.to_dict())))
    assert [c.to_dict() for c in again.chunks] == [c.to_dict() for c in idx.chunks]
    assert again.links == idx.links and again.topic_names() == idx.topic_names()
    assert again.search("collagen")[0][0].id == idx.search("collagen")[0][0].id


def test_source_ref_labels():
    ref = SourceRef.of(Chunk("t1", SOURCE_TRANSCRIPT, "x", start=750.0, end=778.0))
    assert ref.label == "Transcript 12:30-12:58"
    assert SourceRef.of(Chunk("p3", SOURCE_PDF, "x", page=14)).label == "Slide 14"


# ----------------------------------------------------------------------------------------------------------------- loading sources
def test_transcript_json_of_the_transcriber_is_read(tmp_path):
    path = tmp_path / "local_transcript.json"
    path.write_text(json.dumps({"language": "en", "segments": [{"start": 1.0, "end": 3.5, "text": " Hello. "}, {"start": 4, "end": 5, "text": ""}]}),
                    encoding="utf-8")
    segs, language = segments_from_json(path)
    assert language == "en" and segs == [{"start": 1.0, "end": 3.5, "text": " Hello. "}]
    assert find_transcript(tmp_path) == path


def test_missing_or_broken_inputs_give_a_readable_error(tmp_path):
    with pytest.raises(SourceError, match="nothing to study"):
        load_sources()
    bad = tmp_path / "local_transcript.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(SourceError, match="could not be read"):
        load_sources(transcript=bad)
    broken_pdf = tmp_path / "broken.pdf"
    broken_pdf.write_bytes(b"%PDF-1.4 this is not a pdf")
    with pytest.raises(SourceError):
        pdf_pages(broken_pdf)


def test_broken_pdf_does_not_block_the_transcript(tmp_path):
    good = tmp_path / "local_transcript.json"
    good.write_text(json.dumps({"segments": [{"start": 0, "end": 2, "text": "Glycolysis happens in the cytoplasm."}]}), encoding="utf-8")
    broken_pdf = tmp_path / "broken.pdf"
    broken_pdf.write_bytes(b"not a pdf")
    sources = load_sources(transcript=good, pdf=broken_pdf)
    assert sources.mode == "transcript" and sources.segments and any("Slides not used" in w for w in sources.warnings)


def test_pdf_with_text_is_read(tmp_path):
    pypdf = pytest.importorskip("pypdf")
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=200)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"), NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})})
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 14 Tf 20 100 Td (Hexokinase phosphorylates glucose) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    path = tmp_path / "lecture.pdf"
    with path.open("wb") as handle:
        writer.write(handle)
    assert pypdf and "Hexokinase" in pdf_pages(path)[0]
    sources = load_sources(pdf=path)
    assert sources.mode == "pdf" and sources.pdf_pages


def test_title_case_keeps_terms_that_are_written_in_a_special_way():
    from src.study.text import title_case

    assert [title_case(t) for t in ("mTOR", "pH", "glycolysis", "ATP", "DNA polymerase", " hexokinase ", "")] == [
        "mTOR", "pH", "Glycolysis", "ATP", "DNA polymerase", "Hexokinase", ""]
