import json

from src.exporters import (
    OutputPaths, export_all, format_timestamp, make_output_paths, raw_transcript_text, safe_name,
    segments_to_srt, transcript_json, uncertain_json,
)
from src.schema import UncertaintyFlag, Word
from tests.conftest import make_segment


def _segments():
    return [
        make_segment(" Corticospinal tract lesions cause spasticity.", 0.0, 3.2, confidence=0.91),
        make_segment(" Şişman bağırsak ve İleum çalışması.", 3.2, 7.45, language="tr"),
    ]


def test_txt_export_is_raw_one_segment_per_line():
    text = raw_transcript_text(_segments())
    assert text == "Corticospinal tract lesions cause spasticity.\nŞişman bağırsak ve İleum çalışması.\n"


def test_srt_export_format():
    srt = segments_to_srt(_segments())
    assert srt.startswith("1\n00:00:00,000 --> 00:00:03,200\nCorticospinal tract lesions cause\nspasticity.\n")
    assert "\n\n2\n00:00:03,200 --> 00:00:07,450\n" in srt
    assert srt.endswith("\n") and not srt.endswith("\n\n")


def test_srt_timestamp_rounding_and_hours():
    assert format_timestamp(0) == "00:00:00,000"
    assert format_timestamp(3661.5) == "01:01:01,500"
    assert format_timestamp(59.9996) == "00:01:00,000"
    assert format_timestamp(-1) == "00:00:00,000"
    assert format_timestamp(1.5, ".") == "00:00:01.500"


def test_srt_skips_empty_segments_and_fixes_zero_length():
    segs = [make_segment("  ", 0, 1), make_segment("Hello there", 5.0, 5.0)]
    srt = segments_to_srt(segs)
    assert srt.startswith("1\n00:00:05,000 --> 00:00:05,500\nHello there")


def test_json_export_segment_format(tmp_path):
    seg = _segments()[0]
    seg.words = [Word(0.0, 0.5, " Corticospinal", 0.98)]
    data = transcript_json([seg], language="en", language_probability=0.99, model="large-v3",
                           backend="faster-whisper", duration=3.2, source="lecture.mp4")
    first = data["segments"][0]
    assert {"start", "end", "text", "confidence", "language", "model", "backend"} <= set(first)
    assert first["text"] == "Corticospinal tract lesions cause spasticity."
    assert first["confidence"] == 0.91
    assert first["model"] == "large-v3" and first["backend"] == "faster-whisper"
    assert first["words"][0]["probability"] == 0.98
    json.dumps(data, ensure_ascii=False)  # serialisable


def test_uncertain_json_lists_only_flagged_segments_with_reasons():
    ok, bad = _segments()
    bad.flags = [UncertaintyFlag("low_avg_logprob", "avg_logprob -1.50 < -1.0")]
    entries = uncertain_json([ok, bad])
    assert len(entries) == 1
    assert entries[0]["index"] == 1
    assert entries[0]["reasons"][0]["code"] == "low_avg_logprob"


def test_export_all_writes_every_file_utf8(tmp_path):
    paths = make_output_paths(tmp_path, "ders: anatomi/1?")
    assert paths.directory.name == "ders_ anatomi_1_"
    written = export_all(
        paths, _segments(), clean_text="Temiz metin: Şişman bağırsak.\n", metadata={"mode": "quality", "not": "ğüşıöç"},
        language="en", language_probability=0.97, model="large-v3", backend="faster-whisper",
        duration=7.45, source="ders.mp4",
    )
    expected = {paths.raw, paths.clean, paths.timestamped, paths.srt, paths.vtt, paths.json, paths.uncertain, paths.metadata}
    assert set(written.values()) == expected
    for path in expected:
        assert path.is_file() and path.stat().st_size > 0
    assert "ğüşıöç" in paths.metadata.read_text(encoding="utf-8")
    assert "Şişman bağırsak" in paths.raw.read_text(encoding="utf-8")
    assert paths.log.name == "processing.log"
    assert isinstance(paths, OutputPaths)


def test_safe_name_keeps_turkish_and_blocks_reserved_chars():
    assert safe_name("Anatomi Ders 1 - Şişli") == "Anatomi Ders 1 - Şişli"
    assert safe_name('a<b>:"c"|?*') == "a_b___c____"
    assert safe_name("...") == "untitled"
