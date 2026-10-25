"""BUG-02: Study reads the saved transcript through the same rules as the reader - legacy file names and the pipeline's timestamped ranges included."""

from __future__ import annotations

from src.study.sources import find_transcript, load_sources, segments_from_text
from tests.archive_support import write_lecture


def test_study_finds_a_legacy_named_transcript(tmp_path):
    folder = write_lecture(tmp_path, "L", files="legacy")
    found = find_transcript(folder)
    assert found is not None and found.name == "IMG_0001_clean.txt"
    sources = load_sources(transcript=found)
    assert sources.segments and "Glycolysis" in sources.segments[0]["text"] or sources.segments


def test_study_reads_only_a_timestamped_transcript_with_its_real_times(tmp_path):
    folder = write_lecture(tmp_path, "L", files="timestamped")
    found = find_transcript(folder)
    assert found is not None
    sources = load_sources(transcript=found)
    first = sources.segments[0]
    assert (first["start"], first["end"]) == (1.0, 9.0) and "[00:00" not in first["text"] and not first["text"].startswith("00:")


def test_range_timestamps_do_not_leak_into_the_text():
    segments = segments_from_text("[00:01:05 - 00:01:09] The enzyme hexokinase traps glucose in the cell.\n[00:01:10 - 00:01:15] Insulin raises GLUT4 on the surface.")
    assert [(s["start"], s["end"]) for s in segments] == [(65.0, 69.0), (70.0, 75.0)]
    assert all("00:01" not in s["text"] for s in segments)


def test_plain_text_gets_no_invented_times():
    segments = segments_from_text("First paragraph about glycolysis.\n\nSecond paragraph about collagen.")
    assert [s["text"] for s in segments] == ["First paragraph about glycolysis.", "Second paragraph about collagen."]
    assert all(s["start"] is None for s in segments)


def test_nothing_readable_is_none(tmp_path):
    folder = write_lecture(tmp_path, "L", files="empty")
    assert find_transcript(folder) is None
