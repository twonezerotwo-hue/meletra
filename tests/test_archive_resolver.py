"""BUG-01/02/04: a finished lecture is found, read and described from its own files, whatever the job record says - shaped like the real 4.1.1 profile (``output_dir`` = the chosen
root, the lecture folder only in the summary)."""

from __future__ import annotations

import json

import pytest

from src.archive.artifacts import find_artifacts
from src.archive.resolver import MISSING, RECOVERED, COMPLETED, format_clock, lecture_facts, lecture_folder, quick_status
from src.archive.transcript import load_transcript, parse_timestamped
from src.services.job_service import JobRecord, JobStore
from tests.archive_support import finished_record, write_lecture


@pytest.fixture
def store(tmp_path):
    return JobStore(tmp_path / "jobs")


def test_the_lecture_folder_is_found_when_the_record_names_the_chosen_root(tmp_path, store):
    root = tmp_path / "Medical Transcription"
    folder = write_lecture(root, "IMG_0001")
    record = finished_record(store, root, folder)
    assert record.output_dir == str(root)                                              # the shape of every real 4.1.1 job
    assert lecture_folder(record) == folder


def test_without_the_summary_the_job_id_in_the_folder_finds_it(tmp_path, store):
    root = tmp_path / "Medical Transcription"
    folder = write_lecture(root, "IMG_0001")
    write_lecture(root, "IMG_0002", source="IMG_0002.MP4", job_id="other")
    record = finished_record(store, root, folder, with_summary=False)
    assert lecture_folder(record) == folder


def test_the_recording_name_finds_it_when_nothing_else_does(tmp_path, store):
    root = tmp_path / "Medical Transcription"
    folder = write_lecture(root, "Renamed folder", job_id="")
    record = finished_record(store, root, folder, with_summary=False)
    assert lecture_folder(record) == folder                                           # metadata.json names IMG_0001.MP4


def test_a_missing_folder_is_none_not_the_parent(tmp_path, store):
    root = tmp_path / "Medical Transcription"
    root.mkdir()
    record = JobRecord(job_id="x", source="a.mp4", source_name="a.mp4", output_dir=str(root), state="completed")
    assert lecture_folder(record) is None and quick_status(record) == (MISSING, "Transcript not found")


def test_the_parent_folder_is_never_taken_for_a_lecture(tmp_path, store):
    root = tmp_path / "Medical Transcription"
    write_lecture(root, "IMG_0001")
    assert not find_artifacts(root).is_lecture


def test_legacy_names_are_recognised(tmp_path):
    folder = write_lecture(tmp_path, "L", files="legacy")
    art = find_artifacts(folder)
    assert art.has_transcript and art.legacy_names and art.clean.name == "IMG_0001_clean.txt" and art.timestamped.name == "IMG_0001_timestamped.txt"
    assert load_transcript(art).kind == "clean"


def test_only_a_timestamped_transcript_is_read_with_its_real_times(tmp_path):
    folder = write_lecture(tmp_path, "L", files="timestamped")
    transcript = load_transcript(folder)
    assert transcript.kind == "timestamped" and transcript.has_timestamps and len(transcript.segments) == 6
    first = transcript.segments[0]
    assert (first.start, first.end) == (1.0, 9.0) and first.text.startswith("Hello everyone")
    assert "[00:00" not in transcript.text and transcript.words > 40                  # the clock is not part of the text


def test_plain_text_gets_no_invented_times():
    segments = parse_timestamped("First paragraph of a lecture.\n\nSecond paragraph, no clock anywhere.")
    assert [s.text for s in segments] == ["First paragraph of a lecture.", "Second paragraph, no clock anywhere."]
    assert all(s.start is None and s.end is None for s in segments)


def test_range_and_point_timestamps_both_parse():
    segments = parse_timestamped("[00:01:05 - 00:01:09] a b c\n[01:10] d e\ncontinued\n[01:20] f")
    assert [(s.start, s.end, s.text) for s in segments] == [(65.0, 69.0, "a b c"), (70.0, 80.0, "d e continued"), (80.0, 80.0, "f")]


def test_a_unreadable_or_empty_transcript_is_not_a_transcript(tmp_path):
    folder = write_lecture(tmp_path, "L", files="empty")
    (folder / "local_clean_transcript.txt").write_text("  \n", encoding="utf-8")
    (folder / "local_transcript.json").write_bytes(b"\x00\x01 not json")
    art = find_artifacts(folder)
    assert not art.has_transcript and load_transcript(folder) is None


def test_facts_come_from_the_files_when_the_record_is_silent(tmp_path, store):
    root = tmp_path / "Medical Transcription"
    folder = write_lecture(root, "IMG_0001", duration=2548.0)
    record = JobRecord(job_id="job0000000000001", source="IMG_0001.MP4", source_name="IMG_0001.MP4", output_dir=str(root), state="completed", duration_seconds=0.0)
    facts = lecture_facts(record)
    assert facts.status == RECOVERED and facts.status_label == "Completed - metadata recovered"
    assert facts.words and facts.words > 40 and facts.duration_seconds == 2548.0 and facts.duration_source == "metadata" and facts.language == "en"
    assert facts.processing_seconds == 406.9 and facts.transcript_available


def test_unknown_stays_unknown(tmp_path, store):
    root = tmp_path / "Medical Transcription"
    folder = write_lecture(root, "L", files="legacy", duration=0.0)
    (folder / "metadata.json").unlink()
    (folder / "IMG_0001_timestamped.txt").unlink()
    record = JobRecord(job_id="x", source="IMG_0001.MP4", source_name="IMG_0001.MP4", output_dir=str(folder), state="completed")
    facts = lecture_facts(record)
    assert facts.duration_seconds is None and facts.language == "" and facts.processing_seconds is None
    assert format_clock(facts.duration_seconds) == "Unknown" and format_clock(0) == "Unknown" and format_clock(3725) == "01:02:05"


def test_a_complete_record_is_plain_completed(tmp_path, store):
    root = tmp_path / "Medical Transcription"
    folder = write_lecture(root, "IMG_0001")
    record = finished_record(store, root, folder)
    facts = lecture_facts(record)
    assert facts.status == COMPLETED and facts.words == 55 and facts.duration_seconds == 2548.0 and facts.duration_source == "recorded"


def test_a_completed_record_without_any_transcript_says_so(tmp_path, store):
    root = tmp_path / "Medical Transcription"
    folder = write_lecture(root, "IMG_0001", files="empty")
    record = finished_record(store, root, folder)
    facts = lecture_facts(record)
    assert facts.status == MISSING and not facts.transcript_available


def test_the_completion_event_stores_the_lecture_folder(tmp_path, store):
    root = tmp_path / "Medical Transcription"
    folder = write_lecture(root, "IMG_0001")
    store.save(JobRecord(job_id="job1", source="IMG_0001.MP4", source_name="IMG_0001.MP4", output_dir=str(root), state="running"))
    store.apply_event("job1", {"type": "completed", "output_dir": str(folder), "words": 55, "audio_seconds": 2548.0, "language": "en"})
    record = store.get("job1")
    assert record.output_dir == str(folder) and record.output_root == str(root) and lecture_folder(record) == folder


def test_legacy_records_without_the_new_field_read_as_before(tmp_path, store):
    path = store.directory / "old.json"
    path.write_text(json.dumps({"job_id": "old", "source": "a.mp4", "source_name": "a.mp4", "output_dir": str(tmp_path), "state": "completed"}), encoding="utf-8")
    assert store.get("old").output_root == ""


def test_a_jobs_request_file_is_not_a_second_lecture(tmp_path, store):
    """The real jobs folder holds <id>.json, <id>.request.json and <id>.events.jsonl for every job. The request file used to be listed as a QUEUED lecture named after the output folder
    ("Medical Transcription - Queued"), above the real one, and made every finished lecture also look unfinished."""
    root = tmp_path / "Medical Transcription"
    folder = write_lecture(root, "IMG_0001")
    finished_record(store, root, folder)
    (store.directory / "job0000000000001.request.json").write_text(json.dumps({"source": str(tmp_path / "IMG_0001.MP4"), "output_dir": str(root), "job_id": "job0000000000001",
                                                                                 "model_key": "whisper-large-v3", "language": "auto"}), encoding="utf-8")
    (store.directory / "job0000000000001.events.jsonl").write_text("{}\n", encoding="utf-8")
    records = store.list()
    assert [r.job_id for r in records] == ["job0000000000001"] and records[0].state == "completed"
    assert store.unfinished() == [] and store.get("job0000000000001.request") is None
    assert [r.job_id for r in store.archive()] == ["job0000000000001"]
