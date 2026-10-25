"""BUG-03: lectures that exist on disk but not in the archive come back, once; nothing on disk is touched."""

from __future__ import annotations

import pytest

from src.archive.reconcile import reconcile_archive
from src.archive.resolver import lecture_folder
from src.services.job_service import JobRecord, JobStore
from tests.archive_support import finished_record, tree_hashes, write_lecture


@pytest.fixture
def profile(tmp_path):
    """The shape of the real profile: one lecture with a (root-style) record, three whose records vanished, a batch, a study-only folder, a hidden one, one whose folder is gone."""
    root = tmp_path / "Telegram" / "Medical Transcription"
    store = JobStore(tmp_path / "jobs")
    kept = write_lecture(root, "FIXTURE_0005", source="FIXTURE_0005.MP4", job_id="job-fixture-0005")
    finished_record(store, root, kept, source="FIXTURE_0005.MP4", job_id="job-fixture-0005")
    lost = [write_lecture(root, name, source=f"{name}.MP4", job_id=f"job{name}") for name in ("FIXTURE_0001", "FIXTURE_0003", "FIXTURE_0004")]
    write_lecture(root / "Batch_20261011_155826", "Inside_batch", source="Inside_batch.mp3", job_id="jobbatch")
    write_lecture(root, "Document lecture", source="", job_id="", files="empty", study=True)
    hidden = write_lecture(root, "Hidden_one", source="Hidden_one.mp4", job_id="jobhidden")
    finished_record(store, root, hidden, source="Hidden_one.mp4", job_id="jobhidden")
    store.hide("jobhidden")
    gone = JobRecord(job_id="jobgone", source="gone.mp4", source_name="gone.mp4", output_dir=str(tmp_path / "OneDrive" / "Medical Transcription"), state="completed")
    store.save(gone)
    return tmp_path, root, store, lost


def test_vanished_records_come_back_and_the_root_style_record_is_repaired(profile):
    tmp_path, root, store, lost = profile
    report = reconcile_archive(store)
    assert report.recovered == 5 and report.repaired == 2 and report.already_indexed == 0        # 3 lost + the batch lecture + the document lecture; the two root-style records
    ids = {r.job_id for r in store.list()}
    assert {"jobFIXTURE_0001", "jobFIXTURE_0003", "jobFIXTURE_0004", "jobbatch"} <= ids
    kept = store.get("job-fixture-0005")
    assert kept.output_dir == str(root / "FIXTURE_0005") and kept.output_root == str(root) and lecture_folder(kept) == root / "FIXTURE_0005"
    restored = store.get("jobFIXTURE_0001")
    assert restored.state == "completed" and restored.source_name == "FIXTURE_0001.MP4" and restored.duration_seconds == 2548.0 and restored.detected_language == "en"


def test_a_second_repair_changes_nothing_and_creates_no_duplicates(profile):
    tmp_path, root, store, lost = profile
    reconcile_archive(store)
    before = {r.job_id: (r.output_dir, r.updated) for r in store.list()}
    again = reconcile_archive(store)
    assert again.recovered == 0 and again.repaired == 0 and not again.changed
    assert {r.job_id: (r.output_dir, r.updated) for r in store.list()} == before and len(before) == len(store.list())


def test_nothing_on_disk_is_touched(profile):
    tmp_path, root, store, lost = profile
    before = tree_hashes(root)
    reconcile_archive(store)
    assert tree_hashes(root) == before


def test_the_parent_folder_and_the_batch_folder_are_not_lectures(profile):
    tmp_path, root, store, lost = profile
    reconcile_archive(store)
    names = {r.source_name for r in store.list()}
    assert "Medical Transcription" not in names and not any(n.startswith("Batch_") for n in names)
    assert "Inside_batch.mp3" in names


def test_a_lecture_the_student_removed_stays_removed(profile):
    tmp_path, root, store, lost = profile
    report = reconcile_archive(store)
    assert store.get("jobhidden").hidden is True and report.hidden_by_user == 1


def test_a_lecture_whose_folder_is_gone_is_reported_not_deleted(profile):
    tmp_path, root, store, lost = profile
    report = reconcile_archive(store)
    assert any("gone.mp4" in line for line in report.unresolved) and store.get("jobgone") is not None


def test_a_copied_folder_does_not_make_a_second_entry(profile):
    tmp_path, root, store, lost = profile
    reconcile_archive(store)
    write_lecture(root, "FIXTURE_0001 - copy", source="FIXTURE_0001.MP4", job_id="jobFIXTURE_0001")
    report = reconcile_archive(store)
    assert report.duplicates_prevented == 1 and len([r for r in store.list() if r.job_id == "jobFIXTURE_0001"]) == 1


def test_the_restored_dates_are_the_folders_not_today(profile):
    tmp_path, root, store, lost = profile
    reconcile_archive(store)
    assert store.get("jobFIXTURE_0001").created.startswith("2026-10-11")


def test_repair_keeps_the_names_a_student_gave(profile):
    tmp_path, root, store, lost = profile
    store.repair("job-fixture-0005", display_name="My neuro lecture")
    reconcile_archive(store)
    record = store.get("job-fixture-0005")
    assert record.display_name == "My neuro lecture" and lecture_folder(record) == root / "FIXTURE_0005"


def test_a_folder_without_any_job_id_gets_a_stable_id(tmp_path):
    root = tmp_path / "Medical Transcription"
    store = JobStore(tmp_path / "jobs")
    write_lecture(root, "Old lecture", source="Old.mp4", job_id="", files="legacy")
    JobRecord(job_id="seed", source="x", output_dir=str(root), state="completed")
    store.save(JobRecord(job_id="seed", source="x.mp4", source_name="x.mp4", output_dir=str(root), state="completed"))
    first = reconcile_archive(store)
    second = reconcile_archive(store)
    adopted = [r for r in store.list() if r.job_id.startswith("adopted-")]
    assert first.recovered == 1 and second.recovered == 0 and len(adopted) == 1
