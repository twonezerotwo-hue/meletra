"""Phase 21 scenarios for the lecture archive: a renamed lecture, a restart, a deleted recording, and transcripts that are partial, damaged or unreadable. In every one the status the
cards show is the status the result screen shows, no transcript is invented, and nothing in the lecture folder is changed by looking."""

from __future__ import annotations

import json

import pytest

from src.archive.reconcile import reconcile_archive
from src.archive.resolver import MISSING, lecture_facts, lecture_folder, quick_status
from src.gui.main_window import MainWindow
from src.services.job_service import JobStore
from tests.archive_support import finished_record, tree_hashes, write_lecture
from tests.gui_support import ScriptedLauncher, make_services, settle


def world(tmp_path, **kwargs):
    store = JobStore(tmp_path / "jobs")
    root = tmp_path / "Medical Transcription"
    folder = write_lecture(root, "IMG_0001", **kwargs)
    return store, root, folder, finished_record(store, root, folder)


def test_renaming_a_lecture_changes_only_the_name_it_is_shown_by(tmp_path):
    store, root, folder, record = world(tmp_path, study=True)
    before = tree_hashes(folder)
    store.update(record.job_id, display_name="Neuroanatomy - ventricles")
    renamed = store.get(record.job_id)
    assert lecture_folder(renamed) == folder and quick_status(renamed)[1] == "Completed" and lecture_facts(renamed).words
    assert tree_hashes(folder) == before                                              # the lecture's folder, transcript and study pack are untouched
    report = reconcile_archive(store)
    assert report.recovered == 0 and report.duplicates_prevented == 0 and [r.job_id for r in store.list()] == [record.job_id]       # no second entry appears


def test_a_restart_shows_the_same_lecture_with_the_same_status(qapp, tmp_path):
    services = make_services(tmp_path, opened=[])
    root = tmp_path / "Medical Transcription"
    folder = write_lecture(root, "IMG_0001")
    record = finished_record(services.jobs, root, folder)
    first = MainWindow(services, ScriptedLauncher())
    first.files_page.refresh()
    chips = [c.status.text() for c in first.files_page.cards]
    settle(first, first.home, first.study)
    first.deleteLater()
    second_services = make_services(tmp_path, opened=[])                              # the same data folder, a new program start
    second = MainWindow(second_services, ScriptedLauncher())
    second.files_page.refresh()
    assert [c.record.job_id for c in second.files_page.cards] == [record.job_id] and [c.status.text() for c in second.files_page.cards] == chips == ["Completed"]
    second.open_lecture_record(second_services.jobs.get(record.job_id))
    assert second.simple_result.read_button.isEnabled() and second.current_output == folder
    settle(second, second.home, second.study)
    second.deleteLater()


def test_a_deleted_recording_does_not_touch_the_lecture(tmp_path):
    store, root, folder, record = world(tmp_path)
    source = tmp_path / "IMG_0001.MP4"
    assert not source.exists()                                                        # the recording is gone (it never existed in this test)
    assert quick_status(store.get(record.job_id))[1] == "Completed" and lecture_facts(store.get(record.job_id)).transcript_available


def test_a_truncated_transcript_json_with_good_text_files_is_still_a_completed_lecture(tmp_path):
    store, root, folder, record = world(tmp_path)
    text = (folder / "local_transcript.json").read_text(encoding="utf-8")
    (folder / "local_transcript.json").write_text(text[: len(text) // 2], encoding="utf-8")        # the program was closed while it was written
    facts = lecture_facts(store.get(record.job_id))
    assert facts.transcript_available and facts.transcript_kind in ("clean", "raw", "timestamped") and quick_status(store.get(record.job_id))[0] == facts.status


def test_a_lecture_whose_only_transcript_file_is_damaged_says_so_on_every_screen(tmp_path):
    store, root, folder, record = world(tmp_path, files="empty")
    (folder / "local_transcript.json").write_text('{"segments": [{"start": 0, "end": 1, "te', encoding="utf-8")       # cut off in the middle of a word
    current = store.get(record.job_id)
    status, label = quick_status(current)
    facts = lecture_facts(current)
    assert status == facts.status == MISSING and label == facts.status_label == "Transcript not found" and not facts.transcript_available


def test_a_transcript_of_only_noise_is_not_a_transcript(tmp_path):
    store, root, folder, record = world(tmp_path, files="empty")
    (folder / "local_clean_transcript.txt").write_text("\x00\x01\x02 ", encoding="utf-8")
    (folder / "local_raw_transcript.txt").write_bytes(b"\xff\xfe\x00")
    assert quick_status(store.get(record.job_id))[0] == MISSING


def test_looking_at_a_broken_lecture_never_changes_its_files(tmp_path):
    store, root, folder, record = world(tmp_path)
    (folder / "local_transcript.json").write_text("{broken", encoding="utf-8")
    before = tree_hashes(folder)
    for _ in range(2):
        quick_status(store.get(record.job_id))
        lecture_facts(store.get(record.job_id))
        reconcile_archive(store)
    assert tree_hashes(folder) == before


@pytest.mark.parametrize("files", ["full", "legacy", "timestamped"])
def test_every_shape_of_saved_files_gives_the_same_status_everywhere(tmp_path, files):
    store, root, folder, record = world(tmp_path, files=files)
    current = store.get(record.job_id)
    assert quick_status(current)[0] == lecture_facts(current).status
