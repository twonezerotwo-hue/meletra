"""The real user case, end to end in the window: a lecture a real 4.1.1 run left behind (the job record names the CHOSEN folder, the lecture is its sub-folder, a request file sits next to the
record). BUG-01 Read Transcript, BUG-02 Study, BUG-03 the list, BUG-04 the numbers, BUG-05 reading inside MELETRA, BUG-08 translating the saved text."""

from __future__ import annotations

import json

import pytest

from src.gui.main_window import MainWindow
from src.gui.study_view import PAGE_TABS
from src.gui.translate_dialog import TranslateDialog
from src.services.job_service import JobRecord
from src.translation.service import TranslationService
from tests.gui_support import ScriptedLauncher, make_services, settle, wait_for
from tests.study_support import write_lecture_folder


class Engine:
    id = "fake:1"

    def __init__(self):
        self.calls = []

    def translate(self, texts, source, target):
        self.calls.append((source, target, list(texts)))
        return [f"[{target}] {t}" for t in texts]


@pytest.fixture
def env(qapp, tmp_path):
    opened: list = []
    services = make_services(tmp_path, opened=opened)
    root = tmp_path / "Fixture Library" / "Medical Transcription"
    folder = write_lecture_folder(root / "FIXTURE_0005")
    (folder / ".job_id").write_text("0000000000000001", encoding="utf-8")
    record = JobRecord(job_id="0000000000000001", source=str(tmp_path / "Fixture Library" / "FIXTURE_0005.MP4"), source_name="FIXTURE_0005.MP4", output_dir=str(root), state="completed",
                       progress=100.0, duration_seconds=2548.0, summary={"output_dir": str(folder), "words": 4971})
    services.jobs.save(record)
    (services.paths.jobs / "0000000000000001.request.json").write_text(json.dumps({"source": record.source, "output_dir": str(root), "job_id": record.job_id}), encoding="utf-8")
    window = MainWindow(services, ScriptedLauncher())
    window.controller._timer.setInterval(20)
    yield window, services, folder, record, opened
    window.controller._timer.stop()
    settle(window, window.home, window.study)
    window.deleteLater()


def test_the_list_shows_the_lecture_once_and_completed(env):
    window, services, folder, record, _opened = env
    window.files_page.refresh()
    assert len(window.files_page.cards) == 1                                           # no "Medical Transcription - Queued" next to it
    assert window.files_page.cards[0].record.state == "completed" and "completed" in window.files_page.cards[0].accessibleName().lower()
    assert services.jobs.unfinished() == []


def test_a_click_opens_the_lecture_with_a_working_read_button(env):
    window, services, folder, record, opened = env
    window.files_page.refresh()
    window.open_lecture_record(window.files_page.cards[0].record)
    assert window.current_output == folder
    result = window.simple_result
    assert result.read_button.isEnabled() and result.study_button.isEnabled() and result.listen_button.isEnabled()
    assert "words" in result.info_label.text() and "Unknown" not in result.info_label.text()
    assert services.jobs.get(record.job_id).output_dir == str(folder)                  # the old record now names the lecture's own folder


def test_read_transcript_opens_the_text_inside_the_program(env):
    window, services, folder, record, opened = env
    window.open_lecture_record(services.jobs.get(record.job_id))
    window.simple_result.read_button.click()
    assert window.stack is not None and "Glycolysis" in window.transcript_view.text.toPlainText()
    assert opened == []                                                                  # no Explorer window, no text editor


def test_a_lecture_with_only_legacy_files_is_still_read(env):
    window, services, folder, record, opened = env
    for name in ("local_clean_transcript.txt", "local_raw_transcript.txt", "local_transcript.json"):
        (folder / name).unlink()
    (folder / "FIXTURE_0005_clean.txt").write_text("Legacy copy of the transcript with enough words to be a lecture.", encoding="utf-8")
    window.open_lecture_record(services.jobs.get(record.job_id))
    assert window.simple_result.read_button.isEnabled()
    window.simple_result.read_button.click()
    assert "Legacy copy" in window.transcript_view.text.toPlainText() or "Glycolysis" in window.transcript_view.text.toPlainText()


def test_study_uses_the_saved_transcript_without_a_pdf_or_the_video(env):
    window, services, folder, record, opened = env
    window.open_lecture_record(services.jobs.get(record.job_id))
    window.open_study("overview")
    wait_for(lambda: window.study.stack.currentIndex() == PAGE_TABS, timeout=60, what="the study material")
    assert (folder / "study" / "study_pack.json").is_file() and window.study.state.pack.flashcards
    assert not (services.jobs.get(record.job_id).source and __import__("pathlib").Path(services.jobs.get(record.job_id).source).exists())      # the video does not exist, and was not needed


def test_the_technical_details_show_the_numbers_of_the_files(env):
    from src.gui.results_page import summarize

    window, services, folder, record, _opened = env
    numbers = summarize(folder, None)
    assert numbers["Status"].startswith("Completed") and numbers["Words"] != "Unknown" and numbers["Language"] == "English"


def test_translating_the_saved_text_needs_no_video_and_no_job(env):
    window, services, folder, record, opened = env
    window.open_lecture_record(services.jobs.get(record.job_id))
    assert window.simple_result.translate_button.isEnabled()
    engine = Engine()
    dialog = TranslateDialog(services, window.current_output, None, service=TranslationService([], engine=engine))
    seen = []
    dialog.translated.connect(seen.append)
    dialog.combo.setCurrentIndex(dialog.combo.findData("tr"))
    dialog.go_button.click()
    wait_for(lambda: seen, timeout=30, what="the translation")
    assert (folder / "translations" / "tr.json").is_file() and engine.calls
    assert window.controller.is_running is False and services.jobs.unfinished() == []   # no job was started


def test_a_technical_details_file_opens_in_the_reader_not_in_an_outside_editor(env):
    window, services, folder, record, opened = env
    window.open_lecture_record(services.jobs.get(record.job_id))
    window.results.show_results(folder, "FIXTURE_0005.MP4", None)
    window.results.buttons["local_clean_transcript.txt"].click()
    assert opened == []                                                                  # no Notepad, no Explorer
    assert "Glycolysis" in window.transcript_view.text.toPlainText()


def test_finishing_a_transcription_does_not_open_explorer_by_default(env):
    window, services, folder, record, opened = env
    assert services.settings.settings.open_folder_after is False
