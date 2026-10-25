"""The v3 dashboard shell and real recent-lecture list."""

from pathlib import Path

from src.gui.home_page import HomePage
from src.gui.lecture_cards import LectureCard
from src.gui.main_window import MainWindow
from src.gui.simple_home import SimpleHome
from src.pipeline import job_key
from tests.gui_support import ScriptedLauncher, make_services, settle
from tests.study_support import write_lecture_folder


def test_dashboard_shows_real_recent_jobs_and_opens_completed_lecture(qapp, tmp_path):
    services = make_services(tmp_path)
    source = tmp_path / "cardiology.mp4"
    source.write_bytes(b"video")
    output = write_lecture_folder(tmp_path / "Cardiology lecture")
    record = services.jobs.create_or_get(job_key(source), source, output_dir=str(output), state="completed", progress=100)
    services.jobs.update(record.job_id, summary={"output_dir": str(output), "words": 4971})                          # what the worker's completion event leaves in the record
    page = SimpleHome(services, HomePage(services))
    page.refresh()

    cards = page.recent_card.findChildren(LectureCard)
    assert [card.title.text() for card in cards] == [record.source_name] and cards[0].status.text() == "Completed"
    opened = []
    page.lecture_requested.connect(opened.append)
    cards[0].clicked.emit(cards[0].record)                                                  # a click on the card opens the lecture, by its state
    assert [r.job_id for r in opened] == [record.job_id]
    settle(page, page.engine)
    page.deleteLater()


def test_simple_shell_has_requested_navigation_and_tracks_current_page(qapp, tmp_path, monkeypatch):
    opened = []
    services = make_services(tmp_path, opened=opened)
    window = MainWindow(services, ScriptedLauncher())
    assert list(window.simple_shell.buttons) == ["home", "transcription", "study", "files", "settings"]
    assert window.simple_shell.buttons["home"].isChecked()
    assert not hasattr(window.simple_shell, "privacy")                                      # the status chip lives in the Home header only
    chip = window.simple_home.privacy
    assert chip.text() == "Local processing"                                                 # sharing is off by default and the chip says so
    services.update_setting(share_transcripts=True)
    window.simple_home.refresh()
    assert chip.text() == "Local processing"                                                 # a stored wish that was never agreed to is not sharing
    monkeypatch.setenv("MLT_SHARING_ENDPOINT", "https://inbox.example.org/v1/transcripts")
    services.update_setting(share_notice_shown=True)
    window.simple_home.refresh()
    assert chip.text() == "Sharing enabled"                                                  # ... and says so too when the student switched it on and agreed
    services.update_setting(share_transcripts=False)
    window.simple_home.refresh()
    assert chip.text() == "Local processing"
    services.update_setting(cloud_enabled=True)
    window.simple_home.refresh()
    assert chip.text() == "Cloud comparison on"
    window._goto("active")
    assert window.simple_shell.buttons["transcription"].isChecked()
    window.simple_shell.buttons["files"].click()
    assert window.simple_stack.currentWidget() is window.files_page and not opened          # Files is the archive page; the folder is opened from a lecture's menu
    settle(window, window.home, window.study)
    window.deleteLater()
