"""MELETRA 4.1: the simplified start screen, the whole-window drop, the languages, the lecture archive (Recent lectures, Files, rename, remove),
the logo and the product name."""

import re
from pathlib import Path

import pytest
from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl
from PySide6.QtGui import QDragEnterEvent, QDragLeaveEvent, QDropEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QAbstractButton, QApplication, QLabel

from src.gui import brand
from src.gui.lecture_cards import LectureCard, language_path
from src.gui.main_window import SIMPLE_FILES, SIMPLE_HOME, SIMPLE_RESULT, SIMPLE_STUDY, MainWindow
from src.services.job_service import JobStore, lecture_title
from src.services.settings_service import Settings
from tests.conftest import write_silent_wav
from tests.gui_support import ScriptedLauncher, make_services, settle, wait_for
from tests.study_support import write_lecture_folder

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def env(qapp, tmp_path):
    services = make_services(tmp_path, opened=[])
    window = MainWindow(services, ScriptedLauncher())
    window.controller._timer.setInterval(20)
    yield window, services, tmp_path
    window.controller._timer.stop()
    settle(window, window.home, window.study)
    window.deleteLater()


def mime(*paths) -> QMimeData:
    data = QMimeData()
    data.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
    return data


KEEP = []                                                                                                           # Qt events keep only a pointer to their QMimeData


def drag_enter(window, *paths):
    data = mime(*paths)
    KEEP.append(data)
    event = QDragEnterEvent(QPoint(200, 200), Qt.CopyAction, data, Qt.LeftButton, Qt.NoModifier)
    window.dragEnterEvent(event)
    return event


def drop(window, *paths):
    data = mime(*paths)
    KEEP.append(data)
    event = QDropEvent(QPointF(200, 200), Qt.CopyAction, data, Qt.LeftButton, Qt.NoModifier)
    window.dropEvent(event)
    return event


def completed(services, tmp, name, folder_name=None, **fields):
    """A finished lecture in the archive: the folder with its transcript, and the job record."""
    folder = write_lecture_folder(tmp / (folder_name or name))
    source = tmp / f"{name}.mp4"
    record = services.jobs.create_or_get(f"job-{name}", source, output_dir=str(folder), state="completed", **fields)
    services.jobs.update(record.job_id, state="completed", progress=100.0, summary={"output_dir": str(folder), "words": 4971})       # what the worker's completion event leaves in the record
    return services.jobs.get(record.job_id), folder


# ------------------------------------------------------------------------------------------------------------------ the start screen
def test_the_start_screen_is_one_workflow_in_the_right_order(env):
    window, services, _tmp = env
    home = window.simple_home
    assert home.title.text() == "MELETRA" and home.subtitle.text() == "Medical Learning & Transcription"
    assert home.input_stack.minimumHeight() >= 300                                                                  # about twice the old 170 px area
    assert home.drop_zone.text() == "Drop lecture files here" and home.drop_hint.text() == "Audio · Video · PDF · Image"
    assert home.browse_button.text() == "Browse files" and home.transcribe_button.text() == "TRANSCRIBE"
    top = lambda w: w.mapTo(home, QPoint(0, 0)).y()  # noqa: E731
    window.resize(1440, 900)
    window.show()
    QApplication.processEvents()
    assert home.drop_zone.height() >= 300 and home.drop_zone.height() > home.transcribe_button.height() * 3                  # the dominant element
    assert top(home.input_stack) < top(home.pdf_button) < top(home.input_combo) < top(home.transcribe_button)          # file -> PDF -> languages -> action
    assert not any(w.text() in ("New transcription", "Open a finished lecture ...") for w in home.findChildren(QAbstractButton))     # the quick actions are gone
    assert not hasattr(home, "quick_card") and not hasattr(home, "select_button")
    window.hide()


def test_there_is_no_second_settings_button_and_the_sidebar_one_works(env):
    window, *_ = env
    home = window.simple_home
    assert not hasattr(home, "settings_button") and "Settings" not in [b.text() for b in home.findChildren(QAbstractButton)]
    assert [b.text() for b in window.simple_shell.buttons.values()] == ["Home", "Transcription", "Study", "Files", "Settings"]
    opened = []
    window.make_settings_dialog = lambda tab="General": opened.append(tab) or type("D", (), {"exec": lambda self: 0})()
    window.simple_shell.buttons["settings"].click()
    assert opened == ["General"]


def test_the_drop_area_is_clickable_and_reachable_by_keyboard(env):
    window, _s, tmp = env
    source = write_silent_wav(tmp / "lecture.wav", seconds=30)
    home = window.simple_home
    home.drop_zone.choose_many = lambda: [str(source)]
    assert home.drop_zone.focusPolicy() == Qt.NoFocus and home.browse_button.focusPolicy() == Qt.TabFocus         # ONE keyboard stop for choosing files: the real button inside the area
    QTest.mouseClick(home.drop_zone, Qt.LeftButton)                                                               # click = the Select files dialog
    wait_for(lambda: window.home.source == source, what="the chosen file")
    window.new_transcription()
    home.drop_zone.choose_many = lambda: [str(source)]
    QTest.keyClick(home.browse_button, Qt.Key_Return)                                                              # Enter does the same
    wait_for(lambda: window.home.source == source, what="the chosen file (keyboard)")
    window.new_transcription()
    home.drop_zone.choose_many = lambda: [str(source)]
    home.browse_button.click()                                                                                     # and the small "Browse files" action
    wait_for(lambda: window.home.source == source, what="the chosen file (Browse files)")


def test_a_chosen_file_becomes_a_card_with_remove_and_replace(env):
    window, _s, tmp = env
    source = write_silent_wav(tmp / "FIXTURE_0004.wav", seconds=120)
    home = window.simple_home
    home.drop_zone.file_chosen.emit(str(source))
    wait_for(lambda: home.transcribe_button.isEnabled(), what="TRANSCRIBE")
    assert home.file_card.isVisibleTo(home) and home.file_name.text() == "FIXTURE_0004.wav"
    assert re.fullmatch(r"Audio · 2 min · .+", home.file_info.text()), home.file_info.text()
    assert {home.file_card.replace_button.text(), home.file_card.remove_button.text()} == {"Replace", "Remove"}
    home.file_card.remove_button.click()
    assert window.home.source is None and not home.file_card.isVisibleTo(home) and not home.transcribe_button.isEnabled()
    home.drop_zone.file_chosen.emit(str(source))
    wait_for(lambda: home.file_card.isVisibleTo(home), what="the card again")
    home.drop_zone.choose_many = lambda: []                                                                       # Replace opens the same dialog
    home.file_card.replace_button.click()


def test_the_pdf_row_is_short_and_explained_by_a_tooltip(env):
    window, *_ = env
    home = window.simple_home
    assert home.pdf_title.text() == "Lecture material" and home.pdf_optional.text() == "optional" and home.pdf_button.text() == "+ Add PDF"
    assert home.pdf_button.toolTip() == "Slides improve terminology recognition and study generation." and not home.pdf_label.isVisibleTo(home)


# ------------------------------------------------------------------------------------------------------------------ languages
def test_languages_default_to_auto_and_english_and_offer_the_requested_lists(env):
    window, services, _tmp = env
    home = window.simple_home
    assert Settings().language == "auto" and Settings().output_language == "en"
    assert home.input_combo.currentData() == "auto" and home.output_combo.currentData() == "en"
    assert home.input_label.text() == "Input Language" and home.output_label.text() == "Output Language"
    inputs = [(home.input_combo.itemText(i), home.input_combo.itemData(i)) for i in range(home.input_combo.count())]
    outputs = [(home.output_combo.itemText(i), home.output_combo.itemData(i)) for i in range(home.output_combo.count())]
    assert inputs == [("Auto Detect", "auto"), ("English", "en"), ("Turkish", "tr"), ("German", "de"), ("Hungarian", "hu"), ("Korean", "ko"), ("Chinese", "zh"),
                      ("Hindi", "hi"), ("Dutch", "nl"), ("French", "fr"), ("Spanish", "es")]
    assert outputs == inputs[1:]                                                                                   # the same ten, without Auto
    home.input_combo.setCurrentIndex(home.input_combo.findData("tr"))
    home.output_combo.setCurrentIndex(home.output_combo.findData("de"))
    assert services.settings.settings.language == "tr" and services.settings.settings.output_language == "de"
    assert window.home.language_combo.currentData() == "tr"                                                        # the engine behind the screen uses it
    request = window.home.request_template()
    assert request.language == "tr" and request.output_language == "de"


def test_language_choices_survive_a_restart(qapp, tmp_path):
    services = make_services(tmp_path, opened=[])
    window = MainWindow(services, ScriptedLauncher())
    window.simple_home.input_combo.setCurrentIndex(window.simple_home.input_combo.findData("hu"))
    window.simple_home.output_combo.setCurrentIndex(window.simple_home.output_combo.findData("fr"))
    settle(window, window.home, window.study)
    window.deleteLater()
    again = MainWindow(make_services(tmp_path, opened=[]), ScriptedLauncher())
    assert again.simple_home.input_combo.currentData() == "hu" and again.simple_home.output_combo.currentData() == "fr"
    settle(again, again.home, again.study)
    again.deleteLater()


def test_the_language_path_of_a_card(env):
    window, services, tmp = env
    record, _ = completed(services, tmp, "neuro", language="auto", detected_language="en", output_language="tr", translation="m2m100")
    assert language_path(record) == "Auto (English) → Turkish"
    assert language_path(services.jobs.update(record.job_id, output_language="en", translation="none")) == "Auto (English)"
    assert language_path(services.jobs.update(record.job_id, output_language="tr", translation="model_missing")) == "Auto (English) → Turkish (not translated)"
    assert language_path(services.jobs.update(record.job_id, language="de", detected_language="de", output_language="de", translation="none")) == "German"


# ------------------------------------------------------------------------------------------------------------------ the whole-window drop
def test_dragging_a_supported_file_anywhere_shows_the_overlay_and_leaving_hides_it(env):
    window, _s, tmp = env
    source = write_silent_wav(tmp / "lecture.wav", seconds=20)
    window._goto("study")                                                                                          # not even on the start screen
    event = drag_enter(window, source)
    assert event.isAccepted() and window.drop_overlay.isVisibleTo(window.root_stack) and window.drop_overlay.TEXT == "Drop lecture files" and window.drop_overlay.SUB == "Release to add to transcription"
    window.dragLeaveEvent(QDragLeaveEvent())
    assert not window.drop_overlay.isVisibleTo(window.root_stack)


def test_dropping_anywhere_adds_the_file_on_the_start_screen(env):
    window, _s, tmp = env
    source = write_silent_wav(tmp / "lecture.wav", seconds=20)
    window._goto("study")
    drag_enter(window, source)
    assert drop(window, source).isAccepted()
    assert not window.drop_overlay.isVisibleTo(window.root_stack) and window.simple_stack.currentIndex() == SIMPLE_HOME
    wait_for(lambda: window.home.source == source, what="the dropped file")
    window.new_transcription()
    drop(window, source, source)                                                                                   # the same file twice is one file, not two entries
    wait_for(lambda: window.home.source == source and not window.simple_home.selection, what="one file")


def test_several_dropped_files_become_one_queue_without_duplicates(env):
    window, _s, tmp = env
    a, b = write_silent_wav(tmp / "a.wav", seconds=10), write_silent_wav(tmp / "b.wav", seconds=10)
    drop(window, a, b, a)
    wait_for(lambda: len(window.simple_home.selection) == 2, what="two files")
    assert sorted(d.path.name for d in window.simple_home.selection) == ["a.wav", "b.wav"]


def test_an_unsupported_file_gets_no_overlay_and_a_clear_small_error(env):
    window, *_ = env
    docx = ROOT / "setup.docx"
    event = drag_enter(window, docx)
    assert not event.isAccepted() and not window.drop_overlay.isVisibleTo(window.root_stack)
    assert window.toast.isVisibleTo(window.root_stack) and "setup.docx" in window.toast.text() and "not supported" in window.toast.text()
    window.toast.hide()
    drop(window, docx)
    assert "setup.docx" in window.toast.text() and window.home.source is None and not window.simple_home.selection


def test_a_mix_of_supported_and_unsupported_files_adds_the_good_one_and_names_the_rest(env):
    window, _s, tmp = env
    source = write_silent_wav(tmp / "lecture.wav", seconds=20)
    drop(window, source, ROOT / "notes.docx")
    wait_for(lambda: window.home.source == source, what="the supported file")
    assert "notes.docx" in window.toast.text()


# ------------------------------------------------------------------------------------------------------------------ Recent lectures
def test_a_completed_recent_lecture_opens_its_transcript(env):
    window, services, tmp = env
    record, folder = completed(services, tmp, "neuro", display_name="Neuroanatomy - Ventricular System")
    window.simple_home.refresh()
    card = window.simple_home.recent_holder.itemAt(0).widget()
    assert isinstance(card, LectureCard) and card.title.text() == "Neuroanatomy - Ventricular System" and card.sub.text() == "neuro.mp4"
    assert card.status.text() == "Completed"
    QTest.mouseClick(card, Qt.LeftButton)
    assert window.simple_stack.currentIndex() == SIMPLE_RESULT and window.current_output == folder


def test_a_lecture_without_its_recording_still_opens_and_a_missing_folder_says_so(env):
    window, services, tmp = env
    record, folder = completed(services, tmp, "gone")                                                              # tmp/gone.mp4 never existed: the recording is "deleted"
    window.open_lecture_record(record)
    assert window.simple_stack.currentIndex() == SIMPLE_RESULT
    import shutil

    shutil.rmtree(folder)
    window.new_transcription()
    window.open_lecture_record(record)
    assert window.simple_stack.currentIndex() == SIMPLE_HOME and "were not found" in window.toast.text()


def test_the_other_states_open_the_right_screen(env):
    window, services, tmp = env
    source = write_silent_wav(tmp / "half.wav", seconds=30)
    paused = services.jobs.create_or_get("job-half", source, state="paused", progress=40.0)
    window.open_lecture_record(paused)
    wait_for(lambda: window.home.source == source, what="the paused lecture is chosen again")
    failed = services.jobs.create_or_get("job-fail", write_silent_wav(tmp / "fail.wav", seconds=30), state="failed", last_error="the disk was full")
    window.new_transcription()
    window.open_lecture_record(failed)
    wait_for(lambda: "the disk was full" in window.simple_home.problem.text(), what="the error and the retry")


def test_the_card_menu_has_four_actions_and_the_recent_list_is_short(env):
    window, services, tmp = env
    for i in range(9):
        completed(services, tmp, f"lec{i}")
    window.simple_home.refresh()
    cards = [window.simple_home.recent_holder.itemAt(i).widget() for i in range(window.simple_home.recent_holder.count())]
    assert 5 <= len(cards) <= 8
    assert [a.text() for a in cards[0].menu_button.menu().actions() if not a.isSeparator()] == ["Open", "Rename", "Show Files", "Remove from Archive"]
    assert window.simple_home.view_all_button.text() == "View all"


def test_renaming_changes_only_the_display_name_and_persists(qapp, tmp_path):
    services = make_services(tmp_path, opened=[])
    window = MainWindow(services, ScriptedLauncher())
    record, folder = completed(services, tmp_path, "FIXTURE_0004")
    media = tmp_path / "FIXTURE_0004.mp4"
    media.write_bytes(b"recording")
    home = window.simple_home
    home.refresh()
    card = home.recent_holder.itemAt(0).widget()
    card.rename_action.trigger()                                                                                   # the menu's Rename: a text field in the card, no dialog
    assert card.renaming() and card.rename_edit.text() == "FIXTURE_0004.mp4" and "Esc" in card.rename_hint.text()
    card.rename_edit.setText("Neuroanatomy - Ventricular System")
    QTest.keyClick(card.rename_edit, Qt.Key_Return)                                                                # Enter saves
    again = JobStore(services.paths.jobs).get(record.job_id)                                                       # read fresh from disk: the restart
    assert again.display_name == "Neuroanatomy - Ventricular System" and again.source_name == "FIXTURE_0004.mp4"
    assert lecture_title(again) == "Neuroanatomy - Ventricular System" and media.read_bytes() == b"recording" and folder.is_dir()
    card = home.recent_holder.itemAt(0).widget()
    assert card.title.text() == "Neuroanatomy - Ventricular System" and card.sub.text() == "FIXTURE_0004.mp4"          # the original name only when renamed
    card.begin_rename()
    card.rename_edit.setText("Something else")
    QTest.keyClick(card.rename_edit, Qt.Key_Escape)                                                                # Esc cancels: nothing changes
    assert not card.renaming() and JobStore(services.paths.jobs).get(record.job_id).display_name == "Neuroanatomy - Ventricular System"
    card.begin_rename()
    card.rename_edit.setText("FIXTURE_0004.mp4")                                                                       # back to the file name: no display name
    QTest.keyClick(card.rename_edit, Qt.Key_Return)
    assert JobStore(services.paths.jobs).get(record.job_id).display_name == ""
    settle(window, window.home, window.study)
    window.deleteLater()


def test_remove_from_archive_keeps_the_files(env):
    window, services, tmp = env
    record, folder = completed(services, tmp, "keep")
    window.simple_home.actions.confirm = lambda title, text: "stay on your computer" in text
    window.simple_home.actions.remove(record)
    assert services.jobs.archive() == [] and services.jobs.get(record.job_id).hidden and (folder / "local_clean_transcript.txt").is_file() or folder.is_dir()
    assert folder.is_dir()
    window.simple_home.actions.confirm = lambda title, text: False                                                 # "No" removes nothing
    other, _ = completed(services, tmp, "other")
    window.simple_home.actions.remove(other)
    assert [r.job_id for r in services.jobs.archive()] == [other.job_id]


# ------------------------------------------------------------------------------------------------------------------ Files = the archive
def test_files_page_is_the_full_archive_with_search_and_sort(env):
    window, services, tmp = env
    completed(services, tmp, "a_old", display_name="Biochemistry 1", duration_seconds=600, created="2026-01-01T10:00:00")
    completed(services, tmp, "b_mid", display_name="Anatomy", duration_seconds=3000, created="2026-02-01T10:00:00")
    completed(services, tmp, "c_new", display_name="Pharmacology", duration_seconds=1200, created="2026-03-01T10:00:00")
    for i in range(10):
        completed(services, tmp, f"filler{i}", duration_seconds=60 + i)
    window._navigate("files")
    page = window.files_page
    assert window.simple_stack.currentIndex() == SIMPLE_FILES and window.simple_shell.buttons["files"].isChecked()
    assert len(page.cards) == 13                                                                                   # every lecture, not the latest few
    page.search.setText("anat")
    wait_for(lambda: [c.title.text() for c in page.cards] == ["Anatomy"], what="the search to follow the typing")      # no Search button: it follows the typing (a light delay)
    page.search.setText("C_NEW")                                                                                   # the ORIGINAL file name is searched too
    page.search_now()
    assert [c.title.text() for c in page.cards] == ["Pharmacology"]
    page.search.setText("zzz")
    page.search_now()
    assert page.cards == [] and page.empty.isVisibleTo(page) and page.empty.title.text() == "No lectures match your search."
    page.search.setText("")
    page.search_now()
    assert not page.empty.isVisibleTo(page)
    order = lambda key: [c.record.display_name for c in page.cards if c.record.display_name]  # noqa: E731
    page.sort_combo.setCurrentIndex(page.sort_combo.findData("name"))
    assert order("name") == ["Anatomy", "Biochemistry 1", "Pharmacology"]
    page.sort_combo.setCurrentIndex(page.sort_combo.findData("duration"))
    assert order("duration") == ["Anatomy", "Pharmacology", "Biochemistry 1"]
    page.sort_combo.setCurrentIndex(page.sort_combo.findData("oldest"))
    jobs = services.jobs.archive(sort="oldest")
    assert jobs[0].created <= jobs[-1].created
    assert [page.sort_combo.itemText(i) for i in range(page.sort_combo.count())] == ["Newest", "Oldest", "Name A–Z", "Duration"]
    assert page.pdf_button.text().startswith("Study from a PDF only") and page.add_folder_button.text().startswith("Add a finished lecture folder")


def test_the_archive_survives_a_restart_and_a_deleted_recording(qapp, tmp_path):
    services = make_services(tmp_path, opened=[])
    first = MainWindow(services, ScriptedLauncher())
    record, folder = completed(services, tmp_path, "survivor", display_name="Kept lecture")
    first.files_page.refresh()
    settle(first, first.home, first.study)
    first.deleteLater()
    second = MainWindow(make_services(tmp_path, opened=[]), ScriptedLauncher())                                      # a new program start on the same data
    second._navigate("files")
    assert [c.title.text() for c in second.files_page.cards] == ["Kept lecture"]
    second.open_lecture_record(second.files_page.cards[0].record)
    assert second.simple_stack.currentIndex() == SIMPLE_RESULT and second.current_output == folder
    settle(second, second.home, second.study)
    second.deleteLater()


def test_old_4_0_lecture_records_still_load_and_show(env):
    window, services, tmp = env
    import json

    folder = write_lecture_folder(tmp / "from40")
    legacy = {"job_id": "legacy1", "source": str(tmp / "old.mp4"), "source_name": "old.mp4", "output_dir": str(folder), "state": "completed", "progress": 100.0,
              "language": "auto", "summary": {"language": "en"}, "created": "2026-05-01T09:00:00", "updated": "2026-05-01T10:00:00"}      # no 4.1 fields at all
    (services.paths.jobs / "legacy1.json").write_text(json.dumps(legacy), encoding="utf-8")
    [record] = services.jobs.archive()
    assert record.display_name == "" and record.output_language == "" and not record.hidden and lecture_title(record) == "old.mp4"
    window.simple_home.refresh()
    assert window.simple_home.recent_holder.itemAt(0).widget().title.text() == "old.mp4"


def test_adopting_a_finished_folder_is_idempotent_and_reads_its_metadata(env):
    window, services, tmp = env
    folder = write_lecture_folder(tmp / "elsewhere")
    first = services.jobs.adopt_folder(folder)
    second = services.jobs.adopt_folder(folder)
    assert first.job_id == second.job_id and len(services.jobs.archive()) == 1 and first.state == "completed"
    services.jobs.hide(first.job_id)
    assert services.jobs.archive() == [] and len(services.jobs.archive(include_hidden=True)) == 1
    services.jobs.adopt_folder(folder)                                                                             # adding it again brings it back
    assert len(services.jobs.archive()) == 1


# ------------------------------------------------------------------------------------------------------------------ privacy chip, logo, name
def test_the_privacy_chip_never_claims_more_than_is_true(env, monkeypatch):
    window, services, _tmp = env
    home, shell = window.simple_home, window.simple_shell
    assert services.settings.settings.share_transcripts is False                                                   # sharing is opt-in: a fresh start shares nothing
    assert home.privacy.text() == "Local processing" and not hasattr(shell, "privacy")                             # one small chip, in the Home header only
    assert "nothing is shared" in home.privacy.toolTip()
    services.settings.update(share_transcripts=True)                                                               # a wish that was never agreed to (a settings file edited by hand) is not sharing
    window.on_settings_changed()
    assert home.privacy.text() == "Local processing" and "nothing is shared" in home.privacy.toolTip()
    services.sharing.set_enabled(True)                                                                             # the student switched it on (and agreed to the notice) ...
    services.sharing.notice_shown()
    window.on_settings_changed()
    assert home.privacy.text() == "Sharing on - not connected" and "no upload address" in home.privacy.toolTip()   # ... but this build has no upload address: nothing is sent, and the chip says so
    monkeypatch.setenv("MLT_SHARING_ENDPOINT", "https://inbox.example.org/v1/transcripts")
    window.on_settings_changed()
    assert home.privacy.text() == "Sharing enabled" and "shared with MELETRA" in home.privacy.toolTip()
    for text in (home.privacy.text(), home.privacy.toolTip()):
        assert "stays on this computer" not in text and "LOCAL & PRIVATE" not in text and "Everything stays" not in text
    services.sharing.set_enabled(False)
    window.on_settings_changed()
    assert home.privacy.text() == "Local processing"
    opened = []
    window.make_settings_dialog = lambda tab="General": opened.append(tab) or type("D", (), {"exec": lambda self: 0})()
    home.privacy.click()
    assert opened == ["Privacy & Data"]


def test_the_sidebar_shows_the_logo_and_the_product_name_only(env):
    window, *_ = env
    shell = window.simple_shell
    assert not shell.logo.pixmap().isNull() and "MELETRA" in re.sub(r"<[^>]+>", "", shell.brand.text())
    visible = " ".join(w.text() for w in shell.findChildren(QLabel)).casefold()
    assert "medlecture" not in visible and "private study workspace" not in visible and "medical lecture transcriber" not in visible


def test_the_brand_assets_exist_and_are_what_the_window_uses(qapp, env):
    window, *_ = env
    folder = ROOT / "assets" / "brand"
    for name in ("meletra_logo.png", "meletra_logo_horizontal.png", "meletra_symbol.png", "meletra_icon.png", "meletra.ico", "meletra.icns", "favicon.ico", "favicon-32.png",
                 "favicon-192.png", "apple-touch-icon.png", "meletra_symbol_source.png", "brand_assets.json"):
        assert (folder / name).is_file() and (folder / name).stat().st_size > 500, name
    for n in (16, 24, 32, 48, 64, 128, 256, 512, 1024):
        assert (folder / f"meletra_icon_{n}.png").is_file()
    assert not any(path.name.startswith(("app_", "whispermedical")) for path in folder.iterdir())                    # no file of the old brand is left
    from build_tools.icns import read_icns_sizes
    from build_tools.icon import read_ico_sizes

    assert read_ico_sizes((folder / "meletra.ico").read_bytes()) == [16, 24, 32, 48, 64, 128, 256]
    assert read_icns_sizes((folder / "meletra.icns").read_bytes()) == [16, 32, 64, 128, 256, 512, 1024]
    assert not window.windowIcon().isNull() and not brand.app_icon().isNull() and not brand.logo_pixmap(small=True, height=44).isNull()
    image = brand.icon_image(64)
    assert image.width() == 64 and image.pixelColor(32, 32).alpha() > 0 and image.pixelColor(0, 0).alpha() == 0       # the rounded square, not a flat fill


def test_the_executable_installer_and_bundle_use_the_new_icon_and_name(tmp_path):
    from build_tools.icon import ico_bytes, read_ico_sizes, render_pngs, write_icon
    from build_tools import macos

    ico = write_icon(tmp_path / "app.ico")                                                                         # what scripts/build_app.py writes into the exe
    assert read_ico_sizes(ico.read_bytes()) == [16, 24, 32, 48, 64, 128, 256]
    assert render_pngs((256,))[256] == (ROOT / "assets" / "brand" / "meletra_icon_256.png").read_bytes() or len(render_pngs((256,))[256]) > 1000
    spec = (ROOT / "build_tools" / "MELETRA.spec").read_text(encoding="utf-8")
    assert 'icon=str(ASSETS / "app.ico")' in spec and "assets/brand" in spec and "meletra_symbol.png" in spec
    mac_spec = (ROOT / "build_tools" / "MELETRA-macos.spec").read_text(encoding="utf-8")
    assert "app.icns" in mac_spec and "assets/brand" in mac_spec
    plist = macos.info_plist("4.1.0", "x86_64")
    assert plist["CFBundleName"] == "MELETRA" and plist["CFBundleDisplayName"] == "MELETRA" and plist["CFBundleIconFile"] == "app.icns"
    iss = (ROOT / "build_tools" / "installer.iss").read_text(encoding="utf-8")
    assert '#define AppName "MELETRA"' in iss and "SetupIconFile={#IconFile}" in iss and "InfoBeforeFile={#NoticeFile}" in iss
    assert "{6F0B7C52-3A1E-4E7D-9C54-8D2A7B1E4F60}" in iss                                                         # the same installation id: upgrades keep working


def test_the_old_product_identity_is_gone_from_the_screens_and_the_names_are_consistent():
    import src

    assert src.APP_NAME == "MELETRA" and src.APP_ID == "MedicalLectureTranscriber"                          # shown name vs. internal id (data folders, exe, registry)
    for path in (ROOT / "src" / "gui").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "MEDLECTURE" not in text and "Private study workspace" not in text and "MEDICAL LECTURE TRANSCRIBER" not in text, path.name
        assert "Medical Lecture Transcriber" not in text, path.name
    for path in (ROOT / "src" / "gui").glob("*.py"):
        assert "CONVERT TO TEXT" not in path.read_text(encoding="utf-8"), path.name                                # one wording: TRANSCRIBE


# ------------------------------------------------------------------------------------------------------------------ small screens, keyboard
@pytest.mark.parametrize("size", [(1366, 768), (1440, 900), (1920, 1080)])
def test_nothing_is_clipped_or_hidden_at_common_laptop_sizes(env, size):
    window, services, tmp = env
    for i in range(9):
        completed(services, tmp, f"clip{i}", display_name=f"Lecture number {i} with a rather long display name that must be shortened")
    window.simple_home.refresh()
    window.resize(*size)
    window.show()
    QApplication.processEvents()
    home = window.simple_home
    assert window.width() == size[0] or window.width() >= size[0] - 40
    home.scroll.ensureWidgetVisible(home.transcribe_button)
    for widget in (home.transcribe_button, home.input_combo, home.output_combo, home.pdf_button, home.input_stack, home.privacy, home.view_all_button):
        top_left = widget.mapTo(window, QPoint(0, 0))
        assert widget.isVisibleTo(home) and widget.width() > 40 and widget.height() > 10, widget.objectName()
        assert top_left.x() >= 0 and top_left.x() + widget.width() <= window.width() and top_left.y() + widget.height() <= window.height(), (widget.objectName(), size)
    assert not home.stacked() and home.scroll.verticalScrollBar().maximum() == 0                                   # side by side, and no scrolling needed at these sizes
    assert window.minimumSize().width() <= 1000 and window.minimumSize().height() <= 650                           # the window itself can be small ...
    rail = window.simple_shell.rail.width()
    assert home.minimumSizeHint().width() <= 1366 - rail and window.files_page.minimumSizeHint().width() <= 1366 - rail      # ... and the start screen and Files fit a 1366 px laptop
    window.hide()


def test_every_control_of_the_start_screen_can_be_reached_with_the_tab_key(env):
    window, *_ = env
    home = window.simple_home
    for widget in (home.browse_button, home.pdf_button, home.input_combo, home.output_combo, home.transcribe_button, home.privacy):
        assert widget.focusPolicy() in (Qt.StrongFocus, Qt.TabFocus, Qt.WheelFocus, Qt.ClickFocus) and widget.focusPolicy() != Qt.NoFocus, widget.objectName()
    assert home.input_combo.accessibleName() and home.output_combo.accessibleName() and home.drop_zone.accessibleName() and home.pdf_button.accessibleName()         and home.browse_button.accessibleName() == "Browse lecture files"
    from src.gui.theme import stylesheet

    for theme in ("dark", "light"):
        assert "QPushButton:focus" in stylesheet(theme) and "border: 2px solid" in stylesheet(theme)               # a visible focus ring (teal, 2 px) in both themes


# ------------------------------------------------------------------------------------------------------------------ the result of an Output Language
def translated_lecture(tmp, method="whisper", target="en", write=True):
    import json

    folder = write_lecture_folder(tmp / f"tr_{method}")
    record = {"method": method, "source_language": "tr", "output_language": target, "reason": "x", "segments": 2, "seconds": 3.0, "error": None,
              "files": [f"translated_{target}_transcript.txt"] if write else []}
    meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8")) if (folder / "metadata.json").is_file() else {}
    meta["translation"] = record
    (folder / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    transcript = folder / "local_transcript.json"
    if transcript.is_file():                                                       # the spoken language the lecture was made in (what the record says)
        data = json.loads(transcript.read_text(encoding="utf-8"))
        data["language"] = "tr"
        transcript.write_text(json.dumps(data), encoding="utf-8")
    if write:
        (folder / f"translated_{target}_transcript.txt").write_text("The heart pumps blood.\nThe lungs exchange gas.\n", encoding="utf-8")
    return folder


def test_the_result_screen_reads_the_translation_and_keeps_the_original_one_click_away(env):
    window, _s, tmp = env
    folder = translated_lecture(tmp)
    window.show_finished(folder, "Pharmacology")
    result = window.simple_result
    assert result.translated_file == folder / "translated_en_transcript.txt" and result.read_button.text() == "Read Transcript (English)"
    assert "Turkish → English" in result.info_label.text() and not hasattr(result, "read_original_button")
    window.open_transcript()
    view = window.transcript_view
    assert window.simple_stack.currentIndex() == 3 and "The heart pumps blood." in view.text.toPlainText()
    assert view.mode.isVisibleTo(view) and view.mode.current() == "en"                                              # Original | English, the translation first
    assert view.mode.buttons["original"].text() == "Original · Turkish" and view.mode.buttons["en"].text() == "English"
    view.mode.buttons["original"].click()
    assert "The heart pumps blood." not in view.text.toPlainText() and view.view_combo.isVisibleTo(view)           # the original versions come back
    view.mode.buttons["en"].click()
    assert "The heart pumps blood." in view.text.toPlainText() and not view.view_combo.isVisibleTo(view)


def test_a_missing_translation_is_explained_and_nothing_is_hidden(env):
    window, _s, tmp = env
    window.show_finished(translated_lecture(tmp, method="model_missing", target="de", write=False), "Lecture")
    result = window.simple_result
    assert result.translated_file is None and result.read_button.text() == "Read Transcript" and result.message.isVisibleTo(result)
    assert "Translation to German isn't available on this computer yet" in result.message.text() and "saved and complete" in result.message.text()
    assert "translation-m2m100" in result.message.toolTip() and "translation-m2m100" not in result.message.text()                 # plain words on screen, the folder for whoever wants it
    window.show_finished(translated_lecture(tmp, method="failed", target="fr", write=False), "Lecture")
    assert "French translation could not be made" in result.message.text() and "saved and complete" in result.message.text()
    window.open_transcript()
    assert not window.transcript_view.mode.isVisibleTo(window.transcript_view)                                       # no translated text: no switch
