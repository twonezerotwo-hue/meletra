"""The simple window of version 0.2: friendly start / running / result screens, transcript view, technical details, window styles,
settings (General, AI Study, Models, Cloud + Advanced). The classic pages keep doing the work behind them."""

import json

import pytest
from PySide6.QtWidgets import QAbstractButton, QCheckBox, QLabel, QPushButton

from src.gui import friendly
from src.gui.main_window import (PAGE_ACTIVE, PAGE_HOME, PAGE_RESULTS, ROOT_CLASSIC, ROOT_SIMPLE, SIMPLE_ACTIVE, SIMPLE_HOME, SIMPLE_RESULT,
                                 SIMPLE_STUDY, SIMPLE_TRANSCRIPT, MainWindow)
from src.gui.settings_dialog import SettingsDialog
from src.gui.settings_simple import SimpleSettingsDialog
from src.gui.technical import TechnicalDetailsDialog
from src.services.transcription_service import EXIT_PAUSED
from tests.conftest import write_silent_wav
from tests.gui_support import ScriptedLauncher, make_services, progress_event, settle, wait_for
from tests.study_support import write_lecture_folder, write_pdf
from tests.test_gui_job import start_from_window, window_env  # noqa: F401 - the fixture runs the real worker with a fake model

FORBIDDEN = ("cuda", "compute type", "rtf", "chunk", "vram", "ctranslate2", "faster-whisper", "hotword", "vad", "beam", "temperature",
             "checkpoint", "diagnostic", "probabilit", "float16", "int8")


def texts(*pages) -> str:
    out = []
    for page in pages:
        for kind in (QLabel, QAbstractButton):
            out.extend(widget.text() for widget in page.findChildren(kind))
    return "\n".join(out).casefold()


def assert_no_jargon(*pages):
    text = texts(*pages)
    for word in FORBIDDEN:
        assert word not in text, f"technical word '{word}' is visible on a simple screen"


@pytest.fixture
def simple_env(qapp, tmp_path):
    opened = []
    services = make_services(tmp_path, opened=opened)
    window = MainWindow(services, ScriptedLauncher())
    window.controller._timer.setInterval(20)
    yield window, services, opened, tmp_path
    window.controller._timer.stop()
    settle(window, window.home, window.study)
    window.deleteLater()


# ------------------------------------------------------------------------------------------------------------------ friendly text
def test_friendly_durations_and_time_left():
    assert friendly.duration(4320) == "1 h 12 min" and friendly.duration(720) == "12 min" and friendly.duration(45) == "45 sec"
    assert friendly.time_left(None) == "Estimated time remaining: calculating ..."
    assert friendly.time_left(10) == "Estimated time remaining: less than a minute"
    assert friendly.time_left(12 * 60 + 5) == "Estimated time remaining: 12 min"
    assert friendly.time_left(3900) == "Estimated time remaining: 1 h 05 min"
    assert friendly.language_name("en") == "English" and friendly.language_name("tr") == "Turkish" and friendly.language_name("") == "-"
    assert friendly.subject_name("auto") == "Detecting ..." and friendly.subject_name("biochemistry") == "Biochemistry"
    assert friendly.plural(1, "card") == "1 card" and friendly.plural(3, "card") == "3 cards"


def services_archive(window):
    return window.services.jobs.archive()


# ------------------------------------------------------------------------------------------------------------------ start screen
def test_the_simple_window_is_the_default_and_the_start_screen_is_plain(simple_env):
    window, services, _opened, _tmp = simple_env
    assert window.root_stack.currentIndex() == ROOT_SIMPLE and window.simple_stack.currentIndex() == SIMPLE_HOME
    home = window.simple_home
    assert home.title.text() == "MELETRA" and home.subtitle.text() == "Medical Learning & Transcription"
    assert home.transcribe_button.text() == "TRANSCRIBE" and not home.transcribe_button.isEnabled()
    assert home.pdf_button.text() == "+ Add PDF" and home.pdf_title.text() == "Lecture material" and home.pdf_optional.text() == "optional"
    assert home.drop_zone.text() == "Drop lecture files here"
    assert not home.file_card.isVisibleTo(home) and not home.problem.isVisibleTo(home) and not home.model_box.isVisibleTo(home)
    assert_no_jargon(home)


def test_choosing_a_file_enables_transcribe_and_shows_name_and_length(simple_env):
    window, _s, _o, tmp = simple_env
    source = write_silent_wav(tmp / "Biochemistry lecture.wav", seconds=120)
    window.simple_home.drop_zone.file_chosen.emit(str(source))
    wait_for(lambda: window.simple_home.transcribe_button.isEnabled(), what="TRANSCRIBE")
    home = window.simple_home
    assert home.file_card.isVisibleTo(home) and home.file_name.text() == "Biochemistry lecture.wav"
    assert "2 min" in home.file_info.text()
    assert window.home.source == source                           # the classic page did the real work


def test_optional_pdf_is_shown_and_can_be_removed(simple_env):
    window, _s, _o, tmp = simple_env
    pdf = write_pdf(tmp / "slides.pdf", [["Glycolysis"]])
    window.home.set_pdf(pdf)
    home = window.simple_home
    assert "slides.pdf" in home.pdf_label.text() and home.pdf_remove.isVisibleTo(home) and "Change" in home.pdf_button.text()
    home.pdf_remove.click()
    assert window.home.pdf_path is None and not home.pdf_remove.isVisibleTo(home)


def test_unsupported_file_gets_a_plain_message(simple_env):
    window, *_ = simple_env
    window.simple_home.drop_zone.rejected.emit("'notes.docx' is not a supported audio or video file.")
    assert window.simple_home.problem.isVisibleTo(window.simple_home) and "notes.docx" in window.simple_home.problem.text()


def test_missing_model_is_explained_as_a_one_time_setup(qapp, tmp_path):
    services = make_services(tmp_path, installed=False)
    window = MainWindow(services, ScriptedLauncher())
    home = window.simple_home
    assert home.model_box.isVisibleTo(home) and "One-time setup" in home.model_text.text()
    assert home.download_button.isVisibleTo(home) and "GB" in home.model_text.text()
    assert_no_jargon(home)
    settle(window, window.home)
    window.deleteLater()


def test_unfinished_lecture_offers_to_continue(simple_env):
    window, services, _o, tmp = simple_env
    source = write_silent_wav(tmp / "lecture.wav", seconds=120)
    from src.pipeline import job_key

    services.jobs.create_or_get(job_key(source), source, state="paused", progress=45.0, chunk=2, chunks=4)
    window.simple_home.drop_zone.file_chosen.emit(str(source))
    wait_for(lambda: window.simple_home.resume_box.isVisibleTo(window.simple_home), what="resume box")
    assert "45%" in window.simple_home.resume_text.text() and "Continue" in window.simple_home.resume_button.text()


# ------------------------------------------------------------------------------------------------------------------ running + result
@pytest.mark.usefixtures("fake_ffmpeg")
def test_a_whole_run_through_the_simple_screens(window_env):
    env = window_env
    w = env.window
    env.services.update_setting(chunk_minutes=1.0, output_dir=str(env.tmp / "out"))
    w.simple_home.drop_zone.file_chosen.emit(str(env.source))
    wait_for(lambda: w.simple_home.transcribe_button.isEnabled(), what="TRANSCRIBE")
    w.simple_home.transcribe_button.click()
    assert w.simple_stack.currentIndex() == SIMPLE_ACTIVE and w.stack.currentIndex() == PAGE_ACTIVE
    assert w.simple_active.title.text().startswith("Preparing") and "Language:" in w.simple_active.facts_label.text()
    assert_no_jargon(w.simple_active)
    wait_for(lambda: w.simple_stack.currentIndex() == SIMPLE_RESULT, timeout=60, what="the result screen")
    assert w.stack.currentIndex() == PAGE_RESULTS and w.current_output == env.tmp / "out" / "lecture"
    r = w.simple_result
    assert r.title.text() == "Lecture Ready" and r.file_label.text() == "lecture.wav"
    assert [b.text() for b in (r.read_button, r.study_button, r.files_button)] == ["Read Transcript", "Create Study Material", "Show in Explorer"]
    assert set(r.tiles) == {"summary", "flashcards", "questions"}
    assert r.read_button.isEnabled() and "transcript" not in r.tiles and "Made when you press Create Study Material" in r.tiles["summary"].text()
    assert_no_jargon(r)
    r.files_button.click()
    assert env.opened[-1] == env.tmp / "out" / "lecture"
    r.read_button.click()
    assert w.simple_stack.currentIndex() == SIMPLE_TRANSCRIPT and w.transcript_view.text.toPlainText().strip()
    w.transcript_view.back_button.click()
    assert w.simple_stack.currentIndex() == SIMPLE_RESULT
    r.new_button.click()
    assert w.simple_stack.currentIndex() == SIMPLE_HOME and w.stack.currentIndex() == PAGE_HOME
    assert not w.simple_home.file_card.isVisibleTo(w.simple_home)


def test_progress_shows_percent_time_left_and_language_subject(simple_env):
    window, _s, _o, tmp = simple_env
    a = window.simple_active
    a.begin("lecture.mp4", "auto", "auto")
    assert a.eta_label.text() == "" and a.progress_bar.maximum() == 0          # not measurable yet: a busy bar, no time estimate, no 0%
    a.set_progress(progress_event(40.0, rtf=0.25, eta=720.0))
    assert a.percent_label.text() == "40%" and a.progress_bar.value() == 400 and a.eta_label.text() == "Estimated time remaining: 12 min"
    a.set_progress(progress_event(41.0, rtf=0.0))
    assert a.eta_label.text() == "Estimated time remaining: calculating ..."          # no speed estimate yet -> no invented number
    a.set_facts({"language": "en", "subject": "biochemistry", "device": "cuda", "model": "x"})
    assert a.facts_label.text() == "Language: English     Subject: Biochemistry"
    assert_no_jargon(a)


def test_pause_resume_and_failure_wording(simple_env):
    window, *_ = simple_env
    a = window.simple_active
    a.begin("lecture.mp4", "en", "auto")
    got = []
    a.pause_requested.connect(lambda: got.append("pause"))
    a.pause_button.click()
    assert got == ["pause"] and a.pause_button.text() == "Pausing ..." and not a.pause_button.isEnabled()
    window.on_paused({})
    assert a.title.text() == "Paused" and a.resume_button.isVisibleTo(a) and a.back_button.isVisibleTo(a)
    assert "progress is saved" in a.eta_label.text()
    window.on_failed("The speech model could not be loaded.")
    assert a.title.text() == "The transcription stopped" and a.resume_button.text() == "Try again" and "speech files are missing" in a.note.text()


def test_technical_details_of_a_finished_lecture_show_the_classic_results(simple_env, monkeypatch):
    window, _s, _o, tmp = simple_env
    folder = write_lecture_folder(tmp / "done")
    window.show_finished(folder, "Glycolysis.mp4")
    assert window.simple_stack.currentIndex() == SIMPLE_RESULT and window.simple_result.file_label.text() == "Glycolysis.mp4"
    shown = []
    monkeypatch.setattr(TechnicalDetailsDialog, "exec", lambda self: shown.append((self.windowTitle(), type(self.page).__name__)) or 0)
    window.simple_result.details_button.click()
    assert shown == [("Technical details", "ResultsPage")]


def test_technical_details_of_a_running_job_hands_the_live_page_back(simple_env, monkeypatch):
    window, *_ = simple_env
    window._goto("active")

    def exec_and_close(self):
        assert window.stack.indexOf(window.active) < 0                    # the live page is inside the dialog now
        self.done(0)
        return 0

    monkeypatch.setattr(TechnicalDetailsDialog, "exec", exec_and_close)
    window.simple_active.details_button.click()
    assert window.stack.indexOf(window.active) == PAGE_ACTIVE             # ... and back in its place afterwards


# ------------------------------------------------------------------------------------------------------------------ window styles
def test_classic_window_style_shows_the_old_screens(simple_env):
    window, services, *_ = simple_env
    assert type(window.make_settings_dialog()) is SimpleSettingsDialog
    services.update_setting(ui_mode="classic")
    window.apply_ui_mode()
    assert window.root_stack.currentIndex() == ROOT_CLASSIC and not window.open_lecture_action.isEnabled()
    dialog = window.make_settings_dialog()
    assert type(dialog) is SettingsDialog
    dialog.deleteLater()
    services.update_setting(ui_mode="simple")
    window.apply_ui_mode()
    assert window.root_stack.currentIndex() == ROOT_SIMPLE


def test_both_views_agree_about_where_the_job_is(simple_env):
    window, *_ = simple_env
    for name, classic, simple in (("active", PAGE_ACTIVE, SIMPLE_ACTIVE), ("result", PAGE_RESULTS, SIMPLE_RESULT),
                                  ("home", PAGE_HOME, SIMPLE_HOME)):
        window._goto(name)
        assert (window.stack.currentIndex(), window.simple_stack.currentIndex()) == (classic, simple)
    window._goto("transcript")
    assert window.simple_stack.currentIndex() == SIMPLE_TRANSCRIPT and window.stack.currentIndex() == PAGE_HOME


def test_menus_keep_their_titles_and_diagnostics_moved_to_help(simple_env):
    window, *_ = simple_env
    bar = window.menuBar()                                       # keep the wrapper: a temporary one lost its menus under PySide on macOS
    actions = bar.actions()
    assert [a.text() for a in actions] == ["&File", "&Settings", "&Help"]
    settings_menu = actions[1].menu()
    help_menu = actions[2].menu()
    assert window.diagnostics_action not in settings_menu.actions() and window.diagnostics_action in help_menu.actions()
    assert "Advanced" in window.diagnostics_action.text()


# ------------------------------------------------------------------------------------------------------------------ transcript view
def test_transcript_view_switches_files_searches_and_copies(simple_env):
    window, _s, opened, tmp = simple_env
    folder = write_lecture_folder(tmp / "done")
    window.show_finished(folder, "Lecture")
    window.simple_result.read_button.click()
    view = window.transcript_view
    assert view.view_combo.currentData() == "local_clean_transcript.txt" and "Glycolysis is the metabolic pathway" in view.text.toPlainText()
    view.search_edit.setText("hexokinase")
    assert view.hits_label.text().endswith("found") and int(view.hits_label.text().split()[0]) >= 3
    view.find_next()
    assert view.text.textCursor().selectedText().casefold() == "hexokinase"
    view.view_combo.setCurrentIndex(1)
    assert view.text.toPlainText().startswith("[00:00]")
    view.open_button.click()
    assert opened[-1] == folder / "local_transcript_with_timestamps.txt"
    view.copy_button.click()
    from PySide6.QtGui import QGuiApplication

    assert QGuiApplication.clipboard().text() == view.text.toPlainText()


def test_transcript_view_copes_with_missing_files(simple_env):
    window, _s, _o, tmp = simple_env
    folder = tmp / "odd"
    folder.mkdir()
    (folder / "local_raw_transcript.txt").write_text("Only the raw text exists.", encoding="utf-8")
    window.show_finished(folder, "Odd")
    window.transcript_view.show_transcript(folder, "Odd")
    assert window.transcript_view.text.toPlainText() == "Only the raw text exists."
    empty = tmp / "empty"
    empty.mkdir()
    window.transcript_view.show_transcript(empty, "Empty")
    assert "not found" in window.transcript_view.status.text()


# ------------------------------------------------------------------------------------------------------------------ opening older lectures
def test_open_a_finished_lecture_from_the_start_screen(simple_env):
    window, _s, _o, tmp = simple_env
    folder = write_lecture_folder(tmp / "older")
    window.choose_folder = lambda start: str(folder)
    window.files_page.add_folder_button.click()                             # Files -> "Add a finished lecture folder ..."
    assert window.simple_stack.currentIndex() == SIMPLE_RESULT and window.current_output == folder
    assert [r.output_dir for r in services_archive(window)] == [str(folder)]          # and it is in the archive from now on
    window.new_transcription()
    window.choose_folder = lambda start: str(tmp)                          # not a lecture folder
    window.files_page.add_folder_button.click()
    assert window.simple_stack.currentIndex() == SIMPLE_HOME and "does not look like a finished lecture" in window.simple_home.problem.text()
    window.choose_folder = lambda start: ""
    window.open_finished_lecture()                                          # cancelled dialog: nothing happens
    assert window.simple_stack.currentIndex() == SIMPLE_HOME


# ------------------------------------------------------------------------------------------------------------------ settings
def test_simple_settings_have_four_visible_tabs_and_a_hidden_advanced_section(simple_env):
    window, services, *_ = simple_env
    dialog = window.make_settings_dialog()
    groups = [dialog.tabs.tabText(i) for i in range(dialog.tabs.count())]
    assert groups == ["Transcription", "Medical", "Study", "Privacy & Data", "Models", "Advanced"]
    assert [dialog.nav.item(i).text() for i in range(dialog.nav.count())] == groups                 # the same six groups, as a list on the left
    assert [dialog.advanced_tabs.tabText(i) for i in range(dialog.advanced_tabs.count())] == ["Performance", "Medical options", "Diagnostics"]
    dialog.save()
    dialog.deleteLater()


def test_old_tab_names_open_the_advanced_section(simple_env):
    window, services, *_ = simple_env
    dialog = window.make_settings_dialog("Performance")
    assert dialog.tabs.tabText(dialog.tabs.currentIndex()) == "Advanced"
    assert dialog.advanced_tabs.tabText(dialog.advanced_tabs.currentIndex()) == "Performance"
    dialog.select_tab("Cloud")                                                                       # moved into Privacy & Data
    assert dialog.tabs.tabText(dialog.tabs.currentIndex()) == "Privacy & Data"
    dialog.select_tab("General")
    assert dialog.tabs.tabText(dialog.tabs.currentIndex()) == "Transcription"
    dialog.deleteLater()


def test_the_study_page_has_no_method_chooser_and_cloud_needs_its_own_consent(simple_env):
    """FEATURE-06/07: MELETRA chooses the study engine. The page offers no Built-in / Ollama / OpenAI / Anthropic choice; an optional other service is under Advanced, and a cloud service
    is used only after a SEPARATE consent."""
    window, services, *_ = simple_env
    dialog = window.make_settings_dialog("AI Study")
    page_texts = " ".join(w.text() for w in dialog.findChildren(QLabel)) + " " + " ".join(b.text() for b in dialog.findChildren(QPushButton))
    assert "Study method:" not in page_texts and not hasattr(dialog, "study_provider_combo")
    assert not dialog.study_advanced.isVisibleTo(dialog)                                                  # engine details, a preferred model and the other service: only when asked for
    dialog.study_advanced_check.setChecked(True)
    assert dialog.study_advanced.isVisibleTo(dialog) and dialog.study_service_combo.currentData() == "builtin"
    assert not dialog.study_key_edit.isVisibleTo(dialog) and not dialog.study_cloud_check.isVisibleTo(dialog)
    dialog.study_service_combo.setCurrentIndex(dialog.study_service_combo.findData("anthropic"))
    assert dialog.study_key_edit.isVisibleTo(dialog) and "never in a settings file" in dialog.study_key_status.text() and dialog.study_cloud_check.isVisibleTo(dialog)
    assert not dialog.study_cloud_check.isChecked()                                                       # nothing is consented to by choosing a service
    dialog.study_key_edit.setText("sk-test-key-123456")
    dialog.save_study_key()
    assert services.credentials.get_study_key("anthropic") == "sk-test-key-123456"
    assert "3456" in dialog.study_key_status.text() and "sk-test-key" not in dialog.study_key_status.text()
    dialog.study_model_edit.setText("claude-haiku-5-5")
    dialog.study_cards_spin.setValue(30)
    dialog.save()
    s = services.settings.settings
    assert (s.study_provider, s.study_model, s.study_cards) == ("anthropic", "claude-haiku-5-5", 30) and s.study_cloud_consent is False      # a key and a service are not a consent
    assert "sk-test-key" not in json.dumps(s.to_dict())                     # never in the settings file
    dialog2 = window.make_settings_dialog("AI Study")
    dialog2.study_advanced_check.setChecked(True)
    dialog2.study_cloud_check.setChecked(True)
    dialog2.save()
    assert services.settings.settings.study_cloud_consent is True
    dialog3 = window.make_settings_dialog("AI Study")
    dialog3.remove_study_key()
    assert services.credentials.get_study_key("anthropic") is None
    dialog3.study_service_combo.setCurrentIndex(dialog3.study_service_combo.findData("builtin"))
    dialog3.save()
    assert services.settings.settings.study_cloud_consent is False                                       # taking the service away takes the consent away
    for d in (dialog, dialog2, dialog3):
        settle(d)
        d.deleteLater()


def test_the_study_engine_check_says_in_plain_words_what_will_be_used(simple_env):
    window, services, *_ = simple_env
    dialog = window.make_settings_dialog("AI Study")
    wait_for(lambda: "built-in offline method" in dialog.study_engine_label.text(), what="the engine line")          # no AI server in the tests: the offline method
    dialog.test_study_provider()
    wait_for(lambda: dialog.study_test_result.text().startswith("Ready"), what="the engine check")
    assert "built-in offline method" in dialog.study_test_result.text() and "Ollama" not in dialog.study_test_result.text().split("(")[0]
    services.study.transport = lambda url, headers, body, timeout: {"models": [{"name": "qwen3:8b"}, {"name": "nomic-embed-text"}]}
    dialog.test_study_provider()
    wait_for(lambda: "local AI (qwen3:8b)" in dialog.study_test_result.text(), what="the engine check with a local model")
    settle(dialog)
    dialog.deleteLater()


def test_cloud_tab_names_the_right_secure_store(simple_env):
    window, services, *_ = simple_env
    dialog = window.make_settings_dialog("Cloud")
    assert services.credentials.store_label in dialog.key_status.text()
    dialog.deleteLater()


def test_settings_is_in_the_sidebar_only(simple_env):
    window, *_ = simple_env
    home = window.simple_home
    assert not hasattr(home, "settings_button") and not hasattr(home, "settings_requested")          # no second Settings button at the top right
    assert [b.text() for b in home.findChildren(type(window.simple_shell.buttons["settings"])) if b.text() == "Settings"] == []
    assert [b.text() for b in window.simple_shell.buttons.values()] == ["Home", "Transcription", "Study", "Files", "Settings"]
    opened = []
    window.make_settings_dialog = lambda tab="General": opened.append(tab) or type("D", (), {"exec": lambda self: 0})()
    window.simple_shell.buttons["settings"].click()
    assert opened == ["General"]


def test_job_warnings_are_put_in_plain_words_on_the_simple_screen(simple_env):
    window, *_ = simple_env
    a = window.simple_active
    a.begin("lecture.mp4", "en", "auto")
    for raw in ("CUDA acceleration unavailable. Reason: no driver. CPU fallback enabled.",
                "Running on the CPU: Large-v3 can take several times the length of the recording. An NVIDIA GPU is much faster."):
        a.note.hide()
        a.show_warning(raw)
        assert a.note.isVisibleTo(a) and "processor" in a.note.text() and "CUDA" not in a.note.text() and "GPU" not in a.note.text()
    a.note.hide()
    a.show_warning("Hotwords are off: the language has no medical vocabulary.")
    assert not a.note.isVisibleTo(a)                                          # nothing a student can act on
    a.show_warning("An earlier checkpoint for this file belongs to a different file version or different settings and is ignored.")
    assert "starts from the beginning" in a.note.text() and "checkpoint" not in a.note.text().casefold()
    a.show_warning("Another application is using significant GPU memory. Only 1.0 GB of 8.0 GB VRAM are free.")
    assert "graphics memory" in a.note.text() and "VRAM" not in a.note.text()
    a.show_warning("Disk is almost full.")
    assert a.note.text() == "Disk is almost full."                           # an ordinary message stays as it is
    window.controller.warning.emit("Running on the CPU: slow")               # wired through the window
    assert "processor" in a.note.text()
    assert_no_jargon(a)


def test_a_stray_key_press_cannot_pause_the_running_job(simple_env):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    window, *_ = simple_env
    a = window.simple_active
    window._goto("active")
    a.begin("lecture.mp4", "en", "auto")
    got = []
    a.pause_requested.connect(lambda: got.append("pause"))
    assert a.pause_button.focusPolicy() == Qt.ClickFocus and a.focusPolicy() == Qt.StrongFocus
    window.show()
    window.activateWindow()
    a.setFocus()
    for key in (Qt.Key_Space, Qt.Key_Return, Qt.Key_Enter, Qt.Key_Tab, Qt.Key_Space):
        QTest.keyClick(window, key)
    assert got == []                                            # keys do nothing; only a deliberate click pauses
    a.pause_button.click()
    assert got == ["pause"]
    window.hide()
