"""The opt-in diagnostic probe: what the window shows is reported as data (the build machine has no one to look at the screen)."""

import pytest

from src.gui import diagnostics_probe as probe
from src.gui.main_window import MainWindow
from src.gui.study_view import PAGE_TABS, TAB_NAMES
from tests.gui_support import ScriptedLauncher, make_services, settle, wait_for
from tests.study_support import write_lecture_folder


@pytest.fixture
def window(qapp, tmp_path):
    services = make_services(tmp_path)
    w = MainWindow(services, ScriptedLauncher())
    w.show()
    w.study.show_source_dialog = lambda title, text: None
    yield w, tmp_path
    settle(w, w.home, w.study)
    w.deleteLater()


def test_the_probe_is_off_unless_asked(monkeypatch):
    monkeypatch.delenv(probe.ENABLED_BY, raising=False)
    assert probe.enabled() is False
    assert probe.payload(None, None) == {}
    monkeypatch.setenv(probe.ENABLED_BY, "1")
    assert probe.enabled() is True


def test_navigation_and_first_launch_screen_are_reported(window):
    w, _ = window
    nav = probe.home_facts(w)
    assert nav["navigation"] == ["Home", "Transcription", "Study", "Files", "Settings"] and nav["privacy"] in ("Sharing enabled", "Local processing")
    first = probe.first_launch_facts(w)
    assert "model_text" in first and isinstance(first["download_button_hidden"], bool)       # without a model the download button is offered


def test_the_study_workspace_is_reported_with_tabs_stats_buttons_and_a_revealed_card(window):
    w, tmp = window
    folder = write_lecture_folder(tmp / "lecture")
    w.show_finished(folder, "Glycolysis.mp4")
    w.open_study("overview")
    wait_for(lambda: probe.study_ready(w), timeout=60, what="study tabs")
    facts = probe.study_facts(w)
    assert facts["tabs"] == TAB_NAMES and set(facts["stats"]) >= {"concepts", "flashcards", "questions", "mastery"}
    engine = facts["engine"]                                                                      # which engine made the pack is reported from the running window
    assert engine["engine"] == probe.OLD_ENGINE or engine["engine"] == probe.NEW_ENGINE
    assert set(engine) >= {"summary", "key_concepts", "flashcards", "questions", "study_mode"}
    assert {"Back", "Regenerate", "Export"} <= set(facts["actions"])
    cards = facts["tab_content"]["Flashcards"]["buttons"]
    assert {"Previous", "Next", "Know", "Unsure", "Don't know"} <= set(cards), cards            # the answer was revealed by the probe
    assert any(b.startswith("Transcript ") for b in cards), cards                                # ... and so were its source links
    assert w.study.stack.currentIndex() == PAGE_TABS


def test_the_trace_records_each_text_once_with_its_time(window):
    w, _ = window
    p = probe.UiProbe(w, interval_ms=20)
    w._goto("active")
    w.simple_active.title.setText("Preparing video...")
    p.sample()
    p.sample()
    w.simple_active.title.setText("Extracting audio...")
    rows = p.stop()
    assert [r[2] for r in rows if r[1] == "active.title"] == ["Preparing video...", "Extracting audio..."]


def test_a_stage_shorter_than_the_sampling_interval_is_still_recorded(window):
    """The packaged-app CI missed "Extracting audio..." on a fast runner: the clip's audio was out in less than one 100 ms sample."""
    w, _ = window
    p = probe.UiProbe(w, interval_ms=60000)                  # the timer never fires: only the hooks can see the texts
    w._goto("active")
    for text in ("Preparing video...", "Extracting audio...", "Analyzing the recording..."):
        w.simple_active.title.setText(text)
    w.simple_active.percent_label.setText("0%")
    w.simple_active.percent_label.setText("12%")
    rows = p.stop()
    assert [r[2] for r in rows if r[1] == "active.title"] == ["Preparing video...", "Extracting audio...", "Analyzing the recording..."]
    assert [r[2] for r in rows if r[1] == "active.percent"] == ["0%", "12%"]


def test_autorun_study_opens_a_lecture_folder_and_reports_the_workspace(qapp, tmp_path, monkeypatch):
    """The hook the packaged-app CI uses: --autorun-study FOLDER writes the workspace report."""
    import json

    from src.gui import app as gui_app
    from src.gui.main_window import MainWindow

    monkeypatch.setenv(probe.ENABLED_BY, "1")
    services = make_services(tmp_path)
    window = MainWindow(services, ScriptedLauncher())
    window.show()
    window.study.show_source_dialog = lambda title, text: None
    folder = write_lecture_folder(tmp_path / "lecture")
    result = tmp_path / "study_result.json"
    runner = gui_app.AutoRunStudy(window, folder, result, 120)
    wait_for(lambda: runner.finished, timeout=120, what="the study probe")
    data = json.loads(result.read_text(encoding="utf-8"))
    assert data["exit_code"] == 0 and data["ui_study"]["tabs"] == TAB_NAMES
    assert data["ui_home"]["navigation"] == ["Home", "Transcription", "Study", "Files", "Settings"] and "model_text" in data["ui_first_launch"]
    runner.timer.stop()
    settle(window, window.home, window.study)
    window.deleteLater()
