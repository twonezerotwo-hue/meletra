"""Accessibility of the window: what assistive technology is told (the Qt accessibility tree, read back programmatically) and how the keyboard moves.

This checks the TREE (roles, names, descriptions, values, focus) and the KEYBOARD. It is not a test with a screen reader (NVDA, JAWS, Narrator): that has not been done.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from src.gui import accessibility, theme, ui_check
from src.gui.main_window import MainWindow
from tests.gui_support import ScriptedLauncher, make_services, settle, wait_for
from tests.study_support import write_lecture_folder


@pytest.fixture
def env(qapp, tmp_path):
    theme.apply_theme(qapp, "dark")
    services = make_services(tmp_path, opened=[])
    folders = []
    for index, name in enumerate(("Neuroanatomy", "Histology")):
        folder = write_lecture_folder(tmp_path / f"lec{index}")
        folders.append(folder)
        record = services.jobs.create_or_get(f"job-{index}", tmp_path / f"{name}.mp4", output_dir=str(folder), state="completed", display_name=name, duration_seconds=1800)
        services.jobs.update(record.job_id, state="completed", progress=100.0, summary={"output_dir": str(folder), "words": 4971})        # what the worker's completion event leaves in the record
    window = MainWindow(services, ScriptedLauncher())
    window.resize(1366, 768)
    window.show()
    window.simple_home.refresh()
    yield window, services, tmp_path, folders
    window.controller._timer.stop()
    settle(window, window.home, window.study)
    window.deleteLater()


def pump(times: int = 6) -> None:
    for _ in range(times):
        QApplication.processEvents()


def names(chain) -> list[str]:
    return [item["name"] for item in chain]


# ------------------------------------------------------------------------------------------------------------------ the tree
def test_the_start_screen_tree_has_the_names_roles_values_and_focus_a_screen_reader_needs(env):
    window, *_ = env
    report = ui_check.accessibility_report(window)
    failed = {key: value for key, value in report["checks"].items() if value is False}
    assert not failed, {"failed": failed, "input": accessibility.find(report["home"], "Input language"),
                        "output": accessibility.find(report["home"], "Output language"),
                        "evidence": report["checks"].get("language_value_evidence")}
    home = report["home"]
    examples = report["checks"]["lecture_card_examples"]
    assert any(name == "Lecture Neuroanatomy, completed" for name in examples), examples
    if sys.platform == "darwin":
        assert window.simple_home.input_combo.accessibleName() == "Input language"
        assert window.simple_home.input_combo.currentText() == "Auto Detect"
        assert "language spoken" in window.simple_home.input_combo.accessibleDescription()
        assert window.simple_home.output_combo.accessibleName() == "Output language"
        assert window.simple_home.output_combo.currentText() == "English"
    else:
        combo = accessibility.find(home, "Input language", "ComboBox")
        assert combo["value"] == "Auto Detect" and "language spoken" in combo["description"]
        assert accessibility.find(home, "Output language", "ComboBox")["value"] == "English"
    browse = accessibility.find(home, "Browse lecture files", "Button")
    assert browse["focusable"] and "audio, video, PDF or image" in browse["description"]
    area = accessibility.find(home, "Lecture files drop area")
    assert area is not None and not area["focusable"]                                       # ONE keyboard stop for choosing files: the button inside the area
    pdf = accessibility.find(home, "Add lecture PDF", "Button")
    assert pdf is not None and "Slides" in pdf["description"]
    card = accessibility.find(home, "Lecture Neuroanatomy", "Button")
    assert card["focusable"] and "Press Enter" in card["description"]
    menu = accessibility.find(home, "Actions for Neuroanatomy")
    assert menu is not None and menu["focusable"]
    chip = accessibility.find(home, "Privacy status")
    assert chip["role"] == "Button" and "privacy settings" in chip["description"]


@pytest.mark.parametrize("page", ["home", "files", "active", "batch", "result", "transcript"])
def test_every_focusable_control_of_every_page_has_a_name(env, page):
    window, services, tmp, folders = env
    window.show_finished(folders[0], "Neuroanatomy.mp4")
    window.transcript_view.show_transcript(folders[0], "Neuroanatomy")
    if page == "home":
        window.show_home()
    else:
        window._goto(page)
    window.files_page.refresh()
    pump()
    assert accessibility.unnamed_focusable(accessibility.tree(window)) == [], page


def test_every_focusable_control_of_the_study_workspace_has_a_name(env):
    from src.gui import diagnostics_probe as probe

    window, services, tmp, folders = env
    window.show_finished(folders[0], "Neuroanatomy.mp4")
    window.open_study("overview")
    wait_for(lambda: probe.study_ready(window), timeout=30, what="the study material")
    for tab in ("overview", "summary", "concepts", "flashcards", "questions", "study"):
        window.study.select_tab(tab)
        pump(10)
        assert accessibility.unnamed_focusable(accessibility.tree(window)) == [], tab
    window.study.select_tab("flashcards")
    window.study.flashcards.show_button.click()                                                # a revealed card
    pump()
    assert accessibility.unnamed_focusable(accessibility.tree(window)) == []
    window.study.select_tab("study")
    mode = window.study.study_mode
    mode.start()
    pump()
    assert accessibility.unnamed_focusable(accessibility.tree(window)) == []
    sections = accessibility.find(accessibility.tree(window), "Study sections")
    assert sections is not None and sections["focusable"]


def test_every_focusable_control_of_the_dialogs_has_a_name(env):
    from src.gui.about_dialog import AboutDialog
    from src.gui.first_run import FirstRunDialog
    from src.gui.sharing_notice import SharingNoticeDialog

    window, services, *_ = env
    settings = window.make_settings_dialog("General")
    settings.show()
    for name in ("Transcription", "Medical", "Study", "Privacy & Data", "Models", "Advanced"):
        settings.select_tab(name)
        pump(8)
        assert accessibility.unnamed_focusable(accessibility.tree(settings)) == [], name
    settings.close()
    for dialog in (AboutDialog(window), SharingNoticeDialog(services, window), FirstRunDialog(services, window)):
        dialog.show()
        pump()
        assert accessibility.unnamed_focusable(accessibility.tree(dialog)) == [], type(dialog).__name__
        assert dialog.windowTitle(), type(dialog).__name__
        dialog.close()


def test_a_field_is_named_by_its_hint_not_left_unnamed(env):
    window, *_ = env
    window._goto("transcript")
    pump()
    tree = accessibility.tree(window)
    assert accessibility.find(tree, "Search in the transcript", "EditableText") is not None


# ------------------------------------------------------------------------------------------------------------------ the keyboard
def test_tab_order_of_the_start_screen(env):
    window, *_ = env
    home = window.simple_home
    window.show_home()
    home.transcribe_button.setEnabled(True)                                                                         # (a chosen file enables it; a disabled button is skipped)
    pump()
    chain = ui_check.tab_chain(window)
    assert names(chain)[:6] == ["Browse lecture files", "Add lecture PDF", "Input language", "Output language", "TRANSCRIBE", "Lecture Neuroanatomy, completed"], names(chain)
    assert names(chain)[6:10] == ["Actions for Neuroanatomy", "Lecture Histology, completed", "Actions for Histology", "Privacy status: Local processing"]
    assert names(chain)[10:] == ["Home", "Transcription", "Study", "Files", "Settings"]                              # the sidebar last
    assert chain[0]["widget"] is home.browse_button


def test_tab_order_of_files(env):
    window, *_ = env
    window._goto("files")
    window.files_page.refresh()
    pump()
    chain = names(ui_check.tab_chain(window))
    assert chain[:2] == ["Search lectures", "Sort lectures"]
    assert chain[2:6] == ["All", "Completed", "Processing", "Failed"]                                              # the status filter
    cards = [i for i, name in enumerate(chain) if name.startswith("Lecture ")]
    assert cards and cards[0] > chain.index("Failed")                                                               # the lectures after the controls that narrow them
    assert chain[cards[0] + 1].startswith("Actions for ")                                                           # a card, then its menu
    assert chain[-5:] == ["Home", "Transcription", "Study", "Files", "Settings"]


def test_tab_order_of_study(env):
    from src.gui import diagnostics_probe as probe

    window, services, tmp, folders = env
    window.show_finished(folders[0], "Neuroanatomy.mp4")
    window.open_study("overview")
    wait_for(lambda: probe.study_ready(window), timeout=30, what="the study material")
    pump(10)
    chain = names(ui_check.tab_chain(window))
    assert chain[:6] == ["Back", "Add PDF", "Regenerate", "Export", "Start Study Mode", "Study sections"]           # the header, then the sections, then the page
    assert chain[-5:] == ["Home", "Transcription", "Study", "Files", "Settings"]
    window.study.select_tab("flashcards")
    pump(10)
    chain = names(ui_check.tab_chain(window))
    assert chain[6:12] == ["Topic", "Type", "Level", "Show", "Shuffle", "Start"] and "Show answer" in chain


def test_tab_order_of_settings_follows_the_page_then_the_buttons(env):
    window, *_ = env
    dialog = window.make_settings_dialog("General")
    dialog.show()
    for name in ("Transcription", "Medical", "Study", "Privacy & Data", "Models", "Advanced"):
        dialog.select_tab(name)
        pump(8)
        QTest.qWait(30)
        pump(4)
        chain = ui_check.tab_chain(dialog, 60, start=dialog.nav)
        order = names(chain)
        assert order[-2:] == ["Close", "Save"], (name, order)                                                        # the page's fields first, the buttons last, left to right
        assert len(order) >= 3, (name, order)
    dialog.select_tab("Transcription")
    pump(8)
    QTest.qWait(30)
    pump(4)
    order = names(ui_check.tab_chain(dialog, 60, start=dialog.nav))
    assert order.index("Input language") < order.index("Output language") < order.index("Subject") < order.index("Transcription mode")      # top to bottom, as seen
    dialog.close()


def test_enter_and_space_press_the_buttons_and_escape_closes_the_dialogs(env):
    window, services, *_ = env
    home = window.simple_home
    chosen = []
    home.drop_zone.choose_many = lambda: chosen.append(1) or []
    home.browse_button.setFocus(Qt.TabFocusReason)
    QTest.keyClick(home.browse_button, Qt.Key_Return)
    QTest.keyClick(home.browse_button, Qt.Key_Space)
    assert len(chosen) == 2                                                                                         # Enter and Space both press a focused button
    dialog = window.make_settings_dialog("General")
    dialog.show()
    pump()
    QTest.keyClick(dialog, Qt.Key_Escape)
    pump()
    assert not dialog.isVisible()


def test_the_focus_ring_is_visible_in_both_themes_and_not_clipped(qapp):
    for name in ("dark", "light"):
        sheet = theme.stylesheet(name)
        palette = theme.tokens.DARK if name == "dark" else theme.tokens.LIGHT
        assert f"QPushButton:focus {{ border: 2px solid {palette.focus}" in sheet                                  # teal ring
        assert f"QPushButton#primary:focus {{ border: 2px solid {palette.gold}" in sheet                           # gold where the button is teal itself
        assert f"QPushButton#big:focus {{ border: 2px solid {palette.gold}" in sheet
        assert f"QFrame#lectureCard:focus {{ border: 2px solid {palette.focus}" in sheet
        for selector in ("QComboBox:focus", "QLineEdit:focus", "QPushButton#navButton:focus", "QPushButton#segment:focus", "QPushButton#smallButton:focus"):
            assert selector in sheet, selector
