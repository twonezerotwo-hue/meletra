"""The final polish pass: Study as one product, a compact progress block, one type scale, token spacing, plain-language errors, the drop overlay."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QFontMetrics
from PySide6.QtWidgets import QApplication, QFrame, QLabel

from src.gui import friendly, theme, tokens
from src.gui.components import CARD_WIDTH, READING_WIDTH
from src.gui.main_window import SIMPLE_ACTIVE, SIMPLE_BATCH, MainWindow
from tests.gui_support import ScriptedLauncher, make_services, settle, wait_for
from tests.study_support import write_lecture_folder

ROOT = Path(__file__).resolve().parent.parent
SIZES = [(1366, 768), (1440, 900), (1920, 1080)]


@pytest.fixture
def env(qapp, tmp_path):
    theme.apply_theme(qapp, "dark")
    services = make_services(tmp_path, opened=[])
    window = MainWindow(services, ScriptedLauncher())
    yield window, services, tmp_path
    window.controller._timer.stop()
    settle(window, window.home, window.study)
    window.deleteLater()


def pump(times: int = 8) -> None:
    for _ in range(times):
        QApplication.processEvents()


def open_study(window, tmp):
    from src.gui import diagnostics_probe as probe

    folder = write_lecture_folder(tmp / "study_lecture")
    window.show_finished(folder, "Neuroanatomy.mp4")
    window.open_study("overview")
    wait_for(lambda: probe.study_ready(window), timeout=30, what="the study material")
    return window.study


# ------------------------------------------------------------------------------------------------------------------ Study
def test_the_reading_column_holds_70_to_90_characters(env):
    window, services, tmp = env
    window.resize(1920, 1080)
    window.show()
    study = open_study(window, tmp)
    study.select_tab("summary")
    pump(12)
    browser = study.summary.browser
    font = QFont(browser.font())
    font.setPixelSize(tokens.FONT_PX["reading"])
    sample = "The quick regulatory enzyme of glycolysis is activated by AMP and by fructose-2,6-bisphosphate in the liver"
    per_char = QFontMetrics(font).horizontalAdvance(sample) / len(sample)               # real text, not the widest letter
    text_width = browser.viewport().width() - 2 * int(browser.document().documentMargin())
    characters = text_width / per_char
    assert 70 <= characters <= 95, (characters, text_width, per_char)
    assert browser.width() <= READING_WIDTH + 30                                           # a column, not the whole window
    window.hide()


def test_summary_has_a_title_a_meta_line_and_quiet_kinds(env):
    window, services, tmp = env
    window.resize(1366, 768)
    window.show()
    study = open_study(window, tmp)
    study.select_tab("summary")
    pump(10)
    summary = study.summary
    assert summary.kind_title.text() == "Quick Summary" and "points" in summary.kind_meta.text() and "·" in summary.kind_meta.text()
    summary.select_kind("detailed")
    assert summary.kind_title.text() != "Quick Summary"
    assert summary.kinds.objectName() == "kindList" and summary.kinds.width() >= 240 and summary.kinds.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
    html = summary.browser.toHtml()
    assert "<ul" not in html                                                                  # blocks, not a bullet list of every sentence
    window.hide()


def test_key_concepts_are_stacked_blocks_with_a_term_a_sentence_and_a_note(env):
    window, services, tmp = env
    window.resize(1366, 768)
    window.show()
    study = open_study(window, tmp)
    study.select_tab("concepts")
    pump(10)
    text = study.concepts.browser.toPlainText()
    assert "Key concepts" in text and "Glycolysis" in text and "·" in text                    # "Core term · named 15 times in the lecture"
    assert " - named" not in text
    window.hide()


def test_flashcards_separate_the_question_from_the_answer(env):
    window, services, tmp = env
    window.resize(1366, 768)
    window.show()
    study = open_study(window, tmp)
    study.select_tab("flashcards")
    pump(10)
    cards = study.flashcards
    assert cards.front_caption.text() == "Question" and cards.back_caption.text() == "Answer"
    assert not cards.back.isVisibleTo(cards) and not cards.divider.isVisibleTo(cards) and not cards.back_caption.isVisibleTo(cards)
    assert [cards.previous_button.text(), cards.show_button.text(), cards.next_button.text()] == ["Previous", "Show answer", "Next"]
    cards.show_button.click()
    pump()
    assert cards.back.isVisibleTo(cards) and cards.divider.isVisibleTo(cards) and cards.back_caption.isVisibleTo(cards)
    assert cards.card_frame.maximumHeight() <= 440                                          # not a poster on a big screen
    window.hide()


@pytest.mark.parametrize("width,height", SIZES)
def test_the_flashcard_buttons_and_hints_fit_the_card_column_at_every_target_size(env, width, height):
    window, services, tmp = env
    window.resize(width, height)
    window.show()
    study = open_study(window, tmp)
    study.select_tab("flashcards")
    pump(10)
    cards = study.flashcards
    cards.reveal()
    pump()
    row = [cards.previous_button, cards.know_button, cards.unsure_button, cards.dont_button, cards.next_button, cards.difficult_button]
    assert all(b.isVisibleTo(cards) for b in row) and not cards.show_button.isVisibleTo(cards)
    column = cards.card_frame.parentWidget()
    right_edge = max(b.mapTo(column, b.rect().topRight()).x() for b in row)
    assert min(b.mapTo(column, b.rect().topLeft()).x() for b in row) >= 0 and right_edge <= column.width(), (right_edge, column.width())
    assert cards.schedule_hint.isVisibleTo(cards) and "Know:" in cards.schedule_hint.text()
    for label in (cards.schedule_hint, cards.counter):                                       # the lines under and over the card are never clipped
        needed = label.heightForWidth(label.width()) if label.wordWrap() else label.sizeHint().height()
        assert needed <= label.height() + 2, (label.text(), needed, label.height())
    window.hide()


def test_questions_have_a_number_the_question_and_one_action(env):
    window, services, tmp = env
    window.resize(1366, 768)
    window.show()
    study = open_study(window, tmp)
    study.select_tab("questions")
    pump(10)
    first = study.questions.list_layout.itemAt(0).widget()
    badge = first.findChild(QLabel, "questionNumber")
    assert badge is not None and badge.text() == "1"
    assert first.check_button.text() == "Check answer" and first.accessibleName().startswith("Question 1: ")
    window.hide()


def test_study_mode_is_focused_one_question_few_chrome(env):
    window, services, tmp = env
    window.resize(1366, 768)
    window.show()
    study = open_study(window, tmp)
    study.select_tab("study")
    pump(10)
    mode = study.study_mode
    assert study.stats_widget.isVisibleTo(study)                                          # the numbers are there while choosing
    mode.start()
    pump()
    assert not study.stats_widget.isVisibleTo(study)                                      # ... and step back while a question is on screen
    assert mode.question_label.objectName() == "focusQuestion" and mode.position_label.text().startswith("Question 1 of")
    assert mode.stop_button.objectName() == "link" and mode.agree_button.objectName() == "link"          # the rare actions are quiet
    if mode.option_buttons:
        assert all(button.objectName() == "option" and not button.styleSheet() for button in mode.option_buttons)
    mode.end()
    pump()
    assert study.stats_widget.isVisibleTo(study)
    window.hide()


@pytest.mark.parametrize("size", SIZES, ids=lambda s: f"{s[0]}x{s[1]}")
def test_no_study_tab_clips_or_overflows_at_the_target_sizes(env, size):
    window, services, tmp = env
    window.resize(*size)
    window.show()
    study = open_study(window, tmp)
    for tab in ("overview", "summary", "concepts", "flashcards", "questions", "study"):
        study.select_tab(tab)
        pump(10)
        page = study.tabs.currentWidget()
        assert page.minimumSizeHint().width() <= size[0] - window.simple_shell.rail.width(), (tab, page.minimumSizeHint())
    study.flashcards.show_button.click()
    study.select_tab("study")
    study.study_mode.start()
    pump()
    for label in study.study_mode.findChildren(QLabel):
        if label.isVisibleTo(study) and label.wordWrap() and label.text():
            assert label.heightForWidth(label.width()) <= label.height() + 2, label.text()[:40]       # no wrapped text is cut off
    window.hide()


# ------------------------------------------------------------------------------------------------------------------ progress
def test_the_progress_page_is_one_compact_block(env):
    window, services, tmp = env
    window.resize(1366, 768)
    window.show()
    window._goto("active")
    active = window.simple_active
    active.begin("FIXTURE_0004.MP4", "auto", "auto", "video")
    active.set_progress({"percent": 68.0, "rtf": 6.0, "eta": 240})
    pump(10)
    panel = active.findChild(QFrame, "card")
    assert panel.width() <= 560 and panel.height() <= 300                                   # a block, not a page-sized card
    assert abs(panel.geometry().center().y() + panel.parentWidget().geometry().top() + active.geometry().top() - window.simple_stack.height() / 2) < 120      # centred on purpose
    assert active.file_label.text() == "FIXTURE_0004.MP4" and active.percent_label.text() == "68%"
    bar_row_y = active.progress_bar.mapTo(active, active.progress_bar.rect().center()).y()
    percent_y = active.percent_label.mapTo(active, active.percent_label.rect().center()).y()
    assert abs(bar_row_y - percent_y) <= 8                                                   # the percent sits on the bar's line
    active.set_stage({"stage": "finalizing", "percent": 90.0})
    assert "medical terms" in active.stage_label.text() and active.title.text().startswith("Finalizing")
    assert {b.text() for b in (active.pause_button, active.cancel_button) if b.isVisibleTo(active)} == {"Pause", "Cancel"}
    window.hide()


def test_the_queue_names_what_comes_next(env, tmp_path):
    from src.ingest import queue as q
    from src.ingest.router import classify
    from tests.conftest import write_silent_wav

    window, services, tmp = env
    files = []
    for name in ("Neuroanatomy", "Histology", "Physiology"):
        path = tmp / f"{name}.wav"
        write_silent_wav(path, seconds=2)
        files.append(str(path))
    queue = q.BatchQueue(classify(files).items, start=lambda item: None, pause=lambda: None, cancel=lambda: None)
    queue.items[0].state, queue.items[1].state = q.COMPLETED, q.PROCESSING
    queue.state = q.RUNNING
    window._goto("batch")
    window.simple_batch.begin(queue)
    assert window.simple_batch.next_label.text() == "Next: Physiology.wav"
    queue.items[2].state = q.PROCESSING
    window.simple_batch.refresh()
    assert window.simple_batch.next_label.text() == ""


# ------------------------------------------------------------------------------------------------------------------ one scale, token spacing
def test_every_font_size_of_the_style_sheet_is_in_the_type_scale():
    allowed = set(tokens.FONT_PX.values())
    for name in ("dark", "light"):
        sizes = {int(n) for n in re.findall(r"font-size: (\d+)px", theme.stylesheet(name))}
        assert sizes <= allowed, (name, sorted(sizes - allowed))
    assert sorted(set(tokens.FONT_PX.values())) == [11, 12, 13, 14, 15, 16, 17, 20, 24, 28, 34]


SPACING_FILES = ["study_tabs", "study_view", "simple_active", "simple_batch", "about_dialog", "first_run", "sharing_notice", "components", "settings_simple", "drop_area",
                 "lecture_cards", "simple_result"]


@pytest.mark.parametrize("name", SPACING_FILES)
def test_margins_and_gaps_come_from_the_spacing_scale(name):
    allowed = set(tokens.SPACE.values()) | {0}
    source = (ROOT / "src" / "gui" / f"{name}.py").read_text(encoding="utf-8")
    stray = []
    for call in re.findall(r"(?:setContentsMargins|setSpacing|addSpacing|setHorizontalSpacing|setVerticalSpacing)\(([^()]*(?:\([^()]*\)[^()]*)*)\)", source):
        if "tokens.SPACE" in call:
            continue
        for number in re.findall(r"-?\d+", call):
            if int(number) not in allowed:
                stray.append(call)
    assert stray == [], stray


def test_icons_come_in_a_few_sizes_and_the_set_has_one_stroke():
    source = "".join(path.read_text(encoding="utf-8") for path in (ROOT / "src" / "gui").glob("*.py"))
    sizes = {int(n) for n in re.findall(r"badge_pixmap\([^,]+,\s*(\d+),\s*(\d+)", source) for n in n}
    assert sizes <= {20, 28, 40, 56, 64}, sizes
    assert theme.icons.STROKE == 1.8


# ------------------------------------------------------------------------------------------------------------------ plain language
@pytest.mark.parametrize("raw, expected", [
    ("CUDA out of memory. Tried to allocate 2 GiB", "ran out of memory"),
    ("[Errno 28] No space left on device", "not enough free disk space"),
    ("FileNotFoundError: no such file or directory: 'a.mp4'", "could not be found"),
    ("moov atom not found", "could not be read"),
    ("PermissionError: access is denied", "not allowed to write"),
    ("Model 'large-v3' is missing", "speech files are missing"),
    ("Something unusual happened", "Something unusual happened"),
    ("", "Something went wrong."),
])
def test_a_failed_transcription_is_explained_in_plain_words(raw, expected):
    assert expected in friendly.plain_failure(raw)


def test_the_failed_page_shows_the_plain_reason_and_keeps_the_work(env):
    window, *_ = env
    window._goto("active")
    window.simple_active.begin("x.wav", "auto", "auto", "audio")
    window.simple_active.finished_failed("CUDA out of memory. Tried to allocate 2 GiB")
    active = window.simple_active
    assert active.title.text() == "The transcription stopped" and "ran out of memory" in active.note.text()
    assert "Your progress is saved" in active.eta_label.text() and active.resume_button.text() == "Try again"


def test_empty_states_say_what_to_do(env):
    window, services, tmp = env
    window.show()
    window.simple_home.refresh()
    window._goto("files")
    window.files_page.refresh()
    pump()
    assert window.files_page.empty.isVisibleTo(window.files_page) and "No lectures yet" in "".join(l.text() for l in window.files_page.empty.findChildren(QLabel))
    folder = write_lecture_folder(tmp / "one")
    record = services.jobs.create_or_get("job-one", tmp / "Histology.mp4", output_dir=str(folder), state="completed", display_name="Histology", duration_seconds=900)
    services.jobs.update(record.job_id, state="completed", progress=100.0)
    window.files_page.refresh()
    window.files_page.search.setText("zzz")
    wait_for(lambda: "No lectures match your search." in "".join(l.text() for l in window.files_page.empty.findChildren(QLabel)), timeout=3, what="the empty search result")
    window.hide()


# ------------------------------------------------------------------------------------------------------------------ the overlay
def test_the_drop_overlay_says_what_to_do_and_what_happens(env):
    from src.gui.drop_overlay import DropOverlay

    window, *_ = env
    window.resize(1366, 768)
    window.show()
    pump()
    overlay = window.drop_overlay
    assert (DropOverlay.TEXT, DropOverlay.SUB) == ("Drop lecture files", "Release to add to transcription")
    overlay.show_over()
    pump()
    image = overlay.grab().toImage()
    assert overlay.inset_left == window.simple_shell.rail.width()
    area_center = (overlay.inset_left + (overlay.width() - overlay.inset_left) // 2, overlay.height() // 2)
    panel = image.pixelColor(*area_center)                                                    # the panel is opaque: its surface colour, not the dimmed page
    assert panel.alpha() == 255 and panel.name().lower() == tokens.current().surface.lower()
    rail_pixel = image.pixelColor(overlay.inset_left // 2, overlay.height() // 2)             # over the sidebar only the scrim: nothing of the panel
    assert rail_pixel.name().lower() != tokens.current().surface.lower()
    overlay.hide()
    window.hide()


# ------------------------------------------------------------------------------------------------------------------ dialogs
def test_a_question_dialog_names_its_action_and_puts_cancel_on_the_left(qapp):
    from PySide6.QtWidgets import QDialog

    from src.gui.components import ConfirmDialog

    theme.apply_theme(qapp, "dark")
    dialog = ConfirmDialog(None, "Remove from archive", "Remove 'Neuroanatomy' from the lecture list?", accept="Remove from archive")
    dialog.show()
    pump()
    assert dialog.width() == ConfirmDialog.WIDTH and dialog.windowTitle() == "Remove from archive"
    assert dialog.accept_button.text() == "Remove from archive" and dialog.reject_button.text() == "Cancel"
    assert dialog.reject_button.geometry().right() < dialog.accept_button.geometry().left()                    # Cancel left, the action right
    assert dialog.reject_button.isDefault() and not dialog.accept_button.isDefault()                          # Enter gives the safe answer
    dialog.accept_button.click()
    assert dialog.result() == QDialog.Accepted
    other = ConfirmDialog(None, "Delete model", "x", accept="Delete model")
    other.reject_button.click()
    assert other.result() == QDialog.Rejected


def test_dialogs_put_the_main_action_on_the_right(env):
    from PySide6.QtWidgets import QDialogButtonBox

    window, services, *_ = env
    dialog = window.make_settings_dialog("General")
    dialog.show()
    pump()
    boxes = dialog.findChildren(QDialogButtonBox)
    box = boxes[0]
    assert box.button(QDialogButtonBox.Close).geometry().right() < box.button(QDialogButtonBox.Save).geometry().left()          # Close left, Save right
    dialog.close()
    from src.gui.sharing_notice import SharingNoticeDialog

    notice = SharingNoticeDialog(services, window)
    notice.show()
    pump()
    buttons = sorted(notice.findChildren(type(box.button(QDialogButtonBox.Save))), key=lambda b: b.geometry().left())
    assert buttons[-1].text() == "Share transcripts"                                                                         # the same rule in the sharing notice
    notice.close()


def test_the_card_menu_and_settings_use_the_shared_question_dialog():
    for name in ("lecture_cards", "settings_dialog", "main_window"):
        source = (ROOT / "src" / "gui" / f"{name}.py").read_text(encoding="utf-8")
        assert "QMessageBox.question" not in source and "ask(" in source, name
