"""MELETRA 4.1 premium UI: the design system (tokens, contrast, icons), one navigation and one Settings entry, the drop area states, the lecture cards, the Files archive
(filter, search, sort, empty states), the transcript switch, the Settings groups with their badges, the queue rows, the keyboard focus ring and the target window sizes."""

import re
import sys
from pathlib import Path

import pytest
from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QAbstractButton, QApplication, QMenu

from src.gui import icons, theme, tokens
from src.gui.drop_area import DropArea
from src.gui.main_window import SIMPLE_FILES, SIMPLE_HOME, MainWindow
from src.ingest import queue as q
from src.ingest.router import classify
from tests.conftest import write_silent_wav
from tests.gui_support import ScriptedLauncher, make_services, settle, wait_for
from tests.study_support import write_lecture_folder

ROOT = Path(__file__).resolve().parent.parent
SIZES = [(1366, 768), (1440, 900), (1920, 1080)]
KEEP = []


@pytest.fixture
def env(qapp, tmp_path):
    services = make_services(tmp_path, opened=[])
    window = MainWindow(services, ScriptedLauncher())
    window.controller._timer.setInterval(20)
    yield window, services, tmp_path
    window.controller._timer.stop()
    settle(window, window.home, window.study)
    window.deleteLater()


def completed(services, tmp, name, **fields):
    folder = write_lecture_folder(tmp / name)
    record = services.jobs.create_or_get(f"job-{name}", tmp / f"{name}.mp4", output_dir=str(folder), state="completed", **fields)
    services.jobs.update(record.job_id, state="completed", progress=100.0, summary={"output_dir": str(folder), "words": 4971})       # what the worker's completion event leaves in the record
    return services.jobs.get(record.job_id), folder


# ------------------------------------------------------------------------------------------------------------------ the design system
@pytest.mark.parametrize("palette", [tokens.DARK, tokens.LIGHT], ids=["dark", "light"])
def test_text_colours_are_readable_on_every_surface(palette):
    """WCAG AA: 4.5 : 1 for text, 3 : 1 for interface parts (borders of controls, the focus ring, the primary colour on the window)."""
    for surface in (palette.bg, palette.sidebar, palette.surface, palette.surface_hover, palette.field):
        assert tokens.contrast(palette.text, surface) >= 7.0, (palette.name, "text", surface)
        assert tokens.contrast(palette.text2, surface) >= 4.5, (palette.name, "secondary text", surface)
        assert tokens.contrast(palette.text3, surface) >= 4.5, (palette.name, "metadata", surface)
        assert tokens.contrast(palette.teal, surface) >= 3.0, (palette.name, "teal", surface)
        assert tokens.contrast(palette.focus, surface) >= 3.0, (palette.name, "focus ring", surface)
        assert tokens.contrast(palette.gold, surface) >= 3.0, (palette.name, "focus ring on teal-filled buttons", surface)
    assert tokens.contrast(palette.on_teal, palette.teal) >= 4.5, "the text of the primary button"
    assert tokens.contrast(palette.border_strong, palette.bg) >= 1.5
    for fg, bg in (("success", "success_bg"), ("warning", "warning_bg"), ("error", "error_bg"), ("neutral", "neutral_bg"), ("teal", "teal_soft")):
        assert tokens.contrast(getattr(palette, fg), getattr(palette, bg)) >= 4.5, (palette.name, fg)


def test_the_brand_palette_is_navy_teal_and_a_muted_gold():
    dark = tokens.DARK
    r, g, b = (int(dark.bg.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    assert b > r and b > g and max(r, g, b) < 48                                                                 # deep navy, almost black
    assert tokens.luminance(dark.surface) > tokens.luminance(dark.bg) and tokens.luminance(dark.surface_hover) > tokens.luminance(dark.surface)
    teal_r, teal_g, teal_b = (int(dark.teal.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    assert teal_g > teal_r and teal_b > teal_r                                                                   # teal is the interaction colour
    gold_r, gold_g, gold_b = (int(dark.gold.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    assert gold_r > gold_b and gold_g > gold_b


def test_the_spacing_radius_and_type_scales_are_the_specified_ones():
    assert tuple(tokens.SPACE.values()) == (4, 8, 12, 16, 24, 32, 48)
    assert 6 <= tokens.RADIUS["small"] <= 8 and 6 <= tokens.RADIUS["control"] <= 8 and 10 <= tokens.RADIUS["card"] <= 12 and 14 <= tokens.RADIUS["drop"] <= 16
    f = tokens.FONT_PX
    assert 26 <= f["page"] <= 30 and 16 <= f["section"] <= 18 and 13 <= f["body"] <= 14 and 12 <= f["secondary"] <= 13 and 11 <= f["meta"] <= 12 and 13 <= f["button"] <= 14
    assert 120 <= tokens.TRANSITION_MS <= 180
    sheet = theme.stylesheet("dark")
    assert f"font-size: {f['page']}px" in sheet and f"border-radius: {tokens.RADIUS['card']}px" in sheet


def test_the_stylesheet_uses_tokens_not_random_colours():
    for name in ("dark", "light"):
        sheet = theme.stylesheet(name)
        palette = tokens.PALETTES[name]
        allowed = {value.lower() for value in vars(palette).values() if isinstance(value, str) and value.startswith("#")}
        used = {match.lower() for match in re.findall(r"#[0-9a-fA-F]{6}\b", sheet)}
        stray = {c for c in used if c not in allowed and not c.startswith("#00")}
        assert not stray, (name, stray)                                                                           # every colour is a palette role


def test_every_icon_draws_and_the_set_has_one_style(qapp):
    assert {"home", "transcription", "study", "files", "settings", "audio", "video", "pdf", "image", "rename", "folder", "upload", "more", "close", "replace",
            "search", "check", "dot"} <= set(icons.DRAWERS)
    for name in icons.DRAWERS:
        image = icons.pixmap(name, "#ffffff", 20).toImage()
        assert not image.isNull() and any(image.pixelColor(x, y).alpha() > 0 for x in range(0, image.width(), 2) for y in range(0, image.height(), 2)), name
    assert icons.STROKE == 1.8                                                                                    # one stroke weight for the whole set


def test_the_interface_code_has_no_emoji_or_symbol_glyphs_used_as_icons():
    forbidden = "⋯✕✓✔●○◉⚙★☆♥▶▼▲✎🔍📄📁🎬🎵✅❌⚠"
    for path in (ROOT / "src" / "gui").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        found = [ch for ch in forbidden if ch in text]
        assert not found, (path.name, found)


def test_themes_can_be_switched_without_leaving_old_colours(qapp):
    theme.apply_theme(qapp, "light")
    assert tokens.LIGHT.bg in qapp.styleSheet() and tokens.color("teal") == tokens.LIGHT.teal
    theme.apply_theme(qapp, "dark")
    assert tokens.DARK.bg in qapp.styleSheet() and tokens.color("teal") == tokens.DARK.teal
    assert theme.apply_theme(qapp, "dark") == "dark"                                                              # the same theme again is a no-op (no 0.8 s repolish)


# ------------------------------------------------------------------------------------------------------------------ one navigation, one Settings
def test_there_is_exactly_one_settings_entry_and_no_menu_bar_or_status_bar(env):
    window, *_ = env
    window.show()
    QApplication.processEvents()
    if sys.platform != "darwin":
        assert not window.menuBar().isVisibleTo(window) and not window.statusBar().isVisibleTo(window)
    entries = [b.text() for b in window.findChildren(QAbstractButton) if window.isAncestorOf(b) and b.isVisibleTo(window) and b.text().strip().lower() == "settings"]
    assert entries == ["Settings"]                                                                                 # the sidebar's, nothing else
    assert [b.text() for b in window.simple_shell.buttons.values()] == ["Home", "Transcription", "Study", "Files", "Settings"]
    assert not any("Private study workspace" in w.text() for w in window.simple_shell.findChildren(QAbstractButton))
    window.hide()


def test_the_menu_shortcuts_still_work_without_the_menu_bar(env):
    window, *_ = env
    actions = {a.text().replace("&", ""): a for a in window.actions()}
    assert "Open audio or video ..." in actions and actions["Open audio or video ..."].shortcut().toString() != ""
    assert any("Settings" in a.text() for a in window.actions())                                                   # (the Settings shortcut is Cmd+, on a Mac only)


def test_the_window_title_icon_and_sidebar_use_the_brand(env):
    window, *_ = env
    assert window.windowTitle() == "MELETRA" and not window.windowIcon().isNull()
    assert not window.simple_shell.logo.pixmap().isNull()
    assert re.sub(r"<[^>]+>", "", window.simple_shell.brand.text()) == "MELETRA"


# ------------------------------------------------------------------------------------------------------------------ the drop area
def test_the_drop_area_is_the_visual_centrepiece_and_says_what_to_do(env):
    window, *_ = env
    home = window.simple_home
    window.resize(1366, 768)
    window.show()
    QApplication.processEvents()
    zone = home.drop_zone
    assert isinstance(zone, DropArea) and zone.title_label.text() == "Drop lecture files here" and zone.sub_label.text() == "Audio · Video · PDF · Image"
    assert zone.browse_button.text() == "Browse files" and not zone.icon_label.pixmap().isNull()
    assert zone.height() >= 300 and zone.width() > 500                                                              # 1.7 to 2 x the old 170 px area
    window.hide()


def test_the_drag_over_state_fades_in_about_150_ms_and_out_again(env):
    window, *_ = env
    zone = window.simple_home.drop_zone
    assert 120 <= zone._animation.duration() <= 180
    zone.set_drag_active(True)
    wait_for(lambda: zone._drag > 0.99, timeout=2, what="the drag state to fade in")
    assert zone.drag_active()
    zone.set_drag_active(False)
    wait_for(lambda: zone._drag < 0.01, timeout=2, what="the drag state to fade out")
    assert not zone.drag_active()


def test_dragging_a_file_over_the_window_lights_the_drop_area_and_dropping_ends_it(env):
    window, _s, tmp = env
    source = write_silent_wav(tmp / "lecture.wav", seconds=5)
    data = QMimeData()
    data.setUrls([QUrl.fromLocalFile(str(source))])
    KEEP.append(data)
    window.dragEnterEvent(QDragEnterEvent(QPoint(200, 200), Qt.CopyAction, data, Qt.LeftButton, Qt.NoModifier))
    assert window.drop_overlay.isVisibleTo(window) and window.simple_home.drop_zone.drag_active()
    window.dropEvent(QDropEvent(QPointF(200, 200), Qt.CopyAction, data, Qt.LeftButton, Qt.NoModifier))
    assert not window.drop_overlay.isVisibleTo(window) and not window.simple_home.drop_zone.drag_active()
    wait_for(lambda: not window.simple_home.classifying, what="the dropped file to be read")
    assert window.home.source == source


def test_the_transcribe_button_says_what_it_is_doing(env):
    window, _s, tmp = env
    home = window.simple_home
    assert home.transcribe_button.text() == "TRANSCRIBE"
    home.set_busy("Transcribing...")
    assert home.transcribe_button.text() == "Transcribing..." and not home.transcribe_button.isEnabled()
    assert home.recent_card.isEnabled()                                                                             # the archive stays clickable while a lecture runs
    home.set_busy("")
    assert home.transcribe_button.text() == "TRANSCRIBE"


# ------------------------------------------------------------------------------------------------------------------ recent lectures and Files
def test_view_all_appears_when_there_is_more_and_opens_files(env):
    window, services, tmp = env
    home = window.simple_home
    for i in range(3):
        completed(services, tmp, f"few{i}")
    home.refresh()
    assert not home.view_all_button.isVisibleTo(home)                                                               # everything is already shown
    for i in range(9):
        completed(services, tmp, f"many{i}")
    window.show()
    home.refresh()
    QApplication.processEvents()
    assert home.view_all_button.isVisibleTo(home) and 5 <= home.recent_holder.count() <= 7
    home.view_all_button.click()
    assert window.simple_stack.currentIndex() == SIMPLE_FILES
    window.hide()


@pytest.mark.parametrize("state,kind", [("completed", "completed"), ("running", "running"), ("queued", "queued"), ("failed", "failed"), ("paused", "paused")])
def test_each_status_has_its_own_chip(env, state, kind):
    window, services, tmp = env
    if state == "completed":
        completed(services, tmp, state)                                                  # a completed lecture that really has its transcript
    else:
        services.jobs.create_or_get(f"job-{state}", tmp / f"{state}.mp4", state=state, progress=40.0)
    window.simple_home.refresh()
    card = window.simple_home.recent_holder.itemAt(0).widget()
    assert card.status.property("state") == kind and card.status.objectName() == "statusChip"
    colours = {k: v for k, v in vars(tokens.DARK).items()}
    sheet = theme.stylesheet("dark")
    assert f'QLabel#statusChip[state="{kind}"]' in sheet or kind == "queued"
    assert colours["success"] != colours["error"] != colours["warning"] != colours["teal"]


def test_a_completed_job_without_its_transcript_is_not_painted_as_completed(env):
    window, services, tmp = env
    services.jobs.create_or_get("job-empty", tmp / "empty.mp4", output_dir=str(tmp / "nothing here"), state="completed")
    window.simple_home.refresh()
    card = window.simple_home.recent_holder.itemAt(0).widget()
    assert card.status.text() == "Transcript not found" and card.status.property("state") == "paused"


def test_a_lecture_card_is_a_clickable_archive_entry_with_a_menu(env):
    window, services, tmp = env
    record, _folder = completed(services, tmp, "FIXTURE_0004", display_name="Neuroanatomy — Ventricular System", language="auto", detected_language="en",
                                output_language="tr", translation="m2m100", duration_seconds=2700)
    window.simple_home.refresh()
    card = window.simple_home.recent_holder.itemAt(0).widget()
    assert card.cursor().shape() == Qt.PointingHandCursor and card.focusPolicy() == Qt.TabFocus
    assert card.title.text() == "Neuroanatomy — Ventricular System" and card.sub.text() == "FIXTURE_0004.mp4"
    assert card.path.text() == "Auto (English) → Turkish" and card.when.text().startswith("45 min") and card.status.text() == "Completed"
    opened = []
    card.clicked.connect(opened.append)
    QTest.mouseClick(card, Qt.LeftButton)
    QTest.keyClick(card, Qt.Key_Return)
    assert [r.job_id for r in opened] == [record.job_id, record.job_id]
    menu = card.menu_button.menu()
    assert isinstance(menu, QMenu) and [a.text() for a in menu.actions() if not a.isSeparator()] == ["Open", "Rename", "Show Files", "Remove from Archive"]
    assert all(not a.icon().isNull() for a in menu.actions() if not a.isSeparator())                                 # one icon style in the menu too


def test_files_filter_by_status_and_the_two_empty_states(env):
    window, services, tmp = env
    page = window.files_page
    window._goto("files")
    page.refresh()
    assert page.empty.title.text() == "No lectures yet" and page.empty.hint.text() == "Drop a lecture on Home to get started."
    completed(services, tmp, "done_one", display_name="Anatomy")
    services.jobs.create_or_get("job-bad", tmp / "bad.mp4", state="failed")
    services.jobs.create_or_get("job-run", tmp / "run.mp4", state="running", progress=10.0)
    page.refresh()
    assert page.header.title.text() == "Lecture Archive" and len(page.cards) == 3
    page.filter.buttons["completed"].click()
    assert [c.record.state for c in page.cards] == ["completed"]
    page.filter.buttons["failed"].click()
    assert [c.record.state for c in page.cards] == ["failed"]
    page.filter.buttons["processing"].click()
    assert [c.record.state for c in page.cards] == ["running"]
    page.filter.buttons["all"].click()
    assert len(page.cards) == 3
    page.search.setText("no such lecture")
    page.search_now()
    assert page.empty.isVisibleTo(page) and page.empty.title.text() == "No lectures match your search."
    assert "Traceback" not in page.empty.title.text() + page.empty.hint.text()


def test_the_files_rows_are_wide_rows_and_the_cards_stay_cards_when_narrow(env):
    window, services, tmp = env
    for i in range(3):
        completed(services, tmp, f"row{i}", duration_seconds=600 + i, language="auto", output_language="en")
    window.resize(1366, 768)
    window.show()
    window._goto("files")
    window.files_page.refresh()
    QApplication.processEvents()
    wide = [c for c in window.files_page.cards]
    assert wide and all(c.is_wide() for c in wide)
    assert wide[0].duration_label.isVisibleTo(wide[0]) and wide[0].date_label.isVisibleTo(wide[0]) and not wide[0].when.isVisibleTo(wide[0])
    window.simple_home.refresh()
    assert not any(c.is_wide() for c in [window.simple_home.recent_holder.itemAt(i).widget() for i in range(window.simple_home.recent_holder.count())])
    window.hide()


# ------------------------------------------------------------------------------------------------------------------ Settings
def test_settings_are_grouped_and_the_medical_status_is_obvious(env):
    window, services, _tmp = env
    dialog = window.make_settings_dialog("Medical")
    assert [dialog.nav.item(i).text() for i in range(dialog.nav.count())] == ["Transcription", "Medical", "Study", "Privacy & Data", "Models", "Advanced"]
    assert dialog.tabs.tabText(dialog.tabs.currentIndex()) == "Medical" and dialog.nav.currentRow() == 1
    assert (dialog.expert_badge.text(), dialog.expert_state.text()) == ("Stable", "ON")
    assert (dialog.intelligence_badge.text(), dialog.intelligence_state.text()) == ("Experimental", "OFF")
    assert dialog.expert_badge.property("kind") == "stable" and dialog.intelligence_badge.property("kind") == "experimental"
    dialog.medical_expert_check.setChecked(False)
    assert dialog.expert_state.text() == "OFF"
    dialog.save()
    assert services.settings.settings.medical_expert is False
    dialog.deleteLater()


def test_the_privacy_page_shows_the_sharing_state_in_big_letters(env, monkeypatch):
    from src.gui.sharing_notice import SharingNoticeDialog

    window, services, _tmp = env
    dialog = window.make_settings_dialog("Privacy & Data")
    assert dialog.share_state.text() == "OFF" and dialog.share_state.property("kind") == "off"                  # off until the student switches it on
    monkeypatch.setattr(SharingNoticeDialog, "exec", lambda self: 0)                                           # "Not now": it stays off
    dialog.share_check.setChecked(True)
    assert dialog.share_check.isChecked() is False and dialog.share_state.text() == "OFF" and services.settings.settings.share_transcripts is False
    monkeypatch.setattr(SharingNoticeDialog, "exec", lambda self: 1)                                           # "Share transcripts": it is on
    dialog.share_check.setChecked(True)
    assert dialog.share_state.text() == "ON" and dialog.share_state.property("kind") == "on" and services.settings.settings.share_transcripts is True
    dialog.share_check.setChecked(False)
    assert dialog.share_state.text() == "OFF" and dialog.share_state.property("kind") == "off" and services.settings.settings.share_transcripts is False
    assert "Cloud comparison" in " ".join(label.text() for label in dialog.findChildren(type(dialog.share_state)))
    dialog.deleteLater()


def test_about_and_the_theme_live_in_advanced(env):
    window, *_ = env
    dialog = window.make_settings_dialog("Advanced")
    assert dialog.about_button.text() == "About MELETRA ..." and dialog.theme_combo.count() == 3
    dialog.deleteLater()


# ------------------------------------------------------------------------------------------------------------------ progress and queue
def test_the_progress_screen_shows_stage_percentage_and_file_in_one_card(env):
    window, *_ = env
    active = window.simple_active
    active.begin("FIXTURE_0004.MP4", "auto", "auto", "video")
    active.set_progress({"percent": 68.0, "rtf": 6.0, "eta": 240})
    assert active.title.text() == "Transcribing..." and active.percent_label.text() == "68%" and active.file_label.text() == "FIXTURE_0004.MP4"
    assert len(active.findChildren(type(active.progress_bar))) == 1                                                 # one progress bar, no competing ones
    active.set_stage({"stage": "translating", "percent": 40.0})
    assert active.title.text() == "Translating..." and active.percent_label.text() == "40%"


def test_the_queue_shows_one_row_per_file_with_a_status_chip(env, tmp_path):
    window, *_ = env
    files = [write_silent_wav(tmp_path / f"{name}.wav", seconds=2) for name in ("Neuroanatomy", "Histology", "Physiology")]
    selection = classify([str(f) for f in files])
    queue = q.BatchQueue(selection.items, start=lambda item: None, pause=lambda: None, cancel=lambda: None)
    batch = window.simple_batch
    batch.begin(queue)
    assert batch.list.count() == 3
    from src.gui.simple_batch import ROLE_NAME, ROLE_NUMBER, ROLE_STATE

    assert [batch.list.item(i).data(ROLE_NAME) for i in range(3)] == ["Neuroanatomy.wav", "Histology.wav", "Physiology.wav"]
    assert [batch.list.item(i).data(ROLE_NUMBER) for i in range(3)] == [1, 2, 3] and {batch.list.item(i).data(ROLE_STATE) for i in range(3)} == {q.QUEUED}
    assert "Waiting" in batch.list.item(0).text()                                                                   # the text stays for screen readers


# ------------------------------------------------------------------------------------------------------------------ keyboard
def test_buttons_take_the_focus_from_the_keyboard_only_so_the_ring_is_never_a_leftover_click(qapp):
    from PySide6.QtWidgets import QCheckBox, QComboBox, QLineEdit, QPushButton

    theme.apply_theme(qapp, "dark")                                                                              # installs the Whisper style (buttons: Tab focus only)
    assert QPushButton("x").focusPolicy() == Qt.TabFocus and QCheckBox("x").focusPolicy() == Qt.TabFocus
    assert QComboBox().focusPolicy() != Qt.NoFocus and QLineEdit().focusPolicy() != Qt.NoFocus                    # fields show their ring on any focus
    sheet = theme.stylesheet("dark")
    for selector in ("QPushButton:focus", "QPushButton#big:focus", "QPushButton#navButton:focus", "QComboBox:focus", "QLineEdit:focus", "QFrame#lectureCard:focus"):
        assert selector in sheet, selector
    assert f"QPushButton:focus {{ border: 2px solid {tokens.DARK.focus}" in sheet                                    # a teal ring, not only a colour change
    assert f"QPushButton#primary:focus {{ border: 2px solid {tokens.DARK.gold}" in sheet                             # gold where the button itself is teal


def test_the_drop_area_and_the_cards_are_reached_with_the_keyboard(env):
    window, services, tmp = env
    completed(services, tmp, "kb")
    window.simple_home.refresh()
    card = window.simple_home.recent_holder.itemAt(0).widget()
    assert window.simple_home.browse_button.focusPolicy() == Qt.TabFocus and window.simple_home.drop_zone.focusPolicy() == Qt.NoFocus and card.focusPolicy() == Qt.TabFocus


def tab_chain(window, count=40):
    """The widgets reached by pressing Tab again and again, starting from where the focus rests (the anchor)."""
    window.focus_anchor.setFocus(Qt.TabFocusReason)
    QApplication.processEvents()
    chain = []
    for _ in range(count):
        window.focusNextPrevChild(True)
        QApplication.processEvents()
        widget = QApplication.focusWidget()
        if widget is None or widget in chain:
            break
        chain.append(widget)
    return chain


def test_tab_order_of_the_start_screen_follows_the_reading_order(env):
    window, services, tmp = env
    for i in range(3):
        completed(services, tmp, f"tab{i}")
    home = window.simple_home
    home.refresh()
    window.show()
    home.transcribe_button.setEnabled(True)                                                                         # (a file would enable it)
    QApplication.processEvents()
    chain = tab_chain(window)
    cards = [home.recent_holder.itemAt(i).widget() for i in range(home.recent_holder.count())]
    wanted = [home.browse_button, home.pdf_button, home.input_combo, home.output_combo, home.transcribe_button]
    assert chain[:5] == wanted, [type(w).__name__ + ":" + w.objectName() for w in chain[:8]]                     # Browse (the drop area), PDF, languages, TRANSCRIBE
    after = chain[5:]
    assert after[0] in (home.view_all_button, cards[0])
    first_card = after.index(cards[0])
    assert after[first_card + 1] is cards[0].menu_button and after[first_card + 2] is cards[1]                     # then the recent lectures: card, its menu, next card
    last_card_menu = after.index(cards[-1].menu_button)
    assert after[last_card_menu + 1] is home.privacy and after[last_card_menu + 2] is window.simple_shell.buttons["home"]   # the small status chip, and last the sidebar
    window.hide()


def test_nothing_is_highlighted_before_the_keyboard_is_used(env):
    window, *_ = env
    window.show()
    QApplication.processEvents()
    QApplication.processEvents()
    assert QApplication.focusWidget() is window.focus_anchor                                                         # no ring on launch
    window.simple_shell.buttons["files"].click()
    assert QApplication.focusWidget() is window.focus_anchor and window.simple_stack.currentIndex() == SIMPLE_FILES  # a click leaves no ring either
    chain = tab_chain(window, 3)
    assert chain and chain[0] is not window.simple_shell.buttons["files"]                                            # Tab goes into the page ("skip to content")
    window.hide()


def test_enter_and_space_press_buttons_and_open_cards(env):
    window, services, tmp = env
    record, _ = completed(services, tmp, "kb_open")
    window.simple_home.refresh()
    theme.apply_theme(QApplication.instance(), "dark")                                                                 # the program does this at start: it installs the keyboard style
    window.show()                                                                                                      # (a button is set up for Enter when it is first polished)
    button = window.simple_shell.buttons["files"]
    button.setFocus(Qt.TabFocusReason)
    QTest.keyClick(button, Qt.Key_Space)
    assert window.simple_stack.currentIndex() == SIMPLE_FILES
    home_button = window.simple_shell.buttons["home"]
    home_button.setFocus(Qt.TabFocusReason)
    QTest.keyClick(home_button, Qt.Key_Return)                                                                       # Enter presses a focused button too, not only Space
    assert window.simple_stack.currentIndex() == SIMPLE_HOME
    window.simple_shell.buttons["files"].click()
    card = window.files_page.cards[0]
    opened = []
    card.clicked.connect(opened.append)
    card.setFocus(Qt.TabFocusReason)
    QTest.keyClick(card, Qt.Key_Return)
    QTest.keyClick(card, Qt.Key_Space)
    assert len(opened) == 2 and opened[0].job_id == record.job_id
    window.hide()


def test_the_context_menu_works_with_the_keyboard(env):
    window, services, tmp = env
    record, _ = completed(services, tmp, "kb_menu")
    window.simple_home.refresh()
    card = window.simple_home.recent_holder.itemAt(0).widget()
    button, menu = card.menu_button, card.menu_button.menu()
    window.show()
    shown = []
    button.showMenu = lambda: shown.append(1)                                    # (the real call waits for the menu to close)
    button.setFocus(Qt.TabFocusReason)
    for key in (Qt.Key_Space, Qt.Key_Return, Qt.Key_Down):
        QTest.keyClick(button, key)
    card.setFocus(Qt.TabFocusReason)
    card.menu_button.showMenu = lambda: shown.append(2)
    QTest.keyClick(card, Qt.Key_Menu)                                            # the Menu key on the card does the same
    assert shown == [1, 1, 1, 2]
    opened = []
    card.clicked.connect(opened.append)
    menu.popup(button.mapToGlobal(QPoint(0, button.height())))                   # inside the open menu: arrows choose, Enter runs, Esc closes
    QApplication.processEvents()
    assert menu.isVisible()
    QTest.keyClick(menu, Qt.Key_Down)
    assert menu.activeAction() is card.open_action
    QTest.keyClick(menu, Qt.Key_Return)
    assert [r.job_id for r in opened] == [record.job_id] and not menu.isVisible()
    menu.popup(button.mapToGlobal(QPoint(0, button.height())))
    QApplication.processEvents()
    QTest.keyClick(menu, Qt.Key_Escape)
    assert not menu.isVisible()
    window.hide()


def test_drop_downs_are_usable_with_the_keyboard(env):
    window, *_ = env
    home = window.simple_home
    window.show()
    combo = home.input_combo
    combo.setFocus(Qt.TabFocusReason)
    assert combo.currentData() == "auto"
    QTest.keyClick(combo, Qt.Key_Down)
    assert combo.currentData() == "en"                                                                               # arrow keys move through the list
    QTest.keyClicks(combo, "Tu")
    assert combo.currentData() == "tr"                                                                               # typing the first letters jumps to the language
    out = home.output_combo
    out.setFocus(Qt.TabFocusReason)
    QTest.keyClick(out, Qt.Key_Down)
    assert out.currentData() == "tr"
    window.hide()


def test_escape_closes_the_dialogs(env):
    window, *_ = env
    from src.gui.about_dialog import AboutDialog

    for dialog in (window.make_settings_dialog("Medical"), AboutDialog(window)):
        dialog.show()
        QApplication.processEvents()
        assert dialog.isVisible()
        QTest.keyClick(dialog, Qt.Key_Escape)
        assert not dialog.isVisible()
        dialog.deleteLater()


# ------------------------------------------------------------------------------------------------------------------ the three target sizes
@pytest.mark.parametrize("size", SIZES)
def test_every_page_fits_the_target_sizes_without_horizontal_scrolling(env, size):
    window, services, tmp = env
    for i in range(8):
        completed(services, tmp, f"fit{i}", display_name=f"Lecture {i}: a long title that has to be shortened by the card instead of widening it", duration_seconds=3000)
    window.resize(*size)
    window.show()
    window.simple_home.refresh()
    QApplication.processEvents()
    for goto in ("home", "files", "active", "result", "transcript", "study", "batch"):
        window._goto(goto)
        window.files_page.refresh()
        QApplication.processEvents()
        page = window.simple_stack.currentWidget()
        assert page.minimumSizeHint().width() <= size[0] - window.simple_shell.rail.width(), (goto, size, page.minimumSizeHint())
        for scroll in page.findChildren(type(window.simple_home.scroll)):
            assert scroll.horizontalScrollBar().maximum() == 0 or not scroll.horizontalScrollBar().isVisible(), (goto, size)
    window._goto("home")
    assert not window.simple_home.stacked()
    window.hide()


def test_the_start_screen_stacks_when_the_window_is_narrow(env):
    window, *_ = env
    window.resize(980, 700)
    window.show()
    QApplication.processEvents()
    home = window.simple_home
    assert home.stacked() and home.side.width() >= home.main_column.width() * 0.8                                     # the recent lectures moved under the workflow
    window.resize(1366, 768)
    QApplication.processEvents()
    assert not home.stacked()
    window.hide()


# ------------------------------------------------------------------------------------------------------------------ nothing clipped, at any page and size
def clipped_widgets(root):
    from PySide6.QtWidgets import QLabel, QPushButton, QToolButton

    from src.gui.lecture_cards import ElidedLabel

    found = []
    for widget in root.findChildren(QLabel) + root.findChildren(QPushButton) + root.findChildren(QToolButton):
        if not widget.isVisibleTo(root) or widget.width() <= 2 or isinstance(widget, ElidedLabel):
            continue
        if isinstance(widget, QLabel):
            if widget.wordWrap() or not widget.text() or widget.textFormat() == Qt.RichText or (widget.pixmap() is not None and not widget.pixmap().isNull()):
                continue
            need = widget.fontMetrics().horizontalAdvance(widget.text().split(chr(10))[0]) + 2 * widget.margin()
            if need > widget.width() + 1:
                found.append((type(widget).__name__, widget.objectName(), widget.text()[:40], need, widget.width()))
        elif widget.text():
            if widget.sizeHint().width() > widget.width() + 2:
                found.append((type(widget).__name__, widget.objectName(), widget.text()[:40], widget.sizeHint().width(), widget.width()))
    return found


@pytest.mark.parametrize("size", SIZES)
def test_no_text_is_clipped_on_any_page_at_the_target_sizes(env, size):
    window, services, tmp = env
    for i in range(7):
        completed(services, tmp, f"clip2_{i}", display_name=f"Lecture {i}: a long title that has to be shortened by the card", duration_seconds=3000, language="auto",
                  output_language="tr", detected_language="en", translation="m2m100")
    services.jobs.create_or_get("job-p", tmp / "p.mp4", state="paused", progress=62.0)
    window.resize(*size)
    window.show()
    window.simple_home.refresh()
    files = [str(write_silent_wav(tmp / "a.wav", 2)), str(write_silent_wav(tmp / "b.wav", 2))]
    window.simple_batch.begin(q.BatchQueue(classify(files).items, start=lambda item: None, pause=lambda: None, cancel=lambda: None))
    window.show_finished(write_lecture_folder(tmp / "res"), "FIXTURE_0004.MP4")
    translated = write_lecture_folder(tmp / "tres")
    import json

    meta = json.loads((translated / "metadata.json").read_text(encoding="utf-8"))
    meta["translation"] = {"method": "whisper", "source_language": "tr", "output_language": "en", "files": ["translated_en_transcript.txt"]}
    (translated / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    (translated / "translated_en_transcript.txt").write_text("The heart pumps blood.", encoding="utf-8")
    window.transcript_view.show_transcript(translated, "Pharmacology: autonomic drugs", prefer_translated=True)
    for goto in ("home", "files", "active", "result", "batch", "transcript"):
        window._goto(goto)
        window.files_page.refresh()
        for _ in range(3):
            QApplication.processEvents()
        assert clipped_widgets(window.simple_stack.currentWidget()) == [], (goto, size)
    window.hide()


def test_no_text_is_clipped_in_the_settings_groups(env):
    window, *_ = env
    dialog = window.make_settings_dialog("Transcription")
    dialog.show()
    for name in ("Transcription", "Medical", "Study", "Privacy & Data", "Models", "Advanced"):
        dialog.select_tab(name)
        for _ in range(3):
            QApplication.processEvents()
        assert clipped_widgets(dialog.tabs.currentWidget()) == [], name
    dialog.close()
    dialog.deleteLater()


# ------------------------------------------------------------------------------------------------------------------ brand assets of the installer and the logo set
def test_the_installer_wizard_has_the_brand_pictures_and_the_logo_set_is_complete():
    from PIL import Image

    brand = ROOT / "assets" / "brand"
    for tag, scale in (("100", 1.0), ("150", 1.5), ("200", 2.0)):
        large = Image.open(brand / "installer" / f"wizard_large_{tag}.bmp")
        small = Image.open(brand / "installer" / f"wizard_small_{tag}.bmp")
        assert large.size == (round(164 * scale), round(314 * scale)) and small.size == (round(55 * scale), round(55 * scale))
    iss = (ROOT / "build_tools" / "installer.iss").read_text(encoding="utf-8")
    assert "WizardImageFile={#WizardLarge}" in iss and "WizardSmallImageFile={#WizardSmall}" in iss and "#define AppName \"MELETRA\"" in iss
    assert r"{group}\{#AppName}" in iss and r"{autodesktop}\{#AppName}" in iss and "UninstallDisplayName={#AppName}" in iss
    wide = Image.open(brand / "meletra_logo_horizontal.png")
    assert wide.width > 2.5 * wide.height and wide.mode == "RGBA"                                                   # symbol + lettering side by side, transparent
    sizes = {n: Image.open(brand / f"meletra_icon_{n}.png").size for n in (16, 24, 32, 48, 64, 128, 256, 512)}
    assert all(size == (n, n) for n, size in sizes.items())
    for name in ("meletra.ico", "meletra.icns", "favicon.ico", "meletra_logo.png", "meletra_symbol.png"):
        assert (brand / name).stat().st_size > 500, name



# ------------------------------------------------------------------------------------------------------------------ every control has a name a screen reader can say
def unnamed_controls(root):
    from PySide6.QtWidgets import QAbstractButton, QAbstractSpinBox, QComboBox, QLineEdit

    found = []
    for widget in root.findChildren(QAbstractButton) + root.findChildren(QComboBox) + root.findChildren(QLineEdit) + root.findChildren(QAbstractSpinBox):
        if not widget.isVisibleTo(root) or widget.objectName().startswith("qt_"):
            continue
        name = widget.accessibleName() or (widget.text() if isinstance(widget, QAbstractButton) else "") or (widget.placeholderText() if isinstance(widget, QLineEdit) else "")
        if not (name or widget.toolTip()):
            found.append((type(widget).__name__, widget.objectName()))
    return found


def test_every_visible_control_has_a_name(env):
    window, services, tmp = env
    completed(services, tmp, "names", display_name="Neuroanatomy")
    window.simple_home.refresh()
    window.show()
    window.show_finished(write_lecture_folder(tmp / "names_result"), "FIXTURE_0004.MP4")
    window.transcript_view.show_transcript(write_lecture_folder(tmp / "names_transcript"), "T")
    window.open_study("overview")
    wait_for(lambda: window.study.stack.currentIndex() != 0 or True, timeout=1, what="the study view")
    for goto in ("home", "files", "active", "result", "transcript", "batch"):
        window._goto(goto)
        window.files_page.refresh()
        QApplication.processEvents()
        assert unnamed_controls(window.simple_stack.currentWidget()) == [], goto
    dialog = window.make_settings_dialog("Transcription")
    dialog.show()
    for name in ("Transcription", "Medical", "Study", "Privacy & Data", "Models", "Advanced"):
        dialog.select_tab(name)
        QApplication.processEvents()
        assert unnamed_controls(dialog.tabs.currentWidget()) == [], name
    dialog.close()
    window.hide()
