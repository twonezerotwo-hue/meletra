"""Translate to another language: the button on the result screen, the window it opens, the tabs for saved translations, the archive and the study picker.
Nothing here may ask for a file, start a transcription job or touch the speech recogniser."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PySide6.QtWidgets import QDialog

from src.gui.main_window import SIMPLE_RESULT, SIMPLE_STUDY, MainWindow
from src.gui.translate_dialog import TranslateDialog
from src.multilingual import source as source_mod
from src.multilingual.source import SourceSegment, SourceTranscript
from src.translation.service import TranslationService
from tests.gui_support import ScriptedLauncher, make_services, settle, wait_for


class Engine:
    id = "fake:1"

    def __init__(self, fail=False, hold=None):
        self.calls = []
        self.fail = fail
        self.hold = hold

    def translate(self, texts, source, target):
        self.calls.append((source, target, list(texts)))
        if self.hold is not None:
            self.hold()
        if self.fail:
            raise RuntimeError("the engine broke")
        return [f"[{target}] {t}" for t in texts]


SENTENCES = [("ko", "오늘은 신경계에 대해 이야기하겠습니다."), ("hi", "आज हम तंत्रिका तंत्र के बारे में बात करेंगे।"), ("en", "The cerebellum coordinates movement."), ("ko", "감사합니다.")]


def make_lecture(folder: Path, languages=None) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    rows = [SourceSegment(i + 1, i * 6.0, i * 6.0 + 5.0, text, detected_language=lang, language_source="detector") for i, (lang, text) in enumerate(SENTENCES)]
    source_mod.write(folder, SourceTranscript(lecture_id=folder.name, segments=rows))
    (folder / "local_transcript.json").write_text(json.dumps({"language": "ko", "segments": [{"start": s.start, "end": s.end, "text": s.source_text, "language": s.detected_language} for s in rows]},
                                                             ensure_ascii=False), encoding="utf-8")
    (folder / "local_raw_transcript.txt").write_text("\n".join(t for _l, t in SENTENCES) + "\n", encoding="utf-8")
    (folder / "local_clean_transcript.txt").write_text("\n".join(t for _l, t in SENTENCES) + "\n", encoding="utf-8")
    (folder / "metadata.json").write_text(json.dumps({"language": {"detected": "ko"}, "translation": {"method": "none", "output_language": "", "files": []}}), encoding="utf-8")
    return folder


@pytest.fixture
def env(qapp, tmp_path):
    services = make_services(tmp_path, opened=[])
    window = MainWindow(services, ScriptedLauncher())
    window.controller._timer.setInterval(20)
    yield window, services, tmp_path
    window.controller._timer.stop()
    settle(window, window.home, window.study)
    window.deleteLater()


def dialog_for(services, folder, engine, **kw) -> TranslateDialog:
    return TranslateDialog(services, folder, None, service=TranslationService([], engine=engine), **kw)


# ------------------------------------------------------------------------------------------------------------------ the result screen
def test_the_result_screen_offers_translation_of_the_saved_transcript_without_a_file_or_a_job(env):
    window, services, tmp = env
    folder = make_lecture(tmp / "lecture")
    window.show_finished(folder, "Pharmacology")
    result = window.simple_result
    assert result.translate_button.text() == "Translate to another language" and result.translate_button.isEnabled() and result.new_button.text() == "Transcribe another lecture"
    assert "Korean · Hindi · English" in result.info_label.text()                                                    # every recognised language is named, not just the first
    opened = []
    window.make_translate_dialog = lambda f: opened.append(f) or type("D", (), {"translated": type("S", (), {"connect": lambda self, fn: None})(), "exec": lambda self: 0})()
    asked = []
    window.choose_file = lambda *a, **k: asked.append(a) or ""
    jobs_before = len(services.jobs.list())
    result.translate_button.click()
    assert opened == [folder] and asked == [] and not window.controller.is_running and len(services.jobs.list()) == jobs_before


def test_a_document_has_no_translate_button(env):
    window, _services, tmp = env
    folder = tmp / "doc"
    folder.mkdir()
    text = folder / "doc.txt"
    text.write_text("some text", encoding="utf-8")
    (folder / "doc.json").write_text(json.dumps({"label": "text", "pages": [1]}), encoding="utf-8")
    window.simple_result.show_results(folder, "doc", None, text_file=text)
    assert not window.simple_result.translate_button.isVisibleTo(window.simple_result)


# ------------------------------------------------------------------------------------------------------------------ the window
def test_the_window_translates_the_saved_transcript_and_reports_progress(env):
    window, services, tmp = env
    folder = make_lecture(tmp / "lecture")
    engine = Engine()
    dialog = dialog_for(services, folder, engine)
    assert dialog.windowTitle() == "Translate to another language" and dialog.combo.accessibleName() == "Target Language"
    assert [dialog.combo.itemData(i) for i in range(dialog.combo.count())] == ["en", "tr", "de", "hu", "ko", "zh", "hi", "nl", "fr", "es"]
    assert dialog.target == "en" and "Korean · Hindi · English" in dialog.about.text()                            # English is the default
    seen = []
    dialog.translated.connect(seen.append)
    dialog.combo.setCurrentIndex(dialog.combo.findData("de"))
    dialog.go_button.click()
    wait_for(lambda: seen, timeout=30, what="the translation")
    assert seen == ["de"] and dialog.result_doc.complete and (tmp / "lecture" / "translations" / "de.json").is_file()
    assert {(s, t) for s, t, _x in engine.calls} == {("ko", "de"), ("hi", "de"), ("en", "de")}                       # direct, from each segment's own language
    assert dialog.result() == QDialog.Accepted


def test_a_saved_translation_opens_at_once_and_is_marked_in_the_list(env):
    window, services, tmp = env
    folder = make_lecture(tmp / "lecture")
    TranslationService(engine=Engine()).translate(folder, "en")
    engine = Engine()
    dialog = dialog_for(services, folder, engine)
    assert dialog.combo.itemText(dialog.combo.findData("en")) == "English (saved)" and dialog.target == "tr"      # the next language that is not there yet
    dialog.combo.setCurrentIndex(dialog.combo.findData("en"))
    assert dialog.go_button.text() == "Open translation" and "already saved" in dialog.status.text()
    seen = []
    dialog.translated.connect(seen.append)
    dialog.go_button.click()
    assert seen == ["en"] and engine.calls == []                                                                  # nothing was translated again


def test_a_missing_pack_is_explained_and_offers_the_download(env):
    window, services, tmp = env
    folder = make_lecture(tmp / "lecture")
    dialog = TranslateDialog(services, folder, None, service=TranslationService([tmp / "no_models"]))
    assert not dialog.go_button.isEnabled() and dialog.install_button.isVisibleTo(dialog)
    assert "not installed" in dialog.status.text() and "0.5 GB" in dialog.status.text() and "Your transcript is not affected" in dialog.status.text()


def test_a_failing_engine_keeps_the_original_marks_the_missing_parts_and_retries_only_those(env):
    window, services, tmp = env
    folder = make_lecture(tmp / "lecture")
    dialog = dialog_for(services, folder, Engine(fail=True))
    seen = []
    dialog.translated.connect(seen.append)
    dialog.go_button.click()
    wait_for(lambda: seen, timeout=30, what="the partial translation")
    doc = json.loads((folder / "translations" / "en.json").read_bytes().decode("utf-8"))
    assert doc["status"] == "partial" and doc["counts"].get("failed", 0) >= 1                                       # the failed parts are marked, the same-language part is kept
    assert (folder / "local_raw_transcript.txt").read_text(encoding="utf-8").startswith("오늘은")                    # the original is untouched
    fixed = Engine()
    again = dialog_for(services, folder, fixed, preselect="en")
    assert again.go_button.text() == "Translate the missing parts" and "could not be translated" in again.status.text()
    done = []
    again.translated.connect(done.append)
    again.go_button.click()
    wait_for(lambda: done, timeout=30, what="the retry")
    assert {(s, t) for s, t, _x in fixed.calls} == {("ko", "en"), ("hi", "en")}                                       # English was spoken in English: not sent again; only the failed languages were
    assert json.loads((folder / "translations" / "en.json").read_bytes().decode("utf-8"))["status"] == "complete"


def test_cancel_stops_the_translation_and_stores_nothing(env):
    window, services, tmp = env
    folder = make_lecture(tmp / "lecture")
    holder = {}
    engine = Engine(hold=lambda: holder["dialog"].cancel())
    dialog = dialog_for(services, folder, engine)
    holder["dialog"] = dialog
    # the second batch (another language) is where the cancel request is noticed
    dialog.go_button.click()
    wait_for(lambda: not dialog._busy, timeout=30, what="the end of the task")
    assert not (folder / "translations" / "en.json").exists() or json.loads((folder / "translations" / "en.json").read_bytes().decode("utf-8"))["counts"]
    assert (folder / "source_transcript.json").is_file()


# ------------------------------------------------------------------------------------------------------------------ the transcript view
def test_the_transcript_view_has_a_tab_for_every_saved_translation(env):
    window, _services, tmp = env
    folder = make_lecture(tmp / "lecture")
    service = TranslationService(engine=Engine())
    service.translate(folder, "en")
    service.translate(folder, "de")
    window.show_finished(folder, "Neurology")
    window.open_transcript()
    view = window.transcript_view
    assert list(view.mode.buttons) == ["original", "en", "de"] and view.mode.isVisibleTo(view)
    assert view.mode.buttons["original"].text() == "Original · Korean · Hindi · English" and view.mode.buttons["en"].text() == "English" and view.mode.buttons["de"].text() == "German"
    view.mode.buttons["de"].click()
    assert "[de]" in view.text.toPlainText() and not view.view_combo.isVisibleTo(view)
    view.mode.buttons["original"].click()
    assert "신경계" in view.text.toPlainText() and "[de]" not in view.text.toPlainText() and view.view_combo.isVisibleTo(view)


def test_a_translation_made_from_an_older_transcript_is_not_shown(env):
    window, _services, tmp = env
    folder = make_lecture(tmp / "lecture")
    TranslationService(engine=Engine()).translate(folder, "en")
    source = source_mod.read_file(folder)
    source.segments[0].source_text = "오늘은 소뇌에 대해 이야기하겠습니다."
    source_mod.write(folder, source)
    window.show_finished(folder, "Neurology")
    window.open_transcript()
    assert list(window.transcript_view.mode.buttons) == ["original"] and not window.transcript_view.mode.isVisibleTo(window.transcript_view)


def test_a_translation_made_by_version_4_1_0_is_shown_as_it_is(env):
    window, _services, tmp = env
    folder = make_lecture(tmp / "old")
    (folder / "translated_en_transcript.txt").write_text("Old English text.\n", encoding="utf-8")
    window.show_finished(folder, "Old lecture")
    window.open_transcript()
    view = window.transcript_view
    assert list(view.mode.buttons) == ["original", "en"]
    view.mode.buttons["en"].click()
    assert "Old English text." in view.text.toPlainText()


def test_after_a_translation_the_main_window_opens_its_tab(env):
    window, _services, tmp = env
    folder = make_lecture(tmp / "lecture")
    window.show_finished(folder, "Neurology")
    TranslationService(engine=Engine()).translate(folder, "tr")
    window.show_translation("tr")
    assert window.transcript_view.language == "tr" and "[tr]" in window.transcript_view.text.toPlainText()


# ------------------------------------------------------------------------------------------------------------------ study
def test_a_lecture_in_several_languages_asks_for_a_translation_before_study_and_then_offers_the_languages(env):
    window, services, tmp = env
    folder = make_lecture(tmp / "lecture")
    wait = window.study
    wait.open_lecture(folder, "Neurology")
    assert wait.stack.currentIndex() == 2 and wait.message_translate_button.isVisibleTo(wait) and "Korean" in wait.message_text.text()
    asked = []
    wait.translate_requested.connect(lambda: asked.append(True))
    wait.message_translate_button.click()
    assert asked == [True]
    service = TranslationService(engine=Engine())
    service.translate(folder, "en")
    service.translate(folder, "de")
    wait.open_lecture(folder, "Neurology")
    wait_for(lambda: wait.stack.currentIndex() in (1, 2), timeout=60, what="the study material")
    assert wait.language_row.isVisibleTo(wait) and [wait.language_combo.itemData(i) for i in range(wait.language_combo.count())] == ["en", "de"]
    wait.language_combo.setCurrentIndex(1)
    wait._language_chosen(1)
    wait_for(lambda: wait.stack.currentIndex() in (1, 2), timeout=60, what="the German study material")
    assert services.study.study_language(folder) == "de" and (folder / "study" / "de").is_dir()


# ------------------------------------------------------------------------------------------------------------------ the archive
def test_the_archive_card_names_every_spoken_language_and_every_saved_translation(env):
    from src.gui.lecture_cards import language_path
    from src.services.job_service import JobStore

    window, services, tmp = env
    folder = make_lecture(tmp / "lecture")
    store = services.jobs
    store.create_or_get("job1", tmp / "a.mp4", output_dir=str(folder), state="running", language="auto", output_language="en")
    store.apply_event("job1", {"type": "completed", "language": "ko", "languages": ["ko", "hi", "en"], "translations": ["en"], "audio_seconds": 60.0, "output_dir": str(folder),
                               "translation": {"method": "m2m100"}})
    record = store.get("job1")
    assert language_path(record) == "Auto (Korean · Hindi · English) → English"
    service = TranslationService(engine=Engine())
    service.translate(folder, "en")
    service.translate(folder, "tr")
    window.show_finished(folder, "Neurology")
    window.show_translation("tr")                                                                                 # as after the window: the archive learns about the new translation
    assert store.get("job1").translations == ["en", "tr"]
    assert language_path(store.get("job1")) == "Auto (Korean · Hindi · English) → English, Turkish"
    assert language_path(JobStore(services.paths.jobs).get("job1")) == "Auto (Korean · Hindi · English) → English, Turkish"       # still there after a restart


def test_an_old_record_without_the_new_fields_reads_as_before(env):
    from src.gui.lecture_cards import language_path
    from src.services.job_service import JobRecord

    record = JobRecord.from_dict({"job_id": "x", "source": "a.mp4", "language": "auto", "detected_language": "tr", "output_language": "en", "translation": "whisper"})
    assert record.detected_languages == [] and record.translations == [] and language_path(record) == "Auto (Turkish) → English"


# ------------------------------------------------------------------------------------------------------------------ lectures made by 4.1.0
def test_a_lecture_from_4_1_0_is_translated_from_its_saved_text_and_offers_to_read_the_recording_again(env, tmp_path):
    window, services, tmp = env
    folder = tmp / "old"
    folder.mkdir()
    (folder / "local_transcript.json").write_text(json.dumps({"language": "en", "segments": [{"start": 0, "end": 3, "text": "The heart pumps blood."}]}), encoding="utf-8")
    (folder / "local_raw_transcript.txt").write_text("The heart pumps blood.\n", encoding="utf-8")
    (folder / "metadata.json").write_text(json.dumps({"language": {"detected": "en", "requested": "auto"}}), encoding="utf-8")
    recording = tmp / "recording.mp4"
    recording.write_bytes(b"x")
    services.jobs.create_or_get("old1", recording, output_dir=str(folder), state="completed", language="auto")
    window.show_finished(folder, "Old lecture")
    result = window.simple_result
    assert result.legacy_row.isVisibleTo(result) and result.reprocess_button.text() == "Reprocess Original Audio" and result.translate_button.isEnabled()
    dialog = dialog_for(services, folder, Engine())
    seen = []
    dialog.translated.connect(seen.append)
    dialog.combo.setCurrentIndex(dialog.combo.findData("de"))
    dialog.go_button.click()
    wait_for(lambda: seen, timeout=30, what="the translation")
    doc = TranslationService(engine=Engine()).cached(folder, "de")
    assert doc is not None and any("earlier version" in w for w in doc.warnings) and any("existing English transcript" in w for w in doc.warnings)
    chosen = []
    window.simple_home.set_files = chosen.append
    window.reprocess_original()                                                                                      # a separate action: it goes to the start screen with the recording chosen
    assert window.simple_stack.currentIndex() == 0 and chosen == [[str(recording)]]


def test_reprocessing_without_the_recording_says_so_and_changes_nothing(env):
    window, services, tmp = env
    folder = make_lecture(tmp / "lecture")
    services.jobs.create_or_get("gone", tmp / "missing.mp4", output_dir=str(folder), state="completed")
    window.show_finished(folder, "Lecture")
    window.reprocess_original()
    assert window.simple_stack.currentIndex() == SIMPLE_RESULT and (folder / "local_transcript.json").is_file()


def test_a_new_lecture_does_not_offer_reprocessing(env):
    window, _services, tmp = env
    folder = make_lecture(tmp / "lecture")
    window.show_finished(folder, "Lecture")
    assert not window.simple_result.legacy_row.isVisibleTo(window.simple_result)


def test_after_a_translation_the_student_is_told_that_study_material_can_be_made_in_that_language(env):
    window, _services, tmp = env
    folder = make_lecture(tmp / "lecture")
    window.show_finished(folder, "Neurology")
    TranslationService(engine=Engine()).translate(folder, "de")
    shown = []
    window.toast.show_message = shown.append
    window.show_translation("de")
    assert shown and "Study material can be made in German" in shown[0]
