"""The headless translation (MELETRA.exe --translate-lecture) and the diagnostic hook that drives the real Translate window (--autorun-translate)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.gui.app import AutoRunTranslate
from src.gui.translate_dialog import TranslateDialog
from src.multilingual import source as source_mod
from src.multilingual.source import SourceSegment, SourceTranscript
from src.translation import cli
from src.translation.service import TranslationService
from tests.gui_support import ScriptedLauncher, make_services, settle, wait_for


class Engine:
    id = "fake:1"

    def translate(self, texts, source, target):
        return [f"[{target}] {t}" for t in texts]


def lecture(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    rows = [SourceSegment(1, 0, 5, "오늘은 신경계에 대해 이야기하겠습니다.", detected_language="ko", language_source="detector"),
            SourceSegment(2, 6, 11, "The cerebellum coordinates movement.", detected_language="en", language_source="detector")]
    source_mod.write(folder, SourceTranscript(lecture_id=folder.name, segments=rows))
    (folder / "metadata.json").write_text(json.dumps({"language": {"detected": "ko"}}), encoding="utf-8")
    return folder


def test_the_command_reports_a_missing_pack_without_changing_anything(tmp_path, monkeypatch):
    monkeypatch.setenv("MLT_DATA_DIR", str(tmp_path / "data"))
    folder = lecture(tmp_path / "lecture")
    out = tmp_path / "result.json"
    assert cli.translate_lecture_command(str(folder), "tr", str(out)) == 2
    report = json.loads(out.read_bytes().decode("utf-8"))
    assert report["availability"] == "model_missing" and not report["ok"] and not (folder / "translations").exists()


def test_the_command_translates_the_saved_transcript_and_a_second_run_uses_the_saved_one(tmp_path, monkeypatch):
    monkeypatch.setenv("MLT_DATA_DIR", str(tmp_path / "data"))
    real = TranslationService
    monkeypatch.setattr("src.translation.service.TranslationService", lambda dirs, **k: real(dirs, engine=Engine(), **k))
    folder = lecture(tmp_path / "lecture")
    out = tmp_path / "result.json"
    assert cli.translate_lecture_command(str(folder), "tr", str(out)) == 0
    first = json.loads(out.read_bytes().decode("utf-8"))
    assert first["ok"] and not first["from_cache"] and first["status"] == "complete" and "translated_tr_transcript.txt" in first["files"]
    assert first["speech_recogniser_loaded"] is False                                                           # the speech model was never imported, let alone used
    assert cli.translate_lecture_command(str(folder), "tr", str(out)) == 0
    second = json.loads(out.read_bytes().decode("utf-8"))
    assert second["from_cache"] and second["created_at"] == first["created_at"]
    assert cli.translate_lecture_command(str(folder), "xx", str(out)) == 2


@pytest.fixture
def window(qapp, tmp_path):
    from src.gui.main_window import MainWindow

    services = make_services(tmp_path, opened=[])
    win = MainWindow(services, ScriptedLauncher())
    win.make_translate_dialog = lambda f: TranslateDialog(services, f, win, service=TranslationService([], engine=Engine()))
    yield win, services
    win.controller._timer.stop()
    settle(win, win.home, win.study)
    win.deleteLater()


def test_the_diagnostic_hook_presses_the_real_button_twice_and_reports_no_job_and_a_saved_translation(window, tmp_path):
    win, services = window
    folder = lecture(tmp_path / "lecture")
    result = tmp_path / "autorun.json"
    exits = []
    import src.gui.app as app_module

    original = app_module.QApplication.exit
    app_module.QApplication.exit = staticmethod(lambda code=0: exits.append(code))
    try:
        runner = AutoRunTranslate(win, folder, "tr", result, 60)
        wait_for(lambda: result.exists(), timeout=60, what="the hook's report")
    finally:
        app_module.QApplication.exit = original
    report = json.loads(result.read_bytes().decode("utf-8"))
    assert report["exit_code"] == 0 and report["state"] == "completed" and report["button"] == {"text": "Translate to another language", "visible": True, "enabled": True}
    assert report["reopened_saved_translation"] and report["jobs_created"] == 0 and not report["job_running"]
    assert report["tabs"] == ["original", "tr"] and report["transcript_language"] == "tr" and "[tr]" in report["transcript_head"]
    assert report["stage2_dialog"]["button"] == "Open translation"
    del runner
