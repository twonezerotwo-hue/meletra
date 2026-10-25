"""The offline translation pack from the user's side: Install translation pack in the Translate window, the row in Settings -> Models, and refusing a language the runtime cannot do."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.gui.settings_dialog import SettingsDialog
from src.gui.translate_dialog import TranslateDialog
from src.multilingual import source as source_mod
from src.multilingual.source import SourceSegment, SourceTranscript
from src.services.model_service import TRANSLATION_PACK_KEY, ModelCancelled, ModelDownloadError, ModelService
from src.services.paths import AppPaths
from src.translation.m2m100 import MODEL_FOLDER, REQUIRED_FILES
from src.translation.service import TranslationService
from tests.gui_support import make_services, settle, wait_for


def lecture(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    rows = [SourceSegment(1, 0, 5, "오늘은 신경계에 대해 이야기하겠습니다.", detected_language="ko", language_source="detector"),
            SourceSegment(2, 6, 11, "The cerebellum coordinates movement.", detected_language="en", language_source="detector")]
    source_mod.write(folder, SourceTranscript(lecture_id=folder.name, segments=rows))
    return folder


class FakeModels:
    """Stands in for ModelService.download: puts the four files of the pack into the models folder, or fails / is cancelled."""

    def __init__(self, models_dir: Path, outcome="ok"):
        self.models_dir, self.outcome, self.calls = models_dir, outcome, []

    def search_dirs(self):
        return [self.models_dir]

    def download(self, key, progress=None, cancel=None):
        self.calls.append(key)
        if progress:
            progress({"type": "model_download_progress", "bytes_done": 10, "bytes_total": 100, "percent": 10.0})
        if self.outcome == "fail":
            raise ModelDownloadError("The download server refused model.bin (HTTP 404).")
        if self.outcome == "cancel":
            raise ModelCancelled("Download cancelled. Your progress is kept.")
        folder = self.models_dir / MODEL_FOLDER
        folder.mkdir(parents=True, exist_ok=True)
        for name in (*REQUIRED_FILES, "shared_vocabulary.json"):
            (folder / name).write_bytes(b"x")
        return folder


class Engine:
    id = "fake:1"

    def translate(self, texts, source, target):
        return [f"[{target}] {t}" for t in texts]


@pytest.fixture
def env(qapp, tmp_path):
    services = make_services(tmp_path, opened=[])
    yield services, tmp_path


def dialog(services, folder, models):
    services.models = models
    return TranslateDialog(services, folder, None)


def test_install_translation_pack_downloads_it_then_translation_becomes_possible(env):
    services, tmp = env
    folder = lecture(tmp / "lecture")
    models = FakeModels(tmp / "models")
    d = dialog(services, folder, models)
    assert d.install_button.isVisibleTo(d) and not d.go_button.isEnabled() and "not installed" in d.status.text()
    d.install_button.click()
    wait_for(lambda: not d._busy and d.go_button.isEnabled(), timeout=30, what="the install")
    assert models.calls == [TRANSLATION_PACK_KEY] and not d.install_button.isVisibleTo(d) and (tmp / "models" / MODEL_FOLDER / "model.bin").is_file()
    assert TranslationService([tmp / "models"]).availability(folder, "tr").state == "ready"                      # the same folder is now a usable pack


def test_a_failed_download_is_explained_and_can_be_tried_again(env):
    services, tmp = env
    folder = lecture(tmp / "lecture")
    models = FakeModels(tmp / "models", outcome="fail")
    d = dialog(services, folder, models)
    d.install_button.click()
    wait_for(lambda: not d._busy, timeout=30, what="the failure")
    assert "404" in d.status.text() and d.install_button.isVisibleTo(d) and not d.go_button.isEnabled()
    models.outcome = "ok"
    d.install_button.click()
    wait_for(lambda: not d._busy and d.go_button.isEnabled(), timeout=30, what="the second try")


def test_cancelling_the_download_keeps_everything_as_it_was(env):
    services, tmp = env
    folder = lecture(tmp / "lecture")
    d = dialog(services, folder, FakeModels(tmp / "models", outcome="cancel"))
    d.install_button.click()
    wait_for(lambda: not d._busy, timeout=30, what="the cancel")
    assert "ancel" in d.status.text() and d.install_button.isVisibleTo(d) and not (tmp / "models" / MODEL_FOLDER).exists()


def test_a_language_pair_the_runtime_cannot_do_is_not_pretended(env):
    """An unsupported target never reaches the engine and an unsupported source language is kept, marked, never silently dropped."""
    services, tmp = env
    folder = lecture(tmp / "lecture")
    service = TranslationService(engine=Engine())
    assert service.availability(folder, "xx").state == "unsupported_target"
    rows = source_mod.read_file(folder)
    rows.segments.append(SourceSegment(3, 12, 17, "பிறகு பார்க்கலாம்", detected_language="ta", language_source="detector"))
    source_mod.write(folder, rows)
    doc = service.translate(folder, "en")
    by_id = {s.segment_id: s for s in doc.segments}
    assert by_id[3].translation_status == "untranslated" and by_id[3].text == "" and "does not read ta" in by_id[3].error
    assert any("[not translated - ta]" in line for line in doc.display_lines())


def test_settings_models_lists_the_translation_pack_next_to_the_speech_model(qapp, tmp_path):
    services = make_services(tmp_path)
    services.models = ModelService(AppPaths(tmp_path / "data").ensure(), app_dir=tmp_path / "app")                # the shipped registry
    d = SettingsDialog(services)
    try:
        titles = [d.model_table.item(row, 0).text() for row in range(d.model_table.rowCount())]
        assert any("Whisper Large-v3" in t for t in titles) and any(t.startswith("Offline translation pack") for t in titles)
        row = next(r for r in range(d.model_table.rowCount()) if d.model_table.item(r, 0).text().startswith("Offline translation pack"))
        assert d.model_table.item(row, 1).text() == "Not installed" and "GB download" in d.model_table.item(row, 2).text() or "MB" in d.model_table.item(row, 2).text() or "GB" in d.model_table.item(row, 2).text()
        d.model_table.selectRow(row)
        assert d.download_button.isEnabled() and "Optional" in d.model_note.text()
    finally:
        settle(d)
        d.deleteLater()


def test_the_speech_model_choice_never_offers_the_translation_pack(qapp, tmp_path):
    services = make_services(tmp_path)
    services.models = ModelService(AppPaths(tmp_path / "data").ensure(), app_dir=tmp_path / "app")
    keys = [s.key for s in services.models.list_models()]
    assert TRANSLATION_PACK_KEY not in keys and "whisper-large-v3" in keys
