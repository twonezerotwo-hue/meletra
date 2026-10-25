"""Settings dialog: six tabs, saving, model manager, Deepgram key in the credential store, diagnostics."""

import json
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from src.gui.settings_dialog import TAB_NAMES, SettingsDialog
from src.services.model_service import ModelError
from tests.gui_support import MODEL_KEY, fake_files, install_fake_model, make_gpu, make_services, settle, wait_for


@pytest.fixture
def env(qapp, tmp_path):
    services = make_services(tmp_path)
    dialog = SettingsDialog(services)
    dialog.confirm = lambda title, text: True
    yield type("Env", (), {"services": services, "dialog": dialog, "tmp": tmp_path})
    settle(dialog)
    dialog.deleteLater()


def test_the_six_tabs_exist_in_order(env):
    assert TAB_NAMES == ["General", "Models", "Performance", "Medical", "Cloud", "Diagnostics"]
    assert [env.dialog.tabs.tabText(i) for i in range(env.dialog.tabs.count())] == TAB_NAMES


def test_initial_tab_can_be_chosen(qapp, tmp_path):
    dialog = SettingsDialog(make_services(tmp_path), initial_tab="Cloud")
    assert dialog.tabs.currentIndex() == TAB_NAMES.index("Cloud")


def test_saving_writes_every_changed_value(env):
    d = env.dialog
    d.output_edit.setText(str(env.tmp / "transcripts"))
    d.theme_combo.setCurrentIndex(d.theme_combo.findData("dark"))
    d.open_folder_check.setChecked(False)
    d.language_combo.setCurrentIndex(d.language_combo.findData("tr"))
    d.mode_combo.setCurrentIndex(d.mode_combo.findData("fast"))
    d.device_combo.setCurrentIndex(d.device_combo.findData("cpu"))
    d.compute_combo.setCurrentIndex(d.compute_combo.findData("int8"))
    d.threads_spin.setValue(6)
    d.beam_spin.setValue(3)
    d.vad_check.setChecked(False)
    d.chunk_spin.setValue(10.0)
    d.overlap_spin.setValue(2.0)
    d.hotwords_combo.setCurrentIndex(d.hotwords_combo.findData("off"))
    d.pdf_check.setChecked(False)
    d.format_checks["vtt"].setChecked(False)
    d.save()
    s = env.services.settings.settings
    assert (s.output_dir, s.theme, s.open_folder_after, s.language, s.mode) == (str(env.tmp / "transcripts"), "dark", False, "tr", "fast")
    assert (s.device, s.compute_type, s.cpu_threads, s.beam_size, s.vad) == ("cpu", "int8", 6, 3, False)
    assert (s.chunk_minutes, s.overlap_seconds, s.hotwords, s.pdf_context) == (10.0, 2.0, "off", False)
    assert s.output_formats == ["txt", "srt", "json"]
    stored = json.loads(env.services.paths.settings_file.read_text(encoding="utf-8"))
    assert stored["device"] == "cpu" and stored["chunk_minutes"] == 10.0


def test_saving_emits_settings_changed_and_closes(env):
    got = []
    env.dialog.settings_changed.connect(lambda: got.append(1))
    env.dialog.save()
    assert got == [1] and env.dialog.result() == 1


def test_unchecking_every_format_falls_back_to_all_of_them(env):
    for box in env.dialog.format_checks.values():
        box.setChecked(False)
    env.dialog.save()
    assert env.services.settings.settings.output_formats == ["txt", "srt", "vtt", "json"]


def test_unload_ollama_is_off_by_default_and_labelled_advanced(env):
    assert not env.dialog.ollama_check.isChecked() and "Advanced" in env.dialog.ollama_check.text()


def test_browse_output_uses_the_chosen_folder(env):
    env.dialog.choose_folder = lambda title, start: str(env.tmp / "picked")
    env.dialog._browse_output()
    assert env.dialog.output_edit.text() == str(env.tmp / "picked")


def test_performance_defaults_come_from_the_settings(env):
    d = env.dialog
    assert d.device_combo.currentData() == "auto" and d.compute_combo.currentData() == "auto" and d.vad_check.isChecked()
    assert d.chunk_spin.value() == 15.0 and d.overlap_spin.value() == 3.0


def test_medical_tab_reports_the_vocabulary(env):
    assert "medical terms" in env.dialog.vocab_label.text() and "biochemistry" in env.dialog.vocab_label.text()


# ----------------------------------------------------------------------------------------------- models tab
def test_model_table_shows_status_size_and_location(env):
    d = env.dialog
    assert d.model_table.rowCount() == 1
    texts = [d.model_table.item(0, c).text() for c in range(5)]
    assert texts[0].startswith("Whisper Large-v3") and "Recommended" in texts[0] and texts[1] == "Installed"
    assert texts[4].endswith("whisper-large-v3-folder") and "free on this drive" in d.disk_label.text()
    assert d.verify_button.isEnabled() and not d.download_button.isEnabled() and d.delete_button.isEnabled()


def test_model_not_installed_offers_a_download(qapp, tmp_path):
    dialog = SettingsDialog(make_services(tmp_path, installed=False))
    assert dialog.model_table.item(0, 1).text() == "Not installed" and "GB download" in dialog.model_table.item(0, 2).text()
    assert dialog.download_button.isEnabled() and not dialog.verify_button.isEnabled() and not dialog.delete_button.isEnabled()
    assert dialog.download_button.text() == "Download"


def test_download_button_downloads_through_the_service(qapp, tmp_path, monkeypatch):
    services = make_services(tmp_path, installed=False)
    dialog = SettingsDialog(services)
    done = []
    dialog.models_changed.connect(lambda: done.append(1))

    def fake_download(key, progress=None, cancel=None):
        progress({"type": "model_download_progress", "percent": 40.0, "bytes_done": 4, "bytes_total": 10, "eta_seconds": 120})
        return install_fake_model(services.models, key)

    monkeypatch.setattr(services.models, "download", fake_download)
    dialog.download_button.click()
    wait_for(lambda: services.models.is_installed(MODEL_KEY) and done, what="the download")
    assert dialog.model_table.item(0, 1).text() == "Installed"
    settle(dialog)


def test_download_errors_are_shown_not_raised(qapp, tmp_path, monkeypatch):
    services = make_services(tmp_path, installed=False)
    dialog = SettingsDialog(services)

    def broken(key, progress=None, cancel=None):
        raise ModelError("There is not enough free disk space.")

    monkeypatch.setattr(services.models, "download", broken)
    dialog.download_button.click()
    wait_for(lambda: "not enough free disk space" in dialog.model_progress_text.text(), what="the error text")
    settle(dialog)


def test_verify_reports_a_pass(env):
    env.dialog.verify_button.click()
    wait_for(lambda: "Verification passed" in env.dialog.model_progress_text.text(), what="verification")


def test_verify_reports_a_damaged_model(env):
    folder = env.services.models.locate(MODEL_KEY)
    data = bytearray((folder / "model.bin").read_bytes())
    data[10] ^= 0xFF
    (folder / "model.bin").write_bytes(bytes(data))
    env.dialog.verify_button.click()
    wait_for(lambda: "FAILED" in env.dialog.model_progress_text.text(), what="verification failure")


def test_delete_asks_and_removes_the_model(env):
    asked = []
    env.dialog.confirm = lambda title, text: (asked.append(text), False)[1]
    env.dialog.delete_button.click()
    assert env.services.models.is_installed(MODEL_KEY) and "Delete" in asked[0]
    env.dialog.confirm = lambda title, text: True
    env.dialog.delete_button.click()
    assert not env.services.models.is_installed(MODEL_KEY) and env.dialog.model_table.item(0, 1).text() == "Not installed"


def test_import_model_folder_copies_and_verifies(qapp, tmp_path):
    services = make_services(tmp_path, installed=False)
    source = tmp_path / "usb" / "large-v3"
    source.mkdir(parents=True)
    for name, data in fake_files().items():
        (source / name).write_bytes(data)
    dialog = SettingsDialog(services)
    dialog.choose_folder = lambda title, start: str(source)
    dialog.import_button.click()
    wait_for(lambda: services.models.is_installed(MODEL_KEY), what="the import")
    wait_for(lambda: "imported and verified" in dialog.model_progress_text.text(), what="the import message")
    settle(dialog)


def test_import_of_a_wrong_folder_is_reported(qapp, tmp_path):
    services = make_services(tmp_path, installed=False)
    wrong = tmp_path / "empty"
    wrong.mkdir()
    dialog = SettingsDialog(services)
    dialog.choose_folder = lambda title, start: str(wrong)
    dialog.import_button.click()
    wait_for(lambda: dialog.model_progress_text.text() and not any(t.isRunning() for t in dialog._threads), what="the error")
    assert not services.models.is_installed(MODEL_KEY) and dialog.model_progress_text.text()
    settle(dialog)


def test_extra_search_folder_is_remembered_and_used(qapp, tmp_path):
    services = make_services(tmp_path, installed=False)
    elsewhere = tmp_path / "models-elsewhere"
    folder = elsewhere / services.models.info(MODEL_KEY).folder
    folder.mkdir(parents=True)
    for name, data in fake_files().items():
        (folder / name).write_bytes(data)
    dialog = SettingsDialog(services)
    dialog.choose_folder = lambda title, start: str(elsewhere)
    dialog.folder_button.click()
    assert services.settings.settings.extra_model_dirs == [str(elsewhere)]
    assert services.models.is_installed(MODEL_KEY) and dialog.model_table.item(0, 1).text() == "Installed"
    dialog.folder_button.click()
    assert services.settings.settings.extra_model_dirs == [str(elsewhere)]            # no duplicates


def test_model_note_explains_the_runtime(env):
    assert "Runtime:" in env.dialog.model_note.text() and "Test model" in env.dialog.model_note.text()


# ----------------------------------------------------------------------------------------------- cloud tab
def test_cloud_tab_explains_local_versus_deepgram(env):
    texts = [w.text() for w in env.dialog.tabs.widget(TAB_NAMES.index("Cloud")).findChildren(type(env.dialog.key_status))]
    joined = " ".join(texts)
    assert "LOCAL" in joined and "No audio leaves your computer" in joined and "uploaded to Deepgram" in joined


def test_deepgram_key_goes_to_the_credential_store_only(env):
    d = env.dialog
    d.key_edit.setText("dg_live_ABCDEFGH12345678")
    d.save_key_button.click()
    assert env.services.credentials.get_deepgram_key() == "dg_live_ABCDEFGH12345678" and d.key_edit.text() == ""
    assert "dg_live_ABCDEFGH12345678" not in d.key_status.text()
    assert "Windows Credential Manager" in d.key_status.text() or "memory only" in d.key_status.text()
    d.save()
    assert "dg_live" not in env.services.paths.settings_file.read_text(encoding="utf-8")
    d.remove_key_button.click()
    assert not env.services.credentials.has_deepgram_key() and "No key stored" in d.key_status.text()


def test_an_empty_key_is_rejected(env):
    env.dialog.key_edit.setText("   ")
    env.dialog.save_key_button.click()
    assert not env.services.credentials.has_deepgram_key() and env.dialog.key_status.text()


def test_the_key_field_hides_what_is_typed(env):
    from PySide6.QtWidgets import QLineEdit

    assert env.dialog.key_edit.echoMode() == QLineEdit.Password


def test_cloud_checkbox_is_saved(env):
    env.dialog.cloud_check.setChecked(True)
    env.dialog.save()
    assert env.services.settings.settings.cloud_enabled is True


# ----------------------------------------------------------------------------------------------- diagnostics tab
@pytest.mark.windows_platform
def test_diagnostics_load_on_first_view_and_can_be_copied(env):
    env.dialog.select_tab("Diagnostics")
    wait_for(lambda: "Windows" in env.dialog.diag_text.toPlainText(), timeout=30, what="the diagnostics text")
    text = env.dialog.diag_text.toPlainText()
    assert "GPU" in text and "RTX 2070" in text and "Data folder" in text and "Whisper Large-v3" in text
    env.dialog.copy_button.click()
    assert QApplication.clipboard().text() == text
    settle(env.dialog)


@pytest.mark.windows_platform
def test_diagnostics_never_contain_the_api_key(env):
    env.services.credentials.set_deepgram_key("dg_live_SECRETSECRET99")
    env.dialog.select_tab("Diagnostics")
    wait_for(lambda: "Windows" in env.dialog.diag_text.toPlainText(), timeout=30, what="the diagnostics text")
    assert "SECRETSECRET" not in env.dialog.diag_text.toPlainText()
    settle(env.dialog)


def test_logs_folder_button_opens_the_folder(qapp, tmp_path):
    opened = []
    dialog = SettingsDialog(make_services(tmp_path, opened=opened))
    dialog.diag_text.parent().findChildren(type(dialog.copy_button))[2].click()
    assert opened == [dialog.services.paths.logs]
