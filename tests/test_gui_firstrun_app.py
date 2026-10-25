"""First-run wizard, About box, theme and icon, launcher options, self-test, autorun hook."""

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

import launcher
from src import APP_NAME, COMPANY, __version__
from src.gui import theme, tokens
from src.gui.about_dialog import AboutDialog
from src.gui.app import AutoRun
from src.gui.first_run import FirstRunDialog, needs_first_run, system_check
from src.gui.job_controller import ThreadLauncher
from src.gui.main_window import PAGE_RESULTS, MainWindow
from src.model_manager import ModelSpec
from src.services import selftest, transcription_service as tsvc
from src.services.model_service import ModelService
from tests.conftest import write_silent_wav
from tests.gui_support import MODEL_KEY, fake_files, install_fake_model, make_gpu, make_services, settle, wait_for
from tests.test_expert_pipeline import LectureFake

ROOT = Path(__file__).resolve().parent.parent


# ----------------------------------------------------------------------------------------------- system check
def rows_by_name(rows):
    return {name: (text, level) for name, text, level in rows}


@pytest.mark.windows_platform
def test_system_check_lists_what_the_spec_asks_for(qapp, tmp_path):
    rows = rows_by_name(system_check(make_services(tmp_path)))
    for needed in ("Windows", "CPU", "Memory", "GPU", "CUDA acceleration", "Disk space", "Audio decoder"):
        assert needed in rows, needed
    assert "RTX 2070" in rows["GPU"][0] and rows["CUDA acceleration"] == ("available", "ok")


@pytest.mark.windows_platform
def test_no_gpu_is_a_note_not_an_error(qapp, tmp_path):
    cpu = make_gpu(smi=None, cuda=(False, "No compatible NVIDIA GPU / driver was detected."))
    rows = rows_by_name(system_check(make_services(tmp_path, gpu=cpu)))
    assert rows["GPU"][1] == "warn" and "CPU will be used" in rows["GPU"][0]
    assert "CPU fallback enabled" in rows["CUDA acceleration"][0] and rows["CUDA acceleration"][1] == "warn"


@pytest.mark.windows_platform
def test_broken_cuda_runtime_names_the_reason(qapp, tmp_path):
    gpu = make_gpu(cuda=(False, "A CUDA runtime library is missing or incompatible (cublas64_12.dll)."))
    rows = rows_by_name(system_check(make_services(tmp_path, gpu=gpu)))
    assert "cublas64_12.dll" in rows["CUDA acceleration"][0] and "CPU fallback enabled" in rows["CUDA acceleration"][0]


def test_low_memory_and_disk_are_flagged(qapp, tmp_path, monkeypatch):
    from src.services import diagnostics_service as dsvc

    monkeypatch.setattr(dsvc, "memory_status", lambda: {"total_gb": 4.0, "free_gb": 1.0})
    monkeypatch.setattr("src.gui.first_run.diagnostics_service.memory_status", lambda: {"total_gb": 4.0, "free_gb": 1.0})
    rows = rows_by_name(system_check(make_services(tmp_path)))
    assert rows["Memory"][1] == "warn" and "8 GB" in rows["Memory"][0]


# ----------------------------------------------------------------------------------------------- wizard
@pytest.fixture
def wizard(qapp, tmp_path):
    services = make_services(tmp_path, installed=False)
    dialog = FirstRunDialog(services)
    yield dialog
    settle(dialog)
    dialog.deleteLater()


@pytest.mark.windows_platform
def test_wizard_shows_the_check_and_offers_the_recommended_download(wizard):
    wait_for(lambda: "[OK" in wizard.check_label.text() or "[NOTE" in wizard.check_label.text(), timeout=30, what="the system check")
    assert "Windows" in wizard.check_label.text() and "GPU" in wizard.check_label.text()
    assert wizard.download_button.text() == "Download recommended model" and not wizard.download_button.isHidden()
    assert "about" in wizard.model_label.text() and "GB" in wizard.model_label.text()
    assert not wizard.skip_button.isHidden() and wizard.done_button.isHidden()


def test_wizard_downloads_with_progress_and_finishes(wizard, monkeypatch):
    services = wizard.services
    progress = []

    def fake_download(key, progress_cb=None, cancel=None):
        for pct in (10.0, 60.0):
            progress_cb({"type": "model_download_progress", "percent": pct, "bytes_done": int(pct), "bytes_total": 100, "eta_seconds": 90})
        return install_fake_model(services.models, key)

    monkeypatch.setattr(services.models, "download", lambda key, p=None, c=None: fake_download(key, p, c))
    wizard.progress.valueChanged.connect(progress.append)
    wait_for(lambda: wizard.rows, timeout=30, what="the system check")
    wizard.download_button.click()
    wait_for(lambda: not wizard.done_button.isHidden(), what="the wizard to finish")
    assert 600 in progress and "installed" in wizard.model_label.text() and wizard.download_button.isHidden()
    wizard.done_button.click()
    assert services.settings.settings.first_run_done is True


def test_wizard_failure_keeps_the_download_button(wizard, monkeypatch):
    def broken(key, progress=None, cancel=None):
        raise RuntimeError("No internet connection.")

    monkeypatch.setattr(wizard.services.models, "download", broken)
    wait_for(lambda: wizard.rows, timeout=30, what="the system check")
    wizard.download_button.click()
    wait_for(lambda: "No internet connection" in wizard.progress_text.text(), what="the error")
    assert wizard.download_button.isEnabled() and not wizard.download_button.isHidden()


def test_wizard_import_reuses_an_existing_model_folder(wizard, tmp_path):
    source = tmp_path / "existing"
    source.mkdir()
    for name, data in fake_files().items():
        (source / name).write_bytes(data)
    wizard.choose_folder = lambda: str(source)
    wait_for(lambda: wizard.rows, timeout=30, what="the system check")
    wizard.import_button.click()
    wait_for(lambda: wizard.services.models.is_installed(MODEL_KEY) and not wizard.done_button.isHidden(), what="the import")


def test_skip_marks_first_run_done(wizard):
    assert needs_first_run(wizard.services)
    wizard.skip_button.click()
    assert wizard.services.settings.settings.first_run_done and not needs_first_run(wizard.services)


def test_closing_the_wizard_counts_as_skip(wizard):
    wizard.reject()
    assert wizard.services.settings.settings.first_run_done


def test_wizard_with_the_model_already_present_goes_straight_to_ready(qapp, tmp_path):
    dialog = FirstRunDialog(make_services(tmp_path, installed=True))
    assert "is installed" in dialog.model_label.text() and not dialog.done_button.isHidden() and dialog.download_button.isHidden()
    settle(dialog)


def test_wizard_resumes_a_partial_download(qapp, tmp_path):
    services = make_services(tmp_path, installed=False)
    partial = services.paths.models / ".partial" / services.models.info(MODEL_KEY).folder
    partial.mkdir(parents=True)
    (partial / "model.bin.part").write_bytes(b"1" * 2048)
    dialog = FirstRunDialog(services)
    assert dialog.download_button.text() == "Resume download" and "partial download" in dialog.model_label.text()
    settle(dialog)


# ----------------------------------------------------------------------------------------------- about / theme / icon
def test_about_box_states_version_owner_and_privacy(qapp):
    from PySide6.QtWidgets import QLabel

    dialog = AboutDialog()
    text = " ".join(label.text() for label in dialog.findChildren(QLabel))
    assert APP_NAME in text and __version__ in text and COMPANY in text and "locally" in text and "MIT" in text


def test_icon_is_drawn_at_every_size(qapp):
    icon = theme.app_icon()
    assert not icon.isNull()
    for size in (16, 32, 256):
        assert not icon.pixmap(size).isNull()
    image = theme.draw_icon(64)
    assert image.width() == 64 and image.pixelColor(32, 32).alpha() > 0 and image.pixelColor(0, 0).alpha() == 0


def test_light_and_dark_stylesheets_differ_and_cover_the_main_widgets():
    light, dark = theme.stylesheet("light"), theme.stylesheet("dark")
    assert light != dark and tokens.LIGHT.bg in light and tokens.DARK.bg in dark
    for selector in ("QPushButton#primary", "QFrame#dropArea", "QProgressBar", "QTabBar::tab", "QMenu", "QScrollBar:vertical", "QToolTip"):
        assert selector in light


def test_theme_setting_is_respected(qapp):
    assert theme.effective_theme("dark") == "dark" and theme.effective_theme("light") == "light"
    assert theme.effective_theme("system") in ("light", "dark")
    assert theme.apply_theme(qapp, "dark") == "dark" and "background" in qapp.styleSheet()
    assert theme.apply_theme(qapp, "light") == "light"


def test_changing_the_theme_in_settings_updates_the_application(qapp, tmp_path):
    services = make_services(tmp_path)
    window = MainWindow(services, None)
    services.update_setting(theme="dark")
    window.on_settings_changed()
    assert tokens.DARK.bg in qapp.styleSheet()
    services.update_setting(theme="light")
    window.on_settings_changed()
    settle(window, window.home)


def test_menus_exist(qapp, tmp_path):
    window = MainWindow(make_services(tmp_path), None)
    titles = [a.text() for a in window.menuBar().actions()]
    assert titles == ["&File", "&Settings", "&Help"]
    opened = []
    window.make_settings_dialog = lambda tab="General": opened.append(tab) or type("D", (), {"exec": lambda self: 0})()
    window.models_action.trigger()
    window.diagnostics_action.trigger()
    window.settings_action.trigger()
    assert opened == ["Models", "Diagnostics", "General"]
    settle(window, window.home)


# ----------------------------------------------------------------------------------------------- launcher + self-test
def test_launcher_option_parser():
    assert launcher._option(["--selftest", "out.json"], "--selftest") == "out.json"
    assert launcher._option(["--selftest", "--other"], "--selftest", "x") == "x"
    assert launcher._option([], "--selftest", "d") == "d"


def test_launcher_worker_without_a_request_is_a_usage_error():
    assert launcher.main(["exe", "--worker"]) == 2


def test_selftest_passes_in_the_development_environment():
    report = selftest.run_selftest()
    names = [c.name for c in report.checks]
    assert names == ["Libraries", "Resources", "Medical Expert", "Medical knowledge", "PyAV decode", "FFmpeg", "GPU / CUDA", "CPU fallback", "Model discovery",
                     "Languages", "Translation pack", "User data folder", "Credential store", "Study engine", "Lecture archive", "Speech engines", "File router",
                     "Text recognition", "Read aloud"]  # 0.2 added "Study engine" and "Speech engines", 3.0 the last three, v4 "Medical knowledge", 4.1.1 "Languages" and "Translation pack", 4.1.2 "Lecture archive"
    failed = [(c.name, c.detail) for c in report.checks if not c.ok]
    assert not failed, failed
    by_name = {c.name: c for c in report.checks}
    assert not by_name["Languages"].skipped and "3 language runs" in by_name["Languages"].detail        # the saved-transcript machinery is checked for real
    assert by_name["Translation pack"].skipped or "translated a sentence" in by_name["Translation pack"].detail      # the optional pack: found and used, or reported as not installed
    assert report.ok and json.loads(report.to_json())["app"]["version"] == __version__


def test_selftest_reports_a_failing_check_without_crashing(monkeypatch):
    monkeypatch.setattr(selftest, "_cpu_fallback", lambda: (_ for _ in ()).throw(RuntimeError("no cpu types")))
    report = selftest.run_selftest()
    bad = [c for c in report.checks if not c.ok]
    assert [c.name for c in bad] == ["CPU fallback"] and "no cpu types" in bad[0].detail and not report.ok


def test_selftest_via_the_command_line_writes_json(tmp_path):
    out = tmp_path / "self.json"
    proc = subprocess.run([sys.executable, str(ROOT / "launcher.py"), "--selftest", str(out)], capture_output=True, text=True,
                          timeout=180, cwd=str(tmp_path), env={**__import__("os").environ, "MLT_DATA_DIR": str(tmp_path / "data"),
                                                                  "PYTHONPATH": str(ROOT)})
    assert proc.returncode == 0, proc.stderr[-800:]
    assert json.loads(out.read_text(encoding="utf-8"))["ok"] is True


@pytest.mark.windows_only
def test_diagnostics_via_the_command_line_writes_text(tmp_path):
    out = tmp_path / "diag.txt"
    proc = subprocess.run([sys.executable, str(ROOT / "launcher.py"), "--diagnostics", str(out)], capture_output=True, text=True,
                          timeout=180, cwd=str(tmp_path), env={**__import__("os").environ, "MLT_DATA_DIR": str(tmp_path / "data"),
                                                                  "PYTHONPATH": str(ROOT)})
    assert proc.returncode == 0, proc.stderr[-800:]
    text = out.read_text(encoding="utf-8")
    assert "Windows" in text and "FFmpeg" in text and "MELETRA" in text


def test_selftest_resources_exist_in_the_source_tree():
    for rel in selftest.RESOURCE_FILES:
        assert (ROOT / rel).exists(), rel


# ----------------------------------------------------------------------------------------------- autorun hook (packaged-build test driver)
@pytest.mark.usefixtures("fake_ffmpeg")
def test_autorun_drives_the_real_window_to_a_result_file(qapp, tmp_path, monkeypatch):
    services = make_services(tmp_path)
    monkeypatch.setattr(ModelService, "model_spec", lambda self, key: ModelSpec(key, "faster_whisper", "large-v3"))
    monkeypatch.setattr(tsvc, "STOP_POLL_SECONDS", 0.0)
    monkeypatch.setattr("src.transcriber.create_adapter", lambda *a, **k: LectureFake())
    services.update_setting(output_dir=str(tmp_path / "out"), chunk_minutes=1.0)
    window = MainWindow(services, ThreadLauncher())
    window.controller._timer.setInterval(20)
    source = write_silent_wav(tmp_path / "lecture.wav", seconds=90)
    result = tmp_path / "result.json"
    monkeypatch.setattr("src.gui.app.QApplication.exit", staticmethod(lambda code=0: None))
    runner = AutoRun(window, source, result, timeout_s=90, language="en")
    wait_for(lambda: runner.finished and result.exists(), timeout=90, what="autorun to finish")
    data = json.loads(result.read_text(encoding="utf-8"))
    assert data["exit_code"] == 0 and data["state"] == "completed" and Path(data["output_dir"]).is_dir()
    assert window.stack.currentIndex() == PAGE_RESULTS
    settle(window, window.home)


@pytest.mark.usefixtures("fake_ffmpeg")
def test_autorun_can_press_pause_and_resume_like_a_user(qapp, tmp_path, monkeypatch):
    services = make_services(tmp_path)
    monkeypatch.setattr(ModelService, "model_spec", lambda self, key: ModelSpec(key, "faster_whisper", "large-v3"))
    monkeypatch.setattr(tsvc, "STOP_POLL_SECONDS", 0.0)

    class SlowFake(LectureFake):
        def transcribe(self, request):
            real = request.on_segment

            def on_segment(seg):
                real(seg)
                time.sleep(0.12)                              # slow enough for the "button press" to land mid-run

            if real is not None:
                request.on_segment = on_segment
            return super().transcribe(request)

    monkeypatch.setattr("src.transcriber.create_adapter", lambda *a, **k: SlowFake())
    services.update_setting(output_dir=str(tmp_path / "out"), chunk_minutes=1.0)
    window = MainWindow(services, ThreadLauncher())
    window.controller._timer.setInterval(20)
    source = write_silent_wav(tmp_path / "lecture.wav", seconds=120)
    result = tmp_path / "result.json"
    monkeypatch.setattr("src.gui.app.QApplication.exit", staticmethod(lambda code=0: None))
    runner = AutoRun(window, source, result, timeout_s=90, language="en", pause_at=30.0)
    wait_for(lambda: runner.finished and result.exists(), timeout=90, what="autorun with pause to finish")
    data = json.loads(result.read_text(encoding="utf-8"))
    assert data["exit_code"] == 0 and data["state"] == "completed"
    assert data["pause_requested"] is True and data["resumed"] is True and 30.0 <= data["paused_at_percent"] < 100.0
    assert Path(data["output_dir"], "local_transcript.json").is_file()
    settle(window, window.home)


def test_autorun_gives_up_cleanly_when_nothing_happens(qapp, tmp_path, monkeypatch):
    services = make_services(tmp_path, installed=False)                      # no model: START never becomes available
    window = MainWindow(services, None)
    source = write_silent_wav(tmp_path / "lecture.wav", seconds=5)
    result = tmp_path / "result.json"
    monkeypatch.setattr("src.gui.app.QApplication.exit", staticmethod(lambda code=0: None))
    runner = AutoRun(window, source, result, timeout_s=1.0)
    wait_for(lambda: runner.finished, timeout=10, what="autorun to time out")
    data = json.loads(result.read_text(encoding="utf-8"))
    assert data["exit_code"] == 124 and data["state"] == "timeout"
    settle(window, window.home)
