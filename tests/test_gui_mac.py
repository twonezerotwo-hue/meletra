"""macOS behaviour of the window, checked on this Windows machine with a fake Mac (fake home folder, fake command answers): wording,
system check, GPU status, model defaults, native menu roles, Finder "open file" events. Windows behaviour is asserted to be unchanged."""

from pathlib import Path

import pytest
from PySide6.QtCore import QUrl
from PySide6.QtGui import QAction, QFileOpenEvent
from PySide6.QtWidgets import QApplication

from src.gui import first_run
from src.gui.app import FileOpenFilter
from src.gui.choices import theme_choices
from src.gui.first_run import FirstRunDialog, system_check
from src.gui.main_window import SIMPLE_HOME, MainWindow
from src.platform import PosixPlatform, use
from src.services.gpu_service import GpuService
from src.services.model_service import ModelService, load_registry
from tests.conftest import write_silent_wav
from tests.gui_support import ScriptedLauncher, make_services, settle, wait_for
from tests.test_platform import mac

SYSCTL = {("sysctl", "-n", "hw.memsize"): ("17179869184", 0), ("sysctl", "-n", "machdep.cpu.brand_string"): ("Apple M2 Pro", 0),
          ("sw_vers", "-productVersion"): ("14.5", 0)}


def mac_services(tmp_path, machine="arm64"):
    platform, _ = mac(tmp_path, answers=SYSCTL, machine=machine)
    services = make_services(tmp_path)
    services.models = ModelService(services.paths, registry=load_registry(), app_dir=tmp_path / "appdir", platform=platform)
    return services, platform


def rows(services):
    return {name: (text, level) for name, text, level in system_check(services)}


# ------------------------------------------------------------------------------------------------------------------ wording
@pytest.mark.windows_platform
def test_theme_choice_names_the_system(tmp_path):
    assert theme_choices()[0] == ("Follow Windows", "system")
    platform, _ = mac(tmp_path)
    with use(platform):
        assert theme_choices()[0] == ("Follow macOS", "system")
    with use(PosixPlatform(env={}, home=tmp_path)):
        assert theme_choices()[0] == ("Follow the system", "system")


@pytest.mark.windows_platform
def test_gpu_status_on_a_mac_has_no_nvidia_words(tmp_path):
    arm, _ = mac(tmp_path, machine="arm64")
    intel, _ = mac(tmp_path, machine="x86_64")
    with use(arm):
        status = GpuService(smi=lambda i: pytest.fail("nvidia-smi on a Mac"), cuda=lambda i: pytest.fail("CUDA on a Mac")).detect()
    assert status.device == "metal" and status.headline == "Apple Silicon detected" and "GPU" in status.detail and not status.cuda_available
    with use(intel):
        status = GpuService(smi=lambda i: None, cuda=lambda i: (False, "x")).detect()
    assert status.device == "cpu" and status.headline == "Intel Mac detected" and "CPU" in status.detail
    windows = GpuService(smi=lambda i: None, cuda=lambda i: (False, "no driver")).detect()               # this machine: unchanged
    assert windows.headline == "No compatible NVIDIA GPU detected."


# ------------------------------------------------------------------------------------------------------------------ system check
@pytest.mark.windows_platform
def test_the_windows_system_check_is_unchanged(qapp, tmp_path):
    names = [name for name, _t, _l in system_check(make_services(tmp_path))]
    assert names == ["Windows", "CPU", "Memory", "GPU", "CUDA acceleration", "Disk space", "Audio decoder"]


def test_apple_silicon_system_check(qapp, tmp_path, monkeypatch):
    services, platform = mac_services(tmp_path)
    monkeypatch.setattr("importlib.util.find_spec", lambda name, *a: object())
    with use(platform):
        found = rows(services)
    assert list(found) == ["macOS", "Processor", "Memory", "Speech engine", "Disk space", "Speech model", "Audio decoder"]
    assert "14.5" in found["macOS"][0] and found["Processor"] == ("Apple M2 Pro - Apple Silicon", "ok")
    assert found["Memory"][0].startswith("16.0 GB") and found["Memory"][1] == "ok"
    assert found["Speech engine"] == ("Optimized for Apple Silicon (uses this Mac's GPU)", "ok")
    text, level = found["Speech model"]
    assert "Whisper Large-v3 (Apple Silicon)" in text and "Download recommended model" in text and level == "warn"
    assert not any(word in " ".join(t for t, _l in found.values()) for word in ("CUDA", "NVIDIA", "Windows"))


def test_a_missing_mlx_engine_is_explained_on_apple_silicon(qapp, tmp_path, monkeypatch):
    services, platform = mac_services(tmp_path)
    monkeypatch.setattr("importlib.util.find_spec", lambda name, *a: None if name == "mlx_whisper" else object())
    with use(platform):
        text, level = rows(services)["Speech engine"]
    assert level == "warn" and "not available" in text and "standard engine" in text


def test_intel_mac_system_check_warns_about_speed(qapp, tmp_path):
    services, platform = mac_services(tmp_path, machine="x86_64")
    with use(platform):
        found = rows(services)
    assert found["Processor"][1] == "warn" and "slower" in found["Processor"][0] and found["Speech engine"][0] == "Standard engine on the CPU"
    assert "Whisper Large-v3 is not installed" in found["Speech model"][0]


def test_the_first_run_wizard_offers_the_recommended_mac_model(qapp, tmp_path, monkeypatch):
    services, platform = mac_services(tmp_path)
    monkeypatch.setattr("importlib.util.find_spec", lambda name, *a: object())
    with use(platform):
        dialog = FirstRunDialog(services)
        wait_for(lambda: dialog.rows, what="the system check")
    assert "Whisper Large-v3 (Apple Silicon)" in dialog.model_label.text() and dialog.download_button.text() == "Download recommended model"
    assert "[OK  ]  macOS" in dialog.check_label.text() and "CUDA" not in dialog.check_label.text()
    settle(dialog)
    dialog.deleteLater()


# ------------------------------------------------------------------------------------------------------------------ defaults
def test_a_new_apple_silicon_install_starts_with_the_mlx_model(tmp_path):
    services, platform = mac_services(tmp_path)
    assert services.settings.settings.model_key == "whisper-large-v3"
    with use(platform):
        services.apply_platform_defaults()
    assert services.settings.settings.model_key == "whisper-large-v3-mlx"
    services.update_setting(model_key="whisper-large-v3", first_run_done=True)
    services.apply_platform_defaults()
    assert services.settings.settings.model_key == "whisper-large-v3"                    # a choice after the first run stands


def test_windows_keeps_its_default_model(tmp_path):
    services = make_services(tmp_path)
    services.apply_platform_defaults()
    assert services.settings.settings.model_key == "whisper-large-v3"


# ------------------------------------------------------------------------------------------------------------------ native behaviour
@pytest.fixture
def window(qapp, tmp_path):
    w = MainWindow(make_services(tmp_path), ScriptedLauncher())
    yield w
    settle(w, w.home, w.study)
    w.deleteLater()


def test_menu_items_have_the_roles_macos_moves_into_its_application_menu(window):
    by_text = {a.text(): a for menu in window.menuBar().actions() for a in menu.menu().actions() if a.text()}
    assert by_text["E&xit"].menuRole() == QAction.QuitRole
    assert by_text["&Settings ..."].menuRole() == QAction.PreferencesRole
    assert by_text["&About"].menuRole() == QAction.AboutRole
    assert by_text["&Open audio or video ..."].shortcut().matches(__import__("PySide6.QtGui", fromlist=["QKeySequence"]).QKeySequence.Open)


def test_finder_open_events_select_the_file_or_attach_the_pdf(window, tmp_path):
    audio = write_silent_wav(tmp_path / "lecture.wav", seconds=30)
    pdf = tmp_path / "slides.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    window.open_external_file(str(audio))
    wait_for(lambda: window.home.duration is not None, what="the file probe")
    assert window.home.source == audio and window.simple_stack.currentIndex() == SIMPLE_HOME
    window.open_external_file(str(pdf))
    assert window.home.pdf_path == pdf
    window.open_external_file(str(tmp_path / "notes.docx"))                       # not something the program can use: ignored
    assert window.home.source == audio


def test_a_running_job_ignores_finder_open_events(window, tmp_path, monkeypatch):
    monkeypatch.setattr(type(window.controller), "is_running", property(lambda self: True))
    window.open_external_file(str(write_silent_wav(tmp_path / "other.wav", seconds=10)))
    assert window.home.source is None and "being transcribed" in window.statusBar().currentMessage()


def test_files_opened_before_the_window_exists_wait_for_it(qapp, tmp_path):
    filter_ = FileOpenFilter()
    qapp.installEventFilter(filter_)
    try:
        QApplication.sendEvent(qapp, QFileOpenEvent(QUrl.fromLocalFile(str(tmp_path / "early.wav"))))
        assert [Path(p) for p in filter_.pending] == [tmp_path / "early.wav"]
        got = []
        filter_.attach(got.append)
        assert [Path(p) for p in got] == [tmp_path / "early.wav"] and filter_.pending == []
        QApplication.sendEvent(qapp, QFileOpenEvent(QUrl.fromLocalFile(str(tmp_path / "late.wav"))))
        assert Path(got[-1]) == tmp_path / "late.wav"
    finally:
        qapp.removeEventFilter(filter_)
