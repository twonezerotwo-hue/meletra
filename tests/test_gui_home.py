"""Start page: file selection, drag & drop, model state, options, START."""

from pathlib import Path

import pytest

from src.gui.home_page import HomePage, PRIVACY_CLOUD, PRIVACY_LOCAL
from src.gui.widgets import DropZone, first_media_path, is_supported_media
from src.pipeline import job_key
from src.services.gpu_service import GpuStatus
from tests.conftest import write_silent_wav
from tests.gui_support import (
    MODEL_KEY, SMI, drop_event, make_gpu, make_services, settle, wait_for,
)


@pytest.fixture
def home(qapp, tmp_path):
    services = make_services(tmp_path)
    page = HomePage(services)
    page.confirm = lambda parent, title, text, ok: True
    yield page
    settle(page)
    page.deleteLater()


@pytest.fixture
def lecture(tmp_path):
    return write_silent_wav(tmp_path / "week 2 lecture.wav", seconds=75)


def pick(home, path):
    home.set_file(str(path))
    wait_for(lambda: home.duration is not None or not home.problem_label.isHidden() or home.duration_label.text() != "reading ...",
             what="the file probe")


def test_default_options_match_the_spec(home):
    assert home.language_combo.currentText() == "Auto Detect"
    assert home.subject_combo.currentText() == "Auto Detect"
    assert home.mode_combo.currentText() == "Best quality"
    assert "Whisper Large-v3" in home.model_combo.currentText()
    assert not home.start_button.isEnabled() and home.start_button.text() == "START TRANSCRIPTION"


def test_privacy_badge_is_local_by_default(home):
    assert home.privacy_badge.text() == PRIVACY_LOCAL and "stay on this computer" in PRIVACY_LOCAL and "Privacy & Data" in PRIVACY_LOCAL


def test_cloud_setting_changes_the_badge(qapp, tmp_path):
    services = make_services(tmp_path)
    services.update_setting(cloud_enabled=True)
    page = HomePage(services)
    assert page.privacy_badge.text() == PRIVACY_CLOUD and "uploaded" in PRIVACY_CLOUD
    page.deleteLater()


def test_language_choices(home):
    assert [home.language_combo.itemData(i) for i in range(home.language_combo.count())] == ["auto", "en", "tr", "de", "hu", "ko", "zh", "hi", "nl", "fr", "es", "mixed"]


def test_subject_choices_include_auto_and_the_medical_areas(home):
    values = [home.subject_combo.itemData(i) for i in range(home.subject_combo.count())]
    assert values[0] == "auto" and "biochemistry" in values and len(values) >= 15


def test_choosing_a_file_shows_name_size_and_duration(home, lecture):
    pick(home, lecture)
    assert home.file_label.text() == "week 2 lecture.wav"
    assert "KB" in home.size_label.text() or "MB" in home.size_label.text()
    assert home.duration_label.text() == "00:01:15"
    assert home.start_button.isEnabled()


def test_start_stays_disabled_without_a_file(home):
    assert not home.start_button.isEnabled()


def test_unreadable_file_is_reported_and_cannot_be_started(home, tmp_path):
    bad = tmp_path / "broken.mp3"
    bad.write_bytes(b"this is not audio at all" * 50)
    pick(home, bad)
    assert not home.start_button.isEnabled()
    assert "could not be read" in home.problem_label.text()


def test_missing_file_is_reported(home, tmp_path):
    home.set_file(str(tmp_path / "gone.wav"))
    assert "was not found" in home.problem_label.text() and not home.start_button.isEnabled()


def test_supported_extensions():
    for ext in "mp3 wav m4a aac flac ogg opus mp4 mov mkv webm".split():
        assert is_supported_media(Path(f"x.{ext.upper()}")), ext
    for ext in ("txt", "pdf", "exe", "docx"):
        assert not is_supported_media(Path(f"x.{ext}"))


def test_drop_a_supported_file_selects_it(home, lecture):
    zone: DropZone = home.drop_zone
    chosen = []
    zone.file_chosen.connect(chosen.append)
    zone.dropEvent(drop_event(str(lecture)))
    assert chosen == [str(lecture)]
    wait_for(lambda: home.duration is not None, what="probe after drop")


def test_drop_an_unsupported_file_is_refused_with_a_message(home, tmp_path):
    notes = tmp_path / "notes.txt"
    notes.write_text("hello")
    home.drop_zone.dropEvent(drop_event(str(notes)))
    assert "not a supported audio or video file" in home.problem_label.text() and home.source is None


def test_first_media_path_skips_non_media(qapp, tmp_path):
    from PySide6.QtCore import QMimeData, QUrl

    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(tmp_path / "a.txt")), QUrl.fromLocalFile(str(tmp_path / "b.mp4"))])
    assert first_media_path(mime) == tmp_path / "b.mp4"
    assert first_media_path(QMimeData()) is None


def test_click_opens_the_file_chooser(home, lecture):
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    home.drop_zone.chooser = lambda: str(lecture)
    event = QMouseEvent(QEvent.MouseButtonPress, QPointF(5, 5), QPointF(5, 5), Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
    home.drop_zone.mousePressEvent(event)
    assert home.source == lecture


def test_model_not_installed_shows_download_and_blocks_start(qapp, tmp_path, lecture):
    services = make_services(tmp_path, installed=False)
    page = HomePage(services)
    page.set_file(str(lecture))
    wait_for(lambda: page.duration is not None, what="probe")
    assert not page.model_banner.isHidden() and "Model not installed" in page.model_text.text()
    assert page.download_button.text() == "DOWNLOAD MODEL" and not page.start_button.isEnabled()
    settle(page)


def test_partial_download_offers_resume(qapp, tmp_path):
    services = make_services(tmp_path, installed=False)
    partial = services.paths.models / ".partial" / services.models.info(MODEL_KEY).folder
    partial.mkdir(parents=True)
    (partial / "model.bin.part").write_bytes(b"x" * 4000)
    page = HomePage(services)
    assert page.download_button.text() == "RESUME DOWNLOAD" and "partial download" in page.model_text.text()


def test_download_button_downloads_and_unlocks_start(qapp, tmp_path, lecture, monkeypatch):
    services = make_services(tmp_path, installed=False)
    page = HomePage(services)

    def fake_download(key, progress=None, cancel=None):
        progress({"type": "model_download_progress", "percent": 50.0, "bytes_done": 10, "bytes_total": 20})
        from tests.gui_support import install_fake_model
        return install_fake_model(services.models, key)

    monkeypatch.setattr(services.models, "download", fake_download)
    page.set_file(str(lecture))
    wait_for(lambda: page.duration is not None, what="probe")
    page.download_button.click()
    wait_for(lambda: page.model_banner.isHidden() and page.start_button.isEnabled(), what="download to finish")
    settle(page)


def test_failed_download_shows_the_reason_and_keeps_the_button(qapp, tmp_path, monkeypatch):
    services = make_services(tmp_path, installed=False)
    page = HomePage(services)

    def broken(key, progress=None, cancel=None):
        raise RuntimeError("The download server could not be reached.")

    monkeypatch.setattr(services.models, "download", broken)
    page.download_button.click()
    wait_for(lambda: not page.problem_label.isHidden(), what="error message")
    assert "could not be reached" in page.problem_label.text() and page.download_button.isEnabled()
    settle(page)


def test_cuda_unavailable_banner_tells_the_user_about_the_cpu_fallback(home):
    status = GpuStatus(nvidia_present=False, cuda_available=False, headline="No compatible NVIDIA GPU detected.",
                       detail="CUDA acceleration unavailable.\nCPU fallback enabled.")
    home.set_gpu_status(status)
    assert "CPU fallback enabled" in home.device_banner.text() and home.device_banner.objectName() == "bannerWarn"


@pytest.mark.windows_platform
def test_gpu_banner_names_the_gpu(home):
    home.set_gpu_status(make_gpu().detect())
    assert "RTX 2070" in home.device_banner.text() and home.device_banner.objectName() == "banner"


def test_build_request_carries_the_choices(home, lecture):
    pick(home, lecture)
    home.language_combo.setCurrentIndex(1)
    home.subject_combo.setCurrentIndex(home.subject_combo.findData("biochemistry"))
    home.mode_combo.setCurrentIndex(1)
    request = home.build_request()
    assert (request.language, request.subject, request.mode, request.model_key) == ("en", "biochemistry", "fast", MODEL_KEY)
    assert request.source == str(lecture) and request.device == "auto" and request.cloud_compare is False
    assert request.output_dir.endswith("Medical Transcription")


def test_output_folder_setting_overrides_the_default(home, lecture, tmp_path):
    home.services.update_setting(output_dir=str(tmp_path / "mine"))
    pick(home, lecture)
    assert home.build_request().output_dir == str(tmp_path / "mine")


def test_start_emits_one_request_and_remembers_the_choices(home, lecture):
    pick(home, lecture)
    home.set_gpu_status(make_gpu().detect())
    got = []
    home.start_requested.connect(got.append)
    home.language_combo.setCurrentIndex(2)
    home.start_button.click()
    assert len(got) == 1 and got[0].language == "tr"
    assert home.services.settings.settings.language == "tr"


def test_cloud_on_without_a_key_refuses_to_start(home, lecture):
    home.services.update_setting(cloud_enabled=True)
    pick(home, lecture)
    got = []
    home.start_requested.connect(got.append)
    home.start()
    assert not got and "no Deepgram API key" in home.problem_label.text()


def test_cloud_upload_needs_an_explicit_yes(home, lecture):
    home.services.update_setting(cloud_enabled=True)
    home.services.credentials.set_deepgram_key("dg_secret_value_123456")
    pick(home, lecture)
    got, asked = [], []
    home.start_requested.connect(got.append)
    home.confirm = lambda parent, title, text, ok: (asked.append(text), False)[1]
    home.start()
    assert not got and "uploaded to Deepgram" in asked[0]
    home.confirm = lambda parent, title, text, ok: True
    home.start()
    assert len(got) == 1 and got[0].cloud_compare is True


@pytest.mark.windows_platform
def test_low_gpu_memory_asks_before_starting(home, lecture):
    low = make_gpu(smi={**SMI, "free_mib": 2500}, programs=["ollama.exe"])
    home.services.gpu = low
    home.set_gpu_status(low.detect())
    pick(home, lecture)
    got, asked = [], []
    home.start_requested.connect(got.append)
    home.choose_device = lambda parent, check: (asked.append(check), "cancel")[1]
    home.start()
    assert not got and "ollama.exe" in asked[0].message                                     # Cancel: nothing starts
    home.choose_device = lambda parent, check: "gpu"
    home.start()
    assert len(got) == 1 and got[0].device != "cpu"                                          # Continue on GPU
    home.choose_device = lambda parent, check: "cpu"
    home.start()
    assert len(got) == 2 and got[1].device == "cpu"                                          # Use CPU: this job runs on the CPU


def test_the_recommended_device_and_an_unattended_run_never_wait():
    from src.gui.app import unattended_device_choice
    from src.gui.home_page import recommended_device
    from src.services.gpu_service import VramCheck

    critical = VramCheck(False, "critical", 1500, 4500, ["llama-server.exe"], "Only 1.5 GB free")
    warning = VramCheck(False, "warning", 3000, 4500, [], "Only 2.9 GB free")
    assert recommended_device(critical) == "cpu" and recommended_device(warning) == "gpu"
    assert unattended_device_choice(None, critical) == "cpu" and unattended_device_choice(None, warning) == "gpu"


def test_no_vram_prompt_on_cpu(home, lecture):
    cpu = make_gpu(smi=None, cuda=(False, "No compatible NVIDIA GPU / driver was detected."))
    home.services.gpu = cpu
    home.set_gpu_status(cpu.detect())
    pick(home, lecture)
    home.confirm = lambda *a: pytest.fail("must not ask about GPU memory on a CPU-only machine")
    home.choose_device = lambda *a: pytest.fail("must not ask about GPU memory on a CPU-only machine")
    got = []
    home.start_requested.connect(got.append)
    home.start()
    assert len(got) == 1


def test_pdf_can_be_added_and_removed(home, lecture, tmp_path):
    pdf = tmp_path / "slides.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    home.choose_pdf = lambda: str(pdf)
    pick(home, lecture)
    home.add_pdf()
    assert home.pdf_path == pdf and "slides.pdf" in home.pdf_label.text()
    assert home.build_request().pdf_path == str(pdf)
    home.pdf_clear.click()
    assert home.pdf_path is None and home.build_request().pdf_path == ""


def test_unfinished_job_for_the_same_file_offers_resume_or_start_over(home, lecture):
    pick(home, lecture)
    jid = job_key(lecture)
    home.services.jobs.create_or_get(jid, lecture, state="paused", progress=42.0, chunk=2, chunks=5)
    (home.services.paths.checkpoints / jid).mkdir(parents=True)
    home.refresh_unfinished()
    assert not home.resume_box.isHidden() and "42%" in home.resume_text.text() and "chunk 2 of 5" in home.resume_text.text()
    got = []
    home.start_requested.connect(got.append)
    home.resume_button.click()
    assert got and got[0].resume is True
    got.clear()
    home.start_over_button.click()
    assert got and got[0].resume is False
    assert not (home.services.paths.checkpoints / jid).exists() and home.services.jobs.get(jid) is None


def test_start_over_asks_first(home, lecture):
    pick(home, lecture)
    jid = job_key(lecture)
    home.services.jobs.create_or_get(jid, lecture, state="paused", progress=10.0)
    home.refresh_unfinished()
    home.confirm = lambda *a: False
    got = []
    home.start_requested.connect(got.append)
    home.start_over_button.click()
    assert not got and home.services.jobs.get(jid) is not None


def test_other_unfinished_jobs_are_listed(home, tmp_path):
    other = write_silent_wav(tmp_path / "other.wav", 5)
    home.services.jobs.create_or_get(job_key(other), other, state="paused", progress=20.0)
    home.refresh_unfinished()
    assert "other.wav" in home.unfinished_label.text()


def test_advanced_and_import_buttons_open_the_right_settings_tab(home):
    tabs = []
    home.settings_requested.connect(tabs.append)
    home.advanced_button.click()
    home.import_button.click()
    assert tabs == ["Performance", "Models"]


def test_selecting_a_file_remembers_its_folder(home, lecture):
    pick(home, lecture)
    assert home.services.settings.settings.last_input_dir == str(lecture.parent)
