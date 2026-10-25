"""MELETRA - Medical Learning & Transcription: the product is called MELETRA everywhere a person sees a name, and the old names (WhisperMedical, MEDLECTURE, Medical Lecture
Transcriber) appear only where an old installation has to be recognised.

Allowed old names: the installer's upgrade code and its welcome sentence, the Mac installer script, the release notes and the history of the project.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from PySide6.QtWidgets import QAbstractButton, QApplication, QComboBox, QLabel, QLineEdit, QWidget

import src
from build_tools import layout, macos, macos_intel
from build_tools.versioninfo import version_file_text
from src.gui import brand, theme
from src.gui.main_window import MainWindow
from tests.gui_support import ScriptedLauncher, make_services, settle
from tests.study_support import write_lecture_folder

ROOT = Path(__file__).resolve().parent.parent
OLD_NAMES = ("whispermedical", "whisper medical", "medlecture", "medical lecture transcriber")


def old_names_in(text: str) -> list[str]:
    low = re.sub(r"<[^>]+>", "", text or "").lower()
    return [name for name in OLD_NAMES if name in low]


def visible_texts(root: QWidget) -> list[str]:
    """Every string a person (or a screen reader) gets from the widgets below ``root``: labels, buttons, tooltips, accessible names and descriptions, hints, titles."""
    found: list[str] = []
    for widget in [root] + root.findChildren(QWidget):
        found += [widget.toolTip(), widget.accessibleName(), widget.accessibleDescription(), widget.windowTitle()]
        if isinstance(widget, (QLabel, QAbstractButton)):
            found.append(widget.text())
        if isinstance(widget, QLineEdit):
            found.append(widget.placeholderText())
        if isinstance(widget, QComboBox):
            found += [widget.itemText(i) for i in range(widget.count())]
    return [t for t in found if t]


@pytest.fixture
def window(qapp, tmp_path):
    theme.apply_theme(qapp, "dark")
    services = make_services(tmp_path, opened=[])
    folder = write_lecture_folder(tmp_path / "lec")
    record = services.jobs.create_or_get("job-1", tmp_path / "Neuroanatomy.mp4", output_dir=str(folder), state="completed", display_name="Neuroanatomy", duration_seconds=1800)
    services.jobs.update(record.job_id, state="completed", progress=100.0)
    win = MainWindow(services, ScriptedLauncher())
    win.resize(1366, 768)
    win.show()
    win.simple_home.refresh()
    yield win, services, tmp_path, folder
    settle(win, win.home, win.study)
    win.deleteLater()


# ------------------------------------------------------------------------------------------------------------------ the name
def test_the_product_is_meletra_and_the_data_folder_keeps_its_old_identity():
    assert src.APP_NAME == "MELETRA" and src.APP_TAGLINE == "Medical Learning & Transcription" and src.PRODUCT_ID == "MELETRA"
    assert src.APP_ID == "MedicalLectureTranscriber"                                  # %LOCALAPPDATA% folder, credential keys, registry id: unchanged, so no lecture is lost
    assert src.__version__ == "4.1.2"
    assert brand.NAME == "MELETRA" and brand.TAGLINE == "Medical Learning & Transcription"


def test_home_title_sidebar_and_window_say_meletra(window):
    win, *_ = window
    home, shell = win.simple_home, win.simple_shell
    assert win.windowTitle() == "MELETRA"
    assert home.title.text() == "MELETRA" and home.subtitle.text() == "Medical Learning & Transcription"
    assert re.sub(r"<[^>]+>", "", shell.brand.text()) == "MELETRA" and shell.brand.accessibleName() == "MELETRA" and shell.logo.accessibleName() == "MELETRA logo"
    assert [shell.buttons[k].text() for k in ("home", "transcription", "study", "files", "settings")] == ["Home", "Transcription", "Study", "Files", "Settings"]
    assert not win.windowIcon().isNull() and not brand.logo_pixmap(small=True, height=36).isNull() and not brand.logo_pixmap(small=False, height=200).isNull()


def test_about_says_meletra_4_1_0_and_the_tagline(window):
    from src.gui.about_dialog import AboutDialog

    win, *_ = window
    about = AboutDialog(win)
    about.show()
    QApplication.processEvents()
    text = " ".join(label.text() for label in about.findChildren(QLabel))
    assert about.windowTitle().endswith("MELETRA") and "MELETRA" in text and f"Version {src.__version__}" in text and "Medical Learning & Transcription" in text
    about.close()


@pytest.mark.parametrize("page", ["home", "files", "active", "batch", "result", "transcript"])
def test_no_page_shows_an_old_product_name(window, page):
    win, services, tmp, folder = window
    win.show_finished(folder, "Neuroanatomy.mp4")
    win.transcript_view.show_transcript(folder, "Neuroanatomy")
    win.show_home() if page == "home" else win._goto(page)
    win.files_page.refresh()
    QApplication.processEvents()
    bad = [(t, old_names_in(t)) for t in visible_texts(win) if old_names_in(t)]
    assert bad == [], (page, bad[:3])


def test_study_and_settings_and_the_dialogs_show_no_old_product_name(window):
    from src.gui import diagnostics_probe as probe
    from src.gui.about_dialog import AboutDialog
    from src.gui.components import ConfirmDialog
    from src.gui.first_run import FirstRunDialog
    from src.gui.sharing_notice import SharingNoticeDialog
    from tests.gui_support import wait_for

    win, services, tmp, folder = window
    win.show_finished(folder, "Neuroanatomy.mp4")
    win.open_study("overview")
    wait_for(lambda: probe.study_ready(win), timeout=30, what="the study material")
    for tab in ("overview", "summary", "concepts", "flashcards", "questions", "study"):
        win.study.select_tab(tab)
        QApplication.processEvents()
        assert [t for t in visible_texts(win) if old_names_in(t)] == [], tab
    settings = win.make_settings_dialog("General")
    settings.show()
    for name in ("Transcription", "Medical", "Study", "Privacy & Data", "Models", "Advanced"):
        settings.select_tab(name)
        QApplication.processEvents()
        assert [t for t in visible_texts(settings) if old_names_in(t)] == [], name
    assert settings.about_button.text() == "About MELETRA ..."
    settings.close()
    for dialog in (AboutDialog(win), FirstRunDialog(services, win), SharingNoticeDialog(services, win), ConfirmDialog(win, "Remove from archive", "x", accept="Remove")):
        dialog.show()
        QApplication.processEvents()
        assert [t for t in visible_texts(dialog) if old_names_in(t)] == [], type(dialog).__name__
        dialog.close()
    assert "Welcome" in FirstRunDialog(services, win).windowTitle() and FirstRunDialog(services, win).windowTitle().endswith("MELETRA")


def test_no_shipped_source_file_uses_an_old_name_except_the_history_line():
    allowed = {"src/__init__.py"}
    found = []
    for folder in ("src", "config", "backend"):
        for path in (ROOT / folder).rglob("*"):
            if path.suffix in {".py", ".json", ".yaml", ".yml", ".txt", ".md"} and "__pycache__" not in path.parts and path.relative_to(ROOT).as_posix() not in allowed:
                text = path.read_text(encoding="utf-8", errors="replace")
                if path.relative_to(ROOT).as_posix() == "config/app_models.json":
                    text = text.replace("whispermedical-downloads", "")      # the technical address of the public download repository: not renamed, so that every earlier link keeps working
                if old_names_in(text):
                    found.append(path.relative_to(ROOT).as_posix())
    assert found == [], found


# ------------------------------------------------------------------------------------------------------------------ the files a person sees outside the window
def test_the_exe_resource_and_the_artifacts_carry_the_new_name():
    text = version_file_text("4.1.2", src.COMPANY, src.APP_NAME, f"{src.APP_NAME} - {src.APP_TAGLINE}", f"{src.PRODUCT_ID}.exe")
    assert "StringStruct('ProductName', 'MELETRA')" in text and "StringStruct('FileDescription', 'MELETRA - Medical Learning & Transcription')" in text
    assert "StringStruct('OriginalFilename', 'MELETRA.exe')" in text and "StringStruct('InternalName', 'MELETRA')" in text
    build_app = (ROOT / "scripts" / "build_app.py").read_text(encoding="utf-8")
    assert "{APP_NAME} - {APP_TAGLINE}" in build_app and 'f"{PRODUCT_ID}.exe"' in build_app                    # the real build writes exactly this resource
    from scripts import build_installer, build_portable

    assert build_installer.OUTPUT_BASE == "MELETRA-Setup-Windows-x64" and build_portable.NAME == "MELETRA-Portable-Windows-x64.zip"
    assert layout.dist_dir("4.1.2", ROOT) == ROOT / "dist" / "v4.1.2" / "MELETRA" and layout.dist_dir("4.0.0", ROOT).name == "MedicalLectureTranscriber"
    assert (ROOT / "build_tools" / "MELETRA.spec").is_file() and not (ROOT / "build_tools" / "MedicalLectureTranscriber.spec").exists()
    assert 'name="MELETRA"' in (ROOT / "build_tools" / "MELETRA.spec").read_text(encoding="utf-8")
    assert macos.APP_BUNDLE == "MELETRA.app" and macos_intel.APP_FOLDER == "MELETRA.app"


def test_the_installer_script_installs_meletra_and_upgrades_the_old_names():
    iss = (ROOT / "build_tools" / "installer.iss").read_text(encoding="utf-8")
    setup = iss[iss.index("[Setup]"):iss.index("[Messages]")]
    assert '#define AppName "MELETRA"' in iss and '#define AppExe "MELETRA.exe"' in iss and "AppName={#AppName}" in setup and "UninstallDisplayName={#AppName}" in setup
    assert "WhisperMedical" not in setup and "Medical Lecture Transcriber" not in setup                      # the setup properties are the new product
    messages = iss[iss.index("[Messages]"):iss.index("[Tasks]")]
    assert "Welcome to {#AppName}" in messages and "Medical Learning & Transcription" in messages and "your lectures and settings are kept" in messages
    assert "WizardInfoBefore=Your privacy" in messages
    # the upgrade: the same AppId, the old shortcuts and program file removed, an earlier folder replaced through its own (silent, data-keeping) uninstaller
    assert "{6F0B7C52-3A1E-4E7D-9C54-8D2A7B1E4F60}" in iss and "[InstallDelete]" in iss and "{autoprograms}\\WhisperMedical" in iss and "MedicalLectureTranscriber.exe" in iss
    assert "UsePreviousGroup=no" in setup                                                                    # the shortcuts of an upgrade do not go into the Start menu folder of the old name
    assert "RemovePreviousInstallation" in iss and "/VERYSILENT /SUPPRESSMSGBOXES" in iss and "{localappdata}\\MedicalLectureTranscriber" in iss          # the data folder is the old one
    notice = (ROOT / "build_tools" / "installer_notice.txt").read_text(encoding="utf-8")
    assert "MELETRA" in notice and old_names_in(notice) == []


def test_the_readme_and_the_brand_files_use_the_new_name():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert readme.splitlines()[0].count("meletra_logo.png") == 1 and "# MELETRA" in readme and "Medical Learning & Transcription" in readme
    assert "## Public build mirror" in readme and "## Reproduce the Apple Silicon build" in readme
    assert not any(path.name.startswith(("app_", "whispermedical")) for path in (ROOT / "assets" / "brand").iterdir())


def test_the_installer_wizard_pictures_are_the_new_ones():
    from PIL import Image

    for tag, scale in (("100", 1.0), ("150", 1.5), ("200", 2.0)):
        large = Image.open(ROOT / "assets" / "brand" / "installer" / f"wizard_large_{tag}.bmp")
        assert large.size == (round(164 * scale), round(314 * scale))
        pixels = large.convert("RGB").resize((82, 157)).getdata()
        assert sum(1 for r, g, b in pixels if r > 150 and g > 100 and b < 120) > 80                    # the gold of the MELETRA wordmark and the symbol is on the panel
