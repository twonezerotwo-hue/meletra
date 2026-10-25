"""Build provenance: the commit of an installed build is readable (About / Diagnostics), and the installer guards a running app."""

import json
from pathlib import Path

from src import build_info

ROOT = Path(__file__).resolve().parent.parent


def test_build_info_reads_the_stamped_file(tmp_path, monkeypatch):
    (tmp_path / build_info.FILE_NAME).write_text(json.dumps({"commit": "0123456789abcdef", "branch": "release/x", "build_date": "2026-10-11"}), encoding="utf-8")
    monkeypatch.setattr(build_info, "APP_DIR", tmp_path)
    assert build_info.commit() == "0123456"
    assert build_info.commit(short=False) == "0123456789abcdef"
    assert build_info.load()["branch"] == "release/x"
    assert "commit 0123456" in build_info.line()


def test_build_info_without_a_stamp_is_honest(tmp_path, monkeypatch):
    monkeypatch.setattr(build_info, "APP_DIR", tmp_path)
    monkeypatch.setattr(build_info, "RESOURCE_ROOT", tmp_path)
    assert build_info.commit() == "source checkout"
    (tmp_path / build_info.FILE_NAME).write_text("{broken", encoding="utf-8")
    assert build_info.load()["commit"] == ""


def test_diagnostics_reports_the_commit():
    from src.services import diagnostics_service
    source = Path(diagnostics_service.__file__).read_text(encoding="utf-8")
    assert '"commit": build_info.commit' in source


def test_installer_closes_only_its_own_program_and_never_overwrites_a_locked_install():
    script = (ROOT / "build_tools" / "installer.iss").read_text(encoding="utf-8")
    assert "function PrepareToInstall" in script and "function InitializeUninstall" in script
    # graceful, and only the program file of the install folder (or of the folder of the installation it replaces), never another copy or another Python
    assert "CloseMainWindow" in script and "Split-Path -Parent $_.Path" in script and "$d -contains" in script and "-Name MELETRA,MedicalLectureTranscriber" in script
    assert "taskkill" not in script.lower() and "Stop-Process" not in script    # no forced kills
    assert "CloseApplications=yes" in script


def test_a_build_stamp_left_by_an_older_installation_is_not_believed(tmp_path, monkeypatch):
    """The 4.1.1 setup left the 4.1.0 stamp (and its commit) beside the new program: About named a commit that was not the program."""
    (tmp_path / build_info.FILE_NAME).write_text(json.dumps({"version": "4.0.9", "commit": "e84d681abcdef", "branch": "old", "build_date": "2026-10-01"}), encoding="utf-8")
    monkeypatch.setattr(build_info, "APP_DIR", tmp_path)
    monkeypatch.setattr(build_info, "RESOURCE_ROOT", tmp_path)
    assert build_info.commit() == "source checkout" and build_info.load()["stale_build_file"] == "4.0.9" and build_info.load()["version"] == build_info.__version__
    (tmp_path / build_info.FILE_NAME).write_text(json.dumps({"version": build_info.__version__, "commit": "0123456789abcdef"}), encoding="utf-8")
    assert build_info.commit() == "0123456"                                         # a stamp of THIS version is used


def test_the_installer_removes_the_old_stamp_before_it_writes_the_new_files():
    script = (ROOT / "build_tools" / "installer.iss").read_text(encoding="utf-8")
    delete_section = script.split("[InstallDelete]", 1)[1].split("[Files]", 1)[0]
    assert r'Name: "{app}\build_info.json"' in delete_section
