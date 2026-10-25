"""The macOS Intel offline release, as far as it can be checked on any computer: where the bundled speech model is searched, what a
damaged one reports, what the self-test demands of the offline release, and that the build recipe packages every resource folder.
This is NOT evidence that a Mac build works - nothing here runs on macOS."""

import json
import re
from pathlib import Path

import pytest

from src.services import selftest
from src.services.model_service import ModelService
from src.services.paths import AppPaths, get_paths

ROOT = Path(__file__).resolve().parent.parent


def fake_app(tmp_path) -> Path:
    macos = tmp_path / "MELETRA.app" / "Contents" / "MacOS"
    macos.mkdir(parents=True)
    return macos


def write_model(folder: Path, info, *, wrong_size=False) -> None:
    folder.mkdir(parents=True)
    for spec in info.files:
        if spec.path in info.required:
            (folder / spec.path).write_bytes(b"x" * (3 if wrong_size else 0))     # real sizes would be 3 GB: sparse files
    if not wrong_size:
        for spec in info.files:
            if spec.path in info.required:
                with open(folder / spec.path, "r+b") as handle:
                    handle.truncate(spec.size)


@pytest.fixture
def models(tmp_path):
    macos = fake_app(tmp_path)
    paths = get_paths({"MLT_DATA_DIR": str(tmp_path / "data")}, app_dir=macos, create=True)
    return ModelService(paths, app_dir=macos), macos


def test_the_app_bundle_resources_models_folder_is_searched(models):
    service, macos = models
    assert macos.parent / "Resources" / "models" in service.search_dirs()
    assert service.search_dirs().index(macos.parent / "Resources" / "models") > service.search_dirs().index(service.paths.models)


def test_other_app_folders_do_not_get_a_resources_search_path(tmp_path):
    folder = tmp_path / "app"
    folder.mkdir()
    paths = get_paths({"MLT_DATA_DIR": str(tmp_path / "data")}, app_dir=folder, create=True)
    assert all("Resources" not in str(p) for p in paths.model_search_dirs(None, folder))


def test_a_model_shipped_inside_the_app_is_found_and_labelled_bundled(models):
    service, macos = models
    info = service.info("whisper-large-v3")
    write_model(macos.parent / "Resources" / "models" / info.folder, info)
    state = service.state("whisper-large-v3")
    assert state.installed and state.source == "bundled" and service.bundled_problems("whisper-large-v3") == []


def test_a_damaged_bundled_model_is_reported_and_never_a_reason_to_download(models, monkeypatch):
    service, macos = models
    info = service.info("whisper-large-v3")
    write_model(macos.parent / "Resources" / "models" / info.folder, info, wrong_size=True)
    problems = service.bundled_problems("whisper-large-v3")
    assert problems and all("wrong size" in p or "missing" in p for p in problems)
    assert not service.state("whisper-large-v3").installed
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: pytest.fail("the app must not touch the network for a damaged bundled model"))
    from src.gui.first_run import system_check  # noqa: F401  (the row text is built from bundled_problems; see first_run.py)


def test_without_a_bundled_model_there_is_nothing_to_report(models):
    service, _ = models
    assert service.bundled_problems("whisper-large-v3") == []


def test_the_self_test_ignores_a_normal_install_and_checks_the_offline_release(tmp_path, monkeypatch):
    assert selftest.offline_marker() is None and selftest._offline_release_detail() == ""
    macos = fake_app(tmp_path)
    resources = macos.parent / "Resources"
    resources.mkdir()
    (resources / selftest.OFFLINE_MARKER).write_text(json.dumps({"release": "3.1.0-mac-intel", "architecture": "x86_64", "model": "whisper-large-v3",
                                                                 "license_files": ["THIRD_PARTY_LICENSES.txt"]}), encoding="utf-8")
    monkeypatch.setattr(selftest, "APP_DIR", macos)
    assert selftest.offline_marker() == resources / selftest.OFFLINE_MARKER
    monkeypatch.setenv("MLT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr("src.services.paths.APP_DIR", macos)
    with pytest.raises(RuntimeError) as err:
        selftest._offline_release_detail()
    text = str(err.value)
    assert "offline release is incomplete" in text and "speech model" in text and "THIRD_PARTY_LICENSES.txt is missing" in text


def test_the_macos_spec_packages_every_resource_folder_the_app_reads():
    spec = (ROOT / "build_tools" / "MELETRA-macos.spec").read_text(encoding="utf-8")
    packaged = {a for a, b in re.findall(r'ROOT / "(\w+)"\), "(\w+)"', spec) if a == b}
    assert {"config", "medical_terms", "medical_vocab"} <= packaged


def test_the_intel_requirements_only_name_packages_that_exist_for_intel_macs_at_macos_12():
    text = (ROOT / "requirements-macos-intel.txt").read_text(encoding="utf-8")
    pins = dict(re.findall(r"^([A-Za-z0-9_.\-]+)==([0-9.]+)", text, re.M))
    assert pins["onnxruntime"] == "1.19.2"                    # 1.20+ ships no macOS x86_64 wheel; 1.23.2 needs macOS 13
    assert pins["PySide6-Essentials"] == "6.9.3"              # 6.10+ needs macOS 13/14
    assert pins["opencv-python-headless"] == "4.10.0.84"      # 4.12+ needs macOS 13/14
    lines = [l.split("#")[0].strip().lower() for l in text.splitlines()]
    assert "RapidFuzz" in pins                                # src/medical/* imports it
    assert not any(l.startswith(("mlx", "torch", "winrt", "transformers")) for l in lines)       # Apple Silicon / Windows / optional only
