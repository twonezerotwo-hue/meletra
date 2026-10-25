"""The model registry knows which models belong to which platform: Windows keeps its list, an Apple Silicon Mac gets the MLX Large-v3,
an Intel Mac and Linux get the CTranslate2 one, and nobody downloads a second multi-gigabyte model they do not need."""

import hashlib
import re
from dataclasses import replace

import pytest

from src.model_manager import get_model_spec, load_model_specs
from src.services.model_service import ModelFile, ModelService, load_registry, platform_key
from src.services.paths import AppPaths
from tests.test_platform import mac


@pytest.fixture()
def registry():
    return load_registry()


def service(tmp_path, platform=None, registry=None):
    return ModelService(AppPaths(tmp_path / "data").ensure(), registry=registry, app_dir=tmp_path / "app", platform=platform)


def install_small_copy(service, key, files):
    """Replace a registry entry by one with tiny files and install them (the real weights are gigabytes)."""
    info = service.info(key)
    small = tuple(ModelFile(name, len(data), hashlib.sha256(data).hexdigest()) for name, data in files.items())
    service.registry[key] = replace(info, files=small, required=tuple(files), approx_bytes=sum(len(d) for d in files.values()))
    folder = service.paths.models / info.folder
    folder.mkdir(parents=True)
    for name, data in files.items():
        (folder / name).write_bytes(data)
    return folder


def test_the_shipped_registry_pins_the_macos_model(registry):
    mlx = registry["whisper-large-v3-mlx"]
    assert mlx.engine == "mlx_whisper" and mlx.repo == "mlx-community/whisper-large-v3-mlx" and mlx.models_yaml_key == "mlx_large_v3"
    assert re.fullmatch(r"[0-9a-f]{40}", mlx.revision) and mlx.platforms == ("macos-arm64",) and mlx.recommended_on == ("macos-arm64",)
    assert mlx.file("weights.npz").size > 3_000_000_000 and re.fullmatch(r"[0-9a-f]{64}", mlx.file("weights.npz").sha256)
    assert set(mlx.required) == {"config.json", "weights.npz"} and mlx.approx_bytes == sum(f.size for f in mlx.files)
    assert get_model_spec(mlx.models_yaml_key, load_model_specs()).engine == "mlx_whisper"
    large = registry["whisper-large-v3"]
    assert large.platforms == ("windows", "macos", "linux") and "macos-arm64" not in large.recommended_on
    assert registry["turkish-medical-large-v3"].platforms == ("windows", "linux")


def test_platform_keys(tmp_path):
    arm, _ = mac(tmp_path, machine="arm64")
    intel, _ = mac(tmp_path, machine="x86_64")
    assert platform_key(arm) == "macos-arm64" and platform_key(intel) == "macos-x86_64"


@pytest.mark.windows_platform
def test_windows_sees_exactly_what_it_saw_in_011(tmp_path, registry):
    s = service(tmp_path, registry=registry)
    assert {m.key for m in s.list_models()} == {"whisper-large-v3", "turkish-medical-large-v3"}
    assert s.platform_key() == "windows" and s.recommended_key() == "whisper-large-v3" and s.is_recommended("whisper-large-v3")
    assert not s.available("whisper-large-v3-mlx") and "whisper-large-v3-mlx" in {m.key for m in s.list_models(all_platforms=True)}
    assert s.preferred_key() == "whisper-large-v3"


def test_apple_silicon_recommends_the_mlx_model(tmp_path, registry):
    arm, _ = mac(tmp_path, machine="arm64")
    s = service(tmp_path, arm, registry)
    shown = {m.key for m in s.list_models()}
    assert shown == {"whisper-large-v3", "whisper-large-v3-mlx"}                    # no PyTorch-only model on a Mac
    assert s.recommended_key() == "whisper-large-v3-mlx" and s.is_recommended("whisper-large-v3-mlx")
    assert not s.is_recommended("whisper-large-v3")
    assert s.preferred_key() == "whisper-large-v3-mlx"


def test_intel_mac_and_linux_use_the_ctranslate2_model(tmp_path, registry):
    intel, _ = mac(tmp_path, machine="x86_64")
    s = service(tmp_path, intel, registry)
    assert {m.key for m in s.list_models()} == {"whisper-large-v3"} and s.recommended_key() == "whisper-large-v3"
    from src.platform import PosixPlatform

    linux = service(tmp_path, PosixPlatform(env={}, home=tmp_path), registry)
    assert linux.platform_key() == "linux" and linux.recommended_key() == "whisper-large-v3"
    assert {m.key for m in linux.list_models()} == {"whisper-large-v3", "turkish-medical-large-v3"}


def test_the_mlx_runtime_must_be_present(tmp_path, registry, monkeypatch):
    arm, _ = mac(tmp_path)
    s = service(tmp_path, arm, registry)
    monkeypatch.setattr("importlib.util.find_spec", lambda name, *a: None)
    ok, note = s.runtime_status(s.info("whisper-large-v3-mlx"))
    assert not ok and "mlx-whisper" in note
    monkeypatch.setattr("importlib.util.find_spec", lambda name, *a: object())
    assert s.runtime_status(s.info("whisper-large-v3-mlx")) == (True, "")
    intel, _ = mac(tmp_path, machine="x86_64")
    ok, note = service(tmp_path, intel, load_registry()).runtime_status(s.info("whisper-large-v3-mlx"))
    assert not ok and "Apple Silicon" in note


def test_an_installed_model_is_reused_instead_of_a_second_download(tmp_path, registry, monkeypatch):
    monkeypatch.setattr("importlib.util.find_spec", lambda name, *a: object())
    arm, _ = mac(tmp_path)
    s = service(tmp_path, arm, registry)
    assert s.preferred_key() == "whisper-large-v3-mlx"                              # nothing installed: the recommended one
    install_small_copy(s, "whisper-large-v3", {"config.json": b"{}", "model.bin": b"x" * 100, "tokenizer.json": b"{}"})
    assert s.preferred_key() == "whisper-large-v3"                                  # the CTranslate2 copy works: no 3 GB download
    install_small_copy(s, "whisper-large-v3-mlx", {"config.json": b'{"n_mels": 128}', "weights.npz": b"y" * 100})
    assert s.preferred_key() == "whisper-large-v3-mlx"                              # the recommended one is there now


def test_the_pipeline_spec_of_the_mlx_model_points_at_the_installed_folder(tmp_path, registry, monkeypatch):
    monkeypatch.setattr("importlib.util.find_spec", lambda name, *a: object())
    arm, _ = mac(tmp_path)
    s = service(tmp_path, arm, registry)
    folder = install_small_copy(s, "whisper-large-v3-mlx", {"config.json": b"{}", "weights.npz": b"y" * 50})
    spec = s.model_spec("whisper-large-v3-mlx")
    assert spec.engine == "mlx_whisper" and spec.model == str(folder) and spec.key == "whisper-large-v3-mlx"
    from src.asr import create_backend, MacOptimizedBackend

    assert type(create_backend(spec)) is MacOptimizedBackend


def test_model_labels_follow_the_platform(tmp_path, registry):
    from tests.gui_support import make_services

    arm, _ = mac(tmp_path)
    s = service(tmp_path, arm, registry)
    assert s.is_recommended("whisper-large-v3-mlx") and not s.is_recommended("whisper-large-v3")
