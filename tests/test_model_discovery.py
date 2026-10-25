"""v3 zero-config model discovery: a complete Whisper copy in a Hugging Face cache is used where it is (no folder picker, no copy,
no second download); incomplete or foreign copies are ignored; nothing changes unless discovery is switched on."""

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from src.gui.first_run import FirstRunDialog
from src.platform.windows import WindowsPlatform
from src.services.model_service import ModelFile, ModelService, load_registry
from src.services.paths import AppPaths
from tests.gui_support import settle
from tests.test_platform import mac

FILES = {"config.json": b'{"alignment_heads": []}', "model.bin": b"weights" * 50, "tokenizer.json": b"{}"}
KEY = "whisper-large-v3"


def small_registry():
    registry = load_registry()
    info = registry[KEY]
    files = tuple(ModelFile(n, len(d), hashlib.sha256(d).hexdigest()) for n, d in FILES.items())
    registry[KEY] = replace(info, files=files, required=tuple(FILES), approx_bytes=sum(map(len, FILES.values())))
    return registry


def make(tmp_path, platform=None, discover=True):
    registry = small_registry()
    env = {}
    platform = platform or WindowsPlatform(env=env, home=tmp_path / "home")
    return ModelService(AppPaths(tmp_path / "data").ensure(), registry=registry, app_dir=tmp_path / "app", platform=platform,
                        discover=discover), registry[KEY]


def snapshot(root: Path, info, revision=None, files=FILES) -> Path:
    folder = root / ("models--" + info.repo.replace("/", "--")) / "snapshots" / (revision or info.revision)
    folder.mkdir(parents=True)
    for name, data in files.items():
        (folder / name).write_bytes(data)
    return folder


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    for name in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        monkeypatch.delenv(name, raising=False)


def test_default_hf_cache_copy_is_used_in_place(tmp_path):
    service, info = make(tmp_path)
    assert not service.is_installed(KEY)
    snap = snapshot(tmp_path / "home" / ".cache" / "huggingface" / "hub", info)
    assert service.locate(KEY) == snap and service.state(KEY).installed and service.state(KEY).source == "cache"
    assert service.model_spec(KEY).model == str(snap)                       # the pipeline loads it from the cache
    assert not any(service.paths.models.iterdir())                           # nothing was copied or downloaded


@pytest.mark.parametrize("variable, sub", [("HF_HOME", "hub"), ("HF_HOME", ""), ("HUGGINGFACE_HUB_CACHE", ""), ("HF_HUB_CACHE", ""),
                                           ("TRANSFORMERS_CACHE", "")])
def test_environment_cache_locations_are_searched(tmp_path, variable, sub):
    env = {variable: str(tmp_path / "elsewhere")}
    service, info = make(tmp_path, WindowsPlatform(env=env, home=tmp_path / "home"))
    snap = snapshot(tmp_path / "elsewhere" / sub, info)
    assert service.locate(KEY) == snap


def test_macos_library_caches_is_searched(tmp_path):
    platform, _ = mac(tmp_path)
    service, info = make(tmp_path, platform)
    snap = snapshot(tmp_path / "home" / "Library" / "Caches", info)
    assert service.locate(KEY) == snap


def test_the_pinned_revision_wins_and_broken_copies_are_skipped(tmp_path):
    service, info = make(tmp_path)
    root = tmp_path / "home" / ".cache" / "huggingface" / "hub"
    snapshot(root, info, "aaaa", {**FILES, "model.bin": b"short"})              # wrong size
    snapshot(root, info, "bbbb", {**FILES, "config.json": b"not json at all......"[:len(FILES["config.json"])]})
    assert service.locate(KEY) is None
    other = snapshot(root, info, "cccc")
    pinned = snapshot(root, info, info.revision)
    assert service.locate(KEY) == pinned and other.is_dir()


def test_nothing_is_discovered_unless_switched_on(tmp_path):
    service, info = make(tmp_path, discover=False)
    snapshot(tmp_path / "home" / ".cache" / "huggingface" / "hub", info)
    assert service.locate(KEY) is None


def test_an_app_managed_copy_still_wins_over_the_cache(tmp_path):
    service, info = make(tmp_path)
    snapshot(tmp_path / "home" / ".cache" / "huggingface" / "hub", info)
    own = service.paths.models / info.folder
    own.mkdir(parents=True)
    for name, data in FILES.items():
        (own / name).write_bytes(data)
    assert service.locate(KEY) == own


def test_first_run_says_ready_and_asks_for_nothing(qapp, tmp_path):
    from tests.gui_support import make_services
    services = make_services(tmp_path, installed=False)
    service, info = make(tmp_path)
    snapshot(tmp_path / "home" / ".cache" / "huggingface" / "hub", info)
    services.models = service
    services.settings.update(model_key=KEY)
    dialog = FirstRunDialog(services)
    assert "is ready" in dialog.model_label.text() and dialog.done_button.text() == "Continue"
    assert not dialog.done_button.isHidden() and dialog.download_button.isHidden() and dialog.import_button.isHidden()
    settle(dialog)
