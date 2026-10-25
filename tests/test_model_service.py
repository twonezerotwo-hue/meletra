"""Model manager: registry, resumable download, integrity, verify, delete, import - against a REAL local HTTP server
(with Range support and injected failures), so the download code path is exercised end to end without the internet."""

import hashlib
import http.server
import os
import random
import re
import shutil
import threading
from pathlib import Path

import pytest

from src.services import model_service as ms
from src.services.model_service import (
    ModelCancelled, ModelDownloadError, ModelError, ModelFile, ModelInfo, ModelIntegrityError, ModelNotInstalled,
    ModelService, load_registry,
)
from src.services.paths import AppPaths

REPO, REVISION = "acme/tiny-model", "deadbeef" * 5


def make_files() -> dict[str, bytes]:
    rng = random.Random(7)
    return {"config.json": b'{"model": "tiny"}', "model.bin": bytes(rng.getrandbits(8) for _ in range(2_600_000)),
            "tokenizer.json": b'{"vocab": [1, 2, 3]}'}


def make_info(files: dict[str, bytes], engine: str = "faster_whisper", key: str = "tiny") -> ModelInfo:
    return ModelInfo(
        key=key, title="Tiny test model", repo=REPO, revision=REVISION, engine=engine, models_yaml_key="tiny_test",
        folder="tiny-test", description="", runtime="", required=tuple(files), approx_bytes=sum(map(len, files.values())),
        files=tuple(ModelFile(n, len(b), hashlib.sha256(b).hexdigest()) for n, b in files.items()))


class Handler(http.server.BaseHTTPRequestHandler):
    files: dict[str, bytes] = {}
    requests: list[tuple[str, str]] = []
    drop_after: dict[str, int] = {}            # file -> bytes to send before cutting the connection (first request only)
    corrupt: set = set()
    ignore_range = False
    missing: set = set()

    def log_message(self, *args):
        pass

    def do_GET(self):
        match = re.fullmatch(rf"/{REPO}/resolve/{REVISION}/(.+)", self.path)
        name = match.group(1) if match else ""
        Handler.requests.append((name, self.headers.get("Range", "")))
        if name in Handler.missing or name not in Handler.files:
            self.send_error(404)
            return
        body = Handler.files[name]
        if name in Handler.corrupt:
            body = bytes([body[0] ^ 0xFF]) + body[1:]
        start, status = 0, 200
        spec = self.headers.get("Range")
        if spec and not Handler.ignore_range:
            start, status = int(re.match(r"bytes=(\d+)-", spec).group(1)), 206
        data = body[start:]
        self.send_response(status)
        self.send_header("Content-Length", str(len(data)))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{len(body) - 1}/{len(body)}")
        self.end_headers()
        limit = Handler.drop_after.pop(name, None)
        if limit is not None:
            self.wfile.write(data[:limit])
            self.wfile.flush()
            self.connection.close()
            return
        self.wfile.write(data)


@pytest.fixture
def server():
    Handler.files, Handler.requests, Handler.drop_after = make_files(), [], {}
    Handler.corrupt, Handler.ignore_range, Handler.missing = set(), False, set()
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture
def service(tmp_path, server):
    info = make_info(Handler.files)
    paths = AppPaths(tmp_path / "data").ensure()
    return ModelService(paths, registry={"tiny": info}, base_url=server, sleep=lambda s: None, retries=4,
                        app_dir=tmp_path / "app")


# ----------------------------------------------------------------------------------------------- registry
def test_shipped_registry_pins_revisions_and_checksums():
    registry = load_registry()
    assert {"whisper-large-v3", "turkish-medical-large-v3", "whisper-tiny-test"} <= set(registry)
    assert registry["translation-m2m100"].kind == "translation" and registry["translation-m2m100"].license == "MIT"
    for info in registry.values():
        if info.source_url:                                                            # hosted as a release asset: the revision is the release tag, the files are pinned by size and SHA-256
            assert re.fullmatch(r"[a-z0-9][a-z0-9.\-]+", info.revision) and "{path}" in info.source_url and info.files
        else:
            assert re.fullmatch(r"[0-9a-f]{40}", info.revision) and info.files
        for f in info.files:
            assert re.fullmatch(r"[0-9a-f]{64}", f.sha256) and f.size > 0
    large = registry["whisper-large-v3"]
    assert large.recommended and large.engine == "faster_whisper" and large.file("model.bin").size > 3_000_000_000
    assert registry["turkish-medical-large-v3"].engine == "transformers" and not registry["turkish-medical-large-v3"].recommended


@pytest.mark.windows_platform
def test_hidden_models_are_not_listed_to_users(tmp_path):
    service = ModelService(AppPaths(tmp_path).ensure())
    shown = {s.key for s in service.list_models()}
    assert shown == {"whisper-large-v3", "turkish-medical-large-v3"} and "whisper-tiny-test" in {
        s.key for s in service.list_models(include_hidden=True)}
    assert service.recommended_key() == "whisper-large-v3"


# ----------------------------------------------------------------------------------------------- download
def test_download_installs_verifies_and_reports_progress(service):
    events = []
    folder = service.download("tiny", progress=events.append)
    assert folder == service.paths.models / "tiny-test" and service.is_installed("tiny")
    state = service.state("tiny")
    assert state.installed and state.verified is True and state.size_on_disk >= 2_600_000 and state.source == "app"
    assert not (service.paths.models / ".partial" / "tiny-test").exists()
    percents = [e["percent"] for e in events if e["type"] == "model_download_progress" and "message" not in e]
    assert percents == sorted(percents) and percents[-1] == pytest.approx(100.0)
    last = events[-1]
    assert last["bytes_done"] == last["bytes_total"] and last["files"] == 3
    assert any(e.get("speed_bps") for e in events)


def test_an_interrupted_connection_resumes_where_it_stopped(service):
    Handler.drop_after = {"model.bin": 1_200_000}
    service.download("tiny")
    assert service.is_installed("tiny")
    ranges = [r for name, r in Handler.requests if name == "model.bin"]
    assert ranges[0] == "" and re.fullmatch(r"bytes=1[0-9]{6}-", ranges[1])          # second request continues at ~1.2 MB
    sent = sum(1 for name, _ in Handler.requests if name == "model.bin")
    assert sent == 2


def test_cancel_keeps_the_partial_file_and_the_next_call_resumes(service):
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] > 4                                                          # after ~2 MB of the 2.6 MB file

    with pytest.raises(ModelCancelled):
        service.download("tiny", cancel=cancel)
    assert not service.is_installed("tiny") and service.partial_bytes("tiny") > 1_000_000
    assert service.state("tiny").partial_bytes > 0
    Handler.requests.clear()
    service.download("tiny")
    assert service.is_installed("tiny")
    resumed = [r for name, r in Handler.requests if name == "model.bin"]
    assert resumed and resumed[0].startswith("bytes=") and resumed[0] != "bytes=0-"


def test_a_server_that_ignores_range_still_gives_a_correct_file(service):
    Handler.drop_after = {"model.bin": 800_000}
    Handler.ignore_range = True
    service.download("tiny")
    assert service.verify("tiny").ok


def test_damaged_data_is_rejected_and_nothing_is_installed(service):
    Handler.corrupt = {"model.bin"}
    with pytest.raises(ModelIntegrityError, match="checksum"):
        service.download("tiny")
    assert not service.is_installed("tiny")
    assert not (service.paths.models / "tiny-test" / "model.bin").exists()


def test_a_missing_file_on_the_server_is_a_clear_error(service):
    Handler.missing = {"tokenizer.json"}
    with pytest.raises(ModelDownloadError, match="HTTP 404"):
        service.download("tiny")


def test_no_internet_gives_a_resumable_message_not_a_traceback(tmp_path):
    info = make_info(make_files())
    service = ModelService(AppPaths(tmp_path / "d").ensure(), registry={"tiny": info}, base_url="http://127.0.0.1:9",
                           sleep=lambda s: None, retries=2)
    with pytest.raises(ModelDownloadError, match="resume"):
        service.download("tiny")
    assert not service.is_installed("tiny")


def test_not_enough_disk_space_is_reported_before_downloading(service, monkeypatch):
    monkeypatch.setattr(ms.shutil, "disk_usage", lambda p: shutil._ntuple_diskusage(100, 100, 1_000))
    with pytest.raises(ModelDownloadError, match="free disk space"):
        service.download("tiny")
    assert Handler.requests == []


def test_download_of_an_installed_model_does_nothing(service):
    service.download("tiny")
    Handler.requests.clear()
    service.download("tiny")
    assert Handler.requests == []


# ----------------------------------------------------------------------------------------------- verify / delete
def test_verify_ok_then_detects_tampering_and_missing_files(service):
    service.download("tiny")
    assert service.verify("tiny").ok
    folder = service.locate("tiny")
    data = bytearray((folder / "model.bin").read_bytes())
    data[100] ^= 1
    (folder / "model.bin").write_bytes(bytes(data))
    result = service.verify("tiny")
    assert not result.ok and "model.bin does not match its checksum" in result.problems
    assert service.state("tiny").verified is False
    (folder / "model.bin").unlink()
    assert not service.is_installed("tiny")                                            # a missing required file = not installed
    assert not service.verify("tiny").ok


def test_delete_removes_the_model_and_any_partial_download(service):
    Handler.drop_after = {"model.bin": 500_000}
    service.retries = 1
    with pytest.raises(ModelDownloadError):
        service.download("tiny")
    assert service.partial_bytes("tiny") > 0
    assert service.delete("tiny") and service.partial_bytes("tiny") == 0
    service.download("tiny")
    assert service.delete("tiny") and not service.is_installed("tiny")
    assert not service.delete("tiny")


def test_delete_never_touches_a_model_outside_the_app_folder(service, tmp_path):
    extra = tmp_path / "usb" / "tiny-test"
    extra.mkdir(parents=True)
    for name, data in Handler.files.items():
        (extra / name).write_bytes(data)
    service.extra_dirs = [str(tmp_path / "usb")]
    assert service.is_installed("tiny") and service.state("tiny").source == "extra"
    assert not service.delete("tiny")
    assert (extra / "model.bin").is_file()


def test_models_in_the_app_folder_count_for_portable_and_offline_packages(service, tmp_path):
    folder = tmp_path / "app" / "models" / "tiny-test"
    folder.mkdir(parents=True)
    for name, data in Handler.files.items():
        (folder / name).write_bytes(data)
    assert service.locate("tiny") == folder and service.state("tiny").source == "portable"


# ----------------------------------------------------------------------------------------------- import
def test_import_folder_copies_and_verifies(service, tmp_path):
    src = tmp_path / "usb" / "whatever"
    src.mkdir(parents=True)
    for name, data in Handler.files.items():
        (src / name).write_bytes(data)
    events = []
    folder, result = service.import_folder("tiny", tmp_path / "usb", progress=events.append)          # parent of the model folder
    assert result.ok and folder == service.paths.models / "tiny-test" and service.is_installed("tiny")
    assert (src / "model.bin").is_file(), "the source is copied, never moved"
    assert events and events[-1]["percent"] == pytest.approx(100.0)
    assert service.state("tiny").verified is True


def test_import_of_the_wrong_folder_is_a_clear_error(service, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ModelError, match="does not contain Tiny test model"):
        service.import_folder("tiny", empty)
    with pytest.raises(ModelError, match="not a folder"):
        service.import_folder("tiny", tmp_path / "nope")


def test_import_of_same_sized_but_different_files_is_flagged_not_trusted(service, tmp_path):
    src = tmp_path / "other"
    src.mkdir()
    for name, data in Handler.files.items():
        (src / name).write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    _, result = service.import_folder("tiny", src)
    assert not result.ok and service.state("tiny").verified is False


def test_hf_cache_copy_is_found_when_present(service, tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    assert service.find_hf_cache_copy("tiny") is None
    snap = tmp_path / "home" / ".cache" / "huggingface" / "hub" / "models--acme--tiny-model" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    for name, data in Handler.files.items():
        (snap / name).write_bytes(data)
    assert service.find_hf_cache_copy("tiny") == snap


# ----------------------------------------------------------------------------------------------- pipeline spec / runtime / disk
def test_model_spec_points_at_the_local_folder_and_keeps_the_yaml_options(service):
    with pytest.raises(ModelNotInstalled, match="Settings -> Models"):
        service.model_spec("tiny")
    service.download("tiny")
    spec = service.model_spec("tiny")
    assert spec.key == "tiny" and Path(spec.model) == service.paths.models / "tiny-test" and spec.engine == "faster_whisper"


def test_a_transformers_model_without_the_runtime_explains_itself(tmp_path, monkeypatch):
    info = make_info(make_files(), engine="transformers", key="tr")
    service = ModelService(AppPaths(tmp_path / "d").ensure(), registry={"tr": info})
    monkeypatch.setattr(ms.importlib.util, "find_spec", lambda name: None)
    ok, note = service.runtime_status(info)
    assert not ok and "optional" in note
    assert service.state("tr").runtime_ok is False


def test_disk_usage_summary(service):
    service.download("tiny")
    usage = service.disk_usage()
    assert usage["per_model"]["tiny"] >= 2_600_000 and usage["models_bytes"] >= usage["per_model"]["tiny"]
    assert usage["free_bytes"] > 0 and usage["partial_bytes"] == 0


def test_human_bytes():
    assert ms.human_bytes(512) == "512 B" and ms.human_bytes(1536) == "1.5 KB" and ms.human_bytes(3 * 1024 ** 3) == "3.0 GB"


def test_the_translation_pack_is_never_offered_as_a_speech_model(tmp_path):
    paths = AppPaths(tmp_path / "data").ensure()
    service = ModelService(paths, app_dir=tmp_path / "app")
    assert "translation-m2m100" not in [s.key for s in service.list_models()]                                    # every speech-model picker uses the default
    assert "translation-m2m100" in [s.key for s in service.list_models(kind=None)]                                # Settings -> Models shows both
    assert service.registry[service.recommended_key()].kind == "asr" and service.registry[service.preferred_key()].kind == "asr"
    info = service.info("translation-m2m100")
    assert service._url(info, "model.bin") == "https://github.com/twonezerotwo-hue/whispermedical-downloads/releases/download/translation-m2m100-v1/model.bin"
    assert not service.is_installed("translation-m2m100")
