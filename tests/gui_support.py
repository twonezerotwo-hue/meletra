"""Helpers for the offscreen GUI tests: a tiny fake model, fake GPU, scripted worker, and an event-loop wait."""

from __future__ import annotations

import hashlib
import json
import random
import time
import types
from pathlib import Path
from typing import Any, Callable, Optional

from PySide6.QtCore import QMimeData, QPointF, QThread, QUrl, Qt
from PySide6.QtGui import QDropEvent
from PySide6.QtWidgets import QApplication

from src.services.credential_service import CredentialService, MemoryCredentialBackend
from src.services.gpu_service import GpuService
from src.services.model_service import ModelFile, ModelInfo, ModelService
from src.services.paths import AppPaths
from src.services.transcription_service import JobRequest, events_path
from src.gui.context import AppServices

SMI = {"name": "NVIDIA GeForce RTX 2070", "total_mib": 8192, "used_mib": 1000, "free_mib": 7000, "driver": "617.14"}
MODEL_KEY = "whisper-large-v3"


def fake_files() -> dict[str, bytes]:
    rng = random.Random(3)
    return {"config.json": b'{"model": "tiny"}', "model.bin": bytes(rng.getrandbits(8) for _ in range(50_000)),
            "tokenizer.json": b'{"vocab": [1]}'}


def fake_info(files: Optional[dict[str, bytes]] = None, key: str = MODEL_KEY, recommended: bool = True) -> ModelInfo:
    files = files or fake_files()
    return ModelInfo(
        key=key, title="Whisper Large-v3", repo="acme/model", revision="a" * 40, engine="faster_whisper", models_yaml_key="tiny_test",
        folder=f"{key}-folder", description="Test model", runtime="faster-whisper", required=tuple(files),
        approx_bytes=sum(map(len, files.values())), recommended=recommended,
        files=tuple(ModelFile(n, len(b), hashlib.sha256(b).hexdigest()) for n, b in files.items()))


def install_fake_model(models: ModelService, key: str = MODEL_KEY) -> Path:
    info = models.info(key)
    folder = models.paths.models / info.folder
    folder.mkdir(parents=True, exist_ok=True)
    files = fake_files()
    for name, data in files.items():
        (folder / name).write_bytes(data)
    return folder


def make_gpu(smi: Optional[dict] = SMI, cuda: tuple[bool, str] = (True, "CUDA acceleration is available."), programs=()) -> GpuService:
    return GpuService(smi=lambda i: smi, cuda=lambda i: cuda, processes=lambda: list(programs))


def make_services(tmp_path: Path, installed: bool = True, gpu: Optional[GpuService] = None, opened: Optional[list] = None
                  ) -> AppServices:
    data = tmp_path / "data"
    paths = AppPaths(data).ensure()
    models = ModelService(paths, registry={MODEL_KEY: fake_info()}, app_dir=tmp_path / "appdir", sleep=lambda s: None, retries=1)
    services = AppServices.create(env={"MLT_DATA_DIR": str(data)}, app_dir=tmp_path / "appdir",
                                  credentials=CredentialService(MemoryCredentialBackend()), gpu=gpu or make_gpu(), models=models,
                                  open_path=(lambda p: opened.append(Path(p))) if opened is not None else (lambda p: None))
    if installed:
        install_fake_model(models)
    return services


def wait_for(predicate: Callable[[], Any], timeout: float = 15.0, what: str = "condition") -> Any:
    """Run the Qt event loop until ``predicate()`` is truthy. Fails (never hangs) after ``timeout`` seconds."""
    app = QApplication.instance()
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app.processEvents()
        value = predicate()
        if value:
            return value
        time.sleep(0.01)
    raise AssertionError(f"timed out after {timeout:.0f} s waiting for {what}")


def settle(*widgets) -> None:
    """Let background threads of the given widgets finish before the widget is destroyed."""
    app = QApplication.instance()
    for widget in widgets:
        for thread in widget.findChildren(QThread):
            if thread.isRunning():
                thread.wait(8000)
    app.processEvents()


def drop_event(*paths: str) -> QDropEvent:
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(p) for p in paths])
    event = QDropEvent(QPointF(5, 5), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
    event.keep_alive = mime                  # QDropEvent does not own the mime data: without this it is freed under the event
    return event


# ----------------------------------------------------------------------------------------------- scripted worker
class ScriptedHandle:
    def __init__(self) -> None:
        self.code: Optional[int] = None
        self.killed = False

    def running(self) -> bool:
        return self.code is None and not self.killed

    def exit_code(self) -> Optional[int]:
        return None if self.running() else (-9 if self.killed else self.code)

    def terminate(self) -> None:
        self.killed = True


class ScriptedLauncher:
    """Stands in for the worker process: writes the events a script says, finishes when the test says so."""

    def __init__(self, script: Optional[list[dict]] = None) -> None:
        self.script = script or []
        self.handle: Optional[ScriptedHandle] = None
        self.requests: list[JobRequest] = []
        self.events_file: Optional[Path] = None

    def launch(self, request_file: Path) -> ScriptedHandle:
        request = JobRequest.load(request_file)
        self.requests.append(request)
        self.events_file = events_path(types.SimpleNamespace(jobs=Path(request.data_root) / "jobs"), request.job_id)
        self.emit(*self.script)
        self.handle = ScriptedHandle()
        return self.handle

    def emit(self, *events: dict) -> None:
        with open(self.events_file, "a", encoding="utf-8") as f:
            for event in events:
                f.write(json.dumps(event) + "\n")

    def finish(self, code: int) -> None:
        self.handle.code = code


def progress_event(percent: float, chunk: int = 1, chunks: int = 4, **extra) -> dict:
    return {"type": "chunk_progress", "percent": percent, "chunk": chunk, "chunks": chunks, "processed": percent * 10.0,
            "total": 1000.0, "elapsed": 30.0, "eta": 60.0, "rtf": 0.25, "gpu_dedicated_mib": 4300, "gpu_shared_mib": 50, **extra}
