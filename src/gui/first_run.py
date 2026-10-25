"""First-run wizard: check the computer, then get the speech model (download, import, or skip)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QDialog, QFileDialog, QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout, QWidget,
)

from ..services import diagnostics_service
from ..services.model_service import human_bytes
from . import brand, tokens
from .context import AppServices
from .tasks import TaskThread, stop_threads
from .widgets import card

log = logging.getLogger(__name__)

MIN_RAM_GB = 8.0
MIN_DISK_GB = 8.0


def system_check(services: AppServices) -> list[tuple[str, str, str]]:
    """``(item, text, level)`` rows for the wizard; level is ``ok`` | ``warn`` | ``bad``. Never raises."""
    data = diagnostics_service.collect(services.paths, services.models, services.gpu, check_ffmpeg_version=False)
    if (data.get("platform") or {}).get("name") == "macos":
        return _mac_system_check(services, data)
    rows: list[tuple[str, str, str]] = [("Windows", data["windows"], "ok")]
    cpu = data["cpu"]
    rows.append(("CPU", f"{cpu['name']} ({cpu['logical_cores']} logical cores)", "ok"))
    ram = data["ram"]
    total = float(ram.get("total_gb", 0.0) or 0.0)
    if total and total < MIN_RAM_GB:
        rows.append(("Memory", f"{total} GB RAM - 8 GB or more is recommended; long lectures may be slow", "warn"))
    else:
        rows.append(("Memory", f"{total} GB RAM ({ram.get('free_gb', '?')} GB free)", "ok"))
    gpu = data["gpu"]
    if gpu["name"]:
        rows.append(("GPU", f"{gpu['name']}  -  {gpu['total_mib'] / 1024:.1f} GB video memory", "ok"))
    else:
        rows.append(("GPU", "No NVIDIA GPU found - the CPU will be used (slower, but it works)", "warn"))
    cuda = data["cuda"]
    if cuda["available"]:
        rows.append(("CUDA acceleration", "available", "ok"))
    else:
        rows.append(("CUDA acceleration", f"unavailable - {cuda['reason'] or 'no compatible NVIDIA GPU'}.  CPU fallback enabled.", "warn"))
    free = float(data["disk"]["free_gb"])
    rows.append(("Disk space", f"{free} GB free" + ("" if free >= MIN_DISK_GB else " - the model needs about 3 GB, transcripts a little more"),
                 "ok" if free >= MIN_DISK_GB else "warn"))
    ff = data["ffmpeg"]
    rows.append(("Audio decoder", "FFmpeg (bundled)" if ff["bundled"] else ("FFmpeg (system)" if ff["path"] else "built-in decoder (PyAV)"), "ok"))
    return rows


def _mac_system_check(services: AppServices, data: dict) -> list[tuple[str, str, str]]:
    """The check on a Mac: Apple Silicon or Intel, memory, disk, the speech engine and the model. No NVIDIA / CUDA words."""
    silicon = bool((data.get("platform") or {}).get("apple_silicon"))
    rows: list[tuple[str, str, str]] = [("macOS", data["windows"], "ok")]
    cpu = data["cpu"]
    if silicon:
        rows.append(("Processor", f"{cpu['name']} - Apple Silicon", "ok"))
    else:
        rows.append(("Processor", f"{cpu['name']} - Intel. Speech recognition runs on the CPU and is slower than on Apple Silicon.", "warn"))
    ram = data["ram"]
    total = float(ram.get("total_gb", 0.0) or 0.0)
    if total and total < MIN_RAM_GB:
        rows.append(("Memory", f"{total} GB - 8 GB or more is recommended; long lectures may be slow", "warn"))
    else:
        rows.append(("Memory", f"{total} GB ({ram.get('free_gb', '?')} GB free)", "ok"))
    engines = {e["engine"]: e for e in data.get("engines", [])}
    mlx = engines.get("mlx_whisper", {})
    if silicon and mlx.get("ready") == "yes":
        rows.append(("Speech engine", "Optimized for Apple Silicon (uses this Mac's GPU)", "ok"))
    elif silicon:
        rows.append(("Speech engine", f"The optimized engine is not available ({mlx.get('note', 'unknown reason')}); the standard engine "
                                     "on the CPU will be used (slower, but it works)", "warn"))
    else:
        rows.append(("Speech engine", "Standard engine on the CPU", "ok"))
    free = float(data["disk"]["free_gb"])
    rows.append(("Disk space", f"{free} GB free" + ("" if free >= MIN_DISK_GB else " - the model needs about 3 GB, transcripts a little more"),
                 "ok" if free >= MIN_DISK_GB else "warn"))
    key = services.models.preferred_key()
    state = services.models.state(key)
    damaged = services.models.bundled_problems(key)
    if damaged:
        rows.append(("Speech model", "The model that came with this app is damaged or incomplete - install the app again from the disk image", "bad"))
    else:
        rows.append(("Speech model", f"{services.models.info(key).title} is installed" if state.installed and state.runtime_ok
                     else f"{services.models.info(key).title} is not installed yet - press Download recommended model below",
                     "ok" if state.installed and state.runtime_ok else "warn"))
    ff = data["ffmpeg"]
    rows.append(("Audio decoder", "FFmpeg (bundled)" if ff["bundled"] else ("FFmpeg (system)" if ff["path"] else "built-in decoder (PyAV)"), "ok"))
    return rows


class FirstRunDialog(QDialog):
    def __init__(self, services: AppServices, parent=None) -> None:
        super().__init__(parent)
        self.services = services
        self.setWindowTitle("Welcome - MELETRA")
        self.resize(660, 560)
        self.setModal(True)
        self._threads: list[TaskThread] = []
        self.choose_folder = lambda: QFileDialog.getExistingDirectory(self, "Select the folder that contains the model files", "")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(tokens.SPACE["xl"], tokens.SPACE["xl"], tokens.SPACE["xl"], tokens.SPACE["lg"])
        layout.setSpacing(12)
        head = QHBoxLayout()
        head.setSpacing(16)
        logo = QLabel()
        logo.setPixmap(brand.logo_pixmap(small=True, height=64))
        head.addWidget(logo)
        title = QLabel("Welcome to MELETRA")
        title.setObjectName("title")
        head.addWidget(title, 1)
        layout.addLayout(head)
        intro = QLabel("MELETRA turns lecture recordings into text on THIS computer: your audio, video and files are never uploaded, and nothing is shared "
                       "unless you switch that on in Settings -> Privacy & Data. First, a quick check of your computer:")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.check_card = card()
        self.check_layout = QVBoxLayout(self.check_card)
        self.check_label = QLabel("Checking ...")
        self.check_layout.addWidget(self.check_label)
        layout.addWidget(self.check_card)
        self.model_label = QLabel("")
        self.model_label.setWordWrap(True)
        layout.addWidget(self.model_label)
        row = QHBoxLayout()
        self.download_button = QPushButton("Download recommended model")
        self.download_button.setObjectName("primary")
        self.download_button.clicked.connect(self.start_download)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_download)
        self.cancel_button.hide()
        self.import_button = QPushButton("I already have the model - import ...")
        self.import_button.clicked.connect(self.import_model)
        self.import_button.hide()                    # manual import lives in Settings -> Models; found models are used in place
        row.addWidget(self.download_button)
        row.addWidget(self.cancel_button)
        row.addWidget(self.import_button)
        row.addStretch(1)
        layout.addLayout(row)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.hide()
        self.progress_text = QLabel("")
        self.progress_text.setObjectName("muted")
        self.progress_text.setWordWrap(True)
        layout.addWidget(self.progress)
        layout.addWidget(self.progress_text)
        layout.addStretch(1)
        bottom = QHBoxLayout()
        bottom.addStretch(1)
        self.skip_button = QPushButton("Skip for now")
        self.skip_button.clicked.connect(self.finish)
        self.done_button = QPushButton("Continue")
        self.done_button.setObjectName("primary")
        self.done_button.clicked.connect(self.finish)
        self.done_button.hide()
        bottom.addWidget(self.skip_button)
        bottom.addWidget(self.done_button)
        layout.addLayout(bottom)

        self.rows: list[tuple[str, str, str]] = []
        self._start_check()
        self.refresh_model()

    # ------------------------------------------------------------------ system check
    def _start_check(self) -> None:
        thread = TaskThread(lambda progress, cancelled: system_check(self.services), self)
        thread.succeeded.connect(self._show_rows)
        thread.failed.connect(lambda msg: self.check_label.setText(f"The system check could not finish: {msg}"))
        self._threads.append(thread)
        self.check_thread = thread
        thread.start()

    def _show_rows(self, rows: list[tuple[str, str, str]]) -> None:
        self.rows = rows
        marks = {"ok": "OK  ", "warn": "NOTE", "bad": "FAIL"}
        self.check_label.setText("\n".join(f"[{marks[level]}]  {name}:  {text}" for name, text, level in rows))

    # ------------------------------------------------------------------ model
    def _key(self) -> str:
        return self.services.models.preferred_key()           # the recommended model - or one that is already installed

    def refresh_model(self) -> None:
        models = self.services.models
        key = self._key()
        info = models.info(key)
        state = models.state(key)
        if state.installed and state.runtime_ok:
            where = f" ({state.path})" if state.path else ""
            self.model_label.setText(f"{info.title} is ready.\nIt is installed{where} and is used where it is: nothing is copied or downloaded.")
            self.download_button.hide()
            self.import_button.hide()
            self.skip_button.hide()
            self.done_button.show()
            return
        damaged = models.bundled_problems(key)
        if damaged:
            self.model_label.setText(f"The speech model that came with this app is damaged or incomplete ({'; '.join(damaged[:3])}).\n"
                                     "Nothing is downloaded: drag the app from the disk image into Applications again.")
            self.download_button.hide()
            self.import_button.hide()
            self.skip_button.show()
            self.done_button.hide()
            return
        extra = ""
        resume = f"  A partial download of {human_bytes(state.partial_bytes)} will be resumed." if state.partial_bytes else ""
        self.model_label.setText(f"The speech model is needed once: {info.title}, about {info.approx_bytes / 1024 ** 3:.1f} GB. "
                                 f"It is stored on this computer and works offline afterwards.{resume}{extra}")
        self.download_button.setText("Resume download" if state.partial_bytes else "Download recommended model")
        self.download_button.setEnabled(True)
        self.download_button.show()
        self.skip_button.show()
        self.done_button.hide()

    def _busy(self, busy: bool) -> None:
        self.download_button.setEnabled(not busy)
        self.import_button.setEnabled(not busy)
        self.cancel_button.setVisible(busy)
        self.progress.setVisible(busy)
        if busy:
            self.progress.setValue(0)

    def start_download(self) -> None:
        if any(t.isRunning() for t in self._threads if t is not getattr(self, "check_thread", None)):
            return
        key = self._key()
        self._run(lambda progress, cancelled: self.services.models.download(key, progress, cancelled))

    def import_model(self) -> None:
        folder = self.choose_folder()
        if folder:
            key = self._key()
            self._run(lambda progress, cancelled: self.services.models.import_folder(key, Path(folder), progress))

    def _run(self, fn) -> None:
        self._busy(True)
        thread = TaskThread(fn, self)
        thread.progressed.connect(self._progress)
        thread.succeeded.connect(lambda _r: self._finished(None))
        thread.failed.connect(self._finished)
        self._threads.append(thread)
        self.task = thread
        thread.start()

    def _progress(self, event: dict) -> None:
        if event.get("message"):
            self.progress_text.setText(event["message"])
            return
        percent = float(event.get("percent", 0.0))
        self.progress.setValue(int(percent * 10))
        eta = event.get("eta_seconds")
        self.progress_text.setText(f"{percent:.0f}%   {human_bytes(event.get('bytes_done', 0))} / {human_bytes(event.get('bytes_total', 0))}"
                                   + (f"   about {int(eta // 60)} min left" if eta else ""))

    def cancel_download(self) -> None:
        task = getattr(self, "task", None)
        if task is not None:
            task.cancel()

    def _finished(self, error: Optional[str]) -> None:
        self._busy(False)
        self.progress_text.setText(error or "")
        self.refresh_model()

    def finish(self) -> None:
        self.services.update_setting(first_run_done=True)
        self.accept()

    def reject(self) -> None:                       # closing the window also counts as "skip"
        self.services.update_setting(first_run_done=True)
        super().reject()

    def done(self, result: int) -> None:
        stop_threads(self)                           # never destroy the dialog while a download thread is still running
        super().done(result)


def needs_first_run(services: AppServices) -> bool:
    return not services.settings.settings.first_run_done
