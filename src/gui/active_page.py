"""While a transcription runs: progress, chunk, position, elapsed, ETA, RTF, device, VRAM, live log, PAUSE SAFELY."""

from __future__ import annotations

from typing import Any, Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout, QWidget

from .context import AppServices
from .widgets import LogPanel, StatGrid, card, format_clock

STATS = ["Chunk", "Processed", "Elapsed", "ETA", "RTF", "Language", "Subject", "Model", "Device", "GPU", "VRAM"]


class ActivePage(QWidget):
    pause_requested = Signal()
    open_log_requested = Signal()
    back_requested = Signal()
    resume_requested = Signal()

    def __init__(self, services: AppServices, parent=None) -> None:
        super().__init__(parent)
        self.services = services
        self.gpu_name = ""
        self.vram_total_mib = 0
        root = QVBoxLayout(self)
        root.setSpacing(12)
        self.file_label = QLabel("")
        self.file_label.setObjectName("title")
        root.addWidget(self.file_label)
        self.status_label = QLabel("Starting ...")
        self.status_label.setObjectName("statusLine")
        root.addWidget(self.status_label)

        panel = card()
        pl = QVBoxLayout(panel)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setTextVisible(False)
        self.percent_label = QLabel("0%")
        self.percent_label.setObjectName("bigNumber")
        top = QHBoxLayout()
        top.addWidget(QLabel("Progress:"))
        top.addWidget(self.progress_bar, 1)
        top.addWidget(self.percent_label)
        pl.addLayout(top)
        self.stats = StatGrid(STATS, columns=4)
        pl.addWidget(self.stats)
        root.addWidget(panel)

        self.warning_label = QLabel("")
        self.warning_label.setObjectName("bannerWarn")
        self.warning_label.setWordWrap(True)
        self.warning_label.hide()
        root.addWidget(self.warning_label)

        self.log = LogPanel()
        root.addWidget(self.log, 1)

        buttons = QHBoxLayout()
        self.pause_button = QPushButton("PAUSE SAFELY")
        self.pause_button.setObjectName("danger")
        self.pause_button.clicked.connect(self._pause_clicked)
        self.log_button = QPushButton("OPEN LOG")
        self.log_button.clicked.connect(self.open_log_requested.emit)
        self.resume_button = QPushButton("RESUME")
        self.resume_button.setObjectName("primary")
        self.resume_button.clicked.connect(self.resume_requested.emit)
        self.back_button = QPushButton("Back to start")
        self.back_button.clicked.connect(self.back_requested.emit)
        for b in (self.pause_button, self.resume_button, self.log_button, self.back_button):
            buttons.addWidget(b)
        buttons.addStretch(1)
        root.addLayout(buttons)

    # ------------------------------------------------------------------ lifecycle
    def begin(self, file_name: str, facts: dict[str, Any], gpu_name: str = "", vram_total_mib: int = 0) -> None:
        self.gpu_name, self.vram_total_mib = gpu_name, vram_total_mib
        self.file_label.setText(file_name)
        self.status_label.setText("Starting ...")
        self.progress_bar.setValue(0)
        self.percent_label.setText("0%")
        self.log.clear()
        self.warning_label.hide()
        for name in STATS:
            self.stats.set(name, "-")
        self.stats.set("Model", str(facts.get("model", "-")))
        self.stats.set("Language", str(facts.get("language", "-")))
        self.stats.set("Subject", str(facts.get("subject", "-")))
        self.stats.set("GPU", gpu_name or "none (CPU)")
        self.pause_button.setEnabled(True)
        self.pause_button.show()
        self.resume_button.hide()
        self.resume_button.setEnabled(True)
        self.back_button.hide()

    def update_facts(self, facts: dict[str, Any]) -> None:
        self.stats.set("Language", str(facts.get("language", "-")))
        self.stats.set("Subject", str(facts.get("subject", "-")))
        self.stats.set("Model", str(facts.get("model", "-")))
        device = str(facts.get("device", "")).upper()
        compute = facts.get("compute_type", "")
        self.stats.set("Device", (device + (f" / {compute}" if compute else "")) or "-")

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def set_progress(self, event: dict[str, Any]) -> None:
        percent = float(event.get("percent", 0.0))
        self.progress_bar.setValue(int(min(percent, 100.0) * 10))
        self.percent_label.setText(f"{percent:.0f}%")
        self.stats.set("Chunk", f"{event.get('chunk', 0)} / {event.get('chunks', 0)}")
        self.stats.set("Processed", f"{format_clock(event.get('processed'))} / {format_clock(event.get('total'))}")
        self.stats.set("Elapsed", format_clock(event.get("elapsed")))
        rtf = float(event.get("rtf") or 0.0)
        eta = event.get("eta")
        self.stats.set("ETA", format_clock(eta) if rtf > 0 and eta is not None else "calculating ...")
        self.stats.set("RTF", f"{rtf:.2f}" if rtf > 0 else "-")
        used = event.get("gpu_dedicated_mib")
        if used is not None:
            total = f" / {self.vram_total_mib / 1024:.1f} GB" if self.vram_total_mib else ""
            shared = event.get("gpu_shared_mib") or 0
            self.stats.set("VRAM", f"{used / 1024:.1f}{total}" + (f"  (+{shared / 1024:.1f} GB shared)" if shared >= 512 else ""))
        elif not self.gpu_name:
            self.stats.set("VRAM", "n/a (CPU)")

    def show_warning(self, text: str) -> None:
        if text:
            self.warning_label.setText(text)
            self.warning_label.show()

    def _pause_clicked(self) -> None:
        self.pause_button.setEnabled(False)
        self.pause_button.setText("PAUSING ...")
        self.pause_requested.emit()

    def finished_paused(self) -> None:
        self.pause_button.hide()
        self.pause_button.setEnabled(True)
        self.pause_button.setText("PAUSE SAFELY")
        self.resume_button.show()
        self.resume_button.setEnabled(False)           # enabled by the window once the worker process has really ended
        self.back_button.show()
        self.status_label.setText("Paused safely. Your progress is saved - press RESUME to continue.")

    def finished_failed(self, message: str) -> None:
        self.pause_button.hide()
        self.resume_button.show()
        self.resume_button.setEnabled(False)
        self.back_button.show()
        self.status_label.setText("The transcription stopped.")
        self.show_warning(message)
