"""Start page: choose a lecture, check the options, press START TRANSCRIPTION."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QFormLayout, QFrame, QHBoxLayout, QLabel, QMessageBox, QProgressBar, QPushButton, QVBoxLayout,
    QWidget,
)

from ..audio import probe_media_pyav
from ..pipeline import job_key
from ..services.job_service import JobRecord
from ..services.model_service import ModelError, human_bytes
from ..services.paths import default_output_dir
from ..services.transcription_service import JobRequest
from .choices import LANGUAGE_CHOICES, MODE_CHOICES, index_of, subject_choices
from .context import AppServices
from .tasks import TaskThread
from .widgets import DropZone, card, format_hms, format_size

log = logging.getLogger(__name__)

PRIVACY_LOCAL = "LOCAL PROCESSING  -  Audio and files stay on this computer. Sharing of completed transcripts: Settings -> Privacy & Data."
PRIVACY_CLOUD = "CLOUD COMPARISON ON  -  Audio is uploaded to Deepgram."


def default_confirm(parent: QWidget, title: str, text: str, ok_text: str) -> bool:
    box = QMessageBox(QMessageBox.Warning, title, text, QMessageBox.NoButton, parent)
    ok = box.addButton(ok_text, QMessageBox.AcceptRole)
    box.addButton("Cancel", QMessageBox.RejectRole)
    box.exec()
    return box.clickedButton() is ok


GPU, CPU, CANCEL = "gpu", "cpu", "cancel"


def recommended_device(check) -> str:
    """The safe default when the GPU is short of memory: with enough left for the smaller compute type ("warning") the GPU is still the faster choice; with less than that
    ("critical") the driver would swap video memory to system RAM - slower than the CPU and sometimes stuck - so the CPU is recommended."""
    return CPU if getattr(check, "severity", "") == "critical" else GPU


def default_vram_choice(parent: QWidget, check) -> str:
    """GPU memory low: Continue on GPU / Use CPU / Cancel, the recommended one is the default button. Returns GPU, CPU or CANCEL."""
    box = QMessageBox(QMessageBox.Warning, "GPU memory low", check.message + "\n\nRecommended: " +
                      ("use the CPU (slower, but safe with this little GPU memory)." if recommended_device(check) == CPU else "continue on the GPU (it may be somewhat slower)."),
                      QMessageBox.NoButton, parent)
    gpu = box.addButton("Continue on GPU", QMessageBox.AcceptRole)
    cpu = box.addButton("Use CPU", QMessageBox.AcceptRole)
    box.addButton("Cancel", QMessageBox.RejectRole)
    box.setDefaultButton(cpu if recommended_device(check) == CPU else gpu)
    box.exec()
    return GPU if box.clickedButton() is gpu else CPU if box.clickedButton() is cpu else CANCEL


class HomePage(QWidget):
    start_requested = Signal(object)          # JobRequest
    settings_requested = Signal(str)          # tab name
    models_changed = Signal()
    state_changed = Signal()                  # file / PDF / model / message changed: the simple start page mirrors this page
    download_progress = Signal(dict)          # a model download reports progress (also read by the simple start page)
    download_done = Signal(str)               # "" = finished, otherwise the message of the failure or cancellation

    def __init__(self, services: AppServices, parent=None) -> None:
        super().__init__(parent)
        self.services = services
        self.source: Optional[Path] = None
        self.duration: Optional[float] = None
        self.pdf_path: Optional[Path] = None
        self.unfinished: Optional[JobRecord] = None
        self.gpu_status = None
        self._threads: list[TaskThread] = []
        self.confirm: Callable[[QWidget, str, str, str], bool] = default_confirm       # replaced in tests
        self.choose_device: Callable[[QWidget, object], str] = default_vram_choice       # replaced in tests and by unattended (--autorun) runs
        self.choose_pdf: Callable[[], str] = self._choose_pdf_dialog

        root = QVBoxLayout(self)
        root.setSpacing(12)
        head = QHBoxLayout()
        title = QLabel("MELETRA")
        title.setObjectName("title")
        head.addWidget(title)
        head.addStretch(1)
        self.privacy_badge = QLabel(PRIVACY_LOCAL)
        self.privacy_badge.setObjectName("badgeLocal")
        head.addWidget(self.privacy_badge)
        root.addLayout(head)

        self.device_banner = QLabel("Checking your computer ...")
        self.device_banner.setObjectName("banner")
        self.device_banner.setWordWrap(True)
        root.addWidget(self.device_banner)

        self.model_banner = QFrame()
        self.model_banner.setObjectName("card")
        mb = QVBoxLayout(self.model_banner)
        self.model_text = QLabel("")
        self.model_text.setWordWrap(True)
        mb.addWidget(self.model_text)
        row = QHBoxLayout()
        self.download_button = QPushButton("DOWNLOAD MODEL")
        self.download_button.setObjectName("primary")
        self.download_button.clicked.connect(self.start_model_download)
        self.cancel_download_button = QPushButton("Cancel download")
        self.cancel_download_button.clicked.connect(self.cancel_model_download)
        self.cancel_download_button.hide()
        self.import_button = QPushButton("Import model folder ...")
        self.import_button.clicked.connect(lambda: self.settings_requested.emit("Models"))
        row.addWidget(self.download_button)
        row.addWidget(self.cancel_download_button)
        row.addWidget(self.import_button)
        row.addStretch(1)
        mb.addLayout(row)
        self.model_progress = QProgressBar()
        self.model_progress.setRange(0, 1000)
        self.model_progress.hide()
        mb.addWidget(self.model_progress)
        self.model_progress_text = QLabel("")
        self.model_progress_text.setObjectName("muted")
        self.model_progress_text.hide()
        mb.addWidget(self.model_progress_text)
        root.addWidget(self.model_banner)

        self.drop_zone = DropZone()
        self.drop_zone.file_chosen.connect(self.set_file)
        self.drop_zone.rejected.connect(self.show_problem)
        root.addWidget(self.drop_zone)

        self.problem_label = QLabel("")
        self.problem_label.setObjectName("bannerBad")
        self.problem_label.setWordWrap(True)
        self.problem_label.hide()
        root.addWidget(self.problem_label)

        self.resume_box = QFrame()
        self.resume_box.setObjectName("card")
        rb = QVBoxLayout(self.resume_box)
        self.resume_text = QLabel("")
        self.resume_text.setWordWrap(True)
        rb.addWidget(self.resume_text)
        rrow = QHBoxLayout()
        self.resume_button = QPushButton("RESUME")
        self.resume_button.setObjectName("primary")
        self.resume_button.clicked.connect(lambda: self.start(resume=True))
        self.start_over_button = QPushButton("START OVER")
        self.start_over_button.clicked.connect(self.start_over)
        rrow.addWidget(self.resume_button)
        rrow.addWidget(self.start_over_button)
        rrow.addStretch(1)
        rb.addLayout(rrow)
        self.resume_box.hide()
        root.addWidget(self.resume_box)

        form_card = card()
        form = QFormLayout(form_card)
        form.setLabelAlignment(Qt.AlignLeft)
        self.file_label = QLabel("No file selected")
        self.file_label.setWordWrap(True)
        self.size_label = QLabel("-")
        self.duration_label = QLabel("-")
        form.addRow("Selected file:", self.file_label)
        form.addRow("Size:", self.size_label)
        form.addRow("Duration:", self.duration_label)
        self.language_combo = QComboBox()
        for label, value in LANGUAGE_CHOICES:
            self.language_combo.addItem(label, value)
        self.subject_combo = QComboBox()
        for label, value in subject_choices():
            self.subject_combo.addItem(label, value)
        self.model_combo = QComboBox()
        self.model_combo.currentIndexChanged.connect(lambda _i: self._model_changed())
        self.mode_combo = QComboBox()
        for label, value in MODE_CHOICES:
            self.mode_combo.addItem(label, value)
        form.addRow("Language:", self.language_combo)
        form.addRow("Subject:", self.subject_combo)
        form.addRow("Model:", self.model_combo)
        form.addRow("Mode:", self.mode_combo)
        pdf_row = QHBoxLayout()
        self.pdf_button = QPushButton("Add PDF")
        self.pdf_button.clicked.connect(self.add_pdf)
        self.pdf_clear = QPushButton("Remove")
        self.pdf_clear.clicked.connect(lambda: self.set_pdf(None))
        self.pdf_label = QLabel("Optional lecture slides: terms, abbreviations and subject are used as context.")
        self.pdf_label.setObjectName("muted")
        self.pdf_label.setWordWrap(True)
        pdf_row.addWidget(self.pdf_button)
        pdf_row.addWidget(self.pdf_clear)
        pdf_row.addWidget(self.pdf_label, 1)
        form.addRow("Lecture materials:", pdf_row)
        root.addWidget(form_card)

        action = QHBoxLayout()
        self.start_button = QPushButton("START TRANSCRIPTION")
        self.start_button.setObjectName("primary")
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(lambda: self.start())
        self.advanced_button = QPushButton("Advanced settings ...")
        self.advanced_button.clicked.connect(lambda: self.settings_requested.emit("Performance"))
        action.addWidget(self.start_button)
        action.addWidget(self.advanced_button)
        action.addStretch(1)
        root.addLayout(action)

        self.unfinished_label = QLabel("")
        self.unfinished_label.setObjectName("muted")
        self.unfinished_label.setWordWrap(True)
        root.addWidget(self.unfinished_label)
        root.addStretch(1)

        self.load_settings()
        self.refresh_models()
        self.refresh_unfinished()

    # ------------------------------------------------------------------ settings -> widgets
    def load_settings(self) -> None:
        s = self.services.settings.settings
        self.language_combo.setCurrentIndex(index_of(LANGUAGE_CHOICES, s.language))
        self.subject_combo.setCurrentIndex(index_of(subject_choices(), s.subject))
        self.mode_combo.setCurrentIndex(index_of(MODE_CHOICES, s.mode))
        self.privacy_badge.setText(PRIVACY_CLOUD if s.cloud_enabled else PRIVACY_LOCAL)
        self.privacy_badge.setObjectName("badgeCloud" if s.cloud_enabled else "badgeLocal")
        self.privacy_badge.style().unpolish(self.privacy_badge)
        self.privacy_badge.style().polish(self.privacy_badge)

    def remember_choices(self) -> None:
        self.services.update_setting(language=self.language_combo.currentData(), subject=self.subject_combo.currentData(),
                                     mode=self.mode_combo.currentData(), model_key=self.model_combo.currentData() or "whisper-large-v3")

    # ------------------------------------------------------------------ device / models
    def set_gpu_status(self, status) -> None:
        self.gpu_status = status
        text = status.headline + ("\n" + status.detail if status.detail else "")
        self.device_banner.setText(text)
        self.device_banner.setObjectName("banner" if status.cuda_available else "bannerWarn")
        self.device_banner.style().unpolish(self.device_banner)
        self.device_banner.style().polish(self.device_banner)

    def refresh_models(self) -> None:
        keep = self.model_combo.currentData() or self.services.settings.settings.model_key
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for state in self.services.models.list_models():
            info = self.services.models.info(state.key)
            label = info.title + (" (Recommended)" if self.services.models.is_recommended(state.key) else "") + (
                "" if state.installed else "  -  not installed")
            self.model_combo.addItem(label, state.key)
        self.model_combo.setCurrentIndex(max(self.model_combo.findData(keep), 0))
        self.model_combo.blockSignals(False)
        self._model_changed()

    def _model_changed(self) -> None:
        key = self.model_combo.currentData()
        if not key:
            return
        state = self.services.models.state(key)
        info = self.services.models.info(key)
        if state.installed and state.runtime_ok:
            self.model_banner.hide()
        else:
            self.model_banner.show()
            if state.installed and not state.runtime_ok:
                self.model_text.setText(f"{info.title} is installed, but {state.runtime_note}")
                self.download_button.hide()
            else:
                resume = f" (a partial download of {human_bytes(state.partial_bytes)} will be resumed)" if state.partial_bytes else ""
                self.model_text.setText(f"Model not installed.\n{info.title}  ~{info.approx_bytes / 1024 ** 3:.1f} GB download{resume}")
                self.download_button.setText("RESUME DOWNLOAD" if state.partial_bytes else "DOWNLOAD MODEL")
                self.download_button.show()
        self._update_start_enabled()

    def start_model_download(self) -> None:
        key = self.model_combo.currentData()
        if not key or any(t.isRunning() for t in self._threads):
            return
        self.download_button.setEnabled(False)
        self.cancel_download_button.show()
        self.model_progress.show()
        self.model_progress_text.show()
        thread = TaskThread(lambda progress, cancelled: self.services.models.download(key, progress, cancelled), self)
        thread.progressed.connect(self._download_progress)
        thread.succeeded.connect(lambda _r: self._download_finished(None))
        thread.failed.connect(self._download_finished)
        self._threads.append(thread)
        self.download_thread = thread
        thread.start()

    def cancel_model_download(self) -> None:
        thread = getattr(self, "download_thread", None)
        if thread is not None:
            thread.cancel()

    def _download_progress(self, event: dict) -> None:
        self.download_progress.emit(event)
        if event.get("message"):
            self.model_progress_text.setText(event["message"])
            return
        self.model_progress.setValue(int(event.get("percent", 0) * 10))
        eta = event.get("eta_seconds")
        self.model_progress_text.setText(
            f"Downloading model  {event.get('percent', 0):.0f}%   {human_bytes(event.get('bytes_done', 0))} / "
            f"{human_bytes(event.get('bytes_total', 0))}" + (f"   {human_bytes(event.get('speed_bps', 0))}/s   about {int(eta // 60)} min left" if eta else ""))

    def _download_finished(self, error: Optional[str]) -> None:
        self.download_button.setEnabled(True)
        self.cancel_download_button.hide()
        self.model_progress.hide()
        self.model_progress_text.hide()
        if error:
            self.show_problem(error if "resume" in error.lower() or "cancel" in error.lower() else f"Model download failed: {error}")
        self.refresh_models()
        self.models_changed.emit()
        self.download_done.emit(error or "")

    # ------------------------------------------------------------------ file selection
    def show_problem(self, text: str) -> None:
        self.problem_label.setText(text)
        self.problem_label.show()
        self.state_changed.emit()

    def set_file(self, path: str) -> None:
        source = Path(path)
        self.problem_label.hide()
        if not source.is_file():
            self.show_problem(f"'{source.name}' was not found.")
            return
        self.source, self.duration = source, None
        self.file_label.setText(source.name)
        self.size_label.setText(format_size(source.stat().st_size))
        self.duration_label.setText("reading ...")
        self.services.update_setting(last_input_dir=str(source.parent))
        thread = TaskThread(lambda progress, cancelled: probe_media_pyav(source), self)
        thread.succeeded.connect(lambda info, s=source: self._probed(s, info))
        thread.failed.connect(lambda msg, s=source: self._probed(s, {}))
        self._threads.append(thread)
        thread.start()
        self.refresh_unfinished()
        self._update_start_enabled()

    def _probed(self, source: Path, info: dict) -> None:
        if source != self.source:
            return
        seconds = float(info.get("duration_s") or 0.0)
        if seconds > 0:
            self.duration = seconds
            self.duration_label.setText(format_hms(seconds))
        else:
            self.duration_label.setText("unknown (the file could not be read - it may be damaged)")
            self.show_problem(f"'{source.name}' could not be read as audio or video. It may be damaged or contain no audio track.")
        self._update_start_enabled()

    def add_pdf(self) -> None:
        path = self.choose_pdf()
        if path:
            self.set_pdf(Path(path))

    def _choose_pdf_dialog(self) -> str:
        start = str(self.source.parent) if self.source else ""
        path, _ = QFileDialog.getOpenFileName(self, "Select lecture slides (PDF)", start, "PDF files (*.pdf)")
        return path

    def set_pdf(self, path: Optional[Path]) -> None:
        self.pdf_path = path
        self.state_changed.emit()
        if path is None:
            self.pdf_label.setText("Optional lecture slides: terms, abbreviations and subject are used as context.")
        else:
            self.pdf_label.setText(f"{path.name} - medical terms, acronyms and subject context will be extracted")

    # ------------------------------------------------------------------ unfinished work
    def refresh_unfinished(self) -> None:
        self.unfinished = self.services.jobs.find_for_source(self.source) if self.source else None
        if self.unfinished is None:
            self.resume_box.hide()
        else:
            job = self.unfinished
            why = f"\n{job.last_error}" if job.last_error else ""
            self.resume_text.setText(f"Previous unfinished transcription found.\nProgress: {job.progress:.0f}%  "
                                     f"(chunk {job.chunk} of {job.chunks}){why}")
            self.resume_box.show()
        others = [j for j in self.services.jobs.unfinished() if self.unfinished is None or j.job_id != self.unfinished.job_id]
        if others:
            names = ", ".join(f"{j.source_name} ({j.progress:.0f}%)" for j in others[:4])
            self.unfinished_label.setText(f"Unfinished jobs: {names}. Select the file again to resume it.")
        else:
            self.unfinished_label.setText("")

    def start_over(self) -> None:
        if self.unfinished is None:
            return
        if not self.confirm(self, "Start over", f"Discard the saved progress ({self.unfinished.progress:.0f}%) of "
                            f"'{self.unfinished.source_name}' and transcribe it from the beginning?", "Start over"):
            return
        self.discard_unfinished(self.unfinished)
        self.start(resume=False)

    def discard_unfinished(self, job: JobRecord) -> None:
        shutil.rmtree(self.services.paths.checkpoints / job.job_id, ignore_errors=True)
        self.services.jobs.delete(job.job_id)
        self.refresh_unfinished()

    # ------------------------------------------------------------------ start
    def _update_start_enabled(self) -> None:
        key = self.model_combo.currentData()
        ready = bool(key) and self.services.models.is_installed(key) and self.services.models.state(key).runtime_ok
        self.start_button.setEnabled(self.source is not None and ready and self.duration is not None)
        self.state_changed.emit()

    def request_template(self) -> JobRequest:
        """The job settings without a file: the model, language, subject and mode shown here, and the settings of the program."""
        s = self.services.settings.settings
        return JobRequest(
            source="", output_dir=s.output_dir,
            model_key=self.model_combo.currentData(), language=self.language_combo.currentData(),
            subject=self.subject_combo.currentData(), mode=self.mode_combo.currentData(), device=s.device,
            compute_type=s.compute_type, beam_size=s.beam_size, vad=s.vad, chunk_minutes=s.chunk_minutes,
            overlap_seconds=s.overlap_seconds, medical_context=s.medical_context, hotwords=s.hotwords,
            use_pdf=s.pdf_context, cpu_threads=s.cpu_threads, cloud_compare=s.cloud_enabled, unload_ollama=s.unload_ollama,
            medical_expert=s.medical_expert, custom_terms=list(s.custom_terms), medical_intelligence=s.medical_intelligence,
            output_language=s.output_language)

    def build_request(self, resume: bool = True) -> JobRequest:
        assert self.source is not None
        request = self.request_template()
        request.source = str(self.source)
        request.output_dir = request.output_dir or str(default_output_dir(self.source))
        request.pdf_path = str(self.pdf_path) if self.pdf_path else ""
        request.resume = resume
        return request

    def reset_file(self) -> None:
        """No recording is selected (the start screen shows a queue or a document instead)."""
        self.source, self.duration = None, None
        self.file_label.setText("No file selected")
        self.size_label.setText("-")
        self.duration_label.setText("-")
        self.set_pdf(None)
        self.refresh_unfinished()
        self._update_start_enabled()

    def disk_problem(self) -> str:
        """Not enough room for the temporary 16 kHz audio (about 1.9 MB per minute) or for the results? Says so before the job starts."""
        if self.source is None:
            return ""
        wav_bytes = int((self.duration or 0.0) * 16000 * 2)
        checks = [(self.services.paths.temp, int(wav_bytes * 1.5) + 300 * 1024 ** 2, "for the temporary audio and progress files")]
        s = self.services.settings.settings
        out = Path(s.output_dir) if s.output_dir else self.source.parent
        checks.append((out, 100 * 1024 ** 2, "for the transcripts"))
        for folder, need, why in checks:
            probe = folder
            while not probe.exists() and probe != probe.parent:
                probe = probe.parent
            try:
                free = shutil.disk_usage(probe).free
            except OSError:
                continue
            if free < need:
                return (f"Not enough free disk space on {probe.anchor or probe} {why}: {free / 1024 ** 3:.1f} GB free, "
                        f"about {need / 1024 ** 3:.1f} GB needed. Free some space or choose another output folder (Settings -> General).")
        return ""

    def start(self, resume: bool = True) -> None:
        if self.source is None:
            return
        key = self.model_combo.currentData()
        if not key or not self.services.models.is_installed(key):
            self.show_problem("The selected model is not installed. Download it first (see the box above).")
            return
        problem = self.disk_problem()
        if problem:
            self.show_problem(problem)
            return
        request = self.build_request(resume)
        s = self.services.settings.settings
        if s.cloud_enabled:
            if not self.services.credentials.has_deepgram_key():
                self.show_problem("Cloud comparison is on but no Deepgram API key is stored (Settings -> Cloud). "
                                  "Turn it off or add the key.")
                return
            if not self.confirm(self, "Upload audio to Deepgram?", "Cloud comparison is ON. The audio of this lecture will be "
                                "uploaded to Deepgram (a third party). The local transcription still runs on this computer.",
                                "Upload and continue"):
                return
        use_gpu = request.device in ("auto", "cuda") and self.gpu_status is not None and self.gpu_status.cuda_available
        if use_gpu and s.vram_warnings:
            check = self.services.gpu.vram_check()
            if not check.ok:
                choice = self.choose_device(self, check)
                if choice == CANCEL:
                    return
                if choice == CPU:
                    request.device = "cpu"
                    log.info("GPU memory low (%s MiB free): this job runs on the CPU", check.free_mib)
        self.remember_choices()
        self.start_requested.emit(request)
