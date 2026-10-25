"""Settings: General, Models, Performance, Medical, Cloud, Diagnostics."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QPlainTextEdit, QProgressBar, QPushButton, QSpinBox, QTableWidget,
    QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from ..services import diagnostics_service
from ..services.model_service import ModelError, human_bytes
from ..services.settings_service import OUTPUT_FORMATS
from ..vocabulary import DEFAULT_TERMS_DIR, available_subjects, load_terms
from .choices import (
    COMPUTE_CHOICES, DEVICE_CHOICES, HOTWORD_CHOICES, LANGUAGE_CHOICES, MODE_CHOICES, OUTPUT_LANGUAGE_CHOICES, THEME_CHOICES, index_of, subject_choices, theme_choices,
)
from .components import ask
from .context import AppServices
from .tasks import TaskThread, stop_threads

log = logging.getLogger(__name__)

TAB_NAMES = ["General", "Models", "Performance", "Medical", "Cloud", "Diagnostics"]


def _add(form: QFormLayout, label: str, field) -> None:
    """One row of a settings form; the field also carries the label as its accessible name."""
    if hasattr(field, "setAccessibleName") and not field.accessibleName():
        field.setAccessibleName(label.rstrip(":"))
    form.addRow(label, field)


def _combo(choices: list[tuple[str, str]], current: str) -> QComboBox:
    box = QComboBox()
    for label, value in choices:
        box.addItem(label, value)
    box.setCurrentIndex(index_of(choices, current))
    return box


class SettingsDialog(QDialog):
    settings_changed = Signal()
    models_changed = Signal()

    def __init__(self, services: AppServices, parent=None, initial_tab: str = "General") -> None:
        super().__init__(parent)
        self.services = services
        self.setWindowTitle("Settings")
        self.resize(760, 560)
        self._threads: list[TaskThread] = []
        self.choose_folder = lambda title, start: QFileDialog.getExistingDirectory(self, title, start)
        self.confirm = lambda title, text: ask(self, title, text, accept=title)
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        s = services.settings.settings
        self._build_tabs(s)
        self.tabs.currentChanged.connect(self._tab_changed)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Close)
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.select_tab(initial_tab)

    def _build_tabs(self, s) -> None:
        self.tabs.addTab(self._general_tab(s), "General")
        self.tabs.addTab(self._models_tab(), "Models")
        self.tabs.addTab(self._performance_tab(s), "Performance")
        self.tabs.addTab(self._medical_tab(s), "Medical")
        self.tabs.addTab(self._cloud_tab(s), "Cloud")
        self.tabs.addTab(self._diagnostics_tab(), "Diagnostics")

    def done(self, result: int) -> None:                    # Save, Close, Esc and the window's X all end here
        stop_threads(self)                                   # a download that is still running is cancelled (it resumes next time)
        super().done(result)

    def select_tab(self, name: str) -> None:
        if name in TAB_NAMES:
            self.tabs.setCurrentIndex(TAB_NAMES.index(name))

    # ------------------------------------------------------------------ General
    def _general_tab(self, s) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self.output_edit = QLineEdit(s.output_dir)
        self.output_edit.setPlaceholderText("Next to the source file, in a 'Medical Transcription' folder")
        browse = QPushButton("Browse ...")
        browse.clicked.connect(self._browse_output)
        row = QHBoxLayout()
        row.addWidget(self.output_edit, 1)
        row.addWidget(browse)
        _add(form, "Default output folder:", row)
        self.theme_combo = _combo(theme_choices(), s.theme)
        self.open_folder_check = QCheckBox("Open the output folder when a transcription is finished")
        self.open_folder_check.setChecked(s.open_folder_after)
        _add(form, "Theme:", self.theme_combo)
        form.addRow("", self.open_folder_check)
        self.language_combo = _combo(LANGUAGE_CHOICES, s.language)
        self.output_language_combo = _combo(OUTPUT_LANGUAGE_CHOICES, s.output_language)
        self.subject_combo = _combo(subject_choices(), s.subject)
        self.mode_combo = _combo(MODE_CHOICES, s.mode)
        _add(form, "Default input language:", self.language_combo)
        _add(form, "Default output language:", self.output_language_combo)
        _add(form, "Default subject:", self.subject_combo)
        _add(form, "Default mode:", self.mode_combo)
        note = QLabel("Your transcripts are written to the output folder. Program data (models, logs, unfinished work) is kept in "
                      f"{self.services.paths.root}")
        note.setObjectName("muted")
        note.setWordWrap(True)
        form.addRow("", note)
        return page

    def _browse_output(self) -> None:
        folder = self.choose_folder("Choose the output folder", self.output_edit.text())
        if folder:
            self.output_edit.setText(folder)

    # ------------------------------------------------------------------ Models
    def _models_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.model_table = QTableWidget(0, 5)
        self.model_table.setAccessibleName("Installed models")
        self.model_table.setHorizontalHeaderLabels(["Model", "Status", "Size on disk", "Verified", "Location"])
        header = self.model_table.horizontalHeader()
        for column in range(4):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.Stretch)
        self.model_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.model_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.model_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.model_table.itemSelectionChanged.connect(self._model_selection)
        layout.addWidget(self.model_table)
        self.model_note = QLabel("")
        self.model_note.setObjectName("muted")
        self.model_note.setWordWrap(True)
        layout.addWidget(self.model_note)
        row = QHBoxLayout()
        self.download_button = QPushButton("Download")
        self.cancel_button = QPushButton("Cancel")
        self.verify_button = QPushButton("Verify")
        self.delete_button = QPushButton("Delete")
        self.import_button = QPushButton("Import Model Folder ...")
        self.folder_button = QPushButton("Add search folder ...")
        self.download_button.clicked.connect(self.download_selected)
        self.cancel_button.clicked.connect(self.cancel_task)
        self.verify_button.clicked.connect(self.verify_selected)
        self.delete_button.clicked.connect(self.delete_selected)
        self.import_button.clicked.connect(self.import_selected)
        self.folder_button.clicked.connect(self.add_search_folder)
        for b in (self.download_button, self.cancel_button, self.verify_button, self.delete_button, self.import_button,
                  self.folder_button):
            row.addWidget(b)
        layout.addLayout(row)
        self.model_progress = QProgressBar()
        self.model_progress.setRange(0, 1000)
        self.model_progress.hide()
        self.model_progress_text = QLabel("")
        self.model_progress_text.setObjectName("muted")
        layout.addWidget(self.model_progress)
        layout.addWidget(self.model_progress_text)
        self.disk_label = QLabel("")
        self.disk_label.setObjectName("muted")
        layout.addWidget(self.disk_label)
        self.cancel_button.hide()
        self.refresh_models()
        return page

    def _selected_key(self) -> Optional[str]:
        rows = self.model_table.selectionModel().selectedRows() if self.model_table.selectionModel() else []
        if not rows:
            return None
        return self.model_table.item(rows[0].row(), 0).data(Qt.UserRole)

    def refresh_models(self) -> None:
        models = self.services.models
        states = models.list_models(kind=None)                       # the speech models and the offline translation pack
        self.model_table.setRowCount(len(states))
        for row, state in enumerate(states):
            info = models.info(state.key)
            status = ("Installed" if state.installed else (f"Partial download ({human_bytes(state.partial_bytes)})" if state.partial_bytes
                                                           else "Not installed"))
            if state.installed and not state.runtime_ok:
                status = "Installed - runtime missing"
            verified = {True: "Yes", False: "FAILED", None: "-"}[state.verified] if state.installed else "-"
            cells = [info.title + (" (Recommended)" if self.services.models.is_recommended(state.key) else ""), status,
                     human_bytes(state.size_on_disk) if state.installed else f"~{info.approx_bytes / 1024 ** 3:.1f} GB download",
                     verified, str(state.path) if state.path else ""]
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setData(Qt.UserRole, state.key)
                self.model_table.setItem(row, column, item)
        usage = models.disk_usage()
        self.disk_label.setText(f"Models use {human_bytes(usage['models_bytes'])}; {human_bytes(usage['free_bytes'])} free on this drive.")
        if self.model_table.rowCount() and not self.model_table.selectionModel().hasSelection():
            self.model_table.selectRow(0)
        self._model_selection()

    def _model_selection(self) -> None:
        key = self._selected_key()
        busy = any(t.isRunning() for t in self._threads)
        if key is None:
            return
        state = self.services.models.state(key)
        info = self.services.models.info(key)
        self.download_button.setEnabled(not busy and not state.installed)
        self.download_button.setText("Resume download" if state.partial_bytes and not state.installed else "Download")
        self.verify_button.setEnabled(not busy and state.installed)
        self.delete_button.setEnabled(not busy and (state.installed and state.source == "app" or state.partial_bytes > 0))
        self.import_button.setEnabled(not busy)
        self.model_note.setText(f"{info.description}\nRuntime: {info.runtime}" + (f"\n{state.runtime_note}" if state.runtime_note else ""))

    def _start_task(self, fn, on_success=None) -> None:
        thread = TaskThread(fn, self)
        thread.progressed.connect(self._task_progress)
        thread.succeeded.connect(lambda result: self._task_finished(None, on_success, result))
        thread.failed.connect(lambda msg: self._task_finished(msg, None, None))
        self._threads.append(thread)
        self.task = thread
        self.model_progress.setValue(0)
        self.model_progress.show()
        self.cancel_button.show()
        for b in (self.download_button, self.verify_button, self.delete_button, self.import_button):
            b.setEnabled(False)
        thread.start()

    def _task_progress(self, event: dict) -> None:
        if event.get("message"):
            self.model_progress_text.setText(event["message"])
            return
        percent = float(event.get("percent", 0.0))
        self.model_progress.setValue(int(percent * 10))
        if event.get("type") == "model_download_progress":
            eta = event.get("eta_seconds")
            self.model_progress_text.setText(
                f"Downloading model  {percent:.0f}%   {human_bytes(event.get('bytes_done', 0))} / "
                f"{human_bytes(event.get('bytes_total', 0))}" + (f"   about {int(eta // 60)} min left" if eta else ""))
        else:
            self.model_progress_text.setText(f"{event.get('type', 'working').replace('model_', '').replace('_', ' ').title()} {percent:.0f}%")

    def _task_finished(self, error: Optional[str], on_success, result) -> None:
        self.model_progress.hide()
        self.cancel_button.hide()
        self.model_progress_text.setText(error or "")
        if error is None and on_success is not None:
            on_success(result)
        self.refresh_models()
        self.models_changed.emit()

    def cancel_task(self) -> None:
        if getattr(self, "task", None) is not None:
            self.task.cancel()

    def download_selected(self) -> None:
        key = self._selected_key()
        if key:
            self._start_task(lambda progress, cancelled: self.services.models.download(key, progress, cancelled))

    def verify_selected(self) -> None:
        key = self._selected_key()
        if key:
            def done(result) -> None:
                self.model_progress_text.setText("Verification passed: every file matches its checksum." if result.ok
                                                 else "Verification FAILED: " + "; ".join(result.problems[:3]))

            self._start_task(lambda progress, cancelled: self.services.models.verify(key, progress), done)

    def delete_selected(self) -> None:
        key = self._selected_key()
        if key and self.confirm("Delete model", f"Delete '{self.services.models.info(key).title}' from this computer? "
                                "You can download it again later."):
            self.services.models.delete(key)
            self.refresh_models()
            self.models_changed.emit()

    def import_selected(self) -> None:
        key = self._selected_key()
        if not key:
            return
        folder = self.choose_folder("Select the folder that contains the model files", "")
        if not folder:
            return

        def done(result) -> None:
            _path, check = result
            self.model_progress_text.setText("Model imported and verified." if check.ok else
                                             "Model imported, but it is NOT identical to the tested version: " + "; ".join(check.problems[:2]))

        self._start_task(lambda progress, cancelled: self.services.models.import_folder(key, Path(folder), progress), done)

    def add_search_folder(self) -> None:
        folder = self.choose_folder("Folder that contains model folders (e.g. on a USB drive)", "")
        if folder:
            extra = list(self.services.settings.settings.extra_model_dirs)
            if folder not in extra:
                extra.append(folder)
            self.services.update_setting(extra_model_dirs=extra)
            self.refresh_models()
            self.models_changed.emit()

    # ------------------------------------------------------------------ Performance
    def _performance_tab(self, s) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self.device_combo = _combo(DEVICE_CHOICES, s.device)
        self.compute_combo = _combo(COMPUTE_CHOICES, s.compute_type)
        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(0, 256)
        self.threads_spin.setSpecialValueText("Automatic")
        self.threads_spin.setValue(s.cpu_threads)
        self.beam_spin = QSpinBox()
        self.beam_spin.setRange(0, 20)
        self.beam_spin.setSpecialValueText("Mode default (5 in Quality)")
        self.beam_spin.setValue(s.beam_size)
        self.vad_check = QCheckBox("Skip silence with voice activity detection (recommended)")
        self.vad_check.setChecked(s.vad)
        self.chunk_spin = QDoubleSpinBox()
        self.chunk_spin.setRange(1.0, 60.0)
        self.chunk_spin.setSuffix(" min")
        self.chunk_spin.setValue(s.chunk_minutes)
        self.overlap_spin = QDoubleSpinBox()
        self.overlap_spin.setRange(0.0, 15.0)
        self.overlap_spin.setSuffix(" s")
        self.overlap_spin.setValue(s.overlap_seconds)
        self.vram_check = QCheckBox("Warn me when another program uses much GPU memory")
        self.vram_check.setChecked(s.vram_warnings)
        self.ollama_check = QCheckBox("Advanced: unload idle Ollama models before transcribing (off by default)")
        self.ollama_check.setChecked(s.unload_ollama)
        self.ollama_check.setToolTip("Runs 'ollama stop <model>' for loaded models. The app never closes other programs by itself.")
        _add(form, "Device:", self.device_combo)
        _add(form, "Compute type:", self.compute_combo)
        _add(form, "CPU threads:", self.threads_spin)
        _add(form, "Beam size:", self.beam_spin)
        form.addRow("", self.vad_check)
        _add(form, "Chunk size:", self.chunk_spin)
        _add(form, "Chunk overlap:", self.overlap_spin)
        form.addRow("", self.vram_check)
        form.addRow("", self.ollama_check)
        hint = QLabel("Without an NVIDIA GPU the program uses the CPU automatically. Large-v3 on a CPU can take several times the length "
                      "of the recording.")
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        form.addRow("", hint)
        return page

    # ------------------------------------------------------------------ Medical
    def _medical_tab(self, s) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        self.context_check = QCheckBox("Medical context: detect the subject and give the model the relevant terms")
        self.context_check.setChecked(s.medical_context)
        self.hotwords_combo = _combo(HOTWORD_CHOICES, s.hotwords)
        self.pdf_check = QCheckBox("Use lecture slides (PDF) for terms, abbreviations and subject when available")
        self.pdf_check.setChecked(s.pdf_context)
        form.addRow("", self.context_check)
        _add(form, "Hotwords:", self.hotwords_combo)
        form.addRow("", self.pdf_check)
        try:
            files = available_subjects(DEFAULT_TERMS_DIR)
            total = len(load_terms("all", None, DEFAULT_TERMS_DIR))
            vocab_text = f"{total:,} medical terms in {len(files)} subject files ({', '.join(files)})"
        except Exception:  # noqa: BLE001
            vocab_text = "vocabulary not found"
        self.vocab_label = QLabel(vocab_text)
        self.vocab_label.setWordWrap(True)
        self.vocab_label.setObjectName("muted")
        _add(form, "Vocabulary:", self.vocab_label)
        self.medical_vocab_button = QPushButton("Medical vocabulary - add terms ...")
        self.medical_vocab_button.clicked.connect(self.open_medical_vocabulary)
        _add(form, "My terms:", self.medical_vocab_button)
        formats = QWidget()
        fl = QHBoxLayout(formats)
        fl.setContentsMargins(0, 0, 0, 0)
        self.format_checks: dict[str, QCheckBox] = {}
        for name in OUTPUT_FORMATS:
            box = QCheckBox(name.upper())
            box.setChecked(name in s.output_formats)
            self.format_checks[name] = box
            fl.addWidget(box)
        fl.addStretch(1)
        _add(form, "Output formats:", formats)
        note = QLabel("All transcript files are always written (raw, clean, timestamped, SRT, VTT, JSON and the medical reports). "
                      "The raw transcript is never edited by any setting.")
        note.setObjectName("muted")
        note.setWordWrap(True)
        form.addRow("", note)
        return page

    # ------------------------------------------------------------------ Cloud
    def _cloud_tab(self, s) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        privacy = QLabel("LOCAL (default)\nNo audio leaves your computer.\n\nDEEPGRAM (optional)\nAudio is uploaded to Deepgram, a third-party service.")
        privacy.setObjectName("banner")
        layout.addWidget(privacy)
        self.cloud_check = QCheckBox("Also compare with Deepgram Nova-3 Medical (uploads the audio)")
        self.cloud_check.setChecked(s.cloud_enabled)
        layout.addWidget(self.cloud_check)
        form = QFormLayout()
        self.key_edit = QLineEdit()
        self.key_edit.setEchoMode(QLineEdit.Password)
        self.key_edit.setPlaceholderText("Paste your Deepgram API key")
        self.save_key_button = QPushButton("Save key")
        self.remove_key_button = QPushButton("Remove key")
        self.save_key_button.clicked.connect(self.save_key)
        self.remove_key_button.clicked.connect(self.remove_key)
        row = QHBoxLayout()
        row.addWidget(self.key_edit, 1)
        row.addWidget(self.save_key_button)
        row.addWidget(self.remove_key_button)
        _add(form, "API key:", row)
        self.key_status = QLabel("")
        self.key_status.setObjectName("muted")
        self.key_status.setWordWrap(True)
        form.addRow("", self.key_status)
        layout.addLayout(form)
        layout.addStretch(1)
        self._refresh_key_status()
        return page

    def _refresh_key_status(self) -> None:
        creds = self.services.credentials
        where = creds.store_label
        if creds.has_deepgram_key():
            self.key_status.setText(f"A key is stored in {where}: {creds.masked_deepgram_key()}. It is never written to a settings file or a log.")
        else:
            self.key_status.setText(f"No key stored. Keys are kept in {where}, never in a settings file.")

    def save_key(self) -> None:
        text = self.key_edit.text()
        try:
            self.services.credentials.set_deepgram_key(text)
        except (ValueError, OSError) as exc:
            self.key_status.setText(str(exc))
            return
        self.key_edit.clear()
        self._refresh_key_status()

    def remove_key(self) -> None:
        self.services.credentials.delete_deepgram_key()
        self._refresh_key_status()

    # ------------------------------------------------------------------ Diagnostics
    def _diagnostics_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.diag_text = QPlainTextEdit()
        self.diag_text.setReadOnly(True)
        layout.addWidget(self.diag_text, 1)
        row = QHBoxLayout()
        self.copy_button = QPushButton("COPY DIAGNOSTICS")
        self.copy_button.setObjectName("primary")
        self.copy_button.clicked.connect(self.copy_diagnostics)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh_diagnostics)
        logs = QPushButton("Open logs folder")
        logs.clicked.connect(lambda: self.services.open_path(self.services.paths.logs))
        for b in (self.copy_button, refresh, logs):
            row.addWidget(b)
        row.addStretch(1)
        layout.addLayout(row)
        self._diag_loaded = False
        return page

    def _tab_changed(self, index: int) -> None:
        if self.tabs.tabText(index) == "Diagnostics" and not self._diag_loaded:
            self.refresh_diagnostics()

    def refresh_diagnostics(self) -> None:
        self._diag_loaded = True
        self.diag_text.setPlainText("Collecting ...")
        thread = TaskThread(lambda progress, cancelled: diagnostics_service.as_text(
            diagnostics_service.collect(self.services.paths, self.services.models, self.services.gpu)), self)
        thread.succeeded.connect(self.diag_text.setPlainText)
        thread.failed.connect(lambda msg: self.diag_text.setPlainText(f"Could not collect diagnostics: {msg}"))
        self._threads.append(thread)
        self.diag_thread = thread
        thread.start()

    def copy_diagnostics(self) -> None:
        QGuiApplication.clipboard().setText(self.diag_text.toPlainText())

    # ------------------------------------------------------------------ save
    def open_medical_vocabulary(self) -> None:
        from .medical_vocab_dialog import open_medical_vocabulary

        open_medical_vocabulary(self.services, self)

    def save(self) -> None:
        formats = [name for name, box in self.format_checks.items() if box.isChecked()] or list(OUTPUT_FORMATS)
        self.services.update_setting(
            output_dir=self.output_edit.text().strip(), theme=self.theme_combo.currentData(),
            open_folder_after=self.open_folder_check.isChecked(), language=self.language_combo.currentData(), output_language=self.output_language_combo.currentData(),
            subject=self.subject_combo.currentData(), mode=self.mode_combo.currentData(), device=self.device_combo.currentData(),
            compute_type=self.compute_combo.currentData(), cpu_threads=self.threads_spin.value(), beam_size=self.beam_spin.value(),
            vad=self.vad_check.isChecked(), chunk_minutes=self.chunk_spin.value(), overlap_seconds=self.overlap_spin.value(),
            vram_warnings=self.vram_check.isChecked(), unload_ollama=self.ollama_check.isChecked(),
            medical_context=self.context_check.isChecked(), hotwords=self.hotwords_combo.currentData(),
            pdf_context=self.pdf_check.isChecked(), output_formats=formats, cloud_enabled=self.cloud_check.isChecked())
        self.settings_changed.emit()
        self.accept()
