"""Translate to another language: a small window that translates the SAVED transcript of a lecture. It never asks for a file, never starts a transcription job and never runs the speech recogniser."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QDialog, QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout

from ..language_catalogue import OUTPUT_LANGUAGES, language_name
from ..multilingual import source as source_mod
from ..services.model_service import TRANSLATION_PACK_KEY, human_bytes
from ..translation.service import TranslationDoc, TranslationService
from . import tokens
from .context import AppServices
from .tasks import TaskThread

log = logging.getLogger(__name__)

SAVED = "saved"


class TranslateDialog(QDialog):
    """``translated(code)`` is emitted when a translation is stored (or was already stored and the student asked to open it)."""

    translated = Signal(str)

    def __init__(self, services: AppServices, folder: Path, parent=None, service: Optional[TranslationService] = None, preselect: str = "") -> None:
        super().__init__(parent)
        self.services = services
        self.folder = Path(folder)
        self.service = service or TranslationService(self._model_dirs())
        self.setWindowTitle("Translate to another language")
        self.setModal(True)
        self.setMinimumWidth(460)
        self._thread: Optional[TaskThread] = None
        self._busy = ""                                  # "translate" | "install" | ""
        self._failure = ""
        self.result_doc: Optional[TranslationDoc] = None
        self.chosen: str = ""

        layout = QVBoxLayout(self)
        layout.setSpacing(tokens.SPACE["md"])
        heading = QLabel("Translate this lecture")
        heading.setObjectName("title")
        layout.addWidget(heading)
        self.about = QLabel("")
        self.about.setObjectName("muted")
        self.about.setWordWrap(True)
        layout.addWidget(self.about)

        row = QHBoxLayout()
        label = QLabel("Target Language")
        self.combo = QComboBox()
        self.combo.setAccessibleName("Target Language")
        label.setBuddy(self.combo)
        row.addWidget(label)
        row.addWidget(self.combo, 1)
        layout.addLayout(row)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setObjectName("banner")
        self.status.hide()
        layout.addWidget(self.status)
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.hide()
        layout.addWidget(self.bar)

        buttons = QHBoxLayout()
        self.install_button = QPushButton("Install translation pack")
        self.install_button.clicked.connect(self.install_pack)
        self.install_button.hide()
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel)
        self.go_button = QPushButton("Translate")
        self.go_button.setObjectName("primary")
        self.go_button.setDefault(True)
        self.go_button.clicked.connect(self.start)
        buttons.addWidget(self.install_button)
        buttons.addStretch(1)
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.go_button)
        layout.addLayout(buttons)

        self._fill(preselect)
        self.combo.currentIndexChanged.connect(lambda _i: self.refresh())
        self.refresh()

    # ------------------------------------------------------------------------------------------------------------------ content
    def _model_dirs(self) -> list[Path]:
        try:
            return list(self.services.models.search_dirs())
        except Exception:  # noqa: BLE001 - the dialog still opens; the model is reported missing
            return [self.services.paths.models]

    def _fill(self, preselect: str = "") -> None:
        source = source_mod.load(self.folder)
        spoken = source.detected_languages if source is not None else []
        stored = self.service.stored_targets(self.folder, any_engine=True) if source is not None else {}
        self.combo.blockSignals(True)
        self.combo.clear()
        for code, name in OUTPUT_LANGUAGES:
            tag = " (saved)" if code in stored else (" (spoken)" if spoken == [code] else "")
            self.combo.addItem(name + tag, code)
        only = spoken[0] if len(spoken) == 1 else ""                                  # a lecture spoken in one language needs no translation INTO it; a mixed one does (English is the default)
        wanted = preselect or next((c for c, _n in OUTPUT_LANGUAGES if c not in stored and c != only), "en")
        index = max(self.combo.findData(wanted), 0)
        self.combo.setCurrentIndex(index)
        self.combo.blockSignals(False)
        if source is None:
            self.about.setText("This lecture has no saved transcript to translate.")
        else:
            names = " · ".join(language_name(c) for c in spoken) or "unknown"
            note = "  It was made by an earlier version of the program, so only one language was recorded." if source.origin == source_mod.ORIGIN_LEGACY else ""
            self.about.setText(f"Spoken in this lecture: {names}. The saved transcript is translated - no file is needed and nothing is transcribed again.{note}"
                               "  This is machine translation by a small offline model: check medical terms against the original.")

    @property
    def target(self) -> str:
        return str(self.combo.currentData() or "")

    def refresh(self) -> None:
        """Say what will happen for the chosen language and enable only what can work."""
        if self._busy:
            return
        availability = self.service.availability(self.folder, self.target)
        self.install_button.hide()
        self.status.hide()
        self.go_button.setEnabled(True)
        self.go_button.setText("Translate")
        if availability.state == "cached":
            doc = self.service.cached(self.folder, self.target)
            if doc is not None and doc.failed:
                self.go_button.setText("Translate the missing parts")
                self.status.setText(f"A {language_name(self.target)} translation is saved, but {doc.failed} part(s) could not be translated. Only those parts are tried again.")
            else:
                self.go_button.setText("Open translation")
                self.status.setText(f"A {language_name(self.target)} translation of this lecture is already saved. It opens instantly.")
            self.status.show()
        elif availability.state == "model_missing":
            self.go_button.setEnabled(False)
            self.status.setText(f"{availability.message} It is a one-time download of about 0.5 GB; afterwards translations work without internet. Your transcript is not affected.")
            self.status.show()
            self.install_button.show()
        elif availability.state in ("nothing_to_translate", "unsupported_target"):
            self.go_button.setEnabled(False)
            self.status.setText(availability.message)
            self.status.show()

    # ------------------------------------------------------------------------------------------------------------------ actions
    def start(self) -> None:
        target = self.target
        if not target or self._busy:
            return
        cached = self.service.cached(self.folder, target)
        if cached is not None and not cached.failed:
            self.result_doc, self.chosen = cached, target
            self.translated.emit(target)
            self.accept()
            return
        retry = cached is not None
        self._set_busy("translate", f"Translating to {language_name(target)} ...")

        def work(progress: Callable[[dict], None], cancelled: Callable[[], bool]) -> TranslationDoc:
            return self.service.translate(self.folder, target, progress=lambda f: progress({"percent": round(100 * f, 1)}), should_cancel=cancelled, retry_failed=retry)

        self._run(work, self._translated_ok)

    def install_pack(self) -> None:
        if self._busy:
            return
        self._set_busy("install", "Downloading the translation pack ...")

        def work(progress: Callable[[dict], None], cancelled: Callable[[], bool]) -> Any:
            return self.services.models.download(TRANSLATION_PACK_KEY, progress, cancelled)

        self._run(work, self._installed_ok)

    def cancel(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            self._thread.cancel()
            self.status.setText("Cancelling - nothing is changed ...")
            self.status.show()
            self.cancel_button.setEnabled(False)
            return
        self.reject()

    def _run(self, work: Callable[..., Any], on_success: Callable[[Any], None]) -> None:
        thread = TaskThread(work, self)
        thread.progressed.connect(self._progress)
        thread.succeeded.connect(on_success)
        thread.failed.connect(self._failed)
        thread.done.connect(self._finished)
        self._thread = thread
        thread.start()

    def _set_busy(self, what: str, message: str) -> None:
        self._busy = what
        self.combo.setEnabled(False)
        self.go_button.setEnabled(False)
        self.install_button.setEnabled(False)
        self.status.setText(message)
        self.status.show()
        self.bar.setValue(0)
        self.bar.show()

    def _progress(self, info: dict) -> None:
        percent = info.get("percent")
        if percent is not None:
            self.bar.setValue(int(max(0, min(100, percent))))
        if info.get("type") == "model_download_progress":
            done, total = info.get("bytes_done", 0), info.get("bytes_total", 0)
            self.status.setText(f"Downloading the translation pack: {human_bytes(done)} of {human_bytes(total)}" + (f" - {info['message']}" if info.get("message") else ""))

    def _translated_ok(self, doc: TranslationDoc) -> None:
        self.result_doc, self.chosen = doc, doc.target
        self.translated.emit(doc.target)
        self._busy = ""
        self.accept()

    def _installed_ok(self, _result: Any) -> None:
        self.status.setText("The translation pack is installed.")

    def _failed(self, message: str) -> None:
        log.warning("translation dialog task failed: %s", message)
        self._failure = message

    def _finished(self) -> None:
        busy, self._busy = self._busy, ""
        self._thread = None
        self.bar.hide()
        self.combo.setEnabled(True)
        self.install_button.setEnabled(True)
        self.cancel_button.setEnabled(True)
        failure, self._failure = self._failure, ""
        if failure:
            cancelled = "cancel" in failure.lower()
            self.refresh()                                  # what is possible now (Translate stays off while the pack is still missing, the install button stays visible)
            self.status.setText("Cancelled. Nothing was changed." if cancelled else f"{failure} Nothing was changed; you can try again.")
            self.status.show()
            return
        if self.result() == QDialog.Accepted:
            return
        self.refresh()

    def done(self, result: int) -> None:  # noqa: D401 - Qt override
        if self._thread is not None and self._thread.isRunning():
            self._thread.cancel()
            self._thread.wait(8000)
        super().done(result)
