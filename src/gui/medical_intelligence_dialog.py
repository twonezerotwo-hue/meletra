"""Settings > Medical Intelligence: what the medical knowledge database is (version, concepts, terms, sources, languages, profile, status) and a small search.

The switch is the only setting here: Off (the default - the v3 Medical Expert alone) or On - Experimental (the Medical Expert keeps every decision, Medical Intelligence v4 adds its
high-confidence corrections where the Medical Expert changed nothing). "Shadow" (compare only, transcript unchanged) is a developer mode: it is listed only when it is already the setting
(or set with MLT_V4_MODE=shadow). Any problem with the database falls back to the Medical Expert alone.
"""

from __future__ import annotations

from PySide6.QtWidgets import (QComboBox, QDialog, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QPushButton, QVBoxLayout)

from .tasks import TaskThread

EXPLANATION = "Advanced medical terminology correction. May improve difficult medical terms but remains experimental."
#: what a user can choose (the developer-only shadow mode is listed only when it is already the setting)
MODE_CHOICES = (("off", "Off (default)"), ("on", "On - Experimental"))


def _summary(info: dict) -> str:
    if not info.get("installed"):
        return info.get("status", "No medical knowledge database is installed.")
    c = info.get("counts", {})
    langs = ", ".join(f"{k} {v:,}" for k, v in info.get("languages", {}).items()) or "-"
    lines = [f"{info.get('name', 'MedicalKnowledge')} {info.get('build_version', '')}   (schema {info.get('schema_version', '?')}, profile: {info.get('profile', '')})",
             f"Concepts: {c.get('concepts', 0):,}     Terms: {c.get('terms', 0):,}     Abbreviations: {c.get('abbreviations', 0):,}",
             f"Languages: {langs}", f"File: {info.get('file', '')}  ({info.get('size_bytes', 0) / 1e6:.0f} MB)", "", "Sources:"]
    for s in info.get("sources", []):
        lines.append(f"  {s['name']} {s['version'] or ''} - {s['license_status']}, {'may be redistributed' if s['redistribution_allowed'] else 'private use only'}")
    lines += ["", f"Status: {info.get('status', '')}"]
    return "\n".join(lines)


def _search(path: str, query: str) -> str:
    from ..medical_knowledge.runtime.knowledge import MedicalKnowledge

    with MedicalKnowledge.open(path or None) as kb:
        hits = kb.lookup(query, limit=12)
        if not hits:
            return f"Nothing found for '{query}'."
        out = []
        for h in hits:
            out.append(f"{h.text}   ->   {h.preferred_name}   [{h.match}]   {', '.join(h.domains[:3])}")
        return "\n".join(out)


class MedicalIntelligenceDialog(QDialog):
    def __init__(self, services=None, parent=None) -> None:
        super().__init__(parent)
        self.services = services
        self._threads: list[TaskThread] = []
        self._file = ""
        self.setWindowTitle("Medical Intelligence")
        self.resize(640, 560)
        layout = QVBoxLayout(self)
        title = QLabel("Medical Intelligence v4 - Experimental / Advanced")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        intro = QLabel(EXPLANATION + " Off (default): the Medical Expert corrects medical terms on its own. If the knowledge database is missing or damaged, "
                       "the Medical Expert works alone and the transcription continues.")
        intro.setWordWrap(True)
        intro.setObjectName("muted")
        layout.addWidget(intro)
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Medical Intelligence v4:"))
        self.mode_combo = QComboBox()
        current = getattr(getattr(getattr(services, "settings", None), "settings", None), "medical_intelligence", "off") if services is not None else "off"
        for value, label in MODE_CHOICES + ((("shadow", "Shadow - compare only (developer)"),) if current == "shadow" else ()):
            self.mode_combo.addItem(label, value)
        self.mode_combo.setCurrentIndex(max(0, self.mode_combo.findData(current)))
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        mode_row.addWidget(self.mode_combo, 1)
        layout.addLayout(mode_row)
        self.info_text = QPlainTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setPlainText("Checking the medical knowledge database ...")
        layout.addWidget(self.info_text, 2)
        row = QHBoxLayout()
        self.query_edit = QLineEdit()
        self.query_edit.setPlaceholderText("Search medical knowledge, e.g. dendritic, Purkinje, aconitase")
        self.query_edit.returnPressed.connect(self.search)
        self.search_button = QPushButton("Search Medical Knowledge")
        self.search_button.clicked.connect(self.search)
        self.search_button.setEnabled(False)
        row.addWidget(self.query_edit, 1)
        row.addWidget(self.search_button)
        layout.addLayout(row)
        self.result_text = QPlainTextEdit()
        self.result_text.setReadOnly(True)
        layout.addWidget(self.result_text, 1)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        layout.addWidget(close)
        self.refresh()

    def _mode_changed(self) -> None:
        if self.services is not None:
            self.services.update_setting(medical_intelligence=self.mode_combo.currentData())

    def refresh(self) -> None:
        from ..medical_knowledge.cli import report

        thread = TaskThread(lambda progress, cancelled: report(), self)
        thread.succeeded.connect(self._loaded)
        thread.failed.connect(lambda msg: self.info_text.setPlainText(f"Could not read the medical knowledge database: {msg}"))
        self._threads.append(thread)
        thread.start()

    def _loaded(self, info: dict) -> None:
        self._file = info.get("file", "") if info.get("installed") else ""
        self.info_text.setPlainText(_summary(info))
        self.search_button.setEnabled(bool(self._file) and bool(info.get("ok")))

    def search(self) -> None:
        query = self.query_edit.text().strip()
        if not query or not self._file:
            return
        self.result_text.setPlainText("Searching ...")
        thread = TaskThread(lambda progress, cancelled: _search(self._file, query), self)
        thread.succeeded.connect(self.result_text.setPlainText)
        thread.failed.connect(lambda msg: self.result_text.setPlainText(f"Search failed: {msg}"))
        self._threads.append(thread)
        thread.start()

    def done(self, result: int) -> None:
        for thread in self._threads:
            thread.wait(2000)
        super().done(result)


def open_medical_intelligence(services=None, parent=None) -> None:
    MedicalIntelligenceDialog(services, parent).exec()
