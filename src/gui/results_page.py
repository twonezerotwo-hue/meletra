"""Results: what was produced, and one button for every output file."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFormLayout, QGridLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from .context import AppServices
from .widgets import card, format_clock, format_hms

log = logging.getLogger(__name__)

READABLE_SUFFIXES = (".txt", ".srt", ".vtt", ".json", ".md")

# (button text, file name inside the output folder)
OUTPUT_BUTTONS = [
    ("Open Clean Transcript", "local_clean_transcript.txt"),
    ("Open Raw Transcript", "local_raw_transcript.txt"),
    ("Open Timestamped Transcript", "local_transcript_with_timestamps.txt"),
    ("Open SRT", "local_transcript.srt"),
    ("Open VTT", "local_transcript.vtt"),
    ("Open JSON", "local_transcript.json"),
    ("Medical Terms", "medical_terms_detected.txt"),
    ("Possible Medical Errors", "possible_medical_errors.txt"),
    ("Quality Report", "quality_report.json"),
]


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def count_words(output_dir: Path) -> int:
    try:
        return len((Path(output_dir) / "local_raw_transcript.txt").read_text(encoding="utf-8").split())
    except OSError:
        return 0


def summarize(output_dir: Path, event: Optional[dict[str, Any]] = None) -> dict[str, str]:
    """The numbers of the results screen, read from the files the pipeline wrote (so the page is correct after a restart too).

    ``event`` is the worker's ``completed`` event; it fills in what ``metadata.json`` does not carry (for instance the word
    count when the optional quality audit was switched off).
    """
    from types import SimpleNamespace

    from ..archive.resolver import lecture_facts

    event = event or {}
    facts = lecture_facts(SimpleNamespace(state="completed", summary=event, duration_seconds=float(event.get("audio_seconds") or 0.0), detected_language=str(event.get("language") or ""),
                                          detected_languages=list(event.get("languages") or []), job_id="", source_name="", output_dir=str(output_dir)), Path(output_dir))
    meta = read_json(Path(output_dir) / "metadata.json")
    quality = meta.get("quality") or {}
    timing = meta.get("timing") or {}
    medical = meta.get("medical_terms") or {}
    subject = meta.get("subject") or {}
    language = (meta.get("language") or {})
    repairs = (meta.get("long_audio") or {}).get("gap_repairs") or []
    recovered = [g for g in repairs if g.get("inserted")]
    wall = timing.get("wall_time_cumulative") or timing.get("wall_seconds") or 0.0
    rtf = timing.get("real_time_factor_effective")
    first_pass = meta.get("first_decode_pass") or {}
    if rtf is None:
        rtf = (first_pass.get("timing") or {}).get("real_time_factor")
    names = {"en": "English", "tr": "Turkish"}
    detected = language.get("detected") or language.get("forced") or ""
    areas = subject.get("subjects") or []
    detected = detected or facts.language
    words = quality.get("words") or event.get("words") or facts.words
    return {
        "Status": facts.status_label,
        "Duration": format_hms(facts.duration_seconds) if facts.duration_seconds else "Unknown",
        "Language": names.get(detected, detected) if detected else "Unknown",
        "Subject": ", ".join(a.replace("_", " ").title() for a in areas) or "-",
        "Words": f"{words:,}" if words else "Unknown",
        "Medical terms": f"{medical.get('occurrences', 0):,} ({medical.get('distinct_detected', 0)} distinct)",
        "Uncertain segments": str((meta.get("uncertainty") or {}).get("uncertain_segments", 0)),
        "Recovered gaps": f"{len(recovered)} ({sum(float(g.get('seconds', 0)) for g in recovered):.0f} s of speech)" if recovered else "0",
        "Processing time": format_clock(wall) if wall else "Not recorded",
        "Real-time factor": f"{rtf:.2f}" if rtf else "-",
    }


class ResultsPage(QWidget):
    new_requested = Signal()
    view_requested = Signal(str)                      # a file of the lecture the reader can show: MainWindow opens it inside MELETRA

    def __init__(self, services: AppServices, parent=None) -> None:
        super().__init__(parent)
        self.services = services
        self.output_dir: Optional[Path] = None
        root = QVBoxLayout(self)
        root.setSpacing(12)
        self.title = QLabel("Results")
        self.title.setObjectName("title")
        root.addWidget(self.title)
        self.file_label = QLabel("")
        self.file_label.setObjectName("muted")
        root.addWidget(self.file_label)
        panel = card()
        self.form = QFormLayout(panel)
        self.values: dict[str, QLabel] = {}
        for name in ("Status", "Duration", "Language", "Subject", "Words", "Medical terms", "Uncertain segments",
                     "Recovered gaps", "Processing time", "Real-time factor"):
            label = QLabel("-")
            label.setObjectName("bigNumber")
            self.values[name] = label
            self.form.addRow(name + ":", label)
        root.addWidget(panel)
        self.buttons: dict[str, QPushButton] = {}
        grid = QGridLayout()
        for i, (text, filename) in enumerate(OUTPUT_BUTTONS):
            button = QPushButton(text)
            button.clicked.connect(lambda _c=False, f=filename: self.open_output(f))
            self.buttons[filename] = button
            grid.addWidget(button, i // 3, i % 3)
        root.addLayout(grid)
        row = QGridLayout()
        self.folder_button = QPushButton("Show files in Explorer")                  # a secondary, advanced action: reading happens inside MELETRA
        self.folder_button.clicked.connect(self.open_folder)
        self.new_button = QPushButton("Transcribe another lecture")
        self.new_button.clicked.connect(self.new_requested.emit)
        row.addWidget(self.folder_button, 0, 0)
        row.addWidget(self.new_button, 0, 1)
        root.addLayout(row)
        self.message = QLabel("")
        self.message.setObjectName("muted")
        self.message.setWordWrap(True)
        root.addWidget(self.message)
        root.addStretch(1)

    def show_results(self, output_dir: Path, source_name: str = "", event: Optional[dict[str, Any]] = None) -> None:
        self.output_dir = Path(output_dir)
        self.file_label.setText(f"{source_name}  -  {self.output_dir}")
        self.message.setText("")
        for name, text in summarize(self.output_dir, event).items():
            self.values[name].setText(text)
        for filename, button in self.buttons.items():
            button.setEnabled((self.output_dir / filename).is_file())

    def open_output(self, filename: str) -> None:
        """A file of the lecture opens INSIDE MELETRA (the reader has search and copy); only a file the reader cannot show goes to the system."""
        if self.output_dir is None:
            return
        path = self.output_dir / filename
        if not path.is_file():
            self.message.setText(f"{filename} was not created for this transcription.")
        elif path.suffix.lower() in READABLE_SUFFIXES:
            self.view_requested.emit(str(path))
        else:
            self.services.open_path(path)

    def open_folder(self) -> None:
        if self.output_dir is not None:
            self.services.open_path(self.output_dir)
