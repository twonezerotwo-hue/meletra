"""The result screen (simple view): "Lecture Ready" (or "Document Ready") with Read / Listen / Create Study Material / Show in Explorer (the secondary one: reading happens inside MELETRA).

Quality reports, medical-term lists, uncertainty files and the numbers of the run are behind "Technical details".
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ..archive.artifacts import find_artifacts
from ..archive.transcript import load_transcript
from ..language_catalogue import language_name
from ..multilingual import source as source_mod
from ..translation.outputs import read_record, translated_text_file
from . import icons, tokens
from .components import Tile
from .context import AppServices
from .lecture_cards import ElidedLabel
from .results_page import summarize

log = logging.getLogger(__name__)

SECTIONS = (("summary", "Summary"), ("flashcards", "Flashcards"), ("questions", "Questions"))              # (reading the transcript is the primary button: no second way to the same place)
TILE_ICONS = {"summary": "study", "flashcards": "cards", "questions": "help"}


class SimpleResult(QWidget):
    read_requested = Signal()
    read_original_requested = Signal()        # the spoken-language transcript, when the screen shows the translated one
    listen_requested = Signal()
    study_requested = Signal(str)             # the tab to open: overview | summary | flashcards | questions
    details_requested = Signal()
    new_requested = Signal()
    translate_requested = Signal()            # translate the SAVED transcript (no file, no new job)
    reprocess_requested = Signal()            # read the ORIGINAL recording again (a lecture made by an earlier version): a separate action from translating

    def __init__(self, services: AppServices, parent=None) -> None:
        super().__init__(parent)
        self.services = services
        self.output_dir: Optional[Path] = None
        self.kind = "lecture"                       # lecture (a transcript) | document (text from a PDF, a picture or a text file)
        self.text_file: Optional[Path] = None
        self.translated_file: Optional[Path] = None     # translated_<code>_transcript.txt of the lecture, when the Output Language was made
        outer = QVBoxLayout(self)
        outer.addStretch(1)
        column = QWidget()
        column.setMaximumWidth(780)
        column_layout = QVBoxLayout(column)
        column_layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(column, 100)
        row.addStretch(1)
        outer.addLayout(row)
        outer.addStretch(2)

        panel = QFrame()
        panel.setObjectName("card")
        column_layout.addWidget(panel)
        root = QVBoxLayout(panel)
        root.setContentsMargins(32, 24, 32, 24)
        root.setSpacing(tokens.SPACE["md"])

        head = QHBoxLayout()
        head.setSpacing(tokens.SPACE["lg"])
        self.badge = QLabel()
        self.badge.setPixmap(icons.badge_pixmap("check", 56, 28, background="success_bg", foreground="success"))
        head.addWidget(self.badge, 0, Qt.AlignTop)
        titles = QVBoxLayout()
        titles.setSpacing(tokens.SPACE["xs"])
        self.title = QLabel("Lecture Ready")
        self.title.setObjectName("title")
        self.file_label = ElidedLabel("")
        self.file_label.setObjectName("fileName")
        self.info_label = QLabel("")
        self.info_label.setObjectName("muted")
        self.info_label.setWordWrap(True)
        titles.addWidget(self.title)
        titles.addWidget(self.file_label)
        titles.addWidget(self.info_label)
        head.addLayout(titles, 1)
        root.addLayout(head)
        root.addSpacing(tokens.SPACE["xs"])

        buttons = QHBoxLayout()
        buttons.setSpacing(tokens.SPACE["sm"])
        self.read_button = QPushButton("Read Transcript")
        self.read_button.setObjectName("primary")
        self.read_button.clicked.connect(self.read_requested.emit)
        self.listen_button = QPushButton("Listen to Text")
        self.listen_button.clicked.connect(self.listen_requested.emit)
        self.study_button = QPushButton("Create Study Material")
        self.study_button.clicked.connect(lambda: self.study_requested.emit("overview"))
        self.files_button = QPushButton("Show in Explorer")
        self.files_button.clicked.connect(self.open_files)
        for b in (self.read_button, self.study_button, self.listen_button, self.files_button):
            b.setCursor(Qt.PointingHandCursor)
            buttons.addWidget(b)
        buttons.addStretch(1)
        root.addLayout(buttons)

        grid = QGridLayout()
        grid.setSpacing(tokens.SPACE["md"])
        self.tiles: dict[str, Tile] = {}
        for i, (key, name) in enumerate(SECTIONS):
            tile = Tile(TILE_ICONS[key], name)
            tile.clicked.connect(lambda _c=False, k=key: self._tile_clicked(k))
            self.tiles[key] = tile
            grid.addWidget(tile, 0, i)
            grid.setColumnStretch(i, 1)
        root.addLayout(grid)

        self.message = QLabel("")
        self.message.setObjectName("banner")
        self.message.setWordWrap(True)
        self.message.hide()
        root.addWidget(self.message)

        self.legacy_row = QWidget()                                   # a lecture made by an earlier version: one language was recorded for the whole lecture
        legacy = QHBoxLayout(self.legacy_row)
        legacy.setContentsMargins(0, 0, 0, 0)
        legacy_note = QLabel("Made by an earlier version, which recorded one language for the whole lecture.")
        legacy_note.setObjectName("muted")
        legacy_note.setWordWrap(True)
        self.reprocess_button = QPushButton("Reprocess Original Audio")
        self.reprocess_button.setObjectName("link")
        self.reprocess_button.setToolTip("Reads the original recording again with the new language detection. Your current transcript is kept.")
        self.reprocess_button.clicked.connect(self.reprocess_requested.emit)
        legacy.addWidget(legacy_note, 1)
        legacy.addWidget(self.reprocess_button)
        self.legacy_row.hide()
        root.addWidget(self.legacy_row)

        foot = QHBoxLayout()
        self.details_button = QPushButton("Technical details")
        self.details_button.setObjectName("link")
        self.details_button.clicked.connect(self.details_requested.emit)
        self.translate_button = QPushButton("Translate to another language")
        self.translate_button.setAccessibleName("Translate to another language")
        self.translate_button.setToolTip("Translate the saved transcript of this lecture. No file is needed and nothing is transcribed again.")
        self.translate_button.clicked.connect(self.translate_requested.emit)
        self.new_button = QPushButton("Transcribe another lecture")
        self.new_button.clicked.connect(self.new_requested.emit)
        foot.addWidget(self.details_button)
        foot.addStretch(1)
        foot.addWidget(self.translate_button)
        foot.addWidget(self.new_button)
        root.addLayout(foot)

    def retheme(self) -> None:
        self.badge.setPixmap(icons.badge_pixmap("check", 56, 28, background="success_bg", foreground="success"))
        for tile in self.tiles.values():
            tile.retheme()

    # ------------------------------------------------------------------------------------------------------------------ content
    def show_results(self, output_dir: Path, source_name: str = "", event: Optional[dict[str, Any]] = None,
                     text_file: Optional[Path] = None) -> None:
        """``text_file`` given: a converted document (PDF, picture, text file) instead of a transcribed recording."""
        self.output_dir = Path(output_dir)
        self.kind, self.text_file = ("document" if text_file else "lecture"), (Path(text_file) if text_file else None)
        document = self.kind == "document"
        self.title.setText("Document Ready" if document else "Lecture Ready")
        self.read_button.setText("Read Text" if document else "Read Transcript")
        self.details_button.setVisible(not document)               # the technical reports belong to a transcription
        self.translate_button.setVisible(not document)
        self.legacy_row.hide()
        self.message.setText("")
        self.message.hide()
        self.translated_file = None
        self.file_label.setText(source_name or self.output_dir.name)
        if document:
            self.info_label.setText(self._document_info())
        else:
            numbers = summarize(self.output_dir, event)
            parts = [numbers.get("Duration"), f"{numbers.get('Words')} words" if numbers.get("Words") not in (None, "0") else None,
                     numbers.get("Language") if numbers.get("Language") not in (None, "-") else None,
                     numbers.get("Subject") if numbers.get("Subject") not in (None, "-") else None]
            self._info_parts = [p for p in parts if p and p not in ("--:--:--", "00:00:00")]
            self._language_part = numbers.get("Language") if numbers.get("Language") not in (None, "-") else None
            spoken = self._spoken_languages()
            if len(spoken) > 1 and self._language_part in self._info_parts:        # several languages were recognised: say all of them, not just the first
                names = " · ".join(language_name(c) for c in spoken)
                self._info_parts[self._info_parts.index(self._language_part)] = names
                self._language_part = names
            self.info_label.setText("  ·  ".join(self._info_parts))
            saved = source_mod.load(self.output_dir)
            self.translate_button.setEnabled(saved is not None)
            self.legacy_row.setVisible(saved is not None and saved.origin == source_mod.ORIGIN_LEGACY)
            self._show_translation()
        self.message.setVisible(bool(self.message.text()))
        self.refresh_study()
        self.read_button.setEnabled(self._transcript_file() is not None)
        self.listen_button.setEnabled(self._transcript_file() is not None)

    def _show_translation(self) -> None:
        """Output Language: read the translated transcript, keep the original one a click away, and say plainly when the translation was not made."""
        record = read_record(self.output_dir) or {}
        target = str(record.get("output_language") or "")
        found = translated_text_file(self.output_dir, target) if record.get("files") else None
        if found is not None:
            self.translated_file = found
            self.read_button.setText(f"Read Transcript ({language_name(target)})")
            parts = [p for p in getattr(self, "_info_parts", []) if p != getattr(self, "_language_part", None)]         # "Turkish → English" says it, "English" alone would repeat it
            spoken = self._spoken_languages()
            origin = " · ".join(language_name(c) for c in spoken) if len(spoken) > 1 else language_name(str(record.get("source_language") or ""))
            parts.append(f"{origin} → {language_name(target)}")
            self.info_label.setText("  ·  ".join(parts))
        elif record.get("method") == "model_missing":
            self.message.setText(f"Translation to {language_name(target)} isn't available on this computer yet. Your transcript in the spoken language is saved and complete.")
            self.message.setToolTip("The offline translation files are not installed: put the folder 'translation-m2m100' into the program's models folder.")
        elif record.get("method") == "failed":
            self.message.setText(f"The {language_name(target)} translation could not be made. Your transcript in the spoken language is saved and complete.")

    def _spoken_languages(self) -> list[str]:
        """The languages that were actually recognised in the lecture (from its saved transcript); [] when it has none."""
        try:
            source = source_mod.load(self.output_dir) if self.output_dir is not None else None
        except Exception:  # noqa: BLE001 - a damaged file never hides the transcript
            log.warning("could not read the saved transcript of %s", self.output_dir, exc_info=True)
            return []
        return source.detected_languages if source is not None else []

    def _document_info(self) -> str:
        """Where the text came from, as the conversion recorded it (embedded PDF text, or OCR with its real confidence if it has one)."""
        try:
            data = json.loads(next(self.output_dir.glob("*.json")).read_text(encoding="utf-8"))
            parts = [data.get("label"), f"{len(data.get('pages', []))} pages" if len(data.get("pages", [])) > 1 else None,
                     f"confidence {data['confidence'] * 100:.0f}%" if data.get("confidence") is not None else None]
            return "  ·  ".join(p for p in parts if p)
        except (StopIteration, OSError, ValueError, KeyError, TypeError):
            return ""

    def refresh_study(self) -> None:
        """Say what is already made: the study pack is created the first time the student opens Study Lecture."""
        folder = self.output_dir
        has_text = folder is not None and self._transcript_file() is not None
        pack = None
        if folder is not None:
            try:
                pack = self.services.study.open(folder).pack
            except Exception:                                # noqa: BLE001 - a damaged study file never hides the transcript
                log.warning("could not read the study files of %s", folder, exc_info=True)
        counts = {
            "summary": sum(len(v) for v in pack.summaries.values()) if pack else 0,
            "flashcards": len(pack.flashcards) if pack else 0,
            "questions": len(pack.questions) if pack else 0,
        }
        names = {"summary": "points", "flashcards": "cards", "questions": "questions"}
        for key in ("summary", "flashcards", "questions"):
            label = dict(SECTIONS)[key]
            if counts[key]:
                self.tiles[key].setText(f"{label}\nReady - {counts[key]} {names[key]}")
            else:
                self.tiles[key].setText(f"{label}\nMade when you press Create Study Material")
        self.study_button.setEnabled(has_text or (folder is not None and (folder / "study").is_dir()))

    def _transcript_file(self) -> Optional[Path]:
        if self.output_dir is None:
            return None
        if self.text_file is not None:
            return self.text_file if self.text_file.is_file() else None
        if self.translated_file is not None and self.translated_file.is_file():
            return self.translated_file                         # the Output Language is what is read, listened to and studied
        art = find_artifacts(self.output_dir)
        for path in (art.clean, art.raw):          # the plain-text transcripts (current names, then <name>_clean.txt / _raw.txt copies)
            if path is not None:
                return path
        found = load_transcript(art)                                                       # only a timestamped text or a transcript JSON: Read still works (the reader shows the text)
        return found.path if found is not None else None

    def _tile_clicked(self, key: str) -> None:
        self.study_requested.emit(key)

    def set_listening(self, on: bool) -> None:
        self.listen_button.setText("Stop Listening" if on else "Listen to Text")

    def open_files(self) -> None:
        if self.output_dir is not None:
            self.services.open_path(self.output_dir)
