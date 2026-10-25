"""Read the transcript inside the program: clean text by default, the raw text and the timestamped text one click away, search, copy."""

from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QGuiApplication, QTextBlockFormat, QTextCharFormat, QTextCursor, QTextDocument
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QPushButton, QTextEdit, QVBoxLayout, QWidget)

from ..language_catalogue import language_name
from ..multilingual import source as source_mod
from ..translation.outputs import file_stem
from ..translation.service import TranslationService
from . import icons, tokens
from .components import SegmentedControl, back_button, centered, name_clear_button
from .context import AppServices
from ..archive.artifacts import find_artifacts
from ..archive.transcript import load_transcript

log = logging.getLogger(__name__)

VIEWS = (
    ("Clean transcript", "local_clean_transcript.txt"),
    ("With times", "local_transcript_with_timestamps.txt"),
    ("Raw (exactly what the model wrote)", "local_raw_transcript.txt"),
    ("Medical corrected (terms checked in context)", "medical_corrected_transcript.txt"),
)

TRANSCRIPT_WIDTH = 900                                              # about 120 characters of text per line at the reading size


class TranscriptView(QWidget):
    back_requested = Signal()

    def __init__(self, services: AppServices, parent=None) -> None:
        super().__init__(parent)
        self.services = services
        self.folder: Optional[Path] = None
        self.translations: dict[str, Path] = {}                         # target language -> translated_<code>_transcript.txt of every saved translation of the lecture
        self.language = ""                                              # the tab being shown: "" = the original transcript, else a target language code
        outer = QVBoxLayout(self)
        content = QWidget()
        outer.addWidget(centered(content, TRANSCRIPT_WIDTH), 1)          # header, search, text and buttons in one column: a line of the text never runs across a whole screen
        root = QVBoxLayout(content)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(tokens.SPACE["md"])
        head = QHBoxLayout()
        head.setSpacing(tokens.SPACE["md"])
        self.back_button = back_button()
        self.back_button.clicked.connect(self.back_requested.emit)
        self.title = QLabel("Transcript")
        self.title.setObjectName("title")
        self.mode = SegmentedControl([("original", "Original")])
        self.mode.setAccessibleName("Show the original transcript or one of its translations")
        self.mode.changed.connect(self._tab_changed)
        self.mode.hide()
        head.addWidget(self.back_button)
        head.addWidget(self.title, 1)
        root.addLayout(head)
        root.addWidget(self.mode, 0, Qt.AlignLeft)

        bar = QHBoxLayout()
        bar.setSpacing(tokens.SPACE["sm"])
        self.view_combo = QComboBox()
        for label, filename in VIEWS:
            self.view_combo.addItem(label, filename)
        self.view_combo.setAccessibleName("Transcript version")
        self.view_combo.currentIndexChanged.connect(lambda _i: self._load())
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search in the transcript")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.addAction(icons.icon("search", normal="text3", hover="text3", size=18), QLineEdit.LeadingPosition)
        name_clear_button(self.search_edit, "Clear the search")
        self.search_edit.textChanged.connect(self._search_changed)
        self.search_edit.returnPressed.connect(self.find_next)
        self.prev_button = QPushButton("Previous")
        self.prev_button.setObjectName("smallButton")
        self.prev_button.clicked.connect(self.find_previous)
        self.next_button = QPushButton("Next")
        self.next_button.setObjectName("smallButton")
        self.next_button.clicked.connect(self.find_next)
        self.hits_label = QLabel("")
        self.hits_label.setObjectName("meta")
        bar.addWidget(self.view_combo)
        bar.addWidget(self.search_edit, 1)
        bar.addWidget(self.prev_button)
        bar.addWidget(self.next_button)
        bar.addWidget(self.hits_label)
        root.addLayout(bar)

        self.text = QPlainTextEdit()
        self.text.setObjectName("transcriptText")
        self.text.setAccessibleName("Transcript text")
        self.text.setReadOnly(True)
        self.text.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        root.addWidget(self.text, 1)

        foot = QHBoxLayout()
        foot.setSpacing(tokens.SPACE["sm"])
        self.copy_button = QPushButton("Copy all")
        self.copy_button.setObjectName("smallButton")
        self.copy_button.clicked.connect(self.copy_all)
        self.open_button = QPushButton("Open in another program")
        self.open_button.setObjectName("smallButton")
        self.open_button.clicked.connect(self.open_file)
        self.status = QLabel("")
        self.status.setObjectName("meta")
        foot.addWidget(self.copy_button)
        foot.addWidget(self.open_button)
        foot.addWidget(self.status, 1)
        root.addLayout(foot)

    # ------------------------------------------------------------------------------------------------------------------ content
    def show_document(self, folder: Path, text_file: Path, title: str = "") -> None:
        """The text of a converted document (PDF, picture, text file): one view, same search and copy."""
        self.folder = Path(folder)
        self.translations, self.language = {}, ""
        self.mode.hide()
        self.view_combo.show()
        self.title.setText(title or Path(text_file).name)
        self.view_combo.blockSignals(True)
        self.view_combo.clear()
        self.view_combo.addItem("Text", Path(text_file).name)
        self.view_combo.blockSignals(False)
        self.search_edit.clear()
        self._load()

    def show_transcript(self, folder: Path, title: str = "", prefer_translated: bool = False, language: str = "") -> None:
        """A lecture's transcript. Tabs ``Original | English | Turkish ...`` switch between the transcript in the spoken language(s) and every saved translation.
        ``language``: open that translation (when it exists); ``prefer_translated``: open the translation that was asked for when the lecture was made."""
        self.folder = Path(folder)
        self.title.setText(title or self.folder.name)
        asked = self._find_translations()
        self.view_combo.blockSignals(True)
        if self.view_combo.count() != len(VIEWS):                            # after a document was shown
            self.view_combo.clear()
            for label, filename in VIEWS:
                self.view_combo.addItem(label, filename)
        self.view_combo.setCurrentIndex(0)
        art = find_artifacts(self.folder)                                    # the same file rules as Study, the translation and the result screen (current AND legacy names)
        self._view_paths = {"local_clean_transcript.txt": art.clean, "local_transcript_with_timestamps.txt": art.timestamped or art.corrected_timestamped,
                            "local_raw_transcript.txt": art.raw, "medical_corrected_transcript.txt": art.corrected_text}
        self._synthetic = None
        for i, (_label, filename) in enumerate(VIEWS):                       # show only what this lecture has
            item = self.view_combo.model().item(i)
            if item is not None:
                item.setEnabled(self._view_paths.get(filename) is not None)
        if self._view_paths.get(VIEWS[0][1]) is None:
            for i, (_label, filename) in enumerate(VIEWS):
                if self._view_paths.get(filename) is not None:
                    self.view_combo.setCurrentIndex(i)
                    break
            else:                                                            # only a transcript JSON exists: read its text
                found = load_transcript(art)
                self._synthetic = found.text if found is not None else None
        self.view_combo.blockSignals(False)
        self.search_edit.clear()
        wanted = language if language in self.translations else (asked if (prefer_translated and asked in self.translations) else "")
        self.mode.set_current(wanted or "original")
        self.language = wanted
        self._mode_changed()

    def _find_translations(self) -> str:
        """Tabs for the saved translations of the lecture; returns the Output Language the lecture was made with ("" when none). A translation made from an older version of the
        transcript is not shown. Translations of lectures from 4.1.0 and earlier (``translated_<code>_transcript.txt`` only) are shown as they are."""
        self.translations = {}
        folder = self.folder
        asked = ""
        try:
            source = source_mod.load(folder)
            stored = TranslationService().stored_targets(folder, any_engine=True) if source is not None else {}
            record = (json.loads((folder / "metadata.json").read_text(encoding="utf-8")).get("translation") or {}) if (folder / "metadata.json").is_file() else {}
        except Exception:  # noqa: BLE001 - a damaged file never hides the transcript
            log.warning("could not read the saved translations of %s", folder, exc_info=True)
            source, stored, record = None, {}, {}
        asked = str(record.get("output_language") or "")
        for code in stored:
            path = folder / f"{file_stem(code)}.txt"
            if path.is_file():
                self.translations[code] = path
        for path in sorted(folder.glob("translated_*_transcript.txt")):                   # 4.1.0 and earlier: no revision to compare, shown as they are
            code = path.stem.split("_")[1]
            if code not in self.translations and not (folder / "translations" / f"{code}.json").is_file():
                self.translations[code] = path
        spoken = source.detected_languages if source is not None else []
        origin = " · ".join(language_name(c) for c in spoken) if spoken else (language_name(str(record.get("source_language") or "")) or "")
        items = [("original", f"Original · {origin}" if origin else "Original")] + [(code, language_name(code)) for code in self.translations]
        self.mode.set_items(items)
        self.mode.setVisible(bool(self.translations))
        return asked

    def _tab_changed(self, key: str) -> None:
        self.language = "" if key == "original" else key
        self._mode_changed()

    def showing_translation(self) -> bool:
        return bool(self.language) and self.language in self.translations

    def _mode_changed(self) -> None:
        self.view_combo.setVisible(not self.showing_translation())          # the translated transcript is one text; the versions below belong to the original
        self._load()

    @property
    def current_file(self) -> Optional[Path]:
        if self.showing_translation():
            return self.translations[self.language]
        if self.folder is None:
            return None
        paths = getattr(self, "_view_paths", None)
        key = self.view_combo.currentData() or VIEWS[0][1]
        if paths is not None and key in paths:
            return paths[key]
        return self.folder / key

    def _load(self) -> None:
        path = self.current_file
        if (path is None or not path.is_file()) and getattr(self, "_synthetic", None) and not self.showing_translation():
            content = self._synthetic
        elif path is None or not path.is_file():
            self.text.setPlainText("")
            self.status.setText("This transcript file was not found.")
            return
        else:
            try:
                content = path.read_text(encoding="utf-8-sig")
            except OSError as exc:
                self.text.setPlainText("")
                self.status.setText(f"The file could not be read ({exc.__class__.__name__}).")
                return
        self.text.setPlainText(content)
        cursor = QTextCursor(self.text.document())
        cursor.select(QTextCursor.Document)
        block = QTextBlockFormat()
        block.setLineHeight(150.0, int(QTextBlockFormat.ProportionalHeight.value))       # comfortable line height for reading
        block.setBottomMargin(6)
        cursor.mergeBlockFormat(block)
        self.text.moveCursor(QTextCursor.Start)
        words = len(content.split())
        self.status.setText(f"{words:,} words")
        self._search_changed(self.search_edit.text())

    # ------------------------------------------------------------------------------------------------------------------ search
    def _search_changed(self, query: str) -> None:
        selections = []
        if query:
            fmt = QTextCharFormat()
            fmt.setBackground(QColor(tokens.color("gold")))
            fmt.setForeground(QColor("#08121F"))
            cursor = QTextCursor(self.text.document())
            while True:
                cursor = self.text.document().find(query, cursor)
                if cursor.isNull():
                    break
                extra = QTextEdit.ExtraSelection()
                extra.cursor, extra.format = cursor, fmt
                selections.append(extra)
        self.text.setExtraSelections(selections)
        self.hits_label.setText(f"{len(selections)} found" if query else "")
        if query and selections:
            self.text.setTextCursor(selections[0].cursor)
            self.text.ensureCursorVisible()

    def find_next(self) -> None:
        self._step(forward=True)

    def find_previous(self) -> None:
        self._step(forward=False)

    def _step(self, forward: bool) -> None:
        query = self.search_edit.text()
        if not query:
            return
        flags = QTextDocument.FindFlag(0) if forward else QTextDocument.FindBackward
        if not self.text.find(query, flags):
            self.text.moveCursor(QTextCursor.Start if forward else QTextCursor.End)
            self.text.find(query, flags)

    # ------------------------------------------------------------------------------------------------------------------ actions
    def copy_all(self) -> None:
        QGuiApplication.clipboard().setText(self.text.toPlainText())
        self.status.setText("Copied to the clipboard.")

    def open_file(self) -> None:
        path = self.current_file
        if path is not None and path.is_file():
            self.services.open_path(path)
