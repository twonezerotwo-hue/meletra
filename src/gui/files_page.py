"""Files: the Lecture Archive - every lecture ever transcribed, searchable, sortable and filterable. A completed transcript stays here even when the recording was deleted."""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtWidgets import QComboBox, QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget

from . import icons, tokens
from .components import EmptyState, PageHeader, SegmentedControl, name_clear_button
from .context import AppServices
from .lecture_cards import LectureActions, LectureCard

SORTS = (("Newest", "newest"), ("Oldest", "oldest"), ("Name A–Z", "name"), ("Duration", "duration"))
FILTERS = (("all", "All"), ("completed", "Completed"), ("processing", "Processing"), ("failed", "Failed"))
PROCESSING_STATES = ("running", "queued", "paused")
SHOWN_AT_ONCE = 200
SEARCH_DELAY_MS = 90                       # a light debounce: the list follows the typing without redrawing on every key


class FilesPage(QWidget):
    open_requested = Signal(object)            # JobRecord
    add_folder_requested = Signal()
    pdf_study_requested = Signal()
    repair_requested = Signal()

    def __init__(self, services: AppServices, parent=None) -> None:
        super().__init__(parent)
        self.services = services
        root = QVBoxLayout(self)
        root.setSpacing(tokens.SPACE["lg"])
        self.header = PageHeader("Lecture Archive", "Every lecture you transcribed. Transcripts stay available even if the recording is deleted.")
        root.addWidget(self.header)

        bar = QHBoxLayout()
        bar.setSpacing(tokens.SPACE["md"])
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search by lecture name or file name")
        self.search.setClearButtonEnabled(True)
        self.search.setAccessibleName("Search lectures")
        self.search.addAction(icons.icon("search", normal="text3", hover="text3", size=18), QLineEdit.LeadingPosition)
        name_clear_button(self.search, "Clear the search")
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(SEARCH_DELAY_MS)
        self._timer.timeout.connect(self.refresh)
        self.search.textChanged.connect(lambda _text: self._timer.start())
        self.sort_combo = QComboBox()
        for label, value in SORTS:
            self.sort_combo.addItem(label, value)
        self.sort_combo.setAccessibleName("Sort lectures")
        self.sort_combo.setMinimumWidth(150)
        self.sort_combo.currentIndexChanged.connect(self.refresh)
        bar.addWidget(self.search, 1)
        bar.addWidget(self.sort_combo)
        root.addLayout(bar)

        row = QHBoxLayout()
        row.setSpacing(tokens.SPACE["sm"])
        self.filter = SegmentedControl([(key, label) for key, label in FILTERS])
        self.filter.setAccessibleName("Show lectures by status")
        self.filter.changed.connect(lambda _key: self.refresh())
        row.addWidget(self.filter)
        row.addStretch(1)
        self.count_label = QLabel("")
        self.count_label.setObjectName("meta")
        row.addWidget(self.count_label)
        self.add_folder_button = QPushButton("Add a finished lecture folder ...")
        self.add_folder_button.setObjectName("smallButton")
        self.add_folder_button.setToolTip("Put a lecture folder made by MELETRA (for example from another disk) into this list")
        self.add_folder_button.clicked.connect(self.add_folder_requested.emit)
        self.repair_button = QPushButton("Repair lecture archive")
        self.repair_button.setObjectName("smallButton")
        self.repair_button.setToolTip("Look for lecture folders that are not in this list (or point to the wrong folder) and put them back. Nothing is deleted or changed in your transcripts.")
        self.repair_button.clicked.connect(self.repair_requested.emit)
        self.pdf_button = QPushButton("Study from a PDF only ...")
        self.pdf_button.setObjectName("smallButton")
        self.pdf_button.setToolTip("Make study material from slides without a recording")
        self.pdf_button.clicked.connect(self.pdf_study_requested.emit)
        for button, name in ((self.add_folder_button, "folder"), (self.repair_button, "folder"), (self.pdf_button, "pdf")):
            button.setIconSize(QSize(16, 16))
            icons.bind(button, name, size=16)
            button.setCursor(Qt.PointingHandCursor)
            row.addWidget(button)
        root.addLayout(row)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setFocusPolicy(Qt.NoFocus)
        self.holder = QWidget()
        self.list_layout = QVBoxLayout(self.holder)
        self.list_layout.setContentsMargins(0, 0, 6, 0)
        self.list_layout.setSpacing(tokens.SPACE["sm"])
        self.list_layout.addStretch(1)
        self.scroll.setWidget(self.holder)
        root.addWidget(self.scroll, 1)

        self.empty = EmptyState("files", "No lectures yet", "Drop a lecture on Home to get started.", min_height=240)
        self.empty.hide()
        self.list_layout.insertWidget(0, self.empty)

        self.actions = LectureActions(services, lambda record: self.open_requested.emit(record), self.refresh, self)
        self.cards: list[LectureCard] = []
        self.refresh()

    def search_now(self) -> None:
        """Apply the text typed in the search box without waiting for the short delay (tests, Enter)."""
        self._timer.stop()
        self.refresh()

    def records(self):
        records = self.services.jobs.archive(query=self.search.text(), sort=self.sort_combo.currentData() or "newest")
        mode = self.filter.current()
        if mode == "completed":
            return [r for r in records if r.state == "completed"]
        if mode == "processing":
            return [r for r in records if r.state in PROCESSING_STATES]
        if mode == "failed":
            return [r for r in records if r.state == "failed"]
        return records

    def refresh(self, *_args) -> None:
        self._timer.stop()
        for card in self.cards:
            card.hide()
            card.setParent(None)
            card.deleteLater()
        self.cards = []
        records = self.records()
        total = len(records)
        previous = self.pdf_button
        for record in records[:SHOWN_AT_ONCE]:
            card = LectureCard(record, self.actions, allow_wide=True)
            card.clicked.connect(self.open_requested.emit)
            self.list_layout.insertWidget(self.list_layout.count() - 1, card)
            self.cards.append(card)
            QWidget.setTabOrder(previous, card)                      # toolbar, then the lectures (card, its menu), then the sidebar
            QWidget.setTabOrder(card, card.menu_button)
            previous = card.menu_button
        searching = bool(self.search.text().strip()) or self.filter.current() != "all"
        if not records:
            self.empty.set_text("No lectures match your search." if searching else "No lectures yet",
                                "Try a different name or filter." if searching else "Drop a lecture on Home to get started.")
        self.empty.setVisible(not records)
        self.count_label.setText(f"{total} lecture{'s' if total != 1 else ''}" + (f" (showing {SHOWN_AT_ONCE})" if total > SHOWN_AT_ONCE else ""))
