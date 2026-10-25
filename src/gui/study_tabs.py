"""The tabs of the study workspace: Overview, Summary, Key Concepts, Flashcards, Questions, Study Mode.

Everything here shows ``StudyPack`` data and records the student's answers through the progress file; the generation itself lives in
``src/study`` and ``src/services/study_service.py``. Every item shows where in the lecture it comes from (a small "Slide 14" /
"Transcript 12:30-12:58" link that opens the passage).
"""

from __future__ import annotations

import html
from datetime import date
from typing import Callable, Optional, Sequence

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup, QCheckBox, QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QPlainTextEdit, QPushButton,
    QRadioButton, QScrollArea, QSizePolicy, QSplitter, QStackedWidget, QTextBrowser, QVBoxLayout, QWidget,
)

from ..study.index import DocumentIndex
from ..study.models import (CARD_TYPE_TITLES, DIFFICULTY_TITLES, QUESTION_TYPE_TITLES, SUMMARY_TITLES, Flashcard, Question, SourceRef, StudyPack)
from ..study.objectives import derive_objectives
from ..study.scheduler import QUALITY, next_schedule, when
from ..study.session import (FOCUS_ALL, FOCUS_NEW, FOCUS_WEAK, Feedback, FlashcardSession, QuizSession, build_deck, build_quiz)
from ..study.store import CORRECT, INCORRECT, MASTERED, NEW, WEAK, Progress
from . import friendly, tokens
from .components import CARD_WIDTH, READING_WIDTH, centered

SourceHandler = Callable[[SourceRef], None]
OBJECTIVES = "objectives"                  # the entry of the Summary list that shows the learning objectives (it is not one of the pack's summary kinds)


def _label(text: str, name: str = "", wrap: bool = True) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(wrap)
    label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    if name:
        label.setObjectName(name)
    return label


def _fit(*boxes: QComboBox) -> None:
    for box in boxes:
        box.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        box.setMinimumContentsLength(10)


def _clear(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.hide()                                        # gone at once, not when the event loop gets to deleteLater
            widget.deleteLater()
        elif item.layout() is not None:
            _clear(item.layout())


class SourceChips(QWidget):
    """A row of small links, one per source of an item."""

    def __init__(self, refs: Sequence[SourceRef], handler: SourceHandler, parent=None) -> None:
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(tokens.SPACE["sm"])
        if refs:
            tag = QLabel("Source:")
            tag.setObjectName("muted")
            row.addWidget(tag)
        for ref in refs:
            button = QPushButton(ref.label)
            button.setObjectName("chip")
            button.setCursor(Qt.PointingHandCursor)
            button.clicked.connect(lambda _c=False, r=ref: handler(r))
            row.addWidget(button)
        row.addStretch(1)


class PackBrowser(QTextBrowser):
    """Rich text list of items with clickable source links (``chunk:<id>``)."""

    source_clicked = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setOpenLinks(False)
        self.setOpenExternalLinks(False)
        self.anchorClicked.connect(self._clicked)
        self.setReadOnly(True)
        self.setObjectName("reading")
        self.setFrameShape(QFrame.NoFrame)
        self.document().setDocumentMargin(tokens.SPACE["sm"])         # the callers keep it in a column of about 80 characters (see ``centered``)

    @staticmethod
    def stylesheet() -> str:
        c = tokens.current()
        f = tokens.FONT_PX
        return (f"body {{ color: {c.text}; font-size: {f['reading']}px; }} "
                f"h3 {{ font-size: {f['section']}px; font-weight: 600; color: {c.text}; margin-top: 24px; margin-bottom: 6px; }} "
                f"p.term {{ margin: 0px; font-weight: 600; }} "
                f"p.text {{ margin: 0px 0px 14px 0px; }} "
                f"p.note {{ margin: 0px 0px 14px 0px; color: {c.text3}; font-size: {f['secondary']}px; }} "
                f"a {{ color: {c.teal}; text-decoration: none; }}")

    def _clicked(self, url: QUrl) -> None:
        if url.scheme() == "chunk":
            self.source_clicked.emit(url.path() or url.toString().split(":", 1)[1])

    @staticmethod
    def source_links(refs: Sequence[SourceRef]) -> str:
        return " ".join(f'<a href="chunk:{html.escape(r.chunk_id)}">[{html.escape(r.label).replace(" ", "&nbsp;")}]</a>' for r in refs)

    def show_groups(self, groups: Sequence[tuple[str, Sequence[tuple[str, str, Sequence[SourceRef]]]]], empty: str) -> None:
        """``groups``: (heading, [(term, text, sources)]). A term is shown in bold in front of the text."""
        parts: list[str] = []
        for heading, items in groups:
            if not items:
                continue
            if heading:
                parts.append(f"<h3{' style=' + chr(34) + 'margin-top:0px' + chr(34) if not parts else ''}>{html.escape(heading)}</h3>")
            for item in items:
                term, text, refs = item[0], item[1], item[2]
                note = item[3] if len(item) > 3 else ""
                if term:
                    parts.append(f"<p class='term' style='line-height:150%'>{html.escape(term)}</p>")
                tight = " margin-bottom:2px;" if note else ""             # a note belongs to its text: close to it, then the gap to the next item
                parts.append(f"<p class='text' style='line-height:150%;{tight}'>{html.escape(text)} {self.source_links(refs)}</p>")
                if note:
                    parts.append(f"<p class='note'>{html.escape(note.replace(' - ', '  ·  ', 1))}</p>")
        self.document().setDefaultStyleSheet(self.stylesheet())
        self.setHtml("".join(parts) or f"<p class='note'>{html.escape(empty)}</p>")


# ===================================================================================================================== Overview
class OverviewTab(QWidget):
    start_requested = Signal(str)           # tab name
    regenerate_requested = Signal()
    add_pdf_requested = Signal()
    ask_requested = Signal(str)
    export_requested = Signal()
    source_clicked = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, tokens.SPACE["lg"], 0, 0)
        body = QWidget()
        outer.addWidget(centered(body, READING_WIDTH + 70), 1)
        root = QVBoxLayout(body)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(tokens.SPACE["md"])
        self.heading = QLabel("")
        self.heading.setObjectName("title")
        self.heading.setWordWrap(True)
        root.addWidget(self.heading)
        self.made_from = _label("", "banner")
        root.addWidget(self.made_from)
        self.detail_note = _label("", "meta")                       # how it was checked: small print, not a banner
        root.addWidget(self.detail_note)
        self.warnings = _label("", "bannerWarn")
        self.warnings.hide()
        root.addWidget(self.warnings)
        self.counts = _label("", "statusLine")
        root.addWidget(self.counts)
        self.progress = _label("", "muted")
        root.addWidget(self.progress)
        self.topics = _label("", "muted")
        root.addWidget(self.topics)
        buttons = QHBoxLayout()
        self.flash_button = QPushButton("Practice flashcards")
        self.flash_button.setObjectName("primary")
        self.flash_button.clicked.connect(lambda: self.start_requested.emit("flashcards"))
        self.study_button = QPushButton("Start Study Mode")
        self.study_button.clicked.connect(lambda: self.start_requested.emit("study"))
        self.pdf_button = QPushButton("Add lecture PDF ...")
        self.pdf_button.clicked.connect(self.add_pdf_requested.emit)
        self.regen_button = QPushButton("Make new study material")
        self.regen_button.clicked.connect(self.regenerate_requested.emit)
        self.export_button = QPushButton("Export study guide ...")
        self.export_button.clicked.connect(self.export_requested.emit)
        for b in (self.study_button, self.pdf_button, self.regen_button, self.export_button):
            b.hide()                                                # the header of the workspace has Start Study Mode, Add PDF, Regenerate and Export: no second set here
        buttons.addWidget(self.flash_button)
        buttons.addStretch(1)
        root.addLayout(buttons)
        self.export_note = _label("", "muted")
        root.addWidget(self.export_note)

        ask_row = QHBoxLayout()
        self.ask_edit = QLineEdit()
        self.ask_edit.setPlaceholderText("Ask something about this lecture ...")
        self.ask_edit.returnPressed.connect(self._ask)
        self.ask_button = QPushButton("Ask")
        self.ask_button.clicked.connect(self._ask)
        ask_row.addWidget(self.ask_edit, 1)
        ask_row.addWidget(self.ask_button)
        root.addLayout(ask_row)
        self.answer = PackBrowser()
        self.answer.setAccessibleName("Answer from your lecture")
        self.answer.setMaximumHeight(170)
        self.answer.hide()
        self.answer.source_clicked.connect(self.source_clicked.emit)
        root.addWidget(self.answer)
        root.addStretch(1)

    def _ask(self) -> None:
        question = self.ask_edit.text().strip()
        if question:
            self.ask_button.setEnabled(False)
            self.answer.setHtml("<p><i>Looking it up in your lecture ...</i></p>")
            self.answer.show()
            self.ask_requested.emit(question)

    def show_answer(self, text: str, refs: Sequence[SourceRef]) -> None:
        self.ask_button.setEnabled(True)
        self.answer.show_groups([("", [("", text, refs)])], "")
        self.answer.show()

    def show_pack(self, pack: StudyPack, progress: Progress, provider_line: str, pdf_name: str = "", notes: Sequence[str] = ()) -> None:
        self.heading.setText(pack.title or "Your lecture")
        sources = {"pdf+transcript": "Made from your lecture slides and the transcript. Where both cover a topic, items point to both.",
                   "transcript": "Made from the transcript. Add the lecture PDF to link every item to its slide.",
                   "pdf": "Made from the lecture slides only (no recording)."}[pack.source_mode]
        bound = ("Source-bound: every item comes from your lecture. Nothing is added from outside." if pack.source_bound
                 else "Source-bound is off: items may include general knowledge.")
        dropped = int((pack.stats.get("grounding") or {}).get("dropped", 0))
        checked = (f"\n{dropped} generated item{'s' if dropped != 1 else ''} left out because the lecture does not support "
                   f"{'them' if dropped != 1 else 'it'}." if dropped else "")
        facts = (pack.overview or {}).get("facts") or {}
        validated = (f"Built from {facts['validated']} validated statements out of {facts['sentences']} sentences: every statement is complete, "
                     f"single-topic and traced to the seconds it was said." if facts else "")
        self.made_from.setText(f"{sources}\n{bound}{checked}" + (f"\nSlides: {pdf_name}" if pdf_name else ""))
        premium = pack.stats.get("premium") or {}
        refused = sum(sum(((premium.get(area) or {}).get("reasons") or {}).values()) for area in ("flashcards", "questions"))
        gate = (f"Quality check: {len(pack.flashcards)} flashcards and {len(pack.questions)} questions kept; {refused} weaker item{'s' if refused != 1 else ''} left out "
                f"(unclear, repeated, too easy to guess or not backed by the lecture)." if premium else "")
        self.detail_note.setText("  ·  ".join(part for part in (validated, gate, f"Study engine: {provider_line}") if part))
        self.detail_note.setWordWrap(True)
        shown = list(dict.fromkeys([*pack.warnings, *notes]))
        self.warnings.setText("\n".join(shown))
        self.warnings.setVisible(bool(shown))
        total_summary = sum(len(v) for v in pack.summaries.values())
        self.counts.setText(f"{total_summary} summary points  ·  {len(pack.flashcards)} flashcards  ·  {len(pack.questions)} questions")
        c = progress.counts(pack)
        self.progress.setText(
            f"Flashcards: {c['cards'][MASTERED]} mastered, {c['cards'][WEAK]} to review, {c['cards'][NEW]} new.     "
            f"Questions: {c['questions'][MASTERED]} mastered, {c['questions'][WEAK]} to review, {c['questions'][NEW]} new.")
        self.topics.setText("Topics: " + ("; ".join(pack.topics) if pack.topics else "-"))


# ===================================================================================================================== Summary
class SummaryTab(QWidget):
    source_clicked = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.pack: Optional[StudyPack] = None
        root = QHBoxLayout(self)
        root.setSpacing(tokens.SPACE["lg"])
        self.objectives: list = []
        self.kinds = QListWidget()
        self.kinds.setObjectName("kindList")
        self.kinds.setFixedWidth(252)
        self.kinds.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.kinds.setAccessibleName("Kinds of summary")
        self.kinds.currentRowChanged.connect(self._show_kind)
        self.browser = PackBrowser()
        self.browser.setAccessibleName("Summary text")
        self.browser.source_clicked.connect(self.source_clicked.emit)
        column = QWidget()
        right = QVBoxLayout(column)
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(tokens.SPACE["xs"])
        self.kind_title = QLabel("")
        self.kind_title.setObjectName("title")
        self.kind_meta = QLabel("")
        self.kind_meta.setObjectName("meta")
        right.addWidget(self.kind_title)
        right.addWidget(self.kind_meta)
        right.addSpacing(tokens.SPACE["sm"])
        right.addWidget(self.browser, 1)
        root.addWidget(self.kinds)
        root.addWidget(centered(column, READING_WIDTH + 30), 1)

    def show_pack(self, pack: StudyPack) -> None:
        self.pack = pack
        self.kinds.blockSignals(True)
        self.kinds.clear()
        self.objectives = derive_objectives(pack)                          # worked out from the pack's own validated items: nothing to store, works for older packs too
        entry = QListWidgetItem(f"Learning Objectives  ({len(self.objectives)})")
        entry.setData(Qt.UserRole, OBJECTIVES)
        if not self.objectives:
            entry.setFlags(entry.flags() & ~Qt.ItemIsEnabled)
        self.kinds.addItem(entry)
        for kind, title in SUMMARY_TITLES.items():
            if kind == "key_concepts" or kind == "definitions":
                continue                                                    # these two have their own tab
            items = pack.summaries.get(kind, [])
            entry = QListWidgetItem(f"{title}  ({len(items)})")
            entry.setData(Qt.UserRole, kind)
            if not items:
                entry.setFlags(entry.flags() & ~Qt.ItemIsEnabled)
            self.kinds.addItem(entry)
        self.kinds.blockSignals(False)
        first = next((i for i in range(self.kinds.count()) if self.kinds.item(i).flags() & Qt.ItemIsEnabled and self.kinds.item(i).data(Qt.UserRole) != OBJECTIVES), -1)
        if first < 0 and self.objectives:
            first = 0                                                      # no summary could be made, the objectives are what there is
        self.kinds.setCurrentRow(first)
        if first < 0:
            self.browser.show_groups([], "No summary could be made from these sources.")

    def select_kind(self, kind: str) -> None:
        for i in range(self.kinds.count()):
            if self.kinds.item(i).data(Qt.UserRole) == kind:
                self.kinds.setCurrentRow(i)

    def _show_kind(self, row: int) -> None:
        if self.pack is None or row < 0:
            return
        kind = self.kinds.item(row).data(Qt.UserRole)
        if kind == OBJECTIVES:
            self._show_objectives()
            return
        by_topic: dict[str, list] = {}
        for item in self.pack.summaries.get(kind, []):
            by_topic.setdefault(item.topic, []).append((item.term, item.text, item.sources))
        grouped = kind in ("detailed", "high_yield", "mechanisms", "clinical") and len(by_topic) > 1
        groups = [(topic, items) for topic, items in by_topic.items()] if grouped else \
            [("", [x for items in by_topic.values() for x in items])]
        count = sum(len(items) for _topic, items in groups)
        self.kind_title.setText(SUMMARY_TITLES.get(kind, kind))
        self.kind_meta.setText(friendly.plural(count, "point") + (f"  ·  {len(by_topic)} topics" if len(by_topic) > 1 else ""))
        self.browser.show_groups(groups, "Nothing in this lecture matches this kind of summary.")

    def _show_objectives(self) -> None:
        by_topic: dict[str, list] = {}
        for o in self.objectives:
            by_topic.setdefault(o.topic, []).append(("", o.text, o.sources, o.practice))
        groups = list(by_topic.items()) if len(by_topic) > 1 else [("", [x for items in by_topic.values() for x in items])]
        self.kind_title.setText("Learning Objectives")
        self.kind_meta.setText("After studying this lecture you should be able to:  ·  " + friendly.plural(len(self.objectives), "objective"))
        self.browser.show_groups(groups, "No objectives could be made: the study material has no concept or mechanism question yet.")

    @property
    def current_kind(self) -> str:
        item = self.kinds.currentItem()
        return item.data(Qt.UserRole) if item else ""


class ConceptsTab(QWidget):
    source_clicked = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        self.browser = PackBrowser()
        self.browser.setAccessibleName("Key concepts and definitions")
        self.browser.source_clicked.connect(self.source_clicked.emit)
        root.addWidget(centered(self.browser, READING_WIDTH + 30))

    def show_pack(self, pack: StudyPack) -> None:
        concepts = [(i.term, i.text, i.sources, i.importance) for i in pack.summaries.get("key_concepts", [])]
        definitions = [(i.term, i.text, i.sources) for i in pack.summaries.get("definitions", [])]
        self.browser.show_groups([("Key concepts", concepts), ("Important definitions", definitions)],
                                 "No key concepts could be found in these sources.")


# ===================================================================================================================== Flashcards
class FlashcardsTab(QWidget):
    source_clicked = Signal(str)
    changed = Signal()                      # a rating was saved (the overview refreshes)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.pack: Optional[StudyPack] = None
        self.progress = Progress()
        self.session: Optional[FlashcardSession] = None
        self._revealed = False
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        content = QWidget()
        outer.addWidget(centered(content, 1160), 1)                  # the choices and the card share one column, centred
        root = QVBoxLayout(content)
        root.setContentsMargins(0, tokens.SPACE["md"], 0, 0)
        root.setSpacing(tokens.SPACE["md"])
        filters = QHBoxLayout()
        filters.setSpacing(tokens.SPACE["sm"])
        self.topic_combo, self.type_combo, self.level_combo, self.focus_combo = QComboBox(), QComboBox(), QComboBox(), QComboBox()
        _fit(self.topic_combo, self.type_combo, self.level_combo, self.focus_combo)
        self.shuffle_check = QCheckBox("Shuffle")
        self.shuffle_check.setChecked(True)
        for label, box in (("Topic", self.topic_combo), ("Type", self.type_combo), ("Level", self.level_combo), ("Show", self.focus_combo)):
            caption = QLabel(label)
            caption.setObjectName("fieldLabel")
            box.setAccessibleName(label)
            filters.addWidget(caption)
            filters.addWidget(box)
            filters.addSpacing(tokens.SPACE["sm"])
        filters.addWidget(self.shuffle_check)
        self.start_button = QPushButton("Start")
        self.start_button.setObjectName("primary")
        self.start_button.clicked.connect(self.start)
        filters.addWidget(self.start_button)
        filters.addStretch(1)
        root.addLayout(filters)

        body = QSplitter(Qt.Horizontal)
        body.setChildrenCollapsible(False)
        main = QWidget()
        main_layout = QVBoxLayout(main)
        main_layout.setContentsMargins(0, 0, 0, 0)
        self.stack = QStackedWidget()
        main_layout.addWidget(self.stack, 1)
        self.empty_page = _label("", "muted")
        self.empty_page.setAlignment(Qt.AlignCenter)
        self.stack.addWidget(self.empty_page)
        card_page = QWidget()
        cp = QVBoxLayout(card_page)
        self.counter = QLabel("")
        self.counter.setObjectName("muted")
        cp.addWidget(self.counter)
        self.card_frame = QFrame()
        self.card_frame.setObjectName("flashCard")
        cf = QVBoxLayout(self.card_frame)
        cf.setContentsMargins(tokens.SPACE["xl"], tokens.SPACE["lg"], tokens.SPACE["xl"], tokens.SPACE["lg"])
        cf.setSpacing(tokens.SPACE["sm"])
        self.meta = QLabel("")
        self.meta.setObjectName("meta")
        self.front_caption = QLabel("Question")
        self.front_caption.setObjectName("flashCaption")
        self.front_caption.setAlignment(Qt.AlignCenter)
        self.front = _label("", "flashFront")
        self.front.setAlignment(Qt.AlignCenter)
        self.divider = QFrame()
        self.divider.setObjectName("readingDivider")
        self.divider.hide()
        self.back_caption = QLabel("Answer")
        self.back_caption.setObjectName("flashCaptionAnswer")
        self.back_caption.setAlignment(Qt.AlignCenter)
        self.back_caption.hide()
        self.back = _label("", "flashBack")
        self.back.setAlignment(Qt.AlignCenter)
        self.back.hide()
        self.sources_holder = QVBoxLayout()
        cf.addWidget(self.meta)
        cf.addStretch(1)
        cf.addWidget(self.front_caption)
        cf.addWidget(self.front)
        cf.addSpacing(tokens.SPACE["md"])
        cf.addWidget(self.divider)
        cf.addWidget(self.back_caption)
        cf.addWidget(self.back)
        cf.addLayout(self.sources_holder)
        cf.addStretch(1)
        self.card_frame.setMaximumHeight(440)
        cp.addWidget(self.card_frame, 1)
        buttons = QHBoxLayout()
        buttons.setSpacing(tokens.SPACE["sm"])
        self.show_button = QPushButton("Show answer")
        self.show_button.setObjectName("primary")
        self.show_button.clicked.connect(self.reveal)
        self.previous_button = QPushButton("Previous")
        self.previous_button.clicked.connect(self.previous_card)
        self.know_button = QPushButton("Know")
        self.unsure_button = QPushButton("Unsure")
        self.dont_button = QPushButton("Don't know")
        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.next_card)
        self.difficult_button = QPushButton("Mark difficult")
        self.difficult_button.setCheckable(True)
        self.difficult_button.setToolTip("Mark this card as difficult for you (D). The Show list can then bring up only the difficult cards.")
        self.difficult_button.clicked.connect(self.toggle_difficult)
        self.know_button.clicked.connect(lambda: self.rate("know"))
        self.unsure_button.clicked.connect(lambda: self.rate("unsure"))
        self.dont_button.clicked.connect(lambda: self.rate("dont_know"))
        for b in (self.previous_button, self.show_button, self.know_button, self.unsure_button, self.dont_button, self.next_button, self.difficult_button):
            buttons.addWidget(b)
        buttons.addStretch(1)
        cp.addLayout(buttons)
        self.schedule_hint = _label("", "muted")                             # when each answer would bring the card back
        cp.addWidget(self.schedule_hint)
        cp.addWidget(_label("Keys: Space shows the answer · 1 Know · 2 Unsure · 3 Don't know · Left / Right previous / next · D marks difficult", "muted"))
        cp.addStretch(1)
        cp.setSpacing(tokens.SPACE["md"])
        self.stack.addWidget(centered(card_page, CARD_WIDTH))
        done_page = QWidget()
        dp = QVBoxLayout(done_page)
        self.done_label = _label("", "statusLine")
        dp.addStretch(1)
        dp.addWidget(self.done_label)
        row = QHBoxLayout()
        self.again_button = QPushButton("Practice again")
        self.again_button.clicked.connect(self.start)
        self.weak_button = QPushButton("Only the cards I did not know")
        self.weak_button.clicked.connect(self.practice_weak)
        row.addWidget(self.again_button)
        row.addWidget(self.weak_button)
        for sequence, handler in ((Qt.Key_Left, self.previous_card), (Qt.Key_Right, self.next_card), (Qt.Key_D, self.toggle_difficult)):
            shortcut = QShortcut(QKeySequence(sequence), self, context=Qt.WidgetWithChildrenShortcut)
            shortcut.activated.connect(lambda handler=handler: self._shortcut(handler))
        row.addStretch(1)
        dp.addLayout(row)
        dp.addStretch(1)
        self.stack.addWidget(done_page)

        self.right_rail = QFrame()
        self.right_rail.setObjectName("studyRail")
        rail = QVBoxLayout(self.right_rail)
        rail.setSpacing(tokens.SPACE["md"])
        self.rail_concepts = self._rail_panel(rail, "Key concepts")
        self.rail_question = self._rail_panel(rail, "Question preview")
        self.rail_progress = self._rail_panel(rail, "Study progress")
        rail.addStretch(1)
        body.addWidget(main)
        body.addWidget(self.right_rail)
        body.setStretchFactor(0, 4)
        body.setStretchFactor(1, 1)
        body.setSizes([720, 260])
        root.addWidget(body, 1)

    @staticmethod
    def _rail_panel(layout: QVBoxLayout, title: str) -> QLabel:
        panel = QFrame()
        panel.setObjectName("railPanel")
        box = QVBoxLayout(panel)
        box.setContentsMargins(tokens.SPACE["md"], tokens.SPACE["md"], tokens.SPACE["md"], tokens.SPACE["md"])
        box.setSpacing(tokens.SPACE["xs"])
        heading = QLabel(title)
        heading.setObjectName("sectionTitle")
        value = _label("", "muted")
        box.addWidget(heading)
        box.addWidget(value)
        layout.addWidget(panel)
        return value

    # ---- setup
    def show_pack(self, pack: StudyPack, progress: Progress) -> None:
        self.pack, self.progress = pack, progress
        for combo, entries in ((self.topic_combo, [("All topics", "")] + [(t, t) for t in pack.topics]),
                               (self.type_combo, [("All types", "")] + [(CARD_TYPE_TITLES[t], t) for t in CARD_TYPE_TITLES
                                                                         if any(c.type == t for c in pack.flashcards)]),
                               (self.level_combo, [("All levels", ""), ("Basic", "easy"), ("Intermediate", "medium"), ("Advanced", "hard")]),
                               (self.focus_combo, [("All cards", FOCUS_ALL), ("Due today", "due"), ("New cards", "new"), ("To review", "weak"),
                                                   ("Difficult", "difficult"), ("Not mastered", "unmastered")])):
            combo.clear()
            for label, value in entries:
                combo.addItem(label, value)
        if progress.due_cards(pack):                                          # cards the schedule has brought back: they are what to study first
            self.focus_combo.setCurrentIndex(self.focus_combo.findData("due"))
        self._update_rail()
        self.start()

    def start(self) -> None:
        if self.pack is None:
            return
        if not self.pack.flashcards:
            self.empty_page.setText("No flashcards could be made from these sources.")
            self.stack.setCurrentIndex(0)
            return
        types = [self.type_combo.currentData()] if self.type_combo.currentData() else []
        deck = build_deck(self.pack, self.progress, topic=self.topic_combo.currentData() or "", types=types,
                          difficulty=self.level_combo.currentData() or "", only=self.focus_combo.currentData() or FOCUS_ALL,
                          shuffle=self.shuffle_check.isChecked())
        self._begin(deck)

    def practice_weak(self) -> None:
        if self.pack is None:
            return
        self._begin(build_deck(self.pack, self.progress, only="weak", shuffle=self.shuffle_check.isChecked()))

    def _begin(self, deck: list[Flashcard]) -> None:
        if not deck:
            self.empty_page.setText("No cards match these choices. Change the filters above.")
            self.stack.setCurrentIndex(0)
            return
        self.session = FlashcardSession(deck, self.progress)
        self._show_card()

    # ---- card flow
    def _show_card(self) -> None:
        session = self.session
        card = session.current() if session else None
        if card is None:
            self._finish()
            return
        self._revealed = False
        self.stack.setCurrentIndex(1)
        done = session.summary()
        tally = [f"{done['know']} know", f"{done['unsure']} unsure", f"{done['dont_know']} don't know"] if (done["know"] or done["unsure"] or done["dont_know"]) else []
        self.counter.setText("  ·  ".join([f"Card {session.position + 1} of {len(session.deck)}", *tally]))
        self.meta.setText(f"{CARD_TYPE_TITLES.get(card.type, card.type)}  ·  {DIFFICULTY_TITLES.get(card.difficulty, card.difficulty.title())}  ·  {card.topic}")
        self.difficult_button.setChecked(self.progress.is_difficult(card.id))
        self.difficult_button.setText("Marked difficult" if self.progress.is_difficult(card.id) else "Mark difficult")
        self.schedule_hint.setText("")
        self.front.setText(card.front)
        self.back.setText(card.back)
        self.back.hide()
        self.back_caption.hide()
        self.divider.hide()
        _clear(self.sources_holder)
        self.show_button.show()
        self.previous_button.setEnabled(session.position > 0)
        self.next_button.setEnabled(session.position + 1 < len(session.deck))
        for b in (self.know_button, self.unsure_button, self.dont_button):
            b.hide()
        self._update_rail()

    def reveal(self) -> None:
        card = self.session.current() if self.session else None
        if card is None or self._revealed:
            return
        self._revealed = True
        self.divider.show()
        self.back_caption.show()
        self.back.show()
        _clear(self.sources_holder)
        handler = lambda ref: self.source_clicked.emit(ref.chunk_id)           # noqa: E731
        self.sources_holder.addWidget(SourceChips(card.sources, handler))
        self.show_button.hide()
        for b in (self.know_button, self.unsure_button, self.dont_button):
            b.show()
        current = self.progress.schedule(card.id)
        days = {name: next_schedule(current, name, date.today()).interval for name in QUALITY}
        self.schedule_hint.setText(f"Next review  ·  Know: {when(days['know'])}  ·  Unsure: {when(days['unsure'])}  ·  Don't know: {when(days['dont_know'])}")

    def rate(self, rating: str) -> None:
        if self.session is None or not self._revealed:
            return
        self.session.rate(rating)
        self.changed.emit()
        self._show_card()

    def toggle_difficult(self) -> None:
        if self.session is None or self.session.current() is None:
            return
        self.session.toggle_difficult()
        card = self.session.current()
        self.difficult_button.setChecked(self.progress.is_difficult(card.id))
        self.difficult_button.setText("Marked difficult" if self.progress.is_difficult(card.id) else "Mark difficult")
        self.changed.emit()
        self._update_rail()

    def _shortcut(self, handler: Callable[[], None]) -> None:
        """Left / Right / D act on the card on screen - but not while a list box has the keyboard (they would change its choice) or when no card is showing."""
        if self.session is None or self.stack.currentIndex() != 1 or isinstance(QApplication.focusWidget(), QComboBox):
            return
        handler()

    def previous_card(self) -> None:
        if self.session is None or self.session.position <= 0:
            return
        self.session.position -= 1
        self._show_card()

    def next_card(self) -> None:
        if self.session is None or self.session.current() is None:
            return
        self.session.skip()
        self._show_card()

    def _related_question(self, card):
        """The question that goes with the card on screen: made from the same fact, else from the same source passage - else none (a question that merely shares the topic is not about this card)."""
        if card is None or self.pack is None:
            return None
        questions = self.pack.questions
        if card.fact_id:
            same_fact = next((q for q in questions if q.fact_id == card.fact_id), None)
            if same_fact is not None:
                return same_fact
        wanted = {ref.chunk_id for ref in card.sources}
        if wanted:
            return next((q for q in questions if wanted & {ref.chunk_id for ref in q.sources}), None)
        return None

    def _update_rail(self) -> None:
        if self.pack is None:
            self.rail_concepts.setText("No key concepts available.")
            self.rail_question.setText("No question available.")
            self.rail_progress.setText("No study progress yet.")
            return
        concepts = self.pack.summaries.get("key_concepts", []) or self.pack.summaries.get("definitions", [])
        self.rail_concepts.setText("\n".join(f"• {item.term or item.text}" for item in concepts[:5])
                                   or "No key concepts available.")
        question = self._related_question(self.session.current() if self.session is not None else None)
        if not self.pack.questions:
            self.rail_question.setText("No question available.")
        else:
            self.rail_question.setText((f"{QUESTION_TYPE_TITLES.get(question.type, question.type)} · {DIFFICULTY_TITLES.get(question.difficulty, question.difficulty.title())}\n"
                                        f"{question.question}") if question else "No question about this card yet.")
        counts = self.progress.counts(self.pack)["cards"]
        total = len(self.pack.flashcards)
        mastered = counts[MASTERED]
        percent = round(100 * mastered / total) if total else 0
        plan = self.progress.study_plan(self.pack)
        self.rail_progress.setText(f"{percent}% mastered\n{mastered} mastered · {counts[WEAK]} to review · {counts[NEW]} new\n"
                                   f"{plan['due']} due today · {plan['difficult']} difficult\n"
                                   f"{plan['reviews_today']} review{'s' if plan['reviews_today'] != 1 else ''} today · {plan['reviews']} in all")

    def _next_review_line(self) -> str:
        if self.pack is None:
            return ""
        plan = self.progress.study_plan(self.pack)
        if plan["due"]:
            return f"{plan['due']} card{'s' if plan['due'] != 1 else ''} due today."
        later = self.progress.next_due(self.pack)
        if later is None:
            return ""
        days = (later - date.today()).days
        return f"Nothing is due today. The next review is {when(max(days, 0))}."

    def _finish(self) -> None:
        summary = self.session.summary() if self.session else {}
        self.done_label.setText(f"Deck finished.\n\nKnow: {summary.get('know', 0)}     Unsure: {summary.get('unsure', 0)}     "
                                f"Don't know: {summary.get('dont_know', 0)}\n\n{self._next_review_line()}".rstrip())
        self.weak_button.setEnabled(bool(summary.get("dont_know")) or any(self.progress.card_state(c.id) == WEAK
                                                                        for c in (self.pack.flashcards if self.pack else [])))
        self.stack.setCurrentIndex(2)

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt API
        key = event.key()
        if key == Qt.Key_Space:
            self.reveal()
        elif key in (Qt.Key_1, Qt.Key_2, Qt.Key_3) and self._revealed:
            self.rate({Qt.Key_1: "know", Qt.Key_2: "unsure", Qt.Key_3: "dont_know"}[key])
        else:
            super().keyPressEvent(event)


# ===================================================================================================================== Questions
class QuestionCard(QFrame):
    """One question to answer: options (or a text box), Check answer, then the result with explanation and sources."""

    answered = Signal(str, str)             # question id, outcome

    def __init__(self, question: Question, number: int, handler: SourceHandler, evaluator: Optional[Callable] = None, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        self.question, self.handler, self.evaluator = question, handler, evaluator
        self.done = False
        self.setAccessibleName(f"Question {number}: {question.question}")           # the group of answers is read with its question
        layout = QVBoxLayout(self)
        layout.setContentsMargins(tokens.SPACE["lg"], tokens.SPACE["lg"], tokens.SPACE["lg"], tokens.SPACE["lg"])
        layout.setSpacing(tokens.SPACE["sm"])
        head = QHBoxLayout()
        head.setSpacing(tokens.SPACE["md"])
        badge = QLabel(str(number))
        badge.setObjectName("questionNumber")
        badge.setAlignment(Qt.AlignCenter)
        badge.setAccessibleName(f"Question {number}")
        column = QVBoxLayout()
        column.setSpacing(tokens.SPACE["xs"])
        column.addWidget(_label(f"{QUESTION_TYPE_TITLES.get(question.type, question.type)}  ·  {DIFFICULTY_TITLES.get(question.difficulty, question.difficulty.title())}  ·  {question.topic}", "meta"))
        column.addWidget(_label(question.question, "questionText"))
        head.addWidget(badge, 0, Qt.AlignTop)
        head.addLayout(column, 1)
        layout.addLayout(head)
        self.buttons: list[QRadioButton] = []
        self.group = QButtonGroup(self)
        self.text_edit: Optional[QPlainTextEdit] = None
        if question.options:
            for option in question.options:
                radio = QRadioButton(option)
                self.buttons.append(radio)
                self.group.addButton(radio)
                layout.addWidget(radio)
        else:
            self.text_edit = QPlainTextEdit()
            self.text_edit.setPlaceholderText("Write your answer ...")
            self.text_edit.setMaximumHeight(90)
            layout.addWidget(self.text_edit)
        row = QHBoxLayout()
        self.check_button = QPushButton("Check answer")
        self.check_button.setObjectName("smallButton")
        self.check_button.clicked.connect(self.check)
        row.addWidget(self.check_button)
        row.addStretch(1)
        layout.addLayout(row)
        self.result = QVBoxLayout()
        layout.addLayout(self.result)

    def response(self) -> str:
        if self.text_edit is not None:
            return self.text_edit.toPlainText().strip()
        for radio in self.buttons:
            if radio.isChecked():
                return radio.text()
        return ""

    def check(self) -> None:
        if self.done:
            return
        response = self.response()
        if not response:
            return
        self.done = True
        q = self.question
        if q.options:
            correct, extra, explanation = response.casefold() == q.answer.casefold(), "", q.explanation
        else:
            ev = self.evaluator(q, response) if self.evaluator else None
            if ev is None:
                from ..study.extractive import ExtractiveProvider
                from ..study.index import DocumentIndex
                from ..study.providers import StudyContext

                ev = ExtractiveProvider().evaluate_answer(q, response, [], StudyContext(index=DocumentIndex([])))
            correct, explanation = ev.correct, ev.feedback or q.explanation
            extra = ("Missing: " + ", ".join(ev.missing)) if ev.missing and not ev.correct else ""
        self.check_button.setEnabled(False)
        if self.text_edit is not None:
            self.text_edit.setReadOnly(True)
        for radio in self.buttons:
            radio.setEnabled(False)
        self.result.addWidget(_label("Correct!" if correct else "Not quite.", "resultGood" if correct else "resultBad"))
        self.result.addWidget(_label(f"Correct answer: {q.answer}", "statusLine"))
        if extra:
            self.result.addWidget(_label(extra, "muted"))
        if explanation:
            self.result.addWidget(_label(explanation, "muted"))
        self.result.addWidget(SourceChips(q.sources, self.handler))
        self.answered.emit(q.id, CORRECT if correct else INCORRECT)


class QuestionsTab(QWidget):
    source_clicked = Signal(str)
    changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.pack: Optional[StudyPack] = None
        self.progress = Progress()
        self.evaluator: Optional[Callable] = None
        self.cards: list[QuestionCard] = []
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        content = QWidget()
        outer.addWidget(centered(content, CARD_WIDTH), 1)            # the choices and the questions share one column, centred
        root = QVBoxLayout(content)
        root.setContentsMargins(0, tokens.SPACE["md"], 0, 0)
        root.setSpacing(tokens.SPACE["md"])
        filters = QHBoxLayout()
        filters.setSpacing(tokens.SPACE["sm"])
        self.topic_combo, self.type_combo, self.level_combo = QComboBox(), QComboBox(), QComboBox()
        _fit(self.topic_combo, self.type_combo, self.level_combo)
        for label, box in (("Topic", self.topic_combo), ("Type", self.type_combo), ("Level", self.level_combo)):
            caption = QLabel(label)
            caption.setObjectName("fieldLabel")
            box.setAccessibleName(label)
            filters.addWidget(caption)
            filters.addWidget(box)
            filters.addSpacing(tokens.SPACE["sm"])
            box.currentIndexChanged.connect(lambda _i: self._rebuild())
        self.count_label = QLabel("")
        self.count_label.setObjectName("muted")
        filters.addWidget(self.count_label)
        filters.addStretch(1)
        root.addLayout(filters)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setFocusPolicy(Qt.NoFocus)
        self.holder = QWidget()
        self.list_layout = QVBoxLayout(self.holder)
        self.list_layout.setContentsMargins(0, 0, tokens.SPACE["sm"], 0)
        self.list_layout.setSpacing(tokens.SPACE["md"])
        self.scroll.setWidget(self.holder)
        root.addWidget(self.scroll, 1)

    def show_pack(self, pack: StudyPack, progress: Progress, evaluator: Optional[Callable] = None) -> None:
        self.pack, self.progress, self.evaluator = pack, progress, evaluator
        for combo, entries in ((self.topic_combo, [("All topics", "")] + [(t, t) for t in pack.topics]),
                               (self.type_combo, [("All types", "")] + [(QUESTION_TYPE_TITLES[t], t) for t in QUESTION_TYPE_TITLES
                                                                         if any(q.type == t for q in pack.questions)]),
                               (self.level_combo, [("All levels", ""), ("Basic", "easy"), ("Intermediate", "medium"), ("Advanced", "hard")])):
            combo.blockSignals(True)
            combo.clear()
            for label, value in entries:
                combo.addItem(label, value)
            combo.blockSignals(False)
        self._rebuild()

    def _rebuild(self) -> None:
        _clear(self.list_layout)
        self.cards = []
        if self.pack is None:
            return
        topic, qtype, level = self.topic_combo.currentData() or "", self.type_combo.currentData() or "", self.level_combo.currentData() or ""
        chosen = [q for q in self.pack.questions if (not topic or q.topic == topic) and (not qtype or q.type == qtype)
                  and (not level or q.difficulty == level)]
        self.count_label.setText(friendly.plural(len(chosen), "question"))
        if not chosen:
            self.list_layout.addWidget(_label("No questions match these choices." if self.pack.questions else
                                              "No questions could be made from these sources.", "muted"))
        handler = lambda ref: self.source_clicked.emit(ref.chunk_id)           # noqa: E731
        for number, question in enumerate(chosen, start=1):
            card = QuestionCard(question, number, handler, self.evaluator)
            card.answered.connect(self._answered)
            self.cards.append(card)
            self.list_layout.addWidget(card)
        self.list_layout.addStretch(1)

    def _answered(self, question_id: str, outcome: str) -> None:
        question = next((q for q in (self.pack.questions if self.pack else []) if q.id == question_id), None)
        self.progress.record_answer(question_id, outcome, question.topic if question else "")
        self.progress.save()
        self.changed.emit()


# ===================================================================================================================== Study Mode
class StudyModeTab(QWidget):
    source_clicked = Signal(str)
    changed = Signal()
    focus_changed = Signal(bool)            # True while a question is on screen

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.pack: Optional[StudyPack] = None
        self.progress = Progress()
        self.evaluator: Optional[Callable] = None
        self.session: Optional[QuizSession] = None
        self.feedback: Optional[Feedback] = None
        root = QVBoxLayout(self)
        self.stack = QStackedWidget()
        root.addWidget(self.stack)
        self.stack.currentChanged.connect(lambda index: self.focus_changed.emit(index == 1))      # a question on screen: the rest of the workspace steps back

        # --- setup page
        setup = QWidget()
        sl = QVBoxLayout(setup)
        sl.setSpacing(tokens.SPACE["md"])
        sl.addSpacing(tokens.SPACE["lg"])
        intro = QLabel("Study Mode")
        intro.setObjectName("title")
        sl.addWidget(intro)
        sl.addWidget(_label("One question at a time. You answer, see what was right and read the explanation. At the end you get your score and what to review.", "muted"))
        form = QGridLayout()
        form.setHorizontalSpacing(tokens.SPACE["lg"])
        form.setVerticalSpacing(tokens.SPACE["sm"])
        self.count_combo, self.topic_combo, self.focus_combo, self.type_combo = QComboBox(), QComboBox(), QComboBox(), QComboBox()
        for box in (self.count_combo, self.topic_combo, self.focus_combo, self.type_combo):
            box.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            box.setMinimumContentsLength(14)
            box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        for index, (label, box) in enumerate((("Questions", self.count_combo), ("Topic", self.topic_combo), ("Focus", self.focus_combo), ("Type", self.type_combo))):
            caption = QLabel(label)
            caption.setObjectName("fieldLabel")
            box.setAccessibleName(label)
            form.addWidget(caption, (index // 2) * 2, index % 2)
            form.addWidget(box, (index // 2) * 2 + 1, index % 2)
        form.setColumnStretch(0, 1)
        form.setColumnStretch(1, 1)
        sl.addLayout(form)
        self.setup_note = _label("", "muted")
        sl.addWidget(self.setup_note)
        self.resume_note = _label("", "statusLine")                         # an unfinished session (also after the program was closed)
        self.resume_note.hide()
        sl.addWidget(self.resume_note)
        row = QHBoxLayout()
        self.resume_button = QPushButton("Resume session")
        self.resume_button.setObjectName("primary")
        self.resume_button.clicked.connect(self.resume)
        self.resume_button.hide()
        self.start_button = QPushButton("Start session")
        self.start_button.setObjectName("primary")
        self.start_button.clicked.connect(self.start)
        row.addWidget(self.resume_button)
        row.addWidget(self.start_button)
        row.addStretch(1)
        sl.addLayout(row)
        sl.addStretch(1)
        self.stack.addWidget(centered(setup, 720))

        # --- question page
        page = QWidget()
        ql = QVBoxLayout(page)
        ql.setSpacing(tokens.SPACE["md"])
        ql.addSpacing(tokens.SPACE["lg"])
        self.position_label = QLabel("")
        self.position_label.setObjectName("muted")
        ql.addWidget(self.position_label)
        self.question_label = _label("", "focusQuestion")
        ql.addWidget(self.question_label)
        self.meta_label = QLabel("")
        self.meta_label.setObjectName("meta")
        ql.addWidget(self.meta_label)
        self.option_holder = QVBoxLayout()
        ql.addLayout(self.option_holder)
        self.text_edit = QPlainTextEdit()
        self.text_edit.setMaximumHeight(110)
        self.text_edit.setPlaceholderText("Write your answer ...")
        ql.addWidget(self.text_edit)
        self.feedback_box = QFrame()
        self.feedback_box.setObjectName("card")
        self.feedback_layout = QVBoxLayout(self.feedback_box)
        ql.addWidget(self.feedback_box)
        actions = QHBoxLayout()
        self.submit_button = QPushButton("Submit answer")
        self.submit_button.setObjectName("primary")
        self.submit_button.clicked.connect(self.submit)
        self.skip_button = QPushButton("Skip")
        self.skip_button.clicked.connect(self.skip)
        self.next_button = QPushButton("Next question")
        self.next_button.setObjectName("primary")
        self.next_button.clicked.connect(self.next)
        self.agree_button = QPushButton("I was right")
        self.agree_button.clicked.connect(self.override_right)
        self.stop_button = QPushButton("End session")
        self.stop_button.clicked.connect(self.end)
        self.agree_button.setObjectName("link")
        self.stop_button.setObjectName("link")
        actions.setSpacing(tokens.SPACE["sm"])
        for b in (self.submit_button, self.next_button, self.skip_button):
            actions.addWidget(b)
        actions.addStretch(1)
        for b in (self.agree_button, self.stop_button):                 # the main action first; the rare ones quietly at the end
            actions.addWidget(b)
        ql.addLayout(actions)
        ql.addStretch(1)
        self.stack.addWidget(centered(page, 720))
        self.option_buttons: list[QPushButton] = []

        # --- results page
        results = QWidget()
        rl = QVBoxLayout(results)
        rl.setSpacing(tokens.SPACE["md"])
        rl.addSpacing(tokens.SPACE["lg"])
        self.score_label = QLabel("")
        self.score_label.setObjectName("hero")
        rl.addWidget(self.score_label)
        self.counts_label = _label("", "statusLine")
        rl.addWidget(self.counts_label)
        self.weak_label = _label("", "muted")
        self.strong_label = _label("", "muted")
        self.review_label = _label("", "muted")
        for w in (self.weak_label, self.strong_label, self.review_label):
            rl.addWidget(w)
        row = QHBoxLayout()
        self.again_button = QPushButton("New session")
        self.again_button.setObjectName("primary")
        self.again_button.clicked.connect(self._back_to_setup)
        self.review_button = QPushButton("Practise my weak questions")
        self.review_button.clicked.connect(self.start_weak)
        row.addWidget(self.again_button)
        row.addWidget(self.review_button)
        row.addStretch(1)
        rl.addLayout(row)
        rl.addStretch(1)
        self.stack.addWidget(centered(results, 720))

    # ---- setup
    def show_pack(self, pack: StudyPack, progress: Progress, evaluator: Optional[Callable] = None) -> None:
        self.pack, self.progress, self.evaluator = pack, progress, evaluator
        self.session = None
        for combo, entries in ((self.count_combo, [("10", 10), ("20", 20), ("All", 0)]),
                               (self.topic_combo, [("All topics", "")] + [(t, t) for t in pack.topics]),
                               (self.focus_combo, [("Mixed (weak questions first)", FOCUS_ALL), ("Only weak questions", FOCUS_WEAK),
                                                   ("Only new questions", FOCUS_NEW)]),
                               (self.type_combo, [("All types", "")] + [(QUESTION_TYPE_TITLES[t], t) for t in QUESTION_TYPE_TITLES
                                                                         if any(q.type == t for q in pack.questions)])):
            combo.clear()
            for label, value in entries:
                combo.addItem(label, value)
        c = progress.counts(pack)["questions"]
        self.setup_note.setText(f"{len(pack.questions)} questions available  ·  mastered: {c[MASTERED]}  ·  to review: {c[WEAK]}  ·  "
                                f"new: {c[NEW]}")
        self.start_button.setEnabled(bool(pack.questions))
        self._offer_resume()
        self.stack.setCurrentIndex(0)

    def _back_to_setup(self) -> None:
        self._offer_resume()
        self.stack.setCurrentIndex(0)

    def _offer_resume(self) -> None:
        left = QuizSession.resumable(self.pack, self.progress) if self.pack is not None else None
        self.resume_button.setVisible(left is not None)
        self.resume_note.setVisible(left is not None)
        self.start_button.setObjectName("" if left else "primary")
        self.start_button.style().unpolish(self.start_button)
        self.start_button.style().polish(self.start_button)
        if left:
            self.resume_note.setText(f"You have an unfinished session: {left['answered']} of {left['total']} questions answered, {left['left']} left. "
                                     "Starting a new session replaces it.")

    def resume(self) -> None:
        if self.pack is None:
            return
        session = QuizSession.resume(self.pack, self.progress, evaluator=self.evaluator)
        if session is None:
            self._offer_resume()
            return
        self.session = session
        self._show_question()

    def start(self) -> None:
        if self.pack is None:
            return
        types = [self.type_combo.currentData()] if self.type_combo.currentData() else []
        questions = build_quiz(self.pack, self.progress, count=self.count_combo.currentData() or 0, topic=self.topic_combo.currentData() or "",
                               types=types, focus=self.focus_combo.currentData() or FOCUS_ALL)
        self._begin(questions)

    def start_weak(self) -> None:
        if self.pack is not None:
            self._begin(build_quiz(self.pack, self.progress, count=0, focus=FOCUS_WEAK))

    def _begin(self, questions: list[Question]) -> None:
        if not questions:
            self.setup_note.setText("No questions match these choices.")
            self.stack.setCurrentIndex(0)
            return
        self.session = QuizSession(questions, self.progress, evaluator=self.evaluator)
        self._show_question()

    # ---- question flow
    def _show_question(self) -> None:
        view = self.session.current() if self.session else None
        if view is None:
            self._show_results()
            return
        self.stack.setCurrentIndex(1)
        self.feedback = None
        self.position_label.setText(f"Question {view.number} of {view.total}")
        self.meta_label.setText(f"{QUESTION_TYPE_TITLES.get(view.type, view.type)}  ·  {DIFFICULTY_TITLES.get(view.difficulty, view.difficulty.title())}  ·  {view.topic}")
        self.question_label.setText(view.question)
        _clear(self.option_holder)
        self.option_buttons = []
        for number, option in enumerate(view.options):
            button = QPushButton(f"{chr(65 + number)}.  {option}" if len(view.options) > 2 else option)
            button.setObjectName("option")
            button.clicked.connect(lambda _c=False, n=number: self.choose(n))
            self.option_holder.addWidget(button)
            self.option_buttons.append(button)
        self.text_edit.setVisible(view.free_text)
        self.text_edit.clear()
        self.text_edit.setReadOnly(False)
        _clear(self.feedback_layout)
        self.feedback_box.hide()
        self.submit_button.setVisible(view.free_text)
        self.skip_button.show()
        self.next_button.hide()
        self.agree_button.hide()

    def choose(self, number: int) -> None:
        view = self.session.current() if self.session else None
        if view is None or self.feedback is not None or not 0 <= number < len(view.options):
            return
        self._show_feedback(self.session.answer_option(number))

    def submit(self) -> None:
        if self.session is None or self.feedback is not None:
            return
        self._show_feedback(self.session.answer(self.text_edit.toPlainText()))

    def skip(self) -> None:
        if self.session is None or self.feedback is not None:
            return
        self._show_feedback(self.session.skip())

    def _show_feedback(self, feedback: Feedback) -> None:
        self.feedback = feedback
        _clear(self.feedback_layout)
        if feedback.skipped:
            self.feedback_layout.addWidget(_label("Skipped.", "muted"))
        else:
            self.feedback_layout.addWidget(_label("Correct!" if feedback.correct else "Not quite.",
                                                  "resultGood" if feedback.correct else "resultBad"))
        self.feedback_layout.addWidget(_label(f"Correct answer: {feedback.correct_answer}", "statusLine"))
        if feedback.missing and not feedback.correct:
            self.feedback_layout.addWidget(_label("Missing: " + ", ".join(feedback.missing), "muted"))
        if feedback.explanation:
            self.feedback_layout.addWidget(_label(feedback.explanation, "muted"))
        handler = lambda ref: self.source_clicked.emit(ref.chunk_id)           # noqa: E731
        self.feedback_layout.addWidget(SourceChips(feedback.sources, handler))
        self.feedback_box.show()
        for button in self.option_buttons:
            button.setEnabled(False)
        self.text_edit.setReadOnly(True)
        self.submit_button.hide()
        self.skip_button.hide()
        self.next_button.show()
        self.next_button.setText("Finish" if self.session and self.session.position + 1 >= len(self.session.questions) else "Next question")
        view = self.session.current() if self.session else None
        self.agree_button.setVisible(not feedback.correct and not feedback.skipped and view is not None and view.free_text)
        self.changed.emit()

    def override_right(self) -> None:
        if self.session is not None and self.feedback is not None:
            self._show_feedback(self.session.override(True))

    def next(self) -> None:
        if self.session is None:
            return
        self.session.next()
        self._show_question()

    def end(self) -> None:
        if self.session is not None:
            while not self.session.finished:
                self.session.next()
        self._show_results()

    def _show_results(self) -> None:
        if self.session is None:
            self.stack.setCurrentIndex(0)
            return
        r = self.session.results()
        self.session._finish()
        self.score_label.setText(f"Score: {r.correct} / {r.total}  ({r.score_percent}%)")
        self.counts_label.setText(f"Correct: {r.correct}     Incorrect: {r.incorrect}     Skipped: {r.skipped}")
        self.weak_label.setText("Weak topics: " + ("; ".join(f"{t} ({round(a * 100)}%)" for t, a in r.weak_topics) if r.weak_topics else "none"))
        self.strong_label.setText("Strong topics: " + ("; ".join(f"{t} ({round(a * 100)}%)" for t, a in r.strong_topics)
                                                       if r.strong_topics else "none yet"))
        self.review_label.setText("Suggested review:\n" + "\n".join(f"- {line}" for line in r.suggested_review))
        self.review_button.setEnabled(bool(r.weak_question_ids))
        self.stack.setCurrentIndex(2)
        self.changed.emit()
