"""Settings of MELETRA, grouped the way people think about them: Transcription, Medical, Study, Privacy & Data, Models, Advanced.

A list of the six groups on the left, one page on the right. The technical settings of 0.1.1 (device, compute type, chunking, hotwords, diagnostics ...) are all still there;
they sit inside "Advanced". Changes are written when "Save" is pressed, except the transcript-sharing switch, which acts at once.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QPoint, QSize, Qt, QTimer
from PySide6.QtWidgets import (QAbstractSpinBox, QCheckBox, QComboBox, QDialogButtonBox, QFormLayout, QFrame, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
                               QPushButton, QScrollArea, QSizePolicy, QSpinBox, QTabWidget, QVBoxLayout, QWidget)

from .. import APP_NAME, __version__
from ..archive.resolver import lecture_folder
from ..language_catalogue import OUTPUT_LANGUAGES
from ..study.llm import OPENAI_URL, is_local_url
from . import icons, tokens
from .choices import (INPUT_LANGUAGE_CHOICES, LANGUAGE_CHOICES, MODE_CHOICES, OUTPUT_LANGUAGE_CHOICES, index_of, subject_choices, theme_choices)
from .components import make_badge, repolish
from .settings_dialog import SettingsDialog, _combo
from .tasks import TaskThread

PAGES = (("Transcription", "audio"), ("Medical", "plus"), ("Study", "study"), ("Privacy & Data", "lock"), ("Models", "chip"), ("Advanced", "settings"))
TAB_NAMES = [name for name, _icon in PAGES]
ADVANCED_TABS = ("Performance", "Medical options", "Diagnostics")
#: the names older code and tests use -> where they live now
ALIASES = {"General": "Transcription", "AI Study": "Study", "Cloud": "Privacy & Data"}
SERVICE_CHOICES = [("None - MELETRA chooses automatically", "builtin"), ("OpenAI-compatible server (on this computer or in the cloud)", "openai"), ("Anthropic (cloud)", "anthropic")]
STUDY_LANGUAGE_CHOICES = [("Same as the Output Language", "auto")] + [(label, code) for code, label in OUTPUT_LANGUAGES]
WINDOW_CHOICES = [("Simple (recommended)", "simple"), ("Classic technical window", "classic")]
INTELLIGENCE_STATES = {"off": ("OFF", "off"), "shadow": ("COMPARE ONLY", "off"), "on": ("ON", "on")}
SERVICE_HINTS = {
    "builtin": "",
    "openai": "A server on this computer (for example LM Studio) keeps everything local. A cloud address sends the lecture text - never the audio - to that company, and only after you tick the box below.",
    "anthropic": "A cloud model: the lecture text (never the audio) is sent to Anthropic when you make study material, and only after you tick the box below.",
}


def _card(heading: str = "", hint: str = "") -> tuple[QFrame, QVBoxLayout]:
    """A rounded settings group: an optional heading and hint, then the controls."""
    frame = QFrame()
    frame.setObjectName("card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(tokens.SPACE["xl"], tokens.SPACE["lg"], tokens.SPACE["xl"], tokens.SPACE["lg"])
    layout.setSpacing(tokens.SPACE["md"])
    if heading:
        title = QLabel(heading)
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
    if hint:
        note = QLabel(hint)
        note.setObjectName("muted")
        note.setWordWrap(True)
        layout.addWidget(note)
    return frame, layout


def _form() -> QFormLayout:
    form = QFormLayout()
    form.setHorizontalSpacing(tokens.SPACE["lg"])
    form.setVerticalSpacing(tokens.SPACE["md"])
    form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
    return form


def _add(form: QFormLayout, label: str, field) -> None:
    """One row of a settings form; the field also carries the label as its accessible name (a screen reader says "Input language", not just "combo box")."""
    if hasattr(field, "setAccessibleName") and not field.accessibleName():
        field.setAccessibleName(label.rstrip(":"))
    form.addRow(label, field)


def _page(title: str, subtitle: str, *blocks, stretch: bool = True) -> QScrollArea:
    """One settings page: its title and sentence, then the groups; the page scrolls when the window is small."""
    holder = QWidget()
    column = QVBoxLayout(holder)
    column.setContentsMargins(4, 0, 12, 4)
    column.setSpacing(tokens.SPACE["lg"])
    heading = QLabel(title)
    heading.setObjectName("title")
    sub = QLabel(subtitle)
    sub.setObjectName("muted")
    sub.setWordWrap(True)
    column.addWidget(heading)
    column.addWidget(sub)
    for block in blocks:
        column.addWidget(block)
    if stretch:
        column.addStretch(1)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)                      # never silently clip a page that is wider than a small window
    scroll.setFocusPolicy(Qt.NoFocus)
    scroll.setWidget(holder)
    return scroll


class SimpleSettingsDialog(SettingsDialog):
    def __init__(self, services, parent=None, initial_tab: str = "General") -> None:
        super().__init__(services, parent, initial_tab)
        buttons = self.findChild(QDialogButtonBox)
        if buttons is not None:
            save = buttons.button(QDialogButtonBox.Save)
            if save is not None:
                save.setObjectName("primary")
        self.setMinimumSize(780, 520)
        self.resize(980, 660)
        self.tabs.currentChanged.connect(lambda _i: QTimer.singleShot(0, self._set_tab_order))

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        QTimer.singleShot(0, self._set_tab_order)                # (the positions of the fields are known once the page has been laid out)

    def _set_tab_order(self) -> None:
        """Tab goes: the list of groups, then the fields of the page that is shown from top to bottom (left to right in a row), then Close and Save (left to right). (Qt's own order is the
        order the widgets were created in, which is not the order they are seen in, and the base dialog creates its buttons first.)"""
        page = self.tabs.currentWidget()
        if page is None:
            return
        found = []
        for widget in page.findChildren(QWidget):
            if widget.focusPolicy() == Qt.NoFocus or widget.focusProxy() is not None or isinstance(widget.parent(), (QAbstractSpinBox, QComboBox)):
                continue
            if not widget.isVisibleTo(page):
                continue
            point = widget.mapTo(page, QPoint(0, 0))
            found.append((point.y(), point.x(), widget))
        found.sort(key=lambda item: (item[0], item[1]))
        rows, row_top = [], None                                   # widgets whose tops are within 24 px of each other are one row: left to right
        for y, x, widget in found:
            if row_top is None or y - row_top > 24:
                rows.append([])
                row_top = y
            rows[-1].append((x, widget))
        order = [self.nav] + [widget for row in rows for _x, widget in sorted(row, key=lambda item: item[0])]
        buttons = self.findChild(QDialogButtonBox)
        if buttons is not None:
            order += [b for b in (buttons.button(QDialogButtonBox.Close), buttons.button(QDialogButtonBox.Save)) if b is not None]      # as they stand: Close, then Save
        for first, second in zip(order, order[1:]):
            QWidget.setTabOrder(first, second)

    def _build_tabs(self, s) -> None:
        self.tab_names = list(TAB_NAMES)
        self.tabs.addTab(self._transcription_page(s), "Transcription")
        self.tabs.addTab(self._medical_page(s), "Medical")
        self.tabs.addTab(self._study_page(s), "Study")
        self.tabs.addTab(self._privacy_page(s), "Privacy & Data")
        self.tabs.addTab(self._models_page(), "Models")
        self.tabs.addTab(self._advanced_page(s), "Advanced")
        # the groups are chosen from a list on the left; the tab bar of the base dialog is not shown
        self.tabs.tabBar().hide()
        self.tabs.setFocusPolicy(Qt.NoFocus)                    # the pages are chosen in the list on the left: the hidden bar is no keyboard stop
        self.tabs.setDocumentMode(True)
        self.nav = QListWidget()
        self.nav.setObjectName("settingsNav")
        self.nav.setFixedWidth(200)
        self.nav.setIconSize(QSize(20, 20))
        self.nav.setAccessibleName("Settings groups")
        for name, icon_name in PAGES:
            item = QListWidgetItem(icons.icon(icon_name, normal="text2", hover="text", selected="teal", size=20), name)
            item.setSizeHint(QSize(180, 40))
            self.nav.addItem(item)
        self.nav.setCurrentRow(0)
        self.nav.currentRowChanged.connect(self._nav_changed)
        layout = self.layout()
        layout.removeWidget(self.tabs)
        body = QHBoxLayout()
        body.setSpacing(tokens.SPACE["lg"])
        body.addWidget(self.nav)
        body.addWidget(self.tabs, 1)
        layout.insertLayout(0, body)
        self.tabs.currentChanged.connect(lambda i: self.nav.setCurrentRow(i) if self.nav.currentRow() != i else None)

    def _nav_changed(self, row: int) -> None:
        if row >= 0 and self.tabs.currentIndex() != row:
            self.tabs.setCurrentIndex(row)

    # ------------------------------------------------------------------------------------------------------------------ Transcription
    def _transcription_page(self, s) -> QWidget:
        files, fl = _card("Where your files go")
        form = _form()
        self.output_edit = QLineEdit(s.output_dir)
        self.output_edit.setPlaceholderText("Next to the recording, in a 'Medical Transcription' folder")
        self.output_edit.setAccessibleName("Default output folder")
        browse = QPushButton("Browse ...")
        browse.clicked.connect(self._browse_output)
        row = QHBoxLayout()
        row.setSpacing(tokens.SPACE["sm"])
        row.addWidget(self.output_edit, 1)
        row.addWidget(browse)
        _add(form, "Output folder:", row)
        fl.addLayout(form)
        self.open_folder_check = QCheckBox("Show the lecture folder in Explorer when a transcription is finished")
        self.open_folder_check.setChecked(s.open_folder_after)
        fl.addWidget(self.open_folder_check)
        note = QLabel(f"Program data (models, logs, unfinished work) is kept in {self.services.paths.root}")
        note.setObjectName("meta")
        note.setWordWrap(True)
        fl.addWidget(note)

        defaults, dl = _card("Defaults for new transcriptions", "You can still change the languages for each lecture on the Home screen.")
        form2 = _form()
        self.language_combo = _combo(LANGUAGE_CHOICES, s.language)
        self.output_language_combo = _combo(OUTPUT_LANGUAGE_CHOICES, s.output_language)
        self.subject_combo = _combo(subject_choices(), s.subject)
        self.mode_combo = _combo(MODE_CHOICES, s.mode)
        for combo in (self.language_combo, self.output_language_combo, self.subject_combo, self.mode_combo):
            combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            combo.setMaximumWidth(380)
        _add(form2, "Input language:", self.language_combo)
        _add(form2, "Output language:", self.output_language_combo)
        _add(form2, "Subject:", self.subject_combo)
        _add(form2, "Transcription mode:", self.mode_combo)
        dl.addLayout(form2)
        return _page("Transcription", "Where transcripts are saved and the defaults for every new lecture.", files, defaults)

    # ------------------------------------------------------------------------------------------------------------------ Medical
    def _medical_page(self, s) -> QWidget:
        expert, el = _card()
        head = QHBoxLayout()
        head.setSpacing(tokens.SPACE["sm"])
        title = QLabel("Medical Expert v3")
        title.setObjectName("sectionTitle")
        self.expert_badge = make_badge("Stable", "stable")
        self.expert_state = make_badge("ON" if s.medical_expert else "OFF", "on" if s.medical_expert else "off")
        head.addWidget(title)
        head.addWidget(self.expert_badge)
        head.addWidget(self.expert_state)
        head.addStretch(1)
        el.addLayout(head)
        text = QLabel("Checks medical terms, drug names and abbreviations in the context of the lecture and writes a medical corrected transcript next to the original one. "
                      "Your original transcript is never changed.")
        text.setObjectName("muted")
        text.setWordWrap(True)
        el.addWidget(text)
        self.medical_expert_check = QCheckBox("Use Medical Expert")
        self.medical_expert_check.setChecked(s.medical_expert)
        self.medical_expert_check.toggled.connect(self._expert_toggled)
        el.addWidget(self.medical_expert_check)

        intelligence, il = _card()
        head2 = QHBoxLayout()
        head2.setSpacing(tokens.SPACE["sm"])
        title2 = QLabel("Medical Intelligence v4")
        title2.setObjectName("sectionTitle")
        title2.setToolTip("An extra check against a large medical knowledge base. Experimental: it is off unless you turn it on.")
        self.intelligence_badge = make_badge("Experimental", "experimental")
        label, kind = INTELLIGENCE_STATES.get(s.medical_intelligence, ("OFF", "off"))
        self.intelligence_state = make_badge(label, kind)
        head2.addWidget(title2)
        head2.addWidget(self.intelligence_badge)
        head2.addWidget(self.intelligence_state)
        head2.addStretch(1)
        il.addLayout(head2)
        text2 = QLabel("An extra, experimental check against a large medical knowledge base. It is off by default and never changes your original transcript.")
        text2.setObjectName("muted")
        text2.setWordWrap(True)
        text2.setToolTip("Experimental: results are still being evaluated.")
        il.addWidget(text2)
        self.medical_intelligence_button = QPushButton("Medical Intelligence settings ...")
        self.medical_intelligence_button.clicked.connect(self.open_medical_intelligence)
        row = QHBoxLayout()
        row.addWidget(self.medical_intelligence_button)
        row.addStretch(1)
        il.addLayout(row)

        terms, tl = _card("My medical terms", "Add words your lectures use that should always be spelled your way.")
        self.medical_vocab_general_button = QPushButton("Medical vocabulary - add terms ...")
        self.medical_vocab_general_button.clicked.connect(self.open_medical_vocabulary)
        row2 = QHBoxLayout()
        row2.addWidget(self.medical_vocab_general_button)
        row2.addStretch(1)
        tl.addLayout(row2)
        return _page("Medical", "How MELETRA checks medical terms.", expert, intelligence, terms)

    def _expert_toggled(self, on: bool) -> None:
        self.expert_state.setText("ON" if on else "OFF")
        self.expert_state.setProperty("kind", "on" if on else "off")
        repolish(self.expert_state)

    def open_medical_intelligence(self) -> None:
        from .medical_intelligence_dialog import open_medical_intelligence

        open_medical_intelligence(self.services, self)
        label, kind = INTELLIGENCE_STATES.get(self.services.settings.settings.medical_intelligence, ("OFF", "off"))
        self.intelligence_state.setText(label)
        self.intelligence_state.setProperty("kind", kind)
        repolish(self.intelligence_state)

    def _add_status(self, layout, s) -> None:
        """Read-only: what the next lecture will use (settings_status.py). The knowledge line is filled in by a background read."""
        from .settings_status import knowledge_line, status_lines

        form = _form()
        installed = getattr(getattr(self.services, "models", None), "is_installed", lambda key: False)
        self.status_values: dict[str, QLabel] = {}
        for label, value in status_lines(s, installed):
            value_label = QLabel(value)
            value_label.setWordWrap(True)
            self.status_values[label] = value_label
            caption = QLabel(label)
            caption.setObjectName("muted")
            form.addRow(caption, value_label)
        layout.addLayout(form)

        def read(progress, cancelled):
            from ..medical_knowledge.cli import report

            return report()

        thread = TaskThread(read, self)
        thread.succeeded.connect(lambda info: self.status_values["Medical knowledge:"].setText(knowledge_line(info)))
        thread.failed.connect(lambda msg: self.status_values["Medical knowledge:"].setText(f"could not be read ({msg}) - the Medical Expert works alone"))
        self._status_thread = thread
        thread.start()

    # ------------------------------------------------------------------------------------------------------------------ Privacy & Data
    def _privacy_page(self, s) -> QWidget:
        card, layout = _card("Transcript sharing")
        top = QHBoxLayout()
        top.setSpacing(tokens.SPACE["lg"])
        on = self.services.sharing.active()                       # one answer for the checkbox, the label, the header chip and the uploader
        self.share_state = QLabel("ON" if on else "OFF")
        self.share_state.setObjectName("stateBig")
        self.share_state.setProperty("kind", "on" if on else "off")
        self.share_state.setAccessibleName("Transcript sharing is " + ("on" if on else "off"))
        self.share_check = QCheckBox("Share completed transcript text with MELETRA   (default: OFF)")
        self.share_check.setChecked(on)
        self.share_check.toggled.connect(self._share_toggled)
        top.addWidget(self.share_state)
        top.addWidget(self.share_check, 1)
        layout.addLayout(top)
        intro = QLabel("Transcription itself always runs on this computer, and nothing is shared unless you switch this on. If you do, completed transcripts are shared with MELETRA "
                       "to improve transcription quality. You can turn it off again at any time; the change takes effect at once.")
        intro.setObjectName("muted")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        what = QLabel("Sent, after the transcript is saved on your computer: the transcript text (and the medical corrected transcript), the lecture name, the languages, "
                      "the program version, a random installation number and the time.\n"
                      "Never sent: audio, video, PDF files, pictures, file paths, your user name, your computer's name or any hardware number.\n"
                      "Turning sharing off stops all future uploads and deletes everything still waiting to be sent.")
        what.setWordWrap(True)
        layout.addWidget(what)
        heading = QLabel("Upload status")
        heading.setObjectName("sectionTitle")
        layout.addWidget(heading)
        self.share_status = QLabel("")
        self.share_status.setObjectName("muted")
        self.share_status.setWordWrap(True)
        layout.addWidget(self.share_status)
        row = QHBoxLayout()
        row.setSpacing(tokens.SPACE["sm"])
        self.share_refresh = QPushButton("Refresh")
        self.share_refresh.clicked.connect(self._refresh_share_status)
        self.share_retry = QPushButton("Try to send now")
        self.share_retry.clicked.connect(lambda: (self.services.sharing.uploader.nudge(), self._refresh_share_status()))
        self.share_retry.setEnabled(on)                           # nothing can be sent while sharing is off: the button does not pretend otherwise
        self.share_preview = QPushButton("Show what would be shared")
        self.share_preview.clicked.connect(self.show_share_preview)
        row.addWidget(self.share_refresh)
        row.addWidget(self.share_retry)
        row.addWidget(self.share_preview)
        row.addStretch(1)
        layout.addLayout(row)
        self._refresh_share_status()

        cloud, cl = _card("Cloud comparison (optional)")
        cl.addWidget(self._cloud_tab(s))
        return _page("Privacy & Data", "What stays on this computer and what can leave it.", card, cloud)

    def _cloud_tab(self, s) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(tokens.SPACE["md"])
        note = QLabel("Off by default. When you turn it on, the audio of a lecture is uploaded to Deepgram, a third-party service, to compare its result with yours.")
        note.setObjectName("muted")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.cloud_check = QCheckBox("Also compare with Deepgram Nova-3 Medical (uploads the audio)")
        self.cloud_check.setChecked(s.cloud_enabled)
        layout.addWidget(self.cloud_check)
        form = _form()
        self.key_edit = QLineEdit()
        self.key_edit.setEchoMode(QLineEdit.Password)
        self.key_edit.setPlaceholderText("Paste your Deepgram API key")
        self.save_key_button = QPushButton("Save key")
        self.remove_key_button = QPushButton("Remove key")
        self.save_key_button.clicked.connect(self.save_key)
        self.remove_key_button.clicked.connect(self.remove_key)
        row = QHBoxLayout()
        row.setSpacing(tokens.SPACE["sm"])
        row.addWidget(self.key_edit, 1)
        row.addWidget(self.save_key_button)
        row.addWidget(self.remove_key_button)
        _add(form, "API key:", row)
        self.key_status = QLabel("")
        self.key_status.setObjectName("meta")
        self.key_status.setWordWrap(True)
        form.addRow("", self.key_status)
        layout.addLayout(form)
        self._refresh_key_status()
        return page

    def show_share_preview(self) -> None:
        """The exact fields and a sample of the text of the latest lecture - looking at it shares nothing."""
        from .sharing_notice import SharingPreviewDialog

        record = next((r for r in self.services.jobs.archive() if r.state == "completed"), None)
        folder = lecture_folder(record) if record is not None else None
        dialog = SharingPreviewDialog(self.services.sharing.preview(folder, record.display_name if record is not None else ""), self)
        self.sharing_preview = dialog
        dialog.exec()

    def _share_toggled(self, on: bool) -> None:
        if on:                                                       # switching it on is a decision: what would be shared is shown first, and "Not now" leaves it off
            from .sharing_notice import SharingNoticeDialog

            notice = SharingNoticeDialog(self.services, self)
            self.sharing_notice = notice
            if not notice.exec():
                self.share_check.blockSignals(True)
                self.share_check.setChecked(False)
                self.share_check.blockSignals(False)
                return
        self.services.sharing.set_enabled(on)
        self.share_state.setText("ON" if on else "OFF")
        self.share_state.setProperty("kind", "on" if on else "off")
        self.share_state.setAccessibleName("Transcript sharing is " + ("on" if on else "off"))
        repolish(self.share_state)
        self.share_retry.setEnabled(on)
        self._refresh_share_status()
        self.settings_changed.emit()

    def _refresh_share_status(self) -> None:
        sharing = self.services.sharing
        st = sharing.status()
        lines = [sharing.state_text(), f"Waiting to be sent: {st['queued']}   ·   Sent so far: {st['sent']}   ·   Failed attempts: {st['failed_attempts']}"]
        if st.get("last_attempt"):
            lines.append(f"Last attempt: {st['last_attempt']}" + (f"   ·   last success: {st['last_success']}" if st.get("last_success") else ""))
        if st.get("last_error"):
            lines.append(f"Last problem: {st['last_error']} (it will be tried again later; your transcripts are not affected)")
        self.share_status.setText("\n".join(lines))

    # ------------------------------------------------------------------------------------------------------------------ Models
    def _models_page(self) -> QWidget:
        table = self._models_tab()
        self.model_table.verticalHeader().hide()
        return _page("Models", "The speech recognition model that works without internet.", table, stretch=False)

    def _study_page(self, s) -> QWidget:
        inner = self._study_tab(s)
        card, layout = _card()
        layout.addWidget(inner)
        return _page("Study", "How summaries, flashcards and questions are written.", card)

    # ------------------------------------------------------------------------------------------------------------------ Study
    def _study_tab(self, s) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        intro = QLabel("MELETRA chooses how your summaries, flashcards and questions are written. It uses an AI model on this computer when one is available and its own offline "
                       "method otherwise. Your lecture text stays on this computer unless you switch on a cloud service yourself under Advanced study engine.")
        intro.setWordWrap(True)
        intro.setObjectName("muted")
        layout.addWidget(intro)
        engine_row = QHBoxLayout()
        self.study_engine_label = QLabel("")
        self.study_engine_label.setWordWrap(True)
        self.study_engine_label.setObjectName("fileName")
        self.study_test_button = QPushButton("Check the study engine")
        self.study_test_button.clicked.connect(self.test_study_provider)
        engine_row.addWidget(self.study_engine_label, 1)
        engine_row.addWidget(self.study_test_button)
        layout.addLayout(engine_row)
        self.study_test_result = QLabel("")
        self.study_test_result.setWordWrap(True)
        self.study_test_result.setObjectName("muted")
        layout.addWidget(self.study_test_result)
        form = _form()
        self.study_form = form
        self.study_language_combo = _combo(STUDY_LANGUAGE_CHOICES, s.study_language if s.study_language in dict(STUDY_LANGUAGE_CHOICES).values() else "auto")
        self.study_cards_spin = QSpinBox()
        self.study_cards_spin.setRange(0, 500)
        self.study_cards_spin.setSpecialValueText("Automatic")
        self.study_cards_spin.setValue(s.study_cards)
        self.study_questions_spin = QSpinBox()
        self.study_questions_spin.setRange(0, 500)
        self.study_questions_spin.setSpecialValueText("Automatic")
        self.study_questions_spin.setValue(s.study_questions)
        for field in (self.study_language_combo, self.study_cards_spin, self.study_questions_spin):
            field.setMinimumWidth(320)                                                    # one width for every field of the page
            field.setMaximumWidth(420)
        _add(form, "Write in:", self.study_language_combo)
        _add(form, "Flashcards:", self.study_cards_spin)
        _add(form, "Questions:", self.study_questions_spin)
        layout.addLayout(form)
        source_note = QLabel("Study items only say what your lecture says. Every flashcard and question is checked before you see it: no fill-in-the-blank, no repeats, "
                             "nothing the lecture does not support.")
        source_note.setWordWrap(True)
        source_note.setObjectName("muted")
        layout.addWidget(source_note)

        self.study_advanced_check = QCheckBox("Advanced study engine")
        self.study_advanced_check.setToolTip("Engine details for troubleshooting, a preferred local model, and the optional cloud service")
        layout.addWidget(self.study_advanced_check)
        self.study_advanced = QWidget()
        advanced = QVBoxLayout(self.study_advanced)
        advanced.setContentsMargins(0, 0, 0, 0)
        self.study_details = QLabel("")
        self.study_details.setWordWrap(True)
        self.study_details.setObjectName("muted")
        self.study_details.setTextInteractionFlags(Qt.TextSelectableByMouse)
        advanced.addWidget(self.study_details)
        adv_form = _form()
        self.study_adv_form = adv_form
        self.study_model_edit = QLineEdit(s.study_model)
        self.study_model_edit.setPlaceholderText("Automatic (for example qwen3:8b)")
        self.study_service_combo = _combo(SERVICE_CHOICES, s.study_provider if s.study_provider in ("openai", "anthropic") else "builtin")
        self.study_url_edit = QLineEdit(s.study_base_url)
        for field in (self.study_model_edit, self.study_service_combo, self.study_url_edit):
            field.setMinimumWidth(320)
            field.setMaximumWidth(420)
        _add(adv_form, "Preferred model:", self.study_model_edit)
        _add(adv_form, "Other AI service:", self.study_service_combo)
        _add(adv_form, "Server address:", self.study_url_edit)
        key_row = QHBoxLayout()
        self.study_key_edit = QLineEdit()
        self.study_key_edit.setEchoMode(QLineEdit.Password)
        self.study_key_edit.setPlaceholderText("Paste your API key")
        self.study_save_key = QPushButton("Save key")
        self.study_remove_key = QPushButton("Remove key")
        self.study_save_key.clicked.connect(self.save_study_key)
        self.study_remove_key.clicked.connect(self.remove_study_key)
        key_row.addWidget(self.study_key_edit, 1)
        key_row.addWidget(self.study_save_key)
        key_row.addWidget(self.study_remove_key)
        self.study_key_label = QLabel("API key:")
        adv_form.addRow(self.study_key_label, key_row)
        self.study_key_status = QLabel("")
        self.study_key_status.setObjectName("muted")
        self.study_key_status.setWordWrap(True)
        adv_form.addRow("", self.study_key_status)
        advanced.addLayout(adv_form)
        self.study_cloud_check = QCheckBox("I agree to send my lecture text to this cloud service")
        self.study_cloud_check.setChecked(bool(getattr(s, "study_cloud_consent", False)))
        advanced.addWidget(self.study_cloud_check)
        self.study_cloud_note = QLabel("Only the text of the lecture you make study material for - never audio, video or PDF files. Nothing is sent without this box, and you can take it back here at any time.")
        self.study_cloud_note.setWordWrap(True)
        self.study_cloud_note.setObjectName("muted")
        advanced.addWidget(self.study_cloud_note)
        self.study_hint = QLabel("")
        self.study_hint.setWordWrap(True)
        self.study_hint.setObjectName("banner")
        advanced.addWidget(self.study_hint)
        layout.addWidget(self.study_advanced)
        layout.addStretch(1)
        self.study_advanced.setVisible(False)
        self.study_advanced_check.toggled.connect(self._study_advanced_toggled)
        self.study_service_combo.currentIndexChanged.connect(lambda _i: self._study_service_changed())
        self._study_service_changed()
        self._refresh_study_engine_label()
        return page

    def _study_service(self) -> str:
        return self.study_service_combo.currentData() or "builtin"

    def _study_service_is_cloud(self) -> bool:
        service = self._study_service()
        if service == "anthropic":
            return True
        return service == "openai" and not is_local_url(self.study_url_edit.text().strip() or OPENAI_URL)

    def _study_service_changed(self) -> None:
        service = self._study_service()
        none = service == "builtin"
        for widget in (self.study_url_edit,):
            self.study_adv_form.setRowVisible(widget, not none)
        for widget in (self.study_key_edit, self.study_save_key, self.study_remove_key, self.study_key_label, self.study_key_status):
            widget.setVisible(not none)
        self.study_url_edit.setPlaceholderText({"openai": "http://localhost:1234/v1 (a server on this computer) or https://api.openai.com/v1", "anthropic": "https://api.anthropic.com/v1"}.get(service, ""))
        self.study_cloud_check.setVisible(self._study_service_is_cloud())
        self.study_cloud_note.setVisible(self._study_service_is_cloud())
        self.study_hint.setText(SERVICE_HINTS[service])
        self.study_hint.setVisible(not none)
        if not none:
            self._refresh_study_key_status()

    def _study_advanced_toggled(self, shown: bool) -> None:
        self.study_advanced.setVisible(shown)
        if shown:
            self._run_study_check(lambda: "\n".join(self.services.study.diagnostics()), self.study_details.setText, "Looking at this computer ...")

    def _refresh_study_engine_label(self) -> None:
        self._run_study_check(lambda: self.services.study.provider_summary(), self.study_engine_label.setText, "MELETRA Automatic Study Engine")

    def _run_study_check(self, work, show, waiting: str) -> None:
        """Looking for a local AI model is a request to this computer: it runs off the window's thread."""
        show(waiting)
        thread = TaskThread(lambda progress, cancelled: work(), self)
        thread.succeeded.connect(show)
        thread.failed.connect(lambda msg: show(f"Not ready: {msg}"))
        self._threads.append(thread)
        self.study_thread = thread
        thread.start()

    def _refresh_study_key_status(self) -> None:
        creds, provider = self.services.credentials, self._study_service()
        masked = creds.masked_study_key(provider) if provider in ("openai", "anthropic") else ""
        if masked:
            self.study_key_status.setText(f"A key is stored in {creds.store_label}: {masked}. It is never written to a settings file or a log.")
        else:
            self.study_key_status.setText(f"No key stored. Keys are kept in {creds.store_label}, never in a settings file."
                                          + (" Local servers often need no key." if provider == "openai" else ""))

    def save_study_key(self) -> None:
        provider = self._study_service()
        try:
            self.services.credentials.set_study_key(provider, self.study_key_edit.text())
        except (ValueError, OSError) as exc:
            self.study_key_status.setText(str(exc))
            return
        self.study_key_edit.clear()
        self._refresh_study_key_status()

    def remove_study_key(self) -> None:
        self.services.credentials.delete_study_key(self._study_service())
        self._refresh_study_key_status()

    def test_study_provider(self) -> None:
        """Save the study choices (so the check uses them), then see what MELETRA would use right now."""
        self._save_study_settings()
        self._run_study_check(lambda: self.services.study.test_provider(), lambda r: self.study_test_result.setText(r if isinstance(r, str) else ("Ready: " if r[0] else "Not ready: ") + r[1]),
                              "Checking ...")
        self._refresh_study_engine_label()

    def _save_study_settings(self) -> None:
        service = self._study_service()
        current = self.services.settings.settings.study_provider
        if service == "builtin" and current == "ollama":
            service = "ollama"                                                            # a setting made by an earlier version stays as it was
        self.services.update_setting(
            study_provider=service, study_model=self.study_model_edit.text().strip(),
            study_base_url=self.study_url_edit.text().strip() if service in ("openai", "anthropic") else self.services.settings.settings.study_base_url,
            study_language=self.study_language_combo.currentData(), study_cards=self.study_cards_spin.value(), study_questions=self.study_questions_spin.value(),
            study_cloud_consent=bool(self._study_service_is_cloud() and self.study_cloud_check.isChecked()))

    # ------------------------------------------------------------------------------------------------------------------ Advanced
    def _advanced_page(self, s) -> QWidget:
        look, ll = _card("Appearance")
        form = _form()
        self.theme_combo = _combo(theme_choices(), s.theme)
        self.window_combo = _combo(WINDOW_CHOICES, s.ui_mode)
        _add(form, "Theme:", self.theme_combo)
        _add(form, "Window style:", self.window_combo)
        ll.addLayout(form)

        about, al = _card("About")
        version = QLabel(f"{APP_NAME} {__version__}")
        version.setObjectName("fileName")
        al.addWidget(version)
        row = QHBoxLayout()
        row.setSpacing(tokens.SPACE["sm"])
        self.about_button = QPushButton(f"About {APP_NAME} ...")
        self.about_button.clicked.connect(self.show_about)
        self.logs_button = QPushButton("Open logs folder")
        self.logs_button.clicked.connect(lambda: self.services.open_path(self.services.paths.logs))
        row.addWidget(self.about_button)
        row.addWidget(self.logs_button)
        row.addStretch(1)
        al.addLayout(row)

        status, sl = _card("What the next lecture will use")
        self._add_status(sl, s)

        tech, tl = _card("Technical settings", "For troubleshooting. The defaults are right for almost everyone.")
        self.advanced_tabs = QTabWidget()
        self.advanced_tabs.setAccessibleName("Technical settings sections")
        self.advanced_tabs.tabBar().setAccessibleName("Technical settings sections")
        self.advanced_tabs.addTab(self._performance_tab(s), "Performance")
        self.advanced_tabs.addTab(self._medical_tab(s), "Medical options")
        self.advanced_tabs.addTab(self._diagnostics_tab(), "Diagnostics")
        self.advanced_tabs.setMinimumHeight(360)
        self.advanced_tabs.currentChanged.connect(lambda i: self._advanced_changed(i))
        tl.addWidget(self.advanced_tabs)
        return _page("Advanced", "Appearance, version information and technical settings.", look, about, status, tech)

    def show_about(self) -> None:
        from .about_dialog import AboutDialog

        AboutDialog(self).exec()

    def _advanced_changed(self, index: int) -> None:
        if self.advanced_tabs.tabText(index) == "Diagnostics" and not self._diag_loaded:
            self.refresh_diagnostics()

    def _tab_changed(self, index: int) -> None:
        if self.tabs.tabText(index) == "Advanced" and self.advanced_tabs.tabText(self.advanced_tabs.currentIndex()) == "Diagnostics" \
                and not self._diag_loaded:
            self.refresh_diagnostics()

    def select_tab(self, name: str) -> None:
        """Old names still work: General, AI Study and Cloud moved; Performance / Medical options / Diagnostics open inside Advanced."""
        name = ALIASES.get(name, name)
        if name in ADVANCED_TABS:
            self.tabs.setCurrentIndex(TAB_NAMES.index("Advanced"))
            self.advanced_tabs.setCurrentIndex(ADVANCED_TABS.index(name))
            if name == "Diagnostics" and not self._diag_loaded:
                self.refresh_diagnostics()
        elif name in TAB_NAMES:
            self.tabs.setCurrentIndex(TAB_NAMES.index(name))

    # ------------------------------------------------------------------------------------------------------------------ save
    def save(self) -> None:
        self._save_study_settings()
        self.services.update_setting(ui_mode=self.window_combo.currentData(), medical_expert=self.medical_expert_check.isChecked())
        super().save()

