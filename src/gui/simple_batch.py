"""While several files (or a document) are being converted: what is happening now, one line per file, PAUSE / CANCEL.

Used for batches and for documents / pictures. A single recording keeps the 0.2 "Transcribing..." screen. Nothing technical here.
"""

from __future__ import annotations

from typing import Any, Optional

from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QProgressBar, QPushButton, QStyle, QStyledItemDelegate,
                               QVBoxLayout, QWidget)

from ..ingest import queue as q
from . import friendly, tokens

STATE_TEXT = {q.QUEUED: "Waiting", q.PROCESSING: "Working ...", q.COMPLETED: "Done", q.FAILED: "Failed", q.PAUSED: "Paused",
              q.CANCELLED: "Cancelled"}


ROLE_NAME, ROLE_STATE, ROLE_NUMBER = Qt.UserRole + 1, Qt.UserRole + 2, Qt.UserRole + 3
#: queue state -> (chip colour role, chip background role)
CHIP_ROLES = {q.QUEUED: ("neutral", "neutral_bg"), q.PROCESSING: ("teal", "teal_soft"), q.COMPLETED: ("success", "success_bg"),
              q.FAILED: ("error", "error_bg"), q.PAUSED: ("warning", "warning_bg"), q.CANCELLED: ("neutral", "neutral_bg")}


class QueueRowDelegate(QStyledItemDelegate):
    """One clean row per file: "1  Neuroanatomy.mp4 ........ [Transcribing]". The item text (name - status) stays for screen readers and tests."""

    ROW = 48

    def sizeHint(self, option, index) -> QSize:  # noqa: N802
        return QSize(option.rect.width(), self.ROW)

    def paint(self, painter: QPainter, option, index) -> None:  # noqa: N802
        c = tokens.current()
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(option.rect).adjusted(2, 3, -2, -3)
        state = index.data(ROLE_STATE)
        current = state == q.PROCESSING
        painter.setPen(QColor(c.teal if current else c.border))
        painter.setBrush(QColor(c.surface_hover if option.state & QStyle.State_MouseOver else c.surface))
        painter.drawRoundedRect(rect, tokens.RADIUS["control"], tokens.RADIUS["control"])
        number_font = QFont(option.font)
        number_font.setPixelSize(tokens.FONT_PX["secondary"])
        painter.setFont(number_font)
        painter.setPen(QColor(c.text3))
        painter.drawText(QRectF(rect.left() + 14, rect.top(), 26, rect.height()), Qt.AlignVCenter | Qt.AlignLeft, str(index.data(ROLE_NUMBER)))
        label = STATE_TEXT.get(state, str(state))
        chip_font = QFont(option.font)
        chip_font.setPixelSize(tokens.FONT_PX["meta"])
        chip_font.setWeight(QFont.DemiBold)
        metrics = QFontMetrics(chip_font)
        chip_w = metrics.horizontalAdvance(label) + 22
        chip = QRectF(rect.right() - 14 - chip_w, rect.center().y() - 11, chip_w, 22)
        fg_role, bg_role = CHIP_ROLES.get(state, ("neutral", "neutral_bg"))
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(getattr(c, bg_role)))
        painter.drawRoundedRect(chip, 11, 11)
        painter.setFont(chip_font)
        painter.setPen(QColor(getattr(c, fg_role)))
        painter.drawText(chip, Qt.AlignCenter, label)
        name_font = QFont(option.font)
        name_font.setPixelSize(tokens.FONT_PX["body"])
        painter.setFont(name_font)
        painter.setPen(QColor(c.text))
        name_rect = QRectF(rect.left() + 46, rect.top(), chip.left() - rect.left() - 58, rect.height())
        painter.drawText(name_rect, Qt.AlignVCenter | Qt.AlignLeft, QFontMetrics(name_font).elidedText(str(index.data(ROLE_NAME)), Qt.ElideMiddle, int(name_rect.width())))
        painter.restore()


class SimpleBatch(QWidget):
    pause_requested = Signal()
    resume_requested = Signal()
    cancel_requested = Signal()
    open_item_requested = Signal(int)
    open_folder_requested = Signal()
    new_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.queue: Optional[q.BatchQueue] = None
        outer = QVBoxLayout(self)
        outer.addStretch(1)
        column = QWidget()
        column.setMaximumWidth(760)
        column_layout = QVBoxLayout(column)
        column_layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(column, 100)
        row.addStretch(1)
        outer.addLayout(row)
        outer.addStretch(1)

        panel = QFrame()
        panel.setObjectName("card")
        column_layout.addWidget(panel)
        root = QVBoxLayout(panel)
        root.setContentsMargins(tokens.SPACE["xl"], tokens.SPACE["xl"], tokens.SPACE["xl"], tokens.SPACE["lg"])
        root.setSpacing(tokens.SPACE["sm"])

        titles = QVBoxLayout()
        titles.setSpacing(tokens.SPACE["xs"])
        self.title = QLabel("Converting ...")
        self.title.setObjectName("title")
        self.stage = QLabel("")
        self.stage.setObjectName("statusLine")
        self.stage.setWordWrap(True)
        titles.addWidget(self.title)
        titles.addWidget(self.stage)
        root.addLayout(titles)
        bar = QHBoxLayout()                                         # the same "bar and percent on one line" as the single-file page
        bar.setSpacing(tokens.SPACE["md"])
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setTextVisible(False)
        self.percent_label = QLabel("0%")
        self.percent_label.setObjectName("percent")
        self.percent_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        bar.addWidget(self.progress_bar, 1)
        bar.addWidget(self.percent_label)
        root.addLayout(bar)
        under_bar = QHBoxLayout()
        self.next_label = QLabel("")                                # "Next: Physiology.wav"
        self.next_label.setObjectName("meta")
        self.eta_label = QLabel("")
        self.eta_label.setObjectName("meta")
        under_bar.addWidget(self.next_label, 1)
        under_bar.addWidget(self.eta_label)
        root.addLayout(under_bar)

        self.list = QListWidget()
        self.list.setObjectName("queueList")
        self.list.setItemDelegate(QueueRowDelegate(self.list))
        self.list.setMinimumHeight(150)
        self.list.setMouseTracking(True)
        self.list.setAccessibleName("Files in this queue")
        self.list.itemDoubleClicked.connect(lambda item: self.open_item_requested.emit(self.list.row(item)))
        root.addWidget(self.list)
        self.hint = QLabel("")
        self.hint.setObjectName("meta")
        self.hint.setWordWrap(True)
        root.addWidget(self.hint)

        buttons = QHBoxLayout()
        buttons.setSpacing(tokens.SPACE["sm"])
        self.pause_button = QPushButton("Pause")
        self.pause_button.setFocusPolicy(Qt.ClickFocus)
        self.pause_button.clicked.connect(self._pause_clicked)
        self.resume_button = QPushButton("Resume")
        self.resume_button.setObjectName("primary")
        self.resume_button.clicked.connect(self.resume_requested.emit)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setFocusPolicy(Qt.ClickFocus)
        self.cancel_button.clicked.connect(self._cancel_clicked)
        self.open_button = QPushButton("Open Files")
        self.open_button.clicked.connect(self.open_folder_requested.emit)
        self.new_button = QPushButton("Convert more files")
        self.new_button.clicked.connect(self.new_requested.emit)
        for b in (self.pause_button, self.resume_button, self.cancel_button, self.open_button, self.new_button):
            buttons.addWidget(b)
        buttons.addStretch(1)
        root.addLayout(buttons)

    # ------------------------------------------------------------------------------------------------------------------ content
    def begin(self, queue: q.BatchQueue) -> None:
        self.queue = queue
        self.progress_bar.setValue(0)
        self.percent_label.setText("0%")
        self.eta_label.setText("")
        self.stage.setText("")
        self.hint.setText("Double-click a finished file to open it." if len(queue.items) > 1 else "")
        self.pause_button.setEnabled(True)
        self.cancel_button.setEnabled(True)
        self.refresh()

    def refresh(self) -> None:
        queue = self.queue
        if queue is None:
            return
        self.list.clear()
        for number, item in enumerate(queue.items, 1):
            text = f"{item.name}  ·  {STATE_TEXT[item.state]}"
            if item.state in (q.FAILED, q.CANCELLED) and item.error:
                text += f": {item.error[:140]}"
            row = QListWidgetItem(text)
            row.setData(ROLE_NAME, item.name)
            row.setData(ROLE_STATE, item.state)
            row.setData(ROLE_NUMBER, number)
            row.setToolTip(item.error or item.detected.label)
            self.list.addItem(row)
        state, total = queue.state, len(queue.items)
        coming = next((item for item in queue.items if item.state == q.QUEUED), None)
        self.next_label.setText(f"Next: {coming.name}" if coming is not None and state in (q.RUNNING, q.PAUSED) else "")
        number = queue.position[0]
        only_document = total == 1 and not queue.items[0].detected.is_media
        running = state in (q.RUNNING,)
        if state == q.RUNNING:
            self.title.setText("Reading document..." if only_document else (f"Converting {number} of {total}" if total > 1 else "Transcribing..."))
        elif state == q.PAUSED:
            self.title.setText("Paused")
            self.stage.setText("Your progress is saved. Press Resume to continue.")
        elif state == q.DONE:
            done, failed = queue.count(q.COMPLETED), queue.count(q.FAILED) + queue.count(q.CANCELLED)
            self.title.setText("Done" if not failed else f"{done} of {total} finished")
            self.stage.setText("" if not failed else "Some files could not be converted. Double-click a line to see what happened.")
            self.progress_bar.setValue(1000 if not failed else self.progress_bar.value())
        elif state == q.CANCELLED:
            self.title.setText("Cancelled")
            self.stage.setText("Files that were already finished are kept.")
        self.pause_button.setVisible(running)
        self.cancel_button.setVisible(running or state == q.PAUSED)
        self.resume_button.setVisible(state == q.PAUSED)
        self.resume_button.setEnabled(state == q.PAUSED)
        finished = state in (q.DONE, q.CANCELLED)
        self.open_button.setVisible(finished)
        self.new_button.setVisible(finished)
        if running:
            self.pause_button.setEnabled(True)
            self.pause_button.setText("Pause")

    def set_stage(self, event: dict[str, Any]) -> None:
        """Before the first words of a recording exist: the stage in plain words, a real percent or a busy bar."""
        item = self.queue.current if self.queue is not None else None
        kind = "video" if item is not None and item.detected.kind == "video" else "audio"
        title, line = friendly.stage_texts(str(event.get("stage", "")), kind)
        if event.get("stage") == "language" and event.get("total"):
            line = f"{line} ({int(event.get('done', 0))} of {int(event['total'])})"
        self.stage.setText(f"{title}\n{line}")
        self.eta_label.setText("")
        if event.get("percent") is None:
            self.progress_bar.setRange(0, 0)
            self.percent_label.setText("")
        else:
            self.progress_bar.setRange(0, 1000)
            percent = min(max(float(event["percent"]), 0.0), 100.0)
            self.progress_bar.setValue(int(percent * 10))
            self.percent_label.setText(f"{percent:.0f}%")

    def set_progress(self, index: int, event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "stage":
            self.set_stage(event)
            return
        percent = min(max(float(event.get("percent", 0.0)), 0.0), 100.0)
        if self.progress_bar.maximum() == 0:
            self.progress_bar.setRange(0, 1000)
        if kind == "document_progress":
            if event.get("stage") == "recognizing":
                self.stage.setText(f"Recognizing text...\nPage {event.get('page')} of {event.get('pages')}")
            else:
                self.stage.setText("Reading document...")
            self.eta_label.setText("")
        else:
            self.stage.setText("Transcribing...")
            rtf = float(event.get("rtf") or 0.0)
            eta = event.get("eta")
            self.eta_label.setText(friendly.time_left(eta if rtf > 0 and eta is not None else None))
        self.progress_bar.setValue(int(percent * 10))
        self.percent_label.setText(f"{percent:.0f}%")

    def _pause_clicked(self) -> None:
        self.pause_button.setEnabled(False)
        self.pause_button.setText("Pausing ...")
        self.pause_requested.emit()

    def _cancel_clicked(self) -> None:
        self.cancel_button.setEnabled(False)
        self.cancel_requested.emit()
