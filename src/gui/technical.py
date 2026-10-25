"""The "Technical details" window: the classic pages of version 0.1.1 (live log, speed factor, GPU memory, quality reports ...) in a dialog.

The simple screens never show these; they are one click away for anyone who needs them (support, troubleshooting, curiosity).
"""

from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtWidgets import QDialog, QDialogButtonBox, QVBoxLayout, QWidget


class TechnicalDetailsDialog(QDialog):
    """Holds a page for a while. A live page (the running transcription) is handed back through ``restore`` when the dialog closes."""

    def __init__(self, title: str, page: QWidget, parent=None, restore: Optional[Callable[[QWidget], None]] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(900, 640)
        self.page, self._restore = page, restore
        layout = QVBoxLayout(self)
        layout.addWidget(page)
        page.show()
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def done(self, result: int) -> None:
        self.layout().removeWidget(self.page)
        if self._restore is not None:
            self._restore(self.page)                       # back to where it lives (its parent is set there)
        else:
            self.page.setParent(None)
            self.page.deleteLater()
        super().done(result)
