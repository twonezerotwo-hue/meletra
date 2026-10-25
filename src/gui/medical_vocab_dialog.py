"""Settings > Medical vocabulary > Add terms: words the lectures use that the built-in vocabulary may not have (a drug, a structure, a gene).

The terms only influence the RANKING of the medical expert pass (they make a sound-alike believable when the context agrees); they are never
forced replacements, and they are never added to the speech recogniser's hotwords.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (QCheckBox, QDialog, QFileDialog, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QVBoxLayout)

from ..medical.integration import clean_terms


class MedicalVocabularyDialog(QDialog):
    def __init__(self, services, parent=None) -> None:
        super().__init__(parent)
        self.services = services
        self.setWindowTitle("Medical vocabulary")
        self.resize(520, 460)
        layout = QVBoxLayout(self)
        intro = QLabel("Add terms your lectures use - drug names, structures, genes, abbreviations - one per line. The program already knows thousands; "
                       "these help it recognise YOUR terms when the speech recogniser spells them wrongly. They never replace words by themselves: a "
                       "term is only chosen when the sound and the sentence around it agree. Your raw transcript is never edited.")
        intro.setWordWrap(True)
        intro.setObjectName("muted")
        layout.addWidget(intro)
        self.edit = QPlainTextEdit()
        self.edit.setPlaceholderText("e.g.\nbevacizumab\nBergmann glia\nmTORC1")
        self.edit.setPlainText("\n".join(services.settings.settings.custom_terms))
        layout.addWidget(self.edit, 1)
        self.count_label = QLabel("")
        self.count_label.setObjectName("muted")
        layout.addWidget(self.count_label)
        self.enable_check = QCheckBox("Check medical terms in context and write a medical corrected transcript")
        self.enable_check.setChecked(services.settings.settings.medical_expert)
        layout.addWidget(self.enable_check)
        row = QHBoxLayout()
        self.import_button = QPushButton("Import from a text file ...")
        self.import_button.clicked.connect(self.import_file)
        self.save_button = QPushButton("Save")
        self.save_button.setObjectName("primary")
        self.save_button.clicked.connect(self.save)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        row.addWidget(self.import_button)
        row.addStretch(1)
        row.addWidget(self.cancel_button)
        row.addWidget(self.save_button)
        layout.addLayout(row)
        self.edit.textChanged.connect(self._count)
        self._count()

    def terms(self) -> list[str]:
        return clean_terms(self.edit.toPlainText().splitlines())

    def _count(self) -> None:
        self.count_label.setText(f"{len(self.terms())} term(s)")

    def import_file(self, path: Optional[str] = None) -> None:
        if not path or not isinstance(path, str):
            path, _ = QFileDialog.getOpenFileName(self, "Import terms", "", "Text files (*.txt *.csv);;All files (*)")
        if not path:
            return
        try:
            text = Path(path).read_text(encoding="utf-8-sig")
        except (OSError, UnicodeError):
            self.count_label.setText("That file could not be read as text.")
            return
        merged = clean_terms(self.edit.toPlainText().splitlines() + [part.strip() for line in text.splitlines() for part in line.split(",")])
        self.edit.setPlainText("\n".join(merged))

    def save(self) -> None:
        self.services.update_setting(custom_terms=self.terms(), medical_expert=self.enable_check.isChecked())
        self.accept()


def open_medical_vocabulary(services, parent=None) -> bool:
    dialog = MedicalVocabularyDialog(services, parent)
    return dialog.exec() == QDialog.Accepted
