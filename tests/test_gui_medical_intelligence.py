"""Settings > Medical Intelligence: read-only page with the database facts and a search; the vocabulary page next to it is untouched."""

import pytest

from src.gui.medical_intelligence_dialog import MedicalIntelligenceDialog, _search, _summary
from src.medical_knowledge.ingest.builder import KnowledgeBuilder
from src.medical_knowledge.sources.curated import CuratedSource
from src.medical_knowledge.sources.fixture import FixtureSource
from tests.test_gui_simple import simple_env  # noqa: F401 - fixture


@pytest.fixture(scope="module")
def db_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("gui") / "medical_knowledge_public.db"
    builder = KnowledgeBuilder(path, "fixture", "test")
    for source in (CuratedSource(), FixtureSource()):
        source.export(builder)
    builder.finalize()
    return path


def wait_for(qapp, condition, seconds=10.0):
    import time

    end = time.time() + seconds
    while time.time() < end and not condition():
        qapp.processEvents()
        time.sleep(0.02)
    return condition()


def test_summary_text_of_a_database_and_of_none(db_path):
    from src.medical_knowledge.cli import report

    text = _summary(report(str(db_path)))
    assert "MedicalKnowledge" in text and "Concepts:" in text and "Sources:" in text and "Status: ready" in text
    assert "No medical knowledge" in _summary({"installed": False, "status": "No medical knowledge database is installed."})


def test_search_returns_preferred_names(db_path):
    out = _search(str(db_path), "dendritic")
    assert "dendritic" in out.lower() and "->" in out
    assert "Nothing found" in _search(str(db_path), "qzxvqzxv")


def test_the_dialog_loads_in_the_background_and_enables_the_search(qapp, db_path, monkeypatch):
    monkeypatch.setenv("MLT_KNOWLEDGE_DB", str(db_path))
    dialog = MedicalIntelligenceDialog()
    assert wait_for(qapp, lambda: "Concepts:" in dialog.info_text.toPlainText())
    assert dialog.search_button.isEnabled()
    dialog.query_edit.setText("axon")
    dialog.search()
    assert wait_for(qapp, lambda: "->" in dialog.result_text.toPlainText())
    dialog.accept()
    dialog.deleteLater()


def test_without_a_database_the_page_says_so_and_search_stays_off(qapp, monkeypatch, tmp_path):
    monkeypatch.setenv("MLT_KNOWLEDGE_DB", str(tmp_path / "missing.db"))
    monkeypatch.setattr("src.medical_knowledge.runtime.knowledge.DEFAULT_NAMES", ("no_such_default.db",))
    dialog = MedicalIntelligenceDialog()
    assert wait_for(qapp, lambda: "no medical knowledge database" in dialog.info_text.toPlainText().lower())
    assert not dialog.search_button.isEnabled()
    dialog.accept()
    dialog.deleteLater()


def test_settings_has_the_button_and_keeps_the_vocabulary_page(simple_env):
    window, services, *_ = simple_env
    from src.gui.settings_simple import SimpleSettingsDialog

    dialog = SimpleSettingsDialog(services)
    assert dialog.medical_intelligence_button.text().startswith("Medical Intelligence settings")
    assert (dialog.intelligence_badge.text(), dialog.intelligence_state.text()) == ("Experimental", "OFF")
    assert dialog.medical_vocab_general_button.text().startswith("Medical vocabulary")
    rows = {k: v.text() for k, v in dialog.status_values.items()}                       # Settings > General shows what the next lecture uses
    assert rows["Transcription model:"].startswith("Whisper Large-v3") and rows["Medical correction:"] == "Medical Expert v3 (on)"
    assert rows["Experimental Medical Intelligence:"] == "v4: Off" and rows["Study Quality:"].startswith("MELETRA Automatic Study Engine")
    assert rows["Privacy:"].startswith("transcription runs on this computer; the audio never leaves it") and "transcript sharing is off" in rows["Privacy:"]
    from tests.gui_support import wait_for

    wait_for(lambda: dialog.status_values["Medical knowledge:"].text() != "checking ...", what="the knowledge status line")
    dialog.reject()
    dialog.deleteLater()


def test_status_lines_say_what_the_next_lecture_uses():
    from types import SimpleNamespace

    from src.gui.settings_status import knowledge_line, status_lines

    s = SimpleNamespace(model_key="whisper-large-v3", medical_expert=True, medical_intelligence="on", cloud_enabled=True, study_provider="builtin")
    rows = dict(status_lines(s, lambda key: False, {"installed": True, "ok": True, "name": "MedicalKnowledge", "build_version": "2026.10", "profile": "public",
                                                     "counts": {"concepts": 66668}}))
    assert rows["Transcription model:"] == "Whisper Large-v3 (not installed yet)" and rows["Experimental Medical Intelligence:"] == "v4: On - Experimental"
    assert rows["Medical knowledge:"] == "MedicalKnowledge 2026.10 (public), 66,668 concepts - ready" and "Deepgram" in rows["Privacy:"]
    assert knowledge_line({"installed": False}).startswith("not installed")
    assert "Medical Expert works alone" in knowledge_line({"installed": True, "ok": False, "status": ""})


def test_the_mode_switch_is_off_by_default_and_saves_the_choice(qapp, db_path, monkeypatch):
    from types import SimpleNamespace

    from PySide6.QtWidgets import QLabel

    monkeypatch.setenv("MLT_KNOWLEDGE_DB", str(db_path))
    saved = {}
    services = SimpleNamespace(settings=SimpleNamespace(settings=SimpleNamespace(medical_intelligence="off")), update_setting=lambda **kw: saved.update(kw))
    dialog = MedicalIntelligenceDialog(services)
    assert dialog.mode_combo.currentData() == "off" and [dialog.mode_combo.itemData(i) for i in range(dialog.mode_combo.count())] == ["off", "on"]
    assert [dialog.mode_combo.itemText(i) for i in range(2)] == ["Off (default)", "On - Experimental"]          # the user sees Off / On - Experimental
    texts = " ".join(label.text() for label in dialog.findChildren(QLabel))
    assert "Experimental / Advanced" in texts and "remains experimental" in texts
    assert not any(word in texts for word in ("precision", "heldout", "%", "benchmark"))                          # no benchmark jargon for normal users
    dialog.mode_combo.setCurrentIndex(dialog.mode_combo.findData("on"))
    assert saved == {"medical_intelligence": "on"}
    again = MedicalIntelligenceDialog(SimpleNamespace(settings=SimpleNamespace(settings=SimpleNamespace(medical_intelligence="shadow")), update_setting=lambda **kw: None))
    assert again.mode_combo.currentData() == "shadow" and again.mode_combo.count() == 3                       # the developer mode is shown only when it is the setting
    for d in (dialog, again):
        d.accept()
        d.deleteLater()
