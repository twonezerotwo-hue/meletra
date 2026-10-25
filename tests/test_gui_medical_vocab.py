"""Settings > Medical vocabulary > Add terms, the setting reaching the job, and the medical view in the transcript viewer."""

import pytest

from src.gui.medical_vocab_dialog import MedicalVocabularyDialog
from src.gui.transcript_view import VIEWS
from src.services.settings_service import Settings, sanitize
from src.services.transcription_service import JobRequest, build_run_options
from tests.test_gui_simple import simple_env  # noqa: F401 - fixture


def test_settings_default_to_the_medical_layer_with_no_terms():
    s = Settings()
    assert s.medical_expert is True and s.custom_terms == []
    assert sanitize({"custom_terms": ["bevacizumab", 5, None]}).custom_terms == ["bevacizumab", "5", "None"]
    assert sanitize({"custom_terms": "not a list", "medical_expert": "yes"}).custom_terms == [] and sanitize({"medical_expert": "yes"}).medical_expert is True


def test_terms_are_added_cleaned_and_saved(simple_env):
    window, services, *_ = simple_env
    dialog = MedicalVocabularyDialog(services)
    dialog.edit.setPlainText("bevacizumab\n  mTORC1 \n\nbevacizumab\n# a comment\nBergmann glia")
    assert dialog.terms() == ["bevacizumab", "mTORC1", "Bergmann glia"] and "3 term" in dialog.count_label.text()
    dialog.save()
    assert services.settings.settings.custom_terms == ["bevacizumab", "mTORC1", "Bergmann glia"]
    again = MedicalVocabularyDialog(services)
    assert again.edit.toPlainText().splitlines() == ["bevacizumab", "mTORC1", "Bergmann glia"]
    again.deleteLater()
    dialog.deleteLater()


def test_terms_can_be_imported_from_a_text_file(simple_env, tmp_path):
    window, services, *_ = simple_env
    path = tmp_path / "terms.txt"
    path.write_text("tanycyte, pituicyte\nmicroglia\n", encoding="utf-8")
    dialog = MedicalVocabularyDialog(services)
    dialog.edit.setPlainText("microglia")
    dialog.import_file(str(path))
    assert dialog.terms() == ["microglia", "tanycyte", "pituicyte"]
    dialog.import_file(str(tmp_path / "missing.txt"))
    assert dialog.terms() == ["microglia", "tanycyte", "pituicyte"]
    dialog.deleteLater()


def test_the_medical_layer_can_be_switched_off(simple_env):
    window, services, *_ = simple_env
    dialog = MedicalVocabularyDialog(services)
    dialog.enable_check.setChecked(False)
    dialog.save()
    assert services.settings.settings.medical_expert is False
    dialog.deleteLater()


def test_the_settings_reach_the_job_and_the_pipeline(simple_env):
    window, services, *_ = simple_env
    services.update_setting(custom_terms=["bevacizumab"], medical_expert=True)
    request = window.home.request_template()
    assert request.custom_terms == ["bevacizumab"] and request.medical_expert is True
    from src.services.transcription_service import load_config

    options, _cfg = build_run_options(request, load_config(), services.models, services.paths)
    assert options.medical_expert is True and options.custom_medical_terms == ["bevacizumab"]
    request.medical_expert = False
    assert build_run_options(request, load_config(), services.models, services.paths)[0].medical_expert is False


def test_an_older_request_file_still_loads(tmp_path):
    path = tmp_path / "request.json"
    path.write_text('{"source": "a.mp3", "model_key": "whisper-large-v3"}', encoding="utf-8")
    request = JobRequest.load(path)
    assert request.medical_expert is True and request.custom_terms == []


def test_the_settings_dialogs_offer_the_vocabulary_button(simple_env):
    window, services, *_ = simple_env
    dialog = window.make_settings_dialog()
    assert dialog.medical_vocab_general_button.text().startswith("Medical vocabulary")
    assert dialog.medical_vocab_button.text().startswith("Medical vocabulary")
    dialog.deleteLater()


def test_the_transcript_viewer_offers_the_medical_transcript(simple_env, tmp_path):
    window, services, *_ = simple_env
    assert VIEWS[0][1] == "local_clean_transcript.txt" and VIEWS[-1][1] == "medical_corrected_transcript.txt"
    folder = tmp_path / "lecture"
    folder.mkdir()
    (folder / "local_clean_transcript.txt").write_text("clean text\n", encoding="utf-8")
    (folder / "medical_corrected_transcript.txt").write_text("medical text\n", encoding="utf-8")
    view = window.transcript_view
    view.show_transcript(folder)
    assert view.view_combo.currentData() == "local_clean_transcript.txt" and "clean text" in view.text.toPlainText()
    view.view_combo.setCurrentIndex(view.view_combo.findData("medical_corrected_transcript.txt"))
    assert "medical text" in view.text.toPlainText()
