"""Medical terms survive a translation unchanged: abbreviations, genes, eponyms, drugs and codes are protected by markers; a sentence whose markers did not come back is translated again."""

from __future__ import annotations

import json
from pathlib import Path

from src.multilingual import source as source_mod
from src.multilingual.source import SourceSegment, SourceTranscript
from src.translation.service import TranslationService
from src.translation.terms import MARKER_RE, TermGuard


def guard():
    return TermGuard(["Purkinje", "bevacizumab", "Bergmann glia", "BRCA1", "mTORC1"])


def test_protected_terms_are_replaced_by_markers_in_order_and_put_back():
    text, terms = guard().protect("The MRI showed Purkinje cells and Bergmann glia; bevacizumab was started for high mTORC1 activity (IL-6, HbA1c).")
    assert terms == ["MRI", "Purkinje", "Bergmann glia", "bevacizumab", "mTORC1", "IL-6", "HbA1c"]
    assert text == "The XQ1Z showed XQ2Z cells and XQ3Z; XQ4Z was started for high XQ5Z activity (XQ6Z, XQ7Z)."
    back, complete = TermGuard.restore("Die XQ1Z zeigte XQ2Z-Zellen und XQ3Z; XQ4Z wurde bei hoher XQ5Z-Aktivität begonnen (XQ6Z, XQ7Z).", terms)
    assert complete and back == "Die MRI zeigte Purkinje-Zellen und Bergmann glia; bevacizumab wurde bei hoher mTORC1-Aktivität begonnen (IL-6, HbA1c)."


def test_ordinary_words_and_numbers_are_not_protected():
    text, terms = guard().protect("The patient is OK, the dose is 5 mg and the heart pumps blood.")
    assert terms == [] and text == "The patient is OK, the dose is 5 mg and the heart pumps blood."


def test_a_missing_marker_is_detected():
    _text, terms = guard().protect("MRI and bevacizumab")
    restored, complete = TermGuard.restore("MRI und XQ2Z", terms)
    assert not complete and restored == "MRI und bevacizumab"
    assert TermGuard.restore("XQ1Z XQ1Z XQ2Z", terms)[1] is False                                              # a marker twice is also wrong


def test_suffixes_glued_to_a_marker_are_kept():
    restored, complete = TermGuard.restore("Hasta XQ1Z'de bir hasar gösterdi", ["Purkinje"])
    assert complete and restored == "Hasta Purkinje'de bir hasar gösterdi"


def test_the_shipped_vocabulary_protects_abbreviations_genes_eponyms_and_drugs():
    real = TermGuard.from_lexicon()
    _text, terms = real.protect("Cajal described Purkinje cells; CSF, GFAP and BRCA1 were measured and aspirin was given.")
    assert {"Cajal", "CSF", "GFAP", "BRCA1", "aspirin"} <= set(terms) and any(t.startswith("Purkinje") for t in terms)           # the vocabulary may know the longer form ("Purkinje cells")
    assert "described" not in terms and "measured" not in terms


# ------------------------------------------------------------------------------------------------------------------ in the service
def lecture(folder: Path, sentence: str, language="en") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    source_mod.write(folder, SourceTranscript(lecture_id=folder.name, segments=[SourceSegment(1, 0.0, 5.0, sentence, detected_language=language, language_source="detector")]))
    return folder


class Careful:
    """Keeps markers, 'translates' the rest."""

    id = "careful:1"

    def __init__(self):
        self.calls = []

    def translate(self, texts, source, target):
        self.calls.append(list(texts))
        return [t.replace("The", "Der").replace("showed", "zeigte") for t in texts]


class Careless(Careful):
    """Turns the markers into something else for the first request only (as a model may)."""

    id = "careless:1"

    def translate(self, texts, source, target):
        self.calls.append(list(texts))
        if len(self.calls) == 1:
            return [MARKER_RE.sub("", t) for t in texts]
        return [t.replace("The", "Der") for t in texts]


def test_the_service_sends_markers_to_the_engine_and_stores_the_original_terms(tmp_path):
    folder = lecture(tmp_path / "l", "The MRI showed Purkinje cells.")
    engine = Careful()
    doc = TranslationService(engine=engine).translate(folder, "de")
    assert len(engine.calls) == 1 and engine.calls[0][0].startswith("The XQ1Z showed XQ2Z") and "MRI" not in engine.calls[0][0] and "Purkinje" not in engine.calls[0][0]   # the engine never sees the terms
    assert doc.segments[0].text.startswith("Der MRI zeigte Purkinje") and doc.segments[0].flags == []


def test_a_sentence_whose_markers_were_lost_is_translated_again_without_them_and_flagged(tmp_path):
    folder = lecture(tmp_path / "l", "The MRI showed Purkinje cells.")
    engine = Careless()
    doc = TranslationService(engine=engine).translate(folder, "de")
    assert len(engine.calls) == 2 and engine.calls[1] == ["The MRI showed Purkinje cells."]
    assert doc.segments[0].text == "Der MRI showed Purkinje cells." and doc.segments[0].flags == ["terms_not_protected"]
    stored = json.loads((folder / "translations" / "de.json").read_bytes().decode("utf-8"))
    assert stored["segments"][0]["flags"] == ["terms_not_protected"]                                            # the flag is part of the stored translation
    assert TranslationService(engine=engine).cached(folder, "de").segments[0].flags == ["terms_not_protected"]
