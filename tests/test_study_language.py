"""Study material per language: each language has its own pack (one never overwrites another), it is tied to the version of the saved transcript it was made from, and a lecture
spoken in several languages is studied in one of its translations."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.multilingual import source as source_mod
from src.multilingual.source import SourceSegment, SourceTranscript
from src.services.settings_service import SettingsService
from src.services.study_service import StudyService
from src.study.store import StudyStore
from src.translation.service import TranslationService
from tests.study_support import GLYCOLYSIS, write_lecture_folder


class Engine:
    id = "fake:1"

    def translate(self, texts, source, target):
        return [f"{t}" for t in texts]


def multilingual_lecture(folder: Path) -> Path:
    """Korean, Hindi and English parts built from the synthetic biochemistry lecture (the sentences are English; only the labels differ, which is what the study code looks at)."""
    folder.mkdir(parents=True, exist_ok=True)
    languages = ["ko", "hi", "en"]
    segments = [SourceSegment(i + 1, i * 8.0, i * 8.0 + 7.5, GLYCOLYSIS[i], detected_language=languages[i % 3], language_source="detector") for i in range(len(GLYCOLYSIS))]
    source_mod.write(folder, SourceTranscript(lecture_id=folder.name, segments=segments))
    (folder / "metadata.json").write_text(json.dumps({"language": {"detected": "ko"}, "subject": {"subjects": []}}), encoding="utf-8")
    (folder / "local_transcript.json").write_text(json.dumps({"language": "ko", "segments": [{"start": s.start, "end": s.end, "text": s.source_text, "language": s.detected_language} for s in segments]}),
                                                  encoding="utf-8")
    return folder


@pytest.fixture
def service(tmp_path):
    return StudyService(SettingsService(tmp_path / "settings.json"), credentials=None)


def translate(folder: Path, target: str) -> None:
    TranslationService(engine=Engine()).translate(folder, target)


def test_the_first_language_keeps_the_old_layout_and_another_language_gets_its_own_folder(tmp_path):
    folder = write_lecture_folder(tmp_path / "lecture")
    english = StudyStore(folder, "en")
    english.folder.mkdir(parents=True)
    (english.folder / "study_pack.json").write_text(json.dumps({"language": "en"}), encoding="utf-8")
    assert english.folder == folder / "study"
    german = StudyStore(folder, "de")
    assert german.folder == folder / "study" / "de"                                                          # English is already there: German never overwrites it
    german.folder.mkdir(parents=True)
    (german.folder / "study_pack.json").write_text(json.dumps({"language": "de"}), encoding="utf-8")
    assert StudyStore(folder, "en").folder == folder / "study" and StudyStore(folder, "de").folder == folder / "study" / "de"
    assert StudyStore.languages_on_disk(folder) == ["en", "de"]
    assert StudyStore(folder).folder == folder / "study"                                                     # no language given: the old behaviour


def test_a_lecture_in_several_languages_needs_a_translation_before_it_can_be_studied(tmp_path, service):
    folder = multilingual_lecture(tmp_path / "mixed")
    assert service.study_languages(folder) == [] and service.needs_translation(folder) is True
    translate(folder, "en")
    assert service.study_languages(folder) == ["en"] and service.needs_translation(folder) is False
    translate(folder, "tr")
    assert service.study_languages(folder) == ["en", "tr"]


def test_a_single_language_lecture_can_be_studied_in_its_own_language_and_in_every_translation(tmp_path, service):
    folder = write_lecture_folder(tmp_path / "english")
    (folder / "metadata.json").write_text(json.dumps({"language": {"detected": "en"}}), encoding="utf-8")
    source_mod.ensure_stored(folder)
    translate(folder, "de")
    assert service.study_languages(folder) == ["en", "de"]


def test_each_language_has_its_own_pack_and_the_first_is_not_overwritten(tmp_path, service):
    folder = multilingual_lecture(tmp_path / "mixed")
    translate(folder, "en")
    translate(folder, "tr")
    service.set_language(folder, "en")
    state = service.prepare(folder)
    service.generate(state)
    english_bytes = (folder / "study" / "study_pack.json").read_bytes()
    assert state.pack.language == "en" and state.store.folder == folder / "study"
    service.set_language(folder, "tr")
    assert service.study_language(folder) == "tr"
    turkish = service.prepare(folder)
    service.generate(turkish)
    assert turkish.store.folder == folder / "study" / "tr" and turkish.pack.language == "tr"
    assert (folder / "study" / "study_pack.json").read_bytes() == english_bytes                                   # the English pack is untouched
    assert StudyStore.languages_on_disk(folder) == ["en", "tr"]
    service.set_language(folder, "en")
    assert service.open(folder).pack is not None and service.open(folder).pack.language == "en"


def test_progress_is_kept_per_language(tmp_path, service):
    folder = multilingual_lecture(tmp_path / "mixed")
    translate(folder, "en")
    translate(folder, "tr")
    service.set_language(folder, "en")
    english = service.prepare(folder)
    service.generate(english)
    english.progress.add_session({"kind": "test", "correct": 3})
    english.progress.save()
    service.set_language(folder, "tr")
    turkish = service.prepare(folder)
    assert turkish.progress.sessions == []                                                                          # the Turkish study has its own history
    assert (folder / "study" / "study_progress.json").is_file()


def test_a_pack_made_from_an_older_transcript_is_made_again(tmp_path, service):
    folder = multilingual_lecture(tmp_path / "mixed")
    translate(folder, "en")
    service.set_language(folder, "en")
    state = service.prepare(folder)
    service.generate(state)
    assert service.open(folder).pack is not None
    source = source_mod.read_file(folder)
    source.segments[0].source_text = "A corrected first sentence about glycolysis in the cytoplasm of the cell."
    source_mod.write(folder, source)                                                                              # the saved transcript changed: new revision
    assert service.study_languages(folder) == []                                                                  # the English translation belonged to the old text: it is no longer offered
    translate(folder, "en")                                                                                       # translating again makes the translation of the new text
    again = service.open(folder)
    assert again.pack is None and any("earlier version" in w for w in again.warnings)


def test_the_choice_for_a_lecture_is_remembered_and_forgotten_when_the_language_goes_away(tmp_path, service):
    folder = multilingual_lecture(tmp_path / "mixed")
    translate(folder, "en")
    translate(folder, "de")
    service.set_language(folder, "de")
    assert service.chosen_language(folder) == "de" and service.study_language(folder) == "de"
    for name in ("de.json",):
        (folder / "translations" / name).unlink()                                                                    # the German translation is no longer there
    assert service.chosen_language(folder) == ""


def test_every_output_language_has_its_own_study_wording():
    from src.study.i18n import EN, canonical_true_false, strings, true_false_pair

    for code in ("en", "tr", "de", "hu", "ko", "zh", "hi", "nl", "fr", "es"):
        table = strings(code)
        assert set(table) == set(EN), code                                                                          # every phrase the study method uses exists in every language
        if code != "en":
            assert table["what_is"] != EN["what_is"] and table["true_false"] != EN["true_false"], code               # and it is not the English one
        for key, text in table.items():
            assert text.count("{x}") == EN[key].count("{x}") and text.count("{a}") == EN[key].count("{a}") and text.count("{b}") == EN[key].count("{b}"), (code, key)
        true, false = true_false_pair(code)
        assert canonical_true_false(true.casefold())[0] == true and canonical_true_false(false.casefold())[0] == false, code
    assert canonical_true_false("Wahr")[0] == "Richtig" and canonical_true_false("onwaar")[1] == ("Juist", "Onjuist")


def test_a_german_pack_is_written_with_german_frames(tmp_path, service):
    folder = multilingual_lecture(tmp_path / "mixed")
    translate(folder, "de")
    service.set_language(folder, "de")
    state = service.prepare(folder)
    service.generate(state)
    text = json.dumps(state.pack.to_dict(), ensure_ascii=False)
    assert state.pack.language == "de" and ("Was ist" in text or "Erklären Sie" in text or "Richtig oder falsch" in text or "Füllen Sie die Lücke" in text)
    assert "What is" not in text and "Fill in the blank" not in text and "True or false" not in text                  # nothing is reused from the English wording
