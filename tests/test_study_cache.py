"""Study material is stored WITH the lecture and restored instantly: no engine is probed, nothing is generated, the student's progress is kept. It is made again only when the saved
transcript changed (then it says so) or when the student asks for it (Regenerate). Material made by an earlier version of MELETRA is kept as it is, with a note."""

from __future__ import annotations

import json

import pytest

from src.study.curate import GENERATION_VERSION
from src.study.store import StudyStore
from tests.gui_support import make_services
from tests.study_support import write_lecture_folder


@pytest.fixture
def lecture(qapp, tmp_path):
    services = make_services(tmp_path, opened=[])
    folder = write_lecture_folder(tmp_path / "Medical Transcription" / "IMG_0001")
    return services, folder


def make_pack(services, folder):
    state = services.study.prepare(folder, title="Lecture")
    services.study.generate(state)
    return state


def test_a_stored_pack_is_restored_instantly_without_asking_any_engine(lecture):
    services, folder = lecture
    made = make_pack(services, folder)
    asked = []
    services.study.transport = lambda *a, **k: asked.append(a) or {}
    services.study._choice = None
    state = services.study.open(folder, "Lecture")
    assert state.ready and not state.warnings and asked == []
    assert [c.id for c in state.pack.flashcards] == [c.id for c in made.pack.flashcards]
    assert state.pack.generation["version"] == GENERATION_VERSION and state.pack.generation["engine"] == "MELETRA Automatic Study Engine"


def test_the_pack_says_what_it_belongs_to(lecture):
    services, folder = lecture
    pack = make_pack(services, folder).pack
    generation = pack.generation
    assert generation["language"] == "en" and generation["mode"] == "builtin" and generation["source_revision"]
    assert all(c.generation == GENERATION_VERSION and c.evidence and c.sources[0].start is not None for c in pack.flashcards)       # the moment of the lecture it comes from


def test_material_made_by_an_earlier_version_is_kept_with_a_note(lecture):
    services, folder = lecture
    make_pack(services, folder)
    path = StudyStore(folder, "en").pack_path
    data = json.loads(path.read_text(encoding="utf-8"))
    data.pop("generation")
    for item in data["flashcards"] + data["questions"]:
        item.pop("generation", None)
        item.pop("evidence", None)
    path.write_text(json.dumps(data), encoding="utf-8")
    state = services.study.open(folder, "Lecture")
    assert state.ready and state.pack.flashcards                                       # still there, not silently replaced
    assert any("earlier version of MELETRA" in w and "Regenerate" in w for w in state.warnings)
    assert "earlier version of MELETRA" in services.study.describe_pack(state.pack)


def test_a_changed_transcript_makes_the_stored_material_stale(lecture):
    services, folder = lecture
    make_pack(services, folder)
    data = json.loads((folder / "local_transcript.json").read_text(encoding="utf-8"))
    data["segments"][0]["text"] = "A corrected opening sentence about glycolysis in the cytoplasm of the cell."
    (folder / "local_transcript.json").write_text(json.dumps(data), encoding="utf-8")
    (folder / "source_transcript.json").unlink(missing_ok=True)
    state = services.study.open(folder, "Lecture")
    assert state.pack is None and any("earlier version of the transcript" in w for w in state.warnings)


def test_regenerating_is_explicit_and_keeps_the_students_progress(lecture):
    services, folder = lecture
    state = make_pack(services, folder)
    card = state.pack.flashcards[0]
    state.progress.rate_card(card.id, "know", card.topic)
    state.progress.path = state.store.progress_path
    state.progress.save()
    first_created = state.pack.created
    again = services.study.prepare(folder, title="Lecture")
    services.study.generate(again)
    assert again.pack.flashcards and (again.pack.created >= first_created)
    assert again.store.progress().card_state(card.id) == "learning"                  # the rating of a card that is still in the pack survives


def test_each_language_has_its_own_study_pack_and_neither_replaces_the_other(lecture):
    services, folder = lecture
    english = make_pack(services, folder).pack
    english_bytes = StudyStore(folder, "en").pack_path.read_bytes()
    services.settings.update(study_language="tr")
    turkish_state = services.study.prepare(folder, title="Lecture")
    services.study.generate(turkish_state)
    en_store, tr_store = StudyStore(folder, "en"), StudyStore(folder, "tr")
    assert en_store.pack_path != tr_store.pack_path and en_store.pack_path.is_file() and tr_store.pack_path.is_file()
    assert en_store.pack_path.read_bytes() == english_bytes                           # the first language's material was not touched
    assert en_store.load_pack().language == "en" and tr_store.load_pack().language == "tr" and tr_store.load_pack().generation["language"] == "tr"
    assert english.language == "en"
    assert sorted(StudyStore.languages_on_disk(folder)) == ["en", "tr"]


def test_the_diagnostics_name_the_engine_only_for_advanced_use(lecture):
    services, folder = lecture
    make_pack(services, folder)
    lines = services.study.diagnostics(folder)
    assert any(line.startswith("Engine: MELETRA Automatic Study Engine") for line in lines) and any("Flashcards:" in line and "kept" in line for line in lines)
