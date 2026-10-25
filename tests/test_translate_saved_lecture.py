"""BUG-08: translating a finished lecture works on its SAVED TRANSCRIPT as text - no recording, no speech recogniser, no new job, no file dialog - and the results are kept per lecture,
per version of the transcript, per language and per engine."""

from __future__ import annotations

import json
import subprocess

import pytest

from src.services.job_service import JobStore
from src.services.lecture_service import LectureNotFound, LectureService
from src.translation.service import TranslationService, TranslationUnavailableError
from tests.archive_support import finished_record, write_lecture


class Engine:
    def __init__(self, engine_id="fake:1"):
        self.id = engine_id
        self.calls = 0

    def translate(self, texts, source, target):
        self.calls += 1
        return [f"[{target}] {t}" for t in texts]


@pytest.fixture
def world(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("an outside program (ffmpeg / the recogniser) was started to translate a saved transcript")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    store = JobStore(tmp_path / "jobs")
    root = tmp_path / "Medical Transcription"
    folder = write_lecture(root, "IMG_0001")
    record = finished_record(store, root, folder)
    assert not (tmp_path / "IMG_0001.MP4").exists()                                  # the recording is gone: it must not be needed
    service = LectureService(store)
    return store, service, folder, record


def translator(engine):
    return TranslationService([], engine=engine)


def test_a_saved_lecture_is_translated_from_its_text_alone(world):
    store, service, folder, record = world
    engine = Engine()
    before = [r.job_id for r in store.list()]
    doc = service.translate_saved_lecture(record.job_id, "tr", translator=translator(engine))
    assert doc.complete and engine.calls >= 1 and doc.segments[0].text.startswith("[tr] ")
    assert (folder / "translations" / "tr.json").is_file()
    assert [r.job_id for r in store.list()] == before and store.unfinished() == []   # no job was created
    assert store.get(record.job_id).translations == ["tr"]                           # the lecture card shows the language


def test_asking_again_opens_the_stored_translation_without_translating(world):
    store, service, folder, record = world
    engine = Engine()
    first = service.translate_saved_lecture(record.job_id, "tr", translator=translator(engine))
    calls = engine.calls
    second = service.translate_saved_lecture(record.job_id, "tr", translator=translator(engine))
    assert engine.calls == calls and second.source_revision == first.source_revision
    service.translate_saved_lecture(record.job_id, "tr", translator=translator(engine), force=True)
    assert engine.calls > calls                                                       # only an explicit request translates again


def test_several_languages_live_side_by_side(world):
    store, service, folder, record = world
    engine = Engine()
    for code in ("tr", "de", "fr"):
        service.translate_saved_lecture(record.job_id, code, translator=translator(engine))
    assert sorted(service.saved_translations(record.job_id)) == ["de", "fr", "tr"]
    assert sorted(store.get(record.job_id).translations) == ["de", "fr", "tr"]
    assert all((folder / "translations" / f"{c}.json").is_file() for c in ("tr", "de", "fr"))


def test_a_changed_transcript_makes_the_old_translation_stale(world):
    store, service, folder, record = world
    engine = Engine()
    first = service.translate_saved_lecture(record.job_id, "tr", translator=translator(engine))
    data = json.loads((folder / "local_transcript.json").read_text(encoding="utf-8"))
    (folder / "source_transcript.json").unlink(missing_ok=True)
    data["segments"][0]["text"] = "A corrected first sentence about glycolysis in the cytoplasm."
    (folder / "local_transcript.json").write_text(json.dumps(data), encoding="utf-8")
    calls = engine.calls
    second = service.translate_saved_lecture(record.job_id, "tr", translator=translator(engine))
    assert engine.calls > calls and second.source_revision != first.source_revision      # translated again, from the new text


def test_another_engine_does_not_pass_off_its_translation_as_the_current_ones(world):
    store, service, folder, record = world
    service.translate_saved_lecture(record.job_id, "tr", translator=translator(Engine("fake:1")))
    other = Engine("fake:2")
    service.translate_saved_lecture(record.job_id, "tr", translator=translator(other), force=True)
    stored = json.loads((folder / "translations" / "tr.json").read_text(encoding="utf-8"))
    assert stored["engine"] == "fake:2" or stored.get("engine_id") == "fake:2" or "fake:2" in json.dumps(stored)


def test_a_lecture_with_only_text_files_can_be_translated(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("outside program")))
    store = JobStore(tmp_path / "jobs")
    root = tmp_path / "Medical Transcription"
    folder = write_lecture(root, "IMG_0002", source="IMG_0002.MP4", job_id="job0000000000002", files="legacy")
    record = finished_record(store, root, folder, source="IMG_0002.MP4", job_id="job0000000000002")
    engine = Engine()
    doc = LectureService(store).translate_saved_lecture(record.job_id, "tr", translator=translator(engine))
    assert doc.segments and engine.calls >= 1 and (folder / "translations" / "tr.json").is_file()
    assert not (folder / "local_transcript.json").exists()                            # nothing else was invented in the lecture folder


def test_unknown_lectures_and_languages_are_reported_not_guessed(world):
    store, service, folder, record = world
    with pytest.raises(LectureNotFound):
        service.translate_saved_lecture("nope", "tr", translator=translator(Engine()))
    with pytest.raises(TranslationUnavailableError):
        service.translate_saved_lecture(record.job_id, "xx", translator=translator(Engine()))


def test_a_lecture_without_any_transcript_says_so(tmp_path):
    store = JobStore(tmp_path / "jobs")
    root = tmp_path / "Medical Transcription"
    folder = write_lecture(root, "IMG_0003", job_id="job0000000000003", files="empty")
    record = finished_record(store, root, folder, job_id="job0000000000003")
    with pytest.raises(LectureNotFound) as raised:                                    # a finished lecture whose transcript files are gone is not a lecture that can be translated
        LectureService(store).translate_saved_lecture(record.job_id, "tr", translator=translator(Engine()))
    assert "No saved lecture" in str(raised.value) and not (folder / "translations").exists()


def test_the_original_transcript_is_never_changed_by_translating(world):
    store, service, folder, record = world
    from tests.archive_support import tree_hashes

    before = {k: v for k, v in tree_hashes(folder).items() if not k.startswith(("translations", "translated_", "source_transcript"))}
    service.translate_saved_lecture(record.job_id, "tr", translator=translator(Engine()))
    after = {k: v for k, v in tree_hashes(folder).items() if not k.startswith(("translations", "translated_", "source_transcript"))}
    assert before == after


def test_the_window_uses_the_same_service(qapp, tmp_path):
    from tests.gui_support import make_services

    services = make_services(tmp_path, opened=[])
    assert isinstance(services.lectures, LectureService) and services.lectures.jobs is services.jobs
