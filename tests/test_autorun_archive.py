"""The diagnostic hook the installed-program acceptance uses: --autorun-archive opens ONE lecture from the archive of the real window and reports what a person would see - without any text of
the lecture in the report."""

from __future__ import annotations

import json

import pytest

from src.gui.autorun_archive import AutoRunArchive
from src.gui.main_window import MainWindow
from src.services.job_service import JobRecord
from tests.gui_support import ScriptedLauncher, make_services, settle, wait_for
from tests.study_support import write_lecture_folder


@pytest.fixture
def env(qapp, tmp_path):
    services = make_services(tmp_path, opened=[])
    root = tmp_path / "Medical Transcription"
    folder = write_lecture_folder(root / "FIXTURE_0005")
    (folder / ".job_id").write_text("0000000000000001", encoding="utf-8")
    record = JobRecord(job_id="0000000000000001", source=str(tmp_path / "FIXTURE_0005.MP4"), source_name="FIXTURE_0005.MP4", output_dir=str(root), state="completed", progress=100.0,
                       duration_seconds=2548.0, summary={"output_dir": str(folder), "words": 4971, "audio_seconds": 2548.0, "language": "en", "segments": 171})
    services.jobs.save(record)
    (services.paths.jobs / "0000000000000001.request.json").write_text(json.dumps({"source": record.source, "output_dir": str(root)}), encoding="utf-8")
    window = MainWindow(services, ScriptedLauncher())
    window.controller._timer.setInterval(20)
    yield window, services, folder, tmp_path
    window.controller._timer.stop()
    settle(window, window.home, window.study)
    window.deleteLater()


def run(window, tmp, recording="FIXTURE_0005.MP4", study=False, timeout=120, exercise=False):
    result = tmp / "archive_result.json"
    runner = AutoRunArchive(window, recording, result, timeout, study, exercise)
    wait_for(lambda: runner.finished, timeout=timeout + 10, what="the archive report")
    runner.timer.stop()
    return json.loads(result.read_text(encoding="utf-8")), runner


def test_the_report_describes_what_a_person_sees_when_opening_the_lecture(env):
    window, services, folder, tmp = env
    data, _runner = run(window, tmp)
    assert data["exit_code"] == 0
    assert data["files_list"] == {"cards_for_this_lecture": 1, "status": "Completed", "lectures_listed": 1, "queued_ghosts": 0}
    assert data["opened"]["folder_name"] == "FIXTURE_0005" and data["opened"]["read_enabled"] and data["opened"]["study_enabled"] and data["opened"]["translate_enabled"]
    assert data["technical_details"]["Status"] == "Completed" and data["technical_details"]["Words"] == "4,971" and data["technical_details"]["Duration"] == "00:42:28"
    assert data["reader"]["shown_inside_the_window"] and data["reader"]["characters"] > 500
    assert data["external_programs_started"] == []
    assert data["privacy"]["home_chip"] == "Local processing" and data["privacy"]["sharing_active"] is False


def test_the_report_never_contains_the_text_of_the_lecture(env):
    window, services, folder, tmp = env
    run(window, tmp)
    raw = (tmp / "archive_result.json").read_text(encoding="utf-8")
    assert "Glycolysis" not in raw and "hexokinase" not in raw.lower() and "pyruvate" not in raw.lower()                 # lengths and a fingerprint only


def test_a_click_on_read_transcript_starts_no_outside_program(env):
    window, services, folder, tmp = env
    run(window, tmp)
    assert services.open_path.__name__ == "<lambda>"                                                                      # the hook records instead of starting anything


def test_with_study_the_material_is_made_from_the_saved_transcript_and_described(env):
    window, services, folder, tmp = env
    data, _runner = run(window, tmp, study=True, timeout=180)
    study = data["study"]
    assert data["exit_code"] == 0 and study["flashcards"] > 0 and study["questions"] > 0 and study["items_total"] == study["items_with_evidence"] and study["blanks"] == 0
    assert study["engine"]["engine"] == "MELETRA Automatic Study Engine" and study["label"] == "MELETRA Automatic Study Engine"
    assert (folder / "study" / "study_pack.json").is_file()


def test_the_exercise_makes_two_reviews_a_difficult_mark_and_an_unfinished_session_and_a_second_look_shows_what_was_kept(env):
    window, services, folder, tmp = env
    first, _runner = run(window, tmp, study=True, timeout=180, exercise=True)
    learning = first["study"]["learning"]
    assert first["exit_code"] == 0 and "error" not in learning and len(learning["exercised"]["cards"]) == 3
    assert learning["plan"]["reviews"] == 2 and learning["plan"]["difficult"] == 1 and learning["resumable"]["answered"] == 1
    second, _runner = run(window, tmp, study=True, timeout=60)                                       # the lecture is opened again: what the student did is read from the disk
    kept = second["study"]["learning"]
    assert "exercised" not in kept and "error" not in kept
    assert kept["plan"]["reviews"] == 2 and kept["plan"]["difficult"] == 1 and kept["plan"]["due"] == 1        # the card answered Don't know is due today
    assert kept["deck_opens_on"] == "due" and kept["deck_size"] == 1
    assert kept["resume_button_visible"] and kept["resumable"]["answered"] == 1 and kept["resumable"]["left"] >= 1
    pack = json.loads((folder / "study" / "study_pack.json").read_text(encoding="utf-8"))
    report_text = json.dumps([first, second])
    assert not any(card["front"] in report_text or card["back"] in report_text for card in pack["flashcards"])      # ids and numbers only, no text of the lecture


def test_an_unknown_recording_is_reported_not_guessed(env):
    window, services, folder, tmp = env
    data, _runner = run(window, tmp, recording="IMG_9999.MP4")
    assert data["exit_code"] == 1 and data["records"] == 0 and "IMG_9999.MP4" in data["error"]


def test_the_launcher_understands_the_flag(monkeypatch, tmp_path):
    import launcher

    seen = {}
    monkeypatch.setattr("src.gui.app.run", lambda *a, **k: seen.update(k) or 0)
    launcher.main(["MELETRA.exe", "--autorun-archive", "FIXTURE_0005.MP4", "--study", "--result", str(tmp_path / "r.json"), "--timeout", "90"])
    assert seen["autorun_archive"] == {"recording": "FIXTURE_0005.MP4", "result": str(tmp_path / "r.json"), "study": True, "exercise": False, "timeout": 90.0}
