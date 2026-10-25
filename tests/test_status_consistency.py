"""Phase 19: a lecture has ONE status and every screen says the same thing - Home's Recent lectures, the Files list, the result screen's Technical details and the Study screen.
Completed is never shown over missing output; what had to be recovered from the files says so; what is missing says that."""

from __future__ import annotations

import json

import pytest

from src.archive.resolver import quick_status
from src.gui.lecture_cards import LectureCard, status_state, status_text
from src.gui.main_window import MainWindow
from src.gui.results_page import summarize
from src.services.job_service import JobRecord
from tests.archive_support import finished_record, write_lecture
from tests.gui_support import ScriptedLauncher, make_services, settle

SCENARIOS = {
    "completed": "Completed",
    "recovered": "Completed - metadata recovered",
    "missing": "Transcript not found",
    "no_folder": "Transcript not found",
}


def build(services, tmp_path, scenario):
    root = tmp_path / "Medical Transcription"
    folder = write_lecture(root, "IMG_0001")
    record = finished_record(services.jobs, root, folder)
    if scenario == "recovered":
        services.jobs.update(record.job_id, summary={"output_dir": str(folder)}, duration_seconds=0.0)                       # the job never recorded its numbers
        meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        (folder / "metadata.json").write_text(json.dumps({k: v for k, v in meta.items() if k != "quality"}), encoding="utf-8")
        (folder / ".job_id").unlink()
        for name in ("local_transcript.json", "local_raw_transcript.txt", "local_clean_transcript.txt"):
            (folder / name).unlink()
        (folder / "IMG_0001_clean.txt").write_text("Glycolysis converts glucose to pyruvate in the cytoplasm of every cell of the body.", encoding="utf-8")
        services.jobs.update(record.job_id, summary={})
    elif scenario == "missing":
        for path in folder.iterdir():
            if path.is_file() and path.name not in (".job_id", "metadata.json"):
                path.unlink()
    elif scenario == "no_folder":
        import shutil

        shutil.rmtree(folder)
    return services.jobs.get(record.job_id), folder


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_every_screen_says_the_same_status(qapp, tmp_path, scenario):
    services = make_services(tmp_path, opened=[])
    record, folder = build(services, tmp_path, scenario)
    wanted = SCENARIOS[scenario]
    status, label = quick_status(record)
    assert label == wanted
    window = MainWindow(services, ScriptedLauncher())
    window.files_page.refresh()
    window.simple_home.refresh()
    files_cards = [c for c in window.files_page.cards if c.record.job_id == record.job_id]
    home_cards = [c for c in window.simple_home.findChildren(LectureCard) if c.record.job_id == record.job_id]
    assert files_cards and home_cards
    assert files_cards[0].status.text() == wanted and home_cards[0].status.text() == wanted == status_text(record)
    assert wanted.lower() in files_cards[0].accessibleName().lower()                         # screen readers hear the same thing
    if scenario in ("completed", "recovered"):
        window.open_lecture_record(record)                                                    # the click on the card: the result screen and its Technical details
        assert window.results.values["Status"].text() == wanted                               # the same words as the card
    else:
        assert summarize(folder, None)["Status"] == "Transcript not found" if folder.exists() else True
    settle(window, window.home, window.study)
    window.deleteLater()


def test_a_missing_transcript_is_not_painted_green(qapp, tmp_path):
    services = make_services(tmp_path, opened=[])
    ok, _ = build(services, tmp_path / "a", "completed")
    gone, _ = build(services, tmp_path / "b", "missing") if False else (None, None)
    services_b = make_services(tmp_path / "second", opened=[])
    missing, _ = build(services_b, tmp_path / "b", "missing")
    assert status_state(ok) == "completed" and status_state(missing) == "paused"             # the amber 'needs attention' chip, not the green one


def test_other_states_keep_their_own_words(qapp, tmp_path):
    for state, text in (("queued", "Queued"), ("failed", "Failed"), ("running", "Processing")):
        record = JobRecord(job_id="x" + state, source="a.mp4", source_name="a.mp4", output_dir=str(tmp_path), state=state)
        assert status_text(record).startswith(text) and status_state(record) == state
    interrupted = JobRecord(job_id="xi", source="a.mp4", source_name="a.mp4", output_dir=str(tmp_path), state="paused", last_error="Interrupted: the program was closed")
    assert status_text(interrupted) == "Interrupted"


def test_opening_a_lecture_without_its_transcript_explains_instead_of_showing_an_empty_screen(qapp, tmp_path):
    services = make_services(tmp_path, opened=[])
    record, folder = build(services, tmp_path, "missing")
    window = MainWindow(services, ScriptedLauncher())
    shown = []
    window.toast.show_message = shown.append
    window.open_lecture_record(services.jobs.get(record.job_id))
    assert window.current_output is None and shown and "were not found" in shown[0] and "Repair lecture archive" in shown[0]           # the card said "Transcript not found"; the click says why
    settle(window, window.home, window.study)
    window.deleteLater()
