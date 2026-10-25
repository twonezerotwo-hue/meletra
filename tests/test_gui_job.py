"""Running a transcription from the window: controller, active page, results page, pause/resume, close, crash recovery.

The worker is either scripted (exact events) or the REAL worker code with a fake speech model (``ThreadLauncher``), so the
whole GUI -> request file -> worker -> event file -> GUI path is exercised without a GPU."""

import json
import os
import time
import types
from pathlib import Path

import pytest

from src.gui.active_page import ActivePage
from src.gui.job_controller import JobController, ThreadLauncher
from src.gui.main_window import PAGE_ACTIVE, PAGE_HOME, PAGE_RESULTS, MainWindow
from src.gui.results_page import OUTPUT_BUTTONS, ResultsPage, summarize
from src.gui.widgets import format_clock, format_hms, format_size
from src.model_manager import ModelSpec
from src.pipeline import job_key
from src.services import transcription_service as tsvc
from src.services.job_service import JobStore
from src.services.model_service import ModelService
from src.services.transcription_service import EXIT_COMPLETED, EXIT_FAILED, EXIT_PAUSED, JobRequest, stop_path
from tests.conftest import write_silent_wav
from tests.gui_support import (
    ScriptedLauncher, make_services, progress_event, settle, wait_for,
)
from tests.test_expert_pipeline import LectureFake


def request_for(source, **kw) -> JobRequest:
    return JobRequest(source=str(source), output_dir=str(source.parent / "out"), model_key="whisper-large-v3", language="en", **kw)


@pytest.fixture
def lecture(tmp_path):
    return write_silent_wav(tmp_path / "lecture.wav", seconds=120)


@pytest.fixture
def controller(qapp, tmp_path):
    services = make_services(tmp_path)
    launcher = ScriptedLauncher()
    c = JobController(services, launcher, poll_ms=20)
    c.test_launcher = launcher
    yield c
    c._timer.stop()


def collect(controller):
    seen = {k: [] for k in ("status", "progress", "log", "warning", "completed", "paused", "failed", "info", "running")}
    controller.status.connect(seen["status"].append)
    controller.progress.connect(seen["progress"].append)
    controller.log_line.connect(lambda level, text: seen["log"].append((level, text)))
    controller.warning.connect(seen["warning"].append)
    controller.completed.connect(seen["completed"].append)
    controller.paused.connect(seen["paused"].append)
    controller.failed.connect(seen["failed"].append)
    controller.info.connect(seen["info"].append)
    controller.running_changed.connect(seen["running"].append)
    return seen


# ----------------------------------------------------------------------------------------------- controller
def test_start_writes_the_request_and_registers_the_job(controller, lecture):
    job_id = controller.start(request_for(lecture))
    assert job_id == job_key(lecture)
    paths = controller.services.paths
    saved = JobRequest.load(paths.jobs / f"{job_id}.request.json")
    assert saved.job_id == job_id and saved.data_root == str(paths.root) and saved.parent_pid == os.getpid()
    record = controller.services.jobs.get(job_id)
    assert record.state == "queued" and record.source_name == "lecture.wav"


def test_events_become_signals(controller, lecture):
    seen = collect(controller)
    controller.test_launcher.script = [
        {"type": "job_started", "device": "cuda"}, {"type": "model_loading", "model": "C:/m/large-v3", "device": "cuda", "compute_type": "float16"},
        {"type": "language_detected", "language": "en"}, {"type": "subject_detected", "areas": ["biochemistry", "physiology"]},
        progress_event(25.0), {"type": "warning", "message": "GPU memory is low"}]
    controller.start(request_for(lecture))
    controller.poll()
    assert seen["progress"][0]["percent"] == 25.0 and seen["warning"] == ["GPU memory is low"]
    assert controller.facts["device"] == "cuda" and controller.facts["compute_type"] == "float16"
    assert controller.facts["language"] == "en" and controller.facts["subject"] == "biochemistry, physiology"
    assert any("Loading large-v3 on CUDA" in s for s in seen["status"])
    assert ("WARNING", "GPU memory is low") in seen["log"]


def test_completed_event_ends_the_job(controller, lecture):
    seen = collect(controller)
    controller.test_launcher.script = [{"type": "completed", "output_dir": "X:/out/lecture", "words": 10}]
    controller.start(request_for(lecture))
    controller.test_launcher.finish(EXIT_COMPLETED)
    controller.poll()
    assert seen["completed"][0]["output_dir"] == "X:/out/lecture" and not seen["failed"]
    assert seen["running"] == [True, False] and not controller.is_running


def test_worker_dying_without_a_word_is_reported(controller, lecture):
    seen = collect(controller)
    controller.start(request_for(lecture))
    controller.test_launcher.finish(-1073741819)
    controller.poll()
    assert len(seen["failed"]) == 1 and "ended unexpectedly" in seen["failed"][0] and "-1073741819" in seen["failed"][0]


def test_exit_code_paused_without_event_still_pauses(controller, lecture):
    seen = collect(controller)
    controller.start(request_for(lecture))
    controller.test_launcher.finish(EXIT_PAUSED)
    controller.poll()
    assert len(seen["paused"]) == 1 and not seen["failed"]


def test_exit_zero_without_completion_event_is_a_failure(controller, lecture):
    seen = collect(controller)
    controller.start(request_for(lecture))
    controller.test_launcher.finish(EXIT_COMPLETED)
    controller.poll()
    assert "without a result" in seen["failed"][0]


def test_failed_event_carries_the_friendly_message_once(controller, lecture):
    seen = collect(controller)
    controller.test_launcher.script = [{"type": "failed", "error": "A CUDA runtime library is missing."}]
    controller.start(request_for(lecture))
    controller.test_launcher.finish(EXIT_FAILED)
    controller.poll()
    assert seen["failed"] == ["A CUDA runtime library is missing."]


def test_second_start_while_running_is_refused(controller, lecture):
    controller.start(request_for(lecture))
    with pytest.raises(RuntimeError, match="already running"):
        controller.start(request_for(lecture))


def test_pause_writes_the_stop_file_once(controller, lecture):
    seen = collect(controller)
    job_id = controller.start(request_for(lecture))
    controller.request_pause()
    controller.request_pause()
    assert stop_path(controller.services.paths, job_id).exists()
    assert sum("Pausing safely" in s for s in seen["status"]) == 1


def test_start_clears_a_stale_stop_file(controller, lecture):
    job_id = job_key(lecture)
    stale = stop_path(controller.services.paths, job_id)
    stale.write_text("x")
    controller.start(request_for(lecture))
    assert not stale.exists()


def test_chunk_and_gap_events_are_explained_in_plain_words(controller, lecture):
    seen = collect(controller)
    controller.test_launcher.script = [
        {"type": "resume_started", "completed": 2, "chunks": 5}, {"type": "chunk_started", "index": 3, "chunks": 5},
        {"type": "checkpoint_saved"}, {"type": "gap_found", "seconds": 42.0, "index": 1, "gaps": 2},
        {"type": "gap_repair_finished", "repaired": True}, {"type": "loop_repaired", "repaired": False}]
    controller.start(request_for(lecture))
    controller.poll()
    text = " | ".join(s for _l, s in seen["log"])
    assert "2 of 5 chunks already finished" in text and "Checkpoint saved" in text and "42 s of speech" in text
    assert "Recovered the missing speech" in text and "could not be repaired" in text


# ----------------------------------------------------------------------------------------------- active page
@pytest.fixture
def active(qapp, tmp_path):
    page = ActivePage(make_services(tmp_path))
    yield page
    page.deleteLater()


def test_active_page_shows_every_required_number(active):
    active.begin("lecture.wav", {"model": "Whisper Large-v3", "language": "auto", "subject": "auto"}, "RTX 2070", 8192)
    active.set_progress(progress_event(40.0, chunk=2, chunks=5))
    stats = {name: cell.value.text() for name, cell in active.stats.cells.items()}
    assert active.progress_bar.value() == 400 and active.percent_label.text() == "40%"
    assert stats["Chunk"] == "2 / 5" and stats["Processed"] == "06:40 / 16:40" and stats["Elapsed"] == "00:30"
    assert stats["ETA"] == "01:00" and stats["RTF"] == "0.25" and stats["GPU"] == "RTX 2070"
    assert stats["VRAM"].startswith("4.2 / 8.0 GB") and stats["Model"] == "Whisper Large-v3"


def test_eta_says_calculating_until_the_speed_is_known(active):
    active.begin("x.wav", {}, "", 0)
    active.set_progress({"percent": 1.0, "chunk": 1, "chunks": 3, "processed": 5, "total": 500, "elapsed": 2, "rtf": 0.0, "eta": None})
    assert active.stats.cells["ETA"].value.text() == "calculating ..." and active.stats.cells["RTF"].value.text() == "-"
    assert active.stats.cells["VRAM"].value.text() == "n/a (CPU)" and active.stats.cells["GPU"].value.text() == "none (CPU)"


def test_shared_vram_is_only_mentioned_when_significant(active):
    active.begin("x.wav", {}, "RTX", 8192)
    active.set_progress(progress_event(10.0, gpu_shared_mib=2048))
    assert "+2.0 GB shared" in active.stats.cells["VRAM"].value.text()


def test_facts_update_language_subject_and_device(active):
    active.begin("x.wav", {}, "", 0)
    active.update_facts({"language": "en", "subject": "anatomy", "model": "M", "device": "cuda", "compute_type": "float16"})
    assert active.stats.cells["Device"].value.text() == "CUDA / float16" and active.stats.cells["Language"].value.text() == "en"


def test_pause_button_disables_itself_and_states_the_state(active):
    sent = []
    active.pause_requested.connect(lambda: sent.append(1))
    active.begin("x.wav", {}, "", 0)
    active.pause_button.click()
    assert sent == [1] and not active.pause_button.isEnabled() and active.pause_button.text() == "PAUSING ..."
    active.finished_paused()
    assert active.pause_button.isHidden() and not active.resume_button.isHidden() and "Paused safely" in active.status_label.text()


def test_resume_stays_disabled_until_the_worker_has_really_ended(active):
    active.begin("x.wav", {}, "", 0)
    active.finished_paused()
    assert not active.resume_button.isHidden() and not active.resume_button.isEnabled()
    active.begin("x.wav", {}, "", 0)
    assert active.resume_button.isHidden() and active.resume_button.isEnabled()


def test_failure_offers_resume_and_shows_the_reason(active):
    active.begin("x.wav", {}, "", 0)
    active.finished_failed("The GPU ran out of memory.")
    assert not active.resume_button.isHidden() and "ran out of memory" in active.warning_label.text()


def test_log_panel_keeps_a_bounded_history_and_marks_warnings(active):
    for i in range(5000):
        active.log.add("INFO", f"line {i}")
    active.log.add("WARNING", "careful")
    text = active.log.toPlainText()
    assert active.log.blockCount() <= 4001 and text.rstrip().endswith("! careful")


def test_open_log_and_back_buttons_emit(active):
    got = []
    active.open_log_requested.connect(lambda: got.append("log"))
    active.back_requested.connect(lambda: got.append("back"))
    active.log_button.click()
    active.back_button.click()
    assert got == ["log", "back"]


# ----------------------------------------------------------------------------------------------- formatting
def test_time_and_size_formats():
    assert format_clock(75) == "01:15" and format_clock(3725) == "01:02:05" and format_clock(None) == "--:--" and format_clock(-1) == "--:--"
    assert format_hms(5220) == "01:27:00" and format_hms(None) == "--:--:--"
    assert format_size(512) == "512 B" and format_size(2048) == "2.0 KB" and format_size(3 * 1024 ** 3) == "3.0 GB"


# ----------------------------------------------------------------------------------------------- results page
METADATA = {
    "quality": {"words": 12345}, "timing": {"audio_duration_total": 5220.0, "wall_time_cumulative": 1331.0, "real_time_factor_effective": 0.25},
    "medical_terms": {"occurrences": 321, "distinct_detected": 40}, "subject": {"subjects": ["biochemistry"]},
    "language": {"detected": "en"}, "uncertainty": {"uncertain_segments": 7},
    "long_audio": {"gap_repairs": [{"inserted": True, "seconds": 18.0}, {"inserted": False, "seconds": 3}]},
}


def test_summary_reads_the_numbers_from_metadata(tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps(METADATA), encoding="utf-8")
    (tmp_path / "local_clean_transcript.txt").write_text("Glycolysis converts glucose to pyruvate in the cytoplasm of every cell of the body.", encoding="utf-8")
    s = summarize(tmp_path)
    assert s["Duration"] == "01:27:00" and s["Words"] == "12,345" and s["Medical terms"] == "321 (40 distinct)"
    assert s["Uncertain segments"] == "7" and s["Recovered gaps"] == "1 (18 s of speech)"
    assert s["Processing time"] == "22:11" and s["Real-time factor"] == "0.25" and s["Language"] == "English"
    assert s["Subject"] == "Biochemistry" and s["Status"] == "Completed"


def test_summary_of_a_folder_without_metadata_does_not_crash(tmp_path):
    """BUG-04: a folder that says nothing shows "Unknown", never an invented 0 words / 00:00:00 next to a status of Completed."""
    s = summarize(tmp_path)
    assert s["Words"] == "Unknown" and s["Duration"] == "Unknown" and s["Real-time factor"] == "-" and s["Status"] == "Transcript not found"


def test_summary_with_a_transcript_but_no_metadata_recovers_what_the_files_say(tmp_path):
    (tmp_path / "IMG_0001_clean.txt").write_text("Glycolysis converts glucose to pyruvate in the cytoplasm of every cell of the body.", encoding="utf-8")
    s = summarize(tmp_path)
    assert s["Status"] == "Completed - metadata recovered" and s["Words"] != "Unknown" and s["Duration"] == "Unknown"


def test_results_buttons_follow_the_files_that_exist(qapp, tmp_path):
    opened = []
    page = ResultsPage(make_services(tmp_path, opened=opened))
    out = tmp_path / "result"
    out.mkdir()
    (out / "metadata.json").write_text(json.dumps(METADATA), encoding="utf-8")
    for name in ("local_clean_transcript.txt", "local_transcript.srt", "quality_report.json"):
        (out / name).write_text("x", encoding="utf-8")
    page.show_results(out, "lecture.mp4")
    enabled = {f for f, b in page.buttons.items() if b.isEnabled()}
    assert enabled == {"local_clean_transcript.txt", "local_transcript.srt", "quality_report.json"}
    shown: list[str] = []
    page.view_requested.connect(shown.append)
    page.buttons["local_transcript.srt"].click()
    assert shown == [str(out / "local_transcript.srt")] and opened == []                 # BUG-05: a transcript file is read inside MELETRA, no outside program is started
    page.folder_button.click()
    assert opened == [out]                                                               # only the secondary "Show files in Explorer" leaves the program
    assert page.values["Words"].text() == "12,345" and "lecture.mp4" in page.file_label.text()


def test_results_offers_every_required_output():
    labels = [t for t, _f in OUTPUT_BUTTONS]
    for needed in ("Open Clean Transcript", "Open Raw Transcript", "Open Timestamped Transcript", "Open SRT", "Open VTT", "Open JSON",
                   "Medical Terms", "Possible Medical Errors", "Quality Report"):
        assert needed in labels


def test_results_new_button_emits(qapp, tmp_path):
    page = ResultsPage(make_services(tmp_path))
    got = []
    page.new_requested.connect(lambda: got.append(1))
    page.new_button.click()
    assert got == [1]


# ----------------------------------------------------------------------------------------------- the whole window
@pytest.fixture
def window_env(qapp, tmp_path, monkeypatch):
    """A MainWindow whose jobs run the real worker code in a thread, with a fake speech model."""
    opened = []
    services = make_services(tmp_path, opened=opened)
    monkeypatch.setattr(ModelService, "model_spec", lambda self, key: ModelSpec(key, "faster_whisper", "large-v3"))
    monkeypatch.setattr(tsvc, "STOP_POLL_SECONDS", 0.0)
    monkeypatch.setattr("src.transcriber.create_adapter", lambda *a, **k: LectureFake())
    LectureFake.requests = []
    window = MainWindow(services, ThreadLauncher())
    window.controller._timer.setInterval(20)
    source = write_silent_wav(tmp_path / "lecture.wav", seconds=120)
    env = types.SimpleNamespace(window=window, services=services, opened=opened, source=source, tmp=tmp_path)
    yield env
    window.controller._timer.stop()
    settle(window, window.home)
    window.deleteLater()


def start_from_window(env, chunk_minutes=1.0):
    env.services.update_setting(chunk_minutes=chunk_minutes, output_dir=str(env.tmp / "out"))
    w = env.window
    w.select_file(str(env.source))
    wait_for(lambda: w.home.start_button.isEnabled(), what="start button")
    w.home.language_combo.setCurrentIndex(w.home.language_combo.findData("en"))
    w.home.start_button.click()


@pytest.mark.usefixtures("fake_ffmpeg")
def test_full_run_from_the_window_ends_on_the_results_page(window_env):
    env = window_env
    w = env.window
    assert w.stack.currentIndex() == PAGE_HOME
    start_from_window(env)
    assert w.stack.currentIndex() == PAGE_ACTIVE
    wait_for(lambda: w.stack.currentIndex() == PAGE_RESULTS, timeout=60, what="the results page")
    out = env.tmp / "out" / "lecture"
    assert w.results.output_dir == out and w.results.values["Status"].text() == "Completed"
    assert int(w.results.values["Words"].text().replace(",", "")) > 0
    assert all(w.results.buttons[f].isEnabled() for f in ("local_clean_transcript.txt", "local_raw_transcript.txt",
                                                            "local_transcript_with_timestamps.txt", "local_transcript.srt",
                                                            "local_transcript.vtt", "local_transcript.json"))
    assert env.opened == []                                      # reading happens inside MELETRA: Explorer opens only when the student asked for it
    assert env.services.jobs.get(job_key(env.source)).state == "completed"
    assert "Completed" in w.active.status_label.text() or w.active.progress_bar.value() > 0


@pytest.mark.usefixtures("fake_ffmpeg")
def test_output_folder_is_opened_only_when_the_student_asked_for_it(window_env):
    env = window_env
    env.services.update_setting(open_folder_after=True)
    start_from_window(env)
    wait_for(lambda: env.window.stack.currentIndex() == PAGE_RESULTS, timeout=60, what="results")
    assert env.opened == [env.tmp / "out" / "lecture"]


@pytest.mark.usefixtures("fake_ffmpeg")
def test_pause_then_resume_from_the_window(window_env, monkeypatch):
    env = window_env
    w = env.window
    seen = {"n": 0}

    class PausingFake(LectureFake):
        def transcribe(self, request):
            real = request.on_segment

            def on_segment(seg):
                real(seg)
                seen["n"] += 1
                if seen["n"] == 3:
                    w.active.pause_button.click()                 # the user presses PAUSE SAFELY while the job runs
                    time.sleep(0.3)                               # ... and the GUI thread needs a moment to pass the request on: the
                    #                                               fake model is far faster than any real one (without this wait the
                    #                                               whole fake job can finish first - the test failed on its own at 0.1.1)

            if real is not None:
                request.on_segment = on_segment
            return super().transcribe(request)

    monkeypatch.setattr("src.transcriber.create_adapter", lambda *a, **k: PausingFake())
    start_from_window(env)
    wait_for(lambda: not w.active.resume_button.isHidden() and not w.controller.is_running, timeout=60, what="the paused state")
    assert w.stack.currentIndex() == PAGE_ACTIVE and "Paused safely" in w.active.status_label.text()
    record = env.services.jobs.get(job_key(env.source))
    assert record.state == "paused" and (env.services.paths.checkpoints / record.job_id).exists()

    monkeypatch.setattr("src.transcriber.create_adapter", lambda *a, **k: LectureFake())
    wait_for(lambda: w.active.resume_button.isEnabled(), what="RESUME to be enabled once the worker has ended")
    w.active.resume_button.click()
    wait_for(lambda: w.stack.currentIndex() == PAGE_RESULTS, timeout=60, what="the results after resuming")
    assert env.services.jobs.get(job_key(env.source)).state == "completed"


@pytest.mark.usefixtures("fake_ffmpeg")
def test_a_failing_job_shows_the_reason_and_can_be_retried(window_env, monkeypatch):
    env = window_env
    w = env.window

    def broken(*a, **k):
        raise RuntimeError("CUDA failed with error out of memory")

    monkeypatch.setattr("src.transcriber.create_adapter", broken)
    start_from_window(env)
    wait_for(lambda: not w.active.resume_button.isHidden() and not w.controller.is_running, timeout=60, what="the failure to show")
    assert "GPU ran out of memory" in w.active.warning_label.text() and w.stack.currentIndex() == PAGE_ACTIVE
    w.active.back_button.click()
    assert w.stack.currentIndex() == PAGE_HOME and env.services.jobs.get(job_key(env.source)).state == "failed"


def test_closing_while_running_asks_and_can_be_cancelled(window_env):
    env = window_env
    w = env.window
    launcher = ScriptedLauncher()
    w.controller.launcher = launcher
    w.controller.start(request_for(env.source))
    asked = []
    w.confirm_close = lambda: (asked.append(1), False)[1]
    assert w.close() is False and asked == [1] and w.controller.is_running and not w._closing


def test_closing_after_yes_pauses_safely_and_then_closes(window_env):
    env = window_env
    w = env.window
    launcher = ScriptedLauncher()
    w.controller.launcher = launcher
    job_id = w.controller.start(request_for(env.source))
    w.confirm_close = lambda: True
    w.show()
    w.close()
    assert w._closing and stop_path(env.services.paths, job_id).exists() and w.isVisible()
    launcher.finish(EXIT_PAUSED)
    wait_for(lambda: not w.isVisible(), what="the window to close after the worker paused")


def test_force_close_ends_a_worker_that_does_not_pause(window_env):
    env = window_env
    w = env.window
    launcher = ScriptedLauncher()
    w.controller.launcher = launcher
    w.controller.start(request_for(env.source))
    w._closing = True
    w._force_close()
    assert launcher.handle.killed
    wait_for(lambda: not w.controller.is_running, what="the controller to notice")


def test_unfinished_jobs_of_a_crashed_run_are_marked_paused_at_startup(qapp, tmp_path):
    services = make_services(tmp_path)
    source = write_silent_wav(tmp_path / "crashed.wav", 3)
    services.jobs.create_or_get(job_key(source), source, state="running", pid=2_000_000_000, progress=33.0)
    window = MainWindow(services, ScriptedLauncher())
    assert services.jobs.get(job_key(source)).state == "paused" and "stopped unexpectedly" in window.statusBar().currentMessage()
    settle(window, window.home)


def test_dropping_a_file_on_the_window_selects_it(window_env):
    from tests.gui_support import drop_event

    env = window_env
    env.window.dropEvent(drop_event(str(env.source)))
    wait_for(lambda: not env.window.simple_home.classifying, what="the dropped file to be read")      # in a background thread since 3.0.1
    assert env.window.home.source == env.source
    wait_for(lambda: env.window.home.duration is not None, what="probe")


def test_dropping_text_on_the_window_is_refused(window_env):
    from tests.gui_support import drop_event

    notes = env_file = window_env.tmp / "n.xyz"
    notes.write_text("x")
    window_env.window.dropEvent(drop_event(str(env_file)))
    wait_for(lambda: not window_env.window.simple_home.classifying, what="the dropped file to be read")
    toast = window_env.window.toast
    assert window_env.window.home.source is None and toast.isVisibleTo(window_env.window) and "not supported" in toast.text()


def test_new_transcription_resets_the_start_page(window_env):
    env = window_env
    env.window.select_file(str(env.source))
    wait_for(lambda: env.window.home.duration is not None, what="probe")
    env.window.results.new_button.click()
    assert env.window.home.source is None and env.window.stack.currentIndex() == PAGE_HOME
    assert not env.window.home.start_button.isEnabled()


def test_open_log_opens_the_job_log_or_the_logs_folder(window_env):
    env = window_env
    env.window.open_log()
    assert env.opened[-1] == env.services.paths.logs                  # no job yet: the folder
    job_id = job_key(env.source)
    env.services.paths.job_log(job_id).write_text("log")
    env.window.controller.job_id = job_id
    env.window.open_log()
    assert env.opened[-1] == env.services.paths.job_log(job_id)


def test_completed_event_without_an_output_folder_is_not_shown_as_success(window_env):
    env = window_env
    env.window.on_completed({"type": "completed"})
    assert env.window.stack.currentIndex() != PAGE_RESULTS and "no output folder" in env.window.active.warning_label.text()


def test_the_window_geometry_is_remembered(window_env):
    env = window_env
    env.window.resize(1000, 700)
    env.window.close()
    assert env.services.settings.settings.window_geometry
