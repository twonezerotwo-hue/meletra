"""The screens during the long silent part of a video job: a named stage with real progress (or a busy bar), never a frozen 0%; the window stays
responsive; CANCEL works in every phase."""

import shutil
import time

import pytest
from PySide6.QtCore import QTimer

from src.gui import friendly
from src.gui.job_controller import JobController
from src.gui.main_window import SIMPLE_ACTIVE, SIMPLE_HOME, MainWindow
from src.gui.simple_active import SimpleActive
from src.pipeline import job_key
from tests.gui_support import ScriptedLauncher, make_services, progress_event, settle, wait_for
from tests.test_gui_job import request_for, window_env  # noqa: F401 - the real worker with a fake speech model
from tests.test_gui_queue import env as queue_env       # noqa: F401
from tests.test_ingest_router import encode
from tests.test_video_progress import StagedFake


# ----------------------------------------------------------------------------------------------- the running screen
def test_the_running_screen_names_every_stage_and_never_shows_a_frozen_zero(qapp):
    a = SimpleActive()
    a.begin("lecture.mp4", "auto", "auto", "video")
    assert a.title.text() == "Preparing video..." and a.progress_bar.maximum() == 0 and a.percent_label.text() == "" and a.eta_label.text() == ""
    a.set_stage({"stage": "extracting", "percent": 37.0})
    assert a.title.text() == "Extracting audio..." and a.progress_bar.maximum() == 1000 and a.progress_bar.value() == 370
    assert a.percent_label.text() == "37%" and "video" in a.stage_label.text()
    a.set_stage({"stage": "speech_analysis", "percent": None})
    assert a.progress_bar.maximum() == 0 and a.percent_label.text() == ""                          # not measurable: busy, not "0%"
    a.set_stage({"stage": "language", "percent": 12.5, "done": 3, "total": 24})
    assert "3 of 24" in a.stage_label.text() and a.percent_label.text() in ("12%", "13%")
    a.set_stage({"stage": "transcribing", "percent": None})
    assert a.title.text() == "Transcribing..." and a.progress_bar.maximum() == 0
    a.set_progress(progress_event(5.0, rtf=0.0))
    assert a.progress_bar.maximum() == 1000 and a.percent_label.text() == "5%" and a.stage_label.text() == ""
    assert a.eta_label.text() == "Estimated time remaining: calculating ..."                       # speed data exists only now
    a.set_progress(progress_event(6.0, rtf=0.25, eta=600.0))
    assert a.eta_label.text() == "Estimated time remaining: 10 min"


def test_audio_files_say_audio_and_the_wording_has_no_jargon(qapp):
    a = SimpleActive()
    a.begin("lecture.mp3", "auto", "auto", "audio")
    assert a.title.text() == "Preparing audio..."
    for stage in friendly.STAGES:
        title, line = friendly.stage_texts(stage, "video")
        for word in ("cuda", "gpu", "vad", "ctranslate", "rtf", "chunk", "checkpoint", "hotword"):
            assert word not in (title + line).casefold()
    assert friendly.stage_texts("something new", "video")[0] == "Preparing video..."


def test_the_controller_turns_stage_events_into_one_signal(qapp, tmp_path):
    services = make_services(tmp_path)
    launcher = ScriptedLauncher([
        {"type": "stage", "stage": "probing", "percent": None}, {"type": "stage", "stage": "extracting", "percent": 50.0},
        {"type": "model_loading", "model": "m", "device": "cuda", "compute_type": "float16"},
        {"type": "stage", "stage": "language", "percent": 25.0, "done": 6, "total": 24},
        {"type": "chunk_started", "index": 1, "chunks": 3}, progress_event(2.0), {"type": "chunk_started", "index": 2, "chunks": 3},
        {"type": "gap_scan_started"}, {"type": "quality_audit_started"}])
    controller = JobController(services, launcher, poll_ms=20)
    stages, progress = [], []
    controller.stage.connect(stages.append)
    controller.progress.connect(progress.append)
    source = tmp_path / "lecture.mp4"
    source.write_bytes(b"x")
    controller.start(request_for(source))
    wait_for(lambda: len(stages) >= 6 and progress, what="the stage signals")
    assert [s["stage"] for s in stages] == ["probing", "extracting", "model_loading", "language", "transcribing", "finalizing"]
    assert stages[3]["done"] == 6 and progress                                     # the second chunk_started after real progress adds no busy bar
    launcher.finish(0)
    settle(controller)


# ----------------------------------------------------------------------------------------------- the whole window with a real video
@pytest.fixture
def video(tmp_path):
    return encode(tmp_path / "Fizyoloji dersi çş.mp4", "mp4", "aac", "mpeg4", seconds=60.0)


@pytest.fixture
def staged_window(window_env, monkeypatch):  # noqa: F811
    StagedFake.detections = StagedFake.progress_calls = 0
    StagedFake.on_detect = StagedFake.on_progress = None
    monkeypatch.setattr("src.transcriber.create_adapter", lambda *a, **k: StagedFake())
    window_env.services.update_setting(chunk_minutes=1.0, output_dir=str(window_env.tmp / "out"))
    return window_env


def start_video(env, video):
    w = env.window
    w.simple_home.drop_zone.file_chosen.emit(str(video))
    wait_for(lambda: w.simple_home.transcribe_button.isEnabled(), what="CONVERT TO TEXT")
    w.simple_home.transcribe_button.click()
    return w


def test_a_video_shows_its_stages_before_the_first_words(staged_window, video):
    w = start_video(staged_window, video)
    seen, titles, ranges = [], [], []
    w.controller.stage.connect(lambda e: (seen.append(e["stage"]), titles.append(w.simple_active.title.text()),
                                          ranges.append(w.simple_active.progress_bar.maximum())))
    first_words = []
    w.controller.progress.connect(lambda e: first_words.append(len(seen)) if not first_words else None)
    assert w.simple_stack.currentIndex() == SIMPLE_ACTIVE and w.simple_active.title.text() == "Preparing video..."      # at once
    wait_for(lambda: w.simple_stack.currentIndex() != SIMPLE_ACTIVE, timeout=120, what="the result")
    assert {"probing", "extracting", "language", "analysis"} <= set(seen)
    assert first_words and first_words[0] >= len(seen) - 2 and seen.index("extracting") < seen.index("language") < seen.index("analysis")
    assert "Extracting audio..." in titles and 0 in ranges and 1000 in ranges                       # a busy bar and a measured one


def test_the_window_stays_responsive_while_the_files_are_read(qapp, tmp_path, monkeypatch):
    from src.gui import simple_home as sh
    services = make_services(tmp_path)
    window = MainWindow(services)
    original = sh.classify

    def slow(paths):
        time.sleep(1.5)                                          # a slow drive / a huge file
        return original(paths)
    monkeypatch.setattr(sh, "classify", slow)
    source = encode(tmp_path / "v.mp4", "mp4", "aac", "mpeg4", seconds=5.0)
    ticks = []
    timer = QTimer()
    timer.setInterval(20)
    timer.timeout.connect(lambda: ticks.append(time.monotonic()))
    timer.start()
    window.simple_home.set_files([str(source)])
    started = time.monotonic()
    while window.simple_home.classifying and time.monotonic() - started < 10:
        qapp.processEvents()
        time.sleep(0.005)
    timer.stop()
    gaps = [b - a for a, b in zip(ticks, ticks[1:])]
    assert len(ticks) > 25 and max(gaps) < 0.6                   # the event loop kept running during the 1.5 s wait (a frozen window would show ~1.5 s)
    wait_for(lambda: window.home.source is not None, what="the recording to be selected")
    settle(window, window.home)
    window.deleteLater()


def test_cancel_during_the_language_analysis_goes_back_and_keeps_nothing_running(staged_window, video):
    import threading
    gate = threading.Event()
    StagedFake.on_detect = lambda n: (gate.set(), time.sleep(0.4))
    w = start_video(staged_window, video)
    wait_for(gate.is_set, timeout=60, what="the language analysis to be running")
    assert w.simple_active.cancel_button.isVisibleTo(w.simple_active)
    started = time.monotonic()
    w.simple_active.cancel_button.click()
    wait_for(lambda: w.simple_stack.currentIndex() == SIMPLE_HOME, timeout=60, what="the start screen")
    assert time.monotonic() - started < 20 and StagedFake.detections < 24                      # it stopped promptly, mid-analysis
    assert w.home.source is None
    wait_for(lambda: not w.controller.is_running, what="the worker process to end")
    wait_for(lambda: staged_window.services.jobs.get(job_key(video)).state == "paused", what="the job to be recorded as paused")   # nothing lost
    wait_for(lambda: not [p for p in staged_window.services.paths.temp.glob("mlt_*")], what="the temporary audio to be removed")
    assert video.is_file()                                                                      # the source is untouched


def test_cancel_ends_a_worker_that_cannot_stop_at_once(qapp, tmp_path, monkeypatch):
    """Inside something that cannot be interrupted (loading the model) the safe pause cannot happen: after the grace time the worker is
    ended and the window returns to the start screen instead of showing a failure."""
    import src.gui.main_window as mw
    monkeypatch.setattr(mw, "CANCEL_GRACE_MS", 300)
    services = make_services(tmp_path)
    launcher = ScriptedLauncher([{"type": "stage", "stage": "probing", "percent": None}])
    window = MainWindow(services, launcher)
    window.controller._timer.setInterval(20)
    source = tmp_path / "l.mp4"
    source.write_bytes(b"x")
    window.start_job(request_for(source))
    wait_for(lambda: window.controller.is_running, what="the job")
    killed = []
    monkeypatch.setattr(window.controller, "kill", lambda: (killed.append(1), launcher.finish(1)))
    window.simple_active.cancel_button.click()
    wait_for(lambda: killed, timeout=10, what="the worker to be ended")
    wait_for(lambda: window.simple_stack.currentIndex() == SIMPLE_HOME, timeout=10, what="the start screen")
    assert "stopped" not in window.simple_active.title.text().lower()
    window.controller._timer.stop()
    settle(window, window.home)
    window.deleteLater()


# ----------------------------------------------------------------------------------------------- a queue with media that needs converting first
def test_a_recording_that_needs_converting_is_prepared_off_the_window_thread_with_progress(queue_env, tmp_path):  # noqa: F811
    from src.gui.queue_runner import QueueRunner
    from src.ingest import queue as q
    from src.ingest.router import classify
    odd = tmp_path / "lecture.dat"
    shutil.copyfile(encode(tmp_path / "v.mp4", "mp4", "aac", "mpeg4", seconds=30.0), odd)
    selection = classify([odd])
    assert selection.items[0].needs_wav
    runner = QueueRunner(queue_env.services, queue_env.controller, selection.items, queue_env.template)
    queue_env.runners.append(runner)
    stages = []
    runner.progress.connect(lambda index, event: stages.append(event) if event.get("type") == "stage" else None)
    runner.start()
    assert runner._thread is not None                                       # started in a thread: this call returned at once
    wait_for(lambda: runner.queue.state in (q.DONE, q.CANCELLED), timeout=120, what="the queue")
    assert runner.items[0].state == q.COMPLETED and stages and stages[0]["stage"] == "extracting" and stages[-1]["percent"] >= 90
