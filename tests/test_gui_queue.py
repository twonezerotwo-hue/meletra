"""The batch queue with the REAL worker code (a fake speech model in a thread): sequential jobs, named outputs, failure, pause/resume,
cancel, and documents in the same queue. Nothing here re-implements transcription - the queue only starts the 0.1 worker."""

import threading
import time
import types
from dataclasses import replace
from pathlib import Path

import pytest

from src.gui.job_controller import JobController, ThreadLauncher
from src.gui.queue_runner import QueueRunner
from src.ingest import queue as q
from src.ingest.router import Detected, classify
from src.model_manager import ModelSpec
from src.pipeline import job_key
from src.services import transcription_service as tsvc
from src.services.model_service import ModelService
from src.services.transcription_service import JobRequest
from tests.conftest import write_silent_wav
from tests.gui_support import make_services, settle, wait_for
from tests.test_expert_pipeline import LectureFake
from tests.test_ingest_documents import text_pdf


@pytest.fixture
def env(qapp, tmp_path, monkeypatch, fake_ffmpeg):
    services = make_services(tmp_path)
    monkeypatch.setattr(ModelService, "model_spec", lambda self, key: ModelSpec(key, "faster_whisper", "large-v3"))
    monkeypatch.setattr(tsvc, "STOP_POLL_SECONDS", 0.0)
    monkeypatch.setattr("src.transcriber.create_adapter", lambda *a, **k: LectureFake())
    LectureFake.requests = []
    services.update_setting(chunk_minutes=1.0, output_dir=str(tmp_path / "out"))
    controller = JobController(services, ThreadLauncher(), poll_ms=20)
    template = JobRequest(source="", output_dir="", model_key="whisper-large-v3", language="en", chunk_minutes=1.0)
    holder = types.SimpleNamespace(services=services, controller=controller, tmp=tmp_path, template=template, runners=[])

    def make(paths) -> QueueRunner:
        runner = QueueRunner(services, controller, classify(paths).items, template)
        holder.runners.append(runner)
        return runner
    holder.make = make
    yield holder
    controller._timer.stop()
    for runner in holder.runners:
        runner.queue.cancel()
    settle(controller)


def lectures(tmp, n=3, seconds=40):
    return [write_silent_wav(tmp / f"Lecture_{i:02d}.wav", seconds=seconds) for i in range(1, n + 1)]


def finished(runner):
    return runner.queue.state in (q.DONE, q.CANCELLED)


def test_three_files_run_one_after_another_into_a_batch_folder(env):
    runner = env.make(lectures(env.tmp))
    started = []
    env.controller.job_started.connect(started.append)
    runner.start()
    wait_for(lambda: finished(runner), timeout=120, what="the batch to finish")
    assert [i.state for i in runner.items] == [q.COMPLETED] * 3 and runner.queue.state == q.DONE, [i.error for i in runner.items]
    assert len(started) == 3 and len(set(started)) == 3                        # one job per file; the controller refuses a second
    root = runner.root
    assert root.parent == env.tmp / "out" and root.name.startswith("Batch_")
    for n in (1, 2, 3):
        folder = root / f"Lecture_0{n}"
        names = {p.name for p in folder.iterdir()}
        assert {f"Lecture_0{n}_raw.txt", f"Lecture_0{n}_clean.txt", f"Lecture_0{n}_timestamped.txt", f"Lecture_0{n}.srt",
                f"Lecture_0{n}.vtt", f"Lecture_0{n}.json"} <= names
        assert (folder / f"Lecture_0{n}_raw.txt").read_bytes() == (folder / "local_raw_transcript.txt").read_bytes()
    assert len(LectureFake.requests) >= 3


def test_one_file_keeps_the_normal_output_folder(env):
    source = lectures(env.tmp, 1)[0]
    runner = env.make([source])
    runner.start()
    wait_for(lambda: finished(runner), timeout=60, what="the single job")
    assert runner.items[0].state == q.COMPLETED and "Batch_" not in str(runner.root)
    assert (runner.root / "Lecture_01" / "Lecture_01_raw.txt").is_file()


def test_a_broken_file_fails_alone_and_the_rest_complete(env):
    good = lectures(env.tmp, 2)
    bad = env.tmp / "Lecture_02b.wav"
    bad.write_bytes(b"RIFF....WAVEgarbage" * 4)
    items = [Detected(good[0], kind="audio"), Detected(bad, kind="audio"), Detected(good[1], kind="audio")]
    runner = QueueRunner(env.services, env.controller, items, env.template)
    env.runners.append(runner)
    runner.start()
    wait_for(lambda: finished(runner), timeout=120, what="the batch to finish")
    assert [i.state for i in runner.items] == [q.COMPLETED, q.FAILED, q.COMPLETED] and runner.items[1].error


def test_pause_keeps_the_checkpoint_and_resume_finishes_the_same_item(env, monkeypatch):
    files = lectures(env.tmp, 3, seconds=120)
    reached = threading.Event()
    state = {"n": 0, "armed": True}

    class PausingFake(LectureFake):
        def transcribe(self, request):
            real = request.on_segment

            def on_segment(seg):
                real(seg)
                state["n"] += 1
                if state["armed"] and state["n"] == 20:                   # somewhere inside the second recording
                    reached.set()
                    time.sleep(1.0)                                       # the GUI thread passes the pause request on meanwhile

            if real is not None:
                request.on_segment = on_segment
            return super().transcribe(request)

    monkeypatch.setattr("src.transcriber.create_adapter", lambda *a, **k: PausingFake())
    runner = env.make(files)
    runner.start()
    wait_for(reached.is_set, timeout=60, what="the second recording to be running")
    runner.pause()
    wait_for(lambda: runner.queue.state == q.PAUSED, timeout=60, what="the queue to pause")
    states = [i.state for i in runner.items]
    assert q.COMPLETED in states and states.count(q.PAUSED) == 1 and states[-1] == q.QUEUED
    paused = next(i for i in runner.items if i.state == q.PAUSED)
    assert (env.services.paths.checkpoints / job_key(paused.detected.path)).exists()
    done_before = [i.name for i in runner.items if i.state == q.COMPLETED]
    state["armed"] = False
    runner.resume()
    wait_for(lambda: finished(runner), timeout=120, what="the rest of the batch")
    assert [i.state for i in runner.items] == [q.COMPLETED] * 3
    assert all(runner.items[[i.name for i in runner.items].index(n)].state == q.COMPLETED for n in done_before)


def test_cancel_stops_the_running_file_and_the_waiting_ones(env, monkeypatch):
    files = lectures(env.tmp, 3, seconds=120)
    reached = threading.Event()

    class SlowFake(LectureFake):
        def transcribe(self, request):
            real = request.on_segment

            def on_segment(seg):
                real(seg)
                if not reached.is_set():
                    reached.set()
                    time.sleep(1.5)

            if real is not None:
                request.on_segment = on_segment
            return super().transcribe(request)

    monkeypatch.setattr("src.transcriber.create_adapter", lambda *a, **k: SlowFake())
    runner = env.make(files)
    runner.start()
    wait_for(reached.is_set, timeout=60, what="the first recording to be running")
    runner.cancel()
    wait_for(lambda: not env.controller.is_running, timeout=60, what="the worker to end")
    assert [i.state for i in runner.items] == [q.CANCELLED] * 3 and runner.queue.state == q.CANCELLED


def test_documents_share_the_queue_with_audio(env):
    audio = lectures(env.tmp, 1)[0]
    pdf = text_pdf(env.tmp / "slides.pdf", ["The slide text about the nephron is long enough to be real embedded text.", ""])
    note = env.tmp / "notes.txt"
    note.write_text("Plain lecture notes.", encoding="utf-8")
    runner = env.make([audio, pdf, note])
    events = []
    runner.progress.connect(lambda index, event: events.append((index, event.get("type"))))
    runner.start()
    wait_for(lambda: finished(runner), timeout=120, what="audio + documents")
    assert [i.state for i in runner.items] == [q.COMPLETED] * 3
    assert Path(runner.items[1].files["text"]).name == "slides_text.txt" and "nephron" in Path(runner.items[1].files["text"]).read_text(encoding="utf-8")
    assert Path(runner.items[2].files["json"]).name == "notes.json"
    assert (1, "document_progress") in events


def test_eleven_audio_files_are_refused_before_anything_starts(env):
    files = lectures(env.tmp, 11, seconds=2)
    selection = classify(files)
    assert selection.too_many
    with pytest.raises(q.QueueError):
        QueueRunner(env.services, env.controller, selection.items, env.template)
