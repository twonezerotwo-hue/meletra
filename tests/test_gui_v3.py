"""The v3 screens: one start screen for any file, a queue screen, "Lecture Ready" / "Document Ready", cancel, listen. The older GUI tests
(test_gui_simple, test_gui_job ...) keep covering the single-recording path that runs through the 0.2 pages."""

import threading
import time

import pytest
from PIL import Image, ImageDraw, ImageFont

from src.gui.main_window import SIMPLE_ACTIVE, SIMPLE_BATCH, SIMPLE_HOME, SIMPLE_RESULT, SIMPLE_TRANSCRIPT
from src.ingest import queue as q
from src.ingest.ocr import OcrResult
from tests.conftest import write_silent_wav
from tests.gui_support import settle, wait_for
from tests.study_support import write_pdf
from tests.test_gui_job import window_env  # noqa: F401 - runs the real worker with a fake speech model
from tests.test_gui_simple import assert_no_jargon
from tests.test_ingest_documents import FakeOcr, text_pdf
from tests.test_expert_pipeline import LectureFake


def choose(window, *paths):
    window.simple_home.drop_zone.files_chosen.emit([str(p) for p in paths])
    wait_for(lambda: not window.simple_home.classifying, what="the chosen files to be read")      # read in a background thread


def finished(window):
    return window.runner is not None and window.runner.queue.state in (q.DONE, q.CANCELLED)


@pytest.fixture
def env(window_env, monkeypatch):  # noqa: F811
    window_env.services.update_setting(chunk_minutes=1.0, output_dir=str(window_env.tmp / "out"))
    return window_env


# ----------------------------------------------------------------------------------------------- start screen
def test_start_screen_wording(env):
    home = env.window.simple_home
    assert home.title.text() == "MELETRA" and home.title.font().capitalization().name != "AllUppercase"          # the product name, not a shouting heading
    assert home.drop_zone.text() == "Drop lecture files here" and home.browse_button.text() == "Browse files"
    assert home.transcribe_button.text() == "TRANSCRIBE" and not home.transcribe_button.isEnabled()
    assert_no_jargon(home)


def test_a_pdf_shows_its_kind_and_page_count_then_converts(env):
    pdf = text_pdf(env.tmp / "notes.pdf", ["The nephron is the functional unit of the kidney and filters blood plasma.",
                                           "The loop of Henle creates the concentration gradient of the medulla."])
    w = env.window
    choose(w, pdf)
    home = w.simple_home
    assert home.file_card.isVisibleTo(home) and home.file_name.text() == "notes.pdf" and home.file_info.text() == "PDF · 2 pages"
    assert home.transcribe_button.isEnabled() and home.pdf_button.isHidden() or not home.pdf_button.isVisibleTo(home)
    home.transcribe_button.click()
    wait_for(lambda: w.simple_stack.currentIndex() == SIMPLE_RESULT, timeout=60, what="Document Ready")
    r = w.simple_result
    assert r.title.text() == "Document Ready" and r.read_button.text() == "Read Text" and r.file_label.text() == "notes.pdf"
    assert "Embedded PDF Text" in r.info_label.text() and "confidence" not in r.info_label.text()          # no invented percentage
    assert r.read_button.isEnabled() and r.read_button.text() == "Read Text"
    assert r.listen_button.isEnabled() and r.study_button.isEnabled() and r.study_button.text() == "Create Study Material"
    assert_no_jargon(r)


def test_read_text_opens_the_document_text_and_study_uses_it(env):
    pdf = text_pdf(env.tmp / "slides.pdf", ["Glycolysis converts glucose into pyruvate and produces two molecules of ATP."])
    w = env.window
    choose(w, pdf)
    w.simple_home.transcribe_button.click()
    wait_for(lambda: w.simple_stack.currentIndex() == SIMPLE_RESULT, timeout=60, what="Document Ready")
    w.simple_result.read_button.click()
    assert w.simple_stack.currentIndex() == SIMPLE_TRANSCRIPT and "Glycolysis" in w.transcript_view.text.toPlainText()
    assert w.transcript_view.view_combo.count() == 1
    state = env.services.study.prepare(w.current_output)                   # the converted pages are the study sources
    assert state.index is not None and any("Glycolysis" in c.text for c in state.index.chunks)


def test_a_picture_is_read_with_ocr_and_shows_its_real_confidence(env, monkeypatch):
    engine = FakeOcr(["Slide: the cardiac cycle has systole and diastole"], confidence=0.93)
    monkeypatch.setattr("src.ingest.documents.create_ocr", lambda language="auto": engine)
    picture = env.tmp / "slide.png"
    image = Image.new("RGB", (800, 240), "white")
    ImageDraw.Draw(image).text((20, 90), "cardiac cycle", fill="black", font=ImageFont.load_default(size=40))
    image.save(picture)
    w = env.window
    choose(w, picture)
    assert w.simple_home.file_info.text() == "Image"
    w.simple_home.transcribe_button.click()
    wait_for(lambda: w.simple_stack.currentIndex() == SIMPLE_RESULT, timeout=60, what="Document Ready")
    assert "confidence 93%" in w.simple_result.info_label.text() and "cardiac cycle" in (w.current_output / "slide_text.txt").read_text(encoding="utf-8")


def test_a_recording_with_its_slides_keeps_the_lecture_path(env):
    w = env.window
    pdf = write_pdf(env.tmp / "slides.pdf", [["Glycolysis"]])
    choose(w, env.source, pdf)
    wait_for(lambda: w.home.duration is not None, what="the recording to be read")
    assert w.home.source == env.source and w.home.pdf_path == pdf and not w.simple_home.selection


def test_eleven_recordings_are_refused_with_a_message(env):
    w = env.window
    files = [write_silent_wav(env.tmp / f"l{i:02d}.wav", seconds=2) for i in range(11)]
    choose(w, *files)
    assert not w.simple_home.selection and "limit is 10" in w.simple_home.problem.text()


def test_an_unsupported_file_is_named_in_the_message(env):
    junk = env.tmp / "notes.xyz"
    junk.write_bytes(b"\x00\x01")
    choose(env.window, junk)
    assert "notes.xyz" in env.window.simple_home.problem.text() and not env.window.simple_home.selection


# ----------------------------------------------------------------------------------------------- the queue screen
@pytest.mark.usefixtures("fake_ffmpeg")
def test_three_recordings_convert_one_after_another_and_each_can_be_opened(env):
    w = env.window
    files = [write_silent_wav(env.tmp / f"Lecture_0{i}.wav", seconds=40) for i in (1, 2, 3)]
    choose(w, *files)
    home = w.simple_home
    assert home.file_name.text() == "3 files" and "Lecture_02.wav" in home.file_info.text() and home.transcribe_button.isEnabled()
    home.transcribe_button.click()
    assert w.simple_stack.currentIndex() == SIMPLE_BATCH
    wait_for(lambda: finished(w), timeout=120, what="the batch")
    b = w.simple_batch
    assert b.list.count() == 3 and all("Done" in b.list.item(i).text() for i in range(3)) and b.title.text() == "Done"
    assert b.open_button.isVisibleTo(b) and not b.pause_button.isVisibleTo(b)
    assert_no_jargon(b)
    w.open_queue_item(1)
    assert w.simple_stack.currentIndex() == SIMPLE_RESULT and w.simple_result.title.text() == "Lecture Ready"
    assert (w.current_output / "Lecture_02_raw.txt").is_file() and (w.current_output / "Lecture_02.srt").is_file()
    assert w.current_output.parent.name.startswith("Batch_")


def test_the_batch_screen_stays_while_the_next_file_converts(env):
    """After file 1 is done the window must keep showing the queue (file 2 in progress), not jump to file 1's result screen."""
    w = env.window
    files = [write_silent_wav(env.tmp / f"Lecture_0{i}.wav", seconds=40) for i in (1, 2)]
    choose(w, *files)
    w.simple_home.transcribe_button.click()
    seen = []
    w.runner.changed.connect(lambda: seen.append((w.runner.queue.items[0].state, w.runner.queue.state, w.simple_stack.currentIndex())))
    wait_for(lambda: finished(w), timeout=120, what="the batch")
    between = [index for first, state, index in seen if first == q.COMPLETED and state == q.RUNNING]
    assert between and all(index == SIMPLE_BATCH for index in between), seen
    assert w.simple_stack.currentIndex() == SIMPLE_BATCH and w.simple_batch.title.text() == "Done"
    w.open_queue_item(0)                                       # the first result stays reachable from the list
    assert w.simple_stack.currentIndex() == SIMPLE_RESULT


@pytest.mark.usefixtures("fake_ffmpeg")
def test_a_single_recording_also_gets_the_named_files(env):
    w = env.window
    choose(w, env.source)
    wait_for(lambda: w.simple_home.transcribe_button.isEnabled(), what="CONVERT TO TEXT")
    w.simple_home.transcribe_button.click()
    assert w.simple_stack.currentIndex() == SIMPLE_ACTIVE
    wait_for(lambda: w.simple_stack.currentIndex() == SIMPLE_RESULT, timeout=60, what="Lecture Ready")
    assert (w.current_output / "lecture_raw.txt").is_file() and (w.current_output / "lecture.json").is_file()
    assert w.simple_result.title.text() == "Lecture Ready"


@pytest.mark.usefixtures("fake_ffmpeg")
def test_cancel_during_a_recording_keeps_the_progress_and_returns_to_the_start(env, monkeypatch):
    w = env.window
    reached = threading.Event()

    class Slow(LectureFake):
        def transcribe(self, request):
            real = request.on_segment

            def on_segment(seg):
                real(seg)
                if not reached.is_set():
                    reached.set()
                    time.sleep(1.0)

            if real is not None:
                request.on_segment = on_segment
            return super().transcribe(request)

    monkeypatch.setattr("src.transcriber.create_adapter", lambda *a, **k: Slow())
    choose(w, env.source)
    wait_for(lambda: w.simple_home.transcribe_button.isEnabled(), what="CONVERT TO TEXT")
    w.simple_home.transcribe_button.click()
    wait_for(reached.is_set, timeout=60, what="the job to run")
    assert w.simple_active.cancel_button.isVisibleTo(w.simple_active) and w.simple_active.pause_button.text() == "Pause"
    w.simple_active.cancel_button.click()
    wait_for(lambda: w.simple_stack.currentIndex() == SIMPLE_HOME, timeout=60, what="the start screen")
    assert w.home.source is None
    from src.pipeline import job_key
    assert env.services.jobs.get(job_key(env.source)).state == "paused"                # nothing was lost


@pytest.mark.usefixtures("fake_ffmpeg")
def test_pause_and_cancel_buttons_drive_the_queue(env, monkeypatch):
    w = env.window
    files = [write_silent_wav(env.tmp / f"Lecture_0{i}.wav", seconds=120) for i in (1, 2)]
    reached = threading.Event()
    hold = {"on": True}

    class Slow(LectureFake):
        def transcribe(self, request):
            real = request.on_segment

            def on_segment(seg):
                real(seg)
                if hold["on"]:
                    reached.set()
                    time.sleep(1.0)

            if real is not None:
                request.on_segment = on_segment
            return super().transcribe(request)

    monkeypatch.setattr("src.transcriber.create_adapter", lambda *a, **k: Slow())
    choose(w, *files)
    w.simple_home.transcribe_button.click()
    wait_for(reached.is_set, timeout=60, what="the first recording to run")
    b = w.simple_batch
    assert b.pause_button.isVisibleTo(b) and b.cancel_button.isVisibleTo(b)
    b.pause_button.click()
    wait_for(lambda: w.runner.queue.state == q.PAUSED, timeout=60, what="PAUSED")
    assert b.title.text() == "Paused" and b.resume_button.isVisibleTo(b) and "Paused" in b.list.item(0).text()
    hold["on"] = False
    b.resume_button.click()
    wait_for(lambda: finished(w), timeout=120, what="the rest of the queue")
    assert all("Done" in b.list.item(i).text() for i in range(2))


def test_cancel_marks_the_waiting_files_and_keeps_the_finished_ones(env):
    w = env.window
    a = text_pdf(env.tmp / "a.pdf", ["First document with enough embedded text to be treated as real text."])
    b = text_pdf(env.tmp / "b.pdf", ["Second document with enough embedded text to be treated as real text."])
    choose(w, a, b)
    w.simple_home.transcribe_button.click()
    runner = w.runner
    runner.cancel()
    wait_for(lambda: all(i.state in (q.COMPLETED, q.CANCELLED) for i in runner.items), timeout=30, what="the queue to settle")
    assert runner.queue.state == q.CANCELLED and w.simple_batch.title.text() == "Cancelled"
    settle(w)


# ----------------------------------------------------------------------------------------------- listen
def test_listen_to_text_reads_aloud_and_stops(env):
    pdf = text_pdf(env.tmp / "read.pdf", ["The kidney regulates the volume and the composition of the body fluids every day."])
    w = env.window
    choose(w, pdf)
    w.simple_home.transcribe_button.click()
    wait_for(lambda: w.simple_stack.currentIndex() == SIMPLE_RESULT, timeout=60, what="Document Ready")

    class FakeTts:
        def __init__(self):
            self.spoken, self.stopped = [], threading.Event()

        def speak(self, text, language="auto", should_stop=None, **kw):
            self.spoken.append(text)
            self.stopped.wait(20)
            return False

        def stop(self):
            self.stopped.set()
    fake = w._tts = FakeTts()
    r = w.simple_result
    r.listen_button.click()
    wait_for(lambda: fake.spoken, what="the text to be handed to the voice")
    assert r.listen_button.text() == "Stop Listening" and "kidney" in fake.spoken[0]
    r.listen_button.click()
    wait_for(lambda: r.listen_button.text() == "Listen to Text", what="listening to stop")
    assert not w._listening


def test_a_voice_problem_is_shown_and_nothing_else_breaks(env):
    pdf = text_pdf(env.tmp / "read.pdf", ["The kidney regulates the volume and the composition of the body fluids every day."])
    w = env.window
    choose(w, pdf)
    w.simple_home.transcribe_button.click()
    wait_for(lambda: w.simple_stack.currentIndex() == SIMPLE_RESULT, timeout=60, what="Document Ready")

    class NoVoice:
        def speak(self, *a, **k):
            from src.tts import TtsError
            raise TtsError("No English voice is installed on this computer.")

        def stop(self):
            pass
    w._tts = NoVoice()
    w.simple_result.listen_button.click()
    wait_for(lambda: "No English voice" in w.simple_result.message.text(), what="the message")
    assert w.simple_result.listen_button.text() == "Listen to Text" and w.simple_result.read_button.isEnabled()


def test_create_study_material_works_from_a_converted_document(env):
    from src.gui.main_window import SIMPLE_STUDY
    from src.gui.study_view import PAGE_TABS
    pdf = text_pdf(env.tmp / "pharm.pdf", ["Beta blockers reduce heart rate and blood pressure by blocking beta adrenergic receptors.",
                                           "ACE inhibitors lower blood pressure by preventing the formation of angiotensin II."])
    w = env.window
    choose(w, pdf)
    w.simple_home.transcribe_button.click()
    wait_for(lambda: w.simple_stack.currentIndex() == SIMPLE_RESULT, timeout=60, what="Document Ready")
    w.simple_result.study_button.click()
    assert w.simple_stack.currentIndex() == SIMPLE_STUDY
    wait_for(lambda: w.study.stack.currentIndex() == PAGE_TABS, timeout=60, what="the study workspace")
    assert w.study.state.pack is not None and w.study.state.pack.source_mode == "pdf"          # the converted pages are the slides


def test_new_files_wait_while_a_queue_is_running(env, monkeypatch):
    import src.gui.queue_runner as qr
    real = qr.read_document
    monkeypatch.setattr(qr, "read_document", lambda *a, **k: (time.sleep(0.6), real(*a, **k))[1])     # slow enough to be seen running
    w = env.window
    a = text_pdf(env.tmp / "a.pdf", ["First document with enough embedded text to be treated as real text."])
    b = text_pdf(env.tmp / "b.pdf", ["Second document with enough embedded text to be treated as real text."])
    choose(w, a, b)
    w.simple_home.transcribe_button.click()
    assert w.queue_active
    w.open_external_file(str(a))
    assert "converted" in w.statusBar().currentMessage() or "transcribed" in w.statusBar().currentMessage()
    wait_for(lambda: finished(w), timeout=30, what="the queue")
    assert not w.queue_active
