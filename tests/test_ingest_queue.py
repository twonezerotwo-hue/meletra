"""The batch queue: sequential, max 10, pause/resume without re-running finished items, failures do not stop the others."""

from datetime import datetime
from pathlib import Path

import pytest

from src.ingest import queue as q
from src.ingest.queue import BatchQueue, QueueError, batch_output_root, export_named, prepare_media
from src.ingest.router import Detected, detect
from tests.conftest import write_silent_wav


def media(n: int, kind: str = "audio") -> list[Detected]:
    return [Detected(Path(f"Lecture_{i:02d}.m4a"), kind=kind) for i in range(1, n + 1)]


class Rig:
    """A queue whose 'jobs' are recorded; the test plays the part of the worker."""

    def __init__(self, items):
        self.started, self.paused, self.cancelled, self.changes = [], 0, 0, 0
        self.queue = BatchQueue(items, self.start, self.pause, self.cancel, self.changed)

    def start(self, item):
        self.started.append(item.name)

    def pause(self):
        self.paused += 1

    def cancel(self):
        self.cancelled += 1

    def changed(self):
        self.changes += 1

    @property
    def states(self):
        return [i.state for i in self.queue.items]


def test_items_run_one_at_a_time_in_order():
    rig = Rig(media(3))
    rig.queue.start()
    assert rig.started == ["Lecture_01.m4a"] and rig.states == [q.PROCESSING, q.QUEUED, q.QUEUED]
    rig.queue.item_completed("out/1")
    assert rig.started == ["Lecture_01.m4a", "Lecture_02.m4a"]
    rig.queue.item_completed()
    rig.queue.item_completed()
    assert rig.states == [q.COMPLETED] * 3 and rig.queue.state == q.DONE and len(rig.started) == 3


def test_ten_files_are_allowed_and_eleven_are_an_error():
    assert len(Rig(media(10)).queue.items) == 10
    with pytest.raises(QueueError, match="limit is 10"):
        Rig(media(11))
    Rig(media(10) + [Detected(Path("notes.pdf"), kind="pdf")])                 # documents do not count


def test_a_failed_item_does_not_stop_the_others():
    rig = Rig(media(3))
    rig.queue.start()
    rig.queue.item_failed("Could not decode the audio")
    assert rig.started[-1] == "Lecture_02.m4a"
    rig.queue.item_completed()
    rig.queue.item_completed()
    assert rig.states == [q.FAILED, q.COMPLETED, q.COMPLETED] and rig.queue.items[0].error == "Could not decode the audio"


def test_the_same_failure_twice_in_a_row_stops_the_rest():
    rig = Rig(media(4))
    rig.queue.start()
    rig.queue.item_failed("The selected model is not installed")
    rig.queue.item_failed("The selected model is not installed")
    assert rig.states == [q.FAILED, q.FAILED, q.CANCELLED, q.CANCELLED] and rig.queue.state == q.DONE and len(rig.started) == 2


def test_different_failures_keep_going():
    rig = Rig(media(3))
    rig.queue.start()
    rig.queue.item_failed("a")
    rig.queue.item_failed("b")
    assert rig.states == [q.FAILED, q.FAILED, q.PROCESSING]


def test_a_start_that_raises_fails_only_that_item():
    items = media(2)
    seen = []

    def start(item):
        seen.append(item.name)
        if len(seen) == 1:
            raise RuntimeError("cannot start")
    queue = BatchQueue(items, start, lambda: None, lambda: None)
    queue.start()
    assert [i.state for i in queue.items] == [q.FAILED, q.PROCESSING] and queue.items[0].error == "cannot start"


def test_pause_stops_the_current_job_and_resume_continues_the_same_item():
    rig = Rig(media(3))
    rig.queue.start()
    rig.queue.item_completed()
    rig.queue.pause()
    assert rig.paused == 1
    rig.queue.pause()                                                         # asking twice does nothing more
    assert rig.paused == 1
    rig.queue.item_paused()
    assert rig.queue.state == q.PAUSED and rig.states == [q.COMPLETED, q.PAUSED, q.QUEUED]
    assert rig.started == ["Lecture_01.m4a", "Lecture_02.m4a"]               # nothing new started while paused
    rig.queue.resume()
    assert rig.started[-1] == "Lecture_02.m4a" and len(rig.started) == 3     # the SAME item again; Lecture_01 never reruns
    rig.queue.item_completed()
    rig.queue.item_completed()
    assert rig.states == [q.COMPLETED] * 3 and rig.started.count("Lecture_01.m4a") == 1


def test_pause_that_arrives_after_the_item_finished_still_holds_the_queue():
    rig = Rig(media(3))
    rig.queue.start()
    rig.queue.pause()
    rig.queue.item_completed()                                                # the job ended before it could pause
    assert rig.queue.state == q.PAUSED and rig.states == [q.COMPLETED, q.QUEUED, q.QUEUED] and len(rig.started) == 1
    rig.queue.resume()
    assert rig.started == ["Lecture_01.m4a", "Lecture_02.m4a"]


def test_cancel_stops_the_job_and_everything_waiting_but_keeps_finished_items():
    rig = Rig(media(4))
    rig.queue.start()
    rig.queue.item_completed()
    rig.queue.cancel()
    assert rig.cancelled == 1 and rig.states == [q.COMPLETED, q.CANCELLED, q.CANCELLED, q.CANCELLED] and rig.queue.state == q.CANCELLED
    rig.queue.item_failed("the killed job reports a crash")                   # late reports are ignored
    rig.queue.item_completed()
    assert rig.states[0] == q.COMPLETED and rig.states[1] == q.CANCELLED and len(rig.started) == 2


def test_cancel_while_paused_needs_no_worker():
    rig = Rig(media(2))
    rig.queue.start()
    rig.queue.pause()
    rig.queue.item_paused()
    rig.queue.cancel()
    assert rig.cancelled == 0 and rig.states == [q.CANCELLED, q.CANCELLED]


def test_position_counts_from_one():
    rig = Rig(media(3))
    rig.queue.start()
    assert rig.queue.position == (1, 3)
    rig.queue.item_completed()
    assert rig.queue.position == (2, 3)


# ----------------------------------------------------------------------------------------------- files
def make_outputs(folder: Path, text="x"):
    folder.mkdir(parents=True, exist_ok=True)
    for original, _ in q.NAMED.values():
        (folder / original).write_text(f"{original}:{text}", encoding="utf-8")


def test_named_copies_derive_from_the_source_name(tmp_path):
    make_outputs(tmp_path / "Lecture_01")
    made = export_named(tmp_path / "Lecture_01", Path("C:/in/Lecture_01.m4a"))
    names = sorted(Path(p).name for p in made.values())
    assert names == ["Lecture_01.json", "Lecture_01.srt", "Lecture_01.vtt", "Lecture_01_clean.txt", "Lecture_01_raw.txt",
                     "Lecture_01_timestamped.txt"]
    assert (tmp_path / "Lecture_01" / "local_raw_transcript.txt").is_file()               # the pipeline's own files stay


def test_named_copies_are_never_overwritten_with_different_content(tmp_path):
    folder = tmp_path / "L"
    make_outputs(folder, "first")
    export_named(folder, Path("L.wav"))
    export_named(folder, Path("L.wav"))                                                    # same content: nothing new
    assert len(list(folder.glob("L_raw*.txt"))) == 1
    (folder / "local_raw_transcript.txt").write_text("changed", encoding="utf-8")
    made = export_named(folder, Path("L.wav"))
    assert Path(made["raw"]).name == "L_raw (2).txt" and (folder / "L_raw.txt").read_text(encoding="utf-8").endswith("first")


def test_named_copies_of_a_very_long_video_stay_below_the_windows_path_limit(tmp_path):
    folder = tmp_path / ("results-" + "x" * 90)
    make_outputs(folder)
    source = Path("C:/in") / (("Korean-Mix-Hindi-Songs-" * 8) + ".mp4")
    made = export_named(folder, source)
    assert len(made) == len(q.NAMED)
    assert all(Path(path).is_file() and len(str(Path(path).resolve())) <= q.WINDOWS_SAFE_PATH for path in made.values())
    assert len({Path(path).stem.rsplit("_", 1)[0] for path in made.values()}) >= 1


def test_batch_folder_is_timestamped_and_unique(tmp_path):
    now = datetime(2026, 10, 11, 9, 5, 7)
    first = batch_output_root(tmp_path, now)
    assert first.name == "Batch_20261011_090507"
    first.mkdir()
    assert batch_output_root(tmp_path, now).name == "Batch_20261011_090507_2"


def test_prepare_media_passes_normal_files_through_and_converts_the_rest(tmp_path):
    wav = write_silent_wav(tmp_path / "a.wav", 0.5)
    assert prepare_media(detect(wav), tmp_path / "work") == wav
    odd = tmp_path / "lecture.dat"
    odd.write_bytes(wav.read_bytes())
    found = detect(odd)
    assert found.kind == "audio" and found.needs_wav
    out = prepare_media(found, tmp_path / "work")
    assert out.name == "lecture.wav" and out.is_file() and odd.read_bytes() == wav.read_bytes()
    assert detect(out).kind == "audio" and not detect(out).needs_wav
