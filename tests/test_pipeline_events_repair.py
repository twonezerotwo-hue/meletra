"""Desktop-app pipeline contract: structured events, safe pause + resume, gap repair (accept / reject), timing report.

Everything runs through the real pipeline with a fake ASR backend; no GPU, model, or ffmpeg.exe is needed."""

import json
import random
import string

import pytest

from src.events import EVENT_TYPES
from src.model_manager import ASRAdapter, Capabilities, DecodeRequest, ModelSpec
from src.pipeline import RunOptions, checkpoint_dir_for, job_key, process_files
from src.runstats import SessionStats, safe_rtf, timing_report
from src.schema import Segment, TranscriptionResult
from tests.conftest import write_silent_wav

pytestmark = pytest.mark.usefixtures("fake_ffmpeg")

LETTERS = string.ascii_lowercase


def words_for(second: int) -> str:
    """Six distinct pseudo-words per 10 s slot, so no two segments look like copies of each other."""
    rng = random.Random(second)
    return " " + " ".join("".join(rng.choice(LETTERS) for _ in range(7)) for _ in range(6)) + "."


class GapFake(ASRAdapter):
    """Window-aware fake. With hotwords it can drop whole slots (the real failure: speech without text) or loop."""

    backend_name = "fake"
    capabilities = Capabilities(hotwords=True, vad=True, word_timestamps=True, compute_types=True)
    requests: list[DecodeRequest] = []
    drop_with_hotwords: set[int] = set()        # slot index (t // 10) omitted while hotwords are on
    redecode_text: dict[int, str] = {}          # slot index -> text returned by the plain re-decode instead of words_for
    redecode_no_speech = 0.01
    emitted = 0

    def __init__(self) -> None:
        super().__init__(ModelSpec("default", "faster_whisper", "large-v3"))

    def load(self, compute_type: str) -> None:
        self.compute_type = compute_type

    def transcribe(self, request: DecodeRequest) -> TranscriptionResult:
        GapFake.requests.append(request)
        total = request.total_duration
        begin = request.start_offset
        end = min(total, begin + request.max_seconds) if request.max_seconds else total
        plain = not request.hotwords
        segs, t = [], begin
        while t < end - 0.5:
            stop = min(t + 10.0, end)
            slot = int(t // 10)
            if not plain and slot in GapFake.drop_with_hotwords:
                t += 10.0
                continue
            text = GapFake.redecode_text.get(slot, words_for(int(t))) if (plain and request.max_seconds) else words_for(int(t))
            seg = Segment(t, stop, text, "en", "large-v3", "fake", avg_logprob=-0.2,
                          no_speech_prob=GapFake.redecode_no_speech if (plain and request.max_seconds) else 0.01,
                          compression_ratio=1.1)
            segs.append(seg)
            GapFake.emitted += 1
            if request.on_segment:
                request.on_segment(seg)
            if request.progress:
                request.progress(seg.end, total)
            t += 10.0
        return TranscriptionResult(segs, "en", 0.99, total, "large-v3", "fake", self.compute_type,
                                   hotwords_applied=list(request.hotwords[:5]), hotword_tokens=10)

    def close(self) -> None:
        pass


@pytest.fixture
def fake(monkeypatch):
    GapFake.requests, GapFake.drop_with_hotwords, GapFake.redecode_text = [], set(), {}
    GapFake.redecode_no_speech, GapFake.emitted = 0.01, 0
    monkeypatch.setattr("src.transcriber.create_adapter", lambda *a, **k: GapFake())
    monkeypatch.setattr("src.chunked_decode.time.sleep", lambda s: None, raising=False)


def speech_everywhere(monkeypatch, seconds):
    monkeypatch.setattr("src.quality.compute_speech_ranges", lambda wav, vad: [(0.0, float(seconds))])


def options(tmp_path, events=None, **kw):
    base = dict(outputs_root=tmp_path / "out", language="en", quality_audit=True, auto_topic=False, chunk_minutes=1.0,
                subjects=["biochemistry"], checkpoint_root=tmp_path / "ck", temp_root=tmp_path / "tmp",
                keep_checkpoint=True, events=events.append if events is not None else None)
    base.update(kw)
    return RunOptions(**base)


def media(tmp_path, seconds=200):
    return write_silent_wav(tmp_path / "lecture one.wav", seconds=seconds)


def out_dir(tmp_path):
    return tmp_path / "out" / "lecture one"


# ----------------------------------------------------------------------------------------------- events
def test_a_normal_job_emits_the_documented_events_in_a_sensible_order(tmp_path, cfg, fake, monkeypatch):
    speech_everywhere(monkeypatch, 200)
    events: list[dict] = []
    [summary] = process_files([media(tmp_path)], options(tmp_path, events), cfg)
    kinds = [e["type"] for e in events]
    assert summary.ok and kinds[0] == "job_started" and kinds[-1] == "completed"
    for needed in ("source_validated", "audio_preparing", "audio_ready", "language_detecting", "language_detected",
                   "subject_detected", "model_loading", "model_loaded", "chunk_started", "chunk_progress",
                   "chunk_finished", "checkpoint_saved", "gap_scan_started", "quality_audit_started",
                   "quality_audit_finished"):
        assert needed in kinds, needed
    assert set(kinds) <= set(EVENT_TYPES)
    assert kinds.index("audio_ready") < kinds.index("chunk_started")
    assert kinds.index("model_loading") < kinds.index("model_loaded") < kinds.index("chunk_progress")
    assert kinds.index("chunk_started") < kinds.index("chunk_finished") < kinds.index("gap_scan_started")
    assert kinds.index("quality_audit_finished") < kinds.index("completed")
    for event in events:
        json.dumps(event)                                                   # every payload is JSON serialisable
        assert "time" in event
    progress = [e for e in events if e["type"] == "chunk_progress"][-1]
    assert {"percent", "processed", "total", "elapsed", "eta", "rtf", "chunk", "chunks"} <= set(progress)
    done = events[-1]
    assert done["state"] == "completed" and done["segments"] == summary.segments and done["timing"]["audio_duration_total"] == 200


def test_a_broken_listener_cannot_break_the_job(tmp_path, cfg, fake):
    def explode(event):
        raise RuntimeError("GUI died")

    [summary] = process_files([media(tmp_path)], RunOptions(outputs_root=tmp_path / "out", language="en", quality_audit=False,
                                                           auto_topic=False, events=explode), cfg)
    assert summary.ok


def test_a_failing_job_emits_failed(tmp_path, cfg, fake, monkeypatch):
    class Boom(GapFake):
        def transcribe(self, request):
            raise ValueError("model exploded")

    monkeypatch.setattr("src.transcriber.create_adapter", lambda *a, **k: Boom())
    events: list[dict] = []
    with pytest.raises(Exception):
        process_files([media(tmp_path, 30)], options(tmp_path, events, chunk_minutes=None), no_retries(cfg))
    assert events[-1]["type"] == "failed" and "model exploded" in events[-1]["error"]


def no_retries(cfg):
    return {**cfg, "long_audio": {**cfg["long_audio"], "chunk_retries": 0}}


# ----------------------------------------------------------------------------------------------- pause / resume
def test_pause_stops_at_a_safe_boundary_keeps_the_checkpoint_and_returns_normally(tmp_path, cfg, fake, monkeypatch):
    speech_everywhere(monkeypatch, 200)
    events: list[dict] = []
    opts = options(tmp_path, events, should_stop=lambda: GapFake.emitted >= 3)
    [summary] = process_files([media(tmp_path)], opts, cfg)
    assert summary.paused and not summary.ok and summary.state == "paused" and summary.segments == 3
    kinds = [e["type"] for e in events]
    assert kinds.count("pause_requested") == 1 and kinds[-1] == "paused" and "completed" not in kinds
    paused = events[-1]
    assert paused["segments"] == 3 and paused["position_seconds"] > 0
    assert (out_dir(tmp_path) / "partial_transcript.txt").is_file()
    assert not (out_dir(tmp_path) / "local_raw_transcript.txt").exists()          # no half-finished "final" files
    checkpoint = checkpoint_dir_for(opts, media(tmp_path))
    assert checkpoint.is_dir() and (checkpoint / "chunks").is_dir()                # state lives in the user-data folder
    assert json.loads((checkpoint / "checkpoint.json").read_text(encoding="utf-8"))["status"] == "paused"


def test_resume_continues_after_the_pause_and_never_decodes_finished_work_again(tmp_path, cfg, fake, monkeypatch):
    speech_everywhere(monkeypatch, 200)
    src = media(tmp_path)
    process_files([src], options(tmp_path, should_stop=lambda: GapFake.emitted >= 3), cfg)
    first_requests = len(GapFake.requests)
    events: list[dict] = []
    GapFake.emitted = 0
    [summary] = process_files([src], options(tmp_path, events), cfg)
    assert summary.ok and summary.resumed_segments >= 3
    resume = [e for e in events if e["type"] == "resume_started"]
    assert resume and resume[0]["partial"] == 1 and resume[0]["resume_from_seconds"] > 0
    second = GapFake.requests[first_requests:]
    assert round(second[0].start_offset) >= 20                                      # first chunk continues after its 3 saved segments
    raw = (out_dir(tmp_path) / "local_raw_transcript.txt").read_text(encoding="utf-8").splitlines()
    assert len(raw) == len(set(raw))                                                # every sentence exactly once
    data = json.loads((out_dir(tmp_path) / "local_transcript.json").read_text(encoding="utf-8"))
    starts = [s["start"] for s in data["segments"]]
    assert starts == sorted(starts) and starts[0] == 0


def test_finished_chunks_are_skipped_on_resume(tmp_path, cfg, fake, monkeypatch):
    speech_everywhere(monkeypatch, 200)
    src = media(tmp_path)
    process_files([src], options(tmp_path), cfg)                                    # complete run, checkpoint kept
    GapFake.requests = []
    events: list[dict] = []
    [summary] = process_files([src], options(tmp_path, events), cfg)
    assert summary.ok and GapFake.requests == []                                    # nothing decoded again
    finished = [e for e in events if e["type"] == "chunk_finished"]
    assert finished and all(e.get("skipped") for e in finished)
    assert summary.timing["real_time_factor_this_run"] is None                      # no audio decoded -> no absurd RTF


def test_a_changed_source_file_makes_the_old_checkpoint_unusable(tmp_path, cfg, fake, monkeypatch):
    speech_everywhere(monkeypatch, 200)
    src = media(tmp_path)
    process_files([src], options(tmp_path, should_stop=lambda: GapFake.emitted >= 3), cfg)
    write_silent_wav(src, seconds=200)                                              # same size, but make the content differ
    import wave

    with wave.open(str(src), "rb") as handle:
        params, frames = handle.getparams(), bytearray(handle.readframes(handle.getnframes()))
    frames[1000] = 7
    with wave.open(str(src), "wb") as handle:
        handle.setparams(params)
        handle.writeframes(bytes(frames))
    events: list[dict] = []
    GapFake.emitted = 0
    [summary] = process_files([src], options(tmp_path, events), cfg)
    assert summary.ok and summary.resumed_segments == 0                             # started over
    assert any(e["type"] == "warning" and e.get("code") == "checkpoint_incompatible" for e in events)


def test_job_key_is_stable_and_distinguishes_files(tmp_path):
    a, b = media(tmp_path, 5), write_silent_wav(tmp_path / "other.wav", seconds=6)
    assert job_key(a) == job_key(a) and job_key(a) != job_key(b) and len(job_key(a)) == 16


# ----------------------------------------------------------------------------------------------- gap repair
def test_a_gap_of_real_speech_is_recovered_without_hotwords_and_without_a_full_rerun(tmp_path, cfg, fake, monkeypatch):
    speech_everywhere(monkeypatch, 200)
    GapFake.drop_with_hotwords = {12}                                               # slot 120-130 comes back empty
    events: list[dict] = []
    [summary] = process_files([media(tmp_path)], options(tmp_path, events), cfg)
    assert summary.ok and summary.recovered_gaps == 1 and summary.recovered_gap_seconds >= 5
    plain = [r for r in GapFake.requests if not r.hotwords]
    assert len(plain) == 1                                                          # one small re-decode, nothing else
    request = plain[0]
    assert request.profile["condition_on_previous_text"] is False and request.profile["beam_size"] == 5
    assert request.profile["temperature"] == 0 and request.profile["word_timestamps"] is True
    assert request.max_seconds <= 20                                                # the gap plus padding, not the lecture
    assert len(GapFake.requests) == len([r for r in GapFake.requests if r.hotwords]) + 1
    kinds = [e["type"] for e in events]
    assert {"gap_found", "gap_repair_started", "gap_repair_finished"} <= set(kinds)
    assert [e for e in events if e["type"] == "gap_repair_finished"][0]["repaired"] is True
    data = json.loads((out_dir(tmp_path) / "local_transcript.json").read_text(encoding="utf-8"))
    inside = [s for s in data["segments"] if 126 <= (s["start"] + s["end"]) / 2 <= 138 and s["end"] - s["start"] <= 12]
    assert inside, "recovered text must sit inside the gap with absolute timestamps"
    meta = json.loads((out_dir(tmp_path) / "metadata.json").read_text(encoding="utf-8"))
    [gap] = meta["long_audio"]["gap_repairs"]
    assert gap["inserted"] >= 1 and 125 <= gap["start"] <= 129      # chunk 3's grid: 117, 127 (dropped), 137
    q = meta["quality"]
    assert q["missing_speech_gaps"] == 0                                            # the audit no longer finds the hole


def test_recovered_timestamps_stay_inside_the_gap(tmp_path, cfg, fake, monkeypatch):
    speech_everywhere(monkeypatch, 200)
    GapFake.drop_with_hotwords = {12}
    process_files([media(tmp_path)], options(tmp_path), cfg)
    data = json.loads((out_dir(tmp_path) / "local_transcript.json").read_text(encoding="utf-8"))
    segs = data["segments"]
    starts = [s["start"] for s in segs]
    assert starts == sorted(starts)
    for before, after in zip(segs, segs[1:]):
        assert after["start"] >= before["start"]
    assert max(s["end"] for s in segs) <= 200.5 and min(s["start"] for s in segs) >= 0


@pytest.mark.parametrize("mode", ["loop", "no_speech", "copy", "lone_word"])
def test_a_bad_repair_is_rejected_and_the_gap_is_reported(tmp_path, cfg, fake, monkeypatch, mode):
    speech_everywhere(monkeypatch, 200)
    GapFake.drop_with_hotwords = {12}
    if mode == "loop":
        GapFake.redecode_text = {slot: " glycogen glycogen glycogen glycogen glycogen glycogen." for slot in range(10, 15)}
    elif mode == "no_speech":
        GapFake.redecode_no_speech = 0.95
    elif mode == "copy":
        GapFake.redecode_text = {12: words_for(117), 13: " um"}                      # the padding re-heard the previous sentence
    elif mode == "lone_word":
        GapFake.redecode_text = {slot: " um" for slot in range(10, 15)}
    events: list[dict] = []
    [summary] = process_files([media(tmp_path)], options(tmp_path, events), cfg)
    assert summary.ok and summary.recovered_gaps == 0                               # nothing inserted, nothing broken
    done = [e for e in events if e["type"] == "gap_repair_finished"]
    assert done and done[0]["repaired"] is False
    meta = json.loads((out_dir(tmp_path) / "metadata.json").read_text(encoding="utf-8"))
    assert meta["long_audio"]["gap_repairs"][0]["inserted"] == 0
    assert meta["quality"]["missing_speech_gaps"] >= 1                              # still honestly reported


def test_a_gap_is_re_decoded_only_once_even_across_resumes(tmp_path, cfg, fake, monkeypatch):
    speech_everywhere(monkeypatch, 200)
    GapFake.drop_with_hotwords = {12}
    src = media(tmp_path)
    process_files([src], options(tmp_path), cfg)
    GapFake.requests = []
    [summary] = process_files([src], options(tmp_path), cfg)
    assert summary.ok and summary.recovered_gaps == 1 and GapFake.requests == []     # verdict came from gap_repairs.json


def test_gap_repair_can_be_switched_off(tmp_path, cfg, fake, monkeypatch):
    speech_everywhere(monkeypatch, 200)
    GapFake.drop_with_hotwords = {12}
    off = {**cfg, "long_audio": {**cfg["long_audio"], "gap_repair": False}}
    [summary] = process_files([media(tmp_path)], options(tmp_path), off)
    assert summary.ok and summary.recovered_gaps == 0 and not [r for r in GapFake.requests if not r.hotwords]


def test_uncovered_speech_finder():
    from src.chunked_decode import uncovered_speech

    segs = [Segment(0, 10, "a b c", "en", "m", "b"), Segment(30, 40, "d e f", "en", "m", "b")]
    assert uncovered_speech([(0.0, 40.0)], segs, 5.0) == [(10, 30)]
    assert uncovered_speech([(0.0, 40.0)], segs, 25.0) == []
    assert uncovered_speech([(12.0, 14.0)], segs, 1.0) == [(12.0, 14.0)]
    assert uncovered_speech([(0.0, 10.0)], segs, 1.0) == []


# ----------------------------------------------------------------------------------------------- fragmented VAD ranges
# Measured with the real model: a 9.5 s sentence went missing; the VAD cut the stretch into (22.9, 33.2), (33.4, 38.0) and
# (38.0, 44.8), so the uncovered pieces were 4.3 s and 4.6 s - each below the 5 s threshold - and nothing was reported or repaired.
FRAGMENTED_SPEECH = [(0.0, 4.0), (4.0, 15.0), (15.1, 22.9), (22.9, 33.2), (33.4, 38.0), (38.0, 44.8), (44.8, 51.1)]
HOLE_SEGMENTS = [Segment(0, 3.3, "a b c", "en", "m", "b"), Segment(4.3, 14.4, "d e f", "en", "m", "b"),
                 Segment(15.0, 22.3, "g h i", "en", "m", "b"), Segment(23.0, 28.9, "j k l", "en", "m", "b"),
                 Segment(38.4, 44.1, "m n o", "en", "m", "b"), Segment(45.4, 50.3, "p q r", "en", "m", "b")]


def test_a_hole_split_by_vad_cuts_is_one_gap():
    from src.chunked_decode import uncovered_speech

    [(a, b)] = uncovered_speech(FRAGMENTED_SPEECH, HOLE_SEGMENTS, 5.0)
    assert (a, b) == (28.9, 38.4)


def test_the_audit_sees_the_same_hole():
    from src.quality import missing_speech

    gaps, coverage = missing_speech(HOLE_SEGMENTS, FRAGMENTED_SPEECH, 8.0)
    assert [(g["start"], g["end"]) for g in gaps] == [(28.9, 38.4)] and coverage < 0.9


def test_pieces_are_not_joined_across_covered_text():
    from src.chunked_decode import uncovered_speech

    speech = [(10.0, 13.0), (13.2, 16.0)]
    tiny = [Segment(13.0, 13.2, "ok", "en", "m", "b")]
    assert uncovered_speech(speech, tiny, 5.0) == []                      # real text sits between: two 3 s pieces, not one 6 s hole


def test_pieces_far_apart_stay_separate():
    from src.chunked_decode import uncovered_speech

    assert uncovered_speech([(0.0, 3.0), (6.0, 9.0)], [], 5.0) == []     # a 3 s pause between two 3 s pieces: not one stretch
    assert uncovered_speech([(0.0, 3.0), (3.3, 9.0)], [], 5.0) == [(0.0, 9.0)]


def test_bridge_helpers_directly():
    from src.quality import GAP_BRIDGE_SECONDS, bridge_pieces, uncovered_pieces

    assert GAP_BRIDGE_SECONDS == 0.6
    assert uncovered_pieces([(0, 10)], [(2, 4), (6, 7)]) == [(0, 2), (4, 6), (7, 10)]
    assert bridge_pieces([(0, 2), (2.3, 4)], []) == [(0, 4)]
    assert bridge_pieces([(0, 2), (3, 4)], []) == [(0, 2), (3, 4)]


def fragmented_vad(monkeypatch):
    # the 127-137 s stretch comes back empty (chunk 3 grid); the VAD cuts it into pieces of 4.0 s, 4.3 s and 1.5 s - each below 5 s
    monkeypatch.setattr("src.quality.compute_speech_ranges",
                        lambda wav, vad: [(0.0, 131.0), (131.2, 135.5), (135.5, 200.0)])
    GapFake.drop_with_hotwords = {12}


def _words(*items):
    from src.schema import Word

    return [Word(a, b, w, 0.9) for a, b, w in items]


def test_recovered_text_is_trimmed_to_the_words_inside_the_gap():
    from src.chunked_decode import trim_to_gap

    seg = Segment(28.4, 36.0, " generates NADPH, and its deficiency leads to anemia", "en", "m", "b", words=_words(
        (28.4, 28.9, " generates"), (28.9, 29.2, " NADPH,"), (29.2, 29.4, " and"), (29.6, 30.0, " its"), (30.0, 30.6, " deficiency"),
        (30.6, 31.0, " leads"), (31.0, 31.2, " to"), (31.2, 36.0, " anemia")))
    out = trim_to_gap(seg, 29.5, 40.0)
    assert out.text.strip() == "its deficiency leads to anemia" and out.start == 29.6 and out.end == 36.0
    assert [w.word.strip() for w in out.words] == ["its", "deficiency", "leads", "to", "anemia"]


def test_trimming_leaves_segments_without_word_times_alone_and_drops_segments_with_no_word_inside():
    from src.chunked_decode import trim_to_gap

    plain = Segment(30, 34, " some text here", "en", "m", "b")
    assert trim_to_gap(plain, 29.0, 40.0) is plain
    outside = Segment(10, 12, " old words", "en", "m", "b", words=_words((10.0, 11.0, " old"), (11.0, 12.0, " words")))
    assert trim_to_gap(outside, 29.0, 40.0) is None


def test_the_pipeline_recovers_a_hole_that_the_vad_cut_in_pieces(tmp_path, cfg, fake, monkeypatch):
    fragmented_vad(monkeypatch)
    [summary] = process_files([media(tmp_path)], options(tmp_path), cfg)
    assert summary.ok and summary.recovered_gaps == 1 and summary.recovered_gap_seconds >= 8
    meta = json.loads((out_dir(tmp_path) / "metadata.json").read_text(encoding="utf-8"))
    assert meta["quality"]["missing_speech_gaps"] == 0


def test_without_bridging_that_same_hole_goes_unnoticed(tmp_path, cfg, fake, monkeypatch):
    """The regression this fix is for: proves the scenario above really needs the bridging."""
    fragmented_vad(monkeypatch)
    monkeypatch.setattr("src.quality.bridge_pieces", lambda pieces, covered, bridge=0.6: list(pieces))
    [summary] = process_files([media(tmp_path)], options(tmp_path), cfg)
    assert summary.recovered_gaps == 0


# ----------------------------------------------------------------------------------------------- timing / RTF
def test_rtf_is_none_when_nothing_was_decoded_and_sane_otherwise():
    assert safe_rtf(25.0, 0.0) is None and safe_rtf(25.0, 0.1) is None
    assert safe_rtf(30.0, 300.0) == pytest.approx(0.1)


def test_timing_report_separates_this_run_from_cumulative():
    earlier = [{"wall_seconds": 600.0, "decode_seconds": 500.0, "decoded_audio_seconds": 2000.0, "repair_seconds": 0.0,
                "repair_audio_seconds": 0.0}]
    now = SessionStats(wall_seconds=40.0, model_load_seconds=8.0, decode_seconds=0.0, decoded_audio_seconds=0.0,
                       repair_seconds=20.0, repair_audio_seconds=100.0)
    report = timing_report(audio_total=5241.0, this_run=now, earlier=earlier)
    assert report["audio_duration_total"] == 5241.0 and report["audio_decoded_this_run"] == 100.0
    assert report["audio_decoded_cumulative"] == 2100.0
    assert report["wall_time_this_run"] == 40.0 and report["wall_time_cumulative"] == 640.0
    assert report["model_load_time"] == 8.0 and report["repair_time"] == 20.0
    assert report["real_time_factor_this_run"] == pytest.approx(0.2)
    assert report["real_time_factor_effective"] == pytest.approx(520.0 / 2100.0, rel=1e-3)
    assert report["sessions"] == 2


def test_a_run_that_only_replays_checkpoints_has_no_rtf_at_all():
    now = SessionStats(wall_seconds=3.0)
    report = timing_report(audio_total=5241.0, this_run=now, earlier=[])
    assert report["real_time_factor_this_run"] is None and report["real_time_factor_effective"] is None
    assert report["real_time_factor"] is None


def test_metadata_carries_the_new_timing_keys_after_a_real_run(tmp_path, cfg, fake, monkeypatch):
    speech_everywhere(monkeypatch, 200)
    [summary] = process_files([media(tmp_path)], options(tmp_path), cfg)
    meta = json.loads((out_dir(tmp_path) / "metadata.json").read_text(encoding="utf-8"))
    timing = meta["timing"]
    for key in ("audio_duration_total", "audio_decoded_this_run", "audio_decoded_cumulative", "wall_time_this_run",
                "wall_time_cumulative", "model_load_time", "repair_time", "real_time_factor_this_run",
                "real_time_factor_effective"):
        assert key in timing, key
    assert 200 <= timing["audio_decoded_this_run"] <= 215       # the file plus the 3 s chunk overlaps, nothing more
    assert summary.real_time_factor == timing["real_time_factor_effective"]


def test_checkpoints_live_in_the_user_data_folder_and_are_removed_after_success(tmp_path, cfg, fake):
    src = media(tmp_path, 30)
    opts = options(tmp_path, keep_checkpoint=False, chunk_minutes=None)
    process_files([src], opts, cfg)
    assert not checkpoint_dir_for(opts, src).exists()
    assert not (out_dir(tmp_path) / "checkpoint").exists()
