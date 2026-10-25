"""Long-recording production path through the real pipeline (fake ASR): chunks, per-chunk resume, retries,
partial recovery, chunk-to-chunk context adaptation, timestamp offset correction, safeguard on every request."""

import json

import pytest

from src.model_manager import ASRAdapter, Capabilities, DecodeRequest, ModelSpec
from src.pipeline import RunOptions, process_files
from src.schema import Segment, TranscriptionResult
from src.transcriber import ProgressReporter
from tests.conftest import make_segment, write_silent_wav

pytestmark = pytest.mark.usefixtures("fake_ffmpeg")

CHUNK_MINUTES = 1.0                                                       # smallest chunk the planner allows (60 s)


class WindowFake(ASRAdapter):
    """Honours start_offset / max_seconds like the real adapter: one 10 s segment per step, absolute timestamps."""

    backend_name = "fake"
    capabilities = Capabilities(hotwords=True, vad=True, word_timestamps=True, compute_types=True)
    requests: list[DecodeRequest] = []
    crash_at: dict[float, int] = {}                  # window start (rounded) -> crash after N segments, once
    relative_timestamps = False
    texts: dict[int, str] = {}                       # segment index in the lecture (t // 10) -> text
    loop_with_hotwords: set[int] = set()             # these segments loop only while hotwords are being sent
    loop_always: set[int] = set()                    # ... and these loop no matter what

    def __init__(self) -> None:
        super().__init__(ModelSpec("default", "faster_whisper", "large-v3"))
        self.closed = 0

    def load(self, compute_type: str) -> None:
        self.compute_type = compute_type

    def transcribe(self, request: DecodeRequest) -> TranscriptionResult:
        WindowFake.requests.append(request)
        total = request.total_duration
        begin = request.start_offset
        end = min(total, begin + request.max_seconds) if request.max_seconds else total
        out, made, t = [], 0, begin
        while t < end - 0.5:
            stop = min(t + 10.0, end)
            text = WindowFake.texts.get(int(t // 10), f" sentence at {int(t)}")
            looks_long = (request.max_seconds or 1e9) > 9.5 and int(t // 10) in WindowFake.loop_if_long
            if (request.hotwords and int(t // 10) in WindowFake.loop_with_hotwords) or int(t // 10) in WindowFake.loop_always                     or looks_long:
                text = " glycogen storage disease," * 8          # the hotword-echo loop seen on the real lecture
            shift = begin if WindowFake.relative_timestamps and begin > 0 else 0.0
            seg = Segment(t - shift, stop - shift, text, "en", "large-v3", "fake", avg_logprob=-0.2,
                          no_speech_prob=0.01, compression_ratio=1.1)
            out.append(seg)
            made += 1
            if request.on_segment and not WindowFake.relative_timestamps:
                request.on_segment(seg)
            limit = WindowFake.crash_at.get(round(begin))
            if limit is not None and made >= limit:
                WindowFake.crash_at.pop(round(begin))
                raise RuntimeError("simulated crash in the middle of a chunk")
            t += 10.0
        return TranscriptionResult(out, "en", 1.0, total, "large-v3", "fake", self.compute_type,
                                   hotwords_applied=list(request.hotwords[:5]), hotword_tokens=10)

    def close(self) -> None:
        self.closed += 1


@pytest.fixture
def fake(monkeypatch):
    WindowFake.requests, WindowFake.crash_at, WindowFake.loop_if_long = [], {}, set()
    WindowFake.relative_timestamps, WindowFake.texts = False, {}
    WindowFake.loop_with_hotwords, WindowFake.loop_always = set(), set()
    monkeypatch.setattr("src.transcriber.create_adapter", lambda *a, **k: WindowFake())
    monkeypatch.setattr("src.chunked_decode.time.sleep", lambda s: None, raising=False)


def options(tmp_path, **kw):
    base = dict(outputs_root=tmp_path / "out", language="en", quality_audit=False, auto_topic=False,
                chunk_minutes=CHUNK_MINUTES, subjects=["neuroscience"])
    base.update(kw)
    return RunOptions(**base)


def no_retry(cfg):
    return {**cfg, "long_audio": {**cfg["long_audio"], "chunk_retries": 0}}


def starts(requests):
    return [round(r.start_offset) for r in requests]


def test_a_five_minute_recording_is_decoded_in_chunks_and_stitched_in_order(tmp_path, cfg, fake):
    media = write_silent_wav(tmp_path / "lec.wav", seconds=300)
    [summary] = process_files([media], options(tmp_path), cfg)
    assert summary.ok and summary.chunks == 4
    assert starts(WindowFake.requests) == [0, 57, 117, 177]                  # each window reaches 3 s back
    out = tmp_path / "out" / "lec"
    data = json.loads((out / "local_transcript.json").read_text(encoding="utf-8"))
    begins = [s["start"] for s in data["segments"]]
    assert begins == sorted(begins) and begins[0] == 0 and data["duration"] == 300
    texts = [s["text"].strip() for s in data["segments"]]
    assert len(texts) == len(set(texts))                                          # the overlap produced no duplicate text
    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert meta["long_audio"]["chunks"] == 4 and len(meta["long_audio"]["per_chunk"]) == 4


def test_every_chunk_request_carries_the_hotword_safeguard(tmp_path, cfg, fake):
    media = write_silent_wav(tmp_path / "lec.wav", seconds=300)
    process_files([media], options(tmp_path), cfg)
    assert len(WindowFake.requests) == 4
    for request in WindowFake.requests:
        assert request.hotwords and request.profile["condition_on_previous_text"] is False
        assert request.hotword_token_budget and request.hotword_token_budget <= 223


def test_finished_chunks_are_never_decoded_again_after_a_crash_in_a_later_chunk(tmp_path, cfg, fake):
    media = write_silent_wav(tmp_path / "lec.wav", seconds=300)
    WindowFake.crash_at = {round(117): 2}                                          # chunk 3 dies after two segments
    with pytest.raises(RuntimeError, match="chunk 3"):
        process_files([media], options(tmp_path), no_retry(cfg))
    out = tmp_path / "out" / "lec"
    partial = (out / "partial_transcript.txt").read_text(encoding="utf-8")        # partial recovery: chunks 1-2 + 2 segments
    assert "sentence at 0" in partial and "sentence at 117" in partial and "sentence at 237" not in partial
    assert not (out / "local_raw_transcript.txt").exists()

    WindowFake.requests = []
    [summary] = process_files([media], options(tmp_path), no_retry(cfg))
    assert summary.ok and summary.resumed_segments > 0
    assert 0 not in starts(WindowFake.requests) and 57 not in starts(WindowFake.requests)    # chunks 1-2 not repeated
    assert starts(WindowFake.requests)[0] >= 117                                   # chunk 3 resumes after its saved segments
    raw = (out / "local_raw_transcript.txt").read_text(encoding="utf-8").splitlines()
    assert len(raw) == len(set(line.strip() for line in raw)) and raw[0].strip() == "sentence at 0"
    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert meta["resume"]["resumed_segments"] > 0 and not (out / "checkpoint").exists()


def test_a_failing_chunk_is_retried_with_a_fresh_model_and_the_run_completes(tmp_path, cfg, fake):
    media = write_silent_wav(tmp_path / "lec.wav", seconds=200)
    WindowFake.crash_at = {round(57): 1}
    [summary] = process_files([media], options(tmp_path), cfg)                    # default: 2 retries
    assert summary.ok
    meta = json.loads((tmp_path / "out" / "lec" / "metadata.json").read_text(encoding="utf-8"))
    attempts = [c["attempts"] for c in meta["long_audio"]["per_chunk"]]
    assert max(attempts) == 2 and attempts.count(1) == len(attempts) - 1


def test_a_chunk_that_keeps_failing_gives_up_cleanly_and_stays_resumable(tmp_path, cfg, fake, monkeypatch):
    media = write_silent_wav(tmp_path / "lec.wav", seconds=200)

    class Always(WindowFake):
        def transcribe(self, request):
            if round(request.start_offset) >= 57:
                raise RuntimeError("GPU fell off the bus")
            return super().transcribe(request)

    monkeypatch.setattr("src.transcriber.create_adapter", lambda *a, **k: Always())
    with pytest.raises(RuntimeError, match="failed after 3 attempt"):
        process_files([media], options(tmp_path), cfg)
    out = tmp_path / "out" / "lec"
    assert (out / "partial_transcript.txt").is_file() and (out / "checkpoint").is_dir()


def test_timestamps_are_corrected_when_a_backend_returns_window_relative_times(tmp_path, cfg, fake):
    media = write_silent_wav(tmp_path / "lec.wav", seconds=300)
    WindowFake.relative_timestamps = True
    process_files([media], options(tmp_path), cfg)
    data = json.loads((tmp_path / "out" / "lec" / "local_transcript.json").read_text(encoding="utf-8"))
    begins = [s["start"] for s in data["segments"]]
    assert begins == sorted(begins) and max(begins) > 180                         # not all squeezed into the first minute


def test_the_context_adapts_between_chunks_without_touching_the_transcript(tmp_path, cfg, fake):
    media = write_silent_wav(tmp_path / "lec.wav", seconds=200)
    WindowFake.texts = {1: " Now the periaqueductal gray and the red nucleus are discussed."}
    process_files([media], options(tmp_path), cfg)
    out = tmp_path / "out" / "lec"
    context = json.loads((out / "medical_context.json").read_text(encoding="utf-8"))
    updates = context["rolling_updates"]
    assert updates and updates[0]["chunk"] == 0
    assert {"periaqueductal gray", "red nucleus"} <= set(updates[0]["new_terms"])
    assert "periaqueductal gray" in WindowFake.requests[1].hotwords               # offered to the NEXT chunk
    raw = (out / "local_raw_transcript.txt").read_text(encoding="utf-8")
    assert "Now the periaqueductal gray and the red nucleus are discussed." in raw    # transcript text exactly as decoded


def test_progress_reporter_keeps_one_clock_across_chunks():
    reporter = ProgressReporter(step=5, interval_s=999, sticky=True)
    first_clock = reporter._started
    reporter.reset()                                                               # the transcriber calls this per chunk
    assert reporter._started == first_clock
    reporter.reset(force=True)
    assert reporter._started >= first_clock
    plain = ProgressReporter(step=5, interval_s=999)
    started = plain._started
    plain.reset()
    assert plain._started >= started


# ------------------------------------------------------------------------------------------- 100+ minutes
def write_long_silent_wav(path, minutes):
    """A 16 kHz mono PCM file of the given length, written block by block (no giant bytes object)."""
    import wave

    block = b"\x00\x00" * 16000 * 60
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        for _ in range(int(minutes)):
            handle.writeframes(block)
    return path


def test_a_105_minute_recording_is_chunked_crashes_in_chunk_five_and_resumes(tmp_path, cfg, fake, monkeypatch):
    media = write_long_silent_wav(tmp_path / "long.wav", 105)
    duration = 105 * 60.0
    pauses = [(0.0, 880.0), (884.0, 1790.0), (1796.0, 2700.0), (2704.0, duration)]
    monkeypatch.setattr("src.quality.compute_speech_ranges", lambda wav, vad: pauses)
    opts = lambda **kw: RunOptions(outputs_root=tmp_path / "out", language="en", quality_audit=False, auto_topic=False,  # noqa: E731
                                   subjects=["biochemistry"], **kw)
    WindowFake.crash_at = {}

    # find where chunk 5 starts, crash there once
    from src.long_audio import plan_chunks

    chunks = plan_chunks(duration, 15, 3, pauses)
    assert len(chunks) == 7 and chunks[0].own_end == pytest.approx(882.0)               # first cut snapped onto the pause
    WindowFake.crash_at = {round(chunks[4].start): 3}
    with pytest.raises(RuntimeError, match="chunk 5"):
        process_files([media], opts(), no_retry(cfg))
    out = tmp_path / "out" / "long"
    assert (out / "partial_transcript.txt").is_file() and not (out / "local_raw_transcript.txt").exists()

    WindowFake.requests = []
    [summary] = process_files([media], opts(), no_retry(cfg))
    assert summary.ok and summary.chunks == 7 and summary.resumed_segments > 0
    redone = [round(r.start_offset) for r in WindowFake.requests]
    assert len(redone) == 3 and redone[0] >= round(chunks[4].start)                  # chunks 5 (rest), 6 and 7 only
    data = json.loads((out / "local_transcript.json").read_text(encoding="utf-8"))
    begins = [s["start"] for s in data["segments"]]
    assert begins == sorted(begins) and data["duration"] == duration and begins[-1] > duration - 30
    assert len({s["text"] for s in data["segments"]}) == len(data["segments"])


# ------------------------------------------------------------------------------------------- loop repair (real-lecture bug)
def read_raw(tmp_path):
    return (tmp_path / "out" / "lec" / "local_raw_transcript.txt").read_text(encoding="utf-8")


def test_a_hotword_echo_loop_is_redecoded_without_hotwords_and_only_that_window(tmp_path, cfg, fake):
    media = write_silent_wav(tmp_path / "lec.wav", seconds=300)
    WindowFake.loop_with_hotwords = {12}                                           # t = 127-137 s (chunk 3 window grid)
    [summary] = process_files([media], options(tmp_path), cfg)
    raw = read_raw(tmp_path)
    assert summary.ok and "glycogen storage disease" not in raw and "sentence at 127" in raw
    plain = [r for r in WindowFake.requests if not r.hotwords]
    assert len(plain) == 1 and round(plain[0].start_offset) == 127 and plain[0].max_seconds == pytest.approx(10.0, abs=0.5)
    assert plain[0].profile["condition_on_previous_text"] is False
    meta = json.loads((tmp_path / "out" / "lec" / "metadata.json").read_text(encoding="utf-8"))
    [repair] = meta["long_audio"]["loop_repairs"]
    assert repair["repaired"] is True and repair["start"] == 127 and "glycogen storage disease" in repair["replaced_text"]
    assert repair["new_text"].strip() == "sentence at 127"


def test_the_original_is_kept_and_stays_flagged_when_the_redecode_loops_too(tmp_path, cfg, fake):
    media = write_silent_wav(tmp_path / "lec.wav", seconds=300)
    WindowFake.loop_always = {12}
    process_files([media], options(tmp_path), cfg)
    assert "glycogen storage disease, glycogen storage disease" in read_raw(tmp_path)
    meta = json.loads((tmp_path / "out" / "lec" / "metadata.json").read_text(encoding="utf-8"))
    assert meta["long_audio"]["loop_repairs"][0]["repaired"] is False
    codes = {r["code"] for e in json.loads((tmp_path / "out" / "lec" / "uncertain_segments.json").read_text(encoding="utf-8"))
             for r in e["reasons"]}
    assert "repetition" in codes


def test_without_hotwords_a_loop_is_retried_in_short_pieces_not_by_repeating_the_same_decode(tmp_path, cfg, fake):
    media = write_silent_wav(tmp_path / "lec.wav", seconds=300)
    WindowFake.loop_always = {12}                                  # a window that loops however it is decoded
    process_files([media], options(tmp_path, use_hotwords=False), cfg)
    pieces = [r for r in WindowFake.requests if r.max_seconds and r.max_seconds < 12]
    assert len(WindowFake.requests) == 4 + len(pieces) and 1 <= len(pieces) <= 3        # the four chunks + short pieces only
    assert all(not r.hotwords and r.profile["condition_on_previous_text"] is False for r in pieces)
    meta = json.loads((tmp_path / "out" / "lec" / "metadata.json").read_text(encoding="utf-8"))
    [record] = meta["long_audio"]["loop_repairs"]
    assert record["repaired"] is False and "short pieces" in record["note"]               # the original stays, flagged


def test_a_loop_that_only_happens_on_long_windows_is_repaired_in_pieces_without_hotwords(tmp_path, cfg, fake):
    media = write_silent_wav(tmp_path / "lec.wav", seconds=300)
    WindowFake.loop_if_long = {12}                                 # clean when decoded in windows of <= 9 s
    WindowFake.texts = {12: " the first piece holds plain words", 13: " another clean sentence follows here"}
    short = {**cfg, "long_audio": {**cfg["long_audio"], "loop_piece_seconds": 5.0}}      # pieces of 5 s + padding = 7 s windows
    process_files([media], options(tmp_path, use_hotwords=False), short)
    meta = json.loads((tmp_path / "out" / "lec" / "metadata.json").read_text(encoding="utf-8"))
    [record] = meta["long_audio"]["loop_repairs"]
    assert record["repaired"] is True and "plain words" in record["new_text"]
    assert "glycogen storage disease" not in read_raw(tmp_path)


def test_with_hotwords_a_window_that_still_loops_falls_back_to_pieces(tmp_path, cfg, fake):
    media = write_silent_wav(tmp_path / "lec.wav", seconds=300)
    WindowFake.loop_with_hotwords = {12}
    WindowFake.loop_if_long = {12}
    WindowFake.texts = {12: " the first piece holds plain words", 13: " another clean sentence follows here"}
    short = {**cfg, "long_audio": {**cfg["long_audio"], "loop_piece_seconds": 5.0}}
    process_files([media], options(tmp_path), short)
    meta = json.loads((tmp_path / "out" / "lec" / "metadata.json").read_text(encoding="utf-8"))
    [record] = meta["long_audio"]["loop_repairs"]
    assert record["repaired"] is True
    plain = [r for r in WindowFake.requests if not r.hotwords]
    assert len(plain) >= 3                                         # the whole-window attempt + at least two short pieces


def test_repair_can_be_switched_off(tmp_path, cfg, fake):
    media = write_silent_wav(tmp_path / "lec.wav", seconds=300)
    WindowFake.loop_with_hotwords = {12}
    off = {**cfg, "long_audio": {**cfg["long_audio"], "loop_repair": False}}
    process_files([media], options(tmp_path), off)
    assert "glycogen storage disease" in read_raw(tmp_path)


def test_a_finished_chunk_from_an_earlier_run_is_repaired_once_and_the_fix_is_saved(tmp_path, cfg, fake):
    """The real recording: the loops were found AFTER the run, in chunks that only exist as checkpoints."""
    media = write_silent_wav(tmp_path / "lec.wav", seconds=300)
    WindowFake.loop_with_hotwords = {12}
    off = {**cfg, "long_audio": {**cfg["long_audio"], "loop_repair": False}}
    process_files([media], options(tmp_path, keep_checkpoint=True), off)           # first run keeps the loop + the checkpoint
    assert "glycogen storage disease" in read_raw(tmp_path)

    WindowFake.requests = []
    process_files([media], options(tmp_path, keep_checkpoint=True), cfg)           # same command, repair now on
    assert "glycogen storage disease" not in read_raw(tmp_path)
    assert len(WindowFake.requests) == 1 and not WindowFake.requests[0].hotwords    # only the looped window was decoded

    WindowFake.requests = []
    process_files([media], options(tmp_path, keep_checkpoint=True), cfg)           # and never again
    assert WindowFake.requests == [] and "sentence at 127" in read_raw(tmp_path)


def test_loop_detector_catches_the_patterns_seen_on_the_real_lecture():
    from src.chunked_decode import is_decoder_loop, phrase_loops

    hot = {"acetyl", "coa", "carboxylase", "glycogen", "dehydrogenase", "storage", "disease"}
    assert phrase_loops("glycogenase, acetyl CoA carboxylase, acetyl CoA carboxylase, acetyl CoA carboxylase,")
    assert not phrase_loops("No, no, no, that is not what I said.")                       # natural repetition stays alone
    assert not phrase_loops("The enzyme acetyl CoA carboxylase makes malonyl CoA. Acetyl CoA carboxylase is regulated.")

    def seg(text, lp=-0.2):
        return make_segment(text, 0, 5, avg_logprob=lp)

    assert is_decoder_loop(seg("glycogen sulfate dehydrogenase", lp=-1.06), hot, 4, 4)       # blend of hotwords, unsure
    assert not is_decoder_loop(seg("glycogen sulfate dehydrogenase", lp=-0.2), hot, 4, 4)    # confident = leave it alone
    assert not is_decoder_loop(seg("We study glycogen today in the lecture hall", lp=-1.2), hot, 4, 4)
    assert is_decoder_loop(seg("polysaccharides, polysaccharides, polysaccharides, polysaccharides"), hot, 4, 4)


# ----------------------------------------------------------------------------------------------- loops spread over many segments
def _seg(start, end, text):
    return Segment(start, end, text, "en", "m", "b")


def test_a_short_phrase_repeated_many_times_is_a_run():
    from src.chunked_decode import repeated_runs

    segs = [_seg(i * 2, i * 2 + 2, " This is the lysosome.") for i in range(20)]
    assert repeated_runs(segs) == [(0, 19)]


def test_one_long_sentence_coming_back_is_a_run_after_two_times():
    from src.chunked_decode import repeated_runs

    long = " Köszönöm, ha lehetőszerű, vagyok a Szent György, és nem vagyok biztos, hogy köszönöm a dolgunkban bennetek."
    assert repeated_runs([_seg(0, 30, long), _seg(30, 60, long), _seg(60, 90, " Something else entirely.")]) == [(0, 1)]


def test_natural_repetition_is_not_a_run():
    from src.chunked_decode import repeated_runs

    segs = [_seg(0, 2, " Okay."), _seg(2, 4, " Okay."), _seg(4, 9, " So the enzyme is activated here."),
            _seg(9, 14, " So the enzyme is inhibited there."), _seg(14, 19, " Table 1 shows the results."),
            _seg(19, 24, " Table 2 shows the other results."), _seg(24, 29, " Pyridoxal."), _seg(29, 30, " Pyridoxal.")]
    assert repeated_runs(segs) == []


def test_sentences_that_differ_only_by_a_number_are_not_a_run():
    from src.chunked_decode import repeated_runs

    segs = [_seg(i * 5, i * 5 + 5, f" sentence at {i * 5}") for i in range(8)]
    assert repeated_runs(segs) == []


def test_two_short_repeats_are_not_enough():
    from src.chunked_decode import repeated_runs

    assert repeated_runs([_seg(0, 2, " This is the lysosome."), _seg(2, 4, " This is the lysosome."), _seg(4, 8, " And more.")]) == []


class RunFake(WindowFake):
    """The decoder spreads a loop over several 10 s segments in the main decode; short pieces decode cleanly."""

    run_slots: set[int] = set()

    def transcribe(self, request):
        result = super().transcribe(request)
        long_window = (request.max_seconds or 1e9) > 9.5
        if long_window:
            for seg in result.segments:
                if int(seg.start // 10) in RunFake.run_slots:
                    seg.text = " This is the lysosome and its enzymes."
        return result


@pytest.fixture
def run_fake(monkeypatch, fake):
    RunFake.requests = WindowFake.requests
    RunFake.run_slots = set()
    monkeypatch.setattr("src.transcriber.create_adapter", lambda *a, **k: RunFake())
    return RunFake


def test_a_run_of_repeated_segments_is_repaired_as_one_window_in_pieces(tmp_path, cfg, run_fake):
    media = write_silent_wav(tmp_path / "lec.wav", seconds=300)
    run_fake.run_slots = {12, 13, 14, 15}                          # four segments in a row repeat the same sentence
    WindowFake.texts = {12: " first clean piece of speech", 13: " second clean piece of speech", 14: " third clean piece of speech",
                     15: " fourth clean piece of speech", 16: " fifth clean piece of speech"}
    short = {**cfg, "long_audio": {**cfg["long_audio"], "loop_piece_seconds": 5.0}}
    process_files([media], options(tmp_path, use_hotwords=False), short)
    meta = json.loads((tmp_path / "out" / "lec" / "metadata.json").read_text(encoding="utf-8"))
    records = meta["long_audio"]["loop_repairs"]
    assert len(records) == 1 and records[0]["repaired"] is True and records[0]["segments"] >= 3       # ONE window, not four
    raw = read_raw(tmp_path)
    assert "This is the lysosome" not in raw and "clean piece of speech" in raw


def test_a_repair_that_is_no_better_keeps_the_original(tmp_path, cfg, run_fake):
    media = write_silent_wav(tmp_path / "lec.wav", seconds=300)
    run_fake.run_slots = {12, 13, 14, 15}
    WindowFake.loop_always = {10, 11, 12, 13, 14, 15, 16, 17}          # the pieces loop as well (every window of the stretch)
    process_files([media], options(tmp_path, use_hotwords=False), cfg)
    meta = json.loads((tmp_path / "out" / "lec" / "metadata.json").read_text(encoding="utf-8"))
    assert all(r["repaired"] is False for r in meta["long_audio"]["loop_repairs"])
    assert "glycogen storage disease" in read_raw(tmp_path)         # the original text is still in the raw transcript


# ----------------------------------------------------------------------------------------------- repaired segments stay tidy
def test_clamp_words_keeps_words_inside_a_clipped_segment():
    from src.chunked_decode import clamp_words
    from src.schema import Word

    seg = _seg(10.0, 14.0, " a b c")
    seg.words = [Word(9.5, 10.5, " a", 0.9), Word(11.0, 12.0, " b", 0.9), Word(13.5, 15.0, " c", 0.9)]
    clamp_words(seg)
    assert [(w.start, w.end) for w in seg.words] == [(10.0, 10.5), (11.0, 12.0), (13.5, 14.0)]


def test_overlap_fraction():
    from src.chunked_decode import overlap_fraction

    seg = _seg(10.0, 20.0, " text here")
    assert overlap_fraction(seg, []) == 0.0
    assert overlap_fraction(seg, [_seg(0, 12, " x y"), _seg(15, 30, " z w")]) == pytest.approx(0.7)
    assert overlap_fraction(seg, [_seg(0, 30, " ")]) == 0.0                     # a segment without text covers nothing
    assert overlap_fraction(seg, [_seg(9, 21, " all"), _seg(12, 14, " inside")]) == 1.0


def test_a_recovered_piece_that_a_neighbour_already_covers_is_rejected():
    from src.chunked_decode import ChunkedDecoder

    existing = [_seg(955.1, 968.1, " That's why when we have the aging of this epithelium")]
    fresh = [_seg(955.6, 959.7, " That's why when we have")]
    assert ChunkedDecoder._accept_gap_segments(fresh, 953.0, 961.0, existing) == []
    elsewhere = [_seg(970.0, 975.0, " a different sentence entirely here")]
    assert len(ChunkedDecoder._accept_gap_segments(elsewhere, 968.0, 976.0, existing)) == 1
