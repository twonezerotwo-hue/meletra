import json

from src.checkpoint import SegmentCheckpoint, file_sha256, make_fingerprint
from src.schema import Segment, Word
from tests.conftest import make_segment

FP = make_fingerprint("abc123", model="large-v3", mode="quality")


def seg(i: int) -> Segment:
    s = make_segment(f" sentence number {i}", i * 10.0, i * 10.0 + 9.5)
    s.words = [Word(i * 10.0, i * 10.0 + 0.4, " sentence", 0.97)]
    return s


def test_segment_checkpoint_roundtrip_is_lossless():
    original = seg(3)
    original.start, original.end = 30.123456789, 39.987654321
    restored = Segment.from_checkpoint(json.loads(json.dumps(original.to_checkpoint())))
    assert restored.start == original.start and restored.end == original.end
    assert restored.text == " sentence number 3"               # leading space kept: raw text is never altered
    assert restored.words[0].word == " sentence" and restored.words[0].probability == 0.97
    assert restored.avg_logprob == original.avg_logprob


def test_saved_segments_and_plan_come_back(tmp_path):
    ckpt = SegmentCheckpoint(tmp_path / "ck", FP)
    ckpt.begin({"language": "en", "subjects": ["biochemistry"]}, [])
    for i in range(4):
        ckpt.add(seg(i))
    ckpt.close()
    segments, plan = SegmentCheckpoint(tmp_path / "ck", FP).load()
    assert [s.text.strip() for s in segments] == [f"sentence number {i}" for i in range(4)]
    assert plan == {"language": "en", "subjects": ["biochemistry"]}


def test_checkpoint_from_other_settings_or_source_is_ignored(tmp_path):
    ckpt = SegmentCheckpoint(tmp_path / "ck", FP)
    ckpt.begin({"x": 1}, [])
    ckpt.add(seg(0))
    ckpt.close()
    other_source = make_fingerprint("DIFFERENT", model="large-v3", mode="quality")
    other_mode = make_fingerprint("abc123", model="large-v3", mode="fast")
    assert SegmentCheckpoint(tmp_path / "ck", other_source).load() == ([], None)
    assert SegmentCheckpoint(tmp_path / "ck", other_mode).load() == ([], None)


def test_torn_last_line_after_a_crash_is_dropped_not_fatal(tmp_path):
    ckpt = SegmentCheckpoint(tmp_path / "ck", FP)
    ckpt.begin({}, [])
    ckpt.add(seg(0))
    ckpt.add(seg(1))
    ckpt.close()
    with (tmp_path / "ck" / "partial_segments.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"start": 20.0, "end": 29.5, "text": " half writt')       # power cut mid-write
    segments, _ = SegmentCheckpoint(tmp_path / "ck", FP).load()
    assert len(segments) == 2


def test_rollback_drops_only_the_failed_attempt(tmp_path):
    resumed = [seg(0), seg(1)]
    ckpt = SegmentCheckpoint(tmp_path / "ck", FP)
    ckpt.begin({}, resumed)
    ckpt.add(seg(2))
    ckpt.add(seg(3))
    ckpt.rollback()                      # e.g. float16 -> int8_float16 fallback restarts the decode
    ckpt.add(seg(2))
    ckpt.close()
    segments, _ = SegmentCheckpoint(tmp_path / "ck", FP).load()
    assert [s.text.strip() for s in segments] == [f"sentence number {i}" for i in (0, 1, 2)]


def test_plan_only_checkpoint_has_no_segments_and_discard_cleans_up(tmp_path):
    ckpt = SegmentCheckpoint(tmp_path / "ck", FP)
    ckpt.save_plan({"language": "tr"})
    segments, plan = SegmentCheckpoint(tmp_path / "ck", FP).load()
    assert segments == [] and plan == {"language": "tr"}
    ckpt.discard()
    assert not (tmp_path / "ck").exists()


def test_file_sha256_matches_hashlib(tmp_path):
    import hashlib

    f = tmp_path / "x.bin"
    f.write_bytes(b"medical lecture" * 100000)
    assert file_sha256(f) == hashlib.sha256(f.read_bytes()).hexdigest()
