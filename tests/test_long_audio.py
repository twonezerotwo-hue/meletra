"""Chunk planning, ownership merge, timestamp offset correction and the per-chunk checkpoint store."""

import pytest

from src.long_audio import Chunk, ChunkStore, correct_offsets, own_segments, plan_chunks, stitch
from tests.conftest import make_segment


def test_short_recordings_are_a_single_chunk():
    [chunk] = plan_chunks(600.0, chunk_minutes=15)
    assert (chunk.start, chunk.end, chunk.own_start, chunk.own_end) == (0.0, 600.0, 0.0, 600.0)
    assert len(plan_chunks(15 * 60 * 1.2, chunk_minutes=15)) == 1                 # up to 1.25x is not split


def test_a_hundred_minute_lecture_is_cut_into_fifteen_minute_chunks_with_three_seconds_overlap():
    chunks = plan_chunks(100 * 60.0, chunk_minutes=15, overlap_seconds=3)
    assert len(chunks) == 7
    assert chunks[0].start == 0.0 and chunks[-1].end == 6000.0
    for before, after in zip(chunks, chunks[1:]):
        assert before.own_end == after.own_start                                  # no gap, no double ownership
        assert before.end == pytest.approx(before.own_end + 3.0)                  # decode windows reach into the neighbour
        assert after.start == pytest.approx(after.own_start - 3.0)
    assert all(abs((c.own_end - c.own_start) - 900) < 1 for c in chunks[:-1])


def test_cut_points_move_onto_the_longest_pause_near_the_nominal_cut():
    speech = [(0.0, 880.0), (884.0, 1790.0), (1800.0, 3000.0)]                    # 4 s pause around 882, 10 s around 1795
    chunks = plan_chunks(3000.0, chunk_minutes=15, speech=speech, snap_window_s=60)
    assert chunks[0].own_end == pytest.approx(882.0)                              # the 4 s pause near 900
    assert chunks[1].own_end == pytest.approx(1795.0, abs=1.0)                    # next nominal cut = 882 + 900 -> the 10 s pause


def test_without_vad_information_the_cuts_are_fixed():
    chunks = plan_chunks(3000.0, chunk_minutes=15, speech=None)
    assert [round(c.own_end) for c in chunks[:-1]] == [900, 1800, 2700]


def test_last_chunk_is_never_tiny():
    chunks = plan_chunks(15 * 60 * 2 + 20.0, chunk_minutes=15)
    assert all(c.own_end - c.own_start >= 90 for c in chunks)


def test_ownership_is_by_segment_midpoint_so_overlap_never_duplicates_text():
    first, second = Chunk(0, 0, 900, 0, 903), Chunk(1, 900, 1800, 897, 1800)
    seg_in_overlap = make_segment("Spoken across the cut.", 898.0, 902.0)          # midpoint 900.0 -> belongs to chunk 1
    early, late = make_segment("Early.", 100, 104), make_segment("Late.", 1000, 1004)
    assert own_segments(first, [early, seg_in_overlap], last=False) == [early]
    assert own_segments(second, [seg_in_overlap, late], last=True) == [seg_in_overlap, late]


def test_stitch_orders_chunks_and_drops_a_sentence_decoded_on_both_sides_of_a_cut():
    c0, c1 = Chunk(0, 0, 900, 0, 903), Chunk(1, 900, 1800, 897, 1800)
    sentence = "The corticospinal tract is the main descending motor pathway of the body."
    a = [make_segment("Opening words.", 10, 14), make_segment(sentence, 894.0, 899.5)]
    b = [make_segment(sentence, 900.5, 905.0), make_segment("Next topic starts here.", 910, 914)]
    merged, dropped = stitch([(c1, b), (c0, a)])                                     # given out of order on purpose
    assert [s.text for s in merged] == ["Opening words.", sentence, "Next topic starts here."]
    assert dropped == 1 and [s.start for s in merged] == sorted(s.start for s in merged)


def test_repeats_inside_one_chunk_are_kept_so_a_hallucination_loop_stays_visible():
    chunk = Chunk(0, 0, 100, 0, 100)
    loop = "The corticospinal tract descends through the internal capsule."
    kept, dropped = stitch([(chunk, [make_segment(loop, 10, 14), make_segment(loop, 14, 18), make_segment(loop, 18, 22)])])
    assert len(kept) == 3 and dropped == 0


def test_stitch_never_edits_text():
    c0 = Chunk(0, 0, 100, 0, 100)
    segs = [make_segment(" Raw  text  with   odd spacing. ", 1, 3)]
    [kept], _ = stitch([(c0, segs)])
    assert kept.text == " Raw  text  with   odd spacing. "


def test_window_relative_timestamps_are_shifted_to_absolute_time():
    chunk = Chunk(2, 1800, 2700, 1797, 2703)
    relative = [make_segment("a", 3.0, 9.0), make_segment("b", 20.0, 25.0)]
    assert correct_offsets(chunk, relative) is True
    assert (relative[0].start, relative[1].end) == (1800.0, 1822.0)


def test_absolute_timestamps_are_left_alone():
    chunk = Chunk(2, 1800, 2700, 1797, 2703)
    absolute = [make_segment("a", 1801.0, 1809.0), make_segment("b", 2000.0, 2005.0)]
    assert correct_offsets(chunk, absolute) is False and absolute[0].start == 1801.0
    assert correct_offsets(Chunk(0, 0, 900, 0, 903), [make_segment("x", 1, 2)]) is False


def test_chunk_store_reports_none_partial_and_completed(tmp_path):
    store = ChunkStore(tmp_path, {"source": "abc"})
    chunk = Chunk(0, 0, 100, 0, 100)
    assert store.status(chunk) == ("none", [])
    ck = store.checkpoint(chunk)
    ck.begin({"plan": 1}, [])
    ck.add(make_segment("first", 0, 5))
    ck.close()
    ck.mark({"plan": 1}, "running")
    status, saved = store.status(chunk)
    assert status == "partial" and [s.text for s in saved] == ["first"]
    ck.mark({"plan": 1}, "completed")
    assert store.status(chunk)[0] == "completed"


def test_chunk_checkpoint_is_ignored_when_the_chunk_boundaries_or_source_change(tmp_path):
    chunk = Chunk(0, 0, 100, 0, 100)
    ck = ChunkStore(tmp_path, {"source": "abc"}).checkpoint(chunk)
    ck.begin({}, [])
    ck.add(make_segment("x", 0, 5))
    ck.mark({}, "completed")
    ck.close()
    assert ChunkStore(tmp_path, {"source": "abc"}).status(chunk)[0] == "completed"
    assert ChunkStore(tmp_path, {"source": "different file"}).status(chunk)[0] == "none"
    assert ChunkStore(tmp_path, {"source": "abc"}).status(Chunk(0, 0, 120, 0, 120))[0] == "none"


# ----------------------------------------------------------------------------------------------- sentences cut at the window edge
def _cut_setup():
    c0, c1 = Chunk(0, 0, 960, 0, 963), Chunk(1, 960, 1800, 957, 1800)
    return c0, c1


def test_a_sentence_cut_off_at_the_end_of_a_window_is_replaced_by_its_complete_version():
    c0, c1 = _cut_setup()
    a = [make_segment("not be transparent.", 954.3, 955.1), make_segment("That's why when we have...", 955.6, 959.7)]
    b = [make_segment("That's why when we have the aging of this epithelium, they move towards the center of", 955.1, 968.1),
         make_segment("the lens where they have lost the organelles.", 968.1, 974.3)]
    merged, dropped = stitch([(c0, a), (c1, b)])
    assert [s.text for s in merged] == ["not be transparent.", "That's why when we have the aging of this epithelium, they move towards the center of",
                                        "the lens where they have lost the organelles."]
    assert dropped == 1


def test_a_cut_off_fragment_two_segments_back_is_found_too():
    c0, c1 = _cut_setup()
    a = [make_segment("matrix metal protease", 953.0, 958.0), make_segment("so-called", 958.0, 959.0)]
    b = [make_segment("matrix metal protease. What is special about this type of protease?", 954.0, 970.0)]
    merged, dropped = stitch([(c0, a), (c1, b)])
    texts = [x.text for x in merged]
    assert dropped == 1 and "matrix metal protease" not in texts and any(t.startswith("matrix metal protease. What is special") for t in texts)


def test_the_new_segment_is_dropped_when_it_is_only_a_piece_of_what_was_kept():
    c0, c1 = _cut_setup()
    a = [make_segment("protease inhibitors are small molecules that bind the active site", 952.0, 959.5)]
    b = [make_segment("protease inhibitors are small", 957.0, 965.0)]
    merged, dropped = stitch([(c0, a), (c1, b)])
    assert len(merged) == 1 and dropped == 1


def test_overlapping_but_different_sentences_are_both_kept():
    c0, c1 = _cut_setup()
    a = [make_segment("The first thing is the enzyme.", 950.0, 958.0)]
    b = [make_segment("And the second thing is the substrate.", 955.0, 968.0)]
    merged, dropped = stitch([(c0, a), (c1, b)])
    assert len(merged) == 2 and dropped == 0


def test_tiny_fragments_are_not_treated_as_cut_sentences():
    c0, c1 = _cut_setup()
    a = [make_segment("So", 955.0, 955.6)]
    b = [make_segment("So we continue with the next topic of the lecture.", 955.2, 966.0)]
    merged, dropped = stitch([(c0, a), (c1, b)])
    assert len(merged) == 2 and dropped == 0


def test_a_fragment_that_does_not_overlap_in_time_is_kept():
    c0, c1 = _cut_setup()
    a = [make_segment("That's why when we have", 940.0, 944.0)]
    b = [make_segment("That's why when we have the aging of this epithelium", 955.0, 966.0)]
    merged, dropped = stitch([(c0, a), (c1, b)])
    assert len(merged) == 2 and dropped == 0
