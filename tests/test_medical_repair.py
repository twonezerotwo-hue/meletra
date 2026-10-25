"""Repetition and timestamp repair: decoder artefacts go, the lecturer's own repetition stays, times become consistent."""

from src.medical import repair as rp


def item(i, text, start, end, conf=0.9):
    return rp.Item(i, start, end, text, conf, "en", i)


# ----------------------------------------------------------------------------------------------------- repeated phrases inside a segment
def test_stuttered_phrases_collapse():
    assert rp.collapse_repeats("These fibers are together are together are embedded in the endoneurium.")[0] == "These fibers are together are embedded in the endoneurium."
    assert rp.collapse_repeats("the the neural tube")[0] == "the neural tube"
    assert rp.collapse_repeats("formation, formation of the neural tube") == ("formation of the neural tube", 1)
    assert rp.collapse_repeats("neural tube formation, neural tube formation is the first event")[0] == "neural tube formation is the first event"
    assert rp.collapse_repeats("and then and then and then it closes")[0] == "and then it closes"


def test_loops_of_three_or_more_collapse_whatever_the_length():
    assert rp.collapse_repeats("closure closure closure of the tube")[0] == "closure of the tube"


def test_emphasis_and_normal_text_are_kept():
    for text in ("This is very very important for the exam.", "The neural tube closes and the neural folds fuse.", "It is what it is, he said."):
        assert rp.collapse_repeats(text) == (text, 0)


# ----------------------------------------------------------------------------------------------------- duplicates
def test_an_exact_duplicate_that_follows_at_once_is_removed():
    kept, removed = rp.remove_duplicates([item(0, "The neural tube closes at day 25.", 0, 4), item(1, "The neural tube closes at day 25.", 4.2, 8)])
    assert [k.index for k in kept] == [0] and removed[0].index == 1 and "exact duplicate" in removed[0].reason


def test_an_overlapping_near_duplicate_is_removed_and_the_better_reading_kept():
    kept, removed = rp.remove_duplicates([item(0, "the neural folds fuse in the midline", 0, 4, 0.4), item(1, "The neural folds fuse in the midline.", 3.0, 7, 0.95)])
    assert len(kept) == 1 and kept[0].text.startswith("The neural folds") and len(removed) == 1


def test_a_sentence_the_lecturer_repeats_after_a_pause_stays():
    kept, removed = rp.remove_duplicates([item(0, "You have to know this table for the exam.", 0, 4), item(1, "You have to know this table for the exam.", 9.0, 13)])
    assert len(kept) == 2 and removed == []


def test_short_texts_are_never_treated_as_duplicates_by_similarity():
    kept, removed = rp.remove_duplicates([item(0, "Yes, so.", 0, 1), item(1, "Yes, so.", 1.1, 2)])
    assert removed and len(kept) == 1                                               # exact short repeats ARE artefacts ...
    kept, removed = rp.remove_duplicates([item(0, "Yes so now", 0, 1), item(1, "Yes so no", 1.1, 2)])
    assert len(kept) == 2                                                           # ... but similar short texts are not


def test_the_repeated_head_at_a_chunk_edge_is_trimmed():
    kept, removed = rp.remove_duplicates([item(0, "the cells of the ventricular zone divide and then they migrate outward", 0, 6),
                                          item(1, "and then they migrate outward to form the mantle layer", 5.0, 10)])
    assert len(kept) == 2 and kept[1].text == "to form the mantle layer" or kept[1].text.startswith("to form")
    assert "chunk edge" in " ".join(kept[1].notes)


# ----------------------------------------------------------------------------------------------------- timestamps
def test_overlapping_segments_are_resolved_and_order_is_monotonic():
    items = [item(0, "a b c", 0, 5), item(1, "d e f", 4.2, 9), item(2, "g h i", 8.0, 12)]
    fixed, repairs = rp.repair_timestamps(items, 60.0)
    assert rp.count_overlaps([(i.start, i.end) for i in fixed]) == 0 and repairs
    assert [i.start for i in fixed] == sorted(i.start for i in fixed)


def test_negative_and_reversed_durations_are_repaired():
    fixed, repairs = rp.repair_timestamps([item(0, "one two three four five six", 3.0, 2.0), item(1, "x y", -1.0, 4.0)], 60.0)
    assert all(f.end > f.start for f in fixed) and fixed[1].start >= 0 and len(repairs) >= 2


def test_a_start_before_the_previous_start_is_moved_forward():
    fixed, repairs = rp.repair_timestamps([item(0, "alpha beta", 10, 14), item(1, "gamma delta", 8, 12)], 60.0)
    assert fixed[1].start >= fixed[0].start and "before the previous start" in repairs[0].kind


def test_nothing_ends_after_the_recording():
    fixed, _ = rp.repair_timestamps([item(0, "alpha beta", 58, 70)], 60.0)
    assert fixed[0].end <= 60.5


def test_clean_times_are_untouched_and_the_measures_work():
    items = [item(0, "a b", 0, 2), item(1, "c d", 2, 4), item(2, "e f", 5, 6)]
    fixed, repairs = rp.repair_timestamps(items, 60.0)
    assert repairs == [] and [(f.start, f.end) for f in fixed] == [(0, 2), (2, 4), (5, 6)]
    assert rp.count_overlaps([(0, 5), (4, 8), (8, 9)]) == 1 and rp.count_reversed([(2, 1), (3, 4)]) == 1
