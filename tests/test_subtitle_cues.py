"""Subtitle files: no cue stays on screen for 30 seconds or fills eight lines (measured on a real recording); text and the
raw transcript stay exactly as the model returned them."""

import textwrap

from src.exporters import (
    CUE_MAX_CHARS, CUE_MAX_SECONDS, SRT_LINE_WIDTH, cue_parts, segments_to_srt, segments_to_vtt, split_cue_by_words, subtitle_cues,
)
from src.quality import is_stock_phrase
from src.schema import Segment, Word


def timed_segment(text: str, start: float = 0.0, per_word: float = 0.4, gaps: dict[int, float] | None = None) -> Segment:
    words, t = [], start
    for i, token in enumerate(text.split()):
        t += (gaps or {}).get(i, 0.0)
        words.append(Word(t, t + per_word, " " + token, 0.9))
        t += per_word
    return Segment(start, t, " " + text, "en", "m", "b", words=words)


LONG = ("In glycolysis, glucose is converted to pyruvate. Hexokinase traps glucose inside the cell, and phosphofructokinase is the "
        "main rate limiting enzyme, inhibited by citrate and activated by fructose bisphosphate. Pyruvate dehydrogenase links "
        "glycolysis to the citric acid cycle, and its deficiency causes lactic acidosis.")


def test_a_long_segment_becomes_readable_cues():
    seg = timed_segment(LONG)
    assert seg.end - seg.start > 15
    cues = subtitle_cues([seg])
    assert len(cues) >= 3
    for start, end, text in cues:
        assert end - start <= CUE_MAX_SECONDS + 0.5 and len(text) <= CUE_MAX_CHARS
        assert len(textwrap.wrap(text, SRT_LINE_WIDTH, break_long_words=False)) <= 2
    assert " ".join(t for _s, _e, t in cues) == LONG                        # nothing lost, nothing reordered
    assert cues == sorted(cues)


def test_cues_break_after_sentences_when_possible():
    cues = subtitle_cues([timed_segment(LONG)])
    assert cues[0][2].endswith("pyruvate.")


def test_a_short_segment_is_one_cue_exactly_as_before():
    seg = timed_segment("Short sentence here.")
    assert cue_parts(seg) == [(seg.start, seg.end, "Short sentence here.")]


def test_a_segment_without_word_times_is_left_alone():
    seg = Segment(0.0, 30.0, " " + LONG, "en", "m", "b")
    assert subtitle_cues([seg]) == [(0.0, 30.0, LONG)]


def test_long_silences_still_split_first():
    seg = timed_segment("one two three four five six seven eight", gaps={4: 4.0})
    cues = subtitle_cues([seg])
    assert [t for _s, _e, t in cues] == ["one two three four", "five six seven eight"]


def test_the_split_never_produces_empty_or_overlapping_cues():
    cues = subtitle_cues([timed_segment(LONG, start=100.0)])
    assert all(t.strip() for _s, _e, t in cues)
    assert all(a[1] <= b[0] + 1e-6 for a, b in zip(cues, cues[1:]))
    assert cues[0][0] == 100.0 and cues[-1][1] > 110


def test_srt_and_vtt_use_the_split_cues_and_keep_two_lines_at_most():
    srt = segments_to_srt([timed_segment(LONG)])
    blocks = [b for b in srt.strip().split("\n\n")]
    assert len(blocks) >= 3 and all(len(b.splitlines()) <= 4 for b in blocks)
    assert segments_to_vtt([timed_segment(LONG)]).startswith("WEBVTT")


def _timed(text):
    return [Word(i * 0.5, i * 0.5 + 0.4, " " + w, 0.9) for i, w in enumerate(text.split())]


def test_split_helper_prefers_sentence_then_comma_then_anywhere():
    sentence = _timed("a b c d e f g h. i j k l m n o p q r s t u v w x y z")
    assert split_cue_by_words(sentence, max_seconds=5.0, max_chars=200)[0][-1].word.strip() == "h."
    comma = _timed("a b c d e f g, h i j k l m n o p q r s t u v w x y z")
    assert split_cue_by_words(comma, max_seconds=5.0, max_chars=200)[0][-1].word.strip() == "g,"
    plain = _timed("a b c d e f g h i j k l m n o p q r s t u v w x y z")
    groups = split_cue_by_words(plain, max_seconds=5.0, max_chars=200)
    assert [w for g in groups for w in g] == plain and all(g[-1].end - g[0].start <= 5.0 for g in groups)


def test_one_endless_word_stream_without_punctuation_is_still_cut():
    words = [Word(i * 0.3, i * 0.3 + 0.25, " word", 0.9) for i in range(60)]
    groups = split_cue_by_words(words, max_seconds=7.0, max_chars=84)
    assert len(groups) >= 3 and [w for g in groups for w in g] == words


# ----------------------------------------------------------------------------------------------- stock phrases
def test_stock_phrases_in_several_languages_are_recognised():
    for text in ("Köszönöm, hogy megnézted!", "Köszönöm, hogy megnéztétek.", "Thanks for watching!", "Thank you for watching.",
                 "Vielen Dank fürs Zuschauen.", "Gracias por ver", "Altyazı M.K."):
        assert is_stock_phrase(text), text


def test_real_speech_that_merely_contains_a_thank_you_is_not_a_stock_phrase():
    assert not is_stock_phrase("And with that I want to say thanks for watching the demonstration of the enzyme assay that we prepared.")
    assert not is_stock_phrase("Hexokinase traps glucose inside the cell.")
