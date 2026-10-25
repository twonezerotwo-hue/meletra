"""Decoding by language runs with a fake one-language decoder: each stretch gets its own language, labels and the instrumental guard work, nothing is dropped for being another language."""

from __future__ import annotations

import dataclasses

from src.model_manager import DecodeRequest
from src.multilingual import decode as md
from src.schema import Segment, TranscriptionResult


def request(plan, offset=0.0, seconds=None, total=100.0):
    return DecodeRequest(audio_path="x.wav", language=None, profile={}, decoding={}, vad_parameters={}, start_offset=offset, max_seconds=seconds, total_duration=total,
                         language_plan=plan)


class FakeDecoder:
    """Returns one segment per call, written in the script of the language it was asked for; records the requests."""

    TEXT = {"ko": "안녕하세요 여러분", "hi": "नमस्ते दोस्तों", "en": "hello everyone today we start"}

    def __init__(self, extra=None):
        self.calls: list[tuple[float, float, str, bool]] = []
        self.extra = extra or {}

    def __call__(self, req):
        self.calls.append((req.start_offset, req.start_offset + req.max_seconds, req.language, req.vad_override))
        seg = Segment(req.start_offset + 0.5, req.start_offset + req.max_seconds - 0.5, self.TEXT[req.language], avg_logprob=-0.3, no_speech_prob=0.1, compression_ratio=1.2)
        extra = self.extra.get(round(req.start_offset), [])
        return TranscriptionResult(segments=[seg, *extra], language=req.language, language_probability=None, duration=req.start_offset + req.max_seconds, model="m", backend="b",
                                   load_seconds=0.0)


PLAN = [(0.0, 30.0, "ko", "on"), (30.0, 60.0, "hi", "on"), (60.0, 100.0, "en", "on")]


def test_every_run_is_decoded_in_its_own_language_and_labelled():
    inner = FakeDecoder()
    result = md.transcribe_runs(inner, request(PLAN))
    assert [(c[2]) for c in inner.calls] == ["ko", "hi", "en"]
    assert [s.language for s in result.segments] == ["ko", "hi", "en"]
    assert all(s.language_meta["source"] == "detector" for s in result.segments)
    assert result.language == "en"                                                                        # the language with the most audio, as the result's summary


def test_a_resumed_request_decodes_only_the_runs_after_its_offset():
    inner = FakeDecoder()
    md.transcribe_runs(inner, request(PLAN, offset=45.0, seconds=30.0))
    assert [(round(a), round(b), l) for a, b, l, _v in inner.calls] == [(45, 60, "hi"), (60, 75, "en")]


def test_vad_off_runs_are_decoded_without_the_filter_and_instrumental_output_is_rejected():
    junk = [Segment(5.0, 9.0, "Thank you for watching!", avg_logprob=-0.4, no_speech_prob=0.3),
            Segment(10.0, 14.0, "♪", avg_logprob=-0.9, no_speech_prob=0.5),
            Segment(15.0, 20.0, "la la la la la la la la la la la la", avg_logprob=-0.5, no_speech_prob=0.2),
            Segment(21.0, 25.0, "수고했어요 우리 같이 가요", avg_logprob=-0.4, no_speech_prob=0.2)]
    inner = FakeDecoder(extra={0: junk})
    result = md.transcribe_runs(inner, request([(0.0, 40.0, "ko", "off")]))
    assert inner.calls[0][3] is False                                                                      # the filter was switched off for this stretch
    kept = [s.text for s in result.segments]
    assert "수고했어요 우리 같이 가요" in kept and "안녕하세요 여러분" in kept
    assert not any("Thank you" in t or "la la la" in t or t.strip() == "♪" for t in kept)
    assert len(result.rejected) == 3 and all(r["reason"] for r in result.rejected)                           # rejected stretches are recorded, never silently lost
    assert any("not kept as speech" in n for n in result.notes)


def test_vad_on_runs_keep_everything_the_decoder_returns():
    odd = [Segment(5.0, 9.0, "Thank you for watching!", avg_logprob=-0.4, no_speech_prob=0.3)]
    result = md.transcribe_runs(FakeDecoder(extra={0: odd}), request([(0.0, 40.0, "en", "on")]))
    assert len(result.segments) == 2 and result.rejected == []


def test_a_segment_whose_writing_contradicts_the_run_is_relabelled_by_its_script():
    wrong = [Segment(5.0, 9.0, "여기는 한국어 문장입니다", avg_logprob=-0.4, no_speech_prob=0.1)]
    inner = FakeDecoder(extra={0: wrong})
    result = md.transcribe_runs(inner, request([(0.0, 40.0, "en", "on")]))
    korean = next(s for s in result.segments if "한국어" in s.text)
    assert korean.language == "ko" and korean.language_meta["source"] == "script" and korean.language_meta["candidates"] == ["en"]    # the writing proves it; the text is kept


def test_a_low_confidence_segment_is_flagged_not_dropped():
    shaky = [Segment(5.0, 9.0, "정말 불확실한 문장", avg_logprob=-1.4, no_speech_prob=0.2)]
    result = md.transcribe_runs(FakeDecoder(extra={0: shaky}), request([(0.0, 40.0, "ko", "on")]))
    segment = next(s for s in result.segments if "불확실" in s.text)
    assert segment.language_meta.get("status") == "low_confidence"


def test_run_metadata_is_copied_onto_the_segments():
    meta = [{"start": 30.0, "end": 60.0, "language": "hi", "confidence": 0.81, "status": "confident", "candidates": ["mr"], "source": "detector"}]
    result = md.transcribe_runs(FakeDecoder(), request(PLAN), meta)
    hindi = next(s for s in result.segments if s.language == "hi")
    assert hindi.language_meta["confidence"] == 0.81 and hindi.language_meta["candidates"] == ["mr"]
    korean = next(s for s in result.segments if s.language == "ko")
    assert korean.language_meta["confidence"] is None                                                       # never invented when the plan had none


def test_checkpoint_round_trip_keeps_the_language_label():
    seg = Segment(1.0, 2.0, "text", language="ko", language_meta={"source": "detector", "confidence": 0.9})
    again = Segment.from_checkpoint(seg.to_checkpoint())
    assert again.language == "ko" and again.language_meta == {"source": "detector", "confidence": 0.9}


class BoundaryDecoder:
    """Hindi audio up to 33 s, Korean after; the plan puts the change at 30 s, so the Hindi sentence at 30-33 s is decoded as Korean (garbage, low score)."""

    def __init__(self):
        self.calls = []

    def __call__(self, req):
        self.calls.append((round(req.start_offset, 1), round(req.start_offset + req.max_seconds, 1), req.language))
        lo, hi = req.start_offset, req.start_offset + req.max_seconds
        segments = []
        t = lo
        while t < hi - 0.1:
            end = min(t + 3.0, hi)
            truth = "hi" if t < 33.0 else "ko"
            right = req.language == truth
            text = {"hi": "नमस्ते दोस्तों आज", "ko": "안녕하세요 여러분 오늘은"}[req.language if right else "ko"] if right else "아"
            segments.append(Segment(t, end, text, avg_logprob=-0.3 if right else -1.5, no_speech_prob=0.05, compression_ratio=1.2))
            t = end
        return TranscriptionResult(segments=segments, language=req.language, language_probability=None, duration=hi, model="m", backend="b", load_seconds=0.0)


def test_a_doubtful_segment_next_to_a_change_of_language_is_read_again_in_the_neighbouring_language():
    inner = BoundaryDecoder()
    result = md.transcribe_runs(inner, request([(0.0, 30.0, "hi", "on"), (30.0, 60.0, "ko", "on")]))
    around = [(s.start, s.language, s.text) for s in result.segments if 27.0 <= s.start < 36.0]
    assert [l for _s, l, _t in around if _s < 33.0] == ["hi", "hi"] or all(l == "hi" for _s, l, _t in around if _s < 33.0)       # the Hindi sentence is Hindi again
    wrong = [s for s in result.segments if s.start < 33.0 and s.language == "ko"]
    assert wrong == [] and any("re-read" in n or "read again" in n for n in result.notes)
    assert [s.language for s in result.segments if s.start >= 36.0] == ["ko"] * len([s for s in result.segments if s.start >= 36.0])


def test_confident_segments_are_never_re_read():
    inner = FakeDecoder()
    md.transcribe_runs(inner, request(PLAN))
    assert len(inner.calls) == 3                                                                          # nothing was decoded twice: every segment scored well


def test_a_re_read_that_is_not_clearly_better_is_not_taken():
    class Flat(FakeDecoder):
        def __call__(self, req):
            result = super().__call__(req)
            for s in result.segments:
                s.avg_logprob = -1.3                                                                      # poor in both languages: no evidence for the neighbour
            return result

    result = md.transcribe_runs(Flat(), request([(0.0, 30.0, "ko", "on"), (30.0, 60.0, "hi", "on")]))
    assert [s.language for s in result.segments] == ["ko", "hi"] and result.notes == []


def test_a_line_repeated_far_beyond_a_chorus_is_not_kept_and_the_previous_text_is_not_fed_back_without_the_filter():
    line = "उनको लाने उरगुम इरविल"
    stuck = [Segment(10.0 + i * 8.0, 15.0 + i * 8.0, line, avg_logprob=-0.4, no_speech_prob=0.2, compression_ratio=1.0) for i in range(12)]
    seen = {}

    class Stuck(FakeDecoder):
        def __call__(self, req):
            seen["profile"] = dict(req.profile)
            result = super().__call__(req)
            result.segments = list(stuck)
            return result

    plan = [(0.0, 120.0, "hi", "off")]
    req = request(plan)
    req.profile = {"condition_on_previous_text": True, "beam_size": 3}
    result = md.transcribe_runs(Stuck(), req)
    assert [s.text for s in result.segments].count(line) == md.MAX_SAME_LINE                                   # a chorus of four stays, the other eight are rejected
    assert len(result.rejected) == 8 and all("same line" in r["reason"] for r in result.rejected)
    assert seen["profile"]["condition_on_previous_text"] is False and req.profile["condition_on_previous_text"] is True         # the request itself is not changed
    normal = md.transcribe_runs(Stuck(), request([(0.0, 120.0, "hi", "on")]))
    assert [s.text for s in normal.segments].count(line) == 12                                                    # with the voice filter on, nothing is second-guessed


def test_latin_letters_that_spell_hindi_in_an_english_run_are_not_called_english():
    from src.multilingual.scripts import romanised_indic

    sung = ["Kīyārē dhūbē dhūbē kēsē kīyārē", "Nara dhamana dhim rākasi kāriya Huthi kai pārthi dhāma", "Kīyā kō mērē bāngē"]
    spoken = ["I've been drinking more alcohol for the prosperity", "Did you check on me?", "Śiva is a name", "The café on the Rue de l'Église", "Tōkyō is far from here, she said"]
    assert all(romanised_indic(text) for text in sung) and not any(romanised_indic(text) for text in spoken)
    segments = [Segment(0, 3, sung[0], avg_logprob=-0.7), Segment(3, 6, spoken[0], avg_logprob=-0.4), Segment(6, 9, "यह हिंदी है", avg_logprob=-0.4)]
    md.label_segments(segments, "en", {"status": "verified", "candidates": ["hi", "ko"], "vad": "off"})
    assert [s.language for s in segments] == ["und", "en", "hi"]                                         # the Hindi script is still proven by its script; the English line stays English
    assert segments[0].language_meta["source"] == "script" and segments[0].language_meta["status"] == "uncertain" and segments[0].language_meta["candidates"] == ["en"]
    assert segments[1].language_meta["source"] == "detector" and "status" not in segments[1].language_meta
    german = [Segment(0, 3, sung[0])]
    md.label_segments(german, "de", {})
    assert german[0].language == "und"
    same = [Segment(0, 3, sung[0])]
    md.label_segments(same, "hi", {})
    assert same[0].language == "und"                                                                      # Latin letters that spell Hindi sounds in a Hindi run are romanised speech, not English and not Hindi text


def test_latin_text_in_a_run_of_another_writing_system_is_english_only_when_it_reads_as_english():
    segments = [Segment(0, 3, "Did you check on me?"), Segment(3, 6, "Tu mera koi na hoke bhi kujitage"), Segment(6, 9, "तू मेरा कोई ना होके भी कुछ लागे"), Segment(9, 12, "I love you, I love you")]
    md.label_segments(segments, "hi", {"status": "verified", "candidates": ["en"], "vad": "off"})
    assert [s.language for s in segments] == ["en", "und", "hi", "en"]                                         # a lyric line in English inside a Hindi run; romanised Hindi is not English
    assert segments[0].language_meta["source"] == "script" and "status" not in segments[0].language_meta
    assert segments[1].language_meta["status"] == "uncertain" and segments[1].language_meta["candidates"] == ["hi"]
    korean = [Segment(0, 3, "Baby, calm down, calm down"), Segment(3, 6, "Madena hita karno mage")]
    md.label_segments(korean, "ko", {})
    assert [s.language for s in korean] == ["en", "und"]
    latin_run = [Segment(0, 3, "Madena hita karno mage")]
    md.label_segments(latin_run, "en", {})
    assert latin_run[0].language == "en"                                                                       # nothing is re-labelled in a run whose language is written in Latin letters
