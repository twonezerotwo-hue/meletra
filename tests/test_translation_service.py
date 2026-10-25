"""Text-only translation of a lecture's saved source transcript: direct per-segment translation, the cache, honest failures, legacy lectures, no recording and no recogniser."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.multilingual import source as source_mod
from src.multilingual.source import SourceSegment, SourceTranscript
from src.translation import service as svc
from src.translation.service import TranslationCancelled, TranslationService, TranslationUnavailableError

KO = "안녕하세요 오늘은 신경해부학을 공부합니다"
HI = "नमस्ते आज हम तंत्रिका तंत्र के बारे में पढ़ेंगे"
EN = "Today we will discuss the substantia nigra and GABA in 5 mg per mL."


class FakeEngine:
    """Records every call; 'translates' by tagging, so the test sees exactly which language pair each sentence went through."""

    id = "fake-engine:1"

    def __init__(self, fail_on: str = "") -> None:
        self.calls: list[tuple[str, str, list[str]]] = []
        self.fail_on = fail_on

    def translate(self, texts, source, target):
        self.calls.append((source, target, list(texts)))
        if self.fail_on and self.fail_on == source:
            raise RuntimeError("simulated engine failure")
        return [f"<{source}>{target}:{t}" for t in texts]


def make_lecture(folder: Path, segments=None, origin="pipeline") -> SourceTranscript:
    segments = segments or [
        SourceSegment(1, 0.0, 4.0, KO, "ko", 0.93, "detector"),
        SourceSegment(2, 4.5, 9.0, HI, "hi", 0.88, "detector"),
        SourceSegment(3, 9.5, 14.0, EN, "en", 0.99, "detector"),
        SourceSegment(4, 14.5, 16.0, "...", "und", None, "unknown"),
    ]
    source = SourceTranscript("lec-1", segments, origin=origin, input_language="auto", requested_output_language="en")
    source_mod.write(folder, source)
    return source


def test_each_segment_is_translated_from_its_own_language_directly(tmp_path):
    make_lecture(tmp_path)
    engine = FakeEngine()
    doc = TranslationService(engine=engine).translate(tmp_path, "de")
    pairs = [(s, t) for s, t, _ in engine.calls]
    assert sorted(pairs) == [("en", "de"), ("hi", "de"), ("ko", "de")]                          # direct: never ko -> en -> de
    by_id = {s.segment_id: s for s in doc.segments}
    assert by_id[1].text.startswith("<ko>de:") and by_id[2].text.startswith("<hi>de:") and by_id[3].text.startswith("<en>de:")
    assert all(s.method == "direct" for s in doc.segments if s.translation_status == "translated")


def test_segments_already_in_the_target_language_are_copied_and_unknown_ones_are_not_guessed(tmp_path):
    make_lecture(tmp_path)
    engine = FakeEngine()
    doc = TranslationService(engine=engine).translate(tmp_path, "en")
    by_id = {s.segment_id: s for s in doc.segments}
    assert by_id[3].translation_status == "same_language" and by_id[3].text == EN and by_id[3].method == "copied"
    assert not any(s == "en" for s, _t, _x in engine.calls)                                    # English was not sent through the engine
    assert by_id[4].translation_status == "untranslated" and by_id[4].text == "" and by_id[4].source_text == "..."       # kept, marked, not dropped
    assert doc.status == "partial" and not doc.complete
    lines = doc.display_lines()
    assert any(line.startswith("[not translated") and "..." in line for line in lines)


def test_a_segment_of_unknown_language_is_translated_when_its_writing_system_proves_the_language(tmp_path):
    make_lecture(tmp_path, [SourceSegment(1, 0, 3, KO, "und", None, "unknown")])
    engine = FakeEngine()
    doc = TranslationService(engine=engine).translate(tmp_path, "tr")
    assert engine.calls and engine.calls[0][0] == "ko" and doc.segments[0].translation_status == "translated"      # Hangul can only be Korean


def test_unicode_survives_the_files_a_translation_writes(tmp_path):
    make_lecture(tmp_path)
    doc = TranslationService(engine=FakeEngine()).translate(tmp_path, "ko")                    # target Korean: Hangul in, Hangul out
    cache = (tmp_path / "translations" / "ko.json").read_text(encoding="utf-8")
    assert KO in cache and HI in cache and "\\uc548" not in cache                               # source text stored readable, not ASCII-escaped
    for name in ("txt", "srt", "json"):
        text = (tmp_path / f"translated_ko_transcript.{name}").read_text(encoding="utf-8")
        assert KO in text or "<ko>" in text or "ko" in text
    srt = (tmp_path / "translated_ko_transcript.srt").read_text(encoding="utf-8")
    assert "00:00:00,000 --> 00:00:04,000" in srt and "00:00:04,500 --> 00:00:09,000" in srt   # timestamps and reading order kept
    assert doc.segments[0].start == 0.0 and doc.segments[2].end == 14.0


def test_a_second_request_for_the_same_language_is_served_from_the_cache(tmp_path):
    make_lecture(tmp_path)
    engine = FakeEngine()
    service = TranslationService(engine=engine)
    first = service.translate(tmp_path, "de")
    calls = len(engine.calls)
    again = service.translate(tmp_path, "de")
    assert len(engine.calls) == calls and again.created_at == first.created_at                  # no engine call at all
    assert service.availability(tmp_path, "de").state == "cached"
    assert set(service.stored_targets(tmp_path)) == {"de"}


def test_a_changed_source_makes_the_stored_translation_stale(tmp_path):
    source = make_lecture(tmp_path)
    engine = FakeEngine()
    service = TranslationService(engine=engine)
    service.translate(tmp_path, "de")
    source.segments[0].source_text = KO + " 수정"                                                  # the transcript was corrected
    source_mod.write(tmp_path, source)
    assert service.cached(tmp_path, "de") is None and service.stored_targets(tmp_path) == {}    # never served stale
    before = len(engine.calls)
    service.translate(tmp_path, "de")
    assert len(engine.calls) > before


def test_another_engine_version_is_not_reused(tmp_path):
    make_lecture(tmp_path)
    TranslationService(engine=FakeEngine()).translate(tmp_path, "de")
    newer = FakeEngine()
    newer.id = "fake-engine:2"
    assert TranslationService(engine=newer).cached(tmp_path, "de") is None


def test_a_failed_language_keeps_its_source_text_and_a_retry_redoes_only_it(tmp_path):
    make_lecture(tmp_path)
    broken = FakeEngine(fail_on="hi")
    doc = TranslationService(engine=broken).translate(tmp_path, "de")
    hindi = next(s for s in doc.segments if s.source_language == "hi")
    assert hindi.translation_status == "failed" and hindi.source_text == HI and "simulated" in hindi.error
    assert next(s for s in doc.segments if s.source_language == "ko").translation_status == "translated"     # the other languages are complete
    assert any(HI in line for line in doc.display_lines())                                        # shown with its source text, not dropped
    fixed = FakeEngine()
    retry = TranslationService(engine=fixed).translate(tmp_path, "de", retry_failed=True)
    assert [s for s, _t, _x in fixed.calls] == ["hi"]                                           # only the failed language went through the engine again
    assert next(s for s in retry.segments if s.source_language == "hi").translation_status == "translated"


def test_cancelling_stores_nothing(tmp_path):
    make_lecture(tmp_path)
    with pytest.raises(TranslationCancelled):
        TranslationService(engine=FakeEngine()).translate(tmp_path, "de", should_cancel=lambda: True)
    assert not (tmp_path / "translations").exists() and not list(tmp_path.glob("translated_*"))
    assert source_mod.read_file(tmp_path).output_languages_available == []                     # the source transcript is untouched


def test_a_missing_model_says_so_and_changes_nothing(tmp_path):
    make_lecture(tmp_path)
    service = TranslationService(models_dirs=[tmp_path / "no-models-here"])
    assert service.availability(tmp_path, "de").state == "model_missing"
    with pytest.raises(TranslationUnavailableError) as info:
        service.translate(tmp_path, "de")
    assert info.value.reason == "model_missing" and "Install" in str(info.value)
    assert not (tmp_path / "translations").exists()


def test_a_damaged_model_says_so_and_changes_nothing(tmp_path):
    make_lecture(tmp_path)
    models = tmp_path / "models" / "translation-m2m100"
    models.mkdir(parents=True)
    for name in ("model.bin", "config.json", "sentencepiece.bpe.model", "shared_vocabulary.json"):
        (models / name).write_bytes(b"not a model")
    with pytest.raises(TranslationUnavailableError) as info:
        TranslationService(models_dirs=[tmp_path / "models"]).translate(tmp_path, "de")
    assert info.value.reason in ("model_damaged", "model_missing")
    assert not (tmp_path / "translations").exists()


def test_an_unsupported_target_is_refused_not_answered_in_english(tmp_path):
    make_lecture(tmp_path)
    service = TranslationService(engine=FakeEngine())
    assert service.availability(tmp_path, "xx").state == "unsupported_target"
    with pytest.raises(TranslationUnavailableError):
        service.translate(tmp_path, "xx")


def test_translating_needs_no_recording_and_no_recogniser(tmp_path, monkeypatch):
    """The folder holds only the transcript: no media file exists, and the speech recogniser / model manager is never imported or called."""
    make_lecture(tmp_path)
    import src.model_manager as model_manager

    def boom(*a, **k):
        raise AssertionError("the speech recogniser must not be used to translate a saved transcript")

    monkeypatch.setattr(model_manager.FasterWhisperAdapter, "transcribe", boom, raising=False)
    monkeypatch.setattr(model_manager.FasterWhisperAdapter, "load", boom, raising=False)
    assert not [p for p in tmp_path.iterdir() if p.suffix.lower() in (".mp4", ".wav", ".mp3", ".m4a", ".mkv")]
    doc = TranslationService(engine=FakeEngine()).translate(tmp_path, "tr")
    assert doc.complete is False and doc.counts["translated"] == 3                              # (the undetermined segment stays marked)


def test_long_segments_are_split_into_sentences_and_joined_in_order(tmp_path):
    long_text = " ".join(f"Sentence number {i} is here." for i in range(40))
    make_lecture(tmp_path, [SourceSegment(1, 0, 90, long_text, "en", None, "detector")])
    engine = FakeEngine()
    doc = TranslationService(engine=engine).translate(tmp_path, "tr")
    sent = engine.calls[0][2]
    assert len(sent) > 1 and all(len(t) <= svc.MAX_CHUNK_CHARS for t in sent)
    assert doc.segments[0].text.count("<en>tr:") == len(sent) and "Sentence number 0" in doc.segments[0].text and doc.segments[0].text.index("number 0") < doc.segments[0].text.index("number 39")


def test_split_text_cuts_at_sentence_marks_of_every_script():
    assert svc.split_text("짧은 문장입니다.") == ["짧은 문장입니다."]
    text = "다음 문장입니다。" * 60
    pieces = svc.split_text(text)
    assert len(pieces) > 1 and "".join(pieces).replace(" ", "") == text.replace(" ", "")
    hindi = ("यह एक वाक्य है। " * 40).strip()
    assert all(len(p) <= svc.MAX_CHUNK_CHARS for p in svc.split_text(hindi))


# ------------------------------------------------------------------------------------------------------------------ lectures made by an older version
def write_legacy_lecture(folder: Path, language="en") -> None:
    segments = [{"start": 0.0, "end": 5.0, "text": "This is the English part of the lecture.", "language": language},
                {"start": 5.5, "end": 9.0, "text": "The substantia nigra contains dopaminergic neurons.", "language": language}]
    (folder / "local_transcript.json").write_text(json.dumps({"source": "old.mp4", "language": language, "segments": segments}), encoding="utf-8")
    (folder / "metadata.json").write_text(json.dumps({"language": {"requested": "auto", "detected": language}, "translation": {"output_language": "en", "method": "whisper"}}), encoding="utf-8")


def test_an_old_lecture_is_read_as_what_it_says_and_nothing_more(tmp_path):
    write_legacy_lecture(tmp_path)
    source = source_mod.load(tmp_path)
    assert source.origin == "legacy" and source.detected_languages == ["en"] and len(source.segments) == 2
    assert source.segments[0].language_source == "legacy" and not source.multilingual
    assert source_mod.read_file(tmp_path) is None                                              # reading does not write anything


def test_an_old_lecture_can_be_translated_and_the_result_says_it_is_indirect(tmp_path):
    write_legacy_lecture(tmp_path)
    doc = TranslationService(engine=FakeEngine()).translate(tmp_path, "tr")
    assert doc.complete and any("existing English transcript" in w for w in doc.warnings) and any("Reprocess Original Audio" in w for w in doc.warnings)
    assert (tmp_path / "source_transcript.json").is_file()                                      # the bookkeeping needs a place: derived from the old files, which are unchanged
    assert json.loads((tmp_path / "local_transcript.json").read_text(encoding="utf-8"))["language"] == "en"


def test_the_source_revision_ignores_translation_bookkeeping(tmp_path):
    source = make_lecture(tmp_path)
    before = source.revision
    TranslationService(engine=FakeEngine()).translate(tmp_path, "de")
    after = source_mod.read_file(tmp_path)
    assert after.revision == before and after.output_languages_available == ["de"] and after.translation_engine_versions == {"de": "fake-engine:1"}
    assert after.segments[0].translation_status == {"de": "translated"}


def test_reported_languages_are_only_the_ones_in_the_segments(tmp_path):
    source = make_lecture(tmp_path, [SourceSegment(1, 0, 3, EN, "en"), SourceSegment(2, 3, 6, KO, "ko")])
    assert source.detected_languages == ["en", "ko"] and "hi" not in source.detected_languages       # nothing is "expected": only what was recognised
    other = tmp_path / "x"
    other.mkdir()
    single = make_lecture(other, [SourceSegment(1, 0, 3, EN, "en")])
    assert single.detected_languages == ["en"] and not single.multilingual
