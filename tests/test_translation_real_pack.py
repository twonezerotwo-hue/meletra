"""Real translation with the real offline pack (skipped when the pack is not on this computer: the default suite never downloads a model).

Put the folder ``translation-m2m100`` in the program's models folder (Settings -> Models -> Translation pack, or tools/check_pack_download.py) or set MLT_TRANSLATION_PACK_DIR to its
parent folder. These are the checks that fake engines cannot make: the real tokenizer and model, direct translation between two non-English languages, medical-term markers surviving
the real model, the cache, and no network access.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path

import pytest

from src.multilingual import source as source_mod
from src.multilingual.source import SourceSegment, SourceTranscript
from src.translation.m2m100 import find_model
from src.translation.service import TranslationService


def pack_parent() -> Path | None:
    candidates = [os.environ.get("MLT_TRANSLATION_PACK_DIR"), str(Path(os.environ.get("LOCALAPPDATA", "")) / "MedicalLectureTranscriber" / "models")]
    for item in candidates:
        if item and find_model(Path(item)) is not None:
            return Path(item)
    return None


PARENT = pack_parent()
pytestmark = pytest.mark.skipif(PARENT is None, reason="the offline translation pack is not installed on this computer")


@pytest.fixture(scope="module")
def lecture(tmp_path_factory):
    folder = tmp_path_factory.mktemp("real_pack")
    rows = [SourceSegment(1, 0, 5, "오늘은 신경계의 구조에 대해 이야기하겠습니다.", detected_language="ko", language_source="detector"),
            SourceSegment(2, 6, 11, "आज हम तंत्रिका तंत्र की संरचना के बारे में बात करेंगे।", detected_language="hi", language_source="detector"),
            SourceSegment(3, 12, 17, "The MRI showed a lesion in the cerebellum and bevacizumab was started.", detected_language="en", language_source="detector"),
            SourceSegment(4, 18, 23, "Bugün sinir sisteminin yapısından bahsedeceğiz.", detected_language="tr", language_source="detector")]
    source_mod.write(folder, SourceTranscript(lecture_id="real", segments=rows))
    return folder


@pytest.fixture(scope="module")
def service():
    return TranslationService([PARENT], device="cpu")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(self, address, *a, **k):
        raise OSError("no network access during translation")

    monkeypatch.setattr(socket.socket, "connect", refuse)


def test_every_spoken_language_is_translated_to_english_directly(lecture, service):
    doc = service.translate(lecture, "en")
    by_id = {s.segment_id: s for s in doc.segments}
    assert doc.complete and by_id[3].translation_status == "same_language" and by_id[3].text == "The MRI showed a lesion in the cerebellum and bevacizumab was started."
    assert "nervous system" in by_id[1].text.lower() and "nervous system" in by_id[2].text.lower() and "nervous system" in by_id[4].text.lower()


def test_korean_and_hindi_go_straight_to_turkish_without_english_in_between(lecture, service):
    doc = service.translate(lecture, "tr")
    by_id = {s.segment_id: s for s in doc.segments}
    assert doc.complete and all(by_id[i].method == "direct" for i in (1, 2, 3))
    assert "sinir sistem" in by_id[1].text.lower() and "sinir sistem" in by_id[2].text.lower()


def test_medical_terms_survive_the_real_model(lecture, service):
    for target in ("de", "tr", "fr", "es", "ko", "hi", "zh"):
        text = {s.segment_id: s for s in service.translate(lecture, target).segments}[3].text
        assert "MRI" in text or "mri" in text.lower(), (target, text)


def test_the_second_request_is_answered_from_the_stored_translation(lecture, service):
    first = service.translate(lecture, "de")
    again = service.translate(lecture, "de")
    assert again.created_at == first.created_at and service.cached(lecture, "de") is not None
