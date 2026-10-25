"""Hangul, Devanagari, Han, Latin with diacritics, Greek letters and medical symbols survive every file the program writes and every place it shows text: TXT, SRT, VTT, JSON, the
source transcript, translations, the study pack, the archive record, the transcript viewer. (Transcripts are not stored in SQLite: the only database is the medical vocabulary.)"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.exporters import raw_transcript_text, segments_to_srt, segments_to_vtt, timestamped_text, transcript_json, write_json, write_text
from src.multilingual import source as source_mod
from src.multilingual.source import SourceSegment, SourceTranscript
from src.schema import Segment
from src.services.job_service import JobStore
from src.translation.service import TranslationService

SAMPLES = {
    "ko": "소뇌는 운동을 조절합니다. 환자는 40°C의 열이 있었습니다.",
    "hi": "अनुमस्तिष्क गति का समन्वय करता है। रोगी को 40°C बुखार था।",
    "zh": "小脑协调运动。患者发热40°C，血糖≥7.0 mmol/L。",
    "tr": "Beyincik hareketi koordine eder; şiddetli baş ağrısı, ğ ü ş ı İ.",
    "de": "Das Kleinhirn koordiniert Bewegungen: Größe, Maß, Äpfel, ß.",
    "el": "Η δράση του α-TNF και της β-αδρενεργικής οδού, μ-opioid receptor, ΔF508, Ca²⁺ ± 2 → ↑",
    "fr": "Le cervelet coordonne le mouvement : œdème, ça, déjà, naïf, é à ù.",
    "es": "El cerebelo coordina el movimiento: ¿niño?, ¡dolor!, ñ, á é í ó ú.",
    "hu": "A kisagy a mozgást koordinálja: őrült, űr, árvíztűrő tükörfúrógép.",
}


def segments() -> list[Segment]:
    return [Segment(i * 5.0, i * 5.0 + 4.0, " " + text, code, "m", "fake") for i, (code, text) in enumerate(SAMPLES.items())]


def test_text_and_subtitle_files_keep_every_character(tmp_path):
    segs = segments()
    texts = [t for t in SAMPLES.values()]
    raw = write_text(tmp_path / "raw.txt", raw_transcript_text(segs))
    assert raw.read_bytes().decode("utf-8").splitlines() == texts and not raw.read_bytes().startswith(b"\xef\xbb\xbf") and b"\r" not in raw.read_bytes()
    srt = write_text(tmp_path / "t.srt", segments_to_srt(segs)).read_bytes().decode("utf-8")
    vtt = write_text(tmp_path / "t.vtt", segments_to_vtt(segs)).read_bytes().decode("utf-8")
    stamped = timestamped_text(segs)
    for text in texts:
        words = text.split()
        for word in words:
            assert word in srt.replace("\n", " ") and word in vtt.replace("\n", " ") and word in stamped, word
    assert "[KO]" in stamped and "[HI]" in stamped and "[EL]" in stamped                                           # a recording in several languages names each line's language


def test_json_files_are_written_as_utf_8_not_as_escapes(tmp_path):
    data = transcript_json(segments(), language="ko", language_probability=0.9, model="m", backend="b", duration=50.0, source="x.mp4")
    path = write_json(tmp_path / "t.json", data)
    raw = path.read_bytes()
    assert "소뇌는" in raw.decode("utf-8") and b"\\u" not in raw
    again = json.loads(raw.decode("utf-8"))
    assert [s["text"] for s in again["segments"]] == list(SAMPLES.values()) and [s["language"] for s in again["segments"]] == list(SAMPLES)


def make_source(folder: Path) -> SourceTranscript:
    rows = [SourceSegment(i + 1, i * 5.0, i * 5.0 + 4.0, text, detected_language=code, language_source="detector") for i, (code, text) in enumerate(SAMPLES.items())]
    source = SourceTranscript(lecture_id=folder.name, segments=rows)
    folder.mkdir(parents=True, exist_ok=True)
    source_mod.write(folder, source)
    return source


def test_the_source_transcript_and_a_translation_round_trip_exactly(tmp_path):
    folder = tmp_path / "lecture"
    source = make_source(folder)
    again = source_mod.read_file(folder)
    assert [s.source_text for s in again.segments] == list(SAMPLES.values()) and again.revision == source.revision
    assert b"\\u" not in (folder / "source_transcript.json").read_bytes()

    class Echo:
        id = "echo:1"

        def translate(self, texts, src, target):
            return [t + " | " + target for t in texts]

    doc = TranslationService(engine=Echo()).translate(folder, "de")
    stored = json.loads((folder / "translations" / "de.json").read_bytes().decode("utf-8"))
    assert [s["source_text"] for s in stored["segments"]] == list(SAMPLES.values())
    assert (folder / "translated_de_transcript.txt").read_bytes().decode("utf-8").count("소뇌는") == 1
    srt = (folder / "translated_de_transcript.srt").read_bytes().decode("utf-8")
    assert "अनुमस्तिष्क" in srt and "小脑协调运动" in srt and "Η δράση" in srt and doc.target == "de"


def test_the_archive_record_and_the_study_pack_keep_unicode(tmp_path):
    store = JobStore(tmp_path / "jobs")
    store.create_or_get("j1", tmp_path / "a.mp4", output_dir=str(tmp_path / "out"), state="completed")
    store.rename("j1", "신경해부학 · तंत्रिका तंत्र · 小脑")
    assert JobStore(tmp_path / "jobs").get("j1").display_name == "신경해부학 · तंत्रिका तंत्र · 小脑"
    from src.study.models import StudyPack
    from src.study.store import StudyStore

    folder = tmp_path / "lecture"
    pack = StudyPack(lecture_id="l", title="소뇌 · अनुमस्तिष्क", source_mode="transcript", language="ko", topics=["소뇌", "अनुमस्तिष्क", "小脑", "α-TNF"])
    StudyStore(folder, "ko").save_pack(pack)
    loaded = StudyStore(folder, "ko").load_pack()
    assert loaded.title == "소뇌 · अनुमस्तिष्क" and loaded.topics == ["소뇌", "अनुमस्तिष्क", "小脑", "α-TNF"]


def test_the_transcript_viewer_shows_and_copies_every_script(qapp, tmp_path):
    from PySide6.QtGui import QGuiApplication

    from src.gui.main_window import MainWindow
    from tests.gui_support import ScriptedLauncher, make_services, settle

    services = make_services(tmp_path, opened=[])
    window = MainWindow(services, ScriptedLauncher())
    try:
        folder = tmp_path / "lecture"
        make_source(folder)
        write_text(folder / "local_clean_transcript.txt", "\n".join(SAMPLES.values()) + "\n")
        (folder / "local_transcript.json").write_text(json.dumps({"language": "ko", "segments": [{"start": 0, "end": 1, "text": SAMPLES["ko"]}]}, ensure_ascii=False), encoding="utf-8")
        (folder / "metadata.json").write_text(json.dumps({"language": {"detected": "ko"}}), encoding="utf-8")
        window.show_finished(folder, "신경해부학")
        window.open_transcript()
        view = window.transcript_view
        assert view.text.toPlainText().splitlines()[:len(SAMPLES)] == list(SAMPLES.values())
        view.copy_all()
        assert QGuiApplication.clipboard().text().splitlines()[:len(SAMPLES)] == list(SAMPLES.values())
        view.search_edit.setText("अनुमस्तिष्क")
        assert view.hits_label.text() == "1 found"
        assert "신경해부학" in window.simple_result.file_label.text()
    finally:
        window.controller._timer.stop()
        settle(window, window.home, window.study)
        window.deleteLater()


@pytest.mark.windows_only
def test_the_computer_has_fonts_for_every_script_the_program_shows():
    from PySide6.QtGui import QFontDatabase

    families = QFontDatabase.families()
    wanted = {"Korean": QFontDatabase.Korean, "Devanagari": QFontDatabase.Devanagari, "Simplified Chinese": QFontDatabase.SimplifiedChinese, "Greek": QFontDatabase.Greek,
              "Latin": QFontDatabase.Latin}
    for name, system in wanted.items():
        assert any(system in QFontDatabase.writingSystems(family) for family in families), name
