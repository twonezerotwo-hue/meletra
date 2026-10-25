"""The medical pass as a whole: raw transcript immutable, separate outputs, report fields, vocabulary influence, hallucination protection."""

import json

import pytest

from src.medical.engine import MedicalOptions, run_medical_pass
from tests.medical_support import seg

LECTURE = [
    "In the cell body of the neuron we find the nucleus, the rough endoplasmic reticulum and many ribosomes.",
    "What are these nisal bodies?",
    "They are the basophilic stacks of rough endoplasmic reticulum, seen in the so called crazy violet staining.",
    "Next, the myelin sheath of the axon is interrupted at the nodes of Ranvier,",
    "and this is called the regulatory conduction of the action potential.",
    "Next, the myelin sheath of the axon is interrupted at the nodes of Ranvier,",
    "In the peripheral nervous system the Schwan cell forms the myelin, and the Remak bundle holds unmyelinated axons.",
    "The cell body of the neuron contains the nucleus.",
]


DOUBT = {"crazy": 0.13, "regulatory": 0.22, "nisal": 0.43}             # what the recogniser itself doubted (real values of the benchmark)


def lecture(texts=LECTURE):
    return [seg(t, i * 5.0, i * 5.0 + 4.5, 0.9, words=True, doubt=DOUBT) for i, t in enumerate(texts)]


def run(segments=None, **kw):
    segments = segments if segments is not None else lecture()
    return run_medical_pass(segments, MedicalOptions(language="en", subjects=["neuroanatomy"], duration=60.0, asr_seconds=100.0, **kw))


def test_the_raw_segments_are_never_modified():
    segments = lecture()
    before = [(s.start, s.end, s.text, [(w.word, w.probability) for w in s.words]) for s in segments]
    result = run(segments)
    assert [(s.start, s.end, s.text, [(w.word, w.probability) for w in s.words]) for s in segments] == before
    assert result.quality_report["raw_transcript_modified"] is False and result.correction_report["raw_transcript_modified"] is False


def test_the_medical_transcript_is_written_beside_an_untouched_raw_file(tmp_path):
    raw = tmp_path / "local_raw_transcript.txt"
    raw.write_text("\n".join(LECTURE) + "\n", encoding="utf-8")
    snapshot = raw.read_bytes()
    result = run()
    written = result.write(tmp_path)
    assert raw.read_bytes() == snapshot
    assert set(written) == {"medical_corrected_transcript.txt", "medical_corrected_transcript_timestamped.txt", "medical_corrected_segments.json",
                            "correction_report.json", "medical_quality_report.json"}
    text = (tmp_path / "medical_corrected_transcript.txt").read_text(encoding="utf-8")
    assert "Nissl bodies" in text and "cresyl violet" in text and "saltatory conduction" in text and "Schwann cell" in text
    assert "nisal" not in text and "crazy violet" not in text
    assert "[00:00:05 --> 00:00:09]" in (tmp_path / "medical_corrected_transcript_timestamped.txt").read_text(encoding="utf-8")
    segments = json.loads((tmp_path / "medical_corrected_segments.json").read_text(encoding="utf-8"))
    assert segments[1]["raw_text"] == "What are these nisal bodies?" and segments[1]["text"] == "What are these Nissl bodies?" and segments[1]["corrections"] == 1


def test_the_correction_report_lists_every_change_with_its_evidence():
    report = run().correction_report
    entry = next(c for c in report["corrections"] if c["original"] == "nisal bodies")
    assert {"timestamp", "original", "corrected", "confidence", "domain", "reason", "context", "tier"} <= set(entry)
    assert entry["corrected"] == "Nissl bodies" and entry["timestamp"] == "00:00:05" and entry["domain"] == "neuroanatomy"
    assert report["vocabulary"] and report["domains_used"] and report["topic"]
    assert {c["original"] for c in report["corrections"]} >= {"nisal bodies", "crazy violet", "regulatory conduction"}


def test_the_quality_report_has_every_field():
    q = run().quality_report
    for key in ("primary_language", "language_confidence", "detected_subject", "total_segments", "medical_terms_detected", "medical_corrections",
                "high_confidence_corrections", "medium_confidence_corrections", "uncertain_terms", "language_drift_segments", "redecoded_segments",
                "duplicate_segments_removed", "timestamp_repairs", "remaining_low_confidence_segments"):
        assert key in q, key
    assert q["primary_language"] == "en" and q["detected_subject"] == "neuroanatomy" and q["total_segments"] == len(LECTURE)
    assert q["medical_corrections"] == q["high_confidence_corrections"] + q["medium_confidence_corrections"] >= 4
    assert q["overhead_percent"] is not None and q["seconds"] >= 0


def test_decoder_duplicates_and_stutters_are_repaired_but_the_original_is_reported():
    texts = LECTURE + ["The cell body of the neuron contains the nucleus."]
    segments = lecture(texts)
    segments[-1].start, segments[-1].end = segments[-2].end + 0.1, segments[-2].end + 4.0
    segments[3] = seg("Next, the myelin sheath of the sheath of the axon is interrupted at the nodes of Ranvier,", 15.0, 19.5, 0.9, words=True, doubt=DOUBT)
    q = run(segments)
    assert q.quality_report["duplicate_segments_removed"] == 1 and q.quality_report["repeated_phrases_collapsed"] >= 1
    assert q.correction_report["duplicates_removed"][0]["text"].startswith("The cell body")
    assert q.text.count("The cell body of the neuron contains the nucleus.") == 1


def test_timestamps_are_made_consistent_in_the_medical_transcript_only():
    segments = lecture()
    segments[2].start = segments[1].end - 1.2                                  # overlaps the previous segment
    raw_times = [(s.start, s.end) for s in segments]
    result = run(segments)
    assert [(s.start, s.end) for s in segments] == raw_times
    assert result.quality_report["timestamp_overlaps_before"] >= 1 and result.quality_report["timestamp_overlaps_after"] == 0
    assert result.quality_report["timestamp_repairs"] >= 1


def test_language_drift_is_counted_and_repaired_when_a_redecode_is_available():
    texts = list(LECTURE)
    texts[3] = "Zdravstveno v tem semestru mikroskopična anatomija bo o neuroanatomiji in nervnem tkivu."
    result = run(lecture(texts), redecode=lambda s, e, l: ("The myelin sheath is interrupted at the nodes of Ranvier.", 0.9))
    q = result.quality_report
    assert q["language_drift_segments"] == 1 and q["redecoded_segments"] == 1
    assert "nodes of Ranvier" in result.segments[3].text and result.correction_report["language_drift"][0]["original"].startswith("Zdravstveno")
    assert run(lecture(texts)).quality_report["redecoded_segments"] == 0              # no audio, no re-decode: reported, not rewritten


def test_nothing_is_invented():
    segments = lecture(["The flibbertigibbet of the zorblat is a quarkonian phenomenon in neurons.", "Nothing here resembles a medical term at all, honestly."])
    result = run(segments)
    assert result.quality_report["medical_corrections"] == 0 and result.segments[0].text.startswith("The flibbertigibbet of the zorblat")


def test_slide_and_custom_vocabulary_change_the_ranking_only():
    segments = lecture(["The patient received zorbaxinal for six weeks."])
    assert run(segments).quality_report["medical_corrections"] == 0
    assert run(segments, lecture_terms=["zorbaxinol"]).segments[0].text.endswith("zorbaxinol for six weeks.")
    assert run(segments, custom_terms=["zorbaxinol"]).segments[0].text.endswith("zorbaxinol for six weeks.")


def test_the_pass_never_raises(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("unexpected")
    monkeypatch.setattr("src.medical.engine._run", boom)
    assert run_medical_pass(lecture(), MedicalOptions(language="en")) is None


def test_empty_and_whitespace_segments_are_handled():
    assert run([]).text.strip() == "" or run([]).quality_report["total_segments"] == 0
    result = run(lecture(["", "   ", "Only this."]))
    assert result.quality_report["total_segments"] == 3 and result.text.strip().endswith("Only this.")


def test_it_is_fast_enough():
    many = lecture(LECTURE * 40)
    result = run(many)
    assert result.seconds < 60.0
