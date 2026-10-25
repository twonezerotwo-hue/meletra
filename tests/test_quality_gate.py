"""Merge gate for tuning changes (WER vs medical-term recall)."""

import pytest

from src.gate import DEFAULTS, aggregate, evaluate_gate


def m(wer, recall, **kw):
    return {"wer": wer, "recall": recall, **kw}


def test_the_two_examples_from_the_specification(cfg):
    thresholds = cfg["quality_gate"]
    assert evaluate_gate(m(0.030, 0.80), m(0.031, 0.88), thresholds).accepted          # WER +0.1 pt, recall +8 pt
    assert not evaluate_gate(m(0.030, 0.80), m(0.130, 0.82), thresholds).accepted      # WER +10 pt, recall +2 pt


def test_shipped_thresholds_equal_the_documented_defaults(cfg):
    assert {k: cfg["quality_gate"][k] for k in DEFAULTS} == DEFAULTS


def test_same_or_better_wer_is_accepted():
    for after_wer in (0.030, 0.025, 0.0319):
        result = evaluate_gate(m(0.030, 0.80), m(after_wer, 0.80))
        assert result.accepted and result.rule == "wer_same_or_better"


def test_a_small_wer_loss_needs_a_real_recall_gain():
    assert evaluate_gate(m(0.030, 0.70), m(0.038, 0.78)).rule == "recall_gain_outweighs_small_wer_loss"
    rejected = evaluate_gate(m(0.030, 0.70), m(0.038, 0.72))
    assert not rejected.accepted and rejected.rule == "wer_worse_without_enough_recall_gain"


def test_wer_loss_beyond_the_limit_is_rejected_even_with_big_recall_gain():
    assert not evaluate_gate(m(0.030, 0.50), m(0.060, 0.90)).accepted


def test_recall_must_not_drop():
    result = evaluate_gate(m(0.030, 0.90), m(0.020, 0.80))
    assert not result.accepted and result.rule == "recall_dropped"


def test_new_hallucinations_or_loops_reject_a_change_that_otherwise_looks_fine():
    result = evaluate_gate(m(0.03, 0.8, hallucinations=0, repetitions=0), m(0.02, 0.9, hallucinations=0, repetitions=2))
    assert not result.accepted and result.rule == "new_hallucinations"
    assert evaluate_gate(m(0.03, 0.8, repetitions=2), m(0.03, 0.8, repetitions=1)).accepted


def test_result_is_serialisable_and_reports_points():
    data = evaluate_gate(m(0.03, 0.8), m(0.031, 0.88)).to_dict()
    assert data["verdict"] == "ACCEPT" and data["recall_delta_points"] == pytest.approx(8.0)


def test_aggregation_pools_terms_instead_of_averaging_ratios():
    rows = [{"wer": 0.02, "matched_terms": 9, "reference_terms": 10, "hypothesis_terms": 10, "repetitions": 0},
            {"wer": 0.04, "matched_terms": 1, "reference_terms": 2, "hypothesis_terms": 2, "repetitions": 1}]
    total = aggregate(rows)
    assert total["wer"] == pytest.approx(0.03)
    assert total["recall"] == pytest.approx(10 / 12) and total["precision"] == pytest.approx(10 / 12)
    assert total["repetitions"] == 1
    assert aggregate([])["wer"] == 0.0
