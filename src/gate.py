"""Quality gate for tuning changes: is "after" an acceptable replacement for "before"?

Rules (thresholds in config/transcription.yaml -> quality_gate):

1. medical-term recall may not fall by more than ``recall_drop_max``;
2. no NEW hallucination / repetition flags (a tuning change must not introduce loops);
3. WER equal or better (within ``wer_noise``)  ->  ACCEPT;
4. otherwise recall must rise by at least ``recall_gain_min`` while WER gets worse by at most ``wer_max_loss``  ->  ACCEPT;
5. anything else  ->  REJECT.

Examples (defaults): WER +0.1 pt and recall +8 pt -> accept (rule 3, also 4); WER +10 pt and recall +2 pt -> reject.
All values are fractions (0.02 = 2 %).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

DEFAULTS = {"wer_noise": 0.002, "recall_gain_min": 0.05, "wer_max_loss": 0.01, "recall_drop_max": 0.02}


@dataclass
class GateResult:
    accepted: bool
    rule: str
    wer_delta: float
    recall_delta: float
    reasons: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        return "ACCEPT" if self.accepted else "REJECT"

    def to_dict(self) -> dict[str, Any]:
        return {"verdict": self.verdict, "rule": self.rule, "wer_delta_points": round(100 * self.wer_delta, 2),
                "recall_delta_points": round(100 * self.recall_delta, 2), "reasons": self.reasons}


def evaluate_gate(before: Mapping[str, float], after: Mapping[str, float], thresholds: Mapping[str, float] | None = None
                  ) -> GateResult:
    """Compare two metric dicts with keys ``wer``, ``recall`` and (optional) ``hallucinations`` / ``repetitions``."""
    t = {**DEFAULTS, **(thresholds or {})}
    wer_delta = float(after["wer"]) - float(before["wer"])
    recall_delta = float(after["recall"]) - float(before["recall"])
    new_flags = (float(after.get("hallucinations", 0)) + float(after.get("repetitions", 0))
                 - float(before.get("hallucinations", 0)) - float(before.get("repetitions", 0)))
    if recall_delta < -t["recall_drop_max"]:
        return GateResult(False, "recall_dropped", wer_delta, recall_delta,
                          [f"medical-term recall fell by {-100 * recall_delta:.1f} points (limit {100 * t['recall_drop_max']:.1f})"])
    if new_flags > 0:
        return GateResult(False, "new_hallucinations", wer_delta, recall_delta,
                          [f"{new_flags:g} more hallucination/repetition flag(s) than before"])
    if wer_delta <= t["wer_noise"]:
        return GateResult(True, "wer_same_or_better", wer_delta, recall_delta,
                          [f"WER {100 * wer_delta:+.2f} points (noise limit {100 * t['wer_noise']:.2f})"])
    if recall_delta >= t["recall_gain_min"] and wer_delta <= t["wer_max_loss"]:
        return GateResult(True, "recall_gain_outweighs_small_wer_loss", wer_delta, recall_delta,
                          [f"recall {100 * recall_delta:+.1f} points for WER {100 * wer_delta:+.2f} points"])
    return GateResult(False, "wer_worse_without_enough_recall_gain", wer_delta, recall_delta,
                      [f"WER {100 * wer_delta:+.2f} points is not paid for by recall {100 * recall_delta:+.1f} points"])


def aggregate(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """One set of metrics over several samples: mean WER, term recall/precision pooled over all terms
    (``matched_terms`` / ``reference_terms`` / ``hypothesis_terms`` per row), flags summed."""
    if not rows:
        return {"wer": 0.0, "recall": 0.0, "precision": 0.0, "hallucinations": 0, "repetitions": 0}
    reference = sum(float(r.get("reference_terms") or 0) for r in rows)
    hypothesis = sum(float(r.get("hypothesis_terms") or 0) for r in rows)
    matched = sum(float(r.get("matched_terms") or 0) for r in rows)
    return {"wer": sum(float(r["wer"]) for r in rows) / len(rows),
            "recall": matched / reference if reference else 0.0,
            "precision": matched / hypothesis if hypothesis else 0.0,
            "hallucinations": sum(r.get("hallucinations", 0) for r in rows),
            "repetitions": sum(r.get("repetitions", 0) for r in rows)}
