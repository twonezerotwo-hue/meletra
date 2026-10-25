"""Per-lecture table from a review folder + the label file: pairs (and occurrences) labelled correct / wrong per engine.

    python tools/review_by_lecture.py benchmarks/v4/results_local/review5/review_pairs.json
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
rows = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
labels = json.loads((ROOT / "benchmarks" / "v4" / "labels" / "real_lectures.json").read_text(encoding="utf-8"))["labels"]
table: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
for r in rows:
    label = (labels.get(r["id"]) or {}).get("label", "unlabelled")
    for engine in ("production", "v4_high", "v4_medium"):
        if engine in r["engines"]:
            table[r["lecture"]][engine][label] += r["count"]
print(f"{'lecture':32} {'engine':12} {'correct':>7} {'wrong':>6} {'unsure':>6} {'unlab.':>6}  precision")
for lecture in sorted(table):
    for engine in ("production", "v4_high", "v4_medium"):
        c = table[lecture][engine]
        if not c:
            continue
        done = c["correct"] + c["wrong"]
        print(f"{lecture[:32]:32} {engine:12} {c['correct']:7} {c['wrong']:6} {c['unsure']:6} {c['unlabelled']:6}  {('%.1f%%' % (100 * c['correct'] / done)) if done else '-'}")
