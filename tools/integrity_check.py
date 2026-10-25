"""Integrity metrics of v4 over a set of transcribed runs (see tools/run_v4_corpus.py): things a correction must never break.

    python tools/integrity_check.py --kb data/kb_open_b.db RUNS_DIR [RUNS_DIR ...]

* timestamp integrity   - the corrected transcript has exactly the segments (and times) of the raw one; no correction crosses a segment boundary
* duplicate repair      - number of adjacent repeated words a correction INTRODUCED ("Peyer patches patches"); must be 0
* language drift        - corrections made inside a segment whose language is not English; must be 0
* raw immutability      - the raw segment texts are identical before and after the run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from eval_corpus import apply_tiers, segments  # noqa: E402
from src.medical_knowledge.runtime.engine import V4Engine  # noqa: E402
from src.medical_knowledge.runtime.knowledge import MedicalKnowledge  # noqa: E402


def adjacent_duplicates(text: str) -> int:
    words = re.findall(r"[A-Za-z]+", text.lower())
    return sum(1 for a, b in zip(words, words[1:]) if a == b and len(a) > 2)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--kb", type=Path, required=True)
    a = ap.parse_args(argv)
    engine = V4Engine(MedicalKnowledge.open(a.kb))
    clips = corrections = bad_segments = introduced = drift = raw_changed = crossing = 0
    for runs in a.runs:
        for folder in sorted(runs.iterdir()):
            if not (folder / "local_transcript.json").is_file():
                continue
            segs = segments(folder)
            before = [(s.start, s.end, s.text) for s in segs]
            res = engine.run(segs)
            clips += 1
            for tier_medium in (False, True):
                out = apply_tiers([s.text for s in segs], res, tier_medium)
                if len(out) != len(segs):
                    bad_segments += 1
                introduced += max(0, sum(adjacent_duplicates(t) for t in out) - sum(adjacent_duplicates(s.text) for s in segs))
            items = list(res.corrections) + list(res.uncertain)
            corrections += len(items)
            drift += sum(1 for c in items if (segs[c.segment].language or "en").lower()[:2] != "en")
            crossing += sum(1 for c in items if not (0 <= c.start < c.end <= len(segs[c.segment].text)))
            raw_changed += int(before != [(s.start, s.end, s.text) for s in segs])
    print(json.dumps({"clips": clips, "corrections_checked": corrections, "segment_count_changed": bad_segments, "corrections_outside_their_segment": crossing,
                      "duplicate_words_introduced": introduced, "corrections_in_non_english_segments": drift, "raw_text_or_time_changed": raw_changed}, indent=1))
    return 0 if not (bad_segments or crossing or introduced or drift or raw_changed) else 1


if __name__ == "__main__":
    sys.exit(main())
