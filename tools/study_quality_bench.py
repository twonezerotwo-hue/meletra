"""Measure the study material of finished lectures with the automated rubric of ``src/study/audit.py``.

    python tools/study_quality_bench.py --lectures "C:/path/to/Medical Transcription" [--kb data/medical_knowledge_public.db] [--json benchmarks/study_quality/results.json]

For every lecture folder (a finished transcription) it builds the study pack with the built-in provider, runs the audit and prints the scores. The JSON holds numbers only, under
anonymous labels (L1, L2 ...): no lecture text, no file name. The scores describe the DELIVERED items (their grammar, sources, distractors ...); how much of a noisy recording
could be used at all is reported as ``yield`` next to them. The rubric is automated, not a human rating.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from src.study.audit import TARGETS  # noqa: E402
from src.study.engine import StudyEngine  # noqa: E402
from src.study.sources import build_lecture_index, find_transcript, load_sources  # noqa: E402


def folders(root: Path) -> list[Path]:
    found = []
    for path in sorted(Path(root).rglob("local_transcript.json")):
        found.append(path.parent)
    seen: set[str] = set()
    out = []
    for folder in found:
        key = folder.name[:24]
        if key in seen:
            continue                                                      # the same recording transcribed twice (a batch folder)
        seen.add(key)
        out.append(folder)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lectures", type=Path, required=True)
    ap.add_argument("--kb", type=Path, help="the knowledge base that tells which words are real (default: found like the app does)")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    if a.kb:
        os.environ["MLT_KNOWLEDGE_DB"] = str(a.kb)
    rows = []
    for n, folder in enumerate(folders(a.lectures), 1):
        sources = load_sources(transcript=find_transcript(folder), title=folder.name)
        index = build_lecture_index(sources)
        pack = StudyEngine().generate(index, title=folder.name)
        quality = pack.stats.get("quality")
        if not quality:
            continue
        facts = pack.overview.get("facts", {})
        row = {"lecture": f"L{n}", "medical_corrected": "medical-corrected" in (index.meta or {}).get("source_note", ""), "minutes": round((index.meta or {}).get("duration") or 0) / 60.0,
               "sentences": facts.get("sentences"), "facts": facts.get("validated"), "yield": facts.get("yield"),
               "items": {"summary": sum(len(v) for k, v in pack.summaries.items() if k in ("quick", "detailed", "high_yield")), "key_concepts": len(pack.summaries.get("key_concepts", [])),
                         "flashcards": len(pack.flashcards), "questions": len(pack.questions)}, "scores": quality["scores"]}
        rows.append(row)
        print(f"{row['lecture']:>3}  facts {row['facts']:>3}/{row['sentences']:<4} yield {row['yield']:.2f}  items {row['items']}  " + "  ".join(f"{k} {v:.2f}" for k, v in quality["scores"].items()))
    if not rows:
        print("no lecture folders found")
        return 1
    means = {k: round(sum(r["scores"][k] for r in rows) / len(rows), 2) for k in TARGETS}
    print("\nMEAN   " + "  ".join(f"{k} {v:.2f}" for k, v in means.items()))
    print("TARGET " + "  ".join(f"{k} {v:.2f}" for k, v in TARGETS.items()))
    print("MEETS  " + "  ".join(f"{k} {'yes' if means[k] >= TARGETS[k] else 'NO'}" for k in TARGETS))
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps({"note": "automated rubric (src/study/audit.py), not a human rating; numbers only, anonymous labels", "lectures": rows, "mean": means, "targets": TARGETS,
                                      "meets_targets": {k: means[k] >= TARGETS[k] for k in TARGETS}}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
