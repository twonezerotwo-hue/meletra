"""False-positive stress on plain written English: any text file is cut into sentences and run through the production Medical Expert and v4.
Nothing in the text is a mishearing, so EVERY change is a false correction (the text of this repository's own docs mentions medical terms, spelled correctly).

    python tools/fp_stress.py --kb data/kb_open_b.db README.md docs/*.md [--subjects neuroanatomy] [--show 20]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.medical.corrector import MedicalContextCorrector, Options, SegmentView  # noqa: E402
from src.medical.lexicon import shared_lexicon  # noqa: E402
from src.medical_knowledge.runtime.engine import V4Engine  # noqa: E402
from src.medical_knowledge.runtime.knowledge import MedicalKnowledge  # noqa: E402


def sentences(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"```.*?```", " ", text, flags=re.S)                  # code blocks are not prose
    text = re.sub(r"`[^`]*`|\[([^\]]*)\]\([^)]*\)|[#*|>_]+|https?://\S+", lambda m: m.group(1) or " ", text)
    out = []
    for chunk in re.split(r"(?<=[.!?])\s+|\n{2,}", text):
        chunk = " ".join(chunk.split())
        if len(chunk.split()) >= 5 and re.fullmatch(r"[A-Za-z0-9 ,.;:'\"()\-/%+&?!]+", chunk):
            out.append(chunk)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--kb", type=Path, required=True)
    ap.add_argument("--subjects", default="")
    ap.add_argument("--show", type=int, default=15)
    a = ap.parse_args(argv)
    subjects = tuple(s for s in a.subjects.split(",") if s)
    kb = MedicalKnowledge.open(a.kb)
    engine = V4Engine(kb)
    lexicon = shared_lexicon()
    total_words = 0
    prod_changes: list[tuple[str, str]] = []
    high: list[tuple[str, str]] = []
    medium: list[tuple[str, str]] = []
    for path in a.files:
        sents = sentences(path)
        words = sum(len(s.split()) for s in sents)
        total_words += words
        views = [SegmentView(i, i * 5.0, i * 5.0 + 4.5, s, "en", []) for i, s in enumerate(sents)]
        prod = MedicalContextCorrector(lexicon.fork(), Options(subjects=subjects)).correct(views)
        prod_changes += [(c.original, c.corrected) for c in prod.corrections if c.category == "medical" and c.tier in ("high", "medium")]
        res = engine.run(views, subjects=subjects)
        high += [(c.original, c.replacement) for c in res.corrections]
        medium += [(c.original, c.replacement) for c in res.uncertain]
        print(f"{path.name}: {len(sents)} sentences, {words} words")
    k = total_words / 1000
    print(f"\n{total_words} words")
    for name, rows in (("production (high+medium)", prod_changes), ("v4 high", high), ("v4 medium (not applied)", medium)):
        print(f"{name:26} {len(rows):4} changes  = {len(rows) / k:5.2f} per 1000 words")
        for original, replacement in rows[: a.show]:
            print(f"    {original!r} -> {replacement!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
