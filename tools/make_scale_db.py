"""Build a database with N SYNTHETIC terms on top of the curated vocabulary, for latency / memory / size measurements (never shipped).

    python tools/make_scale_db.py --terms 1000000 --output data/scale_1m.db
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.medical_knowledge.ingest.builder import KnowledgeBuilder  # noqa: E402
from src.medical_knowledge.sources.curated import CuratedSource  # noqa: E402
from src.medical_knowledge.sources.fixture import FixtureSource  # noqa: E402
from src.medical_knowledge.sources.synthetic import SyntheticSource  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--terms", type=int, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    builder = KnowledgeBuilder(a.output, "full", "scale-test")
    for source in (CuratedSource(), FixtureSource(), SyntheticSource(a.terms)):
        s = source.export(builder)
        print(f"  {s.source_id}: {s.concepts:,} concepts, {s.terms:,} terms ({time.time() - t0:.0f} s)", flush=True)
    m = builder.finalize()
    print(f"built {m['counts']['terms']:,} terms, {m['size_bytes'] / 1e6:.0f} MB in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
