"""Render the v4 heldout_v4 corpus to speech: the renderer, the voices, the variants and the three-clips-per-passage rule of heldout_v3, with its own seed, plan and folder.

    python tools/make_v4_heldout4.py --plan            # write benchmarks/v4/corpus/heldout_v4_plan.json (deterministic from the seed) - committed BEFORE any audio exists
    python tools/make_v4_heldout4.py [--only id1,id2] [--force]

See tools/make_v4_heldout3.py for the variants. All speech is SYNTHETIC. Writes ``benchmarks/v4/corpus/audio_v4/<id>_<variant>.wav`` (git-ignored) and the reference text.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_v4_heldout3 as base  # noqa: E402

base.SEED = 20261015
base.AUDIO = base.CORPUS / "audio_v4"
base.PLAN = base.CORPUS / "heldout_v4_plan.json"
base.passages = lambda: json.loads((base.CORPUS / "heldout_v4.json").read_text(encoding="utf-8"))["passages"]

if __name__ == "__main__":
    sys.exit(base.main())
