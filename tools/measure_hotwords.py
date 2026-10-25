"""How many lecture terms fit into the Whisper prompt budget? Uses the REAL tokenizer of the speech model (tokenizer.json of faster-whisper-large-v3).

    python tools/measure_hotwords.py --kb data/kb_open_b.db --runs benchmarks/v4/results_local/corpus_runs/auto [--limit 300]

For each transcribed clip: the lecture vocabulary built from the (raw) transcript, its hotword list cut at ``--limit`` terms with no token budget, and how many of those terms
fit into the 223-token part limit of faster-whisper (src/budget.py). Reports the distribution - the real size of a "small lecture-specific hotword set".
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from eval_corpus import segments  # noqa: E402
from src.budget import LIBRARY_PART_CAP  # noqa: E402
from src.medical_knowledge.runtime.context import LectureContext  # noqa: E402
from src.medical_knowledge.runtime.knowledge import MedicalKnowledge  # noqa: E402
from src.medical_knowledge.runtime.lecture import LectureVocabulary  # noqa: E402


def load_tokenizer():
    from tokenizers import Tokenizer

    pattern = os.path.expanduser("~/.cache/huggingface/hub/models--Systran--faster-whisper-large-v3/snapshots/*/tokenizer.json")
    found = sorted(glob.glob(pattern))
    if not found:
        raise SystemExit("tokenizer.json of faster-whisper-large-v3 not found in the Hugging Face cache")
    tok = Tokenizer.from_file(found[-1])
    return lambda text: len(tok.encode(" " + text.strip(), add_special_tokens=False).ids)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", type=Path, required=True)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    count = load_tokenizer()
    kb = MedicalKnowledge.open(a.kb)
    rows = []
    for folder in sorted(a.runs.iterdir()):
        if not (folder / "local_transcript.json").is_file() or "_neg" in folder.name or folder.name.startswith(("neg", "ho_neg")):
            continue
        views = segments(folder)
        meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8")) if (folder / "metadata.json").is_file() else {}
        subjects = tuple(s for s in ((meta.get("subject") or {}).get("subjects") or ()) if s != "general_medical")
        ctx = LectureContext(kb, views, subjects=subjects)
        words = LectureVocabulary(ctx).hotwords(limit=a.limit, token_budget=10 ** 6)
        used, fits = 0, 0
        for w in words:
            cost = count(w) + 1                                              # the separating comma
            if used + cost > LIBRARY_PART_CAP:
                break
            used += cost
            fits += 1
        rows.append({"clip": folder.name, "terms_available": len(words), "terms_that_fit": fits, "tokens_used": used,
                     "tokens_all_terms": sum(count(w) + 1 for w in words)})
    fit = [r["terms_that_fit"] for r in rows]
    avail = [r["terms_available"] for r in rows]
    summary = {"clips": len(rows), "limit": a.limit, "token_part_cap": LIBRARY_PART_CAP, "terms_available_median": statistics.median(avail), "terms_available_max": max(avail),
               "terms_that_fit_median": statistics.median(fit), "terms_that_fit_min": min(fit), "terms_that_fit_max": max(fit),
               "tokens_for_all_available_median": statistics.median(r["tokens_all_terms"] for r in rows),
               "avg_tokens_per_term": round(sum(r["tokens_all_terms"] for r in rows) / max(1, sum(avail)), 2)}
    print(json.dumps(summary, indent=1))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps({"summary": summary, "clips": rows}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
