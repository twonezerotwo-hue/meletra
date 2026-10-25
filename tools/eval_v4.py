"""Score the current Medical Expert and v4 (shadow) on one corpus set with the separate metrics of the promotion gate.

    python tools/eval_v4.py --set dev --runs auto=benchmarks/v4/results_local/corpus_runs/auto --runs nohot=benchmarks/v4/results_local/corpus_runs/nohot --kb data/medical_knowledge_public.db
    python tools/eval_v4.py --set validation --runs auto=... --gate high=0.74 --gate high_keep_margin=0.3 --json out.json

Every printed number names the corpus set. heldout_v2 may be scored ONCE, at the end; nothing may be tuned on it.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from v4_lib import load_clips, print_report, score  # noqa: E402

from src.medical_knowledge.runtime.engine import V4Engine  # noqa: E402
from src.medical_knowledge.runtime.gate import GateConfig  # noqa: E402
from src.medical_knowledge.runtime.knowledge import MedicalKnowledge  # noqa: E402


def gate_from(pairs) -> GateConfig:
    cfg = GateConfig()
    fields = {f.name: f for f in dataclasses.fields(GateConfig)}
    for pair in pairs or []:
        key, value = pair.split("=", 1)
        if key not in fields:
            raise SystemExit(f"unknown gate setting '{key}' (known: {', '.join(fields)})")
        setattr(cfg, key, type(getattr(cfg, key))(value))
    return cfg


def provenance(run_dirs, set_name: str) -> dict:
    """What a result file stands on: the commit of the code that scored it, the commit of the engine under test, a digest of every raw transcript that was scored."""
    root = Path(__file__).resolve().parents[1]

    def git(*args) -> str:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True).stdout.strip()

    digest = hashlib.sha256()
    n = 0
    for _setup, runs in run_dirs:
        for f in sorted(runs.glob("*/local_raw_transcript.txt")):
            digest.update(f.parent.name.encode() + b"\0" + f.read_bytes() + b"\0")
            n += 1
    out = {"scored_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "scoring_code_commit": git("rev-parse", "HEAD"),
           "working_tree_clean": git("status", "--porcelain", "--", "src", "tools", "tests") == "", "raw_transcripts": n, "raw_transcripts_sha256": digest.hexdigest()}
    corpus = root / "benchmarks" / "v4" / "corpus" / f"{set_name}.json"
    if corpus.is_file():
        out["corpus_provenance"] = json.loads(corpus.read_text(encoding="utf-8")).get("provenance")
        frozen = (out["corpus_provenance"] or {}).get("engine_commit")
        if frozen:
            # the engine is whatever decides a correction: the v4 runtime, its normalisers / indexes / database layer, and the production Medical Expert
            changed = git("diff", "--name-only", frozen, "HEAD", "--", "src/medical_knowledge/runtime", "src/medical_knowledge/normalize", "src/medical_knowledge/indexes",
                          "src/medical_knowledge/database", "src/medical_knowledge/domains.py", "src/medical").splitlines()
            out["engine_commit"] = frozen
            out["engine_files_changed_since_freeze"] = changed
            out["engine_unchanged_since_freeze"] = not changed
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=["dev", "validation", "heldout_v2", "heldout_v3", "heldout_v4", "heldout_v5"])
    ap.add_argument("--runs", action="append", required=True, help="setup=folder, e.g. auto=benchmarks/v4/results_local/corpus_runs/auto")
    ap.add_argument("--kb", type=Path, required=True)
    ap.add_argument("--gate", action="append", help="override a GateConfig field: name=value")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--details", action="store_true", help="keep every proposal and every failure diagnosis in the JSON")
    a = ap.parse_args(argv)
    kb = MedicalKnowledge.open(a.kb)
    engine = V4Engine(kb)
    run_dirs = [(spec.split("=", 1)[0], Path(spec.split("=", 1)[1])) for spec in a.runs]
    clips = load_clips(engine, a.set, run_dirs)
    gate = gate_from(a.gate)
    reports = []
    setups = sorted({c.setup for c in clips})
    for setup in [None] + setups:
        sub = [c for c in clips if setup is None or c.setup == setup]
        rep = score(sub, engine, gate, label=f"{a.set} / {setup or 'all setups'}", keep_details=a.details and setup is None)
        print_report(rep)
        reports.append(rep)
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps({"set": a.set, "kb": str(a.kb), "gate": dataclasses.asdict(gate), "provenance": provenance(run_dirs, a.set), "reports": reports}, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
