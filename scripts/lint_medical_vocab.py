"""Checks the medical_vocab folders: parse errors, duplicate canonical terms, label-like suffixes, unknown kinds, unreadable _domain.json.

    python scripts/lint_medical_vocab.py            # exit 1 on any error
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.medical.lexicon import KINDS, VOCAB_DIR, norm  # noqa: E402

SLIP = re.compile(r"\s(immunology|pathology|physiology|genetics|drug|embryology|histology|development|clinical|metabolic)$", re.I)
KNOWN_LABELS = {"antituberculous drug", "nonsteroidal anti-inflammatory drug"}


def lint(root: Path = VOCAB_DIR, warnings: list[str] | None = None) -> list[str]:
    problems: list[str] = []
    warnings = warnings if warnings is not None else []
    if not root.is_dir():
        return [f"missing folder {root}"]
    for folder in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        meta = folder / "_domain.json"
        if not meta.is_file():
            problems.append(f"{folder.name}: no _domain.json")
        else:
            try:
                data = json.loads(meta.read_text(encoding="utf-8"))
                for related in data.get("related", ()):
                    if not (root / related).is_dir():
                        problems.append(f"{folder.name}/_domain.json: related domain '{related}' does not exist")
            except ValueError as error:
                problems.append(f"{folder.name}/_domain.json: {error}")
        seen: dict[str, str] = {}
        for path in sorted(folder.glob("*.txt")):
            for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                where = f"{folder.name}/{path.name}:{number}"
                parts = [p.strip() for p in line.split("|")]
                canonical = parts[0]
                if not canonical or not norm(canonical):
                    problems.append(f"{where}: empty term")
                    continue
                if len(parts) > 5:
                    problems.append(f"{where}: more than five fields")
                if len(parts) > 4 and parts[4] and parts[4].lower() not in KINDS:
                    problems.append(f"{where}: unknown kind '{parts[4]}'")
                if SLIP.search(canonical) and canonical not in KNOWN_LABELS:
                    problems.append(f"{where}: '{canonical}' looks like a label, not a term")
                key = norm(canonical)
                if key in seen:
                    warnings.append(f"{where}: '{canonical}' also at {seen[key]} (merged when loaded)")
                seen.setdefault(key, where)
    return problems


def main() -> int:
    warnings: list[str] = []
    problems = lint(warnings=warnings)
    for p in problems:
        print("ERROR", p)
    if "--warnings" in sys.argv:
        for w in warnings:
            print("warning", w)
    print(f"{len(problems)} error(s), {len(warnings)} warning(s) (use --warnings to list them)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
