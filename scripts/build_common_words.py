"""Build config/common_english_words.txt from Whisper's own tokenizer vocabulary.

Why: the possible-medical-errors report must not "correct" ordinary English ("advertise" -> "adverse"). A word that Whisper's
tokenizer holds as ONE whole-word token (with the leading space) is a word the model has seen countless times; a mis-heard
medical term ("jejun", "asperate", "gomaglutamyl") is split into pieces. So the vocabulary doubles as an offline
common-word list. The file is generated once and shipped with the app (no tokenizer needed at run time).

Usage:  python scripts/build_common_words.py [path/to/tokenizer.json]
"""

from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "config" / "common_english_words.txt"
SPACE_PREFIX = chr(0x120)          # the byte-level BPE marks a leading space with this character


def find_tokenizer() -> Path:
    patterns = [str(Path.home() / ".cache" / "huggingface" / "hub" / "models--Systran--faster-whisper-*" / "snapshots" / "*" / "tokenizer.json")]
    for pattern in patterns:
        hits = sorted(glob.glob(pattern))
        if hits:
            return Path(hits[0])
    raise SystemExit("No tokenizer.json found; pass its path as the first argument.")


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else find_tokenizer()
    vocab = json.loads(path.read_text(encoding="utf-8"))["model"]["vocab"]
    words = sorted({token[1:] for token in vocab if token.startswith(SPACE_PREFIX) and re.fullmatch(r"[a-z]{3,}", token[1:])})
    header = [
        "# Common words = whole-word tokens of Whisper's tokenizer (see scripts/build_common_words.py). One per line.",
        "# Used ONLY to keep ordinary English out of the possible-medical-errors report. Edit freely.",
    ]
    OUT.write_text("\n".join(header + words) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(words)} words -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
