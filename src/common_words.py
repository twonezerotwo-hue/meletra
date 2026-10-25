"""Offline list of ordinary English words (generated from Whisper's tokenizer; see scripts/build_common_words.py)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from . import RESOURCE_ROOT

COMMON_WORDS_FILE = RESOURCE_ROOT / "config" / "common_english_words.txt"


@lru_cache(maxsize=1)
def common_words(path: str = str(COMMON_WORDS_FILE)) -> frozenset[str]:
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return frozenset()
    return frozenset(line.strip().lower() for line in lines if line.strip() and not line.startswith("#"))


def is_common_word(word: str) -> bool:
    """True for an ordinary English word ("advertise", "regarding", "serious"); False for rare / medical / unknown spellings."""
    return word.strip().lower() in common_words()
