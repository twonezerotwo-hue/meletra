"""SCOWL (Spell Checker Oriented Word Lists, Kevin Atkinson): ordinary English words by commonness, from the ``final/`` folder of the SCOWL release.

Not terminology: these words only tell the correction engine "this is a real word, do not 'correct' it on sound alone". Levels: 10 = the most common words ... 70 = common
dictionary words (about 110,000 inflected words) ... 80 = rarer dictionary words (about 240,000). Proper names, abbreviations and the largest level 95 are left out.
Redistribution requires the SCOWL Copyright text (licenses/SCOWL_COPYRIGHT.txt travels with every pack that contains it).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Iterator

from .base import Record, SourceInfo, TerminologySource, WordRecord

SOURCE_ID = "SCOWL"
KINDS = ("english-words", "american-words", "british-words")
LEVELS = (10, 20, 35, 40, 50, 55, 60, 70, 80)


class ScowlSource(TerminologySource):
    def __init__(self, root: Path, max_level: int = 80) -> None:
        self.root = Path(root)
        self.max_level = max_level
        version = (self.root / "VERSION").read_text(encoding="utf-8").strip() if (self.root / "VERSION").is_file() else "unknown"
        self.info = SourceInfo(SOURCE_ID, "SCOWL English word lists", version, "http://wordlist.aspell.net/", "PERMISSIVE_NOTICE", True, False, "en", version,
                               f"levels up to {max_level}; ordinary words only")

    def verify(self) -> list[str]:
        folder = self.root / "final"
        return [] if (folder / "english-words.10").is_file() else [f"SCOWL 'final' word lists not found under {self.root} (https://sourceforge.net/projects/wordlist/files/SCOWL/)"]

    def parse(self) -> Iterator[dict]:
        best: dict[str, int] = {}
        for kind in KINDS:
            for level in LEVELS:
                if level > self.max_level:
                    continue
                path = self.root / "final" / f"{kind}.{level}"
                if not path.is_file():
                    continue
                for line in path.read_text(encoding="latin-1").splitlines():
                    word = line.strip().lower()
                    if re.fullmatch(r"[a-z]{2,}", word) and (word not in best or level < best[word]):
                        best[word] = level
        for word, level in best.items():
            yield {"word": word, "level": level}

    def normalize(self, raw: Iterable[dict]) -> Iterator[Record]:
        for row in raw:
            yield WordRecord(row["word"], row["level"])
