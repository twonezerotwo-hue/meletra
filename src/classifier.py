"""Rule/score based medical subject classifier (no LLM).

Evidence sources, each turned into a raw score per area:
  * the user's ``--subject`` (strongest prior),
  * the media file name (aliases such as ``biochem`` / ``neuro`` / ``fizyoloji``),
  * the first minutes of transcript,
  * lecture slides (PDF text).

Raw scores are squashed to 0-1 with ``raw / (raw + saturation)`` so the output reads like
``neuroscience: 0.82, anatomy: 0.63, physiology: 0.41`` and the strongest 1-3 areas are used.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

from .knowledge_subjects import (
    AREA_KEYWORDS, AREA_VOCAB_FILES, AREAS, FILENAME_ALIASES, FILENAME_ALIASES_TR,
)
from .vocabulary import DEFAULT_TERMS_DIR, VocabularyIndex, load_terms, normalize_key

PER_KEYWORD_CAP = 5            # one repeated word must not decide the subject alone
AREA_NAME_WEIGHT = 2.0         # "welcome to biochemistry" is a very strong statement
PHRASE_WEIGHT = 1.5            # multi-word keywords are more specific than single words
VOCAB_PHRASE_WEIGHT = 1.2      # an exact multi-word dictionary term from the area's vocabulary file
USER_SUBJECT_RAW = 40.0
FILENAME_RAW = 8.0
PDF_SOURCE_FACTOR = 1.25       # slides are authoritative
MIN_KEYWORD_LENGTH = 3


@dataclass
class AreaScore:
    area: str
    score: float                                   # 0-1
    raw: float = 0.0
    by_source: dict[str, float] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)   # the strongest matched keywords

    def to_dict(self) -> dict:
        return {"area": self.area, "score": round(self.score, 3), "raw": round(self.raw, 2),
                "by_source": {k: round(v, 2) for k, v in self.by_source.items()}, "evidence": self.evidence[:8]}


def _normalise_text(text: str) -> str:
    return " " + re.sub(r"[^a-z0-9]+", " ", text.casefold().replace("ı", "i")) + " "


def _compile(keywords: Iterable[str]) -> dict[str, re.Pattern[str]]:
    patterns: dict[str, re.Pattern[str]] = {}
    for kw in keywords:
        norm = _normalise_text(kw).strip()
        if len(norm) >= MIN_KEYWORD_LENGTH:
            patterns[kw] = re.compile(r"(?<![a-z0-9])" + re.escape(norm).replace(r"\ ", " ") + r"(?![a-z0-9])")
    return patterns


_PATTERNS: dict[str, dict[str, re.Pattern[str]]] = {area: _compile(words) for area, words in AREA_KEYWORDS.items()}


def filename_areas(filename: Optional[str]) -> list[str]:
    """Areas hinted by the file name (``Biochem_lecture3.mp4`` -> biochemistry). Longest alias wins."""
    if not filename:
        return []
    stem = Path(filename).stem
    squashed = re.sub(r"[^a-z0-9]+", "", stem.casefold().replace("ı", "i"))
    tokens = {t for t in re.split(r"[^a-z0-9]+", stem.casefold().replace("ı", "i")) if t}
    found: list[str] = []
    for alias, area in sorted({**FILENAME_ALIASES, **FILENAME_ALIASES_TR}.items(), key=lambda kv: -len(kv[0])):
        # short aliases (<=5 letters) must be whole tokens, otherwise "micro" matches "microphone"
        hit = alias in tokens or (len(alias) > 5 and alias in squashed)
        if hit and area not in found:
            found.append(area)
    return found


class SubjectClassifier:
    """Scores the 19 areas from the available evidence."""

    def __init__(self, terms_dir: Path = DEFAULT_TERMS_DIR, language: Optional[str] = None,
                 saturation: float = 12.0) -> None:
        self.terms_dir = Path(terms_dir)
        self.language = language
        self.saturation = saturation
        self._file_indexes: dict[str, VocabularyIndex] = {}

    def _index(self, vocab_file: str) -> VocabularyIndex:
        if vocab_file not in self._file_indexes:
            self._file_indexes[vocab_file] = VocabularyIndex(load_terms(vocab_file, self.language, self.terms_dir))
        return self._file_indexes[vocab_file]

    def _text_raw(self, text: str, area: str) -> tuple[float, list[tuple[float, str]]]:
        norm = _normalise_text(text)
        raw, hits = 0.0, []
        for position, (kw, pattern) in enumerate(_PATTERNS[area].items()):
            count = len(pattern.findall(norm))
            if not count:
                continue
            weight = AREA_NAME_WEIGHT if position == 0 else (PHRASE_WEIGHT if " " in kw.strip() else 1.0)
            value = weight * min(count, PER_KEYWORD_CAP)
            raw += value
            hits.append((value, kw))
        for rank, vocab_file in enumerate(AREA_VOCAB_FILES.get(area, ())):
            phrases = [t for t in self._index(vocab_file).find_terms(text) if " " in t]
            if phrases:
                value = VOCAB_PHRASE_WEIGHT * min(len(phrases), 10) * (1.0 if rank == 0 else 0.6)
                raw += value
                hits.append((value, f"{len(phrases)} {vocab_file} terms"))
        return raw, hits

    def score(
        self, *, filename: Optional[str] = None, early_text: str = "", pdf_text: str = "",
        user_subjects: Sequence[str] = (),
    ) -> list[AreaScore]:
        results: list[AreaScore] = []
        named = filename_areas(filename)
        wanted = {self._canonical_area(s) for s in user_subjects}
        for area in AREAS:
            by_source: dict[str, float] = {}
            evidence: list[tuple[float, str]] = []
            if early_text:
                raw, hits = self._text_raw(early_text, area)
                if raw:
                    by_source["early_transcript"] = raw
                    evidence += hits
            if pdf_text:
                raw, hits = self._text_raw(pdf_text, area)
                if raw:
                    by_source["pdf"] = raw * PDF_SOURCE_FACTOR
                    evidence += [(v * PDF_SOURCE_FACTOR, f"pdf:{k}") for v, k in hits]
            if area in named:
                by_source["filename"] = FILENAME_RAW
                evidence.append((FILENAME_RAW, f"file name hints '{area}'"))
            if area in wanted:
                by_source["user_subject"] = USER_SUBJECT_RAW
                evidence.append((USER_SUBJECT_RAW, "requested with --subject"))
            total = sum(by_source.values())
            results.append(AreaScore(
                area=area, score=total / (total + self.saturation) if total else 0.0, raw=total, by_source=by_source,
                evidence=[kw for _, kw in sorted(evidence, key=lambda e: -e[0])]))
        results.sort(key=lambda r: (-r.score, r.area))
        return results

    @staticmethod
    def _canonical_area(name: str) -> str:
        key = name.strip().lower().replace(" ", "_")
        return key if key in AREAS else FILENAME_ALIASES.get(key.replace("_", ""), key)

    @staticmethod
    def select(scores: Sequence[AreaScore], max_areas: int = 3, relative_cutoff: float = 0.5,
               min_score: float = 0.2) -> list[AreaScore]:
        """The strongest 1-``max_areas`` areas with a real signal ([] when nothing scores >= ``min_score``)."""
        if not scores or scores[0].score < min_score:
            return []
        top = scores[0].score
        return [s for s in scores if s.score >= min_score and s.score >= relative_cutoff * top][:max_areas]


def vocab_file_scores(selected: Sequence[AreaScore]) -> dict[str, float]:
    """0-1 relevance of every vocabulary file given the selected areas (first file of an area counts fully)."""
    files: dict[str, float] = {}
    for area in selected:
        for rank, vocab_file in enumerate(AREA_VOCAB_FILES.get(area.area, ())):
            files[vocab_file] = max(files.get(vocab_file, 0.0), area.score * (1.0, 0.85, 0.7)[min(rank, 2)])
    return files
