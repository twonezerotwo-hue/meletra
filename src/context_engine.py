"""Medical Context Engine: builds the lecture context and picks the few terms worth sending to the model.

Evidence -> context:
    1. media file name            (``biochem_lecture3.mp4`` -> biochemistry)
    2. the first minutes' text    (probe transcript, or the chunks heard so far)
    3. lecture slides (PDF)       (terms, acronyms, rare proper terms)
    4. the user's --subject

The subject classifier ranks 19 areas, the strongest 1-3 select the vocabulary files, every candidate
term is scored by the priority engine, and only the best terms go to the decoder (a real-tokenizer
token budget makes the final cut). A neuroscience lecture therefore never receives the pharmacology list.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .classifier import AreaScore, SubjectClassifier, vocab_file_scores
from .knowledge_subjects import AREAS
from .priority import TermPriorityEngine, TermScore
from .vocabulary import DEFAULT_TERMS_DIR, VocabularyIndex, available_subjects, load_terms, normalize_key

log = logging.getLogger(__name__)

GENERIC_FILE = "general_medical"
GENERIC_FILE_SCORE = 0.25          # generic clinical vocabulary is always a low-priority fallback


@dataclass
class MedicalContext:
    language: Optional[str] = None
    evidence: dict[str, Any] = field(default_factory=dict)
    area_scores: list[AreaScore] = field(default_factory=list)
    selected: list[AreaScore] = field(default_factory=list)
    confident: bool = False
    vocab_files: dict[str, float] = field(default_factory=dict)
    ranked: list[TermScore] = field(default_factory=list)
    acronyms: list[dict[str, Any]] = field(default_factory=list)
    rolling: list[dict[str, Any]] = field(default_factory=list)
    candidate_count: int = 0

    @property
    def selected_areas(self) -> list[str]:
        return [a.area for a in self.selected]

    def hotword_candidates(self, limit: int = 60) -> list[str]:
        return [t.term for t in self.ranked[:limit]]

    def to_dict(self, top: int = 80) -> dict[str, Any]:
        return {
            "language": self.language, "evidence": self.evidence, "confident": self.confident,
            "selected_areas": [a.to_dict() for a in self.selected],
            "area_scores": {a.area: round(a.score, 3) for a in self.area_scores if a.score > 0},
            "vocabulary_files": {k: round(v, 3) for k, v in self.vocab_files.items()},
            "candidate_terms": self.candidate_count,
            "ranked_terms": [t.to_dict() for t in self.ranked[:top]],
            "acronyms": self.acronyms, "rolling_updates": self.rolling,
        }


class MedicalContextEngine:
    def __init__(
        self, cfg: Mapping[str, Any], terms_dir: Path = DEFAULT_TERMS_DIR, language: Optional[str] = None,
        user_subjects: Sequence[str] = (), filename: Optional[str] = None, pdf_terms: Sequence[str] = (),
        pdf_text: str = "", extra_terms: Sequence[str] = (),
    ) -> None:
        engine_cfg = dict(cfg.get("context_engine", {}))
        self.cfg = engine_cfg
        self.terms_dir = Path(terms_dir)
        self.language = language
        self.user_subjects = [s for s in user_subjects if s]
        self.filename = filename
        self.pdf_terms = list(pdf_terms)
        self.pdf_text = pdf_text
        self.extra_terms = list(extra_terms)
        self.classifier = SubjectClassifier(self.terms_dir, language, float(engine_cfg.get("saturation", 12.0)))
        self.priority = TermPriorityEngine(engine_cfg.get("weights"), engine_cfg.get("multiword_bonus"))
        self.index_all = VocabularyIndex(load_terms("all", language, self.terms_dir) + list(self.pdf_terms))
        self._candidates: dict[str, set[str]] = {}
        self._file_scores: dict[str, float] = {}
        self._early_counts: Counter = Counter()
        self._pdf_counts: Counter = Counter()
        self._context = MedicalContext(language=language)

    # ------------------------------------------------------------------ build
    def _user_vocab_files(self) -> list[str]:
        files = set(available_subjects(self.terms_dir))
        return [s for s in self.user_subjects if s in files]

    def build(self, early_text: str = "") -> MedicalContext:
        cfg = self.cfg
        # "all" keeps the old behaviour: every vocabulary file, no focusing
        use_all = any(s.lower() == "all" for s in self.user_subjects)
        classifier_subjects = [s for s in self.user_subjects if s.lower() != "all"]
        scores = self.classifier.score(filename=self.filename, early_text=early_text, pdf_text=self.pdf_text,
                                       user_subjects=classifier_subjects)
        selected = self.classifier.select(
            scores, int(cfg.get("max_areas", 3)), float(cfg.get("relative_cutoff", 0.5)), float(cfg.get("min_area_score", 0.2)))
        confident = bool(selected)
        file_scores = vocab_file_scores(selected)
        for vocab_file in self._user_vocab_files():                 # --subject neuroscience names a file directly
            file_scores[vocab_file] = max(file_scores.get(vocab_file, 0.0), 0.9)
        if use_all:
            file_scores = {f: 0.5 for f in available_subjects(self.terms_dir)}
        file_scores[GENERIC_FILE] = max(file_scores.get(GENERIC_FILE, 0.0), GENERIC_FILE_SCORE)

        candidates: dict[str, set[str]] = {}
        for vocab_file in file_scores:
            for term in load_terms(vocab_file, self.language, self.terms_dir):
                candidates.setdefault(term, set()).add(vocab_file)
        for term in self.pdf_terms:
            candidates.setdefault(term, set()).add("pdf")
        for term in self.extra_terms:
            candidates.setdefault(term, set()).add("extra")

        self._early_counts = Counter(self.index_all.find_terms(early_text)) if early_text else Counter()
        self._pdf_counts = Counter(self.index_all.find_terms(self.pdf_text)) if self.pdf_text else Counter()
        for term in self._early_counts:                              # spoken terms are candidates even from other files
            candidates.setdefault(term, set()).add("heard")
        self._candidates, self._file_scores = candidates, file_scores

        ctx = self._context
        ctx.area_scores, ctx.selected, ctx.confident = scores, selected, confident
        ctx.vocab_files = file_scores
        ctx.candidate_count = len(candidates)
        ctx.evidence = {
            "filename": self.filename, "filename_areas": [a for a in AREAS if any(
                a == s.area and "filename" in s.by_source for s in scores)],
            "early_transcript_words": len(early_text.split()), "pdf_terms": len(self.pdf_terms),
            "pdf_characters": len(self.pdf_text), "user_subjects": self.user_subjects,
        }
        self._rerank()
        log.info("Medical context: areas %s (confident=%s) | %d candidates -> top: %s",
                 [f"{a.area}:{a.score:.2f}" for a in selected] or "none", confident, len(candidates),
                 ", ".join(self._context.hotword_candidates(8)))
        return ctx

    def _rerank(self) -> None:
        self._context.ranked = self.priority.rank(
            self._candidates, self._file_scores, pdf_terms=self.pdf_terms, pdf_text=self.pdf_text,
            early_counts=self._early_counts, pdf_counts=self._pdf_counts)

    # ------------------------------------------------------------------ rolling adaptation between chunks
    def update_with_heard(self, text: str, chunk_index: int) -> dict[str, Any]:
        """Fold the terms heard in a finished chunk into the context for the next chunk.

        Small by design: only dictionary terms are counted (never the transcript itself), at most
        ``rolling_new_terms`` previously unseen terms are added, and the token budget still applies later.
        """
        limit = int(self.cfg.get("rolling_new_terms", 12))
        heard = Counter(self.index_all.find_terms(text))
        new_terms = [t for t in heard if self._early_counts.get(t, 0) == 0]
        for term, count in heard.items():
            self._early_counts[term] += count
        added: list[str] = []
        for term in sorted(new_terms, key=lambda t: -heard[t])[:limit]:
            self._candidates.setdefault(term, set()).add("heard")
            added.append(term)
        self._rerank()
        entry = {"chunk": chunk_index, "terms_heard": len(heard), "new_terms": added,
                 "top_now": self._context.hotword_candidates(10)}
        self._context.rolling.append(entry)
        log.info("Context adapted after chunk %d: %d terms heard, %d new -> %s", chunk_index, len(heard), len(added), added[:6])
        return entry

    def add_acronyms(self, acronyms: Sequence[dict[str, Any]]) -> None:
        """Learned acronyms (from slides / early speech) become low-cost candidates."""
        self._context.acronyms = list(acronyms)
        for item in acronyms:
            short = item.get("acronym")
            if short and len(short) >= 2:
                self._candidates.setdefault(short, set()).add("acronym")
                self._early_counts[short] += int(item.get("occurrences", 1))
        self._rerank()

    @property
    def context(self) -> MedicalContext:
        return self._context

    def heard_terms(self, limit: int = 40) -> list[str]:
        """Dictionary terms heard so far (probe + finished chunks), most frequent first."""
        return [t for t, _ in self._early_counts.most_common(limit)]
