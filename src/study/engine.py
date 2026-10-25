"""The study engine: index -> provider (with fallback) -> grounding -> :class:`StudyPack`.

Design rules:

* The student can always study. A configured language model that is not installed, not running, out of quota or answering nonsense
  never leaves the Study tabs empty: the built-in provider takes over for the rest of the run and the pack says so (``warnings``,
  ``provider.fallback_from``).
* Nothing is shown that the lecture does not support (``grounding.py``).
* The engine is independent of the transcription pipeline and of the GUI: it takes an index and returns a pack, reporting progress
  through a plain callback and honouring a cancel callback.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional, Sequence

from .extractive import ExtractiveProvider
from .grounding import GroundingReport, ground_items
from .index import DocumentIndex
from .models import (CARD_TYPES, QUESTION_TYPES, SUMMARY_KINDS, SUMMARY_TITLES, Chunk, Flashcard, ProviderInfo, Question, StudyPack,
                     SummaryItem)
from .providers import NOT_IN_SOURCES, AIProvider, Evaluation, ProviderError, StudyCancelled, StudyContext

log = logging.getLogger(__name__)

#: kinds written once for the whole lecture (value: how many items) ...
GLOBAL_KINDS = {"quick": 8, "key_concepts": 24, "definitions": 20, "emphasis": 14, "exam": 14}
#: ... and kinds written for every topic (value: items per topic)
TOPIC_KINDS = {"detailed": 4, "high_yield": 3, "mechanisms": 2, "clinical": 2}

ProgressFn = Callable[[float, str], None]


@dataclass
class StudyOptions:
    summary_kinds: Sequence[str] = SUMMARY_KINDS
    card_types: Sequence[str] = CARD_TYPES
    question_types: Sequence[str] = QUESTION_TYPES
    max_cards: int = 0                            # 0 = decide from the length of the lecture
    max_questions: int = 0
    language: str = "en"
    strict: bool = True                           # SOURCE-BOUND
    allow_external: bool = False                  # External Knowledge mode: not available yet, always off

    def sizes(self, index: DocumentIndex) -> tuple[int, int]:
        words = sum(len(c.text.split()) for c in index.chunks)
        cards = self.max_cards or min(max(round(words / 140), 12), 90)
        questions = self.max_questions or min(max(round(words / 220), 10), 60)
        return cards, questions


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _dedupe(items: list, key: Callable) -> list:
    seen, out = set(), []
    for item in items:
        k = key(item)
        if k not in seen:
            seen.add(k)
            out.append(item)
    return out


class StudyEngine:
    def __init__(self, provider: Optional[AIProvider] = None, fallback: Optional[AIProvider] = None) -> None:
        self.primary: AIProvider = provider or ExtractiveProvider()
        self.fallback: AIProvider = fallback or ExtractiveProvider()
        self.active: AIProvider = self.primary
        self.warnings: list[str] = []
        self.failed_from: str = ""
        self.report = GroundingReport()

    # ------------------------------------------------------------------------------------------------------------ provider handling
    def _start(self) -> None:
        self.active = self.primary
        self.warnings, self.failed_from, self.report = [], "", GroundingReport()
        try:
            self.primary.check()
        except ProviderError as exc:
            self._switch(exc)

    def _switch(self, exc: Exception) -> None:
        if self.active is self.fallback or self.primary is self.fallback:
            return
        message = str(exc) or exc.__class__.__name__
        log.warning("AI provider %s failed: %s", self.primary.name, message)
        self.failed_from = self.primary.name
        self.warnings.append(f"{self.primary.name} could not be used ({message}). The built-in method made the rest of the study material.")
        self.active = self.fallback

    def _call(self, fn: Callable[[AIProvider], list]) -> list:
        """Run ``fn(provider)``; on any provider failure switch to the built-in provider for the rest of the run and retry once."""
        try:
            return fn(self.active)
        except StudyCancelled:
            raise
        except Exception as exc:                  # noqa: BLE001 - a broken provider must never break studying
            if self.active is self.fallback:
                log.exception("built-in study provider failed")
                self.warnings.append(f"A part of the study material could not be made ({exc.__class__.__name__}).")
                return []
            self._switch(exc if isinstance(exc, ProviderError) else ProviderError(f"{exc.__class__.__name__}: {exc}"))
            return self._call(fn)

    option_counts: tuple[int, ...] = (5,)                    # how many options a multiple-choice question may have (the Automatic Study Engine allows four)

    def _context(self, index: DocumentIndex, options: "StudyOptions") -> StudyContext:
        return StudyContext(index=index, language=options.language, strict=options.strict, allow_external=False)

    def _ground(self, items: list, ctx: StudyContext, provider: Optional[AIProvider] = None) -> list:
        provider = provider or self.active
        kept, report = ground_items(items, ctx.index, strict=ctx.strict, check_text=provider.kind != "builtin", option_counts=self.option_counts)
        self.report.merge(report)
        return kept

    def _section(self, produce: Callable[[AIProvider], list], ctx: StudyContext) -> list:
        """One section of the pack: produce, ground; if a language model gave nothing usable, try the built-in provider for it."""
        provider = self.active
        items = self._ground(self._call(produce), ctx, self.active)
        if not items and provider is not self.fallback and self.active is not self.fallback:
            items = self._ground(self._call(lambda _p: produce(self.fallback)), ctx, self.fallback)
            if items:
                self.warnings.append(f"{provider.name} gave nothing usable for one section; the built-in method filled it in.")
        return items

    # ------------------------------------------------------------------------------------------------------------------ generation
    def generate(self, index: DocumentIndex, *, lecture_id: str = "", title: str = "", options: Optional[StudyOptions] = None,
                 progress: Optional[ProgressFn] = None, cancelled: Optional[Callable[[], bool]] = None,
                 only: Optional[set[str]] = None, base: Optional[StudyPack] = None) -> StudyPack:
        """Make the study pack. ``only`` limits the work to some parts ({"summary:high_yield", "flashcards", "questions"}) and
        ``base`` is the pack they are merged into (used to regenerate one tab)."""
        options = options or StudyOptions()
        started = time.monotonic()
        self._start()
        ctx = self._context(index, options)
        pack = base or StudyPack(lecture_id=lecture_id, title=title, source_mode=index.mode, language=options.language)
        pack.created, pack.source_mode, pack.language = _now(), index.mode, options.language
        pack.source_bound = options.strict
        pack.topics = index.topic_names()

        kinds = [k for k in options.summary_kinds if k in SUMMARY_KINDS and (only is None or f"summary:{k}" in only)]
        want_cards = only is None or "flashcards" in only
        want_questions = only is None or "questions" in only
        steps = len(kinds) + int(want_cards) + int(want_questions)
        done = 0

        def tick(message: str) -> None:
            nonlocal done
            if cancelled and cancelled():
                raise StudyCancelled()
            if progress:
                progress(done / max(steps, 1), message)

        topic_chunks = self._topic_scopes(index)
        for kind in kinds:
            tick(f"Writing: {SUMMARY_TITLES[kind]}")
            pack.summaries[kind] = self._summaries(kind, index, topic_chunks, ctx, cancelled)
            done += 1
        cards, questions = options.sizes(index)
        if want_cards:
            tick("Making flashcards")
            pack.flashcards = self._cards(index, topic_chunks, options, cards, ctx, cancelled)
            done += 1
        if want_questions:
            tick("Making questions")
            pack.questions = self._questions(index, topic_chunks, options, questions, ctx, cancelled)
            done += 1
        if progress:
            progress(1.0, "Study material ready")
        pack.summaries = {k: v for k, v in pack.summaries.items() if v}
        pack.provider = self._provider_info()
        pack.warnings = list(dict.fromkeys(self.warnings))
        pack.stats = {"seconds": round(time.monotonic() - started, 2), "chunks": len(index), "topics": len(index.topics),
                      "summaries": {k: len(v) for k, v in pack.summaries.items()}, "flashcards": len(pack.flashcards),
                      "questions": len(pack.questions), "grounding": self.report.to_dict()}
        self._quality(pack, index, ctx)
        return pack

    def _quality(self, pack: StudyPack, index: DocumentIndex, ctx: StudyContext) -> None:
        """Fact-based material (English): the overview, what the validation kept and dropped, the audit of the delivered items, and a warning when the recording gave little."""
        study = ctx.cache.get("factstudy")
        if study is None:
            return
        try:
            from .audit import audit_pack

            pack.overview = study.overview(pack)
            pack.stats["facts"] = study.base.reject_report()
            pack.stats["quality_gate"] = study.gate.to_dict()
            pack.stats["quality"] = audit_pack(pack, index, study)
            facts = pack.overview["facts"]
            uncovered = pack.stats["quality"].get("uncovered_topics") or []
            if uncovered:
                self.warnings.append("No reliable study statement could be made for " + ("this part" if len(uncovered) == 1 else f"these {len(uncovered)} parts") + " of the lecture: "
                                     + "; ".join(uncovered) + ". The recording is unclear there - read or listen to it in the transcript.")
                pack.warnings = list(dict.fromkeys(self.warnings))
            if facts["validated"] < 8 or facts["yield"] < 0.12:
                self.warnings.append(f"Only {facts['validated']} of {facts['sentences']} sentences could be turned into reliable study statements, so this pack is short: the transcript is "
                                     "noisy or conversational. A clearer recording, the lecture slides (PDF) or a reviewed transcript give more and better material.")
                pack.warnings = list(dict.fromkeys(self.warnings))
        except Exception:                          # noqa: BLE001 - the audit must never break studying
            log.exception("study quality report failed")

    def _provider_info(self) -> ProviderInfo:
        if self.failed_from:
            info = self.fallback.info()
            info.fallback_from = self.failed_from
            return info
        return self.primary.info()

    @staticmethod
    def _topic_scopes(index: DocumentIndex) -> list[tuple[str, list[Chunk]]]:
        scopes = [(t.name, index.chunks_of_topic(t.name)) for t in index.topics]
        scopes = [(n, c) for n, c in scopes if c]
        return scopes or [("", list(index.chunks))]

    def _summaries(self, kind: str, index: DocumentIndex, scopes: list[tuple[str, list[Chunk]]], ctx: StudyContext,
                   cancelled: Optional[Callable[[], bool]]) -> list[SummaryItem]:
        if kind in GLOBAL_KINDS:
            limit = GLOBAL_KINDS[kind]
            return _dedupe(self._section(lambda p: p.summarize(kind, index.chunks, topic="", limit=limit, ctx=ctx), ctx),
                           lambda i: i.text.casefold())
        per_topic = TOPIC_KINDS.get(kind, 3)
        out: list[SummaryItem] = []
        for name, chunks in scopes:
            if cancelled and cancelled():
                raise StudyCancelled()
            limit = max(per_topic, round(per_topic * len(chunks) / 8))          # a long topic gets more points than a short one
            out.extend(self._section(lambda p, n=name, c=chunks, k=limit: p.summarize(kind, c, topic=n, limit=k, ctx=ctx), ctx))
        return _dedupe(out, lambda i: i.text.casefold())

    def _cards(self, index: DocumentIndex, scopes: list[tuple[str, list[Chunk]]], options: StudyOptions, total: int, ctx: StudyContext,
               cancelled: Optional[Callable[[], bool]]) -> list[Flashcard]:
        per = max(3, math.ceil(total / max(len(scopes), 1)))
        out: list[Flashcard] = []
        for name, chunks in scopes:
            if cancelled and cancelled():
                raise StudyCancelled()
            out.extend(self._section(lambda p, n=name, c=chunks: p.generate_flashcards(c, topic=n, types=options.card_types, limit=per, ctx=ctx), ctx))
        return _dedupe(out, lambda c: c.id)[:total]

    def _questions(self, index: DocumentIndex, scopes: list[tuple[str, list[Chunk]]], options: StudyOptions, total: int,
                   ctx: StudyContext, cancelled: Optional[Callable[[], bool]]) -> list[Question]:
        per = max(2, math.ceil(total / max(len(scopes), 1)))
        out: list[Question] = []
        for name, chunks in scopes:
            if cancelled and cancelled():
                raise StudyCancelled()
            out.extend(self._section(lambda p, n=name, c=chunks: p.generate_questions(c, topic=n, types=options.question_types, limit=per, ctx=ctx), ctx))
        return _dedupe(out, lambda q: q.id)[:total]

    # ------------------------------------------------------------------------------------------------------------ ask / evaluate
    def explain(self, index: DocumentIndex, question: str, *, language: str = "en", strict: bool = True) -> SummaryItem:
        """Answer a question from the lecture (retrieval, then the provider). Unsupported answers become "not covered"."""
        self._start()
        ctx = StudyContext(index=index, language=language, strict=strict)
        chunks = [c for c, _s in index.search(question, k=6)]
        if not chunks:
            return SummaryItem(text=NOT_IN_SOURCES, sources=[])
        result = self._call(lambda p: [p.explain(question, chunks, ctx)])[0:1]
        if not result:
            return SummaryItem(text=NOT_IN_SOURCES, sources=[])
        answer = result[0]
        if answer.text == NOT_IN_SOURCES:
            return answer
        kept = self._ground([answer], ctx)
        return kept[0] if kept else SummaryItem(text=NOT_IN_SOURCES, sources=[])

    def evaluate(self, index: DocumentIndex, question: Question, answer: str, *, language: str = "en") -> Evaluation:
        """Judge a typed answer. If the AI fails the built-in comparison judges instead - a quiz never gets stuck on a provider."""
        ctx = StudyContext(index=index, language=language)
        chunks = [c for c in (index.get(r.chunk_id) for r in question.sources) if c]
        if self.active is self.fallback or self.primary.kind == "builtin":
            return self.fallback.evaluate_answer(question, answer, chunks, ctx)
        try:
            return self.primary.evaluate_answer(question, answer, chunks, ctx)
        except Exception as exc:                   # noqa: BLE001
            log.warning("AI answer evaluation failed (%s); using the built-in comparison", exc)
            return self.fallback.evaluate_answer(question, answer, chunks, ctx)
