"""MELETRA Automatic Study Engine: the student never chooses a method - MELETRA picks the best one that is available on this computer and checks what it made.

Choosing (``choose_engine``), in this order:

1. A local language model (Ollama, or an OpenAI-compatible server on this computer such as LM Studio): the lecture text stays on the computer.
2. The built-in offline method: always available, needs nothing installed.
3. A cloud model ONLY when the student gave a separate, explicit consent for cloud study (Settings -> Advanced -> Study engine) and saved a key. Lecture text is never sent to a cloud
   service automatically, and never because a key happens to exist.

Making (``AutomaticStudyEngine``): summaries and key concepts come from the validated facts of the lecture (fast, source-bound). Flashcards and questions are written by the language model
when one is available - in the manner of ``premium_prompts.py`` - and EVERY item, from any provider, must pass the premium gate (``curate.py``): a real question, an answer that says something,
evidence the lecture really contains, one defensible best answer with plausible wrong options, nothing near-identical. Refused items are replaced by a bounded number of repair rounds
(each round is told why its items were refused); what is still missing is filled from the built-in method, which passes the same gate. The student always gets a pack.
"""

from __future__ import annotations

import logging
import math
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Sequence

from . import curate
from . import text as T
from . import premium_prompts as PP
from .curate import GENERATION_VERSION
from .engine import StudyCancelled, StudyEngine, StudyOptions
from .extractive import ExtractiveProvider
from .grounding import resolve_sources
from .index import DocumentIndex
from .llm import OLLAMA_URL, OPENAI_URL, LLMProvider, OllamaProvider, Transport, is_local_url, urllib_transport
from .models import CARD_TYPES, QUESTION_TYPES, Chunk, Flashcard, Question, StudyPack, item_id
from .providers import AIProvider, ProviderConfig, ProviderError, StudyContext, create_provider

log = logging.getLogger(__name__)

ENGINE_NAME = "MELETRA Automatic Study Engine"
MAX_ROUNDS = 3                                  # the first request plus at most two repair rounds for one part of the lecture
TIME_BUDGET_SECONDS = 25 * 60                   # after this the rest of the pack is made by the built-in method
PROBE_SECONDS = 2.0
PREFERRED_LOCAL = ("qwen3:8b", "qwen3:14b", "qwen3:32b", "qwen2.5:14b", "qwen2.5:7b", "llama3.1:8b", "gemma3:12b", "gemma2:9b", "mistral", "qwen3:4b", "llama3.2", "phi4", "phi3")
NOT_FOR_TEXT = ("embed", "vl", "vision", "llava", "bge", "nomic", "minicpm", "whisper", "rerank", "moondream", "clip", "coder")
LLM_CARD_KINDS = ("concept", "mechanism", "cause_effect", "comparison", "clinical", "pathway", "exam_recall", "integration")
LLM_QUESTION_KINDS = ("mcq", "mechanism", "clinical_vignette", "short_answer", "explain_why", "true_false")


# ====================================================================================================================== choosing
@dataclass
class EngineChoice:
    provider: AIProvider
    mode: str                         # local | cloud | builtin
    model: str = ""
    reason: str = ""                  # one plain sentence: why this one (shown in the advanced diagnostics)
    notes: list[str] = field(default_factory=list)
    deferred: bool = False            # a better engine exists but waits (a transcription is using the GPU): the material is made again when the student asks (Regenerate)

    @property
    def label(self) -> str:
        if self.mode == "local":
            return f"{ENGINE_NAME} - local AI ({self.model})"
        if self.mode == "cloud":
            return f"{ENGINE_NAME} - cloud AI ({self.provider.name}): lecture text is sent to the provider, with your consent"
        return f"{ENGINE_NAME} - built-in offline method"

    def diagnostics(self) -> list[str]:
        lines = [f"Engine: {ENGINE_NAME} (recipe {GENERATION_VERSION})", f"Mode: {self.mode}" + (f" - {self.model}" if self.model else ""), f"Why: {self.reason}"]
        return lines + self.notes


def installed_ollama_models(base: str = OLLAMA_URL, transport: Optional[Transport] = None, timeout: float = PROBE_SECONDS) -> list[str]:
    """The models Ollama has installed; an empty list when Ollama is not running. Never raises."""
    try:
        data = (transport or urllib_transport)(base.rstrip("/") + "/api/tags", {}, None, timeout)
    except (ProviderError, OSError, ValueError):
        return []
    return [str(m.get("name", "")) for m in data.get("models", []) if isinstance(m, dict) and m.get("name")]


def pick_local_model(names: Sequence[str], wanted: str = "") -> str:
    """The installed model best suited to write study material: the student's own pick when it is installed, else the first of the preferred list, else any text model."""
    usable = [n for n in names if not any(bad in n.casefold() for bad in NOT_FOR_TEXT)]
    if wanted and wanted in names:
        return wanted
    for preferred in PREFERRED_LOCAL:
        for name in usable:
            if name == preferred or name.startswith(preferred + "-") or (":" not in preferred and name.split(":")[0] == preferred):
                return name
    return sorted(usable)[0] if usable else ""


def choose_engine(settings: Any, credentials: Any = None, transport: Optional[Transport] = None, busy: bool = False) -> EngineChoice:
    """The study engine for this run. ``settings`` is the app's Settings; ``credentials`` the CredentialService (for a cloud key); ``transport`` lets tests avoid the network.
    ``busy``: a transcription is running. A model on this computer would share the graphics card with the speech recogniser (and could make the transcription run out of memory), so
    none is asked for or even looked for: the built-in method makes the material, and it says so."""
    kind = str(getattr(settings, "study_provider", "builtin") or "builtin")
    model = str(getattr(settings, "study_model", "") or "")
    url = str(getattr(settings, "study_base_url", "") or "")
    consent = bool(getattr(settings, "study_cloud_consent", False))
    notes: list[str] = []
    if kind in ("openai", "anthropic"):
        local_server = kind == "openai" and is_local_url(url or OPENAI_URL)
        if (not busy) if local_server else consent:                      # consent applies to a cloud service only; a server on this computer shares the graphics card, so it waits while a transcription runs
            try:
                key = credentials.get_study_key(kind) if credentials is not None and not local_server else None
                provider = create_provider(ProviderConfig(kind=kind, model=model, base_url=url, timeout=300.0, context_words=1800), api_key=key, transport=transport)
                provider.check()
                if local_server:
                    return EngineChoice(provider, "local", provider.model, "an OpenAI-compatible server on this computer is set up in Advanced")
                return EngineChoice(provider, "cloud", provider.model, "cloud study is switched on (separate consent) and a key is saved")
            except ProviderError as exc:
                notes.append(f"The cloud/server study engine from Advanced could not be used ({exc}).")
        elif not local_server:
            notes.append("A cloud study provider is configured, but cloud study has not been consented to, so no lecture text is sent anywhere.")
    if busy:
        return EngineChoice(ExtractiveProvider(), "builtin", "", "a transcription is running; the local AI model waits so that the graphics card stays free for it", notes, deferred=True)
    base = url if kind == "ollama" and url else OLLAMA_URL
    names = installed_ollama_models(base, transport)
    picked = pick_local_model(names, model if kind == "ollama" else "")
    if picked:
        provider = OllamaProvider(ProviderConfig(kind="ollama", model=picked, base_url=base, timeout=300.0, temperature=0.2, context_words=1800), transport=transport)
        return EngineChoice(provider, "local", picked, f"Ollama is running on this computer with {len(names)} model{'s' if len(names) != 1 else ''}; {picked} suits study material", notes)
    reason = "no local language model was found (Ollama is not running or has no text model)" if not names else "Ollama has no text model suited to study material"
    return EngineChoice(ExtractiveProvider(), "builtin", "", reason + "; the built-in offline method is used", notes)


# ====================================================================================================================== reading the model's answer
def _ids(raw: Any) -> list[str]:
    if isinstance(raw, str):
        raw = [raw]
    return [str(i).strip().strip("[]") for i in raw] if isinstance(raw, list) else []


def _text(row: dict, key: str) -> str:
    return " ".join(str(row.get(key, "") or "").split())


def parse_cards(data: dict, topic: str, index: DocumentIndex) -> list[Flashcard]:
    out: list[Flashcard] = []
    for row in data.get("cards", []) if isinstance(data.get("cards"), list) else []:
        if not isinstance(row, dict):
            continue
        front, back = _text(row, "front"), _text(row, "back")
        if not front or not back:
            continue
        ctype = str(row.get("type", "")).strip()
        ctype = ctype if ctype in CARD_TYPES else "concept"
        difficulty = str(row.get("difficulty", "medium")).lower()
        out.append(Flashcard(id=item_id("c", ctype, front, back), type=ctype, front=front, back=back, sources=resolve_sources(_ids(row.get("sources")), index),
                             difficulty=difficulty if difficulty in ("easy", "medium", "hard") else "medium", topic=topic, evidence=_text(row, "evidence")))
    return out


def parse_questions(data: dict, topic: str, index: DocumentIndex) -> list[Question]:
    out: list[Question] = []
    for row in data.get("questions", []) if isinstance(data.get("questions"), list) else []:
        if not isinstance(row, dict):
            continue
        stem, answer = _text(row, "question"), _text(row, "answer")
        if not stem or not answer:
            continue
        options = [" ".join(str(o).split()) for o in row.get("options", []) if str(o).strip()] if isinstance(row.get("options"), list) else []
        qtype = str(row.get("type", "")).strip()
        if qtype not in QUESTION_TYPES:
            qtype = "mcq" if len(options) >= 3 else "short_answer"
        if qtype in ("mcq", "mechanism", "clinical_vignette") and options:
            options = list(dict.fromkeys(options))
            random.Random(item_id("q", qtype, stem)).shuffle(options)           # models like to put the right answer second: the position is decided here, the same way every time
        difficulty = str(row.get("difficulty", "medium")).lower()
        out.append(Question(id=item_id("q", qtype, stem, answer), type=qtype, question=stem, answer=answer, explanation=_text(row, "explanation"), options=options,
                            sources=resolve_sources(_ids(row.get("sources")), index), difficulty=difficulty if difficulty in ("easy", "medium", "hard") else "medium",
                            topic=topic, evidence=_text(row, "evidence")))
    return out


def _clinical(chunks: Sequence[Chunk]) -> bool:
    """The passages talk about patients, diseases, deficiencies or treatment (at least two clinical cues)."""
    return T.cue_count(" ".join(c.text for c in chunks), T.CLINICAL_CUES) >= 2


def _position(item) -> float:
    starts = [r.start for r in item.sources if r.start is not None]
    return min(starts) if starts else 0.0


# ====================================================================================================================== making
class AutomaticStudyEngine(StudyEngine):
    option_counts = (4, 5)

    def __init__(self, choice: EngineChoice) -> None:
        super().__init__(choice.provider if choice.mode != "builtin" else None)
        self.choice = choice
        self.stats: dict[str, Any] = {}
        self.engine_notes: list[str] = []                                  # which model, which fallback: Settings -> Advanced shows them, the Study screens do not
        self._deadline = 0.0
        self._card_fronts: list[str] = []                                  # the questions must not repeat the flashcards

    # ---- hooks of the base engine
    def _context(self, index: DocumentIndex, options: StudyOptions) -> StudyContext:
        ctx = super()._context(index, options)
        ctx.cache["premium"] = True                                       # the built-in method writes no fill-in-the-blank items and applies the premium gate itself
        return ctx

    def _summaries(self, kind, index, scopes, ctx, cancelled):
        """Summaries come from the validated facts of the lecture (the built-in method): quick, source-bound and the same every time."""
        active = self.active
        self.active = self.fallback
        try:
            return super()._summaries(kind, index, scopes, ctx, cancelled)
        finally:
            self.active = active

    def _switch(self, exc: Exception) -> None:
        before = len(self.warnings)
        super()._switch(exc)
        self.engine_notes += self.warnings[before:]

    def generate(self, index: DocumentIndex, **kwargs: Any) -> StudyPack:
        self.engine_notes = []
        self._card_fronts = []
        self.stats = {"engine": ENGINE_NAME, "version": GENERATION_VERSION, "mode": self.choice.mode, "model": self.choice.model, "flashcards": self._area(), "questions": self._area()}
        self._deadline = time.monotonic() + TIME_BUDGET_SECONDS
        started = time.monotonic()
        try:
            pack = super().generate(index, **kwargs)
        finally:
            self._release()                                                      # also when the work was cancelled or failed: a transcription may be waiting for the graphics card
        self.stats["seconds"] = round(time.monotonic() - started, 1)
        for area in ("flashcards", "questions"):
            self.stats[area]["reasons"] = dict(self.stats[area]["reasons"].most_common(15))
        pack.warnings = [w for w in pack.warnings if w not in self.engine_notes]
        self.stats["notes"] = list(dict.fromkeys(self.engine_notes))
        pack.stats["premium"] = self.stats
        pack.generation = {"engine": ENGINE_NAME, "version": GENERATION_VERSION, "mode": self.choice.mode, "model": self.choice.model, "language": pack.language,
                           "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
        if self.choice.deferred:
            pack.generation["deferred"] = True
        return pack

    def _release(self) -> None:
        """The model was loaded for this work: free the graphics card as soon as it is done (best effort; the next request loads it again)."""
        release = getattr(self.choice.provider, "release", None)
        if callable(release):
            try:
                release()
            except Exception:  # noqa: BLE001 - nothing may fail because of a tidy-up
                log.debug("could not release the model", exc_info=True)

    # ---- bookkeeping
    @staticmethod
    def _area() -> dict[str, Any]:
        return {"asked": 0, "received": 0, "kept": 0, "rounds": 0, "from_model": 0, "from_builtin": 0, "reasons": Counter()}

    def _tally(self, area: str, source: str, received: int, curation: curate.Curation, rounds: int = 0) -> None:
        stat = self.stats[area]
        stat["received"] += received
        stat["kept"] += len(curation.kept)
        stat["rounds"] += rounds
        stat["from_model" if source == "model" else "from_builtin"] += len(curation.kept)
        stat["reasons"].update(curation.reasons())

    def _model(self) -> Optional[LLMProvider]:
        """The language model, as long as it is still the active provider and the time budget is not used up."""
        if isinstance(self.active, LLMProvider) and time.monotonic() < self._deadline:
            return self.active
        return None

    def _model_failed(self, exc: Exception) -> None:
        self._switch(exc if isinstance(exc, ProviderError) else ProviderError(f"{exc.__class__.__name__}: {exc}"))

    @staticmethod
    def _problems(curation: curate.Curation, ungrounded: int) -> list[str]:
        problems = [reason for reason, _n in curation.reasons().most_common(3)]
        if ungrounded:
            problems.append("statements that the passages do not support")
        return problems

    # ---- flashcards
    def _cards(self, index, scopes, options, total, ctx, cancelled):
        per = max(2, total // max(len(scopes), 1))
        kept: list[Flashcard] = []
        for name, chunks in scopes:
            if cancelled and cancelled():
                raise StudyCancelled()
            model = self._model()
            if model is not None:
                try:
                    kept += self._model_cards(model, name, chunks, per, options, ctx, kept, cancelled)
                    continue
                except StudyCancelled:
                    raise
                except Exception as exc:                                   # noqa: BLE001 - a broken model must never leave the Study tabs empty
                    self._model_failed(exc)
            kept += self._builtin_cards(name, chunks, per, options, ctx, kept)
        if self.choice.mode != "builtin" and len(kept) < max(4, total // 3):
            self.stats["flashcards"].setdefault("supplemented", True)
            self.engine_notes.append("The language model gave few usable flashcards, so the built-in method added some.")
            for name, chunks in scopes:
                kept += self._builtin_cards(name, chunks, per, options, ctx, kept)
        kept.sort(key=_position)
        self._card_fronts = [c.front for c in kept[:total]]
        return kept[:total]

    def _model_cards(self, model: LLMProvider, name: str, chunks: Sequence[Chunk], per: int, options: StudyOptions, ctx: StudyContext, existing: Sequence[Flashcard],
                     cancelled: Optional[Callable[[], bool]]) -> list[Flashcard]:
        kinds = [k for k in LLM_CARD_KINDS if k in options.card_types] or list(LLM_CARD_KINDS)
        made: list[Flashcard] = []
        batches = model._batches(chunks)
        for batch in batches:
            if not _clinical(batch):
                kinds_here = [k for k in kinds if k != "clinical"] or kinds              # no clinical cards from passages with nothing clinical in them
            else:
                kinds_here = kinds
            want = model._split_limit(per, batches, batch)
            got: list[Flashcard] = []
            problems: list[str] = []
            for attempt in range(MAX_ROUNDS):
                if cancelled and cancelled():
                    raise StudyCancelled()
                need = want - len(got)
                ask = need + (2 if attempt == 0 else 1)
                taken = [c.front for c in (*existing, *made, *got)]
                data = model._ask(PP.flashcard_prompt(batch, name, kinds_here, ask, ctx.language, problems, taken, ctx.index.unresolved), ctx, PP.SYSTEM)
                self.stats["flashcards"]["asked"] += ask
                raw = parse_cards(data, name, ctx.index)
                grounded = self._ground(raw, ctx, model)
                curation = curate.curate_cards(grounded, ctx.index, existing=[*existing, *made, *got])
                curation.kept = curation.kept[:need]
                self._tally("flashcards", "model", len(raw), curation, rounds=1)
                got += curation.kept
                if len(got) >= math.ceil(0.8 * want) or not raw:
                    break
                problems = self._problems(curation, len(raw) - len(grounded))
            made += got
        return made

    def _builtin_cards(self, name: str, chunks: Sequence[Chunk], per: int, options: StudyOptions, ctx: StudyContext, existing: Sequence[Flashcard]) -> list[Flashcard]:
        raw = self.fallback.generate_flashcards(chunks, topic=name, types=options.card_types, limit=per, ctx=ctx)
        grounded = self._ground(raw, ctx, self.fallback)
        curation = curate.curate_cards(grounded, ctx.index, existing=existing)
        self._tally("flashcards", "builtin", len(raw), curation)
        return curation.kept

    # ---- questions
    def _questions(self, index, scopes, options, total, ctx, cancelled):
        per = max(2, total // max(len(scopes), 1))
        kept: list[Question] = []
        for name, chunks in scopes:
            if cancelled and cancelled():
                raise StudyCancelled()
            model = self._model()
            if model is not None:
                try:
                    kept += self._model_questions(model, name, chunks, per, options, ctx, kept, cancelled)
                    continue
                except StudyCancelled:
                    raise
                except Exception as exc:                                   # noqa: BLE001
                    self._model_failed(exc)
            kept += self._builtin_questions(name, chunks, per, options, ctx, kept)
        if self.choice.mode != "builtin" and len(kept) < max(3, total // 3):
            self.stats["questions"].setdefault("supplemented", True)
            self.engine_notes.append("The language model gave few usable questions, so the built-in method added some.")
            for name, chunks in scopes:
                kept += self._builtin_questions(name, chunks, per, options, ctx, kept)
        kept = self._balanced(kept)
        kept.sort(key=_position)
        return kept[:total]

    def _balanced(self, questions: list[Question]) -> list[Question]:
        """True/False questions are the easiest to write and the least to learn from: at most one in four of the pack (and two when there is little else). The ones kept alternate
        True and False so the answer is never guessable from the pattern."""
        others = [q for q in questions if q.type != "true_false"]
        flags = [q for q in questions if q.type == "true_false"]
        allowed = max(2, len(others) // 3)
        if len(flags) > allowed:
            falses, trues = [q for q in flags if q.answer == "False"], [q for q in flags if q.answer != "False"]
            chosen: list[Question] = []
            while len(chosen) < allowed and (falses or trues):
                chosen.append((falses if (len(chosen) % 2 == 0 and falses) or not trues else trues).pop(0))
            dropped = len(flags) - len(chosen)
            self.stats["questions"]["reasons"]["too many true/false questions"] += dropped
            flags = chosen
        return others + flags

    def _model_questions(self, model: LLMProvider, name: str, chunks: Sequence[Chunk], per: int, options: StudyOptions, ctx: StudyContext, existing: Sequence[Question],
                         cancelled: Optional[Callable[[], bool]]) -> list[Question]:
        kinds = [k for k in LLM_QUESTION_KINDS if k in options.question_types] or list(LLM_QUESTION_KINDS)
        made: list[Question] = []
        batches = model._batches(chunks)
        for batch in batches:
            kinds_here = kinds if _clinical(batch) else ([k for k in kinds if k != "clinical_vignette"] or kinds)         # a patient scenario needs clinical facts to be built from
            want = model._split_limit(per, batches, batch)
            got: list[Question] = []
            problems: list[str] = []
            for attempt in range(MAX_ROUNDS):
                if cancelled and cancelled():
                    raise StudyCancelled()
                need = want - len(got)
                ask = need + (2 if attempt == 0 else 1)
                taken = [q.question for q in (*existing, *made, *got)]
                data = model._ask(PP.question_prompt(batch, name, kinds_here, ask, ctx.language, problems, taken, ctx.index.unresolved), ctx, PP.SYSTEM)
                self.stats["questions"]["asked"] += ask
                raw = parse_questions(data, name, ctx.index)
                grounded = self._ground(raw, ctx, model)
                curation = curate.curate_questions(grounded, ctx.index, existing=[*existing, *made, *got], unlike=self._card_fronts, true_false_share=1.0)
                curation.kept = curation.kept[:need]
                self._tally("questions", "model", len(raw), curation, rounds=1)
                got += curation.kept
                if len(got) >= math.ceil(0.8 * want) or not raw:
                    break
                problems = self._problems(curation, len(raw) - len(grounded))
            made += got
        return made

    def _builtin_questions(self, name: str, chunks: Sequence[Chunk], per: int, options: StudyOptions, ctx: StudyContext, existing: Sequence[Question]) -> list[Question]:
        raw = self.fallback.generate_questions(chunks, topic=name, types=options.question_types, limit=per, ctx=ctx)
        grounded = self._ground(raw, ctx, self.fallback)
        curation = curate.curate_questions(grounded, ctx.index, existing=existing, unlike=self._card_fronts, true_false_share=1.0)
        self._tally("questions", "builtin", len(raw), curation)
        return curation.kept
