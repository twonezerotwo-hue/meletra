"""The provider-independent AI interface of the study system.

The study features never talk to a particular model. They talk to an :class:`AIProvider`, which can be

* the **built-in extractive provider** (``extractive.py``): rule based, offline, needs nothing installed, quotes the lecture;
* a **local** language model (Ollama, or any OpenAI-compatible local server such as LM Studio / llama.cpp);
* a **cloud** model (OpenAI-compatible endpoints, Anthropic) - only when the student configured it and supplied a key.

This AI is separate from the transcription engine: it never runs while a transcription runs, and when it fails, the transcript is
untouched (the study engine falls back to the built-in provider, see ``engine.py``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from .index import DocumentIndex
from .models import Chunk, Flashcard, ProviderInfo, Question, SummaryItem

NOT_IN_SOURCES = "This is not covered in your lecture sources."


class StudyCancelled(Exception):
    """The student cancelled (or closed the window) while the study material was being made."""


class ProviderError(Exception):
    """The provider could not do its job (not installed, not running, bad key, bad answer, time-out ...). The message is meant to be
    shown to the student; the study engine catches it and uses the built-in provider instead."""


@dataclass
class StudyContext:
    """What every provider call needs to know besides the text it is given."""

    index: DocumentIndex
    language: str = "en"                          # language the output should be written in ("en", "tr", ...)
    strict: bool = True                           # SOURCE-BOUND: say only what the sources say
    allow_external: bool = False                  # reserved for the later "External Knowledge" mode; off by default
    cache: dict[str, Any] = field(default_factory=dict)   # per-run scratch space (the built-in provider keeps its analysis here)


@dataclass
class Evaluation:
    correct: bool
    score: float                                  # 0..1
    feedback: str = ""
    missing: list[str] = field(default_factory=list)


class AIProvider(ABC):
    """Everything the study features need from an AI. Implementations must only use the ``chunks`` they are given."""

    name = "provider"
    kind = "builtin"                              # builtin | local | cloud
    model = ""

    def info(self) -> ProviderInfo:
        return ProviderInfo(name=self.name, kind=self.kind, model=self.model)

    def check(self) -> None:
        """Raise :class:`ProviderError` when the provider cannot be used right now. The default provider is always usable."""

    @abstractmethod
    def summarize(self, kind: str, chunks: Sequence[Chunk], *, topic: str, limit: int, ctx: StudyContext) -> list[SummaryItem]:
        """Summary points of one kind (``models.SUMMARY_KINDS``) about ``chunks``; ``topic`` is the topic name or "" for the whole lecture."""

    @abstractmethod
    def generate_flashcards(self, chunks: Sequence[Chunk], *, topic: str, types: Sequence[str], limit: int,
                            ctx: StudyContext) -> list[Flashcard]:
        ...

    @abstractmethod
    def generate_questions(self, chunks: Sequence[Chunk], *, topic: str, types: Sequence[str], limit: int,
                           ctx: StudyContext) -> list[Question]:
        ...

    @abstractmethod
    def explain(self, question: str, chunks: Sequence[Chunk], ctx: StudyContext) -> SummaryItem:
        """Answer a question from ``chunks`` only; if they do not contain the answer, say :data:`NOT_IN_SOURCES`."""

    @abstractmethod
    def evaluate_answer(self, question: Question, answer: str, chunks: Sequence[Chunk], ctx: StudyContext) -> Evaluation:
        """Judge the free-text answer of a student."""


@dataclass
class ProviderConfig:
    """Which provider the student chose (stored in the settings; the API key is NOT part of it: it lives in the OS credential store)."""

    kind: str = "builtin"                         # builtin | ollama | openai | anthropic
    model: str = ""
    base_url: str = ""
    timeout: float = 180.0
    temperature: float = 0.2
    context_words: int = 2500                     # how much lecture text goes into one request
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "model": self.model, "base_url": self.base_url, "timeout": self.timeout,
                "temperature": self.temperature, "context_words": self.context_words}

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> "ProviderConfig":
        data = data or {}
        return cls(kind=str(data.get("kind", "builtin")), model=str(data.get("model", "")), base_url=str(data.get("base_url", "")),
                   timeout=float(data.get("timeout", 180.0)), temperature=float(data.get("temperature", 0.2)),
                   context_words=int(data.get("context_words", 2500)))


PROVIDER_KINDS = ("builtin", "ollama", "openai", "anthropic")
PROVIDER_TITLES = {"builtin": "Built-in (offline, no setup)", "ollama": "Local model (Ollama)",
                   "openai": "OpenAI-compatible (local or cloud)", "anthropic": "Anthropic (cloud)"}


def create_provider(config: Optional[ProviderConfig], api_key: Optional[str] = None, transport: Any = None) -> AIProvider:
    """The provider the student chose. An unknown kind or a missing model/key does not raise here: ``check()`` tells what is missing,
    and the engine then uses the built-in provider (with a note) so that studying always works."""
    from .extractive import ExtractiveProvider

    config = config or ProviderConfig()
    if config.kind in ("", "builtin"):
        return ExtractiveProvider()
    from . import llm

    if config.kind == "ollama":
        return llm.OllamaProvider(config, transport=transport)
    if config.kind == "openai":
        return llm.OpenAICompatibleProvider(config, api_key=api_key, transport=transport)
    if config.kind == "anthropic":
        return llm.AnthropicProvider(config, api_key=api_key, transport=transport)
    raise ProviderError(f"Unknown AI provider '{config.kind}'.")
