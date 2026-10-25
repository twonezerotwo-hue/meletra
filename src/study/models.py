"""Data of the study system: chunks of the lecture, source references, summaries, flashcards, questions, the saved study pack.

Everything is a plain dataclass with ``to_dict`` / ``from_dict`` so a pack is one JSON file next to the transcript, and so every
generated item can carry the references (``sources``) it was made from.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, fields
from typing import Any, Optional

PACK_VERSION = 1

SOURCE_TRANSCRIPT = "transcript"
SOURCE_PDF = "pdf"

#: how a pack was made, in the order of preference
MODE_BOTH, MODE_TRANSCRIPT, MODE_PDF = "pdf+transcript", "transcript", "pdf"

SUMMARY_KINDS = (
    "quick", "detailed", "high_yield", "key_concepts", "definitions", "mechanisms", "clinical", "emphasis", "exam",
)
SUMMARY_TITLES = {
    "quick": "Quick Summary", "detailed": "Detailed Lecture Summary", "high_yield": "High-Yield Medical Summary",
    "key_concepts": "Key Concepts", "definitions": "Important Definitions", "mechanisms": "Mechanisms / Pathways",
    "clinical": "Clinical Connections", "emphasis": "Lecturer Emphasis", "exam": "Exam-Relevant Points",
}

CARD_TYPES = ("definition", "concept", "mechanism", "cause_effect", "structure_function", "clinical", "comparison", "pathway",
              "enzyme", "drug", "gene_protein", "exam_recall", "integration")
CARD_TYPE_TITLES = {
    "definition": "Definition", "concept": "Concept", "mechanism": "Mechanism", "cause_effect": "Cause -> Effect",
    "structure_function": "Structure -> Function", "clinical": "Clinical", "comparison": "Comparison", "pathway": "Pathway",
    "enzyme": "Enzyme", "drug": "Drug", "gene_protein": "Gene / Protein", "exam_recall": "Active recall", "integration": "Integration",
}
QUESTION_TYPES = ("mcq", "true_false", "short_answer", "explain_why", "clinical_vignette", "mechanism")
QUESTION_TYPE_TITLES = {
    "mcq": "Multiple choice", "true_false": "True / False", "short_answer": "Short answer", "explain_why": "Explain why",
    "clinical_vignette": "Clinical vignette", "mechanism": "Mechanism",
}
DIFFICULTIES = ("easy", "medium", "hard")
DIFFICULTY_TITLES = {"easy": "Basic", "medium": "Intermediate", "hard": "Advanced"}     # what the student sees; the stored values keep their old names so older packs read the same
RATINGS = ("know", "unsure", "dont_know")


def clock(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def item_id(prefix: str, *parts: str) -> str:
    """A stable id from the content of an item: regenerating the same card keeps its id, so saved progress stays attached to it."""
    digest = hashlib.sha1(chr(31).join(p.casefold().strip() for p in parts).encode("utf-8")).hexdigest()
    return f"{prefix}-{digest[:10]}"


def _build(cls: type, data: dict[str, Any]) -> Any:
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class Chunk:
    """A piece of the lecture the study engine can read and cite."""

    id: str
    source: str                                   # transcript | pdf
    text: str
    topic: str = ""
    start: Optional[float] = None                 # transcript: seconds
    end: Optional[float] = None
    page: Optional[int] = None                    # pdf: 1-based page (slide)
    title: str = ""                               # pdf: the slide title
    segs: list = field(default_factory=list)      # transcript: [[start, end, first character, last character], ...] of the segments in `text`

    def label(self) -> str:
        if self.source == SOURCE_PDF:
            return f"Slide {self.page}"
        return f"Transcript {clock(self.start or 0.0)}-{clock(self.end or 0.0)}"

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        return _build(cls, data)


@dataclass
class SourceRef:
    """Where an item comes from: shown to the student as a small tag ("Slide 14", "Transcript 12:30-12:58")."""

    chunk_id: str
    kind: str
    label: str
    start: Optional[float] = None
    end: Optional[float] = None
    page: Optional[int] = None

    @classmethod
    def of(cls, chunk: Chunk) -> "SourceRef":
        return cls(chunk.id, chunk.source, chunk.label(), chunk.start, chunk.end, chunk.page)

    @classmethod
    def of_span(cls, chunk: Chunk, start: Optional[float], end: Optional[float]) -> "SourceRef":
        """A narrow reference: the seconds of the transcript the statement really comes from (not the whole chunk)."""
        if chunk.source != SOURCE_TRANSCRIPT or start is None or end is None:
            return cls.of(chunk)
        return cls(chunk.id, chunk.source, f"Transcript {clock(start)}-{clock(end)}", round(start, 2), round(end, 2), None)

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SourceRef":
        return _build(cls, data)


def _refs(items: list[Any]) -> list[SourceRef]:
    return [i if isinstance(i, SourceRef) else SourceRef.from_dict(i) for i in items]


@dataclass
class SummaryItem:
    text: str
    sources: list[SourceRef] = field(default_factory=list)
    topic: str = ""
    term: str = ""                                # key concepts / definitions: the term the text is about
    external: bool = False                        # not from the lecture (only when the user allowed outside knowledge)
    importance: str = ""                          # key concepts: why it matters ("High-yield - the lecturer stressed it", "Core term - named 11 times")

    def __post_init__(self) -> None:
        self.sources = _refs(self.sources)

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "sources": [s.to_dict() for s in self.sources], "topic": self.topic, "term": self.term,
                "external": self.external, "importance": self.importance}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SummaryItem":
        return _build(cls, data)


@dataclass
class Flashcard:
    id: str
    type: str
    front: str
    back: str
    sources: list[SourceRef] = field(default_factory=list)
    difficulty: str = "medium"
    topic: str = ""
    external: bool = False
    fact_id: str = ""                             # the validated StudyFact it was made from
    high_yield: bool = False                      # the lecturer stressed it, or the slides repeat it
    evidence: str = ""                            # the words of the lecture the card rests on (a quote from the cited passage)
    generation: str = ""                          # the version of the study engine that made it ("" = made before versions were recorded)

    def __post_init__(self) -> None:
        self.sources = _refs(self.sources)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "type": self.type, "front": self.front, "back": self.back,
                "sources": [s.to_dict() for s in self.sources], "difficulty": self.difficulty, "topic": self.topic,
                "external": self.external, "fact_id": self.fact_id, "high_yield": self.high_yield,
                "evidence": self.evidence, "generation": self.generation}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Flashcard":
        return _build(cls, data)


@dataclass
class Question:
    id: str
    type: str
    question: str
    answer: str                                   # mcq: the text of the correct option; true_false: "True" / "False"
    explanation: str = ""
    options: list[str] = field(default_factory=list)
    sources: list[SourceRef] = field(default_factory=list)
    difficulty: str = "medium"
    topic: str = ""
    external: bool = False
    fact_id: str = ""
    high_yield: bool = False
    evidence: str = ""
    generation: str = ""

    def __post_init__(self) -> None:
        self.sources = _refs(self.sources)

    @property
    def correct_index(self) -> int:
        """Position of the correct option (-1 if this is not a multiple-choice question)."""
        return self.options.index(self.answer) if self.options and self.answer in self.options else -1

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "type": self.type, "question": self.question, "answer": self.answer, "explanation": self.explanation,
                "options": list(self.options), "sources": [s.to_dict() for s in self.sources], "difficulty": self.difficulty,
                "topic": self.topic, "external": self.external, "fact_id": self.fact_id, "high_yield": self.high_yield,
                "evidence": self.evidence, "generation": self.generation}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Question":
        return _build(cls, data)


@dataclass
class ProviderInfo:
    name: str = "builtin"
    kind: str = "builtin"                         # builtin | local | cloud
    model: str = ""
    fallback_from: str = ""                       # the configured provider that failed (the built-in method took over)

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProviderInfo":
        return _build(cls, data)


@dataclass
class StudyPack:
    lecture_id: str
    title: str
    source_mode: str                              # pdf+transcript | transcript | pdf
    language: str = "en"
    created: str = ""
    provider: ProviderInfo = field(default_factory=ProviderInfo)
    source_bound: bool = True
    topics: list[str] = field(default_factory=list)
    summaries: dict[str, list[SummaryItem]] = field(default_factory=dict)
    flashcards: list[Flashcard] = field(default_factory=list)
    questions: list[Question] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    version: int = PACK_VERSION
    overview: dict[str, Any] = field(default_factory=dict)       # title, subject, duration, main topics, lecture structure (from the validated facts)
    generation: dict[str, Any] = field(default_factory=dict)     # engine, engine version, model, source revision, language: what the cached material belongs to

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version, "lecture_id": self.lecture_id, "title": self.title, "source_mode": self.source_mode,
            "language": self.language, "created": self.created, "provider": self.provider.to_dict(),
            "source_bound": self.source_bound, "topics": list(self.topics),
            "summaries": {k: [i.to_dict() for i in v] for k, v in self.summaries.items()},
            "flashcards": [c.to_dict() for c in self.flashcards], "questions": [q.to_dict() for q in self.questions],
            "stats": dict(self.stats), "warnings": list(self.warnings), "overview": dict(self.overview), "generation": dict(self.generation),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StudyPack":
        return cls(
            lecture_id=data.get("lecture_id", ""), title=data.get("title", ""), source_mode=data.get("source_mode", MODE_TRANSCRIPT),
            language=data.get("language", "en"), created=data.get("created", ""),
            provider=ProviderInfo.from_dict(data.get("provider", {})), source_bound=bool(data.get("source_bound", True)),
            topics=list(data.get("topics", [])),
            summaries={k: [SummaryItem.from_dict(i) for i in v] for k, v in data.get("summaries", {}).items()},
            flashcards=[Flashcard.from_dict(c) for c in data.get("flashcards", [])],
            questions=[Question.from_dict(q) for q in data.get("questions", [])],
            stats=dict(data.get("stats", {})), warnings=list(data.get("warnings", [])), version=int(data.get("version", PACK_VERSION)),
            overview=dict(data.get("overview", {})), generation=dict(data.get("generation", {})))

    @property
    def empty(self) -> bool:
        return not (self.summaries or self.flashcards or self.questions)
