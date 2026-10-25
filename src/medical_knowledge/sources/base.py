"""Common interface of a terminology source.

A source turns some terminology release (a local file, a licensed download, the project's own vocabulary) into normalised records. Every record keeps
its provenance (source, source code, source concept id); the builder refuses a source that the license policy does not allow in the requested profile.

    download()   fetch the release when that is legal and automatic (open sources); licensed sources only say where to get it and never bypass the license
    verify()     problems with the local copy (missing files, checksum, wrong version); [] = fine
    parse()      raw records of the source format
    normalize()  records in the common shape below
    export()     parse + normalize + hand the records to a ``KnowledgeBuilder``; returns counts
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional


class CredentialsRequired(Exception):
    """The source needs a license / account that only the owner can accept. The importer reports where to get the data and stops."""


@dataclass(frozen=True)
class SourceInfo:
    source_id: str
    source_name: str
    source_version: str
    source_url: str = ""
    license_category: str = "UNKNOWN"       # PROJECT_OWNED | PUBLIC_DOMAIN | CC0 | CC_BY | CUSTOM_OPEN | UMLS_LICENSED | AFFILIATE | UNKNOWN
    redistribution_allowed: bool = False     # informational copy; the license registry decides
    requires_credentials: bool = False
    language: str = "en"
    release_date: str = ""
    notes: str = ""


@dataclass
class ConceptRecord:
    concept_id: str
    preferred_name: str
    semantic_type: str = ""
    domains: tuple[str, ...] = ()
    subdomain: str = ""
    kind: str = "term"
    trust: str = "CURATED"


@dataclass
class TermRecord:
    concept_id: str
    text: str
    language: str = "en"
    term_type: str = "synonym"               # preferred | synonym | abbreviation | latin | eponym | drug | gene | plural
    is_preferred: bool = False
    source_code: str = ""
    source_concept_id: str = ""
    frequency: int = 0
    medical_frequency: float = 0.0
    trust: str = ""                          # empty = the concept's trust
    source_id: str = ""                      # a sub-source of the importing source (UMLS/MSH); empty = the importing source


@dataclass
class RelationRecord:
    concept_a: str
    relation_type: str
    concept_b: str
    weight: float = 1.0
    source_id: str = ""


@dataclass
class AbbreviationRecord:
    short_form: str
    long_form: str
    concept_id: str = ""
    domain: str = ""
    language: str = "en"
    source_id: str = ""


@dataclass
class CueRecord:
    concept_id: str
    cue: str
    weight: float = 1.0


@dataclass
class ConfusionRecord:
    """A known WRONG written form of a concept (an ASR mishearing). Never a valid term."""
    concept_id: str
    heard: str


@dataclass
class WordRecord:
    """An ordinary English word (not a medical term): ``level`` 10 = most common ... 80 = rare dictionary word."""
    word: str
    level: int


Record = SourceInfo | WordRecord | ConceptRecord | TermRecord | RelationRecord | AbbreviationRecord | CueRecord | ConfusionRecord


@dataclass
class ImportStats:
    source_id: str
    concepts: int = 0
    terms: int = 0
    relations: int = 0
    abbreviations: int = 0
    cues: int = 0
    confusions: int = 0
    words: int = 0
    skipped: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"source_id": self.source_id, "concepts": self.concepts, "terms": self.terms, "relations": self.relations,
                "abbreviations": self.abbreviations, "cues": self.cues, "confusions": self.confusions, "words": self.words, "skipped": self.skipped, "notes": self.notes}


class TerminologySource(ABC):
    """Subclasses set ``info`` and implement ``parse`` / ``normalize``."""

    info: SourceInfo

    def download(self, destination: Path) -> None:
        raise NotImplementedError(f"{self.info.source_id}: automatic download is not available; see the importer help for where to get the data")

    def verify(self) -> list[str]:
        return []

    @abstractmethod
    def parse(self) -> Iterator[dict]:
        """Raw records of the source format (one dict per concept / row group)."""

    @abstractmethod
    def normalize(self, raw: Iterable[dict]) -> Iterator[Record]:
        """Raw records -> the common record types above."""

    def export(self, builder) -> ImportStats:
        problems = self.verify()
        if problems:
            raise ValueError(f"{self.info.source_id}: " + "; ".join(problems))
        builder.add_source(self.info)
        stats = ImportStats(self.info.source_id)
        for record in self.normalize(self.parse()):
            sid = getattr(record, "source_id", "") or self.info.source_id
            if isinstance(record, SourceInfo):                       # a sub-source (one UMLS vocabulary) announces itself before its records
                builder.add_source(record)
            elif isinstance(record, ConceptRecord):
                builder.add_concept(self.info.source_id, record)
                stats.concepts += 1
            elif isinstance(record, TermRecord):
                if builder.add_term(sid, record):
                    stats.terms += 1
                else:
                    stats.skipped += 1
            elif isinstance(record, RelationRecord):
                builder.add_relation(sid, record)
                stats.relations += 1
            elif isinstance(record, AbbreviationRecord):
                builder.add_abbreviation(sid, record)
                stats.abbreviations += 1
            elif isinstance(record, CueRecord):
                builder.add_cue(record)
                stats.cues += 1
            elif isinstance(record, ConfusionRecord):
                builder.add_confusion(self.info.source_id, record)
                stats.confusions += 1
            elif isinstance(record, WordRecord):
                builder.add_word(self.info.source_id, record)
                stats.words += 1
        return stats
