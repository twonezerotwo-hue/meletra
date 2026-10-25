"""Plain data containers shared by every module (no heavy imports here)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

SEGMENT_SCOPE = "segment"   # the whole segment is doubtful (acoustic / hallucination signals)
TERM_SCOPE = "term"         # a single word looks like a mis-heard dictionary term


@dataclass
class Word:
    """A single word with timing, when the backend provides it."""

    start: float
    end: float
    word: str
    probability: Optional[float] = None


@dataclass
class UncertaintyFlag:
    """Why (part of) a segment was marked uncertain."""

    code: str
    detail: str
    scope: str = SEGMENT_SCOPE
    token: Optional[str] = None        # the suspicious word (term scope)
    suggestion: Optional[str] = None   # the dictionary term it resembles (term scope)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"code": self.code, "detail": self.detail, "scope": self.scope}
        if self.token is not None:
            data["token"] = self.token
        if self.suggestion is not None:
            data["suggestion"] = self.suggestion
        return data


@dataclass
class Segment:
    """One transcribed segment.

    ``avg_logprob``, ``no_speech_prob`` and ``compression_ratio`` are ``None`` for backends that
    do not expose them (e.g. the plain Transformers pipeline).
    """

    start: float
    end: float
    text: str
    language: Optional[str] = None
    model: str = ""
    backend: str = ""
    avg_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = None
    compression_ratio: Optional[float] = None
    confidence: Optional[float] = None
    words: list[Word] = field(default_factory=list)
    flags: list[UncertaintyFlag] = field(default_factory=list)
    speaker: Optional[str] = None        # only set by the optional WhisperX diarization step
    language_meta: dict[str, Any] = field(default_factory=dict)   # how ``language`` was decided: confidence, source (detector | verified | script | forced), candidates, status

    def to_checkpoint(self) -> dict[str, Any]:
        """Lossless, compact form for the resume checkpoint (full precision, original text)."""
        return {
            "start": self.start, "end": self.end, "text": self.text, "language": self.language,
            "model": self.model, "backend": self.backend, "avg_logprob": self.avg_logprob,
            "no_speech_prob": self.no_speech_prob, "compression_ratio": self.compression_ratio,
            "words": [[w.start, w.end, w.word, w.probability] for w in self.words],
            **({"language_meta": self.language_meta} if self.language_meta else {}),
        }

    @classmethod
    def from_checkpoint(cls, data: dict[str, Any]) -> "Segment":
        return cls(
            start=float(data["start"]), end=float(data["end"]), text=data["text"], language=data.get("language"),
            model=data.get("model", ""), backend=data.get("backend", ""), avg_logprob=data.get("avg_logprob"),
            no_speech_prob=data.get("no_speech_prob"), compression_ratio=data.get("compression_ratio"),
            words=[Word(float(w[0]), float(w[1]), w[2], w[3]) for w in data.get("words", [])],
            language_meta=dict(data.get("language_meta") or {}),
        )

    @property
    def uncertain(self) -> bool:
        return bool(self.flags)

    @property
    def reasons(self) -> list[str]:
        return [flag.code for flag in self.flags]

    def to_dict(self, include_words: bool = True) -> dict[str, Any]:
        def rounded(value: Optional[float], digits: int) -> Optional[float]:
            return None if value is None else round(value, digits)

        data: dict[str, Any] = {
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "text": self.text.strip(),
            "confidence": rounded(self.confidence, 3),
            "language": self.language,
            "model": self.model,
            "backend": self.backend,
            "avg_logprob": rounded(self.avg_logprob, 4),
            "no_speech_prob": rounded(self.no_speech_prob, 4),
            "compression_ratio": rounded(self.compression_ratio, 3),
            "uncertain": self.uncertain,
            "uncertainty_reasons": self.reasons,
        }
        if self.speaker is not None:
            data["speaker"] = self.speaker
        if include_words and self.words:
            data["words"] = [
                {"start": round(w.start, 2), "end": round(w.end, 2), "word": w.word,
                 "probability": rounded(w.probability, 3)}
                for w in self.words
            ]
        return data


@dataclass
class TranscriptionResult:
    """Everything a backend returns for one audio file."""

    segments: list[Segment]
    language: Optional[str]
    language_probability: Optional[float]
    duration: float
    model: str
    backend: str
    compute_type: Optional[str] = None
    fallback_used: bool = False
    load_seconds: float = 0.0
    capabilities_used: dict[str, bool] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    hotwords_applied: list[str] = field(default_factory=list)   # what actually fit the token budget
    hotword_tokens: Optional[int] = None                        # Whisper tokens used by those hotwords
    rejected: list[dict[str, Any]] = field(default_factory=list)   # decoded stretches NOT kept as speech (instrumental hallucinations ...): start, end, reason - for the coverage report
