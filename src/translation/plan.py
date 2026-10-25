"""Which way a transcript gets into the Output Language (pure decision, no I/O)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from . import METHOD_M2M100, METHOD_MISSING, METHOD_NONE, METHOD_WHISPER


@dataclass(frozen=True)
class TranslationPlan:
    method: str
    source: Optional[str]                 # the spoken language (a Whisper code) or None when it is not one language (mixed)
    target: Optional[str]
    reason: str


def decide(source: Optional[str], target: Optional[str], m2m_ready: bool, spoken: Sequence[str] = ()) -> TranslationPlan:
    """``source``: the spoken language Whisper used or detected (None = mixed / unknown); ``target``: the Output Language (None / "" = none asked for);
    ``m2m_ready``: the offline translation model is installed; ``spoken``: every language that was recognised in the recording (several = a mixed recording).

    The text translation (the model, from the saved transcript, each segment from its own language) is the way whenever the model is there. Without it English can still be made from the audio by
    Whisper's translate task - one spoken language at a time - and every other language needs the model."""
    if not target:
        return TranslationPlan(METHOD_NONE, source, None, "no output language asked for")
    languages = [l for l in spoken if l]
    if languages and set(languages) == {target}:
        return TranslationPlan(METHOD_NONE, target, target, "the lecture is spoken in the output language")
    if not languages and source and source == target:
        return TranslationPlan(METHOD_NONE, source, target, "the lecture is spoken in the output language")
    if m2m_ready:
        return TranslationPlan(METHOD_M2M100, source, target, "offline translation model")
    if target == "en":
        return TranslationPlan(METHOD_WHISPER, source, target, "Whisper translates to English")
    return TranslationPlan(METHOD_MISSING, source, target, "the translation model is not installed")
