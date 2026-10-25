"""Rule-based clean transcript.

Guarantees (enforced by tests):
  * the raw transcript is never touched - the cleaner works on copies of the text;
  * no sentence is deleted and no word is added;
  * only deterministic, dictionary-backed fixes are applied (spacing/hyphen variants of a
    known term, capitalisation of eponyms). Anything doubtful is *marked*, never guessed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from .schema import Segment
from .uncertainty import segment_scope_flags, term_scope_flags
from .vocabulary import VocabularyIndex, normalize_key

_WORD = re.compile(r"[^\W\d_][\w'’]*(?:-[\w'’]+)*", re.UNICODE)
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?])")
_MULTI_SPACE = re.compile(r"\s{2,}")
_ONLY_JOINERS = re.compile(r"^[\s\-]+$")


@dataclass
class CleanResult:
    text: str
    changes: list[dict[str, Any]] = field(default_factory=list)


def _normalize_whitespace(text: str) -> str:
    text = _MULTI_SPACE.sub(" ", text.strip())
    return _SPACE_BEFORE_PUNCT.sub(r"\1", text)


def _canonicalize_terms(
    text: str, vocab: VocabularyIndex, fix_spacing: bool, fix_case: bool,
) -> tuple[str, list[dict[str, str]]]:
    """Replace spacing/hyphen/case variants of known terms with the dictionary spelling."""
    tokens = list(_WORD.finditer(text))
    changes: list[dict[str, str]] = []
    pieces: list[str] = []
    cursor = 0
    i = 0
    max_window = vocab.max_words + 2  # allow a term to be split into extra pieces ("spino thalamic")
    while i < len(tokens):
        replaced = False
        for size in range(min(max_window, len(tokens) - i), 0, -1):
            window = tokens[i:i + size]
            gaps_ok = all(
                _ONLY_JOINERS.match(text[window[k].end():window[k + 1].start()]) for k in range(size - 1)
            )
            if not gaps_ok:
                continue
            start, end = window[0].start(), window[-1].end()
            surface = text[start:end]
            canonical = vocab.canonical(surface)
            if canonical is None or surface == canonical:
                if canonical is not None:  # exact match: skip past it unchanged
                    i += size
                    replaced = True
                    break
                continue
            same_letters = surface.casefold() == canonical.casefold()
            if same_letters:
                eponym = any(ch.isupper() for ch in canonical[1:]) or (
                    canonical[:1].isupper() and not canonical.lower() == canonical)
                if not (fix_case and eponym and surface == surface.lower()):
                    i += size
                    replaced = True
                    break
                kind, new = "eponym_case", canonical
            else:
                if not fix_spacing:
                    i += size
                    replaced = True
                    break
                kind, new = "term_spacing", canonical
                if surface[:1].isupper() and canonical[:1].islower():
                    new = canonical[:1].upper() + canonical[1:]
            pieces.append(text[cursor:start])
            pieces.append(new)
            cursor = end
            changes.append({"type": kind, "from": surface, "to": new})
            i += size
            replaced = True
            break
        if not replaced:
            i += 1
    pieces.append(text[cursor:])
    return "".join(pieces), changes


def _mark_terms(text: str, segment: Segment) -> str:
    """Append ``[UNCERTAIN: ...]`` after each near-miss word (the word itself is kept)."""
    for flag in term_scope_flags(segment):
        if not flag.token:
            continue
        pattern = re.compile(rf"(?<!\w){re.escape(flag.token)}(?![\w])")
        marker = f' [UNCERTAIN: resembles "{flag.suggestion}"]'
        text, _ = pattern.subn(lambda m: m.group(0) + marker, text, count=1)
    return text


def clean_segments(
    segments: Sequence[Segment],
    vocab: Optional[VocabularyIndex],
    cleaner_cfg: dict[str, Any],
) -> CleanResult:
    """Build the clean transcript text from (unmodified) segments."""
    gap = float(cleaner_cfg.get("paragraph_gap_seconds", 2.0))
    fix_spacing = bool(cleaner_cfg.get("fix_term_spacing", True))
    fix_case = bool(cleaner_cfg.get("fix_eponym_case", True))
    mark = bool(cleaner_cfg.get("mark_uncertain", True))

    result = CleanResult(text="")
    paragraphs: list[list[str]] = []
    previous_end: Optional[float] = None
    previous_speaker: Optional[str] = None
    for index, seg in enumerate(segments):
        text = _normalize_whitespace(seg.text)
        if not text:
            continue
        if vocab is not None:
            text, changes = _canonicalize_terms(text, vocab, fix_spacing, fix_case)
            result.changes.extend({"segment": index, **c} for c in changes)
        if mark:
            text = _mark_terms(text, seg)
            if segment_scope_flags(seg):
                text = f"[UNCERTAIN: {text}]"
        speaker_change = seg.speaker is not None and seg.speaker != previous_speaker
        if speaker_change:
            text = f"[{seg.speaker}] {text}"
        if previous_end is not None and (seg.start - previous_end > gap or speaker_change) and paragraphs:
            paragraphs.append([])
        if not paragraphs:
            paragraphs.append([])
        paragraphs[-1].append(text)
        previous_end = seg.end
        previous_speaker = seg.speaker if seg.speaker is not None else previous_speaker
    result.text = "\n\n".join(" ".join(p) for p in paragraphs if p) + "\n"
    return result


def strip_markers(text: str) -> str:
    """Remove ``[UNCERTAIN: ...]`` wrappers (used by tests to prove nothing else changed)."""
    text = re.sub(r' \[UNCERTAIN: resembles "[^"]*"\]', "", text)
    return re.sub(r"\[UNCERTAIN: (.*?)\]", r"\1", text, flags=re.S)


def word_keys(text: str) -> list[str]:
    return [normalize_key(m.group(0)) for m in _WORD.finditer(text)]
