"""Medical terms that must survive a translation unchanged: abbreviations (MRI, ATP), gene and protein names (mTORC1, p53), eponyms (Purkinje), drug names (bevacizumab) and
alphanumeric codes (IL-6, HbA1c). A translation model that meets "cerebellum" or "bevacizumab" may turn them into something else; a protected term is swapped for a neutral marker
before the sentence is translated and put back afterwards, and a sentence whose markers did not all come back is translated again without them and flagged.

Measured on the offline model: ``XQ1Z``-style markers came back intact in all seven target languages tried (German, Turkish, French, Spanish, Korean, Hindi, Chinese); brackets, angle
brackets and braces did not (one Korean run looped on ``[1]``).
"""

from __future__ import annotations

import logging
import re
from typing import Iterable, Optional

log = logging.getLogger(__name__)

MARKER = "XQ{n}Z"
MARKER_RE = re.compile(r"XQ(\d+)Z")
PROTECTED_KINDS = ("abbr", "gene", "eponym", "drug")
MAX_WORDS = 4                                                  # the longest term (in words) looked for
_WORD = re.compile(r"[^\W_](?:[\w\-+'’]*[^\W_])?")             # a word, keeping inner hyphens / plus signs / apostrophes (IL-6, BCL-2, Purkinje's)
_ACRONYM = re.compile(r"^[A-Z]{2,6}[0-9]{0,2}$")
_CODE = re.compile(r"^(?=.*[A-Za-z])(?=.*[0-9])[A-Za-z0-9\-+]{2,14}$")
_COMMON_CAPS = {"OK", "TV", "US", "UK", "AM", "PM", "AND", "OR", "NOT", "THE", "FOR", "IS"}


def _norm(text: str) -> str:
    return " ".join(text.casefold().replace("’", "'").split())


class TermGuard:
    """Finds the terms to protect in a sentence and puts them back after the translation."""

    def __init__(self, surfaces: Iterable[str] = ()) -> None:
        self.surfaces = {_norm(s) for s in surfaces if s and s.strip()}

    @classmethod
    def from_lexicon(cls, lexicon: Optional[object] = None) -> "TermGuard":
        """The guard built from the medical vocabulary the program ships (abbreviations, genes, eponyms, drugs). Never raises: without the vocabulary only the patterns apply."""
        try:
            from ..medical.lexicon import shared_lexicon

            lex = lexicon or shared_lexicon()
            surfaces: list[str] = []
            for name in lex.names():                                  # type: ignore[attr-defined]
                domain = lex.domain(name)                              # type: ignore[attr-defined]
                if domain is None:
                    continue
                for term in domain.terms:
                    if term.kind in PROTECTED_KINDS:
                        surfaces.extend(term.surfaces if term.kind != "abbr" else (term.canonical,))   # an abbreviation's spoken-out expansions are ordinary words
            return cls(surfaces)
        except Exception:  # noqa: BLE001 - protection is an aid on top of the translation
            log.warning("medical vocabulary unavailable for term protection", exc_info=True)
            return cls()

    def protect(self, text: str) -> tuple[str, list[str]]:
        """``(text with markers, [the protected terms in marker order])``."""
        words = list(_WORD.finditer(text))
        spans: list[tuple[int, int]] = []
        i = 0
        while i < len(words):
            matched = 0
            for n in range(min(MAX_WORDS, len(words) - i), 0, -1):
                chunk = text[words[i].start():words[i + n - 1].end()]
                if n > 1 and any(text[words[k].end():words[k + 1].start()].strip() for k in range(i, i + n - 1)):
                    continue                                          # punctuation between the words: not one term
                if _norm(chunk) in self.surfaces and not (n == 1 and chunk.upper() in _COMMON_CAPS):
                    spans.append((words[i].start(), words[i + n - 1].end()))
                    matched = n
                    break
            if not matched:
                word = words[i].group(0)
                if (_ACRONYM.match(word) and word not in _COMMON_CAPS) or _CODE.match(word):
                    spans.append((words[i].start(), words[i].end()))
                    matched = 1
            i += matched or 1
        terms: list[str] = []
        out: list[str] = []
        cursor = 0
        for number, (start, end) in enumerate(spans, start=1):
            out.append(text[cursor:start])
            out.append(MARKER.format(n=number))
            terms.append(text[start:end])
            cursor = end
        out.append(text[cursor:])
        return "".join(out), terms

    @staticmethod
    def restore(translated: str, terms: list[str]) -> tuple[str, bool]:
        """``(text with the terms back, True when every marker was found exactly once)``."""
        found = [int(m.group(1)) for m in MARKER_RE.finditer(translated)]
        complete = sorted(found) == list(range(1, len(terms) + 1))
        text = MARKER_RE.sub(lambda m: terms[int(m.group(1)) - 1] if 1 <= int(m.group(1)) <= len(terms) else m.group(0), translated)
        return text, complete
