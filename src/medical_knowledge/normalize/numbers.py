"""Spoken and written numbers inside terms: "CRISPR Cas nine" = "CRISPR-Cas9", "fructose two six bisphosphate" = "Fructose-2,6-Bisphosphate", "DNA polymerase one" = "DNA polymerase I".

``numeric_key`` turns every number WORD into its digits and a roman numeral that ends a multi-word term into its digit, then joins everything without spaces. Two spellings of the
same term have the same key. The key is used for LOOKUP only (candidate retrieval); what is written into a transcript is always the form of a source.
"""

from __future__ import annotations

import re
from functools import lru_cache

from .text import normalize

_ONES = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
         "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8, "ix": 9, "x": 10}
_HAS_NUMBER_WORD = re.compile(r"\b(" + "|".join(sorted({*_ONES, *_TENS}, key=len, reverse=True)) + r")\b")


@lru_cache(maxsize=131072)
def numeric_key(text: str) -> str:
    words = normalize(text).split()
    out: list[str] = []
    i = 0
    while i < len(words):
        w = words[i]
        if w in _TENS and i + 1 < len(words) and words[i + 1] in _ONES and 1 <= _ONES[words[i + 1]] <= 9:        # "twenty one" -> 21
            out.append(str(_TENS[w] + _ONES[words[i + 1]]))
            i += 2
            continue
        if w in _ONES:
            out.append(str(_ONES[w]))
        elif w in _TENS:
            out.append(str(_TENS[w]))
        elif w in _ROMAN and i == len(words) - 1 and i >= 1 and out and not out[-1].isdigit():
            out.append(str(_ROMAN[w]))                                                                      # "polymerase I" -> "polymerase1"
        else:
            out.append(w)
        i += 1
    return "".join(out)


def has_number(text: str) -> bool:
    """True when the text contains a digit, a number word or a final roman numeral: only then is the numeric key different from the plain compact form."""
    n = normalize(text)
    if any(ch.isdigit() for ch in n):
        return True
    words = n.split()
    return bool(_HAS_NUMBER_WORD.search(n)) or (len(words) >= 2 and words[-1] in _ROMAN)
