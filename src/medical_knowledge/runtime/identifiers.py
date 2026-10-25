"""Identifier detector: things that are written the way they are ON PURPOSE and must never be "corrected" as a mishearing.

    gene / protein symbols     BRCA1, TP53, IL6, mTORC1, p53, CD40       (letters next to digits, or capital-letter symbols)
    drug and compound codes    LY294002, ABT199
    dataset / record ids       GSE12345, NCT01234567, rs123456
    software and library names CTranslate2, PyTorch, NumPy, GitHub       (camelCase / acronym + word)
    acronyms                   CNS, PNS, CSF, MS, RA, ER, ATP           (2-8 capital letters)
    paths, addresses, versions model_v2.py, user@host, 3.5.1

The tokenizer of the engine cuts a word at digits ("BRCA1" -> "BRCA" + "1"), so the check looks at the RAW text around a token, not at the token alone.
Only structure is used: no list of names. A recogniser that writes a word like this did so deliberately; correcting it can only damage it (the "CTranslate" -> "translated" false positive).
"""

from __future__ import annotations

import re
from typing import Optional

_CHUNK_CHARS = re.compile(r"[A-Za-z0-9_@./\\\-]")
_CAMEL = re.compile(r"[a-z][A-Z]|[A-Z]{2,}[a-z]")
_ACRONYM = re.compile(r"^[A-Z]{2,8}$")
_ACRONYM_S = re.compile(r"^[A-Z]{2,8}s$")                 # CTs, MRIs
_LETTER_DIGIT = re.compile(r"[A-Za-z]\d|\d[A-Za-z]")
_SYMBOLS = re.compile(r"[_@\\/]|\w\.\w")


def raw_chunk(text: str, start: int, end: int) -> str:
    """The maximal run of word characters (letters, digits and _ @ . / \\ -) around [start, end), without trailing dots and hyphens."""
    lo, hi = start, end
    while lo > 0 and _CHUNK_CHARS.match(text[lo - 1]):
        lo -= 1
    while hi < len(text) and _CHUNK_CHARS.match(text[hi]):
        hi += 1
    chunk = text[lo:hi]
    return chunk.rstrip(".-/\\").lstrip("-")


def identifier_reason(text: str, start: int, end: int) -> Optional[str]:
    """Why the token at [start, end) of ``text`` is an identifier, or None."""
    chunk = raw_chunk(text, start, end)
    if not chunk:
        return None
    for part in re.split(r"-", chunk):
        if not part:
            continue
        if _LETTER_DIGIT.search(part):
            return "alphanumeric identifier (symbol, code or id)"
        if _CAMEL.search(part):
            return "mixed-case identifier (software or protein name)"
        if _ACRONYM.match(part) or _ACRONYM_S.match(part):
            return "acronym written in capitals"
    if _SYMBOLS.search(chunk):
        return "path, address or version"
    return None
