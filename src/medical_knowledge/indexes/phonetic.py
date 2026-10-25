"""Sound-alike keys of terms and spans. ``KEYS`` maps an algorithm name to a function ``text -> key``.

Keys only PROPOSE candidates (retrieval); they never decide a correction. More algorithms are registered by ``indexes.phonetic_extra`` (milestone 4).
"""

from __future__ import annotations

from typing import Callable

from ...medical.phonetics import skeleton
from .double_metaphone import dm_key
from .phonetic_extra import mp_key, msx_key

KeyFunction = Callable[[str], str]

#: every key function (milestone-4 evaluation: docs/v4/PHONETICS.md). ``sk`` and ``dm`` live in the columns ``term.sk`` / ``term.dm``;
#: further algorithms listed in ``STORED`` go to ``phonetic_key``.
KEYS: dict[str, KeyFunction] = {"sk": skeleton, "dm": dm_key, "mp": mp_key, "msx": msx_key}
STORED: tuple[str, ...] = ()
#: the key whose one-letter deletion neighbourhood is indexed (``phonetic_del``): best recall per pool size in the evaluation
DELETION_ALGORITHM = "dm"


def keys_of(text: str, algorithms: tuple[str, ...] = STORED) -> dict[str, str]:
    out = {}
    for name in algorithms:
        key = KEYS[name](text)
        if key:
            out[name] = key
    return out


DELETION_MIN, DELETION_MAX, DELETION_MAX_TOKENS = 3, 24, 4


def deletions(key: str) -> set[str]:
    """The key itself and the key with each single letter removed (the deletion neighbourhood)."""
    return {key} | {key[:i] + key[i + 1:] for i in range(len(key))}


def deletion_eligible(key: str, tokens: int) -> bool:
    return DELETION_MIN <= len(key) <= DELETION_MAX and tokens <= DELETION_MAX_TOKENS
