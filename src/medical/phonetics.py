"""Sound-alike keys for medical terms and ASR errors.

An ASR error keeps the *sound* of the word it replaced (``crazy violet`` for ``cresyl violet``, ``nisal`` for ``Nissl``, ``RIMAC``
for ``Remak``): the consonant skeleton survives, the vowels and the spelling do not. ``skeleton`` keeps what survives; ``similarity``
compares two spans on their sound (and a little on their spelling). Similarity alone never decides a correction - it only proposes
candidates; the context decides (see ``corrector``).
"""

from __future__ import annotations

import re
from functools import lru_cache

from rapidfuzz.distance import Levenshtein

_START = (("kn", "n"), ("gn", "n"), ("wr", "r"), ("ps", "s"), ("pn", "n"), ("pt", "t"), ("x", "s"))
_DIGRAPHS = (("sch", "sk"), ("tch", "X"), ("ph", "f"), ("gh", ""), ("ck", "k"), ("dg", "X"), ("sh", "X"), ("ch", "k"), ("th", "t"),
             ("qu", "kw"), ("tion", "Xn"), ("sion", "Xn"), ("cia", "Xa"), ("x", "ks"), ("q", "k"))
_FOLD = str.maketrans({"b": "p", "d": "t", "g": "k", "v": "f", "z": "s", "m": "n", "j": "s", "c": "k", "X": "s"})
_VOWELS = set("aeiouy")


@lru_cache(maxsize=65536)
def skeleton(text: str) -> str:
    """Consonant skeleton of a word or phrase: voiced/voiceless and nasal pairs folded, vowels dropped, doubles collapsed."""
    word = re.sub(r"[^a-z]", "", text.lower())
    if not word:
        return ""
    for old, new in _START:
        if word.startswith(old):
            word = new + word[len(old):]
            break
    word = re.sub(r"c(?=[eiy])", "s", word)                       # soft c
    for old, new in _DIGRAPHS:
        word = word.replace(old, new)
    if not word:
        return ""
    first_vowel = word[0] if word[0] in _VOWELS else ""
    body = "".join(ch for ch in word[1:] if ch not in _VOWELS and ch not in "hw")
    key = ("a" if first_vowel else word[0]) + body if first_vowel or word[0] not in "hw" else body
    key = key.translate(_FOLD)
    return re.sub(r"(.)\1+", r"\1", key)


def letters(text: str) -> str:
    return re.sub(r"[^a-z]", "", text.lower())


def similarity(heard: str, term: str) -> float:
    """0..1: mostly sound (consonant skeleton), a little spelling. Both sides are phrases without regard to spaces / hyphens."""
    a, b = skeleton(heard), skeleton(term)
    if not a or not b:
        return 0.0
    sound = Levenshtein.normalized_similarity(a, b)
    spelling = Levenshtein.normalized_similarity(letters(heard), letters(term))
    if max(len(a), len(b)) <= 3:                                 # a very short skeleton says almost nothing
        sound *= 0.5 if a != b else 0.65
    elif a != b and min(len(a), len(b)) <= 3:
        sound *= 0.85
    elif min(len(a), len(b)) <= 3:
        sound *= 0.85
    return round(0.7 * sound + 0.3 * spelling, 4)
