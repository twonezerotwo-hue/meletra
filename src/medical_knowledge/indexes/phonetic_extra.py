"""More sound-alike keys, for the milestone-4 evaluation and for ranking.

``mp``   medical phoneme string: medical spellings are mapped to sounds (ph->f, ch->k, ae/oe->e, ps-/pn- ->s/n, x->ks, soft c / g ...), voiced / voiceless and
         nasal pairs are folded like the skeleton, and every VOWEL RUN is kept as one nucleus ``a``. It keeps the syllable structure the skeleton throws away
         ("Nissl" = nasl, "nisal" = nasal = nasal) and is used to RANK candidates, not to retrieve them (the skeleton retrieves: higher recall).
``msx``  medical Soundex: first sound + consonant classes, vowels dropped, full length.
``dm``   Double Metaphone of the words, concatenated (see ``double_metaphone``).
"""

from __future__ import annotations

import re
from functools import lru_cache

from .double_metaphone import dm_key, double_metaphone  # noqa: F401
from ..normalize.text import normalize

_VOWELS = set("aeiou")
_INITIAL = (("kn", "n"), ("gn", "n"), ("pn", "n"), ("wr", "r"), ("ps", "s"), ("pt", "t"), ("mn", "n"), ("x", "s"))
_SUBS = (("sch", "sk"), ("tch", "x"), ("ph", "f"), ("ck", "k"), ("cc", "k"), ("qu", "kw"), ("ch", "k"), ("sh", "x"), ("th", "t"), ("gh", ""), ("ae", "e"),
         ("oe", "e"), ("tion", "xan"), ("sion", "xan"), ("cia", "xa"), ("x", "ks"), ("q", "k"), ("y", "i"), ("w", "u"), ("h", ""))
_FOLD = str.maketrans({"b": "p", "d": "t", "g": "k", "v": "f", "z": "s", "m": "n", "j": "s", "c": "k", "e": "a", "i": "a", "o": "a", "u": "a", "x": "s"})


@lru_cache(maxsize=65536)
def mp_key(text: str) -> str:
    out = []
    for word in normalize(text).split():
        w = re.sub(r"[^a-z]", "", word)
        if not w:
            continue
        for old, new in _INITIAL:
            if w.startswith(old):
                w = new + w[len(old):]
                break
        w = re.sub(r"c(?=[eiy])", "s", w)
        w = re.sub(r"g(?=[eiy])", "j", w)
        for old, new in _SUBS:
            w = w.replace(old, new)
        w = w.translate(_FOLD)
        w = re.sub(r"a+", "a", w)                                           # a vowel run = one syllable nucleus
        w = re.sub(r"(.)\1+", r"\1", w)
        if len(w) > 2 and word.endswith("e") and not word.endswith(("ee", "ie", "ae", "oe")) and w.endswith("a"):
            w = w[:-1]                                                      # a silent final "e" ("cyte", "ose")
        out.append(w)
    return "".join(out)


_SOUNDEX = {"p": "1", "f": "1", "k": "2", "s": "2", "t": "3", "l": "4", "n": "5", "r": "6"}


@lru_cache(maxsize=65536)
def msx_key(text: str) -> str:
    """Medical Soundex: the first sound, then the consonant classes of the phoneme string with vowels dropped and repeats collapsed."""
    key = mp_key(text)
    if not key:
        return ""
    first = key[0]
    digits = []
    for ch in key[1:]:
        d = _SOUNDEX.get(ch, "")
        if d and (not digits or digits[-1] != d):
            digits.append(d)
    return first + "".join(digits)
