"""Matching forms of medical text.

``normalize``  lower case, accents folded, Greek letters spelled out, everything that is not a letter or digit becomes one space
               ("gamma-aminobutyric acid" -> "gamma aminobutyric acid"); the form used for exact lookup and for the full-text index
``compact``    the same without spaces ("gammaaminobutyric acid" -> "gammaaminobutyricacid"); ASR splits and joins compounds at will, so
               exact lookup ignores spaces too
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

_GREEK = {"α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon", "ζ": "zeta", "η": "eta", "θ": "theta", "ι": "iota", "κ": "kappa",
          "λ": "lambda", "μ": "mu", "ν": "nu", "ξ": "xi", "π": "pi", "ρ": "rho", "σ": "sigma", "ς": "sigma", "τ": "tau", "υ": "upsilon",
          "φ": "phi", "χ": "chi", "ψ": "psi", "ω": "omega", "µ": "mu"}
_GREEK_RE = re.compile("[" + "".join(_GREEK) + "]")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_LETTERS = re.compile(r"[^a-z]")


@lru_cache(maxsize=65536)
def normalize(text: str) -> str:
    text = _GREEK_RE.sub(lambda m: f" {_GREEK[m.group(0)]} ", text.lower())
    plain = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return _NON_ALNUM.sub(" ", plain).strip()


def compact(text: str) -> str:
    return normalize(text).replace(" ", "")


_VARIANT_RULES = (("ph", "f"), ("ae", "e"), ("oe", "e"), ("ou", "o"), ("y", "i"), ("z", "s"), ("ck", "k"), ("kh", "k"))
_DOUBLE = re.compile(r"([a-z])\1+")


@lru_cache(maxsize=65536)
def wording_key(text: str) -> str:
    """A key that two spellings of ONE word share: British / American, ph / f, y / i, doubled letters, a plural -s ("glibenclamide" = "glybenclamide" = "glibenclamides").
    Different words never share it: "afferent" and "efferent", "ileum" and "ilium", "meiosis" and "miosis" stay different."""
    words = []
    for w in normalize(text).split():
        for a, b in _VARIANT_RULES:
            w = w.replace(a, b)
        w = _DOUBLE.sub(r"\1", w)
        if len(w) > 4 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
            w = w[:-1]
        if len(w) > 6 and w.endswith("e"):
            w = w[:-1]                                                              # "ezetimib" / "ezetimibe", "glibenclamid" / "glibenclamide"
        words.append(w)
    return " ".join(words)


def tokens(text: str) -> list[str]:
    return normalize(text).split()


def letters(text: str) -> str:
    return _LETTERS.sub("", normalize(text))


def token_count(text: str) -> int:
    return len(tokens(text))
