"""How much a heard word looks like a REAL word - a graded number, not a yes/no.

    1.00 .. 0.70   ordinary English (Whisper's common words, SCOWL levels 10-70): "spinal", "between", "signals", "formation", "body", "cell"
    0.45           a rare dictionary word (SCOWL 80)
    0.55           an inflected form of a dictionary word ("signalled", "cells"), a Latin number form of a known term ("foramina"), a word of the medical vocabulary
    0.40           a derivative of a known word ("-ation", "-ic", "-ous" ... added to a word the dictionary has) or a compound of two known words
    0.00           not a word of any list or recognisable structure

The gate uses it as the "keep the original" prior: the more real the heard word is, the more evidence a replacement needs. Structural rules (inflection, derivation, compounds, Latin number) come
BEFORE any list, so a word that no list contains but that is built correctly from words that are in the lists is still protected - without a hand-written list of protected words.
Morphology never proves a word is real (a garble such as "neuralation" also decomposes); that is why derivatives and compounds only get a MODERATE value.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ...common_words import is_common_word
from .cache import LRU
from .knowledge import MedicalKnowledge

BUILT_MIN, BUILT_MAX = 0.35, 0.80      # a word BUILT from known words (derivative / compound): the longer the known component, the more likely the word is real
_LEVEL_VALUE = ((20, 0.97), (35, 0.92), (50, 0.86), (60, 0.80), (70, 0.72), (80, 0.45))
_DERIVATIONAL = ("ization", "isation", "ational", "ically", "ological", "ology", "ation", "ition", "ative", "ively", "ments", "ment", "ness", "ities", "ity", "ical",
                 "ally", "ious", "eous", "ous", "ive", "ize", "ise", "ing", "ism", "ist", "ful", "less", "able", "ible", "ant", "ent", "ial", "ary", "ory", "ic", "al", "ly", "er", "or")
_LATIN_SINGULARS = (("ae", ("a",)), ("i", ("us", "os")), ("a", ("um", "on")), ("es", ("is", "ex", "ix")), ("ices", ("ex", "ix")), ("ina", ("en",)), ("mata", ("ma",)), ("nges", ("nx",)))
_ALPHA = re.compile(r"^[a-z]+$")


@dataclass(frozen=True)
class Validity:
    value: float
    reason: str


class WordValidity:
    def __init__(self, kb: MedicalKnowledge) -> None:
        self.kb = kb
        self._cache: LRU = LRU(32768)

    # ------------------------------------------------------------------ building blocks
    def _level_value(self, word: str) -> float:
        if is_common_word(word):
            return 0.95
        level = self.kb.word_level(word)
        if level is None:
            return 0.0
        for limit, value in _LEVEL_VALUE:
            if level <= limit:
                return value
        return 0.45

    def _vocab_token(self, word: str, minimum: int = 1) -> bool:
        return self.kb.token_frequency(word) >= minimum

    def _known_base(self, word: str) -> bool:
        """A base word the dictionary (levels up to 70) or the vocabulary (several terms) has - the evidence a derivative / compound is built from."""
        return len(word) >= 4 and (self.kb.is_english_word(word, 70) or is_common_word(word) or self._vocab_token(word, 3))

    def _inflection_bases(self, w: str) -> list[str]:
        bases = []
        if w.endswith("ies") and len(w) > 4:
            bases.append(w[:-3] + "y")
        if w.endswith("es") and len(w) > 4:
            bases.append(w[:-2])
        if w.endswith("s") and len(w) > 3:
            bases.append(w[:-1])
        for suffix, repl in (("ed", ""), ("ed", "e"), ("ing", ""), ("ing", "e"), ("ly", ""), ("er", ""), ("est", "")):
            if w.endswith(suffix) and len(w) > len(suffix) + 3:
                bases.append(w[:-len(suffix)] + repl)
        return bases

    def _latin_bases(self, w: str) -> list[str]:
        out = []
        for ending, singulars in _LATIN_SINGULARS:
            if w.endswith(ending) and len(w) > len(ending) + 3:
                out += [w[:-len(ending)] + s for s in singulars]
        return out

    def _derived(self, w: str) -> int:
        """Length of the known stem of a derivative (0 = not a derivative)."""
        best = 0
        for suffix in _DERIVATIONAL:
            if w.endswith(suffix) and len(w) - len(suffix) >= 4:
                stem = w[:-len(suffix)]
                if self._known_base(stem) or self._known_base(stem + "e") or (stem.endswith("i") and self._known_base(stem[:-1] + "y")):
                    best = max(best, len(stem))
        return best

    def _compound(self, w: str) -> int:
        """Length of the longest known component of a compound (0 = not a compound)."""
        if len(w) < 9:
            return 0
        best = 0
        for i in range(4, len(w) - 3):
            if self._known_base(w[:i]) and self._known_base(w[i:]):
                best = max(best, i, len(w) - i)
        return best

    @staticmethod
    def _built_value(component: int) -> float:
        """0.35 for a short known component (a garble can contain "phage"), up to 0.80 for a long one ("deamination", "polymerase"): a long real word inside a word makes it real."""
        return BUILT_MIN + (BUILT_MAX - BUILT_MIN) * max(0.0, min(1.0, (component - 5) / 5.0))

    # ------------------------------------------------------------------ the answer
    def of(self, word: str) -> Validity:
        w = word.lower()
        return self._cache.get(w, lambda: self._compute(w))

    def _compute(self, w: str) -> Validity:
        if len(w) < 3:
            return Validity(0.9, "short token")
        if not _ALPHA.match(w):
            return Validity(0.9, "contains digits or symbols: an identifier, not a spoken word")
        value = self._level_value(w)
        if value:
            return Validity(value, "dictionary word")
        if self._vocab_token(w):
            return Validity(0.55, "word of the medical vocabulary")
        best = 0.0
        for base in self._inflection_bases(w):
            v = self._level_value(base)
            if v:
                best = max(best, 0.55 if v >= 0.7 else 0.4)
            elif self._vocab_token(base):
                best = max(best, 0.5)
        if best:
            return Validity(best, "inflected form of a known word")
        for base in self._latin_bases(w):
            if self._vocab_token(base) or self.kb.is_english_word(base, 80):
                return Validity(0.55, "Latin number form of a known word")
        derived = self._derived(w)
        if derived:
            return Validity(self._built_value(derived), "derived from a known word")
        compound = self._compound(w)
        if compound:
            return Validity(self._built_value(compound), "compound of known words")
        return Validity(0.0, "not a word")
