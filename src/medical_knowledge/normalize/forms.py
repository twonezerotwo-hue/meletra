"""Form of a replacement: number (singular / plural, Latin and English) and letter case follow the words that were HEARD.

A correction that fixes the term but loses the plural ("exosomatic synapses" -> "axosomatic synapse") or writes a Title Case MeSH name in the middle of a sentence
("nerve system" -> "Nervous System") is a wrong correction. The knowledge base lists one form per term; these helpers adapt it to the sentence.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

from rapidfuzz.distance import Levenshtein

_LATIN_SINGULAR = (("men", "mina"), ("um", "a"), ("on", "a"), ("us", "i"), ("is", "es"), ("ix", "ices"), ("ex", "ices"), ("a", "ae"), ("nx", "nges"), ("ma", "mata"))
_NOT_PLURAL_S = ("ss", "us", "is", "ous", "ics", "sis")
_LATIN_ENDINGS = ("a", "um", "us", "is", "ex", "ix", "on", "ma", "men", "ae", "i", "es")


def _last_word(text: str) -> tuple[str, str]:
    m = re.search(r"([A-Za-z]+)([^A-Za-z]*)$", text)
    if not m:
        return text, ""
    return text[: m.start(1)], m.group(1)


def plural_like(word: str) -> bool:
    w = word.lower()
    if w.endswith(("ae", "ices", "nges", "mata")):
        return True
    return w.endswith("s") and not w.endswith(_NOT_PLURAL_S) and len(w) > 3


def regular_plural(word: str) -> str:
    """The regular English plural (-s / -es / -ies) - never a Latin form."""
    lw = word.lower()
    if lw.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    if lw.endswith("y") and len(lw) > 2 and lw[-2] not in "aeiou":
        return word[:-1] + "ies"
    return word + "s"


def pluralize(word: str) -> str:
    w = word
    lw = w.lower()
    for singular, plural in _LATIN_SINGULAR:
        if lw.endswith(singular) and len(lw) - len(singular) >= 3 and (singular not in ("a", "is", "ex") or lw.endswith(("ia", "ma", "ra"))):
            if singular == "a" and not lw.endswith(("ra", "ia")):
                continue
            return w[: len(w) - len(singular)] + plural
    if lw.endswith(("s", "x", "z", "ch", "sh")):
        return w + "es"
    if lw.endswith("y") and len(lw) > 2 and lw[-2] not in "aeiou":
        return w[:-1] + "ies"
    return w + "s"


def singularize(word: str) -> str:
    lw = word.lower()
    if lw.endswith("ies") and len(lw) > 4:
        return word[:-3] + "y"
    if lw.endswith(("ches", "shes", "xes", "sses")):
        return word[:-2]
    if lw.endswith("s") and not lw.endswith(_NOT_PLURAL_S):
        return word[:-1]
    return word


def _order(candidates: list[str], word: str, heard: str) -> list[str]:
    """The heard word first when it is one of the candidates (it shows which plural the speaker used), then Latin forms for a Latin-looking word, else regular ones."""
    latin = [c for c in candidates if c != regular_plural(word) and c != singularize(word)]
    regular = [c for c in candidates if c not in latin]
    ordered = (latin + regular) if word.lower().endswith(_LATIN_ENDINGS) else (regular + latin)
    first = [c for c in ordered if c.lower() == heard.lower()]
    return list(dict.fromkeys(first + ordered))


def _plural_candidates(word: str, heard: str = "") -> list[str]:
    """Strings that MIGHT be the plural of ``word``. They are lookup keys only: a form is written into the transcript only after it was verified in the sources."""
    out = [regular_plural(word)]
    lw = word.lower()
    for singular, plural in _LATIN_SINGULAR:
        if lw.endswith(singular) and len(lw) - len(singular) >= 3:
            out.append(word[: len(word) - len(singular)] + plural)
    return _order(list(dict.fromkeys(out)), word, heard)


def _singular_candidates(word: str, heard: str = "") -> list[str]:
    lw = word.lower()
    out = [singularize(word)]
    for singular, plural in _LATIN_SINGULAR:
        if lw.endswith(plural) and len(lw) - len(plural) >= 3:
            out.append(word[: len(word) - len(plural)] + singular)
    found = [c for c in dict.fromkeys(out) if c.lower() != lw]
    first = [c for c in found if c.lower() == heard.lower()]
    return list(dict.fromkeys(first + found))


def adapt_number(heard: str, surface: str, proper: Optional[set] = None, verify: Optional[Callable[[str], bool]] = None) -> tuple[str, str]:
    """(text, status): the surface form with the number (singular / plural) of the heard span's last word.

    status  "same"        nothing to adapt (same number, same word, a proper name)
            "adapted"     the other number was needed and the adapted form was VERIFIED (``verify(phrase)`` is true: the sources have exactly this form, or it is a dictionary word)
            "unverified"  the other number was needed but no verified form exists: the text stays as the source wrote it and the caller must not treat the result as safe

    Latin plurals are never invented. "macula densa" + a heard "-s" would be "macula densas"; no source has it, so the result is "unverified", not a fabricated phrase.
    Without ``verify`` only the regular English plural of a word without a Latin ending is accepted."""
    pre_h, hw = _last_word(heard)
    pre_s, sw = _last_word(surface)
    if not hw or not sw or hw.lower() == sw.lower() or (proper and sw in proper):
        return surface, "same"
    h_plural, s_plural = plural_like(hw), plural_like(sw)
    # Latin plural heard against a Latin singular surface: perikarya / perikaryon
    if hw.lower().endswith("a") and sw.lower().endswith(("on", "um")) and hw.lower()[:-1][:4] == sw.lower()[:4]:
        h_plural = True
    if hw.lower().endswith(("i", "ina", "a", "es")) and hw.lower()[:4] == sw.lower()[:4] and not s_plural and pluralize(sw).lower()[:-1] == hw.lower()[:-1]:
        h_plural = True
    if h_plural == s_plural:
        return surface, "same"
    tail = surface[len(pre_s) + len(sw):]

    def check(phrase: str, word: str) -> bool:
        if verify is not None:
            return bool(verify(phrase))
        return not word.lower().endswith(_LATIN_ENDINGS)

    if h_plural and not s_plural:
        if not hw.lower().endswith(("a", "i", "ae")) and Levenshtein.normalized_similarity(singularize(hw).lower(), sw.lower()) < 0.9:
            return surface, "same"                                   # the final "s" is part of the misheard word, not a plural
        for cand in _plural_candidates(sw, hw):
            phrase = pre_s + cand + tail
            if check(phrase, sw):
                return phrase, "adapted"
        return surface, "unverified"
    if not h_plural and s_plural and sw.lower().endswith("s") and not sw.lower().endswith(("ae", "ices", "nges", "mata")):
        for cand in _singular_candidates(sw, hw):
            phrase = pre_s + cand + tail
            if check(phrase, cand):
                return phrase, "adapted"
        return surface, "unverified"
    return surface, "same"


def match_number(heard: str, surface: str, proper: Optional[set] = None, verify: Optional[Callable[[str], bool]] = None) -> str:
    """``adapt_number`` without the status."""
    return adapt_number(heard, surface, proper, verify)[0]


def strip_qualifier(surface: str) -> str:
    """A parenthetical qualifier ("Carboxykinase (GTP)") is a dictionary disambiguation, never spoken."""
    return re.sub(r"\s*\([^)]*\)", "", surface).strip()


def trim_neighbours(surface: str, before: str, after: str) -> str:
    """Words the replacement shares with the text right after / before the span are already there: "pyre" + "patches" -> "Peyer", not "Peyer patches patches"."""
    words = surface.split(" ")
    nxt = after.split()
    while len(words) > 1 and nxt and words[-1].lower().strip(".,") == nxt[0].lower().strip(".,"):
        words.pop()
        nxt = nxt[1:]
    prev = before.split()
    while len(words) > 1 and prev and words[0].lower().strip(".,") == prev[-1].lower().strip(".,"):
        words.pop(0)
        prev = prev[:-1]
    return " ".join(words)


_EPONYM_HEADS = re.compile(r"^(disease|syndrome|nerve|cell|cells|reflex|sign|test|law|effect|pathway|ligament|gland|duct|organ|body|bodies|fibers?|fibres?|bundle|tract|cycle|method|stain|"
                           r"reaction|reagent|virus|bacillus|apparatus|complex|angle|canal|plexus|vein|artery|muscle|membrane|layer|fascia|sphincter|nucleus|lines?|rings?|rods?)$", re.I)


def match_case(heard: str, surface: str, is_word: Optional[Callable[[str], bool]] = None, sentence_start: bool = False,
               eponyms: Optional[set] = None) -> str:
    """Letter case of the replacement: a lower-case heard span gets a natural-case term (MeSH writes "Nervous System", "Aconitase"); acronyms, mixed case and
    proper names (eponyms of the vocabulary, or a name followed by "disease", "syndrome", "cell" ...) stay capitalised."""
    out = surface
    if heard[:1].islower() and out[:1].isupper():
        words = out.split(" ")
        title = all(w[:1].isupper() for w in words if w[:1].isalpha() and len(w) > 3) or len(words) == 1
        if title:
            def lower_it(i: int, w: str) -> bool:
                if not w[:1].isupper() or w[1:].isupper() or re.search(r"\d|[A-Z]", w[1:]):
                    return False
                if eponyms and w in eponyms:
                    return False
                nxt = words[i + 1] if i + 1 < len(words) else ""
                return not (nxt and _EPONYM_HEADS.match(nxt) and not (is_word and is_word(w.lower())))
            out = " ".join(w.lower() if lower_it(i, w) else w for i, w in enumerate(words))
    if sentence_start and out[:1].islower() and heard[:1].isupper():
        out = out[:1].upper() + out[1:]
    return out
