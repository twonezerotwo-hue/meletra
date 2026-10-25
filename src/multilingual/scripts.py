"""Writing-system evidence: which script a text is written in, and which spoken languages that script allows.

Whisper reports ONE language per decoding window and writes everything in the script of the language it was told to use, so the text itself is the cheapest honest witness of what
was recognised: Hangul can only be Korean, Devanagari (here) Hindi, Hiragana / Katakana Japanese, Han characters Chinese (or Japanese, when kana are present). Latin text can be any
of the Latin-script languages the program offers, so it says nothing by itself. Nothing here guesses a language from words; it only refuses combinations that cannot be true.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Iterable, Optional

#: script -> the program's language codes written in it (Latin is shared by several)
SCRIPT_LANGUAGES: dict[str, tuple[str, ...]] = {
    "hangul": ("ko",),
    "devanagari": ("hi",),
    "han": ("zh",),
    "kana": ("ja",),
    "cyrillic": ("ru", "uk", "bg", "sr"),
    "arabic": ("ar", "fa", "ur"),
    "greek": ("el",),
    "latin": ("en", "tr", "de", "hu", "nl", "fr", "es"),
}
LATIN_LANGUAGES = SCRIPT_LANGUAGES["latin"]
#: the language each non-Latin script stands for when the program offers exactly one (used as evidence, not as a detector)
UNIQUE_SCRIPT_LANGUAGE = {"hangul": "ko", "devanagari": "hi", "han": "zh", "kana": "ja"}


def script_of(char: str) -> Optional[str]:
    """The script of one character (``None`` for digits, punctuation, spaces, symbols)."""
    name = unicodedata.name(char, "")
    if not char.isalpha():
        if unicodedata.category(char).startswith("M") and "DEVANAGARI" in name:                                       # Hindi vowel signs and the virama are part of the word, not punctuation
            return "devanagari"
        return None
    if "HANGUL" in name:
        return "hangul"
    if "DEVANAGARI" in name:
        return "devanagari"
    if "HIRAGANA" in name or "KATAKANA" in name:
        return "kana"
    if "CJK" in name or "IDEOGRAPH" in name:
        return "han"
    if "CYRILLIC" in name:
        return "cyrillic"
    if "ARABIC" in name:
        return "arabic"
    if "GREEK" in name:
        return "greek"
    if "LATIN" in name:
        return "latin"
    return "other"


def script_counts(text: str) -> Counter:
    """Letters per script (digits, punctuation and symbols are not counted)."""
    counts: Counter = Counter()
    for char in text or "":
        script = script_of(char)
        if script:
            counts[script] += 1
    return counts


def dominant_script(text: str, minimum_share: float = 0.5) -> Optional[str]:
    """The script most of the letters belong to, when it has at least ``minimum_share`` of them (Han + kana together are reported as kana)."""
    counts = script_counts(text)
    if counts.get("kana"):
        counts["kana"] += counts.pop("han", 0)
    total = sum(counts.values())
    if not total:
        return None
    script, n = counts.most_common(1)[0]
    return script if n / total >= minimum_share else None


def languages_for_script(script: Optional[str]) -> tuple[str, ...]:
    return SCRIPT_LANGUAGES.get(script or "", ())


def script_language(text: str, minimum_share: float = 0.5) -> Optional[str]:
    """The language the writing system proves, or ``None`` when it proves none (Latin letters fit several languages; mixed scripts fit none)."""
    script = dominant_script(text, minimum_share)
    return UNIQUE_SCRIPT_LANGUAGE.get(script or "")


def compatible(text: str, language: Optional[str], minimum_share: float = 0.5) -> bool:
    """Can ``text`` have been written in ``language``? A text without letters, an unknown language and a mixed-script text are not contradicted."""
    if not language:
        return True
    script = dominant_script(text, minimum_share)
    if script is None:
        return True
    allowed = languages_for_script(script)
    return not allowed or language in allowed or script == "other"


#: letters of the scholarly Latin spelling of Indian languages (IAST: macron vowels, dotted consonants, ś): none of the Latin-script languages the program offers is written with them
INDIC_ROMAN_LETTERS = frozenset("āēīōūṛṝḷḹṃḥṅṭḍṇṣś")


def romanised_indic(text: str, minimum_words: int = 3, minimum_share: float = 0.04) -> bool:
    """Is ``text`` Latin letters that spell Hindi / Sanskrit-family sounds ("Kīyārē dhūbē kēsē")? Whisper forced into English writes sung Hindi that way: it is not English, and
    translating it as English would be invented. A name or two (Śiva, Tōkyō) is not enough: at least ``minimum_words`` words carry such letters and they are ``minimum_share``
    of the letters."""
    normal = unicodedata.normalize("NFC", text or "")
    letters = [c for c in normal if c.isalpha()]
    if len(letters) < 6:
        return False
    marks = sum(1 for c in letters if c.lower() in INDIC_ROMAN_LETTERS)
    words = sum(1 for word in normal.split() if any(c.lower() in INDIC_ROMAN_LETTERS for c in word))
    return words >= minimum_words and marks / len(letters) >= minimum_share


#: closed-class English words (articles, pronouns, auxiliaries, prepositions): a line made of them is English whatever the language of the run it was decoded in
ENGLISH_FUNCTION_WORDS = frozenset(
    "a an the and or but if so as of to in on at by for with from up down out over into about than then there here this that these those i me my mine we us our you your he him his "
    "she her it its they them their is am are was were be been being do does did done have has had will would can could should may might must not no yes dont didnt doesnt isnt arent "
    "cant wont im ive youre its thats what who when where why how which all any some one more just only very too also again never ever now oh".split())


def looks_english(text: str, minimum_hits: int = 2, minimum_share: float = 0.3) -> bool:
    """Latin text made of enough English function words ("Did you check on me?", "I love you, I love you"). Romanised Hindi ("Tu mera koi na hoke") has almost none. Used only to decide
    what to call Latin text that sits in a run of a non-Latin language - never to detect a language of a recording."""
    words = [w.replace("'", "").replace("’", "") for w in re.findall(r"[A-Za-z']+", text or "")]
    words = [w.lower() for w in words if w]
    if len(words) < 2:
        return False
    hits = sum(1 for w in words if w in ENGLISH_FUNCTION_WORDS)
    return hits >= minimum_hits and hits / len(words) >= minimum_share


def scripts_present(texts: Iterable[str], minimum_letters: int = 3) -> dict[str, int]:
    """Letters per script over several texts, ignoring scripts with fewer than ``minimum_letters`` letters (a stray character is not a script)."""
    total: Counter = Counter()
    for text in texts:
        total.update(script_counts(text))
    return {script: n for script, n in total.items() if n >= minimum_letters}


def information_units(text: str) -> int:
    """How much written content a text holds, in Latin-letter equivalents: a Hangul syllable or a Han character stands for about two letters (used to compare outputs across scripts)."""
    total = 0
    for char in text or "":
        script = script_of(char)
        if script:
            total += 2 if script in ("hangul", "han", "kana") else 1
    return total
