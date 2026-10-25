"""More validated facts from a noisy spoken lecture, without lowering the bar a fact has to pass (``factbase.py`` re-validates everything these rules produce).

Three kinds of sentence were lost although the lecture said something clear:

* a word the recogniser was unsure of, although it is an ordinary or a known medical word that the lecture uses again and again ("neural", "tube", "cord"): such a doubt is
  CORROBORATED by the rest of the lecture; a doubted word that is rare, or unknown, still rejects the sentence;
* the lecturer's own lead-in ("we can see that ...", "remember that ...", "as I mentioned, ..."): the statement after it stands on its own; "cells ..., they locate" loses the
  resumptive pronoun;
* a statement that continues the previous sentence with "It" / "They" ("The intermediate zone contains neuronal cell bodies. It later forms the gray matter."): the pronoun is
  replaced by the subject term of the previous sentence when the number agrees - the fact then cites both sentences.

A fact made this way is MEDIUM (directly supported, but context-dependent); one that needed nothing of this is HIGH. LOW facts are rejected as before.
"""

from __future__ import annotations

import re
from typing import Callable, Mapping, Optional

from . import facts as F
from . import text as T

HIGH, MEDIUM, LOW = "HIGH", "MEDIUM", "LOW"
CORROBORATING_MENTIONS = 3                        # a doubted word heard at least this often in the lecture (the doubted instance included) is corroborated

_LEAD_INS = (
    (re.compile(r"^(?:and\s+|so\s+|now\s+)?(?:based\s+on\s+(?:this|the|these)\s+[\w-]+(?:\s+[\w-]+)?,?\s+)?(?:we|you)\s+(?:can|could)\s+(?:see|say|notice|observe|conclude)\s+"
                r"(?:here\s+)?that\s+", re.IGNORECASE), False),
    (re.compile(r"^(?:and\s+|so\s+)?(?:we|you)\s+(?:know|learned|have\s+learned|have\s+seen|saw)\s+that\s+", re.IGNORECASE), False),
    (re.compile(r"^(?:and\s+|so\s+)?(?:we|you)\s+(?:should|need\s+to|have\s+to|must)\s+(?:know|remember|keep\s+in\s+mind|understand)\s+that\s+", re.IGNORECASE), True),
    (re.compile(r"^(?:and\s+|so\s+)?(?:please\s+)?(?:remember|note|keep\s+in\s+mind|notice|do\s+not\s+forget|don't\s+forget)\s+that\s+", re.IGNORECASE), True),
    (re.compile(r"^(?:and\s+|so\s+)?(?:trust|believe)\s+me(?:\s+that)?,?\s+", re.IGNORECASE), False),
    (re.compile(r"^(?:and\s+|so\s+)?(?:it\s+is|it's)\s+(?:important|essential|crucial|worth\s+knowing)\s+to\s+(?:know|remember|note|understand)\s+that\s+", re.IGNORECASE), True),
    (re.compile(r"^(?:and\s+|so\s+)?as\s+(?:you\s+know|we\s+know|we\s+learned|we\s+have\s+seen|i\s+(?:mentioned|said|told\s+you)(?:\s+before)?|mentioned\s+before),\s+",
                re.IGNORECASE), False),
)
_RESUMPTIVE = re.compile(r"^(?P<np>[A-Z][^,]{4,80}),\s+(?:they|it)\s+(?P<rest>[a-z][\w-]*\b.*)$")
_PRONOUN_LEAD = re.compile(r"^(?P<pron>It|They)\s+(?P<rest>(?P<verb>[a-z][\w-]*)\b.*)$")
_IMPERSONAL = re.compile(r"^(?:is|was|will\s+be|can\s+be|may\s+be|would\s+be|seems?|looks?|depends|means|happens|takes|makes\s+sense)\b|"
                         r"^(?:is|was)\s+(?:also\s+)?(?:important|possible|impossible|difficult|easy|hard|clear|necessary|interesting|known|thought|believed|likely|unlikely|"
                         r"said|shown|true|false|enough|time|here|there|not|very|so|quite|about|like|what|how|why|\d)", re.IGNORECASE)


_ADVERBS = frozenset("later also then often usually normally still already soon first finally eventually gradually mainly partly only always never further".split())


def doubt_is_corroborated(word: str, counts: Mapping[str, int], unknown: Optional[Callable[[str], bool]] = None) -> bool:
    """A doubted word is no reason to drop the sentence when it is a function word, or a known word the lecture says again and again."""
    low = word.casefold()
    if len(low) < 4 or low in T.STOPWORDS:
        return True
    if unknown is not None and unknown(low):
        return False
    return counts.get(low, 0) >= CORROBORATING_MENTIONS


def strip_lead_in(text: str) -> tuple[str, bool]:
    """``("the statement after the lecturer's lead-in", emphasis)``; ``("", False)`` when the sentence has no such lead-in."""
    body = text.strip()
    for pattern, emphatic in _LEAD_INS:
        m = pattern.match(body)
        if m and len(body) - m.end() >= 12:
            rest = drop_resumptive_pronoun(F.clean_sentence(body[m.end():]).text)
            return (rest[:1].upper() + rest[1:]) if rest else "", emphatic
    return "", False


def drop_resumptive_pronoun(text: str) -> str:
    """"Cells with a high proliferation capacity, they locate near the central canal." -> "Cells with a high proliferation capacity locate near the central canal.":
    only when the part before the comma is a noun phrase (no verb in it) and a verb follows the pronoun."""
    m = _RESUMPTIVE.match(text.strip())
    if not m:
        return text
    np_words, rest_words = T.words(m.group("np")), T.words(m.group("rest"))
    if not rest_words or F._has_verb(np_words) or not (rest_words[0] in F.VERB_FORMS or rest_words[0] in F._AUX):
        return text
    return f"{m.group('np')} {m.group('rest')}"


def resolve_pronoun(text: str, antecedent: str, plural: bool, article: str = "The") -> str:
    """"It later forms the gray matter." + antecedent "intermediate zone" -> "The intermediate zone later forms the gray matter." ("" when the rule does not apply):
    It needs a singular antecedent, They a plural one; impersonal "It is important / It can be 1 to 500 ..." is never resolved."""
    m = _PRONOUN_LEAD.match(text.strip())
    if not m or not antecedent:
        return ""
    pron, rest = m.group("pron"), m.group("rest")
    if (pron == "It") == plural:
        return ""
    if _IMPERSONAL.match(rest):
        return ""
    first = T.words(rest)[:2]
    adverb = bool(first) and (first[0].endswith("ly") or first[0] in _ADVERBS)
    if not first or not (first[0] in F.VERB_FORMS or first[0] in F._AUX or (len(first) > 1 and adverb and (first[1] in F.VERB_FORMS or first[1] in F._AUX))):
        return ""
    name = antecedent if (antecedent[:2].isupper() or not article) else antecedent[:1].lower() + antecedent[1:]
    head = f"{article} {name}" if article else name
    return head[:1].upper() + head[1:] + " " + rest


_POINTS_BACK = re.compile(r"(?<!^)\b(?:this|these|that|those)\s+(?!is\b|are\b|was\b|were\b|will\b|can\b|may\b|has\b|have\b|means\b)([a-z][\w-]+)|"
                         r"\b(?:prevent|avoid|do|does|did|explain|explains|show|shows|mean|means|with|after|before|like|from|by|of)\s+(?:this|that)\s*[,.;]|"
                         r"\b(?:above|below|beyond|than|besides|unlike)\s+(?:this|that|these|those)\b|\b(?:anywhere|somewhere|nowhere)\s+else\b|\belsewhere\b|"
                         r"^the\s+(?:first|second|third|next|last|final|main)\s+[a-z]+\s+(?:is|are|was|were)\b|"
                         r"\bsuch\s+(?!as\b)[a-z][\w-]+|^(?:or|more|less|fewer|another|other|both|either|neither|also|again|then|therefore|thus|hence|otherwise|instead|however|"
                         r"similarly|likewise|accordingly|consequently|meanwhile)\b|^one\s+(?:is|was|of\s+them|component|part|side|type|kind)\b|^the\s+other\b|"
                         r"\b(?:the\s+same\s+(?!time\b)[a-z][\w-]+|these\s+two|those\s+two|the\s+former|the\s+latter|the\s+above|the\s+following|the\s+previous|the\s+other\s+one)\b",
                         re.IGNORECASE)


_GENERIC_SUBJECT = re.compile(r"^(?:[A-Za-z][\w ]{1,30},\s+)?[Tt]he\s+(?:enzyme|enzymes|process|processes|structure|structures|proteins?|molecules?|factors?|cells|phenomenon|substances?|area|region|part|step|stage|"
                              r"mechanism|reaction|change|changes|thing|things|products?|results?|outcome|task|ending|networks?)\s+(?:is|are|was|were|will|can|could|may|has|have|had|[a-z]+s)\b")


def points_back(statement: str) -> str:
    """The words with which a statement points at something said before ("Parallel with THIS PROCESS ...", "will have THE SAME organization", "between THESE TWO"): such a
    statement does not stand alone, whatever its grammar. "" when it stands alone. (A demonstrative at the very start is turned into an article earlier.)"""
    for m in _POINTS_BACK.finditer(statement):
        follower = (m.group(1) or "").casefold()
        if m.group(0).casefold().startswith("that ") and (follower in F.VERB_FORMS or follower in F._AUX or follower in T.STOPWORDS or follower.endswith("ly")):
            continue                                                 # "the pathway THAT converts ...": a relative pronoun, not a pointer
        return m.group(0)
    g = _GENERIC_SUBJECT.match(statement.strip())                    # "The process is called primary neurulation.": WHICH process was said before
    return " ".join(g.group(0).split()[:2]) if g else ""


RESCUES = ("lead-in removed", "pronoun resolved from the previous sentence", "clause of a longer sentence")


def usable(fact, known_term, category_of=None) -> bool:
    """HIGH facts are used. A MEDIUM fact only when its support is clear: ONE rescue and nothing else doubtful (no corroborated recogniser doubt, no pointing back), about a
    term the lecture knows that is a real thing (a structure, a cell, a process ... - not an unclassified phrase), and short enough to stand alone."""
    if fact.tier == HIGH:
        return True
    if fact.tier != MEDIUM:
        return False
    if any(not d.startswith(RESCUES) for d in fact.doubts) or sum(1 for d in fact.doubts if d.startswith(RESCUES)) != 1:
        return False
    if len(T.words(fact.statement)) > 28:
        return False
    if not fact.subtopic and fact.doubts == ["lead-in removed"]:
        # the whole sentence minus "we can see that": no fragment, so a statement about an unnamed subject ("Cells with high proliferation capacity locate near the central
        # canal") is fine when it names a known term of a real kind
        return any(known_term(t) and (category_of is None or category_of(t) not in ("", "other")) for t in getattr(fact, "medical_terms", []))
    if not fact.subtopic or not known_term(fact.subtopic):
        return False
    return category_of is None or category_of(fact.subtopic) not in ("", "other")


def tier_of(soft: str, rescued: bool) -> str:
    """HIGH: the statement needed nothing; MEDIUM: it was rescued or carries a soft doubt."""
    return MEDIUM if rescued or any(s for s in soft.split("|") if s) else HIGH
