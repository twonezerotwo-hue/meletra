"""The shared knowledge layer of the Study engine: validated, atomic, source-grounded :class:`StudyFact` objects.

Summary, key concepts, flashcards, questions and Study Mode do NOT read the transcript on their own. They all start from the same facts, so
they cannot contradict each other and a noisy transcript sentence can never become study material just because it was said:

    transcript sentence -> cleaned (fillers, teacher phrases, stutters, deictic words) -> validated (grammar, one proposition, no ASR
    artefact, no unresolved medical term, no language drift) -> typed fact (definition, location, derivative, ...) -> study items

Everything here is rule based and offline. The facts are the lecture's own words, tidied: nothing is added from outside the lecture.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence

from . import text as T

FACT_TYPES = ("definition", "function", "location", "classification", "mechanism", "sequence", "development", "comparison", "cause_effect",
              "clinical_relevance", "structure_function", "derivative", "relationship")

# ================================================================================================================ cleaning
#: how a lecturer introduces what the exam wants ("you have to know ..."): removed from the study text, remembered as emphasis
_TEACHER = re.compile(
    r"""\b(?:
        you\s+(?:have|need|should|must|will\s+have)\s+to\s+(?:know|remember|recogni[sz]e|memori[sz]e|learn|understand)(?:\s+(?:that|this|the))?
      | you\s+should\s+(?:know|remember|recogni[sz]e)(?:\s+(?:that|this))?
      | you\s+(?:need|have)\s+to\s+be\s+able\s+to\s+(?:recogni[sz]e|identify)
      | (?:it\s+is|it's|this\s+is|that\s+is|that's)\s+(?:very\s+|really\s+|quite\s+|also\s+)?(?:important|crucial|essential|key)(?:\s+(?:to\s+(?:know|remember|note|understand)|that))?
      | (?:very\s+|really\s+)?important(?:\s+(?:is|here\s+is))?\s+(?:to\s+(?:know|remember|note)|that)
      | please\s+(?:note|remember)(?:\s+that)?
      | (?:i\s+want\s+you\s+to\s+)?(?:remember|note|keep\s+in\s+mind|don't\s+forget|do\s+not\s+forget)(?:\s+(?:that|this))?
      | this\s+is\s+(?:what\s+)?(?:we|you)\s+need\s+to\s+(?:remember|know)(?:\s+that)?
      | for\s+the\s+(?:final\s+)?exam
      | this\s+is\s+(?:a\s+)?(?:high[-\s]yield|exam\s+(?:question|topic|material))
      | clinically\s+(?:important|relevant)
    )\b[,:]?\s*""", re.IGNORECASE | re.VERBOSE)
_FILLER_START = re.compile(r"^(?:(?:so|okay|ok|well|now|and|but|alright|all\s+right|right|yeah|yes|then|um+|uh+|erm|like|actually|basically|anyway|"
                           r"of\s+course|as\s+i\s+said|as\s+i\s+mentioned|as\s+you\s+know|you\s+know|i\s+mean|in\s+fact|indeed|moreover|furthermore|"
                           r"also|again|therefore|thus|hence|however)\b[,\s]*)+", re.IGNORECASE)
_FILLER_ANY = re.compile(r"\b(?:um+|uh+|erm|hmm+|you\s+know|i\s+mean|kind\s+of|sort\s+of|okay|ok|of\s+course|alright|all\s+right)\b[,.]?\s*", re.IGNORECASE)
_CONTENT_ECHO = re.compile(r"\b(?!(?:very|really|never|there|their|these|those|where|which|other|about|after|again|being|could|would)\b)([A-Za-z]{5,})\s+\1\b", re.IGNORECASE)
_MULTI_SPACE = re.compile(r"\s+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?])")
_TRAILING_ECHO = re.compile(r"\b([A-Za-z][\w-]{3,})[.!?]\s+\1[.!?]?$", re.IGNORECASE)         # "... epithelium. epithelium."

# ---- things that make a sentence unfit to be a fact on its own
_META = re.compile(r"^(?:i|we)(?:'ll|'d|'ve|'m|'re| will| would| shall| am| are| can| could| want| try| tried| hope| think| believe| mentioned| said| "
                   r"discussed| talked| showed| told| have| had| did| do| don't| didn't| won't| cannot| can't)\b|"
                   r"^(?:let's|let me|let us|please|thank|welcome|next slide|next time|see you|any questions)\b", re.IGNORECASE)
_META_ANY = re.compile(r"\b(?:as i (?:said|mentioned|told)|i (?:told|showed|mentioned|will show|will explain|want to (?:show|tell|say))|"
                       r"in the next (?:lecture|slide|video|class)|in the previous (?:lecture|class|semester)|last (?:lecture|week|time)|"
                       r"our (?:previous|last) (?:lecture|semester|class)|this (?:lecture|semester|course)|you (?:already )?(?:know|studied|learned|saw)|"
                       r"(?:can|will|could|would)\s+(?:you|we)\s+(?:see|find|notice)|if you (?:look|see|check|remember)|"
                       r"(?:try|tried)\s+to\s+(?:add|show|explain))\b", re.IGNORECASE)
_DEICTIC = re.compile(r"\b(?:this|that|these|those)\s+(?:slide|picture|figure|image|diagram|graph|table|arrow|photo|video|one\s+here)\b|"
                      r"\bas\s+you\s+can\s+see\b|\bon\s+the\s+(?:left|right)(?:\s+side)?\b|\bhere\s+(?:you|we)\s+(?:can\s+)?(?:see|have|find)\b|"
                      r"\bin\s+this\s+(?:picture|slide|figure|image|case|image)\b|\bthe\s+(?:green|red|blue|yellow|brown|black|white|pink|purple)\s+"
                      r"(?:one|ones|color|colour|stain|staining|area|cells?|arrows?|line|lines|part)\b|\bhere\s+(?:in|at|on|is|are)\s+(?:this|the)\b|"
                      r"\bright\s+(?:here|there)\b|\bover\s+here\b|\bdown\s+here\b|\bup\s+here\b|\bhere,|\b(?:these|those)\s+steps?\b|\bthis\s+step\b", re.IGNORECASE)
_PRONOUN_ANY = re.compile(r"^(?:it|they|he|she|its|their|there|here|which|who|what|his|her)\b", re.IGNORECASE)
_PRONOUN_CONTRACTION = re.compile(r"^(?:it|this|that|they|he|she|there|here)(?:'s|'re|'ll|'d|'ve|\u2019s|\u2019re|\u2019ll|\u2019d|\u2019ve)\b", re.IGNORECASE)
_PRONOUN_SUBJECT = re.compile(r"^(?:it|this|that|these|those|they|he|she|there|here|its|their|which|what|who|where|when|how|why)"
                              r"\s+(?:is|are|was|were|will|can|could|may|might|must|should|has|have|had|does|do|did|means|mean|comes?|goes?|gets?|"
                              r"becomes?|forms?|gives?|also|then|just|always|never|only|not|really|actually|basically|still|usually|"
                              r"later|first|next|called|named|known|seems?|appears?|looks?|leads?|results?|causes?|contains?|represents?|"
                              r"belongs?|originates?|starts?|begins?|ends?|continues?|remains?|stays?)\b", re.IGNORECASE)
_AGREEMENT = re.compile(r"\b(?:this\s+(?:is\s+)?(?:those|these)|these\s+is\b|those\s+is\b|this\s+are\b|that\s+are\b|a\s+(?:those|these)|"
                        r"an\s+(?:the|those|these)|the\s+an?\b|(?:is|are)\s+(?:is|are)\b|to\s+(?:is|are|was|were)\b|"
                        r"(?:the|a|an)\s+(?:the|a|an)\b|\bof\s+of\b|\bin\s+in\b|\bis\s+(?:that\s+)?is\b)", re.IGNORECASE)
_DANGLING_END = re.compile(r"\b(?:a|an|the|of|to|and|or|but|that|which|is|are|was|were|with|for|in|on|at|by|from|as|than|because|so|if|when|while|"
                           r"then|this|these|those|its|their|his|her|our|your|my|be|been|being|will|can|could|would|should|has|have|had|do|does|did|"
                           r"also|not|very|more|most|some|any|each|every|other|another|such|into|onto|about|between|during|after|before|within|"
                           r"without|through|over|under|up|down|out|off|where|who|whom|whose)[.!?]?$", re.IGNORECASE)
_FIRST_PERSON = re.compile(r"\b(?:i|i'm|i've|i'll|i'd|me|we|we're|we've|we'll|we'd|my|our|ours|us|let's|you|you're|you'll|your|everybody|everyone|anybody|anyone|somebody)\b",
                           re.IGNORECASE)
_SUBORDINATE_FRAGMENT = re.compile(r"^(?:as|when|while|if|once|after|before|since|although|because|whereas|unless|until|so\s+that)\b", re.IGNORECASE)
_PREP_NO_SUBJECT = re.compile(r"^(?:(?:in|on|at|for|during|with|from|by)\s+(?:the\s+)?[\w-]+\s+(?:is|are|was|were|has|have)\b|"
                              r"(?:according\s+to|without|within|through|after|before|depending\s+on)\s+[^,]{2,60}?\s+(?:can|could|may|might|will|would|must|should|is|are|was|were|has|have)\b)",
                              re.IGNORECASE)
_RELATIVE_START = frozenset(("which", "that", "who", "whom", "whose", "where"))
_PRONOUN_AFTER_INTRO = re.compile(r"^[^.]{3,90}?,\s+(?:this|it|they|these|those|he|she)\s+(?:will|is|are|can|has|have|would|could|does|do|was|were|may|might|also|"
                                 r"[a-z]+s|[a-z]+ed)\b", re.IGNORECASE)
_ONLY_MATTERS = re.compile(r"^[\w\s,()-]{3,70}?\s+(?:is|are|was|were|will\s+be|would\s+be|becomes?)\s+(?:very\s+|also\s+|really\s+|quite\s+|so\s+)?"
                           r"(?:important|relevant|necessary|interesting|essential|crucial|significant|useful|critical|key)\s*[.!]?$", re.IGNORECASE)
_TO_FRAGMENT = re.compile(r"^to\s+[a-z]+\b(?![^,]*,)", re.IGNORECASE)
#: nouns too general to be the answer of a card or a wrong option on their own ("the _____ is called N-acetylglutamate synthase" -> "enzyme")
GENERIC_NOUNS = frozenset("""enzyme enzymes protein proteins product products substance substances molecule molecules reaction reactions factor factors compound compounds
cell cells structure structures process processes tissue tissues organ organs part parts area areas region regions thing things""".split())


def is_bare_generic(term: str) -> bool:
    words = T.words(term)
    return bool(words) and all(w in GENERIC_NOUNS or w in T.STOPWORDS for w in words)
_DISFLUENT = re.compile(r"\b(?:a|an|the|of|to|in|and|or|with|for)\s*,|,\s+(?:this|that)\s+(?:they|it|we|is|are|will)\b|\b(?:as\s+like|like\s+like|so\s+so)\b|\bthe\s+(?:a|an)\b|"
                        r"\b(?:the|a|an)\s+(?:of|to|in|and|or|is|are)\b|,\s+(?:and|but)\s+(?:they|it|this|these|those)\s+(?:all\s+)?(?:will|are|is|can|have|has)\b|,\s+then\s+", re.IGNORECASE)
_HEDGE = re.compile(r"\b(?:somehow|actually|basically|obviously|simply|kind\s+of|sort\s+of|story|stuff|whatever|anyway|maybe|probably|i\s+guess)\b", re.IGNORECASE)
_HERE = re.compile(r"\b(?:here|over\s+here|right\s+now)\b", re.IGNORECASE)
_BACK_REFERENCE = re.compile(r"^the\s+(?:next|first|second|third|last|other|following|same|rest|previous)\s+(?:one\s+|part\s+)?(?:is|are|was|were)\b", re.IGNORECASE)
_NOUN_LIKE_VERBS = frozenset("form forms fold folds function functions plate plates signal signals line lines layer layers pump pumps stage stages mark marks cell cells stain "
                             "stains bud buds fuse fusion release releases transport transports store stores support supports".split())
_DET_OR_PREP = frozenset("the a an some all both each this that these those its their his her our in on at of to from with by for into over under between".split())
_CONNECTORS = re.compile(r"\b(?:and|but|because|which|that|so|then|although|while|whereas|when|if)\b", re.IGNORECASE)
_OPEN_QUOTE_PAIRS = (("(", ")"), ("[", "]"), ('"', '"'))
_TERM_LIKE = re.compile(r"[A-Za-z]{4,}")

_BASE_VERBS = """
accumulate act activate adapt add affect aggregate allow alter appear arise arrange ascend associate attach attract bind block break bring build
carry cause change clear close come communicate compare complete compose condense conduct connect consist contain continue contract contribute
control convert coordinate cover create cross cut decrease deliver depend depress derive descend describe detect determine develop differ
differentiate diffuse disappear divide drain drive eliminate enable end enter equal establish evaginate exchange exhibit exist expand
express extend fail fall fill find finish fold follow form function fuse gain generate give go grow guide happen have help hold identify
include increase induce influence inhibit initiate insert interact interfere interpret invade involve join keep lack lead leave limit line
link localize locate lie lose maintain make mean measure mediate migrate mix modify move occupy occur open operate organize originate
oxidize pass penetrate perform permit phosphorylate play position prevent proceed produce project promote propagate protect protrude provide
proliferate pump reabsorb reach receive recognize recruit reduce refer reflect regenerate regulate release remain remove repair replace
represent require reside respond restore result retain return reverse rise rotate secrete see seem send separate serve show signal sit
split spread start stay stimulate stop store supply support suppress surround synthesize take target terminate transmit transport trigger
turn undergo unite use vary wrap yield become begin call name contain constitute comprise enclose engulf phagocytose degrade digest sense
sever survive tolerate unfold appear acquire adopt allow rely run set sort store tend transfer translate transform transition need interconvert
hydroxylate catalyze catalyse cleave hydrolyze hydrolyse esterify deaminate carboxylate decarboxylate methylate acetylate glycosylate ubiquitinate
transcribe replicate dephosphorylate amplify bleed degenerate dilate constrict relax shorten lengthen elevate lower raise tighten loosen
mineralize calcify ossify innervate vascularize perfuse empty emerge exit extrude invaginate sprout bud pinch assemble disassemble
polymerize depolymerize anchor adhere detach glide slide shed renew replenish
""".split()
_IRREGULAR = {"be": "is are was were been being am", "have": "has had having", "do": "does did doing done", "go": "goes went gone going",
              "give": "gives gave given giving", "make": "makes made making", "take": "takes took taken taking", "see": "sees saw seen",
              "find": "finds found", "lie": "lies lay lain lying", "lead": "leads led", "hold": "holds held", "keep": "keeps kept",
              "begin": "begins began begun", "become": "becomes became", "fall": "falls fell fallen", "grow": "grows grew grown",
              "show": "shows showed shown", "mean": "means meant", "send": "sends sent", "set": "sets", "run": "runs ran running",
              "rise": "rises rose risen rising", "arise": "arises arose arisen arising", "come": "comes came coming", "split": "splits splitting",
              "spread": "spreads", "undergo": "undergoes underwent undergone", "bring": "brings brought", "lose": "loses lost losing",
              "cut": "cuts cutting", "sit": "sits sat sitting", "build": "builds built", "bind": "binds bound", "can": "could may might must shall should will would"}


def _inflect(base: str) -> set[str]:
    forms = {base, base + "s", base + "ed", base + "ing", base + "es"}
    if base.endswith("e"):
        forms |= {base + "d", base[:-1] + "ing"}
    if base.endswith("y"):
        forms |= {base[:-1] + "ies", base[:-1] + "ied"}
    if base.endswith(("ate", "ize", "ise")):
        forms.add(base[:-1] + "ing")
    return forms


VERB_FORMS: frozenset[str] = frozenset(f for v in _BASE_VERBS for f in _inflect(v)) | frozenset(
    w for v in _IRREGULAR.values() for w in v.split()) | frozenset(_IRREGULAR)
_AUX = frozenset("is are was were be been being am has have had do does did can could will would may might must should shall".split())
_DET_START = re.compile(r"^(?:the|a|an|this|that|these|those|each|every|all|both|some|any|no|its|their|his|her|our|your|my|one|two|three|four|five)\s+",
                        re.IGNORECASE)


@dataclass
class Cleaned:
    text: str                                    # the study sentence ("" when nothing usable is left)
    emphasis: bool = False                       # the lecturer stressed it ("you have to know ...")
    notes: list[str] = field(default_factory=list)


def clean_sentence(raw: str) -> Cleaned:
    """A spoken sentence made fit to read: fillers, teacher phrases, stutters, ``epithelium. epithelium.`` and a ragged start removed."""
    text = T.clean(raw)
    notes: list[str] = []
    text = re.sub(r"\((?:ok|okay|yes|right|no)\)", "", text, flags=re.IGNORECASE)
    emphasis = False
    if _TEACHER.search(text):
        emphasis = True
        text = _TEACHER.sub("", text, count=3)
        notes.append("teacher phrase removed")
    text = _FILLER_ANY.sub("", text)
    text = _CONTENT_ECHO.sub(r"\1", text)
    text = _FILLER_START.sub("", text).strip(" ,;-–")
    text = _TRAILING_ECHO.sub(lambda m: m.group(1) + ".", text)
    try:                                         # repeated words / phrases ("for for the for the", "formation, formation")
        from ..medical.repair import collapse_repeats

        collapsed, n = collapse_repeats(text)
        if n:
            text = collapsed
            notes.append(f"{n} repeated word(s) removed")
    except Exception:                            # pragma: no cover - the repair module is optional
        pass
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", _MULTI_SPACE.sub(" ", text)).strip(" ,;-–")
    text = re.sub(r"([.!?])\s*[.!?]+$", r"\1", text)
    if not text:
        return Cleaned("", emphasis, notes)
    text = text[0].upper() + text[1:]
    if not text.endswith((".", "!", "?", "…")):
        text += "."
    return Cleaned(text, emphasis, notes)


# ================================================================================================================ validation
@dataclass
class Check:
    ok: bool
    issues: list[str] = field(default_factory=list)        # hard problems: the sentence is not used
    soft: list[str] = field(default_factory=list)          # doubts that lower the confidence

    @property
    def confidence(self) -> float:
        return 0.0 if not self.ok else max(0.3, 1.0 - 0.12 * len(self.soft))


def _has_verb(words: Sequence[str]) -> bool:
    for i, w in enumerate(words):
        if w in _AUX:
            return True
        if w in VERB_FORMS:
            if w in _NOUN_LIKE_VERBS and i > 0 and (words[i - 1] in _DET_OR_PREP or words[i - 1] in ("one", "two", "several", "many", "main", "other")):
                continue                                                       # "the functions of", "one of the folds": a noun here
            return True
    return False


_ADJ_WITHOUT_NOUN = re.compile(r"\bthe\s+([a-z]+(?:ar|al|ic|ous))\s+(?:and|or)\s+(?:the\s+)?([a-z]+)\b", re.IGNORECASE)
_ADJ_END = re.compile(r"(?:ar|al|ic|ous|ive|ary|ior|ial|ent|ant)$")
_COORD_SUBJECT = re.compile(r",\s*(?:and\s+)?(?:the\s+)?[\w-]+(?:\s+[\w-]+)?\s*,?\s+and\s+|\b[\w-]+\s+and\s+(?:the\s+)?[\w-]+", re.IGNORECASE)


#: nouns with an adjective-like ending ("the central canal", "the action potential", "the axon terminal")
_NOUNS_LIKE_ADJECTIVES = frozenset("""canal canals signal signals terminal terminals animal animals potential potentials interval intervals material materials hospital capital total
trial trials vial metal portal arrival removal survival approval journal clinic topic logic music panic mimic tunic pillar collar sugar sugars molar molars cellar nodular""".split())
_DANGLING_ADJ = re.compile(r"\bthe\s+([a-z]+(?:ar|al|ic|ous))\s*[.!?]?$", re.IGNORECASE)


def adjective_without_noun(body: str) -> bool:
    """"the molecular and the cell membrane", "... will reorganize the molecular.": an adjective with no noun of its own - a garbled phrase. "the central and the peripheral
    nervous system" (two adjectives sharing one noun) and "the central canal" (a noun with that ending) are fine."""
    for m in _ADJ_WITHOUT_NOUN.finditer(body):
        if not _ADJ_END.search(m.group(2).casefold()) and m.group(1).casefold() not in _NOUNS_LIKE_ADJECTIVES:
            return True
    end = _DANGLING_ADJ.search(body.strip())
    return bool(end) and end.group(1).casefold() not in _NOUNS_LIKE_ADJECTIVES


def subject_verb_disagree(words: Sequence[str], body: str) -> bool:
    """A coordinated subject ("The dorsal horn, lateral horn, and the ventral horn ...", "X and Y ...") with a singular verb ("develops")."""
    end = _subject_end(list(body.split()))
    if end <= 1:
        return False
    head = " ".join(body.split()[:end])
    verb = body.split()[end].casefold().strip(",.")
    if not _COORD_SUBJECT.search(head) or re.match(r"^(?:in|on|at|during|after|before|within|between|among)\b", head, re.IGNORECASE):
        return False
    return verb in ("is", "was", "has", "does") or (verb in VERB_FORMS and verb.endswith("s") and not verb.endswith("ss") and verb not in _AUX)


_SINGULAR_S = frozenset("lens series species means gas bias corpus plexus virus nucleus basis axis thesis process access abscess mass atlas pancreas iris pelvis testis".split())


def article_number_clash(words: Sequence[str]) -> bool:
    """"A dendritic spines ...": an indefinite article in front of a plural noun (two words at most between them)."""
    for i, w in enumerate(words[:-1]):
        if w not in ("a", "an"):
            continue
        for j in range(i + 1, min(i + 3, len(words))):
            noun = words[j]
            if noun in T.STOPWORDS or noun in VERB_FORMS and j == i + 1:
                break
            if noun.endswith(("es", "s")) and not noun.endswith(("ss", "us", "is", "ous", "ics")) and noun not in _SINGULAR_S and len(noun) > 4:
                nxt = words[j + 1] if j + 1 < len(words) else ""
                if nxt not in ("of", "and", "or"):                                     # "a means of", "a series of": the noun phrase continues
                    return True
            if not noun.endswith(("al", "ar", "ic", "ous", "ive", "ary", "ed", "ing", "ate")):
                break                                                                 # the noun came and was singular
    return False


def no_main_verb(words: Sequence[str]) -> bool:
    """"The cells which make up the cortex and the cortical lamination.": every verb stands in a relative clause - there is a subject and no statement about it."""
    finite = [i for i, w in enumerate(words) if w in _AUX or (w in VERB_FORMS and w not in _NOUN_LIKE_VERBS)]
    if not finite:
        return False
    relative = [i for i, w in enumerate(words) if w in _RELATIVE_START and i > 0]
    if not relative:
        return False
    first_rel = relative[0]
    if first_rel + 1 < len(words) and words[first_rel + 1] in VERB_FORMS and first_rel + 1 not in finite:
        finite = sorted(finite + [first_rel + 1])                                             # "that LINE the canal": a noun-like verb right after the pronoun is its verb
    groups = [i for k, i in enumerate(finite) if k == 0 or finite[k - 1] != i - 1]                 # "are going to send": one verb group
    # every verb after the relative pronoun, and only one verb group there: the relative clause's own verb ("The neurons that line the canal DIVIDE ..." has a second one)
    return all(i > first_rel for i in finite) and len(groups) == 1 and not any(w in ("is", "are", "was", "were") for w in words[:first_rel])


def subsumed(a: str, b: str, threshold: float = 0.8) -> bool:
    """The shorter statement says nothing the longer one does not ("The intermediate zone cells contain the precursors for the neurons" inside "The intermediate zone contains the
    precursors for the glia and neurons ..."): a summary keeps one of them."""
    sa, sb = {T.stem(w) for w in T.content_words(a)}, {T.stem(w) for w in T.content_words(b)}
    small, large = (sa, sb) if len(sa) <= len(sb) else (sb, sa)
    return len(small) >= 4 and len(small & large) / len(small) >= threshold


def _coordinated(words: Sequence[str], i: int, j: int) -> bool:
    """The same word twice in a coordination - "the central nervous system and the peripheral nervous system", "inner layer or outer layer": two different modifiers
    before it and an "and" / "or" between the two places. A stutter ("formation, formation", "the cells the cells") has neither."""
    between = words[i + 1:j]
    if not any(w in ("and", "or", "nor", "versus", "vs") for w in between):
        return False
    phrase_i = [w for w in words[max(0, i - 2):i] if w not in T.STOPWORDS]
    phrase_j = [w for w in words[max(0, j - 2):j] if w not in T.STOPWORDS and w not in ("and", "or", "nor")]
    return bool(phrase_i) and bool(phrase_j) and phrase_i != phrase_j            # "central nervous system" and "peripheral nervous system": the noun phrases differ


def sentence_issues(text: str, *, unknown: Optional[Callable[[str], bool]] = None, unresolved: Iterable[str] = (),
                    max_words: int = 42, min_words: int = 6) -> Check:
    """Is this cleaned sentence a usable, standalone statement? ``unknown(word)`` tells whether a word is not a known English / medical word."""
    issues: list[str] = []
    soft: list[str] = []
    words = T.words(text)
    n = len(words)
    body = text.rstrip(".!?… ")
    if n < min_words:
        issues.append("too short")
    if n > max_words:
        issues.append("too long")
    if text.endswith("?") or re.match(r"^(?:why|what|how|where|when|who|which|is|are|do|does|did|can|could|should|would)\b", text, re.IGNORECASE):
        issues.append("question")
    if _META.search(body) or _META_ANY.search(body):
        issues.append("lecturer talk")
    if _FIRST_PERSON.search(body):
        issues.append("speaker's voice")
    if _SUBORDINATE_FRAGMENT.match(body) and "," not in body:
        issues.append("dependent clause only")
    if _BACK_REFERENCE.match(body):
        issues.append("refers to the previous sentence")
    if _PREP_NO_SUBJECT.match(body):
        issues.append("missing subject")
    if _PRONOUN_AFTER_INTRO.match(body):
        issues.append("pronoun subject after an introductory phrase")
    if _DISFLUENT.search(body):
        issues.append("broken grammar")
    if _HERE.search(body):
        issues.append("refers to the picture")
    if _ONLY_MATTERS.match(body):
        issues.append("says only that it matters")
    if no_main_verb(words):
        issues.append("no main verb")
    if article_number_clash(words):
        issues.append("broken grammar")
    if re.search(r",\s+(?:or\s+|and\s+)?(?:it's|it\s+is|they're|they\s+are)\s+(?:also\s+)?called\b", body, re.IGNORECASE):
        issues.append("two statements run together")                             # "X is made in the ornithine cycle, or it's called also urea cycle, ..."
    if re.search(r",\s+right[.!?]?$", text.strip(), re.IGNORECASE):
        issues.append("spoken filler")
    if _TO_FRAGMENT.match(body):
        issues.append("dependent clause only")
    if _HEDGE.search(body):
        issues.append("spoken filler")
    content = [(i, w) for i, w in enumerate(words) if len(w) >= 5 and w not in T.STOPWORDS]
    if any(w == w2 and j - i <= 5 and not _coordinated(words, i, j) for k, (i, w) in enumerate(content) for (j, w2) in content[k + 1:k + 4]):
        issues.append("repeated words")
    if len(words) >= 2 and words[0] in VERB_FORMS and words[0] not in _NOUN_LIKE_VERBS and words[0] not in _AUX and words[1] in _DET_OR_PREP:
        issues.append("no subject")
    if len(words) >= 3 and words[0] in ("the", "a", "an") and words[1] in VERB_FORMS and words[1].endswith("s") and words[1] not in _NOUN_LIKE_VERBS \
            and words[2] in ("the", "a", "an", "this", "these", "its", "their"):
        issues.append("no subject")                                               # "The gives the name of the group ...": a word is missing
    grams = Counter(tuple(words[i:i + 3]) for i in range(max(0, n - 2)))
    if any(c >= 2 for g, c in grams.items() if not all(w in T.STOPWORDS for w in g)):
        issues.append("repeated phrase")
    if _DEICTIC.search(body):
        issues.append("refers to the slide or picture")
    if _PRONOUN_SUBJECT.match(body) or _PRONOUN_CONTRACTION.match(body) or _PRONOUN_ANY.match(body):
        issues.append("starts with a pronoun")
    if _AGREEMENT.search(body) or adjective_without_noun(body):
        issues.append("broken grammar")
    if subject_verb_disagree(words, body):
        issues.append("subject and verb disagree")
    if _DANGLING_END.search(body):
        issues.append("unfinished sentence")
    if not _has_verb([w for w in words]):
        issues.append("no verb")
    if re.search(r"\b([\w-]{3,})\b(?:[\s,]+)\1\b", body, re.IGNORECASE):
        issues.append("repeated words")
    for left, right in _OPEN_QUOTE_PAIRS:
        if body.count(left) != body.count(right) and left != right:
            issues.append("unbalanced brackets")
    if sum(1 for c in body if c.isdigit()) > 0 and re.search(r"\b\d+\s*(?:to|-)\s*\d+\s*(?:of|out)\b", body):
        soft.append("odd number")
    aux = [i for i, w in enumerate(words) if w in _AUX and w not in ("be", "been", "being")]       # finite forms: they open a clause
    if len(aux) >= 3 or (len(aux) == 2 and n > 16 and not _CONNECTORS.search(" ".join(words[aux[0] + 1:aux[1]])) and "," not in body):
        issues.append("two statements run together")
    if len(_CONNECTORS.findall(body)) >= 4 or body.count(",") >= 5:
        issues.append("several statements in one sentence")
    elif len(_CONNECTORS.findall(body)) == 3 or body.count(",") == 4:
        soft.append("long sentence")
    if re.search(r"\b(?:okay|ok|yeah|um+|uh+|kind of|sort of|you know)\b", body, re.IGNORECASE):
        issues.append("spoken filler")
    low = body.casefold()
    bad = [u for u in (x.casefold() for x in unresolved) if u and u in low]
    if bad:
        issues.append(f"unresolved term: {bad[0]}")
    if unknown is not None:
        odd = [w for w in words if len(w) >= 5 and unknown(w)]
        if odd:                                                                           # a word nobody knows is a recogniser error until proven otherwise
            issues.append(f"unrecognised words: {', '.join(odd[:3])}")
    return Check(not issues, issues, soft)


# ================================================================================================================ labels
_GENERIC_WORDS = frozenset("""
cranial caudal abnormal normal early late first second third fourth fifth next last later different same other another many more most several
various certain general main major minor important special specific particular central lateral medial dorsal ventral upper lower inner outer
total whole entire early stage point step part parts type types kind form forms case cases example examples thing things way ways time times
week weeks day days month months year years process level levels number amount case area areas region regions side sides group groups
""".split())
_GENERIC_LABELS = re.compile(r"^(?:abnormal|normal|early|late|first|second|third|fourth|fifth|next|last|later|different|same|other|many|more|several|"
                             r"various|certain|general|main|major|special|specific|important|cranial|caudal)\b", re.IGNORECASE)


def is_generic_label(term: str, known: Optional[Callable[[str], bool]] = None) -> bool:
    """"cranial caudal", "abnormal neural tube", "third week", "give rise": a label that names no medical concept. ``known(term)`` is True for
    terms of the medical vocabulary, the slides or the lecture glossary (those are never generic)."""
    if known is not None and known(term):
        return False
    words = T.words(term)
    if not words:
        return True
    if all(w in _GENERIC_WORDS or w in T.STOPWORDS or w in VERB_FORMS for w in words):
        return True
    if _GENERIC_LABELS.match(term.strip()) and len(words) <= 3:
        return True
    if any(w in VERB_FORMS and w not in ("function", "form", "plate", "fold", "cell", "signal", "mark", "line", "pump", "stain", "stage") for w in words[:1]):
        return True
    return False


# ================================================================================================================ typing of facts
_REL_LOCATION = re.compile(r"\b(?:is|are)\s+(?:located|found|situated|sitting|seen|present|positioned)\b|\b(?:lies|lie|sits|sit|surrounds?|lines?|"
                           r"covers?|occupies|occupy)\b|\b(?:innermost|outermost|next to|adjacent to|beneath|surrounding)\b", re.IGNORECASE)
_REL_DERIVATIVE = re.compile(r"\b(?:gives?|give)\s+rise\s+to\b|\bdifferentiat\w+\s+(?:into|from|to)\b|\bdevelop\w*\s+(?:into|from)\b|\bderived?\s+from\b|"
                             r"\barises?\s+from\b|\boriginat\w+\s+from\b|\bbecomes?\b|\bturns?\s+into\b|\bforms?\s+(?:the|a|an|all)\b|\bgenerates?\b",
                             re.IGNORECASE)
_REL_CLASSIFICATION = re.compile(r"\b(?:is|are)\s+(?:a|an|one\s+of\s+the|the)\s+(?:type|kind|form|class|group|subtype|category|division|"
                                 r"subdivision|layer|component|part)s?\s+of\b|\bdivided?\s+into\b|\bsubdivid\w+\s+into\b|\b(?:two|three|four|five|six)\s+"
                                 r"(?:types|kinds|forms|groups|layers|zones|parts|divisions|regions|vesicles|stages)\b|\bsplits?\s+into\b",
                                 re.IGNORECASE)
_REL_FUNCTION = re.compile(r"\b(?:releases?|secretes?|produces?|induces?|initiates?|stimulates?|activates?|inhibits?|regulates?|controls?|allows?|"
                           r"enables?|provides?|supplies|supply|transmits?|carries|carry|supports?|protects?|responsible\s+for|important\s+for|"
                           r"required\s+for|needed\s+for|involved\s+in|contributes?\s+to|plays?\s+a\s+role|serves?\s+as|acts?\s+as|function)\b",
                           re.IGNORECASE)
_REL_DEVELOPMENT = re.compile(r"\b(?:fuses?|fuse|closure|closes?|folds?|proliferat\w+|migrat\w+|evaginat\w+|depress\w*|invaginat\w+|"
                              r"condens\w+|differentiat\w+|elevat\w+|forms?\b)", re.IGNORECASE)


def classify_statement(text: str, base_kind: str = "concept") -> str:
    """The :data:`FACT_TYPES` entry for a statement, from the kind the analysis found and the relation the sentence expresses."""
    mapping = {"definition": "definition", "cause_effect": "cause_effect", "mechanism": "mechanism", "clinical": "clinical_relevance",
               "comparison": "comparison", "structure": "structure_function", "pathway": "sequence", "enzyme": "function", "drug": "function",
               "gene": "function"}
    if base_kind in mapping and base_kind not in ("enzyme", "drug", "gene"):
        return mapping[base_kind]
    for pattern, kind in ((_REL_DERIVATIVE, "derivative"), (_REL_LOCATION, "location"), (_REL_CLASSIFICATION, "classification"),
                          (_REL_FUNCTION, "function"), (_REL_DEVELOPMENT, "development")):
        if pattern.search(text):
            return kind
    return mapping.get(base_kind, "relationship")


@dataclass
class StudyFact:
    fact_id: str
    statement: str                               # one clean, grammatical sentence in the lecture's own words
    topic: str = ""
    subtopic: str = ""                           # the main term the statement is about
    fact_type: str = "relationship"
    importance: float = 0.0
    difficulty: str = "medium"
    source_type: str = "transcript"              # transcript | pdf
    source_start: Optional[float] = None
    source_end: Optional[float] = None
    source_page: Optional[int] = None
    source_excerpt: str = ""                     # the sentence as it stands in the source, before cleaning
    confidence: float = 1.0
    medical_terms: list[str] = field(default_factory=list)
    teacher_emphasis: bool = False
    uncertainty: float = 0.0                     # 0 = nothing doubtful, 1 = everything
    related_facts: list[str] = field(default_factory=list)
    chunk_id: str = ""
    position: int = -1                           # index of the sentence unit in the lecture
    subject: str = ""
    object: str = ""
    tier: str = "HIGH"                           # HIGH: directly supported; MEDIUM: supported, but rescued or context-dependent (recall.py); LOW facts are never kept
    doubts: list[str] = field(default_factory=list)   # why it is MEDIUM ("lead-in removed", "points back to an earlier sentence" ...)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def fact_id_of(statement: str) -> str:
    from .models import item_id

    return item_id("f", statement)


def stem_key(term: str) -> str:
    """"Neural fold" / "neural folds" / "Neural-fold" share one key (a plural or inflection is the same concept)."""
    return " ".join(T.stem(w) for w in T.words(term))


def near_duplicate(a: str, b: str, threshold: float = 0.78) -> bool:
    sa, sb = {T.stem(w) for w in T.content_words(a)}, {T.stem(w) for w in T.content_words(b)}
    if not sa or not sb:
        return False
    return len(sa & sb) / min(len(sa), len(sb)) >= threshold and abs(len(sa) - len(sb)) <= max(2, len(sa) // 2)


def link_related(facts: list[StudyFact], limit: int = 4) -> None:
    """Cheap relation graph: facts of the same topic that share a medical term (used for distractors and grouping)."""
    by_term: dict[str, list[StudyFact]] = {}
    for fact in facts:
        for term in fact.medical_terms:
            by_term.setdefault(stem_key(term), []).append(fact)
    for fact in facts:
        scored: dict[str, int] = {}
        for term in fact.medical_terms:
            for other in by_term.get(stem_key(term), ()):
                if other.fact_id != fact.fact_id:
                    scored[other.fact_id] = scored.get(other.fact_id, 0) + 1
        fact.related_facts = [fid for fid, _n in sorted(scored.items(), key=lambda x: -x[1])[:limit]]


# ================================================================================================================ restatements and atoms
_RESTATE = re.compile(r"^((?:the|a|an|this|these|those|each|all)\s+[\w\s'-]{3,60}?),\s+(?:it|they)\s+(?=(?:is|are|was|were|will|can|has|have|do|does)\b|"
                      r"[a-z]+(?:s|es)\b)", re.IGNORECASE)
_SUBORDINATE_START = re.compile(r"^(?:because|when|while|if|once|after|before|since|although|as|whereas|unless|until)\b", re.IGNORECASE)
_PRONOUN_WORD = re.compile(r"^(?:it|they|he|she|this|that|these|those|there|which|who)$", re.IGNORECASE)


_RESTATE_AFTER_CLAUSE = re.compile(r"^(?P<subj>(?:the|a|an)\s+[\w\s-]{3,50}?),\s+(?P<sub>(?:as|when|while|if|once)\s+(?:they|it)\s+[\w\s-]{2,25}?),\s+(?:they|it)\s+(?P<rest>.+)$",
                                   re.IGNORECASE)
_DEMONSTRATIVE_START = re.compile(r"^(?:this|these|those|that)\s+(?=(?!is\b|are\b|was\b|were\b|will\b|can\b|has\b|have\b|means\b|may\b)[a-z-]+\s+[a-z-]+)", re.IGNORECASE)


def demonstrative_to_article(text: str) -> str:
    """"This intermediate zone contains ..." -> "The intermediate zone contains ...": a statement that stands alone does not point back at a previous sentence."""
    return _DEMONSTRATIVE_START.sub("The ", text, count=1)


_RESTATE_LONG = re.compile(r"^(?P<head>[A-Z][\w\s'-]{10,110}?),\s+(?:it|they)\s+(?P<rest>(?:is|are|was|were|will|can|has|have)\b.+)$")


def _finite(words: list[str]) -> bool:
    return any(w.casefold() in _AUX or (w.casefold() in VERB_FORMS and w.casefold().endswith("s") and not w.casefold().endswith("ss")) for w in words)


def fix_restatement(text: str) -> str:
    """Spoken sentences restate their subject: "These radial glial cells, they are a guide" -> "These radial glial cells are a guide"."""
    m2 = _RESTATE_AFTER_CLAUSE.match(text.rstrip(".!? "))
    if m2:
        return f"{m2.group('subj')}, {m2.group('sub')}, {m2.group('rest')}."
    m3 = _RESTATE_LONG.match(text.rstrip(".!? "))
    if m3 and not _finite(m3.group("head").split()) and not _SUBORDINATE_START.match(m3.group("head")):
        return f"{m3.group('head')} {m3.group('rest')}."
    m = _RESTATE.match(text)
    if not m:
        return text
    head = m.group(1).strip()
    if _SUBORDINATE_START.match(head):
        return text
    return head + " " + text[m.end():]


def _subject_end(words: list[str]) -> int:
    """Index of the first verb / auxiliary: the words before it are the subject."""
    for i, w in enumerate(words):
        if i > 0 and (w.casefold() in _AUX or w.casefold() in VERB_FORMS):
            return i
    return -1


def split_atomic(text: str) -> list[str]:
    """One fact = one proposition. A sentence with two predicates of the same subject ("X is the innermost layer and contains the stem cells") or
    two independent clauses ("...; ...") becomes two sentences; a sentence that cannot be split safely stays as it is (and may then be rejected)."""
    body = text.rstrip(".!? ")
    parts = [p.strip() for p in re.split(r";\s+", body) if p.strip()]
    out: list[str] = []
    for part in parts:
        words = part.split()
        cut = re.search(r",?\s+(?:and|but)\s+(?:then\s+)?(?=(?:[a-z]+(?:s|es)\b|is\b|are\b|was\b|were\b|has\b|have\b|can\b|will\b|may\b|"
                        r"must\b|does\b|do\b))", part)
        if cut and cut.start() > 0:
            left, right = part[:cut.start()].strip(" ,"), part[cut.end():].strip()
            first = right.split()[0].casefold() if right.split() else ""
            end = _subject_end(left.split())
            if end > 0 and (first in VERB_FORMS or first in _AUX) and len(right.split()) >= 3 and len(left.split()) >= 5:
                subject = " ".join(left.split()[:end])
                if not _PRONOUN_WORD.match(subject.split()[0]) or len(subject.split()) > 1:
                    out.append(left)
                    out.append(subject + " " + right)
                    continue
        out.append(part)
    return [fix_restatement(clean_sentence(p).text) or p for p in out]


# ================================================================================================================ word knowledge
_KB_CACHE: dict = {}


def default_knowledge():
    """The v4 medical knowledge base (read-only), when the installation has one; None otherwise. Used only to recognise words, never to add facts."""
    if "kb" not in _KB_CACHE:
        kb = None
        try:
            from ..medical_knowledge.runtime.knowledge import open_knowledge

            kb = open_knowledge()
        except Exception:                                      # noqa: BLE001 - the knowledge base is optional for the study side
            kb = None
        _KB_CACHE["kb"] = kb
    return _KB_CACHE["kb"]


class WordKnowledge:
    """Which words are real (English, medical vocabulary, the slides, the glossary, or said so often that they are the lecturer's own term)."""

    def __init__(self, extra: Iterable[str] = (), counts: Optional[dict[str, int]] = None, frequent: int = 5, knowledge="default") -> None:
        self._kb = default_knowledge() if knowledge == "default" else knowledge
        from ..common_words import common_words

        self.common = common_words()
        self.extra = {w.casefold() for term in extra for w in T.words(term)}
        self.counts = counts or {}
        self.frequent = frequent
        try:
            from ..medical.lexicon import shared_lexicon

            self._index = shared_lexicon().subset(None)
        except Exception:                                      # pragma: no cover - the vocabulary folder is optional for the study side
            self._index = None
        self._cache: dict[str, bool] = {}

    def known(self, word: str) -> bool:
        w = word.casefold().strip("-'’")
        if w in self._cache:
            return self._cache[w]
        ok = (w in self.common or w in self.extra or w in T.STOPWORDS or any(c.isdigit() for c in w) or len(w) <= 3
              or T.stem(w) in self.common or w.rstrip("s") in self.common or w in VERB_FORMS or self.counts.get(w, 0) >= self.frequent)
        if not ok and self._index is not None:
            ok = bool(self._index.exact(w)) or w in self._index.known or T.stem(w) in self._index.known
        if not ok and self._kb is not None:
            try:
                ok = self._kb.token_frequency(w) > 0 or self._kb.is_english_word(w, 80)
            except Exception:                                  # noqa: BLE001
                ok = False
        if not ok and "-" in w:
            ok = all(self.known(part) for part in w.split("-") if part)
        self._cache[w] = ok
        return ok

    def unknown(self, word: str) -> bool:
        return not self.known(word)
