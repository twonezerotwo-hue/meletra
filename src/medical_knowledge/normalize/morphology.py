"""Medical word morphology: Greek / Latin building blocks and Latin number forms.

Used for three things, always to HELP retrieval and to PROTECT valid words - never to invent terms:

    decompose(word)       a word that is built entirely from known prefixes / roots / suffixes ("visceromotor", "fibrocytes") is a PLAUSIBLE real term, so it needs
                          stronger evidence before it may be called a mishearing
    variants(word)        singular / plural and Latin case forms (perikaryon <-> perikarya, nucleus <-> nuclei, vertebra <-> vertebrae) so exact lookup is
                          number-insensitive
    analyse(word)         what the ``morphology`` table stores: lemma, Latin case / number, Greek root, prefix, suffix
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

PREFIXES = ("a", "an", "ab", "ad", "ambi", "amphi", "ana", "ante", "anti", "apo", "auto", "bi", "brady", "cata", "circum", "co", "con", "contra", "de", "dia", "dis",
            "dys", "ec", "ecto", "en", "endo", "epi", "eso", "eu", "ex", "exo", "extra", "hemi", "hetero", "homo", "homeo", "hyper", "hypo", "ileo", "im", "in", "infra",
            "inter", "intra", "iso", "juxta", "macro", "mal", "mega", "meso", "meta", "micro", "mid", "mono", "multi", "neo", "non", "ob", "oligo", "pan", "para", "per",
            "peri", "poly", "post", "pre", "pro", "pseudo", "re", "retro", "semi", "sub", "super", "supra", "sym", "syn", "tachy", "trans", "tri", "ultra", "uni", "xeno")
ROOTS = ("acr", "aden", "adip", "adren", "aer", "alb", "algi", "alveol", "amnio", "amyl", "andr", "angi", "ankyl", "anter", "aort", "arachn", "arteri", "arthr", "articul",
         "ather", "audi", "axon", "bacteri", "bil", "bio", "blast", "brachi", "bronch", "bucc", "calc", "capill", "carb", "carcin", "cardi", "carp", "caud", "cav", "cele",
         "cephal", "cerebell", "cerebr", "cervic", "chem", "chol", "chondr", "chord", "chrom", "chromat", "chym", "cili", "circul", "clav", "coccy", "cochle", "col", "colp",
         "cor", "cort", "cost", "crani", "crypt", "cutane", "cyan", "cyst", "cyt", "dendr", "dent", "derm", "dextr", "didym", "digit", "dipl", "disc", "dors", "duct", "duoden",
         "dur", "ectas", "embol", "embry", "emes", "encephal", "enter", "epend", "ependym", "epitheli", "erythr", "esthes", "estr", "fasci", "femor", "fet", "fibr", "fibul",
         "flex", "follic", "gangli", "gastr", "gen", "genit", "gingiv", "glia", "glial", "gloss", "gluc", "glute", "glyc", "gnath", "gon", "gonad", "gran", "gyn", "gyr",
         "hal", "hem", "hemat", "hepat", "heter", "hist", "hidr", "humer", "hydr", "hymen", "hyster", "ile", "ili", "immun", "kary", "kerat", "kines", "kinet", "labi",
         "lacrim", "lact", "lamin", "laryng", "later", "leuk", "lien", "ligament", "lingu", "lip", "lith", "lob", "lumb", "lymph", "lys", "lyt", "malac", "mamm", "mandibul",
         "mast", "maxill", "medull", "melan", "mening", "menisc", "ment", "mes", "metr", "microglia", "mit", "morph", "mot", "motor", "muc", "muscul", "my", "myel", "myo",
         "narc", "nas", "nat", "necr", "nephr", "neur", "neuro", "nod", "nucle", "ocul", "odont", "olig", "onc", "oo", "ophthalm", "opt", "or", "orchi", "orth", "oste",
         "ot", "ovari", "ox", "palat", "pancreat", "papill", "patell", "path", "pector", "ped", "pelv", "pen", "peps", "pept", "perine", "phag", "phalang", "pharyng", "phleb",
         "phon", "phot", "phren", "pil", "pineal", "pituit", "plas", "pleur", "pne", "pneum", "pod", "poie", "poly", "pont", "port", "pos", "proct", "prostat", "pseud",
         "psych", "pub", "pulmon", "pyel", "py", "pyr", "quadr", "rach", "radi", "rect", "ren", "reticul", "retin", "rhin", "sacr", "salping", "sarc", "scap", "schiz",
         "scler", "scop", "sebace", "seps", "sept", "ser", "sial", "sigmoid", "sinus", "som", "somat", "son", "spasm", "sperm", "spher", "sphygm", "spin", "spir", "splen",
         "spondyl", "squam", "stapes", "stas", "stat", "sten", "stern", "stom", "strat", "stri", "synap", "tars", "tax", "tel", "tempor", "ten", "tend", "test", "thalam",
         "therm", "thorac", "thromb", "thym", "thyr", "tibi", "toc", "tom", "ton", "tonsill", "top", "tox", "trache", "trich", "trigon", "troph", "tuber", "tympan", "uln",
         "ur", "ureter", "urethr", "uter", "uvul", "vagin", "valv", "vas", "vascul", "ven", "ventr", "ventricul", "vers", "vertebr", "vesic", "vill", "viscer", "vit",
         "vitre", "zyg", "zym", "viscero", "sens", "sensor", "afferent", "efferent", "cortic", "limb", "lemn", "lemm", "schwann", "oligodendr", "astr", "astrocyt", "somat")
SUFFIXES = ("aemia", "algia", "al", "ar", "ary", "ase", "asis", "blast", "cele", "centesis", "cide", "clast", "cyte", "cytes", "cytic", "cytosis", "ectasia", "ectasis",
            "ectomy", "emesis", "emia", "esis", "form", "gen", "genesis", "genic", "gram", "graph", "graphy", "ia", "iasis", "iatric", "iatry", "ic", "ical", "ide", "ile",
            "in", "ine", "ing", "ion", "ism", "ist", "itis", "ium", "ize", "kinesis", "lepsy", "lith", "logy", "lysis", "lytic", "malacia", "mania", "megaly", "meter",
            "metry", "oid", "ol", "ologist", "ology", "oma", "omas", "omata", "one", "opia", "opsy", "orrhaphy", "ose", "osis", "osmia", "ostomy", "otic", "otomy", "ous",
            "pathy", "penia", "pexy", "phage", "phagia", "phagy", "phasia", "philia", "phobia", "phonia", "phoresis", "phrenia", "plasia", "plasm", "plasty", "plegia",
            "pnea", "poiesis", "ptosis", "ptysis", "rrhage", "rrhagia", "rrhaphy", "rrhea", "rrhexis", "scope", "scopy", "sis", "some", "spasm", "stasis", "stenosis",
            "stomy", "therapy", "thorax", "tic", "tome", "tomy", "tonia", "tripsy", "trophy", "tropic", "tropin", "um", "uria", "us", "y")
_LINKS = ("o", "i", "")
_ALL = {m for m in PREFIXES + ROOTS + SUFFIXES}
_BY_LEN = sorted(_ALL, key=len, reverse=True)


@lru_cache(maxsize=65536)
def decompose(word: str) -> tuple[float, tuple[str, ...]]:
    """(share of the word covered by known morphs, the morphs) for the best split; 1.0 = built entirely from known parts."""
    w = re.sub(r"[^a-z]", "", word.lower())
    n = len(w)
    if n < 5:
        return 0.0, ()
    best: dict[int, tuple[int, tuple[str, ...]]] = {0: (0, ())}                # position -> (covered letters, morphs)
    for i in range(n):
        if i not in best:
            continue
        covered, morphs = best[i]
        for m in _BY_LEN:
            if len(m) < 2 and not (m == "a" and i == 0):
                continue
            if w.startswith(m, i):
                j = i + len(m)
                cand = (covered + len(m), morphs + (m,))
                if j not in best or cand[0] > best[j][0] or (cand[0] == best[j][0] and len(cand[1]) < len(best[j][1])):
                    best[j] = cand
                for link in ("o", "i"):                                           # linking vowel: cardi-o-vascular
                    if w.startswith(link, j):
                        k = j + 1
                        cand2 = (covered + len(m) + 1, morphs + (m, link))
                        if k not in best or cand2[0] > best[k][0]:
                            best[k] = cand2
    full = best.get(n)
    if full:
        return round(full[0] / n, 3), full[1]
    top = max(best.values(), key=lambda v: v[0])
    return round(top[0] / n, 3), top[1]


def plausible_term(word: str, minimum: float = 1.0) -> bool:
    """True when the word is built entirely from known medical morphs (a real term the vocabulary may simply not list)."""
    share, morphs = decompose(word)
    return share >= minimum and len(morphs) >= 2


# ------------------------------------------------------------------ Latin / Greek number and case forms
_PAIRS = (("us", "i"), ("um", "a"), ("a", "ae"), ("on", "a"), ("ma", "mata"), ("is", "es"), ("ix", "ices"), ("ex", "ices"), ("nx", "nges"), ("x", "ces"),
          ("is", "ides"), ("y", "ies"), ("", "s"), ("", "es"), ("us", "era"), ("us", "ora"), ("en", "ina"), ("men", "mina"), ("ion", "ia"))


def variants(word: str) -> set[str]:
    """The word and its likely singular / plural forms (English and Latin / Greek). Used for number-insensitive lookup only."""
    w = word.lower()
    out = {w}
    for singular, plural in _PAIRS:
        if plural and w.endswith(plural) and len(w) > len(plural) + 2:
            out.add(w[: -len(plural)] + singular)
        if w.endswith(singular) and len(w) > len(singular) + 2:
            out.add(w[: len(w) - len(singular)] + plural)
    return out


@dataclass(frozen=True)
class Analysis:
    lemma: str
    latin_case: str
    latin_number: str
    greek_root: str
    prefix: str
    suffix: str


_NOMINATIVE_PLURAL = (("ae", "a"), ("i", "us"), ("a", "um"), ("ices", "ix"), ("nges", "nx"), ("mata", "ma"), ("era", "us"), ("ora", "us"))


def analyse(word: str) -> Analysis:
    w = re.sub(r"[^a-z]", "", word.lower())
    lemma, number = w, "singular"
    for plural, singular in _NOMINATIVE_PLURAL:
        if w.endswith(plural) and len(w) > len(plural) + 3:
            lemma, number = w[: -len(plural)] + singular, "plural"
            break
    _, morphs = decompose(w)
    prefix = next((m for m in morphs[:1] if m in PREFIXES and len(morphs) > 1), "")
    suffix = next((m for m in morphs[-1:] if m in SUFFIXES and len(morphs) > 1), "")
    root = next((m for m in morphs if m in ROOTS), "")
    return Analysis(lemma, "nominative", number, root, prefix, suffix)


# ------------------------------------------------------------------ related word forms (candidate RETRIEVAL only)
_ENDINGS = ("ization", "isation", "ation", "ition", "ically", "ical", "ized", "izes", "ize", "ised", "ise", "ated", "ates", "ate", "ating", "ative", "ing", "ed", "al", "ar", "ary",
            "ic", "ous", "ive", "ity", "ism", "ist", "ies", "ices", "mata", "ae", "es", "s", "y", "e", "a", "i", "us", "um", "on", "is", "ia", "ion", "en", "ix", "ex", "ma")


def related_forms(word: str) -> set[str]:
    """Words that share the stem of ``word`` and differ in the ending: plural / singular, Latin forms, adjectival ("tracheal" ~ "trachea"), verbal ("embolize" ~ "embolus", "emboli",
    "embolic"). Used ONLY to find candidates that a source actually has; a related form is not the same word and never replaces it automatically (the gate compares the grammatical
    role and the meaning)."""
    w = re.sub(r"[^a-z]", "", word.lower())
    if len(w) < 5:
        return set()
    stems = {w}
    for e in _ENDINGS:
        if w.endswith(e) and len(w) - len(e) >= 4:
            stems.add(w[: -len(e)])
    out = set(stems)
    for s in stems:
        for e in _ENDINGS:
            out.add(s + e)
    out.discard(w)
    return {x for x in out if len(x) >= 4}
