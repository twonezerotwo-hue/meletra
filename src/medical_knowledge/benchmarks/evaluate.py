"""Scoring of a correction engine against KNOWN text.

Three texts per clip are aligned: GOLD (what was spoken), RAW (what the recogniser wrote) and ENGINE (what the correction layer produced from RAW). Every
stretch where they differ is a REGION, and each region falls into exactly one class:

    corrected   RAW was wrong, ENGINE equals GOLD                     (a true positive)
    wrong       RAW was wrong, ENGINE changed it to something else    (a false correction that also misses)
    false       RAW was right, ENGINE changed it                      (the worst case: correct speech damaged)
    missed      RAW was wrong, ENGINE left it                         (a false negative)

Spacing-only differences ("brain stem" / "brainstem") are not errors. A region is MEDICAL when it overlaps an occurrence of one of the passage's medical terms.
Numbers are compared as words ("53" = "fifty three"). Nothing here is a human judgement: every figure is a count of regions.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def _number_words(n: int) -> list[str]:
    if n < 20:
        return [_ONES[n]]
    if n < 100:
        return [_TENS[n // 10]] + ([_ONES[n % 10]] if n % 10 else [])
    if n < 1000:
        return [_ONES[n // 100], "hundred"] + (_number_words(n % 100) if n % 100 else [])
    return [str(n)]


_ROMAN = {"ii": "two", "iii": "three", "iv": "four", "vi": "six", "vii": "seven", "viii": "eight"}


def words(text: str) -> list[str]:
    plain = unicodedata.normalize("NFKD", text.replace("\u2019", "'")).encode("ascii", "ignore").decode("ascii").lower().replace("-", " ")
    plain = re.sub(r"'s\b", "", plain).replace("'", "")                  # a possessive ("Wernicke's") is not a different term
    out: list[str] = []
    for w in re.findall(r"[a-z]+|\d+", plain):
        out.extend(_number_words(int(w)) if w.isdigit() else [_ROMAN.get(w, w)])
    return out


def compact(ws: Sequence[str]) -> str:
    return "".join(ws)


@dataclass
class Region:
    kind: str                      # corrected | wrong | false | missed
    medical: bool
    gold: str
    raw: str
    engine: str
    position: int = 0              # index of the first RAW word
    term: str = ""                 # the medical term of the passage this region overlaps ("" for an ordinary region)


@dataclass
class Evaluation:
    regions: list[Region] = field(default_factory=list)
    gold_words: int = 0
    raw_words: int = 0
    terms_total: int = 0
    terms_raw_ok: int = 0
    terms_engine_ok: int = 0
    term_details: list[dict] = field(default_factory=list)

    def count(self, kind: str, medical: Optional[bool] = None) -> int:
        return sum(1 for r in self.regions if r.kind == kind and (medical is None or r.medical == medical))

    def summary(self) -> dict:
        tp, wrong, false, missed = self.count("corrected"), self.count("wrong"), self.count("false"), self.count("missed")
        changes = tp + wrong + false
        med_errors = self.count("corrected", True) + self.count("wrong", True) + self.count("missed", True)
        return {"corrected": tp, "wrong": wrong, "false": false, "missed": missed, "changes": changes,
                "precision": None if changes == 0 else round(tp / changes, 4), "medical_error_regions": med_errors,
                "medical_recall": None if med_errors == 0 else round(self.count("corrected", True) / med_errors, 4),
                "false_per_1000_words": round(1000 * false / max(1, self.gold_words), 3), "terms_total": self.terms_total,
                "term_recall_raw": None if not self.terms_total else round(self.terms_raw_ok / self.terms_total, 4),
                "term_recall_engine": None if not self.terms_total else round(self.terms_engine_ok / self.terms_total, 4)}


def _bound(opcodes, pos: int, left: bool) -> int:
    best = None
    for tag, i1, i2, j1, j2 in opcodes:
        if i1 <= pos <= i2:
            one_to_one = tag == "equal" or (tag == "replace" and i2 - i1 == j2 - j1)                  # positions inside the block map word for word
            j = j1 + (pos - i1) if one_to_one else (j1 if pos == i1 else j2 if pos == i2 else None)
            if j is not None and (best is None or (j < best if left else j > best)):
                best = j
    return best if best is not None else 0


def _term_spans(gold: list[str], terms: Iterable[str]) -> list[tuple[int, int, str]]:
    spans = []
    for term in terms:
        t = words(term)
        if not t:
            continue
        for i in range(len(gold) - len(t) + 1):
            if gold[i:i + len(t)] == t:
                spans.append((i, i + len(t), term))
    return spans


def _intervals(opcodes) -> list[tuple[int, int]]:
    """Differing stretches of RAW. A replacement of n words by n words is n independent word substitutions (difflib joins them into one block only because they
    are adjacent), so each is its own interval; an uneven replacement stays one interval."""
    out = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            continue
        if tag == "replace" and i2 - i1 == j2 - j1 and i2 - i1 > 1:
            out.extend((i, i + 1) for i in range(i1, i2))
        else:
            out.append((i1, i2))
    return out


def evaluate(gold_text: str, raw_text: str, engine_text: str, terms: Iterable[str] = ()) -> Evaluation:
    g, r, e = words(gold_text), words(raw_text), words(engine_text)
    spans = _term_spans(g, terms)
    gr = difflib.SequenceMatcher(None, r, g, autojunk=False).get_opcodes()
    re_ = difflib.SequenceMatcher(None, r, e, autojunk=False).get_opcodes()
    # the recogniser's errors (GOLD vs RAW) are split word by word; the engine's changes (RAW vs ENGINE) stay whole: one correction is one unit
    intervals = sorted(_intervals(gr) + [(i1, i2) for tag, i1, i2, _, _ in re_ if tag != "equal"])
    merged: list[list[int]] = []
    for a, b in intervals:
        # intervals that only TOUCH are independent edits (a neighbouring recogniser error next to a corrected term must not turn the corrected term into a "wrong"
        # region); two edits at the same insertion point, or an insertion touching another edit, are one region
        if merged and (a < merged[-1][1] or (a == merged[-1][1] and (b == a or merged[-1][0] == merged[-1][1]))):
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    ev = Evaluation(gold_words=len(g), raw_words=len(r))
    for a, b in merged:
        ga, gb = _bound(gr, a, True), _bound(gr, b, False)
        ea, eb = _bound(re_, a, True), _bound(re_, b, False)
        gs, rs, es = g[ga:gb], r[a:b], e[ea:eb]
        raw_ok, same_as_raw, same_as_gold = compact(gs) == compact(rs), compact(es) == compact(rs), compact(es) == compact(gs)
        if raw_ok and same_as_raw:
            continue
        hit = next((name for ts, tb, name in spans if ga < tb and ts < gb or (ga == gb and ts <= ga <= tb)), "")
        medical = bool(hit)
        if raw_ok:
            kind = "false"
        elif same_as_raw:
            kind = "missed"
        elif same_as_gold:
            kind = "corrected"
        else:
            kind = "wrong"
        ev.regions.append(Region(kind, medical, " ".join(gs), " ".join(rs), " ".join(es), a, hit))
    # term recall: a term occurrence is right when its gold span lies inside an "equal" block of the alignment to the text
    ge = difflib.SequenceMatcher(None, g, e, autojunk=False).get_opcodes()
    gr2 = difflib.SequenceMatcher(None, g, r, autojunk=False).get_opcodes()

    def covered(opcodes, s: int, t: int) -> bool:
        return any(tag == "equal" and i1 <= s and t <= i2 for tag, i1, i2, _, _ in opcodes)

    for s, t, term in spans:
        raw_ok, eng_ok = covered(gr2, s, t), covered(ge, s, t)
        ev.terms_total += 1
        ev.terms_raw_ok += raw_ok
        ev.terms_engine_ok += eng_ok
        ev.term_details.append({"term": term, "raw_ok": raw_ok, "engine_ok": eng_ok})
    return ev


def merge(evals: Sequence[Evaluation]) -> Evaluation:
    total = Evaluation()
    for ev in evals:
        total.regions += ev.regions
        total.gold_words += ev.gold_words
        total.raw_words += ev.raw_words
        total.terms_total += ev.terms_total
        total.terms_raw_ok += ev.terms_raw_ok
        total.terms_engine_ok += ev.terms_engine_ok
        total.term_details += ev.term_details
    return total
