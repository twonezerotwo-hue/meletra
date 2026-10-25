"""Medical acronym engine and abbreviation disambiguation (reporting only - the transcript is never edited).

* learns acronyms from definitions in slides / speech:  ``cerebrospinal fluid (CSF)``  or  ``CSF (cerebrospinal fluid)``
* finds acronyms in the transcript, including spaced / dotted spellings ("C S F", "c.s.f.") that hint at ASR trouble
* scores the meaning of ambiguous abbreviations (MS, PE, ER, ...) from the words around each occurrence
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from .knowledge_acronyms import ACRONYMS, ACRONYMS_THAT_ARE_AMBIGUOUS, AMBIGUOUS
from .schema import Segment
from .vocabulary import normalize_key, tokenize

_STOP = {"of", "and", "the", "a", "an", "to", "in", "for", "or"}
_DEF_FORWARD = re.compile(r"((?:[A-Za-z][A-Za-z\-']*\s+){1,7}[A-Za-z][A-Za-z\-']*)\s*\(\s*([A-Za-z][A-Za-z0-9\-]{1,9})\s*\)")
_DEF_REVERSE = re.compile(r"\b([A-Za-z][A-Za-z0-9\-]{1,9})\s*\(\s*((?:[A-Za-z][A-Za-z\-']*\s+){0,7}[A-Za-z][A-Za-z\-']*)\s*\)")
_SPACED = re.compile(r"(?<![A-Za-z])((?:[A-Za-z]\.?[ \-]){1,5}[A-Za-z]\.?)(?![A-Za-z])")


def _letters(acronym: str) -> str:
    """Alphabetic letters of an acronym, lower-case (digits dropped: FADH2 -> fadh)."""
    return re.sub(r"[^a-z]", "", acronym.lower())


def looks_like_acronym(token: str) -> bool:
    letters = re.sub(r"[^A-Za-z]", "", token)
    if not 2 <= len(letters) <= 8:
        return False
    upper = sum(1 for c in letters if c.isupper())
    return upper >= 2 or (upper == 1 and letters[0].islower() and len(letters) <= 6 and any(c.isupper() for c in letters[1:]))


def _matches_initials(acronym: str, phrase: str) -> bool:
    """True if the acronym's letters follow the phrase: the first letter is the first word's initial, the other
    letters are an ordered subsequence of the phrase's letters, and the last letter falls inside the last word
    (CSF <- cerebrospinal fluid, cAMP <- cyclic adenosine monophosphate, BBB <- blood-brain barrier)."""
    want = _letters(acronym)
    words = [w.lower() for w in re.findall(r"[A-Za-z]+", phrase) if w.lower() not in _STOP]
    if len(want) < 2 or not words or want[0] != words[0][0]:
        return False
    blob, starts = "", []
    for word in words:
        starts.append(len(blob))
        blob += word
    position, last_found = 0, -1
    for ch in want:
        last_found = blob.find(ch, position)
        if last_found < 0:
            return False
        position = last_found + 1
    return last_found >= starts[-1]


def extract_definitions(text: str) -> dict[str, str]:
    """acronym -> expansion for every ``phrase (ACR)`` / ``ACR (phrase)`` whose letters fit the phrase."""
    found: dict[str, str] = {}
    for match in _DEF_FORWARD.finditer(text):
        phrase, acronym = match.group(1).strip(), match.group(2)
        if looks_like_acronym(acronym):
            words = phrase.split()
            for n in range(1, min(len(words), 7) + 1):                     # shortest tail of the phrase that fits
                tail = " ".join(words[-n:])
                if _matches_initials(acronym, tail):
                    found.setdefault(acronym, tail)
                    break
    for match in _DEF_REVERSE.finditer(text):
        acronym, phrase = match.group(1), match.group(2).strip()
        if looks_like_acronym(acronym) and _matches_initials(acronym, phrase):
            found.setdefault(acronym, phrase)
    return found


@dataclass
class AcronymHit:
    acronym: str
    expansion: str
    origin: str                                   # glossary | defined in slides | defined in speech
    occurrences: int = 0
    first_start: Optional[float] = None
    variants: Counter = field(default_factory=Counter)      # how it was written ("CSF", "C S F", "csf")
    expansion_mentions: int = 0
    odd_spellings: int = 0                        # spaced / dotted forms: possible ASR artefacts

    def to_dict(self) -> dict[str, Any]:
        return {"acronym": self.acronym, "expansion": self.expansion, "origin": self.origin,
                "occurrences": self.occurrences, "first_start": self.first_start, "variants": dict(self.variants),
                "expansion_mentions": self.expansion_mentions, "odd_spellings": self.odd_spellings}


class AcronymEngine:
    def __init__(self, glossary: Optional[dict[str, str]] = None) -> None:
        glossary = ACRONYMS if glossary is None else glossary
        self._known: dict[str, tuple[str, str]] = {
            a: (e, "glossary") for a, e in glossary.items() if a not in ACRONYMS_THAT_ARE_AMBIGUOUS}
        self.learned: dict[str, tuple[str, str]] = {}

    def learn(self, text: str, origin: str) -> dict[str, str]:
        """Add definitions found in ``text`` (slides or speech). Learned meanings override the glossary."""
        found = extract_definitions(text)
        for acronym, expansion in found.items():
            entry = (expansion, origin)
            self._known[acronym] = entry
            self.learned[acronym] = entry
        return found

    @property
    def known(self) -> dict[str, tuple[str, str]]:
        return dict(self._known)

    def scan(self, segments: Sequence[Segment]) -> list[AcronymHit]:
        """Acronyms present in the transcript (by exact token, spaced letters, or spoken expansion)."""
        hits: dict[str, AcronymHit] = {}
        by_letters: dict[str, str] = defaultdict(str)
        for acronym in self._known:
            by_letters.setdefault(_letters(acronym), acronym)
        expansion_keys = {a: normalize_key(e) for a, (e, _) in self._known.items()}
        for segment in segments:
            text = segment.text
            tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-]*", text)
            for token in tokens:
                acronym = token if token in self._known else next(
                    (a for a in self._known if a.lower() == token.lower() and len(a) >= 3 and a.upper() == a), None)
                if acronym:
                    self._record(hits, acronym, token, segment.start)
            for match in _SPACED.finditer(text):
                raw = match.group(1)
                letters = re.sub(r"[^A-Za-z]", "", raw).lower()
                if len(letters) >= 3 and letters in by_letters and re.search(r"[ .\-]", raw):
                    self._record(hits, by_letters[letters], raw.strip(), segment.start, odd=True)
            blob = normalize_key(text)
            for acronym, key in expansion_keys.items():
                if len(key) >= 10 and key in blob:
                    self._record(hits, acronym, None, segment.start, expansion=True)
        return sorted(hits.values(), key=lambda h: (-h.occurrences - h.expansion_mentions, h.acronym))

    def _record(self, hits: dict[str, AcronymHit], acronym: str, written: Optional[str], start: float,
                odd: bool = False, expansion: bool = False) -> None:
        expansion_text, origin = self._known[acronym]
        hit = hits.setdefault(acronym, AcronymHit(acronym, expansion_text, origin))
        if hit.first_start is None:
            hit.first_start = start
        if expansion:
            hit.expansion_mentions += 1
        elif written is not None:
            hit.occurrences += 1
            hit.variants[written] += 1
            hit.odd_spellings += int(odd)

    def for_hotwords(self, hits: Sequence[AcronymHit]) -> list[dict[str, Any]]:
        """Acronyms worth biasing the decoder with: defined in this lecture and actually used."""
        return [{"acronym": h.acronym, "expansion": h.expansion, "occurrences": h.occurrences or 1}
                for h in hits if h.origin != "glossary" and h.acronym in self.learned]


# ----------------------------------------------------------------------------------- ambiguity
@dataclass
class AbbreviationFinding:
    abbreviation: str
    start: float
    best: str
    confidence: str                               # CONFIDENT | REVIEW
    scores: dict[str, float]
    evidence: list[str]
    defined_in_lecture: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"abbreviation": self.abbreviation, "start": self.start, "best_meaning": self.best, "confidence": self.confidence,
                "scores": {k: round(v, 2) for k, v in self.scores.items()}, "evidence": self.evidence,
                "defined_in_lecture": self.defined_in_lecture}


def disambiguate(segments: Sequence[Segment], learned: Optional[dict[str, tuple[str, str]]] = None,
                 window_segments: int = 1) -> list[AbbreviationFinding]:
    """Score the meaning of each ambiguous abbreviation from the neighbouring segments' words."""
    learned = learned or {}
    findings: list[AbbreviationFinding] = []
    for index, segment in enumerate(segments):
        for match in re.finditer(r"(?<![A-Za-z])([A-Z]{2,4})(?![A-Za-z])", segment.text):
            abbreviation = match.group(1)
            meanings = AMBIGUOUS.get(abbreviation)
            if not meanings:
                continue
            lo, hi = max(0, index - window_segments), min(len(segments), index + window_segments + 1)
            window = " " + " ".join(re.sub(r"[^a-z0-9]+", " ", s.text.lower()) for s in segments[lo:hi]) + " "
            scores: dict[str, float] = {}
            evidence: dict[str, list[str]] = {}
            for meaning, cues in meanings.items():
                matched = [c for c in cues if f" {c} " in window or (" " not in c and c in window)]
                scores[meaning] = float(len(matched))
                evidence[meaning] = matched
            defined = False
            if abbreviation in learned:
                expansion = learned[abbreviation][0].lower()
                for meaning in scores:
                    if normalize_key(meaning)[:8] == normalize_key(expansion)[:8]:
                        scores[meaning] += 10.0
                        defined = True
            ranked = sorted(scores.items(), key=lambda kv: -kv[1])
            best, top = ranked[0]
            second = ranked[1][1] if len(ranked) > 1 else 0.0
            confident = top >= 2.0 and top >= 2 * second
            findings.append(AbbreviationFinding(
                abbreviation, segment.start, best if top > 0 else "undetermined",
                "CONFIDENT" if confident else "REVIEW", scores, evidence.get(best, []), defined))
    return findings


def acronyms_report_text(hits: Sequence[AcronymHit], findings: Sequence[AbbreviationFinding]) -> str:
    from .exporters import format_hms

    out = ["MEDICAL ACRONYMS", "Reporting only: the transcript is never rewritten.", "",
           f"{len(hits)} acronym(s) found", "",
           f"{'acronym':<9} {'count':>5} {'first':<9} {'origin':<20} expansion  [how it was written]"]
    for h in hits:
        written = ", ".join(f"'{v}' x{n}" for v, n in h.variants.most_common(4)) or "(only spoken in full)"
        flag = "  <-- spaced/dotted spelling: possible ASR artefact" if h.odd_spellings else ""
        first = format_hms(h.first_start) if h.first_start is not None else "-"
        out.append(f"{h.acronym:<9} {h.occurrences:>5} {first:<9} {h.origin:<20} {h.expansion}  [{written}]"
                   f"{'  (+%d spoken in full)' % h.expansion_mentions if h.expansion_mentions else ''}{flag}")
    out += ["", "AMBIGUOUS ABBREVIATIONS (meaning scored from the surrounding words; REVIEW = not decisive)", ""]
    if not findings:
        out.append("  (none)")
    for f in findings:
        ranked = ", ".join(f"{m}: {s:g}" for m, s in sorted(f.scores.items(), key=lambda kv: -kv[1]))
        out += [f"[{format_hms(f.start)}] {f.abbreviation} -> {f.best}  ({f.confidence}"
                f"{', defined in this lecture' if f.defined_in_lecture else ''})",
                f"    scores: {ranked}", f"    evidence: {', '.join(f.evidence) or '-'}"]
    return "\n".join(out) + "\n"
