"""MedicalContextCorrector: finds the medical term the speaker most likely said where the ASR wrote something that only sounds like it.

It never rewrites the lecturer. A span of one to four words is replaced only when SEVERAL independent kinds of evidence agree:

    sound            the span sounds like a known term (consonant skeleton, ``phonetics``)            - proposes candidates, never decides
    context          the words around it are the words that go with that term (cues, neighbouring terms, same field)
    topic            the lecture is about that field (counted from the terms the lecture really contains)
    lecture          the term is in the slides / custom vocabulary / was already confirmed earlier in this lecture
    asr              how sure the recogniser was about the span (weak evidence on its own: ASR is often confidently wrong)
    oddness          an ordinary phrase of common words is only changed when the context really asks for it

HIGH -> applied; MEDIUM -> applied in the medical transcript and logged; LOW -> the original text stays, the doubt is only recorded.
The raw transcript is never touched: the corrector returns new text and a report.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional, Sequence

from ..common_words import common_words, is_common_word
from .lexicon import Index, Lexicon, Term, norm, stem_phrase, words
from rapidfuzz.distance import Levenshtein

from .phonetics import similarity, skeleton

log = logging.getLogger(__name__)

HIGH, MEDIUM, LOW = "high", "medium", "low"
CORE_DOMAINS = ("abbreviations", "eponyms", "latin")

_TOKEN = re.compile(r"[A-Za-zÀ-ɏ]+(?:['’\-][A-Za-zÀ-ɏ]+)*")
_SENTENCE_END = re.compile(r"[.!?;:]")
_UNIT = re.compile(r"^(?:\d+[.,]?\d*)?(?:mg|µg|ug|ng|pg|kg|g|ml|µl|ul|l|dl|mmhg|mmol|mol|meq|hz|khz|mv|v|µm|um|nm|mm|cm|m|ms|s|min|h|iu|u)(?:/[a-z]+)?$", re.I)
_GREEK = {"alpha", "beta", "gamma", "delta", "epsilon", "kappa", "lambda", "mu", "sigma", "omega", "theta", "tau", "pi", "phi"}
_STOP = {"the", "a", "an", "of", "and", "or", "to", "in", "on", "at", "is", "are", "was", "were", "it", "this", "that", "these", "those", "as",
         "by", "for", "with", "from", "we", "you", "they", "he", "she", "be", "so", "if", "then", "but", "not", "have", "has", "had", "can",
         "will", "which", "what", "who", "there", "here", "also", "very", "just", "all", "some", "one", "two", "three"}

_FUNCTION = _STOP | {"i", "me", "my", "us", "am", "do", "no", "up", "go", "an", "its", "his", "her", "our", "out", "off", "over", "into", "than",
                    "when", "where", "while", "because", "about", "after", "before", "between", "under", "does", "did", "been", "being", "were"}

WEIGHTS = {"sound": 0.30, "context": 0.34, "topic": 0.10, "lecture": 0.10, "oddness": 0.10, "asr": 0.06}


@dataclass
class SegmentView:
    index: int
    start: float
    end: float
    text: str
    language: Optional[str] = None
    probs: list[tuple[str, Optional[float]]] = field(default_factory=list)       # (word, probability) in order


@dataclass
class Correction:
    segment: int
    timestamp: float
    original: str
    corrected: str
    confidence: float
    tier: str
    domain: str
    reason: str
    context: str
    kind: str = "term"
    category: str = "medical"                  # medical | case (capitalisation / spelling of a term that was right)
    evidence: dict[str, float] = field(default_factory=dict)
    span: tuple[int, int] = (0, 0)
    candidate_key: str = ""

    def to_dict(self) -> dict:
        h, rest = divmod(int(self.timestamp), 3600)
        m, s = divmod(rest, 60)
        return {"timestamp": f"{h:02d}:{m:02d}:{s:02d}", "segment": self.segment, "original": self.original, "corrected": self.corrected,
                "confidence": round(self.confidence, 3), "tier": self.tier, "domain": self.domain, "reason": self.reason,
                "context": self.context, "category": self.category, "evidence": {k: round(v, 3) for k, v in self.evidence.items()}}


@dataclass
class CorrectionResult:
    texts: list[str]
    corrections: list[Correction]
    uncertain: list[Correction]
    term_hits: Counter
    topic: dict[str, float]
    domains_used: list[str]
    options: dict = field(default_factory=dict)

    @property
    def applied(self) -> list[Correction]:
        return [c for c in self.corrections if c.tier in (HIGH, MEDIUM)]


@dataclass
class Options:
    language: str = "en"
    subjects: Sequence[str] = ()
    custom_terms: Sequence[str] = ()
    lecture_terms: Sequence[str] = ()
    apply_medium: bool = True
    max_span: int = 4
    window: int = 4                                      # segments before / after used as context (about 12 s each way)
    high: float = 0.76
    medium: float = 0.60
    low: float = 0.55
    refine: bool = True                                  # second pass: corrected neighbours are context for the doubtful ones
    max_per_segment: int = 4
    max_share: float = 0.04                              # corrections above this share of the words: the thresholds are raised


class MedicalContextCorrector:
    def __init__(self, lexicon: Optional[Lexicon] = None, options: Optional[Options] = None) -> None:
        self.lexicon = lexicon or Lexicon()
        self.options = options or Options()
        self.last_proposals: list[dict] = []
        self._why = ""
        self._related: set[str] = set()
        self._fixed: Optional[tuple[list[str], list]] = None             # (texts with the first pass applied, their exact hits)
        self.trace: Optional[list] = None                                # set to [] to collect (heard, candidate, reason) of dropped candidates

    # ================================================================== public
    def correct(self, segments: Sequence[SegmentView]) -> CorrectionResult:
        opt = self.options
        if opt.custom_terms:
            self.lexicon.add_terms("custom", opt.custom_terms)
        if opt.lecture_terms:
            self.lexicon.add_terms("lecture", opt.lecture_terms)
        full = self.lexicon.subset(None)
        segments = [dataclasses.replace(s, index=k) for k, s in enumerate(segments)]
        texts = [s.text for s in segments]
        tokens = [self._tokens(s.text) for s in segments]
        hits = [self._exact_hits(toks, full) for toks in tokens]
        counts: Counter = Counter()
        for per in hits:
            for _, _, term in per:
                counts[term.key] += 1
        topic = self._topic(hits, full)
        focus_names = self._focus(topic)
        self._related = {r for d, w in topic.items() if w >= 0.2 for r in self.lexicon.related(d)}
        focus = self.lexicon.subset(focus_names)
        lecture_keys = {norm(t) for t in opt.lecture_terms} | {norm(t) for t in opt.custom_terms}
        self._fixed = None
        proposals = self._propose(segments, tokens, hits, focus, full, topic, lecture_keys, counts)
        chosen = self._consistency(segments, proposals)
        corrections, uncertain = self._select(chosen, segments, tokens)
        if opt.refine and corrections:
            fixed = self._apply(texts, corrections)
            fixed_hits = [self._exact_hits(self._tokens(text), full) for text in fixed]
            counts2: Counter = Counter()
            for per in fixed_hits:
                for _, _, term in per:
                    counts2[term.key] += 1
            self._fixed = (fixed, fixed_hits)
            proposals = self._propose(segments, tokens, hits, focus, full, topic, lecture_keys, counts2)
            chosen = self._consistency(segments, proposals)
            corrections, uncertain = self._select(chosen, segments, tokens)
            self._fixed = None
        self.last_proposals = proposals                                  # kept for diagnostics / tuning
        case = [c for c in self._case_fixes(segments, tokens, hits)
                if not any(m.segment == c.segment and m.span[0] < c.span[1] and c.span[0] < m.span[1] for m in corrections)]
        out_texts = self._apply(texts, corrections + case)
        return CorrectionResult(out_texts, corrections + case, uncertain, counts, topic, focus_names)

    # ================================================================== tokens
    @staticmethod
    def _tokens(text: str) -> list[tuple[str, int, int]]:
        return [(m.group(0), m.start(), m.end()) for m in _TOKEN.finditer(text)]

    @staticmethod
    def _protected(token: str) -> bool:
        low = token.lower()
        return low in _GREEK or bool(_UNIT.match(low)) or len(low) < 3 or bool(_CONTRACTION.match(low))

    # ================================================================== exact terms and topic
    def _exact_hits(self, toks: list[tuple[str, int, int]], index: Index) -> list[tuple[int, int, Term]]:
        """Spans of the text that already ARE a known term: ``[(first token, end token, term)]`` (longest first, no overlap)."""
        found: list[tuple[int, int, Term]] = []
        taken: set[int] = set()
        for n in range(min(index.max_tokens, 6), 0, -1):
            for i in range(len(toks) - n + 1):
                if any(j in taken for j in range(i, i + n)) or (n == 1 and toks[i][0].lower() in _STOP):
                    continue
                terms = index.exact(" ".join(t[0] for t in toks[i:i + n]))
                if terms:
                    found.append((i, i + n, terms[0]))
                    taken.update(range(i, i + n))
        found.sort()
        return found

    def _topic(self, hits: list, index: Index) -> dict[str, float]:
        weight: Counter = Counter()
        for per in hits:
            for first, last, term in per:
                w = 1.0 + 0.5 * (term.tokens - 1)
                if term.domain not in CORE_DOMAINS and term.domain not in ("custom", "lecture"):
                    weight[term.domain] += w
        for subject in self.options.subjects:
            for domain in _SUBJECT_DOMAINS.get(subject.lower(), ()):
                weight[domain] += 2.0
        total = sum(weight.values())
        return {d: round(w / total, 4) for d, w in weight.most_common(8)} if total else {}

    def _focus(self, topic: dict[str, float]) -> list[str]:
        names = {d for d, w in topic.items() if w >= 0.06}
        for d in list(names):
            names.update(self.lexicon.related(d))
        names.update(CORE_DOMAINS)
        if len(names) <= len(CORE_DOMAINS):                              # no topic could be found: everything is possible
            return self.lexicon.names()
        return sorted(n for n in names if self.lexicon.domain(n) is not None)

    # ================================================================== proposals
    def _window(self, segments: Sequence[SegmentView], hits: list, i: int) -> tuple[str, list[Term]]:
        lo, hi = max(0, i - self.options.window), min(len(segments), i + self.options.window + 1)
        fixed = self._fixed
        text = " ".join(norm_words(fixed[0][j] if fixed and j != i else segments[j].text) for j in range(lo, hi))
        terms = [t for j in range(lo, hi) for _, _, t in (fixed[1][j] if fixed and j != i else hits[j])]
        return text, terms

    def _propose(self, segments, tokens, hits, focus: Index, full: Index, topic, lecture_keys, counts) -> list[dict]:
        out: list[dict] = []
        for i, seg in enumerate(segments):
            if not tokens[i] or not _english_text(tokens[i]):
                continue
            toks = tokens[i]
            covered = {j for first, last, _ in hits[i] for j in range(first, last)
                       if norm(" ".join(x[0] for x in toks[first:last])) not in focus.by_confusable}   # a known mix-up stays correctable
            window_text, window_terms = self._window(segments, hits, i)
            probs = _align(toks, seg.probs)
            for n in range(min(self.options.max_span, focus.max_tokens + 1), 0, -1):
                for s in range(len(toks) - n + 1):
                    span = toks[s:s + n]
                    flags = [(s + k) in covered for k in range(n)]
                    if _boundary(seg.text, span) or all(flags) or seg.text[span[-1][2]:span[-1][2] + 1] == "-" or seg.text[max(0, span[0][1] - 1):span[0][1]] == "-":
                        continue
                    if span[0][0].lower() in _STOP or span[-1][0].lower() in _STOP:
                        continue
                    if all(self._protected(t[0]) for t in span):
                        continue
                    heard = " ".join(t[0] for t in span)
                    cands = sorted(focus.candidates(heard, limit=24), key=lambda c: -similarity(heard, c[0]))[:12]
                    if n >= 2 and any(flags):
                        for k, f in enumerate(flags):
                            if f:
                                cands += [(sf, tm, 0.0) for sf, tm in focus.containing(span[k][0])[:40]]
                    for term in focus.by_confusable.get(norm(heard), ()):
                        cands.extend((surf, term, 0.0) for surf in _number_forms(term, heard))
                    seen: set = set()
                    for surface, term, sim in cands:
                        if (surface, term.key, term.domain) in seen:
                            continue
                        seen.add((surface, term.key, term.domain))
                        if re.search(r"\d", surface) and not re.search(r"\d", heard):
                            continue
                        listed = (norm(surface) in lecture_keys or term.domain in ("custom", "lecture")      # slides / own terms: a one-letter slip is a slip
                                  or term.kind == "eponym")                                                  # so is a dropped letter of a name (Schwan)
                        if len(_TOKEN.findall(surface)) > n or abs(len(_TOKEN.findall(surface)) - n) > 1 or norm(surface) == norm(heard) or (not listed and _same_form(heard, surface, n))                                 or _contained(heard, surface):
                            continue
                        near = {norm(x[0]) for x in toks[max(0, s - 2):s] + toks[s + n:s + n + 2]}
                        changed = _changed(span, surface, flags, listed, strong=n >= 2 and similarity(heard, surface) >= 0.9)
                        if changed is not None and any(norm(w) in near for w in changed[1].split() if len(norm(w)) >= 4):
                            changed = None
                        if changed is None:
                            if self.trace is not None:
                                self.trace.append((heard, surface, "alignment"))
                            continue
                        prop = self._score(heard, surface, term, span, probs[s:s + n], window_text, window_terms, topic, lecture_keys,
                                           confusable=term in focus.by_confusable.get(norm(heard), ()), changed=changed,
                                           known=counts.get(term.key, 0), rare=focus.token_freq, cue_df=focus.cue_df, known_words=full.known)
                        if prop is None and self.trace is not None:
                            self.trace.append((heard, surface, self._why))
                        if prop is not None:
                            prop.update(segment=i, first=s, last=s + n, span=(span[0][1], span[-1][2]), heard=heard)
                            out.append(prop)
        return out

    def _score(self, heard, surface, term: Term, span, probs, window_text, window_terms, topic, lecture_keys, confusable: bool,
               changed: tuple, known: int = 0, rare: Optional[dict] = None, cue_df: Optional[dict] = None,
               known_words: Optional[set] = None) -> Optional[dict]:
        sound = similarity(heard, surface)
        if confusable:
            sound = max(sound, 0.72)
        rare = rare or {}
        changed_heard, changed_surface, anchor, kept, gated, pairs, covered_changed = changed
        # a rare word that stays (gamma-aminobutyric ...) identifies the term: the one word that differs need not sound like it
        anchored = anchor >= 0.6 and any(rare.get(norm(part), 9) <= 2 and len(norm(part)) >= 11 and not is_common_word(part.lower())
                                         for k in kept for part in k.replace("-", " ").split())
        if sound < 0.58:
            self._why = f"sound {sound:.2f}"
            return None
        if gated and not confusable and not anchored and similarity(changed_heard, changed_surface) < 0.5:
            self._why = f"changed words {similarity(changed_heard, changed_surface):.2f}"
            return None
        if gated and not confusable and not anchored and len(pairs) > 1 and min(similarity(h, s) for h, s in pairs) < 0.45:
            self._why = "one changed word sounds nothing like its replacement"
            return None
        cue = _cue_score(term, window_text, cue_df)
        coherence = _coherence(term, window_terms)
        context = min(1.0, 0.65 * cue + 0.55 * coherence)
        if anchor >= 0.3:                                                    # words of the span that already ARE the term are evidence too
            context = max(context, min(1.0, 0.9 * anchor))
        dom = min(1.0, sum(topic.get(d, 0.0) for d in {term.domain}) * 1.6) if topic else 0.4
        if term.domain in CORE_DOMAINS:
            dom = max(dom, 0.4)
        elif topic and term.domain in self._related:
            dom = max(dom, 0.5)
        explicit = 1.0 if norm(surface) in lecture_keys or term.domain in ("custom", "lecture") else 0.0
        lecture = max(explicit, min(0.8, 0.2 * math.log2(1 + known)))        # slides / custom terms, and terms this lecture keeps using
        by_word = {norm(w): q for w, q in probs if q is not None}
        parts = [by_word[norm(w)] for w in changed_heard.replace("-", " ").split() if norm(w) in by_word] or [w for _, w in probs if w is not None]
        p_mean = sum(parts) / len(parts) if parts else None
        asr = 0.5 if p_mean is None else max(0.0, min(1.0, 1.0 - p_mean))
        if not gated and not confusable and not (parts and min(parts) < 0.6) and context < 0.9:       # a compound ("mitral glial" -> microglial)
            self._why = "compound of words the recogniser was sure of"                                 # needs a doubted word or overwhelming context
            return None
        if covered_changed and not (context >= 0.7 and ((parts and min(parts) < 0.5) or context >= 0.85)):
            self._why = "an exact word inside a phrase: needs doubt or overwhelming context"
            return None
        backed = context >= 0.60 and sound >= 0.78
        long_word = len(span) == 1 and len(norm(heard)) >= 8 and p_mean is not None and p_mean >= 0.6 and sound < 0.85 and not confusable and not backed
        if long_word and not (context >= 0.85 and sound >= 0.72):
            self._why = "long confident word"
            return None
        penalty = 0.08 if long_word else 0.0                          # overwhelming context may still win, but with less credit
        if parts and min(parts) >= 0.85 and sound < 0.80 and not confusable and not explicit and not backed:
            self._why = "confident words, weak sound"
            return None
        commons = [_plain(t[0]) for t in span]
        rare_term = any(not _plain(w) for w in _TOKEN.findall(surface))
        odd = (0.6 if rare_term else 0.2) if all(commons) else (0.6 if any(commons) else 1.0)
        if len(span) == 1 and not span[0][0].islower() and term.kind in ("eponym", "gene", "abbr"):
            odd = max(odd, 0.7)
        score = (WEIGHTS["sound"] * sound + WEIGHTS["context"] * context + WEIGHTS["topic"] * dom + WEIGHTS["lecture"] * lecture
                 + WEIGHTS["oddness"] * odd + WEIGHTS["asr"] * asr)
        # an ordinary phrase of ordinary words changes only when the words around it really ask for the term
        slip = (len(changed_heard.split()) == 1 and len(norm(changed_heard)) >= 8 and len(changed_surface.split()) == 1
                and Levenshtein.distance(norm(changed_heard), norm(changed_surface)) <= 1 and not is_common_word(changed_heard.lower()) and sound >= 0.9)
        if (known_words and gated and not slip and not explicit and not confusable and similarity(changed_heard, changed_surface) < (0.85 if len(changed_heard.split()) == 1 else 0.66) and not all(_plain(w) for w in changed_heard.split())
                and all(_plain(w) or stem_phrase(w) in known_words for w in changed_heard.split())):
            self._why = "words the vocabulary itself uses"                    # "inhibitory neurotransmitter" is not a mishearing
            return None
        if all(_plain(w) for w in changed_heard.split()) and not explicit and not confusable and not slip:
            low = min(parts) if parts else 1.0                               # the least sure word of the span, as the recogniser saw it
            if len(changed_heard.split()) == 1 and low >= 0.5:
                self._why = "single ordinary word the ASR was sure of"
                return None
            doubted = low < 0.6
            need = 0.40 if low < 0.35 else (0.48 if doubted else 0.85)       # a word the ASR itself doubted needs less outside support
            if context < need or sound < (0.85 if len(span) == 1 else 0.78):
                self._why = f"ordinary words ctx={context:.2f} doubted={doubted}"
                return None
        score -= penalty
        if slip:
            score += 0.12                                                    # fibrilation -> fibrillation, spliceosom -> spliceosome
        if explicit:                                                         # a slide / own term that sounds almost exactly like an unknown word
            score += 0.15 * max(0.0, (sound - 0.85) / 0.15)
        if term.kind == "eponym" and sound >= 0.93 and not any(_plain(w) for w in changed_heard.split()):    # a name spelled one letter off (Schwan)
            score += 0.12 * min(1.0, (sound - 0.9) / 0.1)
        if _plural(heard) != _plural(surface):
            score -= 0.04                                                    # "Schwann shells" is "Schwann cells", not "Schwann cell"
        score = _risk(term, score, sound)
        return {"score": score, "sound": sound, "context": context, "topic": dom, "lecture": lecture, "explicit": explicit, "asr": asr,
                "odd": odd, "cue": cue, "coherence": coherence, "term": term, "surface": surface, "confusable": confusable,
                "ordinary": all(_plain(w) for w in changed_heard.split()) or covered_changed, "slip": slip}

    # ================================================================== choose
    def _tier(self, p: dict, boost: float = 0.0) -> str:
        opt, score = self.options, p["score"] + boost
        strict = 0.06 if p["term"].kind == "drug" else 0.0
        support = max(p["context"], 0.9 * p["explicit"], 0.3 if p.get("slip") else 0.0)       # a one-letter slip of a long medical word is its own evidence
        if score >= opt.high + strict and p["sound"] >= 0.75 and support >= 0.40 and not p.get("ordinary"):
            return HIGH                                                             # an ordinary word turned into a term is never "sure"
        if score >= opt.medium + strict and p["sound"] >= 0.66 and support >= 0.25:
            return MEDIUM
        if score >= opt.medium + 0.10 + strict and p["sound"] >= 0.60 and support >= 0.80:      # little sound, overwhelming context
            return MEDIUM
        if score >= opt.low and p["sound"] >= 0.62:
            return LOW
        return ""

    def _consistency(self, segments, proposals: list[dict]) -> list[dict]:
        """A mishearing repeats: once a (heard, term) pair is HIGH somewhere in the lecture, its other occurrences are believed more."""
        confirmed = Counter((norm(p["heard"]), p["term"].key) for p in proposals if self._tier(p) == HIGH)
        repeated = Counter((norm(p["heard"]), p["term"].key) for p in proposals if self._tier(p) == MEDIUM)
        for p in proposals:
            pair = (norm(p["heard"]), p["term"].key)
            boost = 0.10 if confirmed.get(pair, 0) >= 1 else (0.06 if repeated.get(pair, 0) >= 2 else 0.0)
            p["tier"] = self._tier(p, boost)
            p["boost"] = boost
        return [p for p in proposals if p["tier"]]

    def _select(self, proposals: list[dict], segments, tokens) -> tuple[list[Correction], list[Correction]]:
        by_seg: dict[int, list[dict]] = defaultdict(list)
        for p in proposals:
            by_seg[p["segment"]].append(p)
        order = {HIGH: 2, MEDIUM: 1, LOW: 0}
        applied: list[Correction] = []
        doubt: list[Correction] = []
        words_total = max(1, sum(len(t) for t in tokens))
        for i, plist in sorted(by_seg.items()):
            plist.sort(key=lambda p: (-(p["score"] + p["boost"] + 0.03 * (p["last"] - p["first"] - 1)), -(p["last"] - p["first"])))
            used: set[int] = set()
            count = 0
            for p in plist:
                span_ids = set(range(p["first"], p["last"]))
                if span_ids & used:
                    continue
                rivals = [q for q in plist if q is not p and q["term"].key != p["term"].key and q["first"] == p["first"] and q["last"] == p["last"]
                          and skeleton(q["surface"]) != skeleton(p["surface"])
                          and p["score"] + p["boost"] - (q["score"] + q["boost"]) < 0.10 and p["sound"] - q["sound"] < 0.15
                          and q["score"] + q["boost"] >= self.options.medium - 0.02]       # only a credible rival makes a reading doubtful
                if rivals and p["tier"] != LOW:
                    p["tier"] = LOW
                    p["ambiguous"] = rivals[0]["surface"]
                spelled = [q for q in plist if q is not p and q["first"] == p["first"] and q["last"] == p["last"] and q["term"].key == p["term"].key
                           and norm(q["surface"]) == norm(p["term"].canonical) and p["score"] + p["boost"] - (q["score"] + q["boost"]) < 0.04]
                if spelled and norm(p["surface"]) != norm(p["term"].canonical):
                    p["surface"] = spelled[0]["surface"]
                used |= span_ids
                correction = self._make(p, segments[i])
                if p["tier"] == LOW or count >= self.options.max_per_segment:
                    correction.tier = LOW
                    doubt.append(correction)
                else:
                    applied.append(correction)
                    count += 1
        limit = max(3, int(words_total * self.options.max_share))
        if len(applied) > limit:                                       # over-correcting: keep only the strongest
            applied.sort(key=lambda c: -c.confidence)
            keep = applied[:limit]
            doubt += [c for c in applied if c not in keep]
            for c in doubt:
                c.tier = LOW if c not in keep else c.tier
            applied = keep
        if not self.options.apply_medium:
            doubt += [c for c in applied if c.tier == MEDIUM]
            applied = [c for c in applied if c.tier == HIGH]
        return sorted(applied, key=lambda c: (c.segment, c.span)), sorted(doubt, key=lambda c: (c.segment, c.span))

    def _make(self, p: dict, seg: SegmentView) -> Correction:
        term: Term = p["term"]
        surface = _recase(p["surface"], seg.text, p["span"], term)
        e = {k: p[k] for k in ("sound", "context", "topic", "lecture", "asr", "odd")}
        reasons = [f"sounds like “{p['surface']}” ({p['sound']:.2f})"]
        if p["cue"] >= 0.3:
            reasons.append("surrounding words belong to it")
        if p["coherence"] >= 0.3:
            reasons.append("neighbouring terms are from the same field")
        if p["lecture"]:
            reasons.append("term is in the slides / custom vocabulary")
        if p["topic"] >= 0.4:
            reasons.append(f"lecture topic ({term.domain})")
        if p.get("boost"):
            reasons.append("same mishearing confirmed elsewhere in the lecture")
        if p.get("ambiguous"):
            reasons.append(f"ambiguous: could also be “{p['ambiguous']}”")
        if p["confusable"]:
            reasons.append("a known ASR confusion")
        lo, hi = max(0, p["span"][0] - 70), min(len(seg.text), p["span"][1] + 70)
        return Correction(seg.index, seg.start, seg.text[p["span"][0]:p["span"][1]], surface, round(min(p["score"] + p["boost"], 0.99), 3),
                          p["tier"], term.domain, "; ".join(reasons), seg.text[lo:hi].strip(), term.kind, "medical", e, p["span"], term.key)

    def _case_fixes(self, segments, tokens, hits) -> list[Correction]:
        """A term that was heard right but written in the wrong case (``gaba``, ``schwann cell``, ``mtor``): the known spelling."""
        out: list[Correction] = []
        for i, per in enumerate(hits):
            seg = segments[i]
            for first, last, term in per:
                span = (tokens[i][first][1], tokens[i][last - 1][2])
                heard = seg.text[span[0]:span[1]]
                if len(norm(heard)) < 3 or (is_common_word(heard.lower()) and term.kind != "eponym"):
                    continue                                              # "er", "ms", "da" ...: an ordinary word more often than an abbreviation
                same = [s for s in term.surfaces if norm(s) == norm(heard)]
                joined = next((s for s in same if not re.search(r"[\s-]", s)), None)
                if joined and re.search(r"\s", heard.strip()) and heard.lower().replace(" ", "") == joined.lower() and len(joined) >= 8:
                    out.append(Correction(i, seg.start, heard, _recase(joined, seg.text, span, term), 0.95, HIGH, term.domain, "a word split in two",
                                          "", term.kind, "case", {}, span, term.key))                     # "athero sclerosis" -> "atherosclerosis"
                    continue
                surface = next((s for s in same if not s.isascii()), None) or (same[0] if same else term.canonical)     # diacritics are restored
                want = _recase(surface, seg.text, span, term)
                accents = not want.isascii() and want != heard
                case = heard.lower().replace("-", " ") == want.lower().replace("-", " ") and (term.kind != "term" or want[1:] != want[1:].lower())
                if heard != want and norm(heard) == norm(want) and (accents or case):
                    out.append(Correction(i, seg.start, heard, want, 0.99, HIGH, term.domain, "known spelling / capitalisation", "",
                                          term.kind, "case", {}, span, term.key))
        return out

    @staticmethod
    def _apply(texts: list[str], corrections: list[Correction]) -> list[str]:
        out = list(texts)
        for i in sorted({c.segment for c in corrections}, reverse=False):
            text = out[i]
            for c in sorted((c for c in corrections if c.segment == i and c.tier in (HIGH, MEDIUM)), key=lambda c: -c.span[0]):
                text = text[:c.span[0]] + c.corrected + text[c.span[1]:]
            out[i] = text
        return out


# ====================================================================== scoring helpers
_SUBJECT_DOMAINS = {
    "neuroanatomy": ("neuroanatomy", "neuroscience", "histology"), "neuroscience": ("neuroscience", "neuroanatomy", "physiology"),
    "anatomy": ("anatomy",), "histology": ("histology",), "physiology": ("physiology",), "biochemistry": ("biochemistry",),
    "molecular_biology": ("molecular_biology", "genetics"), "pharmacology": ("pharmacology",), "pathology": ("pathology",),
}


norm_words = stem_phrase


def _align(toks, probs: list[tuple[str, Optional[float]]]) -> list[tuple[str, Optional[float]]]:
    """Probability of each text token from the word list of the ASR (matched in order by normalised word)."""
    out, j = [], 0
    for tok, _, _ in toks:
        key = norm(tok)
        found = None
        for k in range(j, min(j + 16, len(probs))):                      # words removed by the repetition repair are skipped over
            if norm(probs[k][0]) == key:
                found, j = probs[k][1], k + 1
                break
        out.append((tok, found))
    return out


def _english_text(toks) -> bool:
    """The segment tag of the ASR can be wrong (English words decoded under another language): look at the words themselves."""
    if len(toks) < 2:
        return False
    known = common_words()
    return sum(1 for t in toks if t[0].lower() in known or t[0].lower() in _FUNCTION) / len(toks) >= 0.40


_MEDICAL_ENDING = re.compile(r"(?:ases?|ations?|ated|ating|osis|itis|emia|aemia|pathy|ectomy|otomy|ostomy|plasty|scopy|ology|genesis|lysis|izations?|ized)$")


_CONTRACTION = re.compile(r"^[a-z]+(?:n[’']t|[’']s|[’']re|[’']ve|[’']ll|[’']d|[’']m)$")


def _plain(word: str) -> bool:
    """An ordinary English word - or a word built like a medical one (transaminated, phosphorylation): a valid word, not a mishearing."""
    low = word.lower()
    if "-" in low.strip("-"):
        parts = [p for p in low.split("-") if p]
        return bool(parts) and all(_plain(p) for p in parts)
    return (bool(_CONTRACTION.match(low)) or is_common_word(low) or is_common_word(stem_phrase(low)) or is_common_word(low.rstrip("s"))
            or (len(low) >= 9 and bool(_MEDICAL_ENDING.search(low))))


def _contained(heard: str, surface: str) -> bool:
    """"so-called blood-brain barrier" / "blood-brain barrier", "ventricular" / "ventricular zone": words added or dropped, not misheard."""
    a, b = norm(heard).rstrip("s"), norm(surface).rstrip("s")
    slip = " " not in heard.strip() and abs(len(a) - len(b)) <= 2                      # one word, one or two letters off: a misspelling
    return len(a) >= 5 and len(b) >= 5 and not slip and (a in b or b in a)


def _same_form(heard: str, surface: str, n: int) -> bool:
    """Another inflection / extension of the same word is not a mishearing (spinal / spine, glia / glial, unmyelinated / myelinated)."""
    a, b = norm(heard), norm(surface)
    if n != 1 or not a or not b:
        return False
    prefix = 0
    for x, y in zip(a, b):
        if x != y:
            break
        prefix += 1
    related = a.startswith(b) or b.startswith(a)
    if a + "e" == b or b + "e" == a:
        return False
    distance = Levenshtein.distance(a, b)
    if prefix >= max(4, int(0.7 * min(len(a), len(b)))) and (related or distance >= 2 or (prefix >= 8 and distance <= 3)):
        return True
    if prefix >= max(5, int(0.6 * min(len(a), len(b)))) and distance >= 2:
        return True
    if len(a) <= 6 and distance <= 1 and (is_common_word(a) or is_common_word(b)):
        return True
    return (a.endswith(b) or b.endswith(a)) and abs(len(a) - len(b)) <= 4


def _plural(text: str) -> bool:
    last = norm(text.split()[-1]) if text.split() else ""
    return last.endswith("s") and not last.endswith(("ss", "us", "is"))


def _number_forms(term: Term, heard: str) -> list[str]:
    """Surface of a term for a confusable hit: the plural when the heard word is a plural ("monosaccharides" -> "monocytes")."""
    plural = norm(heard).endswith("s")
    forms = [s for s in term.surfaces if (norm(s).endswith("s") == plural)]
    return forms[:2] or [term.canonical]


def _changed(span, surface: str, flags: list[bool], listed: bool = False, strong: bool = False):
    """Which words of the span really differ from the candidate: ``(heard words, term words, share already matching, kept words, gated)``
    or None. Words that already ARE the term ("Schwann" in "Schwann shells") are evidence, not something to change; another inflection of
    the same word ("glia" / "glial") is not a mishearing; a span that overlaps an exact term but cannot be lined up with the candidate is
    only kept as a compound ("mitral glial" -> "microglial": the exact word is inside the candidate and the rest is not an ordinary word)."""
    stoks = _TOKEN.findall(surface)
    heard = [t[0] for t in span]
    if len(stoks) != len(heard):
        if not any(flags):
            if len(stoks) < len(heard) and (similarity(" ".join(heard), surface) < 0.85 or any(similarity(h, surface) >= 0.78 for h in heard)):
                return None                                          # a merge needs pieces that only TOGETHER sound like the term (not "Porphyria means")
            return " ".join(heard), surface, 0.0, [], True, [], False
        ns = norm(surface)
        covered = [norm(h) for h, f in zip(heard, flags) if f]
        rest = [h for h, f in zip(heard, flags) if not f]
        if len(stoks) == 1 and all(c in ns for c in covered) and rest and not any(is_common_word(h.lower()) for h in rest):
            return " ".join(rest), surface, 0.0, [], False, [], False
        return None
    diff = [(h, s, f) for h, s, f in zip(heard, stoks, flags) if norm(h) != norm(s)]
    if not diff or any(len(norm(h)) <= 2 for h, _, _ in diff) or (any(f for _, _, f in diff) and not strong):
        return None                                                      # (an exact word may change only inside a phrase that sounds like the term)
    if not listed and any(_same_form(h, s, 1) for h, s, _ in diff):
        return None
    same = sum(len(norm(h)) for h, s in zip(heard, stoks) if norm(h) == norm(s))
    total = max(1, sum(len(norm(s)) for s in stoks))
    kept = [h for h, s in zip(heard, stoks) if norm(h) == norm(s)]
    return " ".join(h for h, _, _ in diff), " ".join(s for _, s, _ in diff), same / total, kept, True, [(h, s) for h, s, _ in diff], any(f for _, _, f in diff)


def _boundary(text: str, span) -> bool:
    return bool(_SENTENCE_END.search(text[span[0][1]:span[-1][1]]))


def _cue_score(term: Term, window_text: str, df: Optional[dict] = None) -> float:
    """How many of the term's cue words / phrases are in the window; a cue that many terms share counts for less (IDF)."""
    if not term.cues:
        return 0.0
    padded = f" {window_text} "
    hits = 0.0
    for cue in term.cues:
        phrase = norm_words(cue)
        if phrase and f" {phrase} " in padded or (phrase and phrase in padded and len(phrase) >= 6):
            weight = 1.0 / (1.0 + 0.55 * math.log2(df.get(phrase, 1))) if df else 1.0
            hits += (1.0 if " " in phrase else 0.7) * weight
    return 1.0 - math.exp(-1.1 * hits)


def _coherence(term: Term, window_terms: list[Term]) -> float:
    """Neighbouring terms that are related to this one: listed in each other's cues (under any of their spellings), or the same field."""
    if not window_terms:
        return 0.0
    mine = {norm(c) for c in term.cues}
    my_names = {norm(s) for s in term.surfaces}
    total = 0.0
    for other in window_terms:
        if other.key == term.key:
            continue
        names = {norm(s) for s in other.surfaces}
        related = bool(mine & names) or bool(my_names & {norm(c) for c in other.cues})
        same = other.domain == term.domain
        weight = 1.0 + 0.4 * (other.tokens - 1)
        if related:
            total += 1.3 * weight
        elif same and other.domain not in CORE_DOMAINS:
            total += 0.2 * weight
    return 1.0 - math.exp(-0.45 * total)


def _risk(term: Term, score: float, sound: float) -> float:
    """Short, plain and common-looking terms are easier to wrongly 'find' in ordinary speech: ask for a little more."""
    letters_n = len(norm(term.canonical))
    if letters_n <= 5:
        score -= 0.06
    if term.kind in ("gene", "abbr") and sound < 0.9:
        score -= 0.08
    return score


def _recase(surface: str, text: str, span: tuple[int, int], term: Term) -> str:
    """The lexicon's capitalisation is authoritative (Nissl, GABA, mTOR); an ordinary lower-case term starts a sentence with a capital."""
    if surface[:1].isupper() or not surface:
        return surface
    before = text[: span[0]].rstrip()
    opens = before[-1] in ".!?" if before else text[span[0]:span[0] + 1].isupper()
    return surface[:1].upper() + surface[1:] if opens else surface
