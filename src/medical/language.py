"""Language stability: a soft language lock, and conservative repair of isolated drifted segments.

Whisper detects ONE language per 30-second window. On accented English the detector often spreads its probability over several (Slavic)
languages with none above ~0.75; the old rule ("two windows deviate -> decode every window in its own language") then turned an English
lecture into two thirds Slovenian word salad. The soft lock changes only the DECISION, never the decoder:

* a recording is really multilingual only when at least two languages each have several STRONG windows (p >= 0.8);
* otherwise one primary language is locked. When the detector is not sure which one, a few windows are decoded in each candidate language
  and the language whose decode is the most probable and the most coherent text wins (arbitration);
* the language is switched only for runs of consecutive strong windows, never for one odd window;
* after the decode, isolated segments that are not in the primary language are re-decoded alone (English-constrained) and replaced only when
  the new text is clearly better. Latin / medical terms are not drift. Nothing is translated.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from ..common_words import common_words
from ..language import _EN_STOPWORDS, _TR_STOPWORDS

log = logging.getLogger(__name__)

STRONG = 0.80                    # a window is "strong" evidence at this probability
MIN_STRONG_WINDOWS = 2           # strong windows a language needs to count as really spoken
MIN_REAL_SHARE = 0.15            # ... and the share of all windows it must have
ARBITRATION_WINDOWS = 3
ARBITRATION_WINDOWS_CPU = 2
_FOREIGN_LETTERS = re.compile(r"[čšžćđőűłąęńśźżřěůťďňőéáíóúüöäß]")
_SHORT_ENGLISH = frozenset("a i an as at be by do go he if in is it me my no of on or so to up us we am are can did had has her him his how its may "
                           "not now off one our out she the too two was who why yet you but for and all any new old also".split())
_WORD = re.compile(r"[A-Za-zÀ-ɏ]+(?:['’-][A-Za-zÀ-ɏ]+)*")


# ====================================================================== text language
def words_of(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def english_share(text: str, medical: Optional[Callable[[str], bool]] = None) -> float:
    """Share of the words that are English (ordinary words, function words, or medical vocabulary). 0..1; Latin terms count as English."""
    toks = words_of(text)
    if not toks:
        return 0.0
    known = common_words()
    hits = 0
    for w in toks:
        if _FOREIGN_LETTERS.search(w):
            continue
        if (len(w) >= 3 and w in known) or w in _EN_STOPWORDS or w in _SHORT_ENGLISH or (medical is not None and len(w) >= 3 and medical(w)):
            hits += 1
    return hits / len(toks)


def turkish_share(text: str) -> float:
    toks = words_of(text)
    if not toks:
        return 0.0
    hits = sum(1 for w in toks if w in _TR_STOPWORDS or re.search(r"[çğıöşü]", w))
    return min(1.0, 3.0 * hits / len(toks))


def language_fit(text: str, language: str, medical: Optional[Callable[[str], bool]] = None) -> float:
    """How much ``text`` looks like ``language``. Only English and Turkish can be judged from word lists; for other codes a text that
    is mostly English words is NOT a good fit (English words forced into another language is exactly the failure to detect)."""
    english = english_share(text, medical)
    if language == "en":
        return english
    own = turkish_share(text) if language == "tr" else max(0.0, 1.0 - 2.0 * english)
    return own * 0.5 if english >= 0.55 else own                               # mostly English words cannot be a good fit for another language


# ====================================================================== plan-time soft lock
@dataclass
class LanguageVerdict:
    primary: Optional[str]
    confidence: float
    multilingual: bool
    reason: str
    arbitrated: bool = False
    strong: dict[str, int] = field(default_factory=dict)
    candidates: dict[str, float] = field(default_factory=dict)
    windows: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"primary": self.primary, "confidence": round(self.confidence, 3), "multilingual": self.multilingual, "reason": self.reason,
                "arbitrated": self.arbitrated, "strong_windows": self.strong, "candidate_scores": {k: round(v, 3) for k, v in self.candidates.items()},
                "arbitration_windows": self.windows}


#: ``decode_window(language, start_seconds) -> (text, mean word probability or None)``; one short decode of one 30 s window
DecodeWindow = Callable[[str, float], tuple[str, Optional[float]]]


def real_languages(samples: Sequence[Sequence]) -> dict[str, int]:
    """Languages that are really spoken: strong windows per language, for those with enough of them."""
    rows = [(float(s), str(lang), float(p)) for s, lang, p in samples]
    strong = Counter(lang for _s, lang, p in rows if p >= STRONG)
    total = max(len(rows), 1)
    return {lang: n for lang, n in strong.items() if n >= MIN_STRONG_WINDOWS and n / total >= MIN_REAL_SHARE}


def stabilize(summary: dict, decode_window: Optional[DecodeWindow] = None, *, on_cpu: bool = False, prefer: Sequence[str] = ("en", "tr"),
              medical: Optional[Callable[[str], bool]] = None, should_stop: Optional[Callable[[], bool]] = None,
              probe: Optional[tuple[str, float]] = None) -> LanguageVerdict:
    """Decide the primary language of a recording from the language-detection ``summary`` (see ``language.summarize_language_samples``).

    ``decode_window`` is optional: without it the decision uses the detector's probabilities only (no extra decoding).
    """
    samples = summary.get("samples") or []
    rows = [(float(s), str(lang), float(p)) for s, lang, p in samples]
    if not rows:
        return LanguageVerdict(None, 0.0, False, "no language samples")
    real = real_languages(rows)
    mass: Counter = Counter()
    for _s, lang, p in rows:
        mass[lang] += p
    total_mass = sum(mass.values()) or 1.0
    shares = {lang: mass[lang] / total_mass for lang in mass}
    if len(real) >= 2:
        order = sorted(real, key=lambda lang: -real[lang])
        return LanguageVerdict(order[0], min(0.95, shares.get(order[0], 0.0) + 0.3), True,
                               f"real multilingual: {', '.join(f'{lang} x{n}' for lang, n in real.items())} strong windows",
                               strong=dict(real), candidates=shares)
    if len(real) == 1:
        lang = next(iter(real))
        if probe and probe[0] != lang and probe[1] >= STRONG:               # the recording OPENS in another language: a real switch
            return LanguageVerdict(lang, min(0.95, shares.get(lang, 0.0) + 0.2), True,
                                   f"real multilingual: the first minutes are {probe[0]} but the recording is {lang}", strong=dict(real),
                                   candidates=shares)
        return LanguageVerdict(lang, min(0.99, 0.6 + 0.4 * shares.get(lang, 0.0)), False,
                               f"soft lock: {lang} has {real[lang]} strong windows, no other language does", strong=dict(real), candidates=shares)
    # weak evidence everywhere (accented speech): which language is it? A few decodes answer better than the detector.
    ranked = [lang for lang, _m in mass.most_common()]
    candidates: list[str] = []
    for lang in list(prefer) + ranked:
        if lang not in candidates:
            candidates.append(lang)
    candidates = candidates[: 2 if on_cpu else 3]
    if decode_window is None:
        lang = ranked[0]
        return LanguageVerdict(lang, shares.get(lang, 0.0), False, f"soft lock without arbitration: {lang} has the most detector evidence",
                               candidates=shares)
    starts = _pick_windows(rows, ARBITRATION_WINDOWS_CPU if on_cpu else ARBITRATION_WINDOWS)
    scores: dict[str, list[float]] = {c: [] for c in candidates}
    detail: list[dict] = []
    for start in starts:
        for lang in candidates:
            if should_stop is not None and should_stop():
                break
            try:
                text, prob = decode_window(lang, start)
            except Exception as error:                                         # arbitration is an aid: never fail the run because of it
                log.warning("language arbitration decode failed (%s @ %.0fs): %s", lang, start, error)
                continue
            fit = language_fit(text, lang, medical)
            quality = prob if prob is not None else 0.5
            score = 0.45 * quality + 0.55 * fit
            scores[lang].append(score)
            detail.append({"start": round(start, 1), "language": lang, "probability": None if prob is None else round(prob, 3),
                           "fit": round(fit, 3), "score": round(score, 3), "text": text[:80]})
    means = {lang: sum(v) / len(v) for lang, v in scores.items() if v}
    if not means:
        lang = ranked[0]
        return LanguageVerdict(lang, shares.get(lang, 0.0), False, "arbitration produced nothing: detector majority", candidates=shares)
    best = max(means, key=means.get)
    others = sorted((v for k, v in means.items() if k != best), reverse=True)
    margin = means[best] - (others[0] if others else 0.0)
    return LanguageVerdict(best, max(0.0, min(0.99, means[best] * (0.5 + min(margin, 0.5)))), False,
                           f"soft lock by arbitration: {best} {means[best]:.2f} vs " + ", ".join(f"{k} {v:.2f}" for k, v in means.items() if k != best),
                           arbitrated=True, candidates=means, windows=detail, strong={})


def _pick_windows(rows: Sequence[tuple[float, str, float]], count: int) -> list[float]:
    """Spread windows over the recording, preferring the ones the detector was most sure about (they are real speech)."""
    ordered = sorted(rows, key=lambda r: -r[2])[: max(count * 3, count)]
    ordered.sort(key=lambda r: r[0])
    if len(ordered) <= count:
        return [r[0] for r in ordered]
    step = len(ordered) / count
    return [ordered[int(i * step)][0] for i in range(count)]


def apply_to_summary(summary: dict, verdict: LanguageVerdict) -> dict:
    """Write the verdict into the summary the pipeline plan is built from (``mixed`` / ``dominant`` / ``reason``)."""
    summary["soft_lock"] = verdict.to_dict()
    if verdict.primary:
        summary["dominant"] = verdict.primary
    summary["mixed"] = bool(verdict.multilingual)
    summary["reason"] = verdict.reason
    return summary


# ====================================================================== post-decode drift repair
@dataclass
class DriftRepair:
    segment: int
    start: float
    original: str
    replacement: Optional[str]
    reason: str
    language: str
    redecoded: bool = False

    def to_dict(self) -> dict:
        h, rest = divmod(int(self.start), 3600)
        m, s = divmod(rest, 60)
        return {"timestamp": f"{h:02d}:{m:02d}:{s:02d}", "segment": self.segment, "original": self.original, "replacement": self.replacement,
                "reason": self.reason, "language": self.language, "redecoded": self.redecoded}


#: ``redecode(start, end, language) -> (text, mean word probability or None)``
Redecode = Callable[[float, float, str], tuple[str, Optional[float]]]

DRIFT_SHARE = 0.45               # a segment below this share of primary-language words has drifted
MAX_RUN = 3                      # more consecutive drifted segments than this is a real passage, not an isolated drift


def drifted_segments(texts: Sequence[str], primary: str, medical: Optional[Callable[[str], bool]] = None) -> list[int]:
    out = []
    for i, text in enumerate(texts):
        if len(words_of(text)) >= 3 and language_fit(text, primary, medical) < DRIFT_SHARE:
            out.append(i)
    return out


def other_language(segment, primary: Optional[str]) -> bool:
    """True when the language plan / the recogniser LABELLED this segment as being in a language other than ``primary``: it is a passage in another language, not a drift."""
    label = getattr(segment, "language", None)
    return bool(label) and label not in ("und", primary) and bool((getattr(segment, "language_meta", None) or {}).get("source"))


def repair_drift(segments: Sequence, primary: Optional[str], redecode: Optional[Redecode] = None,
                 medical: Optional[Callable[[str], bool]] = None, max_redecodes: int = 40) -> tuple[list[str], list[DriftRepair], list[int]]:
    """Isolated drifted segments -> a re-decode in the primary language, kept only when it is clearly better.

    Returns ``(new texts, repairs, drifted segment indices)``. Runs longer than ``MAX_RUN`` are left alone (a real switch or a real
    passage in another language): they are reported, never rewritten.
    """
    texts = [s.text for s in segments]
    if not primary:
        return texts, [], []
    drifted = [i for i in drifted_segments(texts, primary, medical) if not other_language(segments[i], primary)]      # a labelled passage in another language is never "repaired" into the primary one
    repairs: list[DriftRepair] = []
    out = list(texts)
    used = 0
    bad = set(drifted)
    for i in drifted:
        run_left, run_right = i, i
        while run_left - 1 in bad:
            run_left -= 1
        while run_right + 1 in bad:
            run_right += 1
        seg = segments[i]
        if run_right - run_left + 1 > MAX_RUN:
            repairs.append(DriftRepair(i, seg.start, seg.text, None, f"part of a run of {run_right - run_left + 1} segments: left as heard", primary))
            continue
        if redecode is None or used >= max_redecodes:
            repairs.append(DriftRepair(i, seg.start, seg.text, None, "isolated drift (no re-decode available)", primary))
            continue
        used += 1
        try:
            text, prob = redecode(max(0.0, seg.start - 0.3), seg.end + 0.3, primary)
        except Exception as error:
            repairs.append(DriftRepair(i, seg.start, seg.text, None, f"re-decode failed: {error}", primary))
            continue
        old_fit, new_fit = language_fit(seg.text, primary, medical), language_fit(text, primary, medical)
        if text.strip() and new_fit >= max(0.5, old_fit + 0.25) and (prob is None or prob >= 0.35):
            out[i] = text.strip()
            repairs.append(DriftRepair(i, seg.start, seg.text, out[i], f"re-decoded in {primary}: fit {old_fit:.2f} -> {new_fit:.2f}", primary, True))
        else:
            repairs.append(DriftRepair(i, seg.start, seg.text, None, f"re-decode not clearly better (fit {old_fit:.2f} -> {new_fit:.2f}): kept", primary, True))
    return out, repairs, drifted
