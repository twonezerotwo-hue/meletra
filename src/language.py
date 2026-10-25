"""Language selection (auto / en / tr / other Whisper codes)."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Optional, Sequence

AUTO = "auto"
PRIMARY_LANGUAGES = {"en": "English", "tr": "Turkish"}
_CODE_RE = re.compile(r"^[a-z]{2,3}$")

MIXED_LANGUAGE_NOTE = (
    "Whisper decodes with ONE language per 30-second window. In auto mode the language is detected on windows spread over the whole "
    "recording (not only the first minutes): a single dominant language is then forced for the whole file, and a recording whose "
    "windows clearly differ is decoded with per-window language detection (multilingual). In a Turkish lecture with English medical "
    "terms either fixed choice has a cost: --language tr keeps Turkish grammar correct but may spell English terms phonetically; "
    "--language en forces English and turns Turkish sentences into garbage. Per-window detection helps when whole passages switch "
    "language, not single words."
)


#: languages with a medical vocabulary of their own (hotwords are only ever built from these)
HOTWORD_LANGUAGES = ("en", "tr")
SAMPLE_WINDOWS = 24
SAMPLE_SECONDS = 30.0
SAMPLE_MIN_PROBABILITY = 0.5
CONFIDENT_PROBABILITY = 0.8


def sampling_starts(speech: Sequence[tuple[float, float]], duration: float, count: int = SAMPLE_WINDOWS,
                    seconds: float = SAMPLE_SECONDS) -> list[float]:
    """Start times of ``count`` windows spread evenly over the SPEECH of a recording (each window fits inside the file).

    "Language: Auto" used to trust the first minutes only. A lecture that opens with small talk in another language was then
    decoded in that language from start to end: measured on a real recording (English lecture, Hungarian chat first), 100 % of the
    English speech came out as Hungarian word salad with decoder loops.
    """
    if duration <= 0 or count <= 0:
        return []
    last_start = max(duration - seconds, 0.0)
    total = sum(max(b - a, 0.0) for a, b in speech)
    if total <= seconds:                                  # no (usable) VAD: spread over the whole file
        return [round(min(last_start * (i + 0.5) / count, last_start), 2) for i in range(count)]
    starts: list[float] = []
    for i in range(count):
        target = total * (i + 0.5) / count
        walked = 0.0
        for a, b in speech:
            if walked + (b - a) >= target:
                starts.append(round(min(a + (target - walked), last_start), 2))
                break
            walked += b - a
    return starts


def summarize_language_samples(samples: Sequence[tuple[float, str, float]], probe: Optional[tuple[str, float]] = None) -> dict:
    """Which language(s) are spoken, from ``(start, language, probability)`` samples taken across the whole recording.

    ``mixed`` (decode every 30 s window in its own language) when a second language is confidently present, when two or more
    windows do not belong to the dominant language, or when the first minutes (``probe``) disagree with the rest of the file.
    A single unsure window in an otherwise one-language lecture does not make it mixed.
    """
    rows = [(float(s), str(lang), float(p)) for s, lang, p in samples]
    confident = [(s, lang, p) for s, lang, p in rows if p >= SAMPLE_MIN_PROBABILITY]
    counts = Counter(lang for _s, lang, _p in confident)
    if not rows or not counts:
        return {"windows": len(rows), "counts": {}, "dominant": None, "dominant_share": 0.0, "mixed": False, "reason": "no usable samples",
                "samples": [[round(s, 1), lang, round(p, 3)] for s, lang, p in rows]}
    dominant, dominant_count = counts.most_common(1)[0]
    deviating = [(s, lang, p) for s, lang, p in rows if not (lang == dominant and p >= SAMPLE_MIN_PROBABILITY)]
    sure_others = [(s, lang, p) for s, lang, p in deviating if lang != dominant and p >= CONFIDENT_PROBABILITY]
    reasons = []
    if sure_others:
        reasons.append(f"{len(sure_others)} window(s) clearly in {', '.join(sorted({lang for _s, lang, _p in sure_others}))}")
    if len(deviating) >= 2:
        reasons.append(f"{len(deviating)} of {len(rows)} windows are not {dominant}")
    if probe and probe[0] != dominant and probe[1] >= CONFIDENT_PROBABILITY:
        reasons.append(f"the first minutes are {probe[0]} but the recording is mostly {dominant}")
    return {"windows": len(rows), "counts": dict(counts), "dominant": dominant, "dominant_share": round(dominant_count / len(rows), 3),
            "deviating": [[round(s, 1), lang, round(p, 3)] for s, lang, p in deviating], "mixed": bool(reasons),
            "reason": "; ".join(reasons) or f"{dominant_count} of {len(rows)} windows are {dominant}",
            "samples": [[round(s, 1), lang, round(p, 3)] for s, lang, p in rows]}


class LanguageError(ValueError):
    """Unsupported or malformed language argument."""


_EN_STOPWORDS = frozenset(
    "the and of to is in that it for with as this are on be by from or an at which not can has have was were "
    "will we you they there their what when because also then these those would should could about into".split())
_TR_STOPWORDS = frozenset(
    "ve bir bu da de için ile çok gibi daha olarak olan ama ne var mi mı mu mü ki şu ben biz siz yani işte "
    "çünkü ise kadar sonra hani değil bunu şey her ya hem veya olduğu oluyor olur yok evet hayır".split())


def estimate_language_mix(text: str, min_hits: int = 20, minority_share: float = 0.15) -> dict:
    """Rough English/Turkish mix from stop-word counts (a hint for metadata, not a detector).

    ``mixed`` is true when both languages contribute at least ``minority_share`` of the stop-word
    hits and there is enough evidence (``min_hits``).
    """
    words = [w.casefold() for w in text.replace("İ", "i").split()]
    en = sum(1 for w in words if w.strip(".,;:!?()\"'") in _EN_STOPWORDS)
    tr = sum(1 for w in words if w.strip(".,;:!?()\"'") in _TR_STOPWORDS)
    total = en + tr
    en_share = en / total if total else 0.0
    return {
        "english_stopwords": en, "turkish_stopwords": tr,
        "english_share": round(en_share, 3), "turkish_share": round(1 - en_share, 3) if total else 0.0,
        "mixed": total >= min_hits and min(en_share, 1 - en_share) >= minority_share,
        "dominant": None if total < min_hits else ("en" if en >= tr else "tr"),
    }


@dataclass(frozen=True)
class LanguageDecision:
    requested: str                 # what the user asked for ("auto", "en", "tr", ...)
    forced: Optional[str]          # code passed to the backend (None => auto-detect)
    detected: Optional[str] = None
    probability: Optional[float] = None


def normalize_language(value: Optional[str]) -> Optional[str]:
    """Map CLI input to a Whisper code. ``auto``/``None`` => ``None`` (auto-detect)."""
    if value is None:
        return None
    cleaned = value.strip().lower()
    if cleaned in ("", AUTO):
        return None
    aliases = {"english": "en", "turkish": "tr", "türkçe": "tr", "turkce": "tr", "ingilizce": "en"}
    cleaned = aliases.get(cleaned, cleaned)
    if not _CODE_RE.match(cleaned):
        raise LanguageError(f"Invalid language '{value}'. Use 'auto', 'en', 'tr' or another ISO 639-1 code.")
    return cleaned


def decide(requested: Optional[str], model_default: Optional[str] = None) -> LanguageDecision:
    """Combine the CLI value with a model's ``default_language`` (e.g. Turkish medical model)."""
    forced = normalize_language(requested)
    shown = forced or AUTO
    if forced is None and model_default:
        return LanguageDecision(requested=shown, forced=model_default)
    return LanguageDecision(requested=shown, forced=forced)
