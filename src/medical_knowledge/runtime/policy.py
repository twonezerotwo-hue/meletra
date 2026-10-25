"""The correction policy: when is changing the transcript JUSTIFIED?

The assumption is NO CHANGE. A correction has to beat the original text, pay its cost, beat every other candidate by a clear margin and be backed by several
INDEPENDENT kinds of evidence. Nothing here looks at a word list of "protected words": the protection comes from how real the heard words look
(``validity.py``), how sure the recogniser was, and how much unrelated evidence supports the replacement.

    keep      the score of the explicit candidate "keep the original text" (0..1): high when the changed words are real words, when the recogniser was sure, when the
              word is a capitalised proper name; low for a non-word the recogniser doubted
    cost      what a change costs: spelling and sound distance, a multi-word edit, a confident recogniser, a frequent original word, and the absence of any support
    adjusted  candidate score - cost; HIGH needs ``adjusted - keep`` >= a margin as well as an absolute level

Independent evidence families (a HIGH correction needs FORM plus at least two others, MEDIUM FORM plus one):

    form        sound AND spelling are close (or the knowledge base records this very mishearing)
    invalid     the words that would change are not real words (graded validity <= 0.25)
    context     a cue word of the candidate nearby, or at least TWO distinct known terms linked to it by the ontology (one relation alone is not evidence)
    domain      the candidate's field is the lecture's field AND the topic classification is confident
    lecture     the slides / the user's own vocabulary name it, or the lecturer already said it several times
    recorded    the knowledge base lists this exact mishearing and there is some context
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Sequence

from ...medical.lexicon import stem
from ..normalize.text import normalize

DOMAIN_CONFIDENCE_MIN = 0.40

FORM, INVALID, CONTEXT, DOMAIN, LECTURE, RECORDED = "form", "invalid", "context", "domain", "lecture", "recorded"


@dataclass
class Assessment:
    keep: float = 0.0
    cost: float = 0.0
    adjusted: float = 0.0
    keep_margin: float = 0.0
    changed_validity: float = 1.0
    families: list[str] = field(default_factory=list)
    parts: dict[str, float] = field(default_factory=dict)

    @property
    def other_families(self) -> int:
        return len([f for f in self.families if f != FORM])

    def to_dict(self) -> dict:
        return {"keep_original": round(self.keep, 3), "correction_cost": round(self.cost, 3), "adjusted_score": round(self.adjusted, 3), "margin_over_original": round(self.keep_margin, 3),
                "changed_word_validity": round(self.changed_validity, 3), "evidence": list(self.families), "parts": {k: round(v, 3) for k, v in self.parts.items()}}


def changed_indices(heard: Sequence[str], term: Sequence[str]) -> list[int]:
    """Indices of the heard words that a replacement by ``term`` would remove or change (the words that survive unchanged are not touched)."""
    hs, ts = [stem(w) for w in heard], [stem(w) for w in term]
    out: list[int] = []
    for tag, i1, i2, _j1, _j2 in difflib.SequenceMatcher(None, hs, ts, autojunk=False).get_opcodes():
        if tag in ("replace", "delete"):
            out.extend(range(i1, i2))
    return out or list(range(len(heard)))


def changed_validity(word_validity: Sequence[float], heard: Sequence[str], surface: str) -> float:
    """Letter-weighted mean validity of the heard words that would change."""
    if not word_validity:
        return 0.0
    term = normalize(surface).split()
    idx = [i for i in changed_indices(heard, term) if i < len(word_validity)] or list(range(len(word_validity)))
    weights = [max(1, len(heard[i])) for i in idx]
    return sum(word_validity[i] * w for i, w in zip(idx, weights)) / sum(weights)


def keep_original(signals: dict, changed_validity_value: float) -> tuple[float, dict]:
    """The score of the candidate "keep the original": 0.10 floor + the realness of the words that would change + a sure recogniser + a capitalised proper name."""
    asr_sure = float(signals.get("asr_sure", 0.0))
    proper = float(signals.get("capitalised", 0.0))
    parts = {"realness": 0.62 * changed_validity_value, "recogniser_sure": 0.16 * asr_sure, "proper_name": 0.08 * proper}
    return min(1.0, 0.10 + sum(parts.values())), parts


def correction_cost(signals: dict, n_changed: int, scale: float) -> tuple[float, dict]:
    domain = float(signals.get("domain", 0.0)) if signals.get("domain_confidence", 0.0) >= DOMAIN_CONFIDENCE_MIN else 0.0
    support = max(domain, float(signals.get("lecture", 0.0)), float(signals.get("context", 0.0)))
    parts = {"base": 0.02, "spelling": 0.10 * (1.0 - float(signals["lexical"])), "sound": 0.08 * (1.0 - float(signals["phonetic"])), "multi_word": 0.04 * max(0, n_changed - 1),
             "recogniser_sure": 0.05 * float(signals.get("asr_sure", 0.0)), "frequent_original": 0.05 * float(signals.get("orig_freq", 0.0)), "no_support": 0.06 * (1.0 - support),
             "other_role": 0.08 * (1.0 - float(signals.get("pos", 1.0))) if float(signals.get("pos", 1.0)) < 0.6 else 0.0,
             "other_kind": 0.08 * (1.0 - float(signals.get("type_fit", 1.0))) if float(signals.get("type_fit", 1.0)) < 0.6 else 0.0}
    return scale * sum(parts.values()), parts


def evidence_families(signals: dict, known_confusion: bool, changed_validity_value: float, min_phonetic: float, min_lexical: float) -> list[str]:
    fam: list[str] = []
    if (signals["phonetic"] >= min_phonetic and signals["lexical"] >= min_lexical) or (known_confusion and signals["phonetic"] >= min_phonetic):
        fam.append(FORM)
    if changed_validity_value <= 0.25:
        fam.append(INVALID)
    if signals.get("cue", 0.0) >= 0.35 or signals.get("link_count", 0.0) >= 2 or (signals.get("cue", 0.0) >= 0.2 and signals.get("ontology", 0.0) >= 0.3):
        fam.append(CONTEXT)
    if signals.get("domain", 0.0) >= 0.7 and signals.get("domain_confidence", 0.0) >= DOMAIN_CONFIDENCE_MIN:
        fam.append(DOMAIN)
    if signals.get("lecture", 0.0) >= 0.6:
        fam.append(LECTURE)
    if known_confusion and signals.get("context", 0.0) >= 0.2:
        fam.append(RECORDED)
    return fam
