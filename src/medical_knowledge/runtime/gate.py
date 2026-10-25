"""The confidence gate: FALSE CORRECTIONS ARE MORE EXPENSIVE THAN MISSED CORRECTIONS.

    HIGH     auto-correct (in v4 shadow mode: recorded as the v4 decision). Almost certainly right.
    MEDIUM   plausible but not safe: advisory only, never applied (a targeted re-decode of that audio window may confirm it)
    LOW      the original text stays

The assumption is NO CHANGE (policy.py). A candidate reaches HIGH only when ALL of these hold:

    * its score MINUS the correction cost clears an absolute level;
    * it beats the explicit candidate "KEEP THE ORIGINAL TEXT" by a clear margin (real words are protected by their realness, not by a word list);
    * it beats the second-best other candidate by a clear margin (0.94 vs 0.91 is ambiguous, 0.96 vs 0.61 is not);
    * form (sound + spelling) plus at least TWO other independent kinds of evidence agree (context, field of study, lecture material, a recorded mishearing,
      a non-word original); a single fuzzy match is never enough;
    * no guard objects: plain English phrases, very short sounds, alignment (the change must correct a mishearing, not edit valid text), siblings listed in
      the same sentence, rare dictionary words, identifiers, ambiguous abbreviations.
"""

from __future__ import annotations

import difflib
import os
import re
from dataclasses import dataclass, field
from typing import Optional

from rapidfuzz.distance import Levenshtein

from ...common_words import is_common_word
from ...medical.corrector import _plain
from ...medical.lexicon import stem
from ...medical.phonetics import skeleton
from ..normalize.morphology import SUFFIXES
from ..normalize.text import normalize, wording_key
from .policy import FORM, Assessment, changed_indices, changed_validity, correction_cost, evidence_families, keep_original
from .retrieve import sound_similarity
from .score import TRUST_ADJUST, Scored
from .spans import Span

HIGH, MEDIUM, LOW = "high", "medium", "low"
WORD_ENDINGS = frozenset(SUFFIXES) | {""}                                  # endings that make a different derived word (morphology.py); a one-letter spelling slip is not among them


@dataclass
class GateConfig:
    # thresholds on the ADJUSTED score (candidate score - correction cost). Chosen on the development and validation corpora (tools/tune_gate.py): the strictest
    # setting that does not give up the recall of the current engine; heldout_v2 was not used.
    high: float = 0.62
    medium: float = 0.50
    # the candidate must beat "keep the original" by this much (adjusted score - keep score)
    high_keep_margin: float = 0.25
    medium_keep_margin: float = 0.15
    # ... and the second-best OTHER candidate by this much (0.96 vs 0.61 is a strong candidate, 0.94 vs 0.91 is ambiguous)
    high_candidate_margin: float = 0.10
    medium_candidate_margin: float = 0.05
    # independent evidence families besides form
    high_other_families: int = 2
    medium_other_families: int = 1
    cost_scale: float = 1.0
    rare_word_lexical: float = 0.9
    min_phonetic_high: float = 0.72
    min_phonetic_medium: float = 0.62
    min_lexical_high: float = 0.55          # the spelling must be recognisably close as well: sound alone ("innervate" ~ "end-feet") is not enough
    min_lexical_medium: float = 0.45
    common_phonetic: float = 0.92           # a span of plain English words: the sound must be (almost) identical ...
    common_lexical: float = 0.70            # ... the spelling close ...
    common_context: float = 0.55            # ... and the context medical, unless the knowledge base lists the span as a mishearing
    max_share: float = 0.04                 # corrections above this share of the words: raise the bar (calibration safety net)
    max_per_segment: int = 4


@dataclass
class Decision:
    span: Span
    tier: str
    best: Optional[Scored]
    runner_up: Optional[Scored]
    ranked: list[Scored] = field(default_factory=list)
    blocked_by: list[str] = field(default_factory=list)
    assessment: Assessment = field(default_factory=Assessment)
    candidate_margin: float = 0.0
    reason: str = ""
    near_high: bool = False                      # MEDIUM that passes every HIGH condition except the number of independent evidence families / the margin over the original:
                                                 # tracked separately (never applied) so that a blind benchmark can show whether it is safe
    pool: list = field(default_factory=list)       # every retrieved candidate: (surface, concept id, prior), best prior first

    @property
    def confidence(self) -> float:
        return self.best.score if self.best else 0.0


def alignment_problem(span: Span, surface: str) -> Optional[str]:
    """Why replacing the span by the term would not be a CORRECTION of a mishearing (None = it would be)."""
    heard = span.words or [w for t in span.tokens for w in normalize(t.text).split()]
    unknown = span.word_unknown or [not (_plain(w) or is_common_word(w)) for w in heard]
    term = normalize(surface).split()
    hs, ts = [stem(w) for w in heard], [stem(w) for w in term]
    if hs == ts:
        return "same words (number / case only)"
    if len(hs) < len(ts):
        it = iter(ts)
        if all(any(h == x for x in it) for h in hs):
            return "the span is an incomplete mention of the term, not a mishearing"
    if len(hs) > len(ts) and any(h in ts for h in hs):                    # some heard words ARE the term's words: the others would be dropped
        extra = [k for k, h in enumerate(hs) if h not in ts]
        if any(not unknown[k] for k in extra):
            return "a real word would be dropped"
    if len(heard) == 1 and len(heard[0]) >= len(compact_words(term)) + 2 and compact_words(term) in heard[0]:
        return "the term is inside the word: it would delete letters"
    if len(heard) == 1 and len(term) == 1 and len(heard[0]) >= 8 and len(term[0]) >= len(heard[0]) + 3 and len(os.path.commonprefix([heard[0], term[0]])) >= len(heard[0]) - 1:
        return "the heard word is the beginning of the term: a different (derived) word, not a mishearing"
    if len(heard) == 1 and len(term) == 1 and heard[0] != term[0]:
        common = len(os.path.commonprefix([heard[0], term[0]]))
        rest_h, rest_t = heard[0][common:], term[0][common:]
        # the same long stem with two DIFFERENT word endings ("embol-ize" / "embol-us", "trachea" / "trachea-lis"): two derived words, not one word misheard
        if common >= 5 and common >= 0.6 * min(len(heard[0]), len(term[0])) and rest_h in WORD_ENDINGS and rest_t in WORD_ENDINGS and (rest_h or rest_t):
            return f"the same stem '{heard[0][:common]}' with a different ending ({rest_h or '-'} / {rest_t or '-'}): a different derived word, not a mishearing"
    if len(heard) == 1 and len(term) >= 2:
        # the heard word contains one word of the term ("tachypolymerase" ~ "DNA polymerase"): the REST of the heard word must sound like the rest of the term
        for k, tw in enumerate(term):
            if len(tw) >= 5 and tw in heard[0] and tw != heard[0]:
                rest_h = heard[0].replace(tw, "", 1)
                rest_t = "".join(w for i, w in enumerate(term) if i != k)
                if rest_h and sound_similarity(rest_h, rest_t) < 0.55 and Levenshtein.normalized_similarity(rest_h, rest_t) < 0.5:
                    return f"the heard word contains '{tw}' but the rest of it ({rest_h}) does not sound like the rest of the term ({rest_t})"
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, hs, ts, autojunk=False).get_opcodes():
        if tag == "replace":
            h_block, t_block = " ".join(heard[i1:i2]), " ".join(term[j1:j2])
            if sound_similarity(h_block, t_block) < 0.55 or Levenshtein.normalized_similarity("".join(h_block.split()), "".join(t_block.split())) < 0.35:
                return f"a changed word sounds nothing like its replacement ({h_block} -> {t_block})"
    return None


def strip_qualifier(surface: str) -> str:
    """"Phosphoenolpyruvate Carboxykinase (GTP)" -> "phosphoenolpyruvate carboxykinase": a parenthetical qualifier is never spoken."""
    return " ".join(re.sub(r"\s*\([^)]*\)", "", surface).lower().split())


def compact_words(ws: list[str]) -> str:
    return "".join(ws)


def sibling(span: Span, scored: Scored, ctx) -> bool:
    """The candidate is already named (exactly) in the same sentence: the speaker is listing OTHER things, so the odd word is probably not that term again."""
    seg = ctx.segments[span.segment]
    return any(scored.candidate.concept_id in h.concepts and not (h.first < span.last and span.first < h.last) for h in seg.hits if not h.common)


def decide(span: Span, ranked: list[Scored], cfg: GateConfig) -> Decision:
    ranked = [r for r in ranked if not r.candidate.exact]
    ranked.sort(key=lambda r: -r.score)
    if not ranked:
        return Decision(span, LOW, None, None, [], ["no candidate"], reason="no candidate")
    best = ranked[0]
    best_key = wording_key(strip_qualifier(best.candidate.surface))
    # the second best is a DIFFERENT word: another concept AND another wording (a plural, a British spelling or a synonym spelled y / i of the best is the same answer, not an alternative)
    runner = next((r for r in ranked[1:] if r.candidate.concept_id != best.candidate.concept_id and wording_key(strip_qualifier(r.candidate.surface)) != best_key), None)
    c, sig = best.candidate, best.signals
    blocked: list[str] = []
    adjust = TRUST_ADJUST.get(c.trust, 0.0) + (0.05 if c.kind in ("drug", "gene") else 0.0)
    high_thr, medium_thr = cfg.high + adjust, cfg.medium + adjust
    sk_len = len(skeleton(span.text))

    if span.common_only and not c.known_confusion:
        if not (sig["phonetic"] >= cfg.common_phonetic and sig["lexical"] >= cfg.common_lexical and sig["context"] >= cfg.common_context and sig["domain"] >= 0.5):
            blocked.append("plain English words: needs a recorded mishearing or near-identical sound + spelling + medical context")
    if sk_len < 4 and not c.known_confusion and sig["lecture"] < 0.9:
        blocked.append("sound too short to say anything")
    if span.n == 1 and not span.reasons == ["confusion"] and c.token_count >= 3:
        blocked.append("one word is not replaced by a three-word term")
    if sig["phonetic"] < cfg.min_phonetic_medium:
        blocked.append(f"sound {sig['phonetic']:.2f}")
    if sig["lexical"] < cfg.min_lexical_medium and not c.known_confusion:
        blocked.append(f"spelling {sig['lexical']:.2f}")
    problem = alignment_problem(span, c.surface)
    if problem:
        blocked.append(problem)
    if sig.get("sibling") and not c.known_confusion:
        blocked.append("the term is already named in this sentence")
    rare = span.word_rare
    if rare and span.word_unknown and not c.known_confusion and all(r for r, u in zip(rare, span.word_unknown) if u) and any(span.word_unknown) \
            and sig["lexical"] < cfg.rare_word_lexical and sig["lecture"] < 0.9:
        blocked.append("a real (rare) dictionary word: replaced only by a near-identical spelling")

    # ---- the policy numbers
    heard = span.words or [w for t in span.tokens for w in normalize(t.text).split()]
    cv = changed_validity(span.word_validity or [], heard, c.surface) if span.word_validity else (0.0 if span.unknown_tokens else 0.9)
    keep, keep_parts = keep_original(sig, cv)
    n_changed = len(changed_indices(heard, normalize(c.surface).split()))
    cost, cost_parts = correction_cost(sig, n_changed, cfg.cost_scale)
    adjusted = best.score - cost
    fam_high = evidence_families(sig, c.known_confusion, cv, cfg.min_phonetic_high, cfg.min_lexical_high)
    fam_medium = evidence_families(sig, c.known_confusion, cv, cfg.min_phonetic_medium, cfg.min_lexical_medium)
    margin = best.score - runner.score if runner is not None else 1.0
    keep_margin = adjusted - keep
    parts = {**{"keep:" + k: v for k, v in keep_parts.items()}, **{"cost:" + k: v for k, v in cost_parts.items()}}
    assessment = Assessment(keep, cost, adjusted, keep_margin, cv, fam_high, parts)

    why: list[str] = []
    tier = LOW
    if not blocked:
        high_ok = (FORM in fam_high and len(fam_high) - 1 >= cfg.high_other_families and adjusted >= high_thr
                   and keep_margin >= cfg.high_keep_margin and margin >= cfg.high_candidate_margin)
        medium_ok = (FORM in fam_medium and len(fam_medium) - 1 >= cfg.medium_other_families and adjusted >= medium_thr
                     and keep_margin >= cfg.medium_keep_margin and margin >= cfg.medium_candidate_margin)
        if high_ok:
            tier = HIGH
        elif medium_ok:
            tier = MEDIUM
        if tier == HIGH and "collision" in span.reasons and sig["context"] < 0.6:
            tier = MEDIUM
            why.append("a sound collision between plain words needs strong context")
        if tier == HIGH and (sig.get("pos", 1.0) == 0.0 or sig.get("type_fit", 1.0) == 0.0) and not c.known_confusion:
            tier = MEDIUM
            why.append("the replacement has another grammatical role (verb / noun / adjective) or another kind of meaning than the sentence asks for")
        if c.kind == "gene" and not c.known_confusion and sig["lecture"] < 0.6 and tier == HIGH:
            tier = MEDIUM
            why.append("a gene / protein symbol is rarely heard by accident: needs lecture evidence")
        if tier != HIGH:
            if FORM not in fam_high:
                why.append("sound + spelling not close enough")
            others = [f for f in fam_high if f != FORM]
            if len(others) < cfg.high_other_families:
                why.append(f"independent evidence {len(others)} of {cfg.high_other_families} needed besides form ({', '.join(others) or 'none'})")
            if adjusted < high_thr:
                why.append(f"adjusted score {adjusted:.2f} < {high_thr:.2f}")
            if keep_margin < cfg.high_keep_margin:
                why.append(f"margin over keeping the original {keep_margin:.2f} < {cfg.high_keep_margin:.2f}")
            if margin < cfg.high_candidate_margin and runner is not None:
                why.append(f"margin to second best {margin:.2f} < {cfg.high_candidate_margin:.2f} ({runner.candidate.preferred_name})")
    near_high = (tier == MEDIUM and not blocked and FORM in fam_high and adjusted >= high_thr and margin >= cfg.high_candidate_margin
                 and sig.get("pos", 1.0) > 0.0 and sig.get("type_fit", 1.0) > 0.0)
    if tier == HIGH:
        reason = "passed: " + ", ".join(fam_high) + f"; adjusted {adjusted:.2f}, over the original +{keep_margin:.2f}, over second best +{margin:.2f}"
    else:
        reason = "; ".join(blocked + why)
    return Decision(span, tier, best, runner, ranked[:5], blocked + why, assessment, margin, reason, near_high)
