"""Local vs Deepgram comparison, medical disagreement report and a reporting-only consensus.

For every local segment the Deepgram words that fall in the same time window are diffed word by word. Differences
that involve a dictionary term (exactly or approximately) become *medical disagreements*; terms both engines wrote
identically and the dictionary/slides confirm are HIGH_CONFIDENCE. Nothing is ever overwritten: every transcript stays
exactly as its engine produced it.

Status vocabulary
    HIGH_CONFIDENCE   local == deepgram == dictionary term (and, when slides exist, the slides agree too)
    REVIEW_REQUIRED   the engines disagree, one omitted the term, or the spelling matches no source - a human decides
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Optional, Sequence

from .schema import Segment
from .vocabulary import VocabularyIndex, normalize_key, tokenize

try:
    from rapidfuzz import fuzz, process
except ImportError:  # pragma: no cover
    fuzz = process = None
import difflib

HIGH, REVIEW = "HIGH_CONFIDENCE", "REVIEW_REQUIRED"
WINDOW_PAD = 0.6                     # seconds of slack when matching Deepgram words to a local segment
NEAR_TERM = 0.80                     # similarity for "this text looks like a dictionary term"
MAX_SPAN_WORDS = 6


@dataclass
class Disagreement:
    start: float
    kind: str                         # spelling | omitted_by_deepgram | omitted_by_local
    local: str
    deepgram: str
    dictionary_term: Optional[str]
    dictionary_similarity: float
    in_pdf: bool
    candidate: Optional[str]
    sources_for_candidate: list[str]
    status: str = REVIEW
    context: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"timestamp": round(self.start, 2), "kind": self.kind, "local": self.local, "deepgram": self.deepgram,
                "medical_dictionary": self.dictionary_term, "dictionary_similarity": round(self.dictionary_similarity, 3),
                "pdf_has_term": self.in_pdf, "consensus_candidate": self.candidate,
                "candidate_supported_by": self.sources_for_candidate, "status": self.status, "context": self.context}


@dataclass
class ConsensusTerm:
    term: str
    count: int
    first_start: float
    sources: list[str]
    status: str = HIGH

    def to_dict(self) -> dict[str, Any]:
        return {"term": self.term, "count": self.count, "first_start": round(self.first_start, 2),
                "sources": self.sources, "status": self.status}


@dataclass
class Comparison:
    statistics: dict[str, Any] = field(default_factory=dict)
    sentence_differences: list[dict[str, Any]] = field(default_factory=list)
    term_differences: dict[str, Any] = field(default_factory=dict)
    disagreements: list[Disagreement] = field(default_factory=list)
    consensus: list[ConsensusTerm] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"statistics": self.statistics, "term_differences": self.term_differences,
                "disagreements": [d.to_dict() for d in self.disagreements],
                "consensus_terms": [c.to_dict() for c in self.consensus],
                "sentence_differences": self.sentence_differences}


def _norm_tokens(text: str) -> list[tuple[str, str]]:
    return [(normalize_key(w) or w.casefold(), w) for w in tokenize(text)]


def _deepgram_window(dg: Sequence[Segment], start: float, end: float) -> str:
    """Deepgram text for [start, end]: words whose midpoint falls inside (utterance text as a fallback)."""
    picked: list[str] = []
    for seg in dg:
        if seg.end < start - 5 or seg.start > end + 5:
            continue
        if seg.words:
            picked += [w.word for w in seg.words if start - WINDOW_PAD <= (w.start + w.end) / 2 <= end + WINDOW_PAD]
        elif seg.end > start and seg.start < end:
            picked.append(seg.text)
    return " ".join(picked)


def _nearest_term(key: str, keys: list[str], index: VocabularyIndex) -> tuple[Optional[str], float]:
    """Dictionary term most similar to the letters in ``key`` (exact match first)."""
    if len(key) < 4:
        return None, 0.0
    exact = index.term_keys.get(key)
    if exact:
        return exact, 1.0
    if process is not None:
        hit = process.extractOne(key, keys, scorer=fuzz.ratio, score_cutoff=NEAR_TERM * 100)
        return (index.term_keys[hit[0]], hit[1] / 100.0) if hit else (None, 0.0)
    close = difflib.get_close_matches(key, keys, n=1, cutoff=NEAR_TERM)
    return (index.term_keys[close[0]], difflib.SequenceMatcher(None, key, close[0]).ratio()) if close else (None, 0.0)


def compare_transcripts(
    local: Sequence[Segment], deepgram: Sequence[Segment], vocab: VocabularyIndex,
    pdf_terms: Sequence[str] = (), pdf_text: str = "", max_sentence_diffs: int = 200,
) -> Comparison:
    keys = list(vocab.term_keys)
    pdf_keys = {normalize_key(t) for t in pdf_terms}
    pdf_blob = normalize_key(pdf_text)

    def term_in_pdf(term: str) -> bool:
        key = normalize_key(term)
        return key in pdf_keys or (len(key) >= 7 and key in pdf_blob)

    matches = ref_tokens = hyp_tokens = 0
    disagreements: list[Disagreement] = []
    sentence_diffs: list[dict[str, Any]] = []
    agreed: dict[str, ConsensusTerm] = {}
    local_terms, dg_terms = Counter(), Counter()

    for seg in local:
        l_pairs = _norm_tokens(seg.text)
        d_text = _deepgram_window(deepgram, seg.start, seg.end)
        d_pairs = _norm_tokens(d_text)
        l_norm, d_norm = [p[0] for p in l_pairs], [p[0] for p in d_pairs]
        ref_tokens += len(l_norm)
        hyp_tokens += len(d_norm)
        local_terms.update(vocab.find_terms(seg.text))
        dg_terms.update(vocab.find_terms(d_text))
        if not l_norm:
            continue
        matcher = SequenceMatcher(None, l_norm, d_norm, autojunk=False)
        seg_matches = 0
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                seg_matches += i2 - i1
                continue
            a_words = [p[1] for p in l_pairs[i1:i2]][:MAX_SPAN_WORDS]
            b_words = [p[1] for p in d_pairs[j1:j2]][:MAX_SPAN_WORDS]
            entry = _medical_difference(seg, a_words, b_words, keys, vocab, term_in_pdf, d_text)
            if entry is not None:
                disagreements.append(entry)
        matches += seg_matches
        similarity = seg_matches / max(len(l_norm), len(d_norm), 1)
        if similarity < 0.85 and d_norm:
            sentence_diffs.append({"timestamp": round(seg.start, 2), "similarity": round(similarity, 3),
                                   "local": seg.text.strip(), "deepgram": d_text.strip()})

        # terms both engines wrote identically
        d_blob = normalize_key(d_text)
        for begin, _end, term in vocab.find_term_spans([p[1] for p in l_pairs]):
            if normalize_key(term) in d_blob:
                sources = ["local", "deepgram", "dictionary"] + (["pdf"] if term_in_pdf(term) else [])
                got = agreed.setdefault(term, ConsensusTerm(term, 0, seg.start, sources))
                got.count += 1

    # a term that also appears in a disagreement is not HIGH_CONFIDENCE for that occurrence (kept as agreed otherwise)
    only_local = {t: n for t, n in local_terms.items() if dg_terms.get(t, 0) == 0}
    only_dg = {t: n for t, n in dg_terms.items() if local_terms.get(t, 0) == 0}
    shared = {t: [local_terms[t], dg_terms[t]] for t in local_terms if dg_terms.get(t, 0) > 0}
    sentence_diffs.sort(key=lambda d: d["similarity"])
    return Comparison(
        statistics={
            "local_segments": len(local), "deepgram_segments": len(deepgram), "local_tokens": ref_tokens,
            "deepgram_tokens_in_windows": hyp_tokens, "matching_tokens": matches,
            "token_agreement": round(matches / max(ref_tokens, 1), 4),
            "estimated_difference_rate_local_vs_deepgram": round(1 - matches / max(ref_tokens, 1), 4),
            "medical_disagreements": len(disagreements), "high_confidence_terms": len(agreed),
            "review_required": len(disagreements),
        },
        sentence_differences=sentence_diffs[:max_sentence_diffs],
        term_differences={"only_local": only_local, "only_deepgram": only_dg, "shared": shared},
        disagreements=sorted(disagreements, key=lambda d: d.start),
        consensus=sorted(agreed.values(), key=lambda c: (-c.count, c.term.casefold())),
    )


def _medical_difference(seg: Segment, a_words: list[str], b_words: list[str], keys: list[str], vocab: VocabularyIndex,
                        term_in_pdf: Any, deepgram_text: str = "") -> Optional[Disagreement]:
    a_text, b_text = " ".join(a_words), " ".join(b_words)
    a_key, b_key = normalize_key(a_text), normalize_key(b_text)
    term_a, ratio_a = _nearest_term(a_key, keys, vocab) if a_key else (None, 0.0)
    term_b, ratio_b = _nearest_term(b_key, keys, vocab) if b_key else (None, 0.0)
    if term_a is None and term_b is None:
        return None
    if a_text and not b_text:
        if ratio_a < 0.95:
            return None
        kind, term, ratio = "omitted_by_deepgram", term_a, ratio_a
    elif b_text and not a_text:
        if ratio_b < 0.95:
            return None
        kind, term, ratio = "omitted_by_local", term_b, ratio_b
    else:
        kind = "spelling"
        term, ratio = (term_a, ratio_a) if ratio_a >= ratio_b else (term_b, ratio_b)
        if normalize_key(a_text) == normalize_key(b_text):
            return None
    assert term is not None
    t_key = normalize_key(term)
    supporters = ["dictionary"]
    if term_in_pdf(term):
        supporters.append("pdf")
    # who wrote the dictionary term: judged on the whole sentence, not only on the words that differ
    # ("cortical spinal" vs "corticospinal" differ, but Deepgram's sentence does contain "corticospinal tract")
    local_blob, deepgram_blob = normalize_key(seg.text), normalize_key(deepgram_text)
    if a_key == t_key or t_key in local_blob:
        supporters.append("local")
    if b_key == t_key or t_key in deepgram_blob:
        supporters.append("deepgram")
    return Disagreement(
        start=seg.start, kind=kind, local=a_text, deepgram=b_text, dictionary_term=term, dictionary_similarity=ratio,
        in_pdf=term_in_pdf(term), candidate=term if len(supporters) >= 2 else None, sources_for_candidate=supporters,
        context=seg.text.strip()[:140])


def disagreement_report_text(comparison: Comparison) -> str:
    from .exporters import format_hms

    s = comparison.statistics
    out = ["MEDICAL DISAGREEMENT REPORT - Local Whisper large-v3 vs Deepgram Nova-3 Medical",
           "Nothing was overwritten: both transcripts are untouched. Candidates are suggestions for a human reviewer.", "",
           f"token agreement      : {s.get('token_agreement')}",
           f"medical disagreements: {s.get('medical_disagreements')}",
           f"HIGH_CONFIDENCE terms: {s.get('high_confidence_terms')}", ""]
    if not comparison.disagreements:
        out.append("(no medical disagreements found)")
    for number, d in enumerate(comparison.disagreements, start=1):
        out += [f"[{number}] {format_hms(d.start)}  {d.kind}  -> {d.status}",
                f"  Local   : {d.local or '(omitted)'}",
                f"  Deepgram: {d.deepgram or '(omitted)'}",
                f"  Medical dictionary: {d.dictionary_term} (similarity {d.dictionary_similarity:.2f})",
                f"  PDF has the term  : {'yes' if d.in_pdf else 'no'}",
                f"  Consensus candidate: {d.candidate or '(none)'}  supported by: {', '.join(d.sources_for_candidate)}",
                f"  Context: {d.context}", ""]
    td = comparison.term_differences
    out += ["TERMS ONLY IN LOCAL:"] + [f"  {t} x{n}" for t, n in sorted(td.get("only_local", {}).items())] \
        + ["TERMS ONLY IN DEEPGRAM:"] + [f"  {t} x{n}" for t, n in sorted(td.get("only_deepgram", {}).items())]
    return "\n".join(out) + "\n"


def consensus_text(comparison: Comparison) -> str:
    from .exporters import format_hms

    out = ["MEDICAL CONSENSUS (reporting only)", "",
           "HIGH_CONFIDENCE = local, Deepgram and the medical dictionary (and the slides, if any) wrote the same term.", "",
           f"{'status':<16} {'count':>5} {'first':<9} term  [sources]"]
    for c in comparison.consensus:
        out.append(f"{c.status:<16} {c.count:>5} {format_hms(c.first_start):<9} {c.term}  [{', '.join(c.sources)}]")
    out += ["", "REVIEW_REQUIRED"]
    for d in comparison.disagreements:
        out.append(f"  {format_hms(d.start)}  local='{d.local or '-'}' deepgram='{d.deepgram or '-'}' "
                   f"-> candidate '{d.candidate or 'none'}' ({', '.join(d.sources_for_candidate)})")
    return "\n".join(out) + "\n"
