"""Why did the engine miss, or wrongly change, something? - the diagnosis behind the benchmark numbers.

For every medical error region of a clip (a stretch where the recogniser was wrong and the passage contains a known medical term) the question is WHERE the engine failed:

    A  the correct term was never a candidate
         A1  the term is not in the knowledge base at all
         A2  no suspicious span covered the heard words (the detector never looked)
         A3  a span covered them but retrieval did not return the term (not in the top 50)
    B  the correct term was retrieved but ranked badly
         B1  retrieved at a prior rank beyond the scored candidates (never scored)
         B2  scored, but another candidate (or the original) came first
    C  the correct term ranked first but the confidence gate rejected it (MEDIUM or LOW)
    D  a WRONG term passed the gate (a HIGH change that is not the gold text, or that damages correct speech)

``recall@k`` is the share of medical error regions whose correct term is within the first k retrieved candidates of a span covering the heard words.
Positions are counted in RAW words (the same normalisation as the evaluator), so a decision and a region overlap when their word ranges intersect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from ...medical.lexicon import stem
from ..normalize.text import normalize
from ..runtime.engine import SCORE_LIMIT
from ..runtime.gate import HIGH, Decision, strip_qualifier
from .evaluate import Region, words


def offsets(texts: Sequence[str]) -> list[int]:
    out, n = [], 0
    for t in texts:
        out.append(n)
        n += len(words(t))
    return out


def raw_range(segment: int, start: int, end: int, texts: Sequence[str], offs: Sequence[int]) -> tuple[int, int]:
    text = texts[segment]
    return offs[segment] + len(words(text[:start])), offs[segment] + len(words(text[:end]))


def region_range(region: Region) -> tuple[int, int]:
    return region.position, region.position + len(region.raw.split())


def overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    if b[0] == b[1]:
        return a[0] <= b[0] <= a[1]
    return a[0] < b[1] and b[0] < a[1]


def _stems(text: str) -> list[str]:
    return [stem(w) for w in normalize(strip_qualifier(text)).split()]


def _contains(big: list[str], small: list[str]) -> bool:
    return bool(small) and any(big[i:i + len(small)] == small for i in range(len(big) - len(small) + 1))


def is_target(surface: str, region: Region) -> bool:
    """The candidate IS the correct answer for the region: it equals the passage's term, or it is a contiguous part of that term that contains the gold words of the region."""
    cand, term, gold = _stems(surface), _stems(region.term), _stems(region.gold)
    if not cand or not term:
        return False
    if cand == term:
        return True
    return _contains(term, cand) and _contains(cand, gold) if gold else False


@dataclass
class RegionDiagnosis:
    region: Region
    failure: str = ""                    # "" for a corrected region, else A1 A2 A3 B1 B2 C D
    rank: Optional[int] = None           # 1-based rank of the correct term in the retrieval order (None = never retrieved)
    examined: bool = False               # a span covered the heard words
    in_kb: bool = True
    detail: str = ""
    retrieved: bool = False              # the correct candidate was among the retrieved ones
    channels: tuple = ()                 # the retrievers that found it
    reason: str = ""                     # NOT_IN_KB | SPAN_NOT_EXAMINED | NO_INDEX_MATCH | FILTERED | RANKED_BELOW | GATE | WRONG_PASSED | OTHER ("" when corrected)
    span_reasons: tuple = ()             # why the span was examined (unknown / confusion / collision / expected / compound)


def diagnose(regions: Sequence[Region], decisions: Sequence[Decision], texts: Sequence[str], term_in_kb, explain=None) -> list[RegionDiagnosis]:
    """One diagnosis per medical error region and per false change (``term_in_kb(term) -> bool``; ``explain(decision, region) -> "FILTERED" | "NO_INDEX_MATCH"`` re-runs
    the retrieval of a span to tell a candidate that a channel found but a cut removed from one that no index returns)."""
    offs = offsets(texts)
    ranges = [(d, raw_range(d.span.segment, d.span.start, d.span.end, texts, offs)) for d in decisions]
    out: list[RegionDiagnosis] = []
    for r in regions:
        rr = region_range(r)
        near = [d for d, rng in ranges if overlaps(rng, rr)]
        diag = RegionDiagnosis(r, examined=bool(near))
        diag.span_reasons = tuple(sorted({x for d in near for x in d.span.reasons}))
        if r.kind == "corrected":
            diag.retrieved = True
            out.append(diag)
        elif r.kind in ("false", "wrong") and not r.medical or r.kind == "false":
            diag.failure, diag.detail, diag.reason = "D", "damaged correct speech" if r.kind == "false" else "changed to a wrong text", "WRONG_PASSED"
            out.append(diag)
        elif not r.medical:
            continue
        elif r.kind == "wrong":
            diag.failure, diag.detail, diag.reason = "D", "a wrong term passed the gate", "WRONG_PASSED"
            out.append(diag)
        else:                                                       # a missed medical error
            diag.in_kb = bool(term_in_kb(r.term, r.gold))
            ranks = []
            channels: tuple = ()
            hit = None
            for d in near:
                for k, entry in enumerate(d.pool, 1):
                    if is_target(entry[0], r):
                        ranks.append(k)
                        if not channels or k == min(ranks):
                            channels = tuple(entry[3]) if len(entry) > 3 else ()
                        break
                if d.best is not None and is_target(d.best.candidate.surface, r):
                    hit = d
            diag.rank = min(ranks) if ranks else None
            diag.retrieved = diag.rank is not None
            diag.channels = channels
            if hit is not None:
                diag.failure, diag.reason, diag.detail = "C", "GATE", f"ranked first, tier {hit.tier}: {hit.reason[:160]}"
            elif diag.rank is not None:
                diag.failure = "B1" if diag.rank > SCORE_LIMIT else "B2"
                diag.reason = "RANKED_BELOW"
                diag.detail = f"retrieved at rank {diag.rank}"
            elif not diag.in_kb:
                diag.failure, diag.reason, diag.detail = "A1", "NOT_IN_KB", "neither the term nor the words of the heard region are in the knowledge base"
            elif not near:
                diag.failure, diag.reason, diag.detail = "A2", "SPAN_NOT_EXAMINED", "no suspicious span covered the heard words"
            else:
                diag.failure = "A3"
                diag.reason = (explain(near[0], r) if explain else "") or "NO_INDEX_MATCH"
                diag.detail = "examined but the term was not retrieved" + (" (a channel found it, a cut removed it)" if diag.reason == "FILTERED" else "")
            out.append(diag)
    return out


def recall_at(diags: Sequence[RegionDiagnosis], ks: Sequence[int] = (10, 25, 50)) -> dict:
    """Medical error regions: share whose correct term is within the first k candidates (and the same share among the regions whose term is in the knowledge base)."""
    med = [d for d in diags if d.region.medical and d.region.kind in ("missed", "wrong", "corrected")]
    kb_med = [d for d in med if d.in_kb or d.region.kind == "corrected"]
    out = {"medical_error_regions": len(med), "with_term_in_knowledge_base": len(kb_med)}
    for k in ks:
        hit = sum(1 for d in med if d.region.kind == "corrected" or (d.rank is not None and d.rank <= k))
        hit_kb = sum(1 for d in kb_med if d.region.kind == "corrected" or (d.rank is not None and d.rank <= k))
        out[f"recall@{k}"] = None if not med else round(hit / len(med), 4)
        out[f"recall@{k}_in_kb"] = None if not kb_med else round(hit_kb / len(kb_med), 4)
    return out


def retrieval_reasons(diags: Sequence[RegionDiagnosis]) -> dict:
    """Why a benchmark target was not retrieved (NOT_IN_KB / SPAN_NOT_EXAMINED / NO_INDEX_MATCH / FILTERED / OTHER) and which retrievers found the ones that were."""
    med = [d for d in diags if d.region.medical and d.region.kind in ("missed", "wrong", "corrected")]
    not_retrieved: dict[str, int] = {}
    found_by: dict[str, int] = {}
    for d in med:
        if d.region.kind == "corrected" or d.retrieved:
            for c in (d.channels or ("(corrected: not traced)",)):
                found_by[c] = found_by.get(c, 0) + 1
        elif d.reason in ("NOT_IN_KB", "SPAN_NOT_EXAMINED", "NO_INDEX_MATCH", "FILTERED"):
            not_retrieved[d.reason] = not_retrieved.get(d.reason, 0) + 1
        else:
            not_retrieved["OTHER"] = not_retrieved.get("OTHER", 0) + 1
    ranks = sorted(d.rank for d in med if d.rank)
    return {"targets": len(med), "span_detection_recall": None if not med else round(sum(1 for d in med if d.examined) / len(med), 4),
            "retrieved": sum(1 for d in med if d.retrieved or d.region.kind == "corrected"), "not_retrieved_reasons": not_retrieved,
            "found_by_channel": dict(sorted(found_by.items(), key=lambda kv: -kv[1])), "first_rank_median": ranks[len(ranks) // 2] if ranks else None,
            "examined_by_reason": _count(d.span_reasons for d in med)}


def _count(groups) -> dict:
    out: dict[str, int] = {}
    for g in groups:
        for x in g:
            out[x] = out.get(x, 0) + 1
    return out


def breakdown(diags: Sequence[RegionDiagnosis]) -> dict:
    failures = [d for d in diags if d.failure]
    counts: dict[str, int] = {}
    for d in failures:
        counts[d.failure] = counts.get(d.failure, 0) + 1
    n = len(failures)
    groups = {"A (retrieval miss)": ("A1", "A2", "A3"), "B (ranking miss)": ("B1", "B2"), "C (gate reject)": ("C",), "D (wrong / false change passed)": ("D",)}
    return {"failures": n, "counts": counts,
            "share": {name: (None if not n else round(sum(counts.get(k, 0) for k in keys) / n, 4)) for name, keys in groups.items()}}


def margins(decisions: Sequence[Decision]) -> dict:
    """Distribution of best-minus-second-best margins over every decision that had a second candidate."""
    values = sorted(round(d.candidate_margin, 3) for d in decisions if d.best is not None and d.runner_up is not None)
    if not values:
        return {"n": 0}

    def q(p: float) -> float:
        return values[min(len(values) - 1, int(p * len(values)))]

    return {"n": len(values), "p10": q(0.10), "p25": q(0.25), "median": q(0.5), "p75": q(0.75), "p90": q(0.90), "share_below_0.05": round(sum(1 for v in values if v < 0.05) / len(values), 3),
            "share_below_0.12": round(sum(1 for v in values if v < 0.12) / len(values), 3)}
