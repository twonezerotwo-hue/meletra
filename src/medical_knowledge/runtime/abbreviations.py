"""Context-aware abbreviation resolution. An abbreviation has NO global meaning ("MS" = multiple sclerosis, mitral stenosis, mass spectrometry, morphine sulfate ...).

Evidence for an expansion, strongest first:

    defined      the lecture spells it out next to the abbreviation ("central nervous system (CNS)", "CNS, the central nervous system") - near-certain
    spoken       the long form was said earlier in this lecture
    context      the expansion's concept is related to the known terms around the abbreviation, or shares their field of study (slides / PDF count double)
    field        the lecture's topic favours the expansion's field of study

When two expansions stay close, the abbreviation is reported AMBIGUOUS and nothing is assumed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from ..domains import SUPPORT_DOMAINS
from ..normalize.text import compact, normalize
from .context import LectureContext, Token

_ABBR = re.compile(r"^[A-Z][A-Za-z]{0,5}[A-Z0-9]$|^[A-Z]{2,6}s?$")
MARGIN = 0.12


@dataclass
class Expansion:
    long_form: str
    concept_id: str
    domain: str
    score: float
    evidence: dict[str, float] = field(default_factory=dict)


@dataclass
class Resolution:
    short: str
    segment: int
    chosen: Optional[Expansion]
    alternatives: list[Expansion]
    ambiguous: bool
    reason: str

    def to_dict(self) -> dict:
        return {"abbreviation": self.short, "segment": self.segment, "expansion": self.chosen.long_form if self.chosen else None,
                "score": round(self.chosen.score, 3) if self.chosen else None, "ambiguous": self.ambiguous, "reason": self.reason,
                "evidence": {k: round(v, 2) for k, v in (self.chosen.evidence if self.chosen else {}).items()},
                "alternatives": [{"expansion": a.long_form, "score": round(a.score, 3)} for a in self.alternatives]}


class AbbreviationResolver:
    def __init__(self, ctx: LectureContext) -> None:
        self.ctx = ctx
        self.kb = ctx.kb
        self._text = [normalize(s.view.text) for s in ctx.segments]

    # ------------------------------------------------------------------ evidence
    def _defined_nearby(self, short: str, long_form: str, segment: int) -> bool:
        short_n, long_n = normalize(short), normalize(long_form)
        for j in range(max(0, segment - 3), min(len(self._text), segment + 2)):
            text = self._text[j]
            if long_n in text and re.search(rf"\b{re.escape(short_n)}\b", text):
                return True
        return False

    def _spoken_before(self, long_form: str, segment: int) -> bool:
        long_n = normalize(long_form)
        return any(long_n in self._text[j] for j in range(0, segment))

    def resolve(self, short: str, segment: int) -> Optional[Resolution]:
        rows = self.kb.db.execute("SELECT long_form, concept_id, domain FROM abbreviation WHERE norm_short = ?", (compact(short),)).fetchall()
        if not rows:
            return None
        ctx = self.ctx
        around = ctx.window_concepts(segment)
        fields = self.kb.concept_domains_many({r[1] for r in rows if r[1]} | around)
        expansions: list[Expansion] = []
        for long_form, cid, domain in rows:
            ev: dict[str, float] = {}
            if self._defined_nearby(short, long_form, segment):
                ev["defined"] = 1.0
            elif self._spoken_before(long_form, segment):
                ev["spoken"] = 0.7
            doms = [d for d in fields.get(cid, ()) if d not in SUPPORT_DOMAINS] or ([domain] if domain else [])
            if cid:
                neigh = {r.concept_id for r in self.kb.related(cid, limit=100)}
                link = len(neigh & around)
                shared = sum(1 for c in around if set(doms) & set(fields.get(c, ())))
                ev["context"] = min(1.0, 0.5 * min(link, 2) / 2 * 2 + 0.5 * min(shared, 6) / 6)
                if cid in ctx.lecture_concepts:
                    ev["context"] = min(1.0, ev["context"] + 0.5)
            ev["field"] = ctx.topic_weight(doms) if doms else 0.3
            score = 0.55 * ev.get("defined", 0.0) + 0.15 * ev.get("spoken", 0.0) + 0.45 * ev.get("context", 0.0) + 0.2 * ev["field"]
            expansions.append(Expansion(long_form, cid or "", doms[0] if doms else "", min(1.0, score), ev))
        expansions.sort(key=lambda e: -e.score)
        best = expansions[0]
        ambiguous = len(expansions) > 1 and expansions[1].score >= best.score - MARGIN
        if best.score < 0.15 and "defined" not in best.evidence:
            return Resolution(short, segment, None, expansions[:3], len(expansions) > 1, "no evidence for any expansion")
        reason = "defined in the lecture" if "defined" in best.evidence else "context" if best.evidence.get("context", 0) >= 0.4 else "the lecture's field"
        return Resolution(short, segment, None if ambiguous else best, expansions[:3], ambiguous, "ambiguous: several expansions fit" if ambiguous else reason)

    def resolve_all(self) -> list[Resolution]:
        """Every abbreviation of the transcript the knowledge base knows, resolved in its context (one entry per occurrence)."""
        out: list[Resolution] = []
        for seg in self.ctx.segments:
            for tok in seg.tokens:
                if _ABBR.match(tok.text) and len(tok.text) >= 2:
                    r = self.resolve(tok.text, seg.index)
                    if r is not None:
                        out.append(r)
        return out
