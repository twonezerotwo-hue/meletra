"""Drug / gene / protein / receptor naming report (reporting only).

Lectures write the same entity in many ways (``TNF-alpha``, ``TNF alpha``, ``TNFα``). This lists, per entity,
which spellings occurred and flags spelling variation or a non-canonical-only spelling, plus near-miss
spellings of dictionary names. Nothing is normalised in the transcript itself.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from .knowledge_acronyms import NAME_VARIANTS
from .schema import Segment
from .vocabulary import normalize_key

REPORTED_CATEGORIES = ("drug", "gene", "protein", "receptor", "enzyme")


@dataclass
class NameFinding:
    canonical: str
    spellings: Counter = field(default_factory=Counter)        # surface form -> count
    first_start: Optional[float] = None
    status: str = "OK"                                        # OK | SPELLING_VARIATION | NON_CANONICAL

    def to_dict(self) -> dict[str, Any]:
        return {"canonical": self.canonical, "spellings": dict(self.spellings), "first_start": self.first_start, "status": self.status}


def _pattern(spelling: str) -> re.Pattern[str]:
    escaped = re.escape(spelling.lower()).replace(r"\ ", r"[\s\-]*")
    return re.compile(r"(?<![a-z0-9])" + escaped + r"(?![a-z0-9])")


_COMPILED = {c: [(v, _pattern(v)) for v in (c, *variants)] for c, variants in NAME_VARIANTS.items()}


def scan_name_variants(segments: Sequence[Segment]) -> list[NameFinding]:
    findings: dict[str, NameFinding] = {}
    for seg in segments:
        lowered = seg.text.lower()
        for canonical, patterns in _COMPILED.items():
            for spelling, pattern in patterns:
                for match in pattern.finditer(lowered):
                    surface = seg.text[match.start():match.end()]
                    finding = findings.setdefault(canonical, NameFinding(canonical))
                    finding.spellings[surface] += 1
                    if finding.first_start is None:
                        finding.first_start = seg.start
    out: list[NameFinding] = []
    for finding in findings.values():
        # collapse case-only differences ("TNF-alpha" / "tnf-alpha" are one spelling); spaces and hyphens DO matter:
        # "acetyl cholinesterase" is an ASR split of "acetylcholinesterase"
        distinct = {s.casefold() for s in finding.spellings}
        if len(distinct) > 1:
            finding.status = "SPELLING_VARIATION"
        elif finding.canonical.casefold() not in distinct:
            finding.status = "NON_CANONICAL"
        out.append(finding)
    return sorted(out, key=lambda f: (f.first_start or 0.0, f.canonical))


def name_report_text(findings: Sequence[NameFinding], entities: Sequence[Any], errors: Sequence[Any]) -> str:
    from .exporters import format_hms

    out = ["DRUG / GENE / PROTEIN / RECEPTOR NAMING REPORT",
           "Reporting only: no spelling is changed in the transcript.", ""]
    out.append("SPELLING VARIANTS OF KNOWN NAMES")
    if not findings:
        out.append("  (none found)")
    for f in findings:
        spelled = ", ".join(f"'{s}' x{n}" for s, n in f.spellings.most_common())
        note = {"OK": "", "SPELLING_VARIATION": "  <-- several spellings used",
                "NON_CANONICAL": f"  <-- only a non-canonical spelling; canonical form is '{f.canonical}'"}[f.status]
        out.append(f"  [{format_hms(f.first_start or 0)}] {f.canonical}: {spelled}{note}")
    out += ["", "DETECTED ENTITIES OF THESE KINDS"]
    shown = [e for e in entities if e.categories and e.categories[0] in REPORTED_CATEGORIES]
    if not shown:
        out.append("  (none)")
    by_cat: dict[str, list[Any]] = defaultdict(list)
    for e in shown:
        by_cat[e.categories[0]].append(e)
    for cat in REPORTED_CATEGORIES:
        for e in sorted(by_cat.get(cat, []), key=lambda x: -x.count):
            out.append(f"  {cat:<9} {e.term:<36} x{e.count:<3} first {format_hms(e.first_start)}")
    out += ["", "POSSIBLE MIS-SPELLINGS OF DICTIONARY NAMES"]
    relevant = [x for x in errors if x.suggestion and any(
        normalize_key(x.suggestion) == normalize_key(e.term) for e in shown)]
    if not relevant:
        out.append("  (none)")
    for x in relevant:
        out.append(f"  [{format_hms(x.start)}] '{x.original}' ~ '{x.suggestion}' (similarity {x.confidence:.2f})")
    return "\n".join(out) + "\n"
