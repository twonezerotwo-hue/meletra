"""What the knowledge base covers - per specialty, per kind of term, per source - and where it is thin.

    python tools/coverage_report.py --kb data/medical_knowledge_public.db [--json benchmarks/v4/coverage_public.json] [--markdown docs/v4/COVERAGE.md]

Counts are taken from the database itself (read-only); nothing is estimated. A specialty is THIN below ``THIN`` concepts: the recogniser's errors in such a lecture will find fewer candidates,
and the report says so instead of hiding it. The thresholds are a reading aid, not a quality claim.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.medical_knowledge.domains import DOMAINS  # noqa: E402

THIN = 500
KINDS = ("term", "gene", "drug", "eponym", "latin")


def report(db_path: Path) -> dict:
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    sources = {r[0]: {"title": r[1], "licence_status": r[5] if len(r) > 5 else ""} for r in db.execute("SELECT * FROM source")}
    total = {"concepts": db.execute("SELECT COUNT(*) FROM concept").fetchone()[0], "terms": db.execute("SELECT COUNT(*) FROM term").fetchone()[0],
             "english_words": db.execute("SELECT COUNT(*) FROM english_word").fetchone()[0]}
    by_domain = {d: n for d, n in db.execute("SELECT domain, COUNT(DISTINCT concept_id) FROM concept_domain GROUP BY domain")}
    terms_by_domain = {d: n for d, n in db.execute("SELECT cd.domain, COUNT(*) FROM concept_domain cd JOIN term t ON t.concept_id = cd.concept_id AND t.language IN ('en','la') GROUP BY cd.domain")}
    kinds = {d: {k: n for k, n in db.execute("SELECT c.kind, COUNT(*) FROM concept c JOIN concept_domain cd ON cd.concept_id = c.concept_id WHERE cd.domain = ? GROUP BY c.kind", (d,))}
             for d in by_domain}
    sources_by_domain = {d: {s: n for s, n in db.execute("SELECT ts.source_id, COUNT(DISTINCT t.concept_id) FROM concept_domain cd JOIN term t ON t.concept_id = cd.concept_id "
                                                         "JOIN term_source ts ON ts.term_id = t.term_id WHERE cd.domain = ? GROUP BY ts.source_id", (d,))} for d in by_domain}
    declared = dict(DOMAINS)
    rows = []
    for dom_id, dom in declared.items():
        n = by_domain.get(dom_id, 0)
        rows.append({"domain": dom_id, "title": dom.title, "group": dom.group, "concepts": n, "terms": terms_by_domain.get(dom_id, 0), "kinds": kinds.get(dom_id, {}),
                     "sources": sources_by_domain.get(dom_id, {}), "status": "NONE" if n == 0 else ("THIN" if n < THIN else "OK")})
    rows.sort(key=lambda r: -r["concepts"])
    by_kind = {k: n for k, n in db.execute("SELECT kind, COUNT(*) FROM concept GROUP BY kind")}
    by_term_type = {k: n for k, n in db.execute("SELECT term_type, COUNT(*) FROM term GROUP BY term_type")}
    sem = {k or "(none)": n for k, n in db.execute("SELECT semantic_type, COUNT(*) FROM concept GROUP BY semantic_type ORDER BY 2 DESC LIMIT 12")}
    wanted = {"Latin / Greek forms": by_domain.get("latin", 0), "abbreviations (table)": db.execute("SELECT COUNT(*) FROM abbreviation").fetchone()[0],
              "eponyms": by_domain.get("eponyms", 0), "drugs (concepts of kind drug)": by_kind.get("drug", 0), "genes": by_kind.get("gene", 0),
              "proteins (gene with protein product)": sem.get("gene with protein product", 0), "laboratory medicine": by_domain.get("laboratory_medicine", 0),
              "surgery (procedures)": by_domain.get("surgery", 0), "internal medicine (syndromes, signs, symptoms)": by_domain.get("internal_medicine", 0)}
    return {"totals": total, "domains": rows, "by_kind": by_kind, "by_term_type": by_term_type, "semantic_types": sem, "categories": wanted, "sources": sources,
            "thin": [r["domain"] for r in rows if r["status"] in ("THIN", "NONE")]}


def markdown(rep: dict) -> str:
    out = ["# Knowledge coverage (public pack)", "",
           f"{rep['totals']['concepts']:,} concepts - {rep['totals']['terms']:,} terms - {rep['totals']['english_words']:,} English dictionary words. Counts come from the database itself.", "",
           "| Specialty | Concepts | Terms | Status | Main sources |", "|---|---|---|---|---|"]
    for r in rep["domains"]:
        top = ", ".join(f"{s} {n:,}" for s, n in sorted(r["sources"].items(), key=lambda kv: -kv[1])[:3])
        out.append(f"| {r['title']} | {r['concepts']:,} | {r['terms']:,} | {r['status']} | {top} |")
    out += ["", f"THIN = fewer than {THIN} concepts. Thin or empty: " + (", ".join(rep["thin"]) or "none") + ".", "", "| Kind of term | Count |", "|---|---|"]
    out += [f"| {k} | {v:,} |" for k, v in rep["categories"].items()]
    out += ["", "Sources: " + "; ".join(f"{k} ({v['title']})" for k, v in rep["sources"].items()) + "."]
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", type=Path, required=True)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--markdown", type=Path)
    a = ap.parse_args(argv)
    rep = report(a.kb)
    print(markdown(rep))
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    if a.markdown:
        a.markdown.parent.mkdir(parents=True, exist_ok=True)
        a.markdown.write_text(markdown(rep), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
