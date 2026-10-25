"""Optional: mine candidate terms from PubMed author keywords (term + PMIDs + years; never article text) into a JSON file.

    python tools/mine_pubmed.py --network --query "neuroanatomy[MeSH Terms] AND Bergmann" --max 300 --domain neuroanatomy --kb data/kb_open_b.db --out data/pubmed_terms.json
    python tools/mine_pubmed.py --xml-file pubmed_dump.xml --out data/pubmed_terms.json          # offline: a PubMed XML file you already have

Then, for a PRIVATE database:  python tools/build_medical_knowledge.py --profile full ... --pubmed data/pubmed_terms.json
The literature source is REVIEW_REQUIRED in the licence registry: a public build refuses it. The network is used only with --network.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common_words import is_common_word  # noqa: E402
from src.medical_knowledge.runtime.knowledge import MedicalKnowledge  # noqa: E402
from src.medical_knowledge.normalize.text import normalize  # noqa: E402
from src.medical_knowledge.sources.pubmed import Evidence, LiteratureMiner, fetch, promote, recognized_identifier, verification_state  # noqa: E402


def variants(term: str) -> list[str]:
    """Spellings of the same term: singular / plural and hyphen / space. A hit on one of them is an ontology ALIAS, not a new concept."""
    t = " ".join(term.split())
    out = {t.replace("-", " "), t.replace(" ", "-")}
    words = t.split()
    if words:
        last = words[-1]
        out |= {" ".join(words[:-1] + [last[:-1]]) if last.endswith("s") and len(last) > 4 else " ".join(words[:-1] + [last + "s"])}
    out.discard(t)
    return sorted(out)


def evidence_for(kb: MedicalKnowledge, term: str) -> Evidence:
    """The independent signals the knowledge base can give for a mined term: an alias of an ontology concept, a recognised identifier (gene symbol, dbSNP, CAS, EC)."""
    alias = any(kb.lookup(v, limit=1) and kb.lookup(v, limit=1)[0].match == "exact" for v in variants(term))
    gene = lambda symbol: any(h.match == "exact" and h.concept_id.startswith("HGNC") for h in kb.lookup(symbol, limit=3))                      # noqa: E731
    return Evidence(ontology_match=alias, recognized_identifier=recognized_identifier(term, gene))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", action="store_true", help="allow requests to NCBI E-utilities (off by default)")
    ap.add_argument("--query", default="")
    ap.add_argument("--max", type=int, default=200)
    ap.add_argument("--xml-file", type=Path, action="append", default=[])
    ap.add_argument("--domain", action="append", default=[])
    ap.add_argument("--kb", type=Path, help="knowledge base: terms it already knows are not candidates")
    ap.add_argument("--email", default="")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    kb = MedicalKnowledge.open(a.kb) if a.kb else None
    miner = LiteratureMiner(is_known=(lambda t: bool(kb.lookup(t, limit=1)) and kb.lookup(t, limit=1)[0].match == "exact") if kb else None,
                            is_plain_english=lambda t: is_common_word(t) or bool(kb and kb.is_english_word(t, 70)))
    for path in a.xml_file:
        miner.add_xml(path.read_bytes())
    if a.query:
        if not a.network:
            print("a query needs --network (the network is never used without it)")
            return 2
        for batch in fetch(a.query, a.max, tool_email=a.email):
            miner.add_xml(batch)
    if not a.xml_file and not a.query:
        print("nothing to mine: give --query with --network, or --xml-file")
        return 2
    if kb:
        for term in miner.candidates():
            miner.evidence[normalize(term.term)] = evidence_for(kb, term.term)
    data = miner.to_json(a.query, a.domain)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    states: dict[str, int] = {}
    for t in miner.candidates():
        state = verification_state(t, miner.evidence.get(normalize(t.term)))
        states[state] = states.get(state, 0) + 1
    print(f"{miner.records} records, {len(data['terms'])} candidate terms, states {states}, rejected {dict(miner.rejected)} -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
