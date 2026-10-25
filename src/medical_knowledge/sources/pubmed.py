"""Optional literature layer: candidate TERMS mined from the author keywords of PubMed records.

What is kept, per term: the term, the number of articles that use it, up to ``MAX_PMIDS`` PubMed identifiers, the first and last publication year. NOTHING else of an
article is stored - no title, no abstract, no sentence. Mined terms enter the knowledge base as UNVERIFIED_CANDIDATE and are only promoted to LITERATURE_VERIFIED by
an explicit rule (``promote``): several independent articles, over more than one year, and a term-like shape. They are PRIVATE: the licence registry lists the source
as REVIEW_REQUIRED, so a public build refuses it. The network is used only by ``tools/mine_pubmed.py --network`` (E-utilities, no credentials); the importer reads
the mined JSON file and never touches the network.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

from ..normalize.text import normalize
from .base import ConceptRecord, Record, SourceInfo, TermRecord, TerminologySource

SOURCE_ID = "PUBMED_MINED"
MAX_PMIDS = 10
MAX_WORDS = 4
MIN_LETTERS = 5
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
PROMOTE_MIN_ARTICLES = 5
PROMOTE_MIN_YEARS = 2


@dataclass
class MinedTerm:
    term: str
    articles: int = 0
    pmids: list[str] = field(default_factory=list)
    first_year: int = 0
    last_year: int = 0
    dois: list[str] = field(default_factory=list)

    def add(self, pmid: str, year: int, doi: str = "") -> None:
        if pmid in self.pmids:
            return                                                   # the same article mined twice counts once
        self.articles += 1
        if len(self.pmids) < MAX_PMIDS:
            self.pmids.append(pmid)
        if doi and doi not in self.dois and len(self.dois) < MAX_PMIDS:
            self.dois.append(doi)
        if year:
            self.first_year = min(self.first_year or year, year)
            self.last_year = max(self.last_year, year)

    @property
    def years(self) -> int:
        return self.last_year - self.first_year + 1 if self.first_year else 0

    def identifiers(self) -> list[str]:
        return [f"PMID:{p}" for p in self.pmids] + [f"DOI:{d}" for d in self.dois]


#: states of a mined term. Even VERIFIED terms still need the context evidence of the decision gate: a literature term is never "HIGH vocabulary" on its own.
UNVERIFIED, MULTI_SOURCE, ONTOLOGY_MATCHED, VERIFIED = "UNVERIFIED", "MULTI-SOURCE", "ONTOLOGY_MATCHED", "VERIFIED"
STATES = (UNVERIFIED, MULTI_SOURCE, ONTOLOGY_MATCHED, VERIFIED)
IDENTIFIER_SHAPES = (re.compile(r"^rs\d{3,}$", re.IGNORECASE),                                     # dbSNP variant
                     re.compile(r"^\d{2,7}-\d{2}-\d$"),                                             # CAS registry number
                     re.compile(r"^(?:EC )?\d+\.(?:\d+|-)\.(?:\d+|-)\.(?:\d+|-)$"))                 # enzyme commission number


@dataclass
class Evidence:
    """Everything besides the publications that speaks for a term. Each field is an independent signal; none of them is derived from another."""

    ontology_match: bool = False           # the term (or its singular / plural / hyphenation) is an alias of a concept of a licensed ontology in the knowledge base
    recognized_identifier: bool = False    # a gene symbol of the gene table, a dbSNP / CAS / EC number
    lectures: int = 0                      # distinct lectures of the user in which the term was heard
    in_pdf: bool = False                   # printed in the slides the user attached
    custom_vocabulary: bool = False        # added by the user in Settings > Medical Vocabulary

    def others(self) -> int:
        return sum([self.ontology_match, self.recognized_identifier, self.lectures >= 2, self.in_pdf, self.custom_vocabulary])


def multi_source(term: "MinedTerm") -> bool:
    return term.articles >= PROMOTE_MIN_ARTICLES and term.years >= PROMOTE_MIN_YEARS and term_shape_ok(term.term)


def verification_state(term: "MinedTerm", evidence: Optional[Evidence] = None) -> str:
    """UNVERIFIED -> MULTI-SOURCE (several publications over more than one year) / ONTOLOGY_MATCHED (an alias of an ontology concept) -> VERIFIED (several publications AND at least
    one independent signal: ontology match, recognised identifier, two lectures, the user's slides, the user's own vocabulary; or two such signals without publications).
    A keyword that is not even shaped like a term stays UNVERIFIED."""
    evidence = evidence or Evidence()
    if not term_shape_ok(term.term):
        return UNVERIFIED
    publications = multi_source(term)
    if (publications and evidence.others() >= 1) or evidence.others() >= 2:
        return VERIFIED
    if evidence.ontology_match:
        return ONTOLOGY_MATCHED
    return MULTI_SOURCE if publications else UNVERIFIED


def confidence(term: "MinedTerm", evidence: Optional[Evidence] = None) -> float:
    """0..1, a reading aid for the review list, not a probability: publications, years and independent signals add up; a shape that is not a term scores 0."""
    if not term_shape_ok(term.term):
        return 0.0
    evidence = evidence or Evidence()
    score = 0.10 + 0.08 * min(term.articles, 6) + 0.07 * min(max(term.years - 1, 0), 3) + 0.15 * evidence.others()
    return round(min(score, 0.99), 3)


def recognized_identifier(term: str, is_gene_symbol: Optional[Callable[[str], bool]] = None) -> bool:
    t = term.strip()
    return any(shape.match(t) for shape in IDENTIFIER_SHAPES) or bool(is_gene_symbol and re.fullmatch(r"[A-Z][A-Z0-9]{1,7}", t) and is_gene_symbol(t))


def parse_articles(xml: bytes | str) -> Iterator[tuple[str, int, list[str]]]:
    """(PMID, publication year, author keywords) of every record. Titles and abstracts are never read."""
    for pmid, year, words, _doi in parse_records(xml):
        yield pmid, year, words


def parse_records(xml: bytes | str) -> Iterator[tuple[str, int, list[str], str]]:
    """(PMID, publication year, author keywords, DOI) of every record that has keywords. Titles and abstracts are never read."""
    root = ET.fromstring(xml)
    for art in root.iter("PubmedArticle"):
        pmid = (art.findtext(".//MedlineCitation/PMID") or "").strip()
        year = 0
        for path in (".//Article/Journal/JournalIssue/PubDate/Year", ".//ArticleDate/Year", ".//PubDate/MedlineDate"):
            raw = art.findtext(path) or ""
            m = re.search(r"(19|20)\d\d", raw)
            if m:
                year = int(m.group(0))
                break
        words = [(k.text or "").strip() for k in art.iter("Keyword")]
        doi = ""
        for node in art.iter("ArticleId"):
            if node.get("IdType") == "doi" and (node.text or "").strip():
                doi = node.text.strip()
                break
        if pmid and words:
            yield pmid, year, [w for w in words if w], doi


_SHAPE = re.compile(r"^[A-Za-z][A-Za-z0-9\-' ]*[A-Za-z0-9]$")


def term_shape_ok(term: str) -> bool:
    """A keyword that can be a TERM: 1-4 words, letters (digits and hyphens inside), no sentence punctuation, long enough."""
    t = " ".join(term.split())
    return bool(_SHAPE.match(t)) and 1 <= len(t.split()) <= MAX_WORDS and len(re.sub(r"[^A-Za-z]", "", t)) >= MIN_LETTERS and not re.search(r"\d{4,}", t)


class LiteratureMiner:
    """Counts keywords. ``is_known(term)`` tells whether the knowledge base already has it (such terms are not candidates: they add nothing)."""

    def __init__(self, is_known: Optional[Callable[[str], bool]] = None, is_plain_english: Optional[Callable[[str], bool]] = None) -> None:
        self.is_known = is_known or (lambda t: False)
        self.is_plain_english = is_plain_english or (lambda t: False)
        self.terms: dict[str, MinedTerm] = {}
        self.evidence: dict[str, Evidence] = {}                      # normalised term -> independent signals (set by the caller: knowledge base, lectures, slides, custom words)
        self.records = 0
        self.rejected = defaultdict(int)

    def add_xml(self, xml: bytes | str) -> int:
        n = 0
        for pmid, year, keywords, doi in parse_records(xml):
            self.records += 1
            seen: set[str] = set()
            for kw in keywords:
                if not term_shape_ok(kw):
                    self.rejected["shape"] += 1
                    continue
                key = normalize(kw)
                if not key or key in seen:
                    continue
                seen.add(key)
                if self.is_known(kw):
                    self.rejected["known"] += 1
                    continue
                if len(key.split()) == 1 and self.is_plain_english(key):
                    self.rejected["plain English"] += 1
                    continue
                self.terms.setdefault(key, MinedTerm(kw.strip())).add(pmid, year, doi)
                n += 1
        return n

    def candidates(self, min_articles: int = 1) -> list[MinedTerm]:
        return sorted((t for t in self.terms.values() if t.articles >= min_articles), key=lambda t: (-t.articles, t.term.lower()))

    def to_json(self, query: str = "", domains: Iterable[str] = ()) -> dict:
        return {"format": "mlt-pubmed-mined-1", "query": query, "domains": list(domains), "mined_at": time.strftime("%Y-%m-%d"), "records": self.records,
                "note": "term candidates and identifiers only; no article text", "terms": [term_row(t, self.evidence.get(normalize(t.term))) for t in self.candidates()]}


def term_row(term: MinedTerm, evidence: Optional[Evidence] = None) -> dict:
    """One candidate as it is stored: term, identifier, source identifiers (PMIDs, DOIs), frequency, first / last seen, confidence, verification state, the independent signals.
    Never any article text."""
    state = verification_state(term, evidence)
    row = {"term": term.term, "identifier": concept_id(term.term), "source_ids": term.identifiers(), "frequency": term.articles, "first_seen": term.first_year, "last_seen": term.last_year,
           "confidence": confidence(term, evidence), "verification_state": state,
           # the keys of the first format stay readable for databases built earlier
           "articles": term.articles, "pmids": list(term.pmids), "dois": list(term.dois), "first_year": term.first_year, "last_year": term.last_year}
    if evidence is not None:
        row["evidence"] = asdict(evidence)
    return row


def promote(term: MinedTerm, evidence: Optional[Evidence] = None) -> str:
    """The trust a mined term gets in the knowledge base: LITERATURE_VERIFIED only in the state VERIFIED (several publications plus an independent signal), else UNVERIFIED_CANDIDATE.
    Publications alone are MULTI-SOURCE: not enough. Even LITERATURE_VERIFIED is not high confidence: the decision gate still wants the context evidence of the lecture."""
    return "LITERATURE_VERIFIED" if verification_state(term, evidence) == VERIFIED else "UNVERIFIED_CANDIDATE"


def concept_id(term: str) -> str:
    return "LIT:" + hashlib.sha1(normalize(term).encode("utf-8")).hexdigest()[:12]


class PubmedSource(TerminologySource):
    """Reads a mined JSON file (``tools/mine_pubmed.py``); offline."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.data: dict = json.loads(self.path.read_text(encoding="utf-8")) if self.path.is_file() else {}
        self.info = SourceInfo(SOURCE_ID, "Terms mined from PubMed keywords (literature layer)", self.data.get("mined_at", "unknown"), "https://pubmed.ncbi.nlm.nih.gov/",
                               "REVIEW_REQUIRED", False, False, "en", self.data.get("mined_at", ""), "candidate terms + PMIDs only; never article text")

    def verify(self) -> list[str]:
        if not self.data:
            return [f"mined terms file not found: {self.path} (python tools/mine_pubmed.py --network --query ... --out FILE)"]
        return [] if self.data.get("format") == "mlt-pubmed-mined-1" else ["not a mined terms file"]

    def parse(self) -> Iterator[dict]:
        yield from self.data.get("terms", [])

    def normalize(self, raw: Iterable[dict]) -> Iterator[Record]:
        domains = tuple(self.data.get("domains") or ())
        for row in raw:
            t = MinedTerm(row["term"], row.get("frequency", row.get("articles", 0)), list(row.get("pmids", [])), row.get("first_seen", row.get("first_year", 0)),
                          row.get("last_seen", row.get("last_year", 0)), list(row.get("dois", [])))
            trust = promote(t, Evidence(**row["evidence"]) if row.get("evidence") else None)
            cid = concept_id(t.term)
            yield ConceptRecord(cid, t.term, "", domains, "", "term", trust)
            yield TermRecord(cid, t.term, "en", "preferred", True, f"{t.first_year}-{t.last_year}", ";".join(f"PMID:{p}" for p in t.pmids), t.articles, 0.0, trust)


# ------------------------------------------------------------------------------------------------------------------ network (opt-in, tools/mine_pubmed.py only)
def _get(url: str, params: dict, timeout: float = 60.0) -> bytes:
    request = urllib.request.Request(url + "?" + urllib.parse.urlencode(params), headers={"User-Agent": "MedicalLectureTranscriber-literature-miner/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:        # nosec - fixed NCBI host
        return response.read()


def fetch(query: str, retmax: int = 200, batch: int = 100, delay: float = 0.4, tool_email: str = "", getter: Callable[[str, dict], bytes] = _get) -> Iterator[bytes]:
    """PubMed XML batches for a search. E-utilities need no credentials; the polite rate (<= 3 requests / s) is kept. ``getter`` is replaceable for tests."""
    common = {"tool": "MedicalLectureTranscriber", **({"email": tool_email} if tool_email else {})}
    found = json.loads(getter(EUTILS + "esearch.fcgi", {**common, "db": "pubmed", "term": query, "retmax": retmax, "retmode": "json", "sort": "relevance"}))
    ids = found.get("esearchresult", {}).get("idlist", [])
    for i in range(0, len(ids), batch):
        time.sleep(delay)
        yield getter(EUTILS + "efetch.fcgi", {**common, "db": "pubmed", "id": ",".join(ids[i:i + batch]), "retmode": "xml"})
