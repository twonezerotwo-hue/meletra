"""SQLite schema of the Medical Knowledge database (own version, separate from the app version).

Everything lives in one file that is opened read-only at run time; nothing is loaded into Python dictionaries at start-up.

    source          one row per imported terminology release (name, version, license category, redistributable?)
    concept         one meaning; ``trust`` says how far it is verified (CURATED, ONTOLOGY_VERIFIED, LITERATURE_VERIFIED, LECTURE_SPECIFIC,
                    USER_DEFINED, UNVERIFIED_CANDIDATE); fields of study are in ``concept_domain`` (a concept may belong to several)
    term            one written form of a concept in one language (unique per concept + compact spelling + language)
    term_source     PROVENANCE: which source(s) delivered a term, under which source code / source concept id
    abbreviation    short form -> long form (ambiguity is a feature: several rows for the same short form)
    relationship    concept A -relation-> concept B (kept light: related_to, part_of, is_a, cue)
    context_cue     plain words that usually appear near a concept (the context that makes a correction believable)
    confusion       KNOWN WRONG spellings (how ASR mishears a concept); never valid terms, only retrieval hints
    term.sk / dm    the consonant-skeleton and the Double Metaphone key of the term (``sk`` indexed: exact sound-alike lookup)
    phonetic_key    ``(algorithm, key) -> term`` exact lookup of further sound-alike keys (algorithms other than the skeleton)
    morphology      lemma / Latin case and number / Greek root / prefix / suffix of a term (helps retrieval, never invents terms)
    token_freq      number of terms containing a word (rare words say more than common ones)
    english_word    ordinary English words by commonness (valid words are never "corrected" on sound alone)
    phonetic_del    deletion neighbourhood of the Double Metaphone key: fuzzy sound-alike candidates in sub-millisecond time (measured, see docs/v4/RETRIEVAL.md)
    term_fts        FTS5 over the normalised text (words and prefixes), for prefix search and the diagnostics page
"""

from __future__ import annotations

SCHEMA_VERSION = 1
KNOWLEDGE_NAME = "MedicalKnowledge"

TRUST_LEVELS = ("CURATED", "ONTOLOGY_VERIFIED", "LITERATURE_VERIFIED", "LECTURE_SPECIFIC", "USER_DEFINED", "UNVERIFIED_CANDIDATE")

TABLES = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE source (
    source_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    url TEXT,
    license_category TEXT NOT NULL,
    license_status TEXT NOT NULL,
    redistribution_allowed INTEGER NOT NULL,
    requires_credentials INTEGER NOT NULL DEFAULT 0,
    language TEXT,
    release_date TEXT,
    notes TEXT
);

CREATE TABLE concept (
    concept_id TEXT PRIMARY KEY,
    preferred_name TEXT NOT NULL,
    semantic_type TEXT,
    domain TEXT,
    subdomain TEXT,
    kind TEXT NOT NULL DEFAULT 'term',
    trust TEXT NOT NULL DEFAULT 'CURATED'
);

CREATE TABLE concept_domain (
    concept_id TEXT NOT NULL,
    domain TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    PRIMARY KEY (concept_id, domain)
) WITHOUT ROWID;

CREATE TABLE term (
    term_id INTEGER PRIMARY KEY,
    concept_id TEXT NOT NULL,
    text TEXT NOT NULL,
    normalized_text TEXT NOT NULL,
    compact_text TEXT NOT NULL,
    sk TEXT NOT NULL DEFAULT '',
    dm TEXT NOT NULL DEFAULT '',
    language TEXT NOT NULL DEFAULT 'en',
    term_type TEXT,
    frequency INTEGER NOT NULL DEFAULT 0,
    medical_frequency REAL NOT NULL DEFAULT 0,
    is_preferred INTEGER NOT NULL DEFAULT 0,
    redistributable INTEGER NOT NULL DEFAULT 0,
    token_count INTEGER NOT NULL,
    trust TEXT NOT NULL DEFAULT 'CURATED',
    UNIQUE (concept_id, compact_text, language)
);

CREATE TABLE term_source (
    term_id INTEGER NOT NULL,
    source_id TEXT NOT NULL,
    source_code TEXT NOT NULL DEFAULT '',
    source_concept_id TEXT NOT NULL DEFAULT '',
    source_term_type TEXT,
    language TEXT,
    PRIMARY KEY (term_id, source_id, source_code)
) WITHOUT ROWID;

CREATE TABLE abbreviation (
    abbr_id INTEGER PRIMARY KEY,
    short_form TEXT NOT NULL,
    norm_short TEXT NOT NULL,
    long_form TEXT NOT NULL,
    concept_id TEXT,
    domain TEXT,
    language TEXT NOT NULL DEFAULT 'en',
    source_id TEXT NOT NULL,
    UNIQUE (short_form, long_form, domain)
);

CREATE TABLE relationship (
    concept_a TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    concept_b TEXT NOT NULL,
    source_id TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    PRIMARY KEY (concept_a, relation_type, concept_b, source_id)
) WITHOUT ROWID;

CREATE TABLE context_cue (
    concept_id TEXT NOT NULL,
    cue TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    PRIMARY KEY (concept_id, cue)
) WITHOUT ROWID;

CREATE TABLE confusion (
    concept_id TEXT NOT NULL,
    heard TEXT NOT NULL,
    norm_heard TEXT NOT NULL,
    source_id TEXT NOT NULL,
    PRIMARY KEY (concept_id, norm_heard, source_id)
) WITHOUT ROWID;

CREATE TABLE phonetic_key (
    term_id INTEGER NOT NULL,
    algorithm TEXT NOT NULL,
    key TEXT NOT NULL
);

-- deletion neighbourhood of the Double Metaphone key (``term.dm``) (every key with one letter removed, plus the key itself): a query key and a stored key that share
-- an entry are at most two edits apart. Only terms of up to four words and keys of 3..24 letters are listed (milestone 3 measurements).
CREATE TABLE phonetic_del (
    key TEXT NOT NULL,
    term_id INTEGER NOT NULL,
    PRIMARY KEY (key, term_id)
) WITHOUT ROWID;

CREATE TABLE morphology (
    term_id INTEGER PRIMARY KEY,
    lemma TEXT,
    latin_case TEXT,
    latin_number TEXT,
    greek_root TEXT,
    prefix TEXT,
    suffix TEXT
);

CREATE TABLE token_freq (token TEXT PRIMARY KEY, n INTEGER NOT NULL) WITHOUT ROWID;

-- the WORDS of the vocabulary as retrieval candidates: a rare word can be the answer even when no whole term is ("paraneoplastic" ~ "Paraneoplastic Syndromes").
-- letters only, length >= 4; pre / suf = first 3 / last 4 letters; sk / dm = sound keys; level = SCOWL level of the word (0 = not an ordinary English word)
CREATE TABLE token_index (token TEXT PRIMARY KEY, n INTEGER NOT NULL, len INTEGER NOT NULL, pre TEXT NOT NULL, suf TEXT NOT NULL, sk TEXT NOT NULL, dm TEXT NOT NULL,
                          level INTEGER NOT NULL) WITHOUT ROWID;

-- terms that contain a number: the key with number words / roman numerals turned into digits ("crisprcas9"), for lookup of spoken forms
CREATE TABLE term_numeric (nkey TEXT NOT NULL, term_id INTEGER NOT NULL, PRIMARY KEY (nkey, term_id)) WITHOUT ROWID;

-- ordinary English words (SCOWL, by commonness level 10 = most common .. 80 = rare dictionary words): a word of this list is a real word, not a mishearing
CREATE TABLE english_word (word TEXT PRIMARY KEY, level INTEGER NOT NULL, source_id TEXT NOT NULL) WITHOUT ROWID;
"""

INDEXES = """
CREATE INDEX ix_concept_domain ON concept_domain (domain, concept_id);
CREATE INDEX ix_term_compact ON term (compact_text);
CREATE INDEX ix_term_norm ON term (normalized_text);
CREATE INDEX ix_term_sk ON term (sk);
CREATE INDEX ix_term_concept ON term (concept_id);
CREATE INDEX ix_abbr_short ON abbreviation (norm_short);
CREATE INDEX ix_abbr_concept ON abbreviation (concept_id);
CREATE INDEX ix_rel_b ON relationship (concept_b, relation_type);
CREATE INDEX ix_cue_cue ON context_cue (cue);
CREATE INDEX ix_confusion_heard ON confusion (norm_heard);
CREATE INDEX ix_phonetic ON phonetic_key (algorithm, key, term_id);
CREATE INDEX ix_term_source_source ON term_source (source_id);
CREATE INDEX ix_token_pre ON token_index (pre, len);
CREATE INDEX ix_token_suf ON token_index (suf, len);
CREATE INDEX ix_token_sk ON token_index (sk, len);
CREATE INDEX ix_token_dm ON token_index (dm, len);
"""

# virtual tables are filled after the bulk insert (much faster than maintaining them row by row)
FTS = """
CREATE VIRTUAL TABLE term_fts USING fts5(normalized_text, content='term', content_rowid='term_id',
                                          tokenize='unicode61 remove_diacritics 2', prefix='2 3 4');
"""

REQUIRED_TABLES = ("meta", "source", "concept", "concept_domain", "term", "term_source", "abbreviation", "relationship", "context_cue", "confusion", "phonetic_key", "phonetic_del",
                   "morphology", "token_freq", "english_word", "term_fts")
