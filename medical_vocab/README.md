# medical_vocab

Modular, versioned, plain-text vocabulary used by the **medical expert pass** (`src/medical/`). It is *not* a replacement table.
A term is only ever proposed for a span of the transcript when the **sound** (consonant skeleton) and the **context** (cue words, neighbouring
terms, lecture topic, slides, terms the lecture already used) agree. Exact terms are never altered; doubtful ones are only reported.

```
medical_vocab/
  <domain>/            anatomy, neuroanatomy, histology, embryology, physiology, biochemistry, molecular_biology, genetics, pharmacology,
                       pathology, microbiology, immunology, neuroscience, radiology, surgery, internal_medicine, cardiology, neurology,
                       gastroenterology, endocrinology, hematology, nephrology, pulmonology, dermatology, ophthalmology, otolaryngology,
                       psychiatry, orthopedics, reproductive_medicine, laboratory_medicine, latin, abbreviations, eponyms
    _domain.json       {"title", "version", "related": [domains whose terms are also plausible in this lecture]}
    *.txt              one concept per line
  manifest.json  SOURCES.json
```

Line format (`|` separated, everything after the first field optional, `#` starts a comment):

```
canonical | aliases; ... | cues; ... | confusables; ... | kind
Nissl body | Nissl bodies; Nissl substance | rough endoplasmic reticulum; ribosomes; soma; stain | | term
```

* **canonical** - the spelling that is written (capitalisation and diacritics are authoritative: `Nissl`, `GABA`, `mTOR`, `Santiago Ramón y Cajal`).
* **aliases** - other *correct* forms (plurals, hyphenation, English/Latin pairs). Matched exactly and never "corrected".
* **cues** - words and phrases that usually appear near the term. They are what makes a correction believable.
* **confusables** - generic ways ASR mishears it. They lower the sound threshold but still need context.
* **kind** - `term` (default), `abbr`, `eponym`, `latin`, `gene`, `drug` (drug names need stronger evidence).

Rules for editors: write the real name (no "..." suffixes such as "mucosa layer"), keep cues specific (a cue shared by many terms counts for little),
never put a lecture-specific sentence or a "blind replacement" in `confusables`. Check with `python scripts/lint_medical_vocab.py`.
Latin terms list their plurals explicitly (`foramen | foramina`); Latin and English forms are never converted into each other.
