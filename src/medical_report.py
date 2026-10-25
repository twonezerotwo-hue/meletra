"""Medical terminology report: terms heard, medical-looking words outside the dictionary, and
phrases that look like mis-heard dictionary terms.

Nothing here edits the transcript. A suspicious phrase is only *listed* (with timestamp, reason, evidence and a confidence
level) so a human decides; the raw transcript always stays exactly as the model wrote it.

A suggestion needs more than spelling similarity. Each candidate is scored from seven kinds of evidence (see
:func:`score_suggestion`) and ordinary English is protected: a heard word that is a common English word is never "corrected"
("advertise" is not "adverse", "serious" is not "serous"), and a common English word is never offered as the correction
unless the lecture slides use it.
"""

from __future__ import annotations

import difflib
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .common_words import is_common_word
from .pdf_context import _COMMON_FALSE_SUFFIX, _EPONYM, _SUFFIX_WORD
from .priority import medical_specificity
from .schema import Segment
from .vocabulary import VocabularyIndex, is_inflection, normalize_key, tokenize

try:  # rapidfuzz is installed with jiwer; it makes the phrase scan ~100x faster
    from rapidfuzz import fuzz, process
except ImportError:  # pragma: no cover - plain-Python fallback
    fuzz = process = None

# Plain English words that only look like medical terms; never reported as "candidates".
_CANDIDATE_STOPLIST = _COMMON_FALSE_SUFFIX | {
    "question", "questions", "section", "direction", "attention", "information", "important", "different",
    "positive", "negative", "relative", "active", "passive", "native", "creative", "describe", "provide",
    "analysis", "emphasis", "basis", "thesis", "diagnosis", "hypothesis",
}

# A phrase that starts or ends with one of these is a piece of a sentence, not a term ("the regulation", "hepatocytes in").
_STOPWORDS = frozenset("""a an the of in on at to and or is are was were be been with for by from that this these those it its
as but not so if then than also we you they i he she can will may would should could do does did has have had there here
which who what when where how why very more most some any each every one two three about into over under between""".split())

#: weights of the evidence behind a suggestion (they sum to 1.0)
SCORE_WEIGHTS = {"phonetic_similarity": 0.30, "medical_specificity": 0.15, "subject_relevance": 0.15,
                 "local_context": 0.10, "pdf_evidence": 0.20, "neighboring_medical_words": 0.10}
HIGH_SCORE, MEDIUM_SCORE = 0.75, 0.60
DEFAULT_MIN_SCORE = 0.45
NONWORD_BONUS = 0.12             # added when the heard text is NOT an ordinary English word and is a near-exact sound-alike


@dataclass
class TermStat:
    term: str
    count: int
    first_start: float
    kind: str = "dictionary"          # "dictionary" | "candidate"


@dataclass
class PossibleError:
    original: str
    suggestion: str
    start: float
    reason: str
    confidence: float                 # similarity 0-1 between the heard text and the dictionary term
    kind: str = "word"                # "word" | "phrase"
    segment_index: int = -1
    subject: str = ""                 # vocabulary file(s) the suggested term belongs to
    pdf_evidence: str = ""            # whether the slides mention the suggested term
    context: str = ""                 # the sentence it was found in (for a quick listen/check)
    score: float = 0.0                # combined evidence score 0-1 (see SCORE_WEIGHTS)
    components: dict[str, float] = field(default_factory=dict)
    label: str = ""                   # high | medium | low (from the score)

    @property
    def similarity(self) -> float:
        return self.confidence

    @property
    def confidence_label(self) -> str:
        if self.label:
            return self.label
        return "high" if self.confidence >= 0.95 else ("medium" if self.confidence >= 0.90 else "low")

    def to_dict(self) -> dict:
        return {"timestamp": round(self.start, 2), "original": self.original, "possible_term": self.suggestion,
                "similarity": round(self.confidence, 3), "score": round(self.score, 3),
                "confidence": self.confidence_label, "subject": self.subject, "pdf_evidence": self.pdf_evidence,
                "reason": self.reason, "kind": self.kind, "context": self.context,
                "components": {k: round(v, 3) for k, v in self.components.items()}}


def _word_time(segment: Segment, index: int, total_words: int) -> float:
    """Best-effort time of the ``index``-th word: real word timestamps when they line up, else interpolation."""
    if segment.words and len(segment.words) == total_words:
        return segment.words[index].start
    span = max(segment.end - segment.start, 0.0)
    return segment.start + span * (index / max(total_words, 1))


def detect_terms(segments: Sequence[Segment], vocab: VocabularyIndex) -> list[TermStat]:
    """Dictionary terms heard in the transcript with counts and first timestamp, most frequent first."""
    counts: Counter[str] = Counter()
    first: dict[str, float] = {}
    for segment in segments:
        words = tokenize(segment.text)
        for begin, _end, term in vocab.find_term_spans(words):
            counts[term] += 1
            first.setdefault(term, _word_time(segment, begin, len(words)))
    return [TermStat(t, n, first[t]) for t, n in counts.most_common()]


def detect_candidate_terms(
    segments: Sequence[Segment], vocab: VocabularyIndex, min_count: int = 2,
) -> list[TermStat]:
    """Medical-looking words (eponyms, -itis/-osis/-emia...) that are NOT in the dictionary.

    The starter dictionaries are small, so this shows what a real lecture uses beyond them and is
    the list to extend ``medical_terms/*.txt`` with.
    """
    counts: Counter[str] = Counter()
    display: dict[str, str] = {}
    first: dict[str, float] = {}
    known_words = {normalize_key(w) for term in vocab.terms for w in tokenize(term)}
    for segment in segments:
        for pattern in (_EPONYM, _SUFFIX_WORD):
            for match in pattern.finditer(segment.text):
                word = match.group(1)
                key = normalize_key(word)
                if len(key) < 6 or word.casefold() in _CANDIDATE_STOPLIST or key in known_words:
                    continue
                if vocab.canonical(word) is not None:
                    continue
                counts[key] += 1
                display.setdefault(key, word)
                first.setdefault(key, segment.start)
    return [TermStat(display[k], n, first[k], "candidate") for k, n in counts.most_common() if n >= min_count]


def _best_match(key: str, term_keys: list[str], cutoff: float) -> Optional[tuple[str, float]]:
    """Most similar dictionary key to ``key`` with ratio >= cutoff (0-1), or None."""
    if process is not None:
        hit = process.extractOne(key, term_keys, scorer=fuzz.ratio, score_cutoff=cutoff * 100)
        return None if hit is None else (hit[0], hit[1] / 100.0)
    close = difflib.get_close_matches(key, term_keys, n=1, cutoff=cutoff)
    if not close:
        return None
    return close[0], difflib.SequenceMatcher(None, key, close[0]).ratio()


def _evidence_for(suggestion: str, pdf_keys: set[str], pdf_blob: str, have_pdf: bool) -> str:
    if not have_pdf:
        return "no lecture PDF supplied"
    if _in_pdf(suggestion, pdf_keys, pdf_blob):
        return f"yes: '{suggestion}' appears in the slides"
    return "no: not found in the slides"


def _in_pdf(suggestion: str, pdf_keys: set[str], pdf_blob: str) -> bool:
    key = normalize_key(suggestion)
    return key in pdf_keys or (len(key) >= 7 and key in pdf_blob)


# ----------------------------------------------------------------------------------------------- scoring
def phonetic_key(text: str) -> str:
    """A small Metaphone-style key: spelling variants that SOUND alike (aspartate/asperate) get similar keys."""
    word = re.sub(r"[^a-z]", "", text.lower())
    if not word:
        return ""
    for old, new in (("ph", "f"), ("ck", "k"), ("gh", ""), ("kn", "n"), ("wr", "r"), ("sch", "sk"), ("ch", "k"), ("sh", "x"),
                     ("th", "t"), ("qu", "kw"), ("x", "ks"), ("q", "k"), ("z", "s"), ("v", "f"), ("w", "")):
        word = word.replace(old, new)
    word = re.sub(r"c(?=[eiy])", "s", word).replace("c", "k")
    head, tail = word[0], re.sub(r"[aeiouyh]", "", word[1:])
    key = head + tail
    return re.sub(r"(.)\1+", r"\1", key)


def phonetic_similarity(heard: str, term: str) -> float:
    """0-1: half spelling similarity, half similarity of the phonetic keys."""
    a, b = normalize_key(heard), normalize_key(term)
    spelling = difflib.SequenceMatcher(None, a, b).ratio()
    sound = difflib.SequenceMatcher(None, phonetic_key(heard), phonetic_key(term)).ratio()
    return 0.5 * spelling + 0.5 * sound


def score_suggestion(
    *, heard: str, suggestion: str, ratio: float, subject_files: Sequence[str], active_subjects: Sequence[str],
    in_pdf: bool, local_terms: int, neighbor_medical: int, heard_is_nonword: bool = False,
) -> tuple[float, dict[str, float]]:
    """Combined evidence score 0-1 for "``heard`` was really ``suggestion``" and the numbers behind it."""
    if active_subjects:
        if set(subject_files) & set(active_subjects):
            subject = 1.0
        elif "general_medical" in subject_files:
            subject = 0.3
        else:
            subject = 0.1 if subject_files else 0.0
    else:
        subject = 0.5 if subject_files else 0.0                 # nothing is known about the lecture's subject: neutral
    components = {
        "phonetic_similarity": max(phonetic_similarity(heard, suggestion), ratio * 0.9),
        "medical_specificity": medical_specificity(suggestion),
        "subject_relevance": subject,
        "local_context": min(1.0, local_terms / 3.0),
        "pdf_evidence": 1.0 if in_pdf else 0.0,
        "neighboring_medical_words": min(1.0, neighbor_medical / 2.0),
    }
    score = sum(SCORE_WEIGHTS[name] * value for name, value in components.items())
    if heard_is_nonword and ratio >= 0.90:
        # "spinothalmic" is not an English word and is one letter away from "spinothalamic": spelling alone is real evidence
        components["nonword_bonus"] = NONWORD_BONUS
        score += NONWORD_BONUS
    return round(min(score, 1.0), 4), {k: round(v, 4) for k, v in components.items()}


def label_for(score: float) -> str:
    return "high" if score >= HIGH_SCORE else ("medium" if score >= MEDIUM_SCORE else "low")


def _common_prefix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _same_family(a: str, b: str) -> bool:
    """Two words of one word family (phosphorylated / phosphorylation, transcriptional / transcription): not a mis-hearing."""
    a, b = normalize_key(a), normalize_key(b)
    if a == b or is_inflection(a, b):
        return True
    prefix = _common_prefix(a, b)
    return prefix >= 8 and abs(len(a) - len(b)) <= 4 and prefix >= min(len(a), len(b)) - 3


def is_distinct_entity(original_words: Sequence[str], suggestion: str, known_words: set[str]) -> bool:
    """True when ``original_words`` is a valid term of its own that merely looks like ``suggestion``.

    * "protein kinase B" / "type 2" next to "protein kinase A" / "type 1": the same name with a different one-character label;
    * "inositol-3" next to "inositol": the term with a number attached;
    * "phosphorylated" when the dictionary knows "phosphorylation" (another word of the same family); "transcriptional factor"
      next to "transcription factor" (every word of the phrase is the same word or a family member).
    Measured on a real lecture: all four kinds were being offered as "possible mis-hearings".
    """
    original_key = normalize_key("".join(original_words))
    suggestion_key = normalize_key(suggestion)
    suggestion_words = tokenize(suggestion)
    if len(original_words) == len(suggestion_words) and len(original_words) >= 2:
        same_head = all(normalize_key(a) == normalize_key(b) for a, b in zip(original_words[:-1], suggestion_words[:-1]))
        last_a, last_b = normalize_key(original_words[-1]), normalize_key(suggestion_words[-1])
        if same_head and last_a != last_b and len(last_a) <= 2 and len(last_b) <= 2:
            return True
        if all(_same_family(a, b) for a, b in zip(original_words, suggestion_words)):
            return True
    if original_key.startswith(suggestion_key) and original_key[len(suggestion_key):].isdigit():
        return True
    if len(original_words) == 1 and len(original_key) >= 9:
        own = {normalize_key(w) for w in suggestion_words}
        return any(w not in own and w != original_key and _common_prefix(original_key, w) >= 9
                   and _common_prefix(original_key, w) >= min(len(original_key), len(w)) - 4 for w in known_words)
    return False


def _looks_medical(word: str, known_words: set[str]) -> bool:
    key = normalize_key(word)
    return len(key) >= 6 and (key in known_words or not is_common_word(word))


def scan_possible_errors(
    segments: Sequence[Segment],
    vocab: VocabularyIndex,
    cutoff: float = 0.86,
    min_word_length: int = 7,
    min_phrase_length: int = 10,
    max_phrase_words: int = 4,
    term_subjects: Optional[Mapping[str, Sequence[str]]] = None,
    pdf_terms: Sequence[str] = (),
    pdf_text: str = "",
    active_subjects: Sequence[str] = (),
    min_score: float = DEFAULT_MIN_SCORE,
) -> list[PossibleError]:
    """Find words / short phrases that closely resemble a dictionary term without being one - and are plausibly that term.

    Examples: ``cortical spinal tract`` ~ ``corticospinal tract``, ``asperate`` ~ ``aspartate``. Exact dictionary terms (also
    split / hyphenated / differently cased), plain inflections (``receptors``/``receptor``), ordinary English words and
    phrases that begin or end with a function word are not reported. A candidate is listed only if its evidence score
    (:func:`score_suggestion`) reaches ``min_score``.
    """
    keys = list(vocab.term_keys)
    results: list[PossibleError] = []
    seen: set[tuple[int, str]] = set()
    pdf_keys = {normalize_key(t) for t in pdf_terms}
    pdf_blob = normalize_key(pdf_text)
    have_pdf = bool(pdf_terms or pdf_text)
    term_subjects = term_subjects or {}
    known_words = {normalize_key(w) for term in vocab.terms for w in tokenize(term)}
    word_files: dict[str, set[str]] = {}                        # single words of multi-word terms -> the files of that term
    for term in vocab.terms:
        files = term_subjects.get(normalize_key(term), ())
        for word in tokenize(term):
            word_files.setdefault(normalize_key(word), set()).update(files)
    token_lists = [tokenize(s.text) for s in segments]
    for index, segment in enumerate(segments):
        words = token_lists[index]
        covered = [False] * len(words)
        spans = vocab.find_term_spans(words)
        for begin, end, _ in spans:
            for i in range(begin, end):
                covered[i] = True
        neighbourhood = " ".join(segments[j].text for j in range(max(0, index - 1), min(len(segments), index + 2)))
        local_terms_all = vocab.find_terms(neighbourhood)
        for size in range(max_phrase_words, 0, -1):
            for begin in range(0, len(words) - size + 1):
                if any(covered[begin:begin + size]):
                    continue
                phrase_words = words[begin:begin + size]
                if size > 1 and (phrase_words[0].lower() in _STOPWORDS or phrase_words[-1].lower() in _STOPWORDS):
                    continue
                joined = normalize_key("".join(phrase_words))
                if size == 1:
                    if len(joined) < min_word_length or is_common_word(phrase_words[0]):
                        continue                                     # ordinary English is never "corrected"
                    suggestion = vocab.near_miss(phrase_words[0], cutoff, min_word_length)
                    if suggestion is None:
                        continue
                    ratio = difflib.SequenceMatcher(None, joined, normalize_key(suggestion)).ratio()
                    original, kind = phrase_words[0], "word"
                else:
                    if len(joined) < min_phrase_length:
                        continue
                    hit = _best_match(joined, keys, cutoff)
                    if hit is None:
                        continue
                    key, ratio = hit
                    if joined == key or abs(len(key) - len(joined)) > 3 or key[0] != joined[0] or is_inflection(joined, key):
                        continue
                    suggestion = vocab.term_keys[key]
                    original, kind = " ".join(phrase_words), "phrase"
                    if all(is_common_word(w) for w in phrase_words) and ratio < 0.92:
                        continue                                     # different ordinary words, not one mis-heard term
                in_pdf = have_pdf and _in_pdf(suggestion, pdf_keys, pdf_blob)
                if len(tokenize(suggestion)) == 1 and is_common_word(suggestion) and not in_pdf:
                    continue                                         # never offer an ordinary word as the "medical" fix
                if is_distinct_entity(phrase_words, suggestion, known_words):
                    continue                                         # a valid term of its own, not a mis-hearing
                marker = (index, normalize_key(original))
                if marker in seen:
                    continue
                suggestion_key = normalize_key(suggestion)
                local_terms = sum(1 for t in local_terms_all if normalize_key(t) != suggestion_key)
                around = words[max(0, begin - 3):begin] + words[begin + size:begin + size + 3]
                neighbor_medical = sum(1 for w in around if _looks_medical(w, known_words))
                subject_files = sorted(term_subjects.get(suggestion_key) or word_files.get(suggestion_key, ()))
                nonword = not all(is_common_word(w) for w in phrase_words)
                score, components = score_suggestion(
                    heard=original, suggestion=suggestion, ratio=ratio, subject_files=subject_files,
                    active_subjects=active_subjects, in_pdf=in_pdf, local_terms=local_terms,
                    neighbor_medical=neighbor_medical, heard_is_nonword=nonword)
                if score < min_score:
                    continue
                seen.add(marker)
                for i in range(begin, begin + size):
                    covered[i] = True      # a longer phrase wins over the words inside it
                label = label_for(score)
                results.append(PossibleError(
                    original=original, suggestion=suggestion, start=_word_time(segment, begin, len(words)),
                    reason=(f"{'phrase' if kind == 'phrase' else 'word'} is {ratio:.0%} similar to dictionary term "
                            f"'{suggestion}' but is not an exact match (possible mis-hearing); evidence score {score:.2f}"),
                    confidence=round(ratio, 3), kind=kind, segment_index=index,
                    subject=", ".join(subject_files),
                    pdf_evidence=_evidence_for(suggestion, pdf_keys, pdf_blob, have_pdf),
                    context=segment.text.strip()[:140], score=score, components=components, label=label,
                ))
    results.sort(key=lambda e: e.start)
    return results


@dataclass
class TermReport:
    terms: list[TermStat] = field(default_factory=list)
    candidates: list[TermStat] = field(default_factory=list)
    errors: list[PossibleError] = field(default_factory=list)


def build_report(segments: Sequence[Segment], vocab: VocabularyIndex, cutoff: float = 0.86,
                 min_word_length: int = 7, candidate_min_count: int = 2,
                 term_subjects: Optional[Mapping[str, Sequence[str]]] = None, pdf_terms: Sequence[str] = (),
                 pdf_text: str = "", active_subjects: Sequence[str] = (), min_score: float = DEFAULT_MIN_SCORE
                 ) -> TermReport:
    return TermReport(
        terms=detect_terms(segments, vocab),
        candidates=detect_candidate_terms(segments, vocab, candidate_min_count),
        errors=scan_possible_errors(segments, vocab, cutoff, min_word_length, term_subjects=term_subjects,
                                    pdf_terms=pdf_terms, pdf_text=pdf_text, active_subjects=active_subjects,
                                    min_score=min_score),
    )


def term_subject_map(terms_dir: Optional[Path] = None) -> dict[str, list[str]]:
    """normalised term key -> vocabulary files that contain it (the 'Subject' column of the error report)."""
    from .vocabulary import DEFAULT_TERMS_DIR, available_subjects, load_terms

    base = terms_dir or DEFAULT_TERMS_DIR
    out: dict[str, list[str]] = {}
    for subject in available_subjects(base):
        for term in load_terms(subject, None, base):
            out.setdefault(normalize_key(term), []).append(subject)
    return out
