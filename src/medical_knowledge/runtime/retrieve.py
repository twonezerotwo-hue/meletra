"""Candidate retrieval: from a span of transcript words to 10-50 plausible medical concepts (never millions, never a replacement decision).

Channels (each candidate remembers which ones proposed it):

    exact        the span already IS a known term (spacing / hyphens / accents ignored)
    numeric      the same term with number words / roman numerals read as digits ("Cas nine" ~ "Cas9", "polymerase one" ~ "polymerase I")
    confusion    the span is a KNOWN mishearing of a concept (``confusion`` table)
    phonetic     same consonant skeleton as a term                              (``term.sk``)
    fuzzy        Double Metaphone key within two edits of a term's key          (``phonetic_del``, deletion neighbourhood)
    partial      a multi-word span shares one rare word with a multi-word term ("regulatory conduction" ~ "saltatory conduction"); the rest is judged by sound
    morphological  a related form of a one-word span that a source has: plural / Latin number / adjectival / verbal ending ("emboli" ~ "embolus", "tracheal" ~ "trachea")
    token-sound / token-spelling   a WORD of the vocabulary (not a whole term) that sounds like, or is spelled like, the span or its concatenation
                 ("perineoplastic" ~ "paraneoplastic", "neural pore" ~ "neuropore"); the candidate is the word itself, taken from a term that contains it
    expected     a term the LECTURE makes likely (related to what it already says, named in its slides) that sounds like the span (runtime/expected.py)

Pre-ranking only (sound blend + spelling + field of study); the decision is made later by the scorer from the context. Measured choices: docs/v4/PHONETICS.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence

from rapidfuzz.distance import Levenshtein

from ...medical.phonetics import skeleton
from ..domains import SUPPORT_DOMAINS, expand
from ..indexes.phonetic import DELETION_MIN, deletions, dm_key, mp_key
from ..normalize.morphology import related_forms
from ..normalize.numbers import has_number, numeric_key
from ..normalize.text import compact, letters, normalize, wording_key
from .cache import LRU
from .knowledge import MedicalKnowledge


def _sim(a: str, b: str) -> float:
    return Levenshtein.normalized_similarity(a, b) if a and b else 0.0


EXACT_POOL, FUZZY_POOL, PARTIAL_POOL = 40, 400, 120
TOKEN_POOL, TOKEN_SOUND_POOL, TOKEN_KEEP, TOKEN_MIN_SIM = 4000, 120, 10, 0.70
LECTURE_BOOST = 0.10                       # prior added to a concept the slides / the user's vocabulary name: it is retrieved earlier, never replaced by that alone
PARTIAL_MIN_SOUND, PARTIAL_RARE = 0.5, 600
#: sound blend measured in milestone 4 (skeleton, medical phoneme string, Double Metaphone); spelling is a separate signal
SOUND_WEIGHTS = (0.45, 0.15, 0.40)


@dataclass(frozen=True)
class Candidate:
    concept_id: str
    term_id: int
    surface: str
    preferred_name: str
    language: str
    term_type: str
    trust: str
    domains: tuple[str, ...]
    token_count: int
    channels: tuple[str, ...]
    kind: str
    sound: float                  # 0..1 sound similarity (skeleton + phoneme string + Double Metaphone)
    ortho: float                  # 0..1 similarity of the spellings
    exact: bool                   # the span is exactly this term
    known_confusion: bool         # the span is a recorded mishearing of this concept
    domain_fit: float             # 1 = in a focus field, 0.6 = related field, 0.5 = eponym / Latin / abbreviation (no single field), 0 = neither or no focus given
    prior: float                  # pre-ranking only

    def to_dict(self) -> dict:
        return {"concept_id": self.concept_id, "surface": self.surface, "preferred_name": self.preferred_name, "language": self.language, "trust": self.trust,
                "domains": list(self.domains), "channels": list(self.channels), "kind": self.kind, "sound": round(self.sound, 3), "ortho": round(self.ortho, 3),
                "exact": self.exact, "known_confusion": self.known_confusion, "domain_fit": round(self.domain_fit, 2), "prior": round(self.prior, 3)}


def sound_similarity(heard: str, term: str) -> float:
    """0..1: how alike two spans sound (skeleton, medical phoneme string, Double Metaphone)."""
    ws, wm, wd = SOUND_WEIGHTS
    return ws * _sim(skeleton(heard), skeleton(term)) + wm * _sim(mp_key(heard), mp_key(term)) + wd * _sim(dm_key(heard), dm_key(term))


class CandidateRetriever:
    def __init__(self, kb: MedicalKnowledge, limit: int = 30, cache_size: int = 1024) -> None:
        self.kb = kb
        self.limit = limit
        self._cache: LRU = LRU(cache_size)

    # ------------------------------------------------------------------ public
    def retrieve(self, span: str, domains: Sequence[str] = (), limit: Optional[int] = None, languages: Sequence[str] = ("en", "la"),
                 extra: Optional[dict[int, set[str]]] = None, boost: Optional[set] = None, trace: Optional[dict] = None) -> list[Candidate]:
        """``extra``: term ids proposed by another stage (term id -> channel names); ``boost``: concept ids the lecture names; ``trace``: filled with what each channel found
        (for diagnostics). Calls with any of the three are not cached."""
        key = compact(span)
        if len(key) < 2:
            return []
        limit = limit or self.limit
        if extra or boost or trace is not None:
            return self._retrieve(span, tuple(domains), limit, tuple(languages), extra, boost, trace)
        return list(self._cache.get((key, tuple(domains), limit, tuple(languages)), lambda: self._retrieve(span, tuple(domains), limit, tuple(languages))))

    # ------------------------------------------------------------------ channels
    def _retrieve(self, span: str, domains: tuple[str, ...], limit: int, languages: tuple[str, ...], extra: Optional[dict[int, set[str]]] = None,
                  boost: Optional[set] = None, trace: Optional[dict] = None) -> list[Candidate]:
        db = self.kb.db
        comp, sk, dm = compact(span), skeleton(span), dm_key(span)
        channels: dict[int, set[str]] = {}
        confused: set[str] = set()
        for tid, names in (extra or {}).items():
            channels.setdefault(tid, set()).update(names)

        for (tid,) in db.execute("SELECT term_id FROM term WHERE compact_text = ? LIMIT ?", (comp, EXACT_POOL)):
            channels.setdefault(tid, set()).add("exact")
        for (cid,) in db.execute("SELECT concept_id FROM confusion WHERE norm_heard = ?", (comp,)):
            confused.add(cid)
            for (tid,) in db.execute("SELECT term_id FROM term WHERE concept_id = ? ORDER BY is_preferred DESC, token_count LIMIT 3", (cid,)):
                channels.setdefault(tid, set()).add("confusion")
        if len(sk) >= 3:
            for (tid,) in db.execute("SELECT term_id FROM term WHERE sk = ? ORDER BY is_preferred DESC, frequency DESC LIMIT ?", (sk, FUZZY_POOL)):
                channels.setdefault(tid, set()).add("phonetic")
        if len(dm) >= max(DELETION_MIN, 3):
            keys = sorted(deletions(dm))
            marks = ",".join("?" * len(keys))
            for (tid,) in db.execute(f"SELECT DISTINCT term_id FROM phonetic_del WHERE key IN ({marks}) LIMIT ?", (*keys, FUZZY_POOL)):
                channels.setdefault(tid, set()).add("fuzzy")
        words = normalize(span).split()
        if len(words) >= 2:
            self._partial(words, channels, comp, sk)
        if self.kb.has_token_index and has_number(span):
            nkey = numeric_key(span)
            if nkey != comp:
                for (tid,) in db.execute("SELECT term_id FROM term_numeric WHERE nkey = ? UNION SELECT term_id FROM term WHERE compact_text = ? LIMIT ?", (nkey, nkey, EXACT_POOL)):
                    channels.setdefault(tid, set()).add("numeric")
        if len(words) == 1 and len(comp) >= 5:
            forms = sorted(related_forms(comp))
            for start in range(0, len(forms), 400):
                chunk = forms[start:start + 400]
                marks = ",".join("?" * len(chunk))
                for (tid,) in db.execute(f"SELECT term_id FROM term WHERE compact_text IN ({marks}) LIMIT ?", (*chunk, EXACT_POOL)):
                    channels.setdefault(tid, set()).add("morphological")
        tokens = self._token_hits(comp, sk, dm)
        if trace is not None:
            trace["channels"] = {tid: set(c) for tid, c in channels.items()}
            trace["tokens"] = {tok: set(c) for tok, (c, _sk, _dm) in tokens.items()}
        if not channels and not tokens:
            return []
        ranked = self._rank(span, sk, dm, channels, confused, domains, limit, languages, boost, trace) if channels else []
        if tokens:
            ranked = self._merge_tokens(span, comp, sk, dm, tokens, ranked, domains, limit, boost)
        return ranked

    # ------------------------------------------------------------------ the words of the vocabulary
    def _token_hits(self, comp: str, sk: str, dm: str) -> dict[str, tuple[set[str], str, str]]:
        """Vocabulary words that sound like or are spelled like ``comp`` (a word, or several words written together): token -> (channels, sk, dm)."""
        kb = self.kb
        if not kb.has_token_index or len(comp) < 4:
            return {}
        db = kb.db
        n = len(comp)
        lo, hi = max(4, n - 3), n + 3
        found: dict[str, tuple[set[str], str, str]] = {}
        ordinary = "AND (level = 0 OR level >= 70)"                                    # an ordinary English word is not an answer: it would be a replacement by "cell"
        for column, key in (("sk", sk), ("dm", dm)):
            if len(key) >= 3:
                for token, tsk, tdm in db.execute(f"SELECT token, sk, dm FROM token_index WHERE {column} = ? AND len BETWEEN ? AND ? {ordinary} LIMIT ?", (key, lo, hi, TOKEN_SOUND_POOL)):
                    if token != comp:
                        found.setdefault(token, (set(), tsk, tdm))[0].add("token-sound")
        if n >= 4:
            minimum = TOKEN_MIN_SIM if n >= 6 else TOKEN_MIN_SIM + 0.04            # a short word needs a closer spelling
            for column, key in (("pre", comp[:3]), ("suf", comp[-4:])):
                if column == "suf" and n < 6:
                    continue
                for token, tsk, tdm in db.execute(f"SELECT token, sk, dm FROM token_index WHERE {column} = ? AND len BETWEEN ? AND ? {ordinary} LIMIT ?", (key, lo, hi, TOKEN_POOL)):
                    if token != comp and _sim(comp, token) >= minimum:
                        found.setdefault(token, (set(), tsk, tdm))[0].add("token-spelling")
        return found

    def _merge_tokens(self, span: str, comp: str, sk: str, dm: str, tokens: dict, ranked: list[Candidate], domains: tuple[str, ...], limit: int,
                      boost: Optional[set]) -> list[Candidate]:
        kb = self.kb
        taken = {wording_key(c.surface) for c in ranked}                          # a word that a whole-term candidate already spells (other spelling variants included) adds nothing
        heard_mp = mp_key(span)
        ws, wm, wd = SOUND_WEIGHTS
        scored = []
        for token, (ch, tsk, tdm) in tokens.items():
            if wording_key(token) in taken:
                continue
            sound = ws * _sim(sk, tsk) + wm * _sim(heard_mp, mp_key(token)) + wd * _sim(dm, tdm)
            ortho = _sim(comp, token)
            scored.append((0.55 * sound + 0.25 * ortho, token, ch, sound, ortho))
        scored.sort(reverse=True)
        core = set(domains)
        near = set(expand(list(domains))) - core if domains else set()
        extra: list[Candidate] = []
        for base, token, ch, sound, ortho in scored[:TOKEN_KEEP]:
            row = kb.term_with_token(token)
            if row is None:
                continue
            tid, cid, text, language, _term_type, trust, _tokens, name, kind = row
            if language not in ("en", "la"):
                continue
            surface = next((w for w in re.split(r"[^A-Za-z0-9]+", text) if normalize(w) == token), token)
            fields = kb.concept_domains_many([cid]).get(cid, ())
            if not domains:
                fit = 0.0
            elif core.intersection(fields):
                fit = 1.0
            elif near.intersection(fields):
                fit = 0.6
            elif fields and all(d in SUPPORT_DOMAINS for d in fields):
                fit = 0.5
            else:
                fit = 0.0
            prior = base + 0.12 * fit + (LECTURE_BOOST if boost and cid in boost else 0.0)
            extra.append(Candidate(cid, tid, surface, text, language, "token", trust, fields, 1, tuple(sorted(ch)), "word", sound, ortho, False, False, fit, prior))
        if not extra:
            return ranked
        merged = ranked + extra
        merged.sort(key=lambda c: (-c.prior, c.token_count, c.surface))
        return merged[:limit]

    def _partial(self, words: list[str], channels: dict[int, set[str]], comp: str, sk: str) -> None:
        """Terms of about the same length that contain one of the span's (rare) words exactly; kept only when the whole phrase still sounds alike."""
        db = self.kb.db
        n = len(words)
        for word in words:
            if len(word) < 5 or self.kb.token_frequency(word) > PARTIAL_RARE:
                continue
            for tid, comp_t, tsk in db.execute("SELECT t.term_id, t.compact_text, t.sk FROM term t WHERE t.term_id IN "
                                               "(SELECT rowid FROM term_fts WHERE term_fts MATCH ? LIMIT ?) AND t.token_count BETWEEN ? AND ?",
                                               ('"' + word + '"', PARTIAL_POOL, n - 1, n + 1)):
                if tid in channels or comp_t == comp:
                    continue
                if _sim(sk, tsk) >= PARTIAL_MIN_SOUND:
                    channels.setdefault(tid, set()).add("partial")

    # ------------------------------------------------------------------ ranking
    def _rank(self, span: str, sk: str, dm: str, channels: dict[int, set[str]], confused: set[str], domains: tuple[str, ...], limit: int,
              languages: tuple[str, ...], boost: Optional[set] = None, trace: Optional[dict] = None) -> list[Candidate]:
        db = self.kb.db
        ids = list(channels)
        # stage 1: sound only (cheap): keep the best few hundred -> no per-candidate SQL for the rest
        stage: list[tuple[float, int]] = []
        keys: dict[int, tuple[str, str]] = {}
        for start in range(0, len(ids), 500):
            chunk = ids[start:start + 500]
            marks = ",".join("?" * len(chunk))
            for tid, tsk, tdm in db.execute(f"SELECT term_id, sk, dm FROM term WHERE term_id IN ({marks})", chunk):
                keys[tid] = (tsk, tdm)
                bonus = 0.3 if channels[tid] & {"exact", "confusion"} else 0.0
                stage.append((0.5 * _sim(sk, tsk) + 0.5 * _sim(dm, tdm) + bonus, tid))
        stage.sort(reverse=True)
        keep = [tid for _, tid in stage[: max(limit * 4, 80)]]
        if trace is not None:
            trace["kept"] = set(keep)
        # stage 2: details, fields of study (one query) and the full pre-ranking of the survivors
        core = set(domains)
        near = set(expand(list(domains))) - core if domains else set()
        heard_letters, heard_mp = letters(span), mp_key(span)
        ws, wm, wd = SOUND_WEIGHTS
        rows = []
        for start in range(0, len(keep), 500):
            chunk = keep[start:start + 500]
            marks = ",".join("?" * len(chunk))
            rows += db.execute("SELECT t.term_id, t.concept_id, t.text, t.language, t.term_type, t.trust, t.token_count, t.is_preferred, c.preferred_name, c.kind "
                               f"FROM term t JOIN concept c ON c.concept_id = t.concept_id WHERE t.term_id IN ({marks})", chunk).fetchall()
        fields = self.kb.concept_domains_many(r[1] for r in rows)
        best: dict[str, Candidate] = {}
        for tid, cid, text, language, term_type, trust, tokens, preferred, name, kind in rows:
            if language not in languages:
                continue
            ch = channels[tid]
            tsk, tdm = keys[tid]
            sound = ws * _sim(sk, tsk) + wm * _sim(heard_mp, mp_key(text)) + wd * _sim(dm, tdm)
            ortho = _sim(heard_letters, letters(text))
            concept_domains = fields.get(cid, ())
            if not domains:
                fit = 0.0
            elif core.intersection(concept_domains):
                fit = 1.0
            elif near.intersection(concept_domains):
                fit = 0.6
            elif concept_domains and all(d in SUPPORT_DOMAINS for d in concept_domains):
                fit = 0.5                                                  # eponyms, Latin forms, abbreviations belong to no single field
            else:
                fit = 0.0
            known = cid in confused and "confusion" in ch
            prior = 0.55 * sound + 0.25 * ortho + 0.12 * fit + (0.2 if known else 0.0) + (0.03 if preferred else 0.0) + (LECTURE_BOOST if boost and cid in boost else 0.0)
            cand = Candidate(cid, tid, text, name, language, term_type or "", trust, concept_domains, tokens, tuple(sorted(ch)), kind or "term", sound, ortho, "exact" in ch, known, fit, prior)
            old = best.get(cid)
            if old is None or cand.prior > old.prior:
                best[cid] = cand
        ranked = sorted(best.values(), key=lambda c: (-c.prior, c.token_count, c.surface))
        return ranked[:limit]
