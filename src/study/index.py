"""The lightweight local document index: transcript and slides cut into citable chunks with page / time metadata, grouped into topics,
searchable with a small BM25 (no vector database, no model download). RAG-compatible: ``search`` returns the chunks a language model
should read for a question; the same call serves the built-in method.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from .models import SOURCE_PDF, SOURCE_TRANSCRIPT, Chunk
from .text import STOPWORDS, clean, content_words, stem, title_case, words

TRANSCRIPT_CHUNK_WORDS = 90            # a transcript chunk closes after about this many words ...
TRANSCRIPT_CHUNK_MAX_WORDS = 170       # ... and never holds more than this
TRANSCRIPT_GAP_SECONDS = 8.0           # a pause this long also closes a chunk
PDF_CHUNK_WORDS = 160                  # slide text longer than this is split by paragraph
MIN_SECTION_CHUNKS = 4
MAX_SECTION_CHUNKS = 12
LINK_MIN_SIMILARITY = 0.12
LF, CR = chr(10), chr(13)


def _terms(text: str) -> list[str]:
    return [stem(w) for w in content_words(text)]


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = sum(v * b.get(k, 0.0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


@dataclass
class TopicSection:
    name: str
    chunk_ids: list[str] = field(default_factory=list)           # transcript chunks, in lecture order
    pages: list[int] = field(default_factory=list)               # slides that talk about it
    start: float = 0.0
    end: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "chunk_ids": self.chunk_ids, "pages": self.pages, "start": self.start, "end": self.end}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TopicSection":
        return cls(data["name"], list(data.get("chunk_ids", [])), list(data.get("pages", [])), float(data.get("start", 0.0)),
                   float(data.get("end", 0.0)))


class DocumentIndex:
    def __init__(self, chunks: Sequence[Chunk], topics: Optional[Sequence[TopicSection]] = None,
                 links: Optional[dict[str, list[str]]] = None) -> None:
        self.chunks: list[Chunk] = list(chunks)
        self.by_id: dict[str, Chunk] = {c.id: c for c in self.chunks}
        self.topics: list[TopicSection] = list(topics or [])
        self.links: dict[str, list[str]] = dict(links or {})             # transcript chunk id -> slide chunk ids
        self.glossary: list[str] = []                                    # known medical terms (only used to name topics / rank terms)
        self.unresolved: list[str] = []                                  # words the medical pass could not resolve: no study item may rest on them
        self.lecture_terms: list[str] = []                               # LectureVocabulary: terms of the slides / PDF
        self.meta: dict[str, Any] = {}                                   # subject, duration, which transcript the material comes from
        self._tokens: dict[str, list[str]] = {c.id: _terms(c.text) for c in self.chunks}
        self._df: Counter = Counter()
        for tokens in self._tokens.values():
            self._df.update(set(tokens))
        self._avg_len = (sum(len(t) for t in self._tokens.values()) / len(self._tokens)) if self._tokens else 1.0

    # ------------------------------------------------------------------ access
    def __len__(self) -> int:
        return len(self.chunks)

    def get(self, chunk_id: str) -> Optional[Chunk]:
        return self.by_id.get(chunk_id)

    def of_source(self, source: str) -> list[Chunk]:
        return [c for c in self.chunks if c.source == source]

    @property
    def mode(self) -> str:
        from .models import MODE_BOTH, MODE_PDF, MODE_TRANSCRIPT

        has_t, has_p = bool(self.of_source(SOURCE_TRANSCRIPT)), bool(self.of_source(SOURCE_PDF))
        return MODE_BOTH if has_t and has_p else (MODE_TRANSCRIPT if has_t else MODE_PDF)

    def topic_names(self) -> list[str]:
        return [t.name for t in self.topics]

    def chunks_of_topic(self, name: str) -> list[Chunk]:
        """Transcript chunks of a topic plus the slides linked to it (the PDF-only case: the slides of that topic)."""
        section = next((t for t in self.topics if t.name == name), None)
        if section is None:
            return []
        out = [self.by_id[i] for i in section.chunk_ids if i in self.by_id]
        have = {c.id for c in out}
        for c in self.chunks:
            if c.source == SOURCE_PDF and c.page in section.pages and c.id not in have:
                out.append(c)
                have.add(c.id)
        return out

    def linked_slides(self, chunk: Chunk) -> list[Chunk]:
        return [self.by_id[i] for i in self.links.get(chunk.id, []) if i in self.by_id]

    # ------------------------------------------------------------------ search (BM25)
    def search(self, query: str, k: int = 5, source: Optional[str] = None) -> list[tuple[Chunk, float]]:
        q = _terms(query)
        if not q or not self.chunks:
            return []
        n = len(self.chunks)
        scored = []
        for chunk in self.chunks:
            if source and chunk.source != source:
                continue
            tokens = self._tokens[chunk.id]
            if not tokens:
                continue
            tf = Counter(tokens)
            score = 0.0
            for term in set(q):
                f = tf.get(term, 0)
                if not f:
                    continue
                idf = math.log(1 + (n - self._df[term] + 0.5) / (self._df[term] + 0.5))
                score += idf * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * len(tokens) / self._avg_len))
            if score > 0:
                scored.append((chunk, score))
        scored.sort(key=lambda cs: cs[1], reverse=True)
        return scored[:k]

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict[str, Any]:
        return {"chunks": [c.to_dict() for c in self.chunks], "topics": [t.to_dict() for t in self.topics], "links": self.links,
                "unresolved": self.unresolved, "lecture_terms": self.lecture_terms, "meta": self.meta}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocumentIndex":
        index = cls([Chunk.from_dict(c) for c in data.get("chunks", [])], [TopicSection.from_dict(t) for t in data.get("topics", [])],
                    data.get("links", {}))
        index.unresolved, index.lecture_terms, index.meta = list(data.get("unresolved", [])), list(data.get("lecture_terms", [])), dict(data.get("meta", {}))
        return index


# ---------------------------------------------------------------------------------------------- building
def transcript_chunks(segments: Sequence[dict[str, Any]]) -> list[Chunk]:
    """Consecutive segments grouped into chunks of about 90 words; each keeps its start and end time (a plain-text transcript
    without times gives chunks without times)."""
    chunks: list[Chunk] = []
    buffer: list[str] = []
    spans: list[list] = []                                              # [start, end, confidence, low words] of every segment in the buffer
    start: Optional[float] = None
    end: Optional[float] = None
    count = 0

    def flush() -> None:
        nonlocal buffer, count, start, end, spans
        text = clean(" ".join(buffer))
        if text:
            offsets, at = [], 0
            for piece, (s, e, conf, low) in zip(buffer, spans):         # character range of each segment inside the chunk text
                at = text.find(clean(piece), at) if clean(piece) else at
                if at < 0:
                    offsets = []
                    break
                offsets.append([s, e, at, at + len(clean(piece)), conf, low])
                at += len(clean(piece))
            chunks.append(Chunk(id=f"t{len(chunks):04d}", source=SOURCE_TRANSCRIPT, text=text,
                                start=None if start is None else round(start, 2), end=None if end is None else round(end, 2),
                                segs=[o for o in offsets if o[0] is not None]))
        buffer, count, start, end, spans = [], 0, None, None, []

    for seg in segments:
        text = clean(str(seg.get("text", "")))
        if not text:
            continue
        s, e = seg.get("start"), seg.get("end")
        s = None if s is None else float(s)
        e = s if e is None else float(e)
        n = len(words(text))
        pause = s is not None and end is not None and s - end > TRANSCRIPT_GAP_SECONDS
        if buffer and (pause or count + n > TRANSCRIPT_CHUNK_MAX_WORDS):
            flush()
        if not buffer:
            start = s
        buffer.append(text)
        spans.append([s, e, seg.get("confidence"), list(seg.get("low_words") or [])])
        end, count = e, count + n
        if count >= TRANSCRIPT_CHUNK_WORDS and text.rstrip().endswith((".", "!", "?", "…")):
            flush()
    flush()
    return chunks


def _pdf_lines(raw: str) -> list[str]:
    return [line for line in (clean(part) for part in raw.replace(CR, "").split(LF)) if line]


def _slide_title(lines: Sequence[str]) -> str:
    for line in lines[:3]:
        if 3 <= len(line) <= 90 and len(words(line)) <= 12:
            return line
    return ""


def pdf_chunks(pages: Sequence[str]) -> list[Chunk]:
    """One chunk per slide (a long page is split in pieces of about 160 words). Line breaks are kept: they separate bullets.
    Pages with no text are skipped; the page number is the number of the page in the PDF."""
    chunks: list[Chunk] = []
    for number, raw in enumerate(pages, start=1):
        lines = _pdf_lines(raw)
        if not lines:
            continue
        title = _slide_title(lines)
        pieces: list[list[str]] = [[]]
        size = 0
        for line in lines:
            if size >= PDF_CHUNK_WORDS:
                pieces.append([])
                size = 0
            pieces[-1].append(line)
            size += len(words(line))
        for k, piece in enumerate(pieces):
            chunk_id = f"p{number:03d}" if k == 0 else f"p{number:03d}-{k + 1}"
            chunks.append(Chunk(id=chunk_id, source=SOURCE_PDF, text=LF.join(piece), page=number, title=title))
    return chunks


def _idf_vectors(token_lists: dict[str, list[str]]) -> dict[str, dict[str, float]]:
    df: Counter = Counter()
    for tokens in token_lists.values():
        df.update(set(tokens))
    n = max(len(token_lists), 1)
    vectors = {}
    for key, tokens in token_lists.items():
        tf = Counter(tokens)
        vectors[key] = {t: (1 + math.log(f)) * math.log(1 + n / df[t]) for t, f in tf.items()}
    return vectors


def _sum_vectors(vectors: Sequence[dict[str, float]]) -> dict[str, float]:
    total: dict[str, float] = {}
    for v in vectors:
        for k, w in v.items():
            total[k] = total.get(k, 0.0) + w
    return total


def _boundaries(vectors: list[dict[str, float]], window: int = 3) -> list[int]:
    """Indices where a new topic starts: the places where the text before and after a point are least alike."""
    n = len(vectors)
    if n < 2 * MIN_SECTION_CHUNKS:
        return []
    dips = []
    for i in range(window, n - window + 1):
        left, right = _sum_vectors(vectors[i - window:i]), _sum_vectors(vectors[i:i + window])
        dips.append((i, _cosine(left, right)))
    values = [s for _i, s in dips]
    mean = sum(values) / len(values)
    spread = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)) or 1e-9
    cut = mean - 0.4 * spread
    picked: list[int] = []
    for i, s in sorted(dips, key=lambda x: x[1]):
        if s > cut:
            break
        if all(abs(i - j) >= MIN_SECTION_CHUNKS for j in picked) and i >= MIN_SECTION_CHUNKS and n - i >= MIN_SECTION_CHUNKS:
            picked.append(i)
    return sorted(picked)


def _split_long(starts: list[int], n: int) -> list[int]:
    """Sections longer than MAX_SECTION_CHUNKS are cut into even parts."""
    marks = [0, *starts, n]
    out = [0]
    for a, b in zip(marks, marks[1:]):
        parts = max(1, math.ceil((b - a) / MAX_SECTION_CHUNKS))
        size = (b - a) / parts
        for p in range(1, parts):
            out.append(a + round(size * p))
        if b < n:
            out.append(b)
    return sorted(set(out))


def _label(glossary: set[str], text: str, distinct: dict[str, float]) -> str:
    """A short topic name: dictionary terms that dominate the section, else the words most typical of it (not of the whole lecture)."""
    low = text.casefold()
    present = set(words(text))
    candidates = [t for t in glossary if len(t) > 3 and t.split()[0].casefold() in present]       # cheap pre-filter for big dictionaries
    glossary_hits = Counter({t: low.count(t.casefold()) for t in candidates if t.casefold() in low})
    named = [t for t, c in glossary_hits.most_common(6) if c >= 2][:3]
    if named:
        return ", ".join(title_case(t) for t in named)
    counts = Counter(w for w in words(text) if len(w) >= 5 and w not in STOPWORDS and "'" not in w)
    ranked = sorted(counts, key=lambda w: counts[w] * distinct.get(w, 1.0), reverse=True)
    return ", ".join(title_case(w) for w in ranked[:3]) or "Lecture"


def build_topics(chunks: list[Chunk], glossary: Sequence[str] = ()) -> list[TopicSection]:
    transcript = [c for c in chunks if c.source == SOURCE_TRANSCRIPT]
    if not transcript:
        return _slide_topics([c for c in chunks if c.source == SOURCE_PDF])
    token_lists = {c.id: _terms(c.text) for c in transcript}
    vectors = [v for v in _idf_vectors(token_lists).values()]
    starts = _split_long(_boundaries(vectors), len(transcript))
    ends = starts[1:] + [len(transcript)]
    groups = [transcript[a:b] for a, b in zip(starts, ends) if transcript[a:b]]
    texts = [" ".join(c.text for c in g) for g in groups]
    section_df: Counter = Counter()
    for text in texts:
        section_df.update({w for w in words(text) if len(w) >= 5})
    distinct = {w: math.log(1 + len(texts) / n) for w, n in section_df.items()}
    gloss = set(glossary)
    sections: list[TopicSection] = []
    used: Counter = Counter()
    for group, text in zip(groups, texts):
        name = _label(gloss, text, distinct)
        used[name] += 1
        if used[name] > 1:
            name = f"{name} ({used[name]})"
        sections.append(TopicSection(name=name, chunk_ids=[c.id for c in group], start=group[0].start or 0.0, end=group[-1].end or 0.0))
    by_id = {c.id: c for c in transcript}
    for section in sections:
        for chunk_id in section.chunk_ids:
            by_id[chunk_id].topic = section.name
    return sections


def _slide_topics(slides: list[Chunk]) -> list[TopicSection]:
    """PDF only: consecutive slides with the same title belong together; every slide group is a topic."""
    sections: list[TopicSection] = []
    for chunk in slides:
        name = chunk.title or f"Slide {chunk.page}"
        if sections and sections[-1].name == name:
            if chunk.page not in sections[-1].pages:
                sections[-1].pages.append(chunk.page or 0)
        else:
            sections.append(TopicSection(name=name, pages=[chunk.page or 0]))
        chunk.topic = name
    return sections


def link_slides(chunks: list[Chunk], topics: list[TopicSection]) -> dict[str, list[str]]:
    """For every transcript chunk the slides that talk about the same thing (transcript + PDF together)."""
    transcript = [c for c in chunks if c.source == SOURCE_TRANSCRIPT]
    slides = [c for c in chunks if c.source == SOURCE_PDF]
    if not transcript or not slides:
        return {}
    vectors = _idf_vectors({c.id: _terms(c.text) for c in chunks})
    links: dict[str, list[str]] = {}
    page_scores: dict[str, Counter] = {t.name: Counter() for t in topics}
    for chunk in transcript:
        scored = sorted(((_cosine(vectors[chunk.id], vectors[s.id]), s) for s in slides), key=lambda x: x[0], reverse=True)
        best = [s for score, s in scored[:2] if score >= LINK_MIN_SIMILARITY]
        if best:
            links[chunk.id] = [s.id for s in best]
            if chunk.topic in page_scores:
                for s in best:
                    page_scores[chunk.topic][s.page] += 1
    for topic in topics:
        pages = [p for p, _n in page_scores.get(topic.name, Counter()).most_common(4) if p]
        topic.pages = sorted(pages)
    return links


def build_index(transcript_segments: Sequence[dict[str, Any]] = (), pdf_pages: Sequence[str] = (),
                glossary: Sequence[str] = ()) -> DocumentIndex:
    """Cut transcript and slides into chunks, find the topics and link each passage to its slides."""
    chunks = transcript_chunks(transcript_segments) + pdf_chunks(pdf_pages)
    topics = build_topics(chunks, glossary)
    links = link_slides(chunks, topics) if transcript_segments and pdf_pages else {}
    for topic in topics:                                                   # a slide belongs to the topic whose passages talk about it
        for chunk in chunks:
            if chunk.source == SOURCE_PDF and not chunk.topic and chunk.page in topic.pages:
                chunk.topic = topic.name
    index = DocumentIndex(chunks, topics, links)
    index.glossary = list(glossary)
    return index
