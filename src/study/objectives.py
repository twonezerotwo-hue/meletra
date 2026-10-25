"""Learning objectives of a lecture, worked out from the study material itself.

An objective here is something the student will be able to do after studying - "Explain how ACE inhibitors reduce mortality in systolic heart failure" - and it is only
ever written when the pack already holds a validated question or concept for it. Nothing is invented: every objective is the instruction form of a flashcard or question
that passed the quality gate (so it carries that item's source passages), or the list of the key concepts of a topic. The number of cards and questions that practise it is
counted from the pack, so the student sees which objectives have material to practise with. It needs no model and works for any pack, including one made by an earlier version.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from . import text as T
from .models import SourceRef, StudyPack

PER_TOPIC = 4                                   # a lecture topic gets at least this many objectives (a list a student reads, not a catalogue) ...
ITEMS_PER_OBJECTIVE = 3                         # ... and one more for every three of its cards and questions, up to MAX_PER_TOPIC
MAX_PER_TOPIC = 9
MAX_WORDS = 26
_NEAR_DUPLICATE = 0.6
_SAME_POINT = 0.35


@dataclass
class Objective:
    text: str
    topic: str = ""
    kind: str = "explain"                       # define | explain | compare | describe | identify
    sources: list[SourceRef] = field(default_factory=list)
    cards: int = 0                              # flashcards that practise it
    questions: int = 0                          # questions that practise it
    from_id: str = ""                           # the card or question it was made from ("" for the key-term objective of a topic)

    @property
    def practice(self) -> str:
        parts = []
        if self.cards:
            parts.append(f"{self.cards} flashcard{'s' if self.cards != 1 else ''}")
        if self.questions:
            parts.append(f"{self.questions} question{'s' if self.questions != 1 else ''}")
        return "Practise with " + " and ".join(parts) if parts else ""


# (pattern on the question with its final "?" removed, objective template, kind). First match wins.
_RULES: list[tuple[re.Pattern, str, str]] = [(re.compile(p, re.IGNORECASE), t, k) for p, t, k in (
    (r"^how do (?P<a>.+?) differ\b ?(?P<b>.*)$", "Compare {a}{b}", "compare"),
    (r"^what (?:is|are) the differences? between (?P<a>.+)$", "Compare {a}", "compare"),
    (r"^what (?:is|are) the sequence of (?P<a>.+)$", "Describe the sequence of {a}", "describe"),
    (r"^what happens (?:when|if|after) (?P<a>.+)$", "Explain what happens when {a}", "explain"),
    (r"^what (?:is|are) the (?P<n>(?:clinical |molecular )?mechanisms?) (?P<a>(?:by which|through which|behind|underlying) .+)$", "Explain the {n} {a}", "explain"),
    (r"^what (?:is|are) (?:a|an) (?P<a>(?:cause|risk factor|feature|sign|symptom|complication|treatment|example|side effect|finding|indication|use|function)\b.+)$", "Identify a {a}", "identify"),
    (r"^what (?:is|are) the (?P<n>(?:clinical )?(?:role|function|purpose|significance|relevance|importance|effect|effects|result|mechanism)) of (?P<a>.+)$", "Explain the {n} of {a}", "explain"),
    (r"^what (?:is|are) the (?P<a>(?:most )?(?:common|typical|main|key|first|usual|characteristic)\b.+)$", "Identify the {a}", "identify"),
    (r"^what (?:is|are) (?:a |an )?(?P<a>(?:typical|common|characteristic|usual)\b.+)$", "Identify a {a}", "identify"),
    (r"^what does (?P<a>.+?) do(?P<b>\b.*)?$", "Explain what {a} does{b}", "explain"),
    (r"^what do (?P<a>.+?) do(?P<b>\b.*)?$", "Explain what {a} do{b}", "explain"),
    (r"^what (?:is|are) (?:the )?(?P<a>.+)$", "Define {a}", "define"),
    (r"^why (?:do|does|did) (?P<a>.+)$", "Explain why {a}", "explain"),
    (r"^how (?:do|does|did) (?P<a>.+)$", "Explain how {a}", "explain"),
)]
_DANGLING = re.compile(r"\b(?:and|or|of|to|in|the|a|an|that|which|with|by|for|from|on|at|as|is|are|when|if|but)$", re.IGNORECASE)
_BAD = re.compile(r"_{2,}|\[unclear\]|\bthis (?:lecture|passage)\b", re.IGNORECASE)


def objective_text(front: str) -> Optional[tuple[str, str]]:
    """The instruction form of a question ("What is X?" -> "Define X") and its kind, or None when it does not read as a clean objective."""
    stem = " ".join(front.replace("\n", " ").split()).rstrip("?").strip().rstrip(".")
    if not stem or _BAD.search(stem):
        return None
    for pattern, template, kind in _RULES:
        m = pattern.match(stem)
        if not m:
            continue
        parts = {k: (v or "").strip() for k, v in m.groupdict().items()}
        if parts.get("b"):
            parts["b"] = " " + parts["b"]
        text = template.format(**{"a": "", "b": "", "n": "", **parts}).strip()
        text = text[0].upper() + text[1:]
        words = len(text.split())
        if words < 2 or words > MAX_WORDS or _DANGLING.search(text) or not T.content_stems(text.split(" ", 1)[1]):          # "Define it" says nothing
            return None
        return text, kind
    return None


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def _practising(origin_text: str, origin_sources: list[SourceRef], origin_fact: str, text: str, sources: list[SourceRef], fact_id: str) -> bool:
    """A card or question practises an objective when it comes from the same fact, or sits on the same source passage and asks nearly the same thing."""
    if fact_id and fact_id == origin_fact:
        return True
    shared = {r.chunk_id for r in sources} & {r.chunk_id for r in origin_sources}
    return bool(shared) and _jaccard(T.content_stems(text), T.content_stems(origin_text)) >= _SAME_POINT


def derive_objectives(pack: StudyPack, per_topic: int = PER_TOPIC) -> list[Objective]:
    """The objectives of a pack, in the order of its topics and, inside a topic, of the lecture."""
    candidates: list[tuple[Objective, str, str]] = []                            # (objective, the origin's own wording, the origin's fact id)
    for card in pack.flashcards:
        made = objective_text(card.front)
        if made:
            candidates.append((Objective(made[0], card.topic, made[1], list(card.sources), from_id=card.id), card.front, card.fact_id))
    for q in pack.questions:
        if q.type == "true_false":
            continue
        made = objective_text(q.question.replace("Which of the following is", "What is").replace("Which of the following are", "What are"))
        if made:
            candidates.append((Objective(made[0], q.topic, made[1], list(q.sources), from_id=q.id), q.question, q.fact_id))
    for o, wording, fact in candidates:
        o.cards = sum(1 for c in pack.flashcards if c.id == o.from_id or _practising(wording, o.sources, fact, c.front, c.sources, c.fact_id))
        o.questions = sum(1 for q in pack.questions if q.id == o.from_id or _practising(wording, o.sources, fact, q.question, q.sources, q.fact_id))
    out: list[Objective] = []
    for topic in dict.fromkeys([*pack.topics, *(o.topic for o, _w, _f in candidates)]):
        mine = [o for o, _w, _f in candidates if o.topic == topic]
        chosen: list[Objective] = []
        terms = [i for i in pack.summaries.get("key_concepts", []) if i.topic == topic and i.term]
        limit = min(MAX_PER_TOPIC, max(per_topic, round((sum(1 for c in pack.flashcards if c.topic == topic) + sum(1 for q in pack.questions if q.topic == topic)) / ITEMS_PER_OBJECTIVE)))
        room = max(limit - (1 if len(terms) >= 2 else 0), 1)
        for o in sorted(mine, key=lambda x: (x.kind == "define", -(x.cards + x.questions))):          # concepts and mechanisms before plain definitions; the best practised first
            stems = T.content_stems(o.text)
            if any(_jaccard(stems, T.content_stems(c.text)) >= _NEAR_DUPLICATE for c in chosen):
                continue
            if o.kind in ("define", "identify") and sum(1 for c in chosen if c.kind == o.kind) >= 2:          # the weaker kinds are capped: a list of "Define"s is a glossary
                continue
            chosen.append(o)
            if len(chosen) >= room:
                break
        if len(terms) >= 2:
            named = [i.term for i in terms[:5]]
            wanted = [t.casefold() for t in named]
            chosen.append(Objective("Define and recognise the key terms: " + ", ".join(named), topic, "define", [r for i in terms[:5] for r in i.sources[:1]],
                                    cards=sum(1 for c in pack.flashcards if any(t in c.front.casefold() for t in wanted))))
        chosen.sort(key=lambda x: min((r.start for r in x.sources if r.start is not None), default=1e12))
        out.extend(chosen[:limit])
    return out
