"""Quality gate of the Study engine: nothing reaches the student (or Study Mode) unless it passes.

Every generated card, question, summary point and key concept is checked before it is kept. A check returns a :class:`Verdict` with hard
problems (the item is dropped and counted) and soft doubts (the item is kept but scores lower). The same checks are used by the tests and by the
audit script, so "quality" is one definition everywhere.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, Sequence

from . import facts as F
from . import text as T
from .models import Flashcard, Question

# vague prompts the old generator produced ("Explain: Primary brain", "What does the lecture say about Neural fold?")
_VAGUE_FRONT = re.compile(r"^(?:explain|describe|discuss|define|tell\s+me\s+about|what\s+about|what\s+is\s+the\s+role\s+of|what\s+role\s+does|"
                          r"what\s+does\s+the\s+lecture\s+say(?:\s+about)?|what\s+is\s+the\s+mechanism\s+involving|clinical\s+connection)\b", re.IGNORECASE)
_ALWAYS_VAGUE_FRONT = re.compile(r"^(?:explain\s*:|describe\s*:|clinical\s+connection\s*:|what\s+is\s+the\s+mechanism\s+involving\b|what\s+does\s+the\s+lecture\s+say|what\s+role\s+does\b.+\bplay\s+in\s+the\s+pathway)",
                                 re.IGNORECASE)
_VAGUE_QUESTION = re.compile(r"(?:what\s+does\s+the\s+lecture\s+say|own\s+words|complete\s+the\s+statement\s+from\s+the\s+lecture|"
                             r"in\s+the\s+pathway\s+described|which\s+of\s+the\s+following\s+is\s+(?:a\s+)?statement)", re.IGNORECASE)
_BLANK = "_____"


@dataclass
class Verdict:
    ok: bool
    issues: list[str] = field(default_factory=list)
    soft: list[str] = field(default_factory=list)
    score: float = 10.0

    def merge(self, other: "Verdict") -> "Verdict":
        issues, soft = self.issues + other.issues, self.soft + other.soft
        return Verdict(not issues, issues, soft, max(0.0, min(self.score, other.score)))


def _verdict(issues: list[str], soft: list[str]) -> Verdict:
    score = max(0.0, 10.0 - 3.5 * len(issues) - 0.8 * len(soft))
    return Verdict(not issues and score >= 7.0, issues, soft, round(score, 1))


def _text_issues(text: str, *, min_words: int = 3, max_words: int = 45, allow_blank: bool = False) -> tuple[list[str], list[str]]:
    """Grammar / repetition / filler problems of a piece of study text (not of a question stem)."""
    issues: list[str] = []
    soft: list[str] = []
    body = text.replace(_BLANK, "term") if allow_blank else text
    words = T.words(body)
    if len(words) < min_words:
        issues.append("too short")
    if len(words) > max_words:
        issues.append("too long")
    if re.search(r"\b([\w-]{3,})\b[\s,]+\1\b", body, re.IGNORECASE):
        issues.append("repeated words")
    if F._AGREEMENT.search(body):
        issues.append("broken grammar")
    if re.search(r"\b(?:okay|ok|um+|uh+|you know|i mean)\b", body, re.IGNORECASE):
        issues.append("spoken filler")
    if re.search(r"\b(?:i|we|you|my|our)\b(?!\s*(?:call|name|define))", body, re.IGNORECASE) and not allow_blank:
        soft.append("speaker's voice")
    if body.count("(") != body.count(")"):
        issues.append("unbalanced brackets")
    return issues, soft


def check_statement(text: str, *, unknown=None, unresolved: Sequence[str] = ()) -> Verdict:
    """A study statement (summary point, T/F statement, answer sentence): a complete, standalone, clean sentence."""
    c = F.sentence_issues(text, unknown=unknown, unresolved=unresolved, min_words=5)
    return _verdict(list(c.issues), list(c.soft))


# ====================================================================================================================== flashcards
def check_card(card: Flashcard, *, unknown=None, unresolved: Sequence[str] = (), known_term=None) -> Verdict:
    issues: list[str] = []
    soft: list[str] = []
    front, back = card.front.strip(), card.back.strip()
    if not front or not back:
        issues.append("empty side")
    if front.casefold() == back.casefold():
        issues.append("answer equals question")
    cloze = _BLANK in front
    if _ALWAYS_VAGUE_FRONT.match(front) or (_VAGUE_FRONT.match(front) and len(T.words(front)) <= 6):
        issues.append("vague prompt")
    if _VAGUE_QUESTION.search(front):
        issues.append("vague prompt")
    if not cloze and not front.endswith("?") and not re.match(r"^(?:compare|name|list|give|state)\b", front, re.IGNORECASE):
        soft.append("not phrased as a question")
    if len(T.words(front)) > (34 if cloze else 24):
        issues.append("prompt too long")
    if len(T.words(back)) > 55:
        issues.append("answer too long")
    elif len(T.words(back)) > 38:
        soft.append("long answer")
    fi, fs = _text_issues(front, min_words=3, max_words=60, allow_blank=True)
    bi, bs = _text_issues(back.split(" - ", 1)[-1] if cloze else back, min_words=1, max_words=60)
    issues += [f"front: {i}" for i in fi if i != "too long"] + [f"back: {i}" for i in bi if i != "too long"]
    soft += fs + bs
    if cloze:
        if front.count(_BLANK) > 2:
            issues.append("more than two blanks")
        sentence = re.sub(r"^(?:fill\s+in\s+the\s+blank|complete\s+the\s+(?:sentence|statement))\s*[:.]\s*", "", front, flags=re.IGNORECASE).replace(_BLANK, "term")
        if len(T.words(sentence)) > 30:
            issues.append("cloze too long")
        c = F.sentence_issues(sentence, unknown=unknown, unresolved=unresolved, min_words=6)
        issues += [f"cloze: {i}" for i in c.issues if i not in ("too long",)]
        answer = back.split(" - ", 1)[0].strip()
        if not answer or answer.casefold() in T.STOPWORDS or F.is_generic_label(answer, known_term):
            issues.append("blank is not a medical term")
    elif len(back.split()) >= 4:
        c = F.sentence_issues(back if back.endswith((".", "!", "?")) else back + ".", unknown=unknown, unresolved=unresolved, min_words=3)
        issues += [f"back: {i}" for i in c.issues if i not in ("too long", "too short", "starts with a pronoun", "no verb", "unfinished sentence",
                                                              "several statements in one sentence", "two statements run together")]
    if not card.sources:
        issues.append("no source")
    return _verdict(issues, soft)


# ====================================================================================================================== questions
def option_lengths_ok(options: Sequence[str]) -> bool:
    sizes = [max(len(o), 1) for o in options]
    return max(sizes) <= 4 * min(sizes) + 12


def check_question(q: Question, *, unknown=None, unresolved: Sequence[str] = (), same_category=None) -> Verdict:
    issues: list[str] = []
    soft: list[str] = []
    stem = q.question.strip()
    if not stem or not q.answer.strip():
        issues.append("empty question or answer")
    if _VAGUE_QUESTION.search(stem) or _ALWAYS_VAGUE_FRONT.match(stem) or (_VAGUE_FRONT.match(stem) and len(T.words(stem)) <= 6):
        issues.append("vague question")
    if q.type != "true_false" and not stem.endswith(("?", ":")) and _BLANK not in stem:
        soft.append("not phrased as a question")
    if not q.sources:
        issues.append("no source")
    if not q.explanation.strip():
        soft.append("no explanation")
    if q.type == "true_false":
        statement = stem.split(":", 1)[-1].strip() if ":" in stem[:20] else stem
        statement = re.sub(r"^(?:true or false|decide whether the following is true or false)\s*[:.]?\s*", "", statement, flags=re.IGNORECASE)
        c = check_statement(statement if statement.endswith((".", "!", "?")) else statement + ".", unknown=unknown, unresolved=unresolved)
        issues += [f"statement: {i}" for i in c.issues]
        soft += [f"statement: {i}" for i in c.soft]
        if len(T.words(statement)) > 32:
            issues.append("statement too long")
        if q.answer not in ("True", "False", "Doğru", "Yanlış"):
            issues.append("unclear truth value")
    else:
        i, s = _text_issues(stem, min_words=3, max_words=60, allow_blank=True)
        issues += [f"question: {x}" for x in i if x != "too long"]
        soft += s
        if len(T.words(stem)) > 55:
            issues.append("question too long")
    if q.options and q.type != "true_false":
        opts = list(q.options)
        folded = [o.casefold().strip() for o in opts]
        if len(set(folded)) != len(folded):
            issues.append("duplicate options")
        if q.answer not in opts:
            issues.append("answer not among the options")
        if len(opts) < 3:
            issues.append("too few options")
        if not option_lengths_ok(opts):
            soft.append("options differ a lot in length")
        key = T.stem(q.answer.casefold())
        if len(q.answer) >= 5 and q.answer.casefold() in stem.casefold() and not stem.count(_BLANK):
            issues.append("the answer is in the question")
        for o in opts:
            if o != q.answer and len(T.words(o)) <= 6:
                i, _s = _text_issues(o, min_words=1, max_words=12)
                if i:
                    issues.append(f"option: {i[0]}")
                    break
        if same_category is not None and not all(same_category(q.answer, o) for o in opts if o != q.answer):
            issues.append("distractors are not of the same kind as the answer")
    elif q.type in ("short_answer", "explain_why"):
        ans_words = len(T.words(q.answer))
        if ans_words > 45:
            issues.append("expected answer too long")
        elif ans_words > 32:
            soft.append("long expected answer")
        c = _text_issues(q.answer, min_words=1, max_words=60)
        issues += [f"answer: {i}" for i in c[0] if i != "too long"]
        if q.answer.casefold().rstrip(". ") in stem.casefold():
            issues.append("the answer is in the question")
    return _verdict(issues, soft)


# ====================================================================================================================== summary / concepts
def check_summary_point(text: str, *, unknown=None, unresolved: Sequence[str] = (), min_words: int = 6) -> Verdict:
    c = F.sentence_issues(text, unknown=unknown, unresolved=unresolved, min_words=min_words)
    return _verdict(list(c.issues), list(c.soft))


def check_concept(term: str, definition: str, *, known_term=None, unknown=None, unresolved: Sequence[str] = ()) -> Verdict:
    issues: list[str] = []
    soft: list[str] = []
    if F.is_generic_label(term, known_term):
        issues.append("generic label")
    if len(T.words(term)) > 5:
        issues.append("label is a phrase")
    if len(T.words(definition)) > 70:
        issues.append("definition too long")
    sentences = [s for s in T.sentences(definition) if s]
    if not sentences or len(sentences) > 3:
        issues.append("not 1-3 sentences")
    for s in sentences:
        c = F.sentence_issues(s, unknown=unknown, unresolved=unresolved, min_words=5)
        issues += c.issues
        soft += c.soft
    if term.casefold() not in definition.casefold() and not any(T.stem(w) in {T.stem(x) for x in T.words(definition)} for w in T.words(term)):
        soft.append("term not mentioned")
    return _verdict(issues[:6], soft)


@dataclass
class Gate:
    """Counts what the gate dropped, by reason: stored in the pack so the student (and the audit) can see that the filter works."""

    kept: Counter = field(default_factory=Counter)
    dropped: Counter = field(default_factory=Counter)
    reasons: Counter = field(default_factory=Counter)
    examples: list[dict] = field(default_factory=list)

    def record(self, area: str, verdict: Verdict, label: str = "") -> bool:
        if verdict.ok:
            self.kept[area] += 1
            return True
        self.dropped[area] += 1
        for issue in verdict.issues:
            self.reasons[f"{area}: {issue.split(':')[0] if issue.startswith(('front', 'back', 'cloze', 'statement', 'question', 'answer', 'option')) else issue}"] += 1
        if len(self.examples) < 40:
            self.examples.append({"area": area, "item": label[:160], "issues": verdict.issues[:4]})
        return False

    def to_dict(self) -> dict:
        return {"kept": dict(self.kept), "dropped": dict(self.dropped), "reasons": dict(self.reasons.most_common(25)), "examples": self.examples}
