"""The premium gate of the Study engine: a flashcard or a question reaches the student only if it is worth studying.

``quality.py`` and ``grounding.py`` ask "is this item well-formed and supported". This module asks the questions a teacher would ask, and it asks them of EVERY item whichever provider
wrote it (a local language model, a cloud model or the built-in method):

* is it a real question, in a sentence the student can answer without the lecture next to them (no blanks, no "what does the lecture say", no dangling fragments)?
* is the answer more than a restatement of the question, and does it answer what was asked?
* does the lecture really say it - the evidence quote is in the cited passage, and its numbers are too?
* for a multiple-choice question: exactly one defensible best answer, distractors of the same kind that the evidence does not also make correct?
* is it new - not a near copy of a card or question that is already kept?
* how hard is it (Basic / Intermediate / Advanced), judged from what answering it takes rather than from what the writer claimed?

Nothing here invents text. An item that fails is dropped and counted with its reason (``Gate`` in the pack statistics); the callers ask for replacements a bounded number of times.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from . import text as T
from .grounding import support_score
from .index import DocumentIndex
from .models import Chunk, Flashcard, Question, SourceRef

#: the version of the whole generation recipe (prompts, gate, calibration). Study material records it; a different value means "made by an older/newer recipe".
GENERATION_VERSION = "2"

MIN_EVIDENCE_OVERLAP = 0.8
MIN_SUPPORT_CARD = 0.75              # share of the answer's content words that the cited passage contains (a quoted sentence is 1.0; a model's elaboration is what this keeps out)
MIN_SUPPORT_QUESTION = 0.7
NEAR_DUPLICATE = 0.6                 # content-word overlap above which two cards, or two questions, ask the same thing
FACT_REPEAT = 0.7                    # two items resting on (nearly) the same sentence of the lecture teach the same fact twice, whatever their wording or format
CARD_REPEAT = 0.55                  # a question this alike (content words) to a flashcard's front is the same task twice: stricter than between two questions, because the pair would sit side by side
SAME_OPENING_MAX = 2                 # at most this many items of a pack may begin with the same four words ("What is the clinical significance of ...")
OPTION_COUNTS = (4, 5)

_BLANK = re.compile(r"_{2,}|\[\s*(?:blank|\.{2,}|…)\s*\]|\bfill\s+in\s+the\s+blank\b|\bcomplete\s+the\s+(?:sentence|statement|following)\b|\bcorrectly\s+completes?\b|\bwhich\s+term\s+completes\b", re.IGNORECASE)
_LECTURE_TALK = re.compile(r"\b(?:what\s+does\s+the\s+lecture|the\s+lecture(?:r)?\s+(?:says?|said|states?|stated|mentions?|explains?)|according\s+to\s+the\s+(?:lecture|speaker|transcript|passage|text)|in\s+your\s+own\s+words|"
                           r"the\s+(?:speaker|professor|teacher)\s+(?:says|said|states))\b", re.IGNORECASE)
_VAGUE_START = re.compile(r"^(?:explain|describe|discuss|define|tell\s+me\s+about|what\s+about|clinical\s+connection|summari[sz]e)\s*[:\-]?\s", re.IGNORECASE)
_DANGLING = re.compile(r"\b(?:result\s+of|effect\s+of|role\s+of|function\s+of|purpose\s+of|difference\s+between|relationship\s+between|happens\s+(?:when|if)|"
                       r"cause\s+of|consequence\s+of)\s+(?:in|on|at|during|with|by|for|from|to|of|and|or|that|which|is|are|was|were|the|a|an)\s*(?:\?|$)", re.IGNORECASE)
_DANGLING_PREP = re.compile(r"\b(?:result\s+of|effect\s+of|happens\s+(?:when|if))\s+(?:in|on|at|during|with|by|for|from|to|of)\b", re.IGNORECASE)
_REPEATED_WORD = re.compile(r"\b([\w-]{3,})\s+\1\b", re.IGNORECASE)
_PRONOUN_START = re.compile(r"^(?:it|its|they|their|these|those|he|she|this|that)\b", re.IGNORECASE)
_GENERIC_END = re.compile(r"\b(?:of|in|for|to|by|on)\s+the\s+(?:pathway|system|process|mechanism|structure|region|area|response|network)\s*\??$", re.IGNORECASE)      # "... in the regulation of the pathway?": which one?
_FILLER = re.compile(r"\b(?:okay|um+|uh+|you\s+know|i\s+mean|so\s+yeah|gonna|kind\s+of)\b", re.IGNORECASE)
_WH_DO = re.compile(r"^what\s+(?:does|do)\s+(?:the\s+)?(?P<subject>.+?)\s+do\s*\?$", re.IGNORECASE)
_WH_IS = re.compile(r"^what\s+(?:is|are)\s+(?:the\s+|a\s+|an\s+)?(?P<subject>[^?]+?)\s*\?$", re.IGNORECASE)
_STATE_VERBS = frozenset("is are was were be been has have had occurs occur happens happen exists exist takes remains remain contains contain lacks lack lives live".split())
_DEFINING_VERBS = frozenset("is are refers means describes denotes represents".split())
_ALL_OF = re.compile(r"\b(?:all|none|both|neither)\s+of\s+the\s+(?:above|following)\b|\b(?:a\s+and\s+b|a,\s*b\s+and\s+c)\b|^(?:all|none)\s+of\s+(?:these|them)\b", re.IGNORECASE)
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


@dataclass
class Review:
    ok: bool
    reasons: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.ok


@dataclass
class Rejected:
    item: Flashcard | Question
    reasons: list[str]


@dataclass
class Curation:
    """What a pass of the gate kept and dropped."""

    kept: list = field(default_factory=list)
    rejected: list[Rejected] = field(default_factory=list)

    def reasons(self) -> Counter:
        counter: Counter = Counter()
        for r in self.rejected:
            counter.update(r.reasons[:1])                       # the first (main) reason of every dropped item
        return counter


def _norm(text: str) -> str:
    return " ".join(T.words(text))


def _stems(text: str) -> set[str]:
    return T.content_stems(text)


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


_DEFINITION = re.compile(r"^(?:what|who)\s+(?:is|are|was|were)\s+(?:the\s+|a\s+|an\s+)?(?P<term>[^?]+)\??$", re.IGNORECASE)


_EXAM_PHRASING = frozenset({"follow", "common", "typic", "main", "key", "major", "primary", "characteristic", "like", "likel", "mostli"})   # how a question is worded, not what it is about


def _ask(text: str) -> tuple[set[str], Optional[frozenset]]:
    """What a front asks: its content stems (without exam phrasing: "Which of the following is a common symptom of X?" and "What are the typical symptoms of X?" ask the same)
    and, for a short definition ("What is systolic heart failure?"), the stems of the term."""
    text = text.replace("-", " ")
    m = _DEFINITION.match(text.strip())
    term = frozenset(_stems(m.group("term"))) if m and len(T.words(m.group("term"))) <= 4 else None
    return {w for w in _stems(text) if w not in _EXAM_PHRASING}, term


def _alike(a: tuple[set[str], Optional[frozenset]], b: tuple[set[str], Optional[frozenset]], threshold: float) -> bool:
    """Two fronts ask the same thing. Two short definitions are the same only when they define the same term - one more word makes it another concept (a subtype, a part)."""
    if a[1] is not None and b[1] is not None:
        return a[1] == b[1]
    return _jaccard(a[0], b[0]) >= threshold


def _numbers(text: str) -> set[str]:
    return {n.replace(",", ".") for n in _NUMBER.findall(text)}


def _chunks(item, index: DocumentIndex) -> list[Chunk]:
    return [c for c in (index.get(r.chunk_id) for r in item.sources) if c]


def doubted_word(text: str, index: DocumentIndex) -> str:
    """A word the medical pass could not resolve (a misheard term such as "nuclides" for "nuclei") that the text uses: no study item may rest on it. "" when there is none."""
    folded = " " + " ".join(T.words(text)) + " "
    for word in getattr(index, "unresolved", []) or []:
        key = " ".join(T.words(word))
        if len(key) >= 6 and f" {key} " in folded:
            return word
    return ""


def opening(text: str) -> str:
    return " ".join(T.words(text)[:4])


# ------------------------------------------------------------------------------------------------------------------------------ evidence
def _sentences_of(chunks: Sequence[Chunk]) -> list[tuple[Chunk, str]]:
    out: list[tuple[Chunk, str]] = []
    for chunk in chunks:
        for sentence in T.sentences(chunk.text) or [T.clean(chunk.text)]:
            out.append((chunk, sentence))
    return out


def best_evidence(claim: str, chunks: Sequence[Chunk]) -> tuple[Optional[Chunk], str, float]:
    """The sentence of the cited passages that backs ``claim`` best: ``(chunk, sentence, share of the claim's content words found in it)``."""
    stems = _stems(claim)
    best: tuple[Optional[Chunk], str, float] = (None, "", 0.0)
    for chunk, sentence in _sentences_of(chunks):
        share = T.overlap(stems, _stems(sentence)) if stems else 0.0
        if share > best[2]:
            best = (chunk, sentence, share)
    return best


def evidence_in(evidence: str, chunks: Sequence[Chunk]) -> bool:
    """The quoted evidence really is (nearly) in the cited passages, with the same numbers."""
    if not evidence.strip():
        return False
    source = " ".join(c.text for c in chunks)
    if not _numbers(evidence) <= _numbers(source):
        return False
    if _norm(evidence) and _norm(evidence) in _norm(source):
        return True
    return T.overlap(_stems(evidence), _stems(source)) >= MIN_EVIDENCE_OVERLAP


def locate(chunk: Chunk, sentence: str) -> SourceRef:
    """A reference narrowed to the transcript segment (and its times) that holds ``sentence``; the whole chunk when it cannot be placed."""
    ref = SourceRef.of(chunk)
    words = sentence.split()[:6]
    if not chunk.segs or not words:
        return ref
    match = re.search(r"\s+".join(re.escape(w) for w in words), chunk.text, re.IGNORECASE)
    if match is None:
        return ref
    for seg in chunk.segs:
        try:
            start, end, first, last = seg[0], seg[1], int(seg[2]), int(seg[3])
        except (TypeError, ValueError, IndexError):
            continue
        if first <= match.start() <= last:
            return SourceRef.of_span(chunk, start, end)
    return ref


_MASK = re.compile(r"\[\s*unclear\s*\]", re.IGNORECASE)


def _unmask(quote: str, chunks: Sequence[Chunk]) -> tuple[Optional[Chunk], str]:
    """Where the quote stands in the cited passages, the masked word being any one word: (chunk, the lecture's own words), or (None, "")."""
    parts = [r"[\w'-]+".join(re.escape(piece) for piece in _MASK.split(token)) for token in quote.split()]
    pattern = re.compile(r"\s+".join(parts), re.IGNORECASE)
    for chunk in chunks:
        found = pattern.search(chunk.text)
        if found:
            return chunk, " ".join(found.group(0).split())
    return None, ""


def attach_evidence(item, index: DocumentIndex) -> None:
    """Fill in what the item rests on: the evidence sentence (when the writer gave none) and a reference narrowed to the segment, so the student can jump to the moment."""
    chunks = _chunks(item, index)
    if not chunks:
        return
    claim = item.back if isinstance(item, Flashcard) else f"{item.answer} {item.explanation}"
    chunk, sentence, _share = best_evidence(claim, chunks)
    if not item.evidence.strip() and sentence:
        item.evidence = sentence
    elif _MASK.search(item.evidence):
        # The model saw a doubtful word as "[unclear]" and quoted it so; the lecture never says "[unclear]". Show what the lecture says at that place: the stretch the quote matches with the
        # masked word as a wildcard; failing that the sentence that backs the quote best; failing that the quote with an ellipsis where the word was.
        quoted_chunk, original = _unmask(item.evidence, chunks)
        if not original:
            quoted_chunk, original, share = best_evidence(_MASK.sub(" ", item.evidence), chunks)
            original = original if share >= 0.6 else ""
        if original:
            item.evidence, chunk = original, quoted_chunk or chunk
        else:
            item.evidence = " ".join(_MASK.sub("...", item.evidence).split())
    if chunk is not None and item.evidence.strip():
        item.sources = [locate(chunk, item.evidence)] + [r for r in item.sources if r.chunk_id != chunk.id]


_WHY = re.compile(r"^why\b", re.IGNORECASE)
_ANSWER_REASON = re.compile(r"\b(?:because|since|so\s+that|thereby|therefore|thus|hence|which\s+is\s+why|that\s+is\s+why|due\s+to|as\s+a\s+result|in\s+order\s+to)\b", re.IGNORECASE)
_CAUSAL_CUE = re.compile(r"\b(?:because|since|so|therefore|thus|hence|due\s+to|which\s+is\s+why|that\s+is\s+why|as\s+a\s+result|in\s+order\s+to|so\s+that|leads?\s+to|led\s+to|results?\s+in|"
                         r"caus(?:e|es|ed|ing)|allow(?:s|ed|ing)?|means|explains?)\b", re.IGNORECASE)


def _evidence_text(item, chunks: Sequence[Chunk]) -> str:
    """The sentence an item rests on: the one it quotes, else the sentence of the cited passages that backs its answer best."""
    claim = item.back if isinstance(item, Flashcard) else f"{item.answer} {item.explanation}"
    if item.evidence.strip():
        text = _MASK.sub(" ", item.evidence)
        sentences = T.sentences(text) or [text]
        stems = _stems(item.back if isinstance(item, Flashcard) else item.answer)          # the answer decides which sentence is "its" sentence (the explanation may repeat the neighbour too)
        return max(sentences, key=lambda s: T.overlap(stems, _stems(s))) if len(sentences) > 1 else text      # a quote of several sentences: the one that backs the answer, not its neighbour
    return best_evidence(claim, chunks)[1]


def _fact_stems(item, index: DocumentIndex) -> set[str]:
    """The sentence an item rests on: the one it quotes, or - when the writer quoted none - the sentence of the cited passage that supports it best."""
    text = item.evidence.strip()
    if not text:
        chunks = _chunks(item, index)
        if chunks:
            claim = item.back if isinstance(item, Flashcard) else f"{item.answer} {item.explanation}"
            text = best_evidence(claim, chunks)[1] or ""
    return _stems(text)


# ------------------------------------------------------------------------------------------------------------------------------ cards
def _blank_or_vague(text: str, reasons: list[str]) -> None:
    if _BLANK.search(text):
        reasons.append("fill-in-the-blank")
    if _LECTURE_TALK.search(text) or _VAGUE_START.match(text.strip()):
        reasons.append("vague prompt")


def _front_problems(front: str, reasons: list[str]) -> None:
    words = T.words(front)
    if len(words) < 3:
        reasons.append("question too short")
    if len(words) > 40:
        reasons.append("question too long")
    if _DANGLING.search(front) or _DANGLING_PREP.search(front) or _REPEATED_WORD.search(front):
        reasons.append("garbled question")
    if _PRONOUN_START.match(front.strip()):
        reasons.append("not self-contained")
    if _GENERIC_END.search(front.strip()):
        reasons.append("not self-contained")                       # it ends on "the pathway" / "the system" without saying which: the student cannot know what is asked
    if _FILLER.search(front):
        reasons.append("spoken filler")
    if front.count("(") != front.count(")"):
        reasons.append("unbalanced brackets")
    if front[:1].islower() and not front[:1].isdigit():
        reasons.append("fragment")


_WHAT_HAPPENS = re.compile(r"^what\s+(?:happens|would\s+happen|occurs)\s+(?:when|if|after|to)\b", re.IGNORECASE)
_CONSEQUENCE = re.compile(r"\b(?:caus\w*|lead\w*|result\w*|produc\w*|reduc\w*|increas\w*|decreas\w*|impair\w*|fail\w*|no\s+longer|without|lower\w*|rais\w*|high\w*|low|deficien\w*|lack\w*|"
                          r"absen\w*|excess\w*|due\s+to|because|so\s+that|then|stain\w*|block\w*|inhibit\w*|stimulat\w*|releas\w*|trigger\w*|prevent\w*|worsen\w*|develop\w*|damage\w*|"
                          r"loss|lose\w*|becom\w*|rise\w*|rising|fall\w*|drop\w*|accumulat\w*|toxic\w*|resist\w*|stop\w*|disrupt\w*|activat\w*|suppress\w*|swell\w*|bleed\w*)\b", re.IGNORECASE)


def _focus_problem(front: str, back: str) -> Optional[str]:
    """The answer must be about what the question asks. Three shapes the writers get wrong: "What does X do?" answered with where/what X is, "What is X?" answered about a longer term,
    and "What happens when X?" answered with a statement that says nothing happens next (it restates a requirement of X)."""
    if _WHAT_HAPPENS.match(front.strip()) and not _CONSEQUENCE.search(back):
        return "answer does not say what follows"
    m = _WH_DO.match(front.strip())
    if m:
        subject = _norm(m.group("subject"))
        tokens = back.strip().split()
        words = [w.casefold().strip(",.;:") for w in tokens]
        for i, w in enumerate(words[:12]):
            if w in _STATE_VERBS and _norm(" ".join(words[:i])).endswith(subject.split()[-1] if subject.split() else subject):
                return "answer does not say what it does"
        return None
    m = _WH_IS.match(front.strip())
    if m:
        subject = _norm(m.group("subject"))
        words = [w.casefold().strip(",.;:") for w in back.strip().split()]
        head: list[str] = []
        for w in words:
            if w in _DEFINING_VERBS or w in _STATE_VERBS:
                break
            head.append(w)
        head_text = _norm(" ".join(head)).replace("the ", "", 1) if head else ""
        head_text = re.sub(r"^(?:the|a|an)\s+", "", head_text)
        if head_text and subject and head_text != subject and head_text.startswith(subject) and len(head) < 8:
            return "answer is about a different term"
    return None


_NUMBER_WORDS = frozenset("one two three four five six seven eight nine ten eleven twelve twenty hundred thousand twice half".split())


def _quantities(text: str) -> set[str]:
    return {w for w in T.words(text) if w in _NUMBER_WORDS or w[:1].isdigit()}


def _answer_in_question(front: str, back: str) -> bool:
    """The answer only repeats the question. A number the question does not contain is new information ("How many ATP ...?" - "Two ATP ...")."""
    fs, bs = _stems(front), _stems(back)
    if not bs:
        return True
    if _quantities(back) - _quantities(front):
        return False
    if _norm(back) and _norm(back) in _norm(front):
        return True
    ratio = len(bs & fs) / len(bs)
    return (len(bs) >= 2 and ratio >= 0.8) or (len(bs) == 1 and ratio == 1.0 and len(T.words(front)) < 12)


def review_card(card: Flashcard, index: DocumentIndex) -> Review:
    reasons: list[str] = []
    front, back = card.front.strip(), card.back.strip()
    if not front or not back:
        return Review(False, ["empty side"])
    _blank_or_vague(front, reasons)
    _blank_or_vague(back, [])
    _front_problems(front, reasons)
    if doubted_word(f"{front} {back} {card.evidence}", index):
        reasons.append("rests on a misheard word")
    back_words = T.words(back)
    if len(back_words) > 60:
        reasons.append("answer too long")
    if _REPEATED_WORD.search(back) or _FILLER.search(back):
        reasons.append("garbled answer")
    if len(back_words) < 3 and len(T.words(front)) < 8:
        reasons.append("trivial")
    if len(T.words(front)) <= 5 and len(back_words) <= 4:
        reasons.append("trivial")
    if _answer_in_question(front, back):
        reasons.append("answer in question")
    focus = _focus_problem(front, back)
    if focus:
        reasons.append(focus)
    chunks = _chunks(card, index)
    if not chunks:
        reasons.append("no source")
    else:
        if support_score(card.back, chunks) < MIN_SUPPORT_CARD:
            reasons.append("unsupported")
        elif card.evidence.strip() and not evidence_in(card.evidence, chunks):
            reasons.append("evidence not in the lecture")
        if (_WHY.match(front) or _ANSWER_REASON.search(back)) and not _CAUSAL_CUE.search(_evidence_text(card, chunks)):
            reasons.append("the lecture gives no reason for this")
    return Review(not reasons, list(dict.fromkeys(reasons)))


# ------------------------------------------------------------------------------------------------------------------------------ questions
def review_question(q: Question, index: DocumentIndex) -> Review:
    reasons: list[str] = []
    stem, answer = q.question.strip(), q.answer.strip()
    if not stem or not answer:
        return Review(False, ["empty question or answer"])
    _blank_or_vague(stem, reasons)
    if doubted_word(" ".join([stem, answer, q.evidence, *q.options]), index):
        reasons.append("rests on a misheard word")
    if q.type != "true_false":
        _front_problems(stem, reasons)
    elif len(T.words(stem)) < 8:                       # a six-word statement ("Drug X is effective against Y.") tests a name, not an idea
        reasons.append("statement too short")
    options = [o.strip() for o in q.options if o.strip()]
    if q.type in ("mcq", "clinical_vignette", "mechanism") and options:
        folded = [_norm(o) for o in options]
        if len(options) not in OPTION_COUNTS:
            reasons.append("needs four or five options")
        if len(set(folded)) != len(folded):
            reasons.append("duplicate options")
        if answer not in options:
            reasons.append("answer not among the options")
        elif sum(1 for o in folded if o == _norm(answer)) != 1:
            reasons.append("answer is not unique")
        if any(_ALL_OF.search(o) for o in options):
            reasons.append("all/none of the above")
        sizes = [max(len(o), 1) for o in options]
        if max(sizes) > 4 * min(sizes) + 20:
            reasons.append("options differ a lot in length")
        for o in options:
            if o != answer and _norm(o) and (_norm(o) in _norm(answer) or _norm(answer) in _norm(o)):
                reasons.append("a distractor contains the answer")
                break
        if _norm(answer) and len(answer) >= 5 and _norm(answer) in _norm(stem):
            reasons.append("answer in question")
        if q.type == "clinical_vignette" and len(T.words(stem)) < 20:
            reasons.append("vignette too short")
    elif q.type == "true_false":
        if answer not in ("True", "False"):
            reasons.append("unclear truth value")
    else:
        words = T.words(answer)
        if len(words) < 3:
            reasons.append("answer too short")
        if len(words) > 70:
            reasons.append("answer too long")
        if _answer_in_question(stem, answer):
            reasons.append("answer in question")
        focus = _focus_problem(stem, answer)
        if focus:
            reasons.append(focus)
    if not q.explanation.strip() or len(T.words(q.explanation)) < 5:
        reasons.append("no explanation")
    chunks = _chunks(q, index)
    if not chunks:
        reasons.append("no source")
    else:
        truth = answer if q.type not in ("true_false",) else stem
        if support_score(f"{truth} {q.explanation}" if q.type != "true_false" else q.explanation or stem, chunks) < MIN_SUPPORT_QUESTION and support_score(truth, chunks) < MIN_SUPPORT_QUESTION:
            reasons.append("unsupported")
        elif q.evidence.strip() and not evidence_in(q.evidence, chunks):
            reasons.append("evidence not in the lecture")
        asserts_reason = _ANSWER_REASON.search(stem if q.type == "true_false" else answer)
        if (_WHY.match(stem) or q.type == "explain_why" or asserts_reason) and not _CAUSAL_CUE.search(_evidence_text(q, chunks)):
            reasons.append("the lecture gives no reason for this")
        if options and q.type in ("mcq", "clinical_vignette", "mechanism"):
            evidence_sentence = q.evidence or best_evidence(f"{q.answer} {q.explanation}", chunks)[1]
            es, ss = _stems(evidence_sentence), _stems(stem)
            for o in options:
                os_ = _stems(o)
                if o == answer or not os_:
                    continue
                if os_ <= ss:
                    reasons.append("an option repeats the question")
                    break
                if es and os_ <= es:
                    reasons.append("a distractor is also in the evidence")
                    break
    return Review(not reasons, list(dict.fromkeys(reasons)))


# ------------------------------------------------------------------------------------------------------------------------------ difficulty
_RECALL_CARDS = {"definition", "structure_function", "exam_recall", "enzyme", "drug", "gene_protein"}
_REASONING_CARDS = {"mechanism", "cause_effect", "comparison", "pathway", "concept", "clinical"}


def calibrate_card(card: Flashcard, index: DocumentIndex) -> str:
    """Basic: a single stated fact. Intermediate: a mechanism, a cause and its effect, a comparison. Advanced: it ties several facts or passages together."""
    level = 0 if card.type in _RECALL_CARDS else 1
    level += card.type == "integration"                                      # by definition it puts two facts together
    chunk_ids = {r.chunk_id for r in card.sources}
    if len(chunk_ids) >= 2 or len(T.sentences(card.back)) >= 2 or len(T.words(card.back)) > 38:
        level += 1
    if card.type == "clinical" and len(T.words(card.back)) > 22:
        level += 1
    return ("easy", "medium", "hard")[min(level, 2)]


def calibrate_question(q: Question, index: DocumentIndex) -> str:
    base = {"true_false": 0, "short_answer": 0, "mcq": 1, "mechanism": 1, "explain_why": 1, "clinical_vignette": 2}.get(q.type, 1)
    chunk_ids = {r.chunk_id for r in q.sources}
    if len(chunk_ids) >= 2:
        base += 1
    if q.type == "mcq" and len(T.words(q.question)) < 9:
        base -= 1                                                              # "Which enzyme ... ?" with a short stem is plain recall
    if q.type in ("short_answer", "explain_why") and len(T.words(q.answer)) > 30:
        base += 1
    return ("easy", "medium", "hard")[max(0, min(base, 2))]


# ------------------------------------------------------------------------------------------------------------------------------ the pass
def curate_cards(cards: Iterable[Flashcard], index: DocumentIndex, *, existing: Sequence[Flashcard] = (), generation: str = GENERATION_VERSION) -> Curation:
    """Review ``cards`` against the lecture and against the cards already kept (``existing`` + the ones accepted earlier in this pass)."""
    result = Curation()
    kept_fronts = [_ask(c.front) for c in existing]
    kept_backs = [_stems(c.back) for c in existing]
    kept_facts = [_stems(c.evidence) for c in existing if c.evidence.strip()]
    openings = Counter(opening(c.front) for c in existing)
    for card in cards:
        verdict = review_card(card, index)
        if verdict.ok and openings[opening(card.front)] >= SAME_OPENING_MAX:
            verdict = Review(False, ["same opening as other cards"])
        if verdict.ok:
            fs, bs = _ask(card.front), _stems(card.back)
            duplicate = any(_alike(fs, k, NEAR_DUPLICATE) for k in kept_fronts) or any(_jaccard(bs, kb) >= 0.8 and _jaccard(fs[0], kf[0]) >= 0.4 for kb, kf in zip(kept_backs, kept_fronts))
            if duplicate:
                verdict = Review(False, ["near-identical to another card"])
            else:
                facts = _fact_stems(card, index)
                if facts and any(_jaccard(facts, k) >= FACT_REPEAT for k in kept_facts):
                    verdict = Review(False, ["same fact as another card"])
        if not verdict.ok:
            result.rejected.append(Rejected(card, verdict.reasons))
            continue
        attach_evidence(card, index)
        card.difficulty = calibrate_card(card, index)
        card.generation = generation
        openings[opening(card.front)] += 1
        kept_fronts.append(_ask(card.front))
        kept_backs.append(_stems(card.back))
        if card.evidence.strip():
            kept_facts.append(_stems(card.evidence))
        result.kept.append(card)
    return result


def curate_questions(questions: Iterable[Question], index: DocumentIndex, *, existing: Sequence[Question] = (), generation: str = GENERATION_VERSION,
                     true_false_share: float = 0.2, unlike: Sequence[str] = ()) -> Curation:
    """``unlike``: texts a question must not repeat - the fronts of the flashcards (the same task twice is not a second chance to learn something).
    Two questions that rest on the same sentence of the lecture are one question asked twice; a question and a flashcard on the same sentence are two formats of one fact and stay."""
    result = Curation()
    kept_stems = [_ask(q.question) for q in existing]
    card_stems = [_ask(text) for text in unlike]
    kept_facts = [_stems(q.evidence) for q in existing if q.evidence.strip()]
    openings = Counter(opening(q.question) for q in existing)
    all_count = len(existing)
    tf_count = sum(1 for q in existing if q.type == "true_false")
    for q in questions:
        verdict = review_question(q, index)
        stems = _ask(q.question)
        if verdict.ok and any(_alike(stems, k, NEAR_DUPLICATE) for k in kept_stems):
            verdict = Review(False, ["near-identical to another question"])
        elif verdict.ok and any(_alike(stems, k, CARD_REPEAT) for k in card_stems):
            verdict = Review(False, ["repeats a flashcard"])
        if verdict.ok:
            facts = _fact_stems(q, index)
            if facts and any(_jaccard(facts, k) >= FACT_REPEAT for k in kept_facts):
                verdict = Review(False, ["same fact as another question"])
        if verdict.ok and openings[opening(q.question)] >= SAME_OPENING_MAX:
            verdict = Review(False, ["same opening as other questions"])
        if verdict.ok and q.type == "true_false" and all_count >= 5 and (tf_count + 1) / (all_count + 1) > true_false_share:
            verdict = Review(False, ["too many true/false questions"])
        if not verdict.ok:
            result.rejected.append(Rejected(q, verdict.reasons))
            continue
        attach_evidence(q, index)
        q.difficulty = calibrate_question(q, index)
        q.generation = generation
        openings[opening(q.question)] += 1
        kept_stems.append(_ask(q.question))
        if q.evidence.strip():
            kept_facts.append(_stems(q.evidence))
        all_count += 1
        tf_count += q.type == "true_false"
        result.kept.append(q)
    return result


def summarise(curation: Curation) -> dict:
    return {"kept": len(curation.kept), "dropped": len(curation.rejected), "reasons": dict(curation.reasons().most_common(12))}
