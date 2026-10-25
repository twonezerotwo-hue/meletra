"""The prompts of the Automatic Study Engine for flashcards and questions.

What they ask for is what a good medical teacher writes: questions that need understanding rather than a remembered sentence, answers that say something, wrong options a
student could really be tempted by - all of it taken from the numbered passages of ONE lecture, with the exact sentence that proves it quoted back (``evidence``).
The prompts do not rely on the model's honesty: ``curate.py`` and ``grounding.py`` check every item afterwards, and what fails is dropped (the second round of a request is told why).
"""

from __future__ import annotations

import re
from typing import Sequence

from .models import Chunk
from .prompts import LANGUAGE_NAMES, render_passages

SYSTEM = (
    "You are an experienced medical educator who writes study material for medical students from ONE lecture. You work ONLY from the numbered passages the user gives you "
    "(a spoken transcript and/or slides). Rules: (1) Everything you write must be answerable from the passages. Never add a fact, number, dose, name or mechanism from your own "
    "knowledge, even when you are sure it is true. (2) Keep the lecture's medical terms and their spelling. (3) The passages are a transcript of speech: ignore greetings, jokes, "
    "remarks about the course or the exam schedule, and anything unclear or doubtful - leave an item out rather than guess. (4) Every item lists the ids of the passages it rests on in "
    "\"sources\" and quotes ONE sentence from those passages, word for word, in \"evidence\". (5) The transcript is machine-made and can contain misheard words (for example \"nuclides\" for \"nuclei\"): "
    "never build an item on a word or phrase that looks wrong or odd, and skip a passage that is garbled. (6) Write answers in the passages' own wording: add no example, consequence or "
    "explanation that the passages do not state. (7) Vary the wording: never begin more than two items with the same words, and do not use \"What is the clinical significance of\" or "
    "\"What is the sequence of events\" unless the passages state exactly that. (8) Reply with one JSON object and nothing else - no prose, no markdown fences."
)

CARD_KINDS = {
    "concept": "conceptual understanding: why or how something works, what it is for, how it fits in (not a bare definition)",
    "mechanism": "a mechanism: what activates, inhibits, converts or regulates what, and what follows",
    "cause_effect": "cause and effect: what happens when X is present, absent, changed or deficient",
    "comparison": "a comparison: how two things the lecture contrasts differ, or what separates look-alikes",
    "clinical": "clinical application: what a deficiency, mutation, drug or condition does to the patient or the test result",
    "pathway": "a sequence: the steps of a process in the order taught, or what comes before/after a step",
    "exam_recall": "active recall: a precise question whose answer is one specific term, number or name the lecture gives",
    "integration": "integration: a question that needs TWO facts of the lecture put together (cite both passages)",
    "definition": "a definition the lecture gives for a term that matters",
}
QUESTION_KINDS = {
    "mcq": "a single-best-answer multiple-choice question about a concept, a mechanism, a cause or a clinical consequence",
    "mechanism": "a multiple-choice question about HOW or WHY something happens (the mechanism, the step, the regulator)",
    "clinical_vignette": "a short patient scenario (3-5 sentences) built ONLY from facts of the passages, then a question the lecture lets you answer; add no lab value or detail the passages do not give",
    "short_answer": "a question answered in one or two sentences",
    "explain_why": "a question that asks the student to explain why something happens, answered from the lecture's reasoning",
    "true_false": "a true/false statement that tests a real distinction (use sparingly)",
}
DIFFICULTY_RULES = (
    "difficulty: \"easy\" = one stated fact; \"medium\" = a mechanism, a cause and its effect, or a comparison; \"hard\" = puts two facts or passages together, or reasons from a clinical picture."
)


def render(chunks: Sequence[Chunk], doubted: Sequence[str] = ()) -> str:
    """The passages as the model sees them. A word the medical pass could not resolve (a misheard term) is hidden as "[unclear]": the model cannot build an item on what is not shown."""
    text = render_passages(chunks)
    for word in doubted:
        word = word.strip()
        if len(word) >= 6:
            text = re.sub(r"(?<![\w-])" + re.escape(word) + r"(?![\w-])", "[unclear]", text, flags=re.IGNORECASE)
    return text


def _language_line(language: str) -> str:
    name = LANGUAGE_NAMES.get(language, language)
    return f"Write every question, answer, option and explanation in {name}; keep medical terms as they appear in the passages."


def _avoid(problems: Sequence[str], taken: Sequence[str]) -> str:
    lines = []
    if problems:
        lines.append("Your previous attempt was refused for these reasons - do not repeat them: " + "; ".join(problems) + ".")
    if taken:
        lines.append("Already written (do not repeat or rephrase): " + " | ".join(t[:90] for t in taken[-14:]) + ".")
    return ("\n".join(lines) + "\n") if lines else ""


def flashcard_prompt(chunks: Sequence[Chunk], topic: str, kinds: Sequence[str], limit: int, language: str, problems: Sequence[str] = (), taken: Sequence[str] = (),
                     doubted: Sequence[str] = ()) -> str:
    kind_lines = "\n".join(f"  - {k}: {CARD_KINDS[k]}" for k in kinds if k in CARD_KINDS)
    schema = ('{"cards": [{"type": "<one of the kinds>", "front": "...", "back": "...", "evidence": "<one sentence copied from a passage>", "difficulty": "easy|medium|hard", '
              '"sources": ["<passage id>"]}]}')
    scope = f"Topic: {topic}\n" if topic else ""
    return (f"Task: write up to {limit} flashcards that teach this part of the lecture. Mix these kinds:\n{kind_lines}\n"
            "Rules for every card: the \"front\" is a complete, self-contained question or prompt a student can answer without the lecture beside them (never a sentence with a blank, "
            "never a yes/no question, never \"what does the lecture say\", no \"this\" or \"it\" without saying what); the \"back\" starts directly with the answer, in one to three sentences "
            "and at most 45 words, is made from the words of your evidence sentence (add nothing it does not say), and says more than the front already says; one idea per card, and no two cards on the same fact; if the passages hold fewer good cards than asked, "
            f"write fewer. {DIFFICULTY_RULES} {_language_line(language)}\n{scope}{_avoid(problems, taken)}Answer as JSON: {schema}\n\nPASSAGES\n{render(chunks, doubted)}")


def question_prompt(chunks: Sequence[Chunk], topic: str, kinds: Sequence[str], limit: int, language: str, problems: Sequence[str] = (), taken: Sequence[str] = (),
                    doubted: Sequence[str] = ()) -> str:
    kind_lines = "\n".join(f"  - {k}: {QUESTION_KINDS[k]}" for k in kinds if k in QUESTION_KINDS)
    schema = ('{"questions": [{"type": "<one of the kinds>", "question": "...", "options": ["...", "...", "...", "..."], "answer": "<for multiple choice exactly one of the options; '
              'for true_false True or False; otherwise the model answer>", "explanation": "...", "evidence": "<one sentence copied from a passage>", "difficulty": "easy|medium|hard", '
              '"sources": ["<passage id>"]}]}')
    scope = f"Topic: {topic}\n" if topic else ""
    return (f"Task: write up to {limit} exam questions for this part of the lecture. Mix these kinds (mostly multiple choice):\n{kind_lines}\n"
            "Rules: every question is a complete question or scenario (never a sentence with a blank). Multiple-choice (mcq, mechanism, clinical_vignette): EXACTLY four options with ONE "
            "defensible best answer; the three wrong options are plausible - things a student who half-knows the topic would choose, of the same kind and similar length as the right "
            "one, preferably other concepts of this lecture - but each is clearly wrong according to the passages; never \"all of the above\" or \"none of the above\"; the question must "
            "not contain the answer. short_answer / explain_why / true_false: \"options\" is [] (true_false: [\"True\", \"False\"]) and the answer is a model answer of one or two sentences. "
            "The \"explanation\" says why the answer is right and, when the passages show it, why the most tempting wrong option is wrong. "
            f"If the passages hold fewer good questions than asked, write fewer. {DIFFICULTY_RULES} {_language_line(language)}\n{scope}{_avoid(problems, taken)}"
            f"Answer as JSON: {schema}\n\nPASSAGES\n{render(chunks, doubted)}")
