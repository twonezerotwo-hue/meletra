"""Prompts for language-model providers. Kept apart from the transport code so they can be read, tested and changed on their own.

The contract with the model: it sees numbered passages (id, label, text) and may use nothing else; it answers with one JSON object;
every item names the passage ids it is based on. ``grounding.py`` verifies those claims afterwards - the prompt asks for honesty, the
code does not rely on it.
"""

from __future__ import annotations

import json
from typing import Sequence

from .models import (CARD_TYPE_TITLES, QUESTION_TYPE_TITLES, SUMMARY_TITLES, Chunk, Question)

LANGUAGE_NAMES = {"en": "English", "tr": "Turkish", "de": "German", "fr": "French", "es": "Spanish", "ru": "Russian", "ar": "Arabic",
                  "bg": "Bulgarian", "hu": "Hungarian", "it": "Italian", "pt": "Portuguese", "pl": "Polish"}

SYSTEM = (
    "You are a careful study assistant for medical students. You work ONLY from the numbered passages the user gives you, which come "
    "from one lecture (spoken transcript and/or slides). Rules: (1) State only what the passages say. Never add facts, numbers, "
    "doses, names or mechanisms from your own knowledge, even if you are sure they are true. (2) Keep the lecture's terminology and "
    "spelling of medical terms. (3) Keep the order of the lecture. (4) If the passages do not contain enough for an item, leave the "
    "item out instead of inventing it. (5) Every item lists the ids of the passages it is based on in \"sources\". (6) Reply with one "
    "JSON object and nothing else - no prose, no markdown fences."
)

SUMMARY_INSTRUCTIONS = {
    "quick": "Write a quick summary: the {limit} most important points of the whole text, each one sentence, in lecture order.",
    "detailed": "Write a detailed summary of this part of the lecture: up to {limit} points that follow the lecture order and cover "
                "what was taught, each one or two sentences.",
    "high_yield": "Pick the high-yield points a student must not miss (what the lecturer stressed, definitions, mechanisms with "
                  "clinical meaning): up to {limit} short points.",
    "key_concepts": "List up to {limit} key concepts. For each give the concept name in \"term\" and one sentence explaining it as "
                    "the lecture does.",
    "definitions": "List up to {limit} important definitions that the passages state. \"term\" is the defined word, \"text\" is the "
                   "definition as the lecture gives it.",
    "mechanisms": "List up to {limit} mechanisms or pathways the lecture explains: what activates / converts / inhibits what, step "
                  "by step as taught.",
    "clinical": "List up to {limit} clinical connections the lecture makes (diseases, deficiencies, symptoms, treatments) - only "
                "those that are in the passages.",
    "emphasis": "List up to {limit} points the lecturer explicitly stressed (\"important\", \"remember\", \"you should know\", "
                "\"exam\"). Skip points that were not stressed.",
    "exam": "List up to {limit} points most likely to be asked in an exam, judged only by how the lecture treats them (stressed, "
            "defined, repeated, shown on slides).",
}


def render_passages(chunks: Sequence[Chunk]) -> str:
    return "\n\n".join(f"[{c.id} | {c.label()}{' | ' + c.title if c.title else ''}]\n{c.text}" for c in chunks)


def _language_line(language: str) -> str:
    name = LANGUAGE_NAMES.get(language, language)
    return f"Write the output in {name}. Keep medical terms as they appear in the passages."


def summary_prompt(kind: str, chunks: Sequence[Chunk], topic: str, limit: int, language: str) -> str:
    instruction = SUMMARY_INSTRUCTIONS[kind].format(limit=limit)
    schema = '{"items": [{"text": "...", "term": "(only for concepts / definitions, else \\"\\")", "sources": ["<passage id>", ...]}]}'
    scope = f"Topic: {topic}\n" if topic else ""
    return (f"Task: {SUMMARY_TITLES[kind]}. {instruction}\n{_language_line(language)}\n{scope}"
            f"Answer as JSON: {schema}\n\nPASSAGES\n{render_passages(chunks)}")


def flashcard_prompt(chunks: Sequence[Chunk], topic: str, types: Sequence[str], limit: int, language: str) -> str:
    kinds = "; ".join(f"{t} ({CARD_TYPE_TITLES[t]})" for t in types)
    schema = '{"cards": [{"type": "<one of the types>", "front": "...", "back": "...", "difficulty": "easy|medium|hard", "sources": ["<passage id>"]}]}'
    scope = f"Topic: {topic}\n" if topic else ""
    return (f"Task: write up to {limit} flashcards. Allowed types: {kinds}. Spread them over the types and the passages; one fact per "
            f"card; the front is a question or prompt, the back is a short answer taken from the passages. {_language_line(language)}\n"
            f"{scope}Answer as JSON: {schema}\n\nPASSAGES\n{render_passages(chunks)}")


def question_prompt(chunks: Sequence[Chunk], topic: str, types: Sequence[str], limit: int, language: str) -> str:
    kinds = "; ".join(f"{t} ({QUESTION_TYPE_TITLES[t]})" for t in types)
    schema = ('{"questions": [{"type": "<one of the types>", "question": "...", "options": ["...", "...", "...", "...", "..."], '
              '"answer": "<for mcq exactly one of the options; for true_false True or False; otherwise the model answer>", '
              '"explanation": "why, based on the passages", "difficulty": "easy|medium|hard", "sources": ["<passage id>"]}]}')
    scope = f"Topic: {topic}\n" if topic else ""
    return (f"Task: write up to {limit} exam questions. Allowed types: {kinds}. Rules: mcq has EXACTLY 5 options, one correct, four "
            f"plausible wrong ones that are different from each other; true_false has options [\"True\", \"False\"]; short_answer, "
            f"explain_why and mechanism need no options (use [] ) and have a short model answer; clinical_vignette is a short patient "
            f"scenario built only from facts of the passages, with 5 options. The explanation must cite what the passages say. "
            f"{_language_line(language)}\n{scope}Answer as JSON: {schema}\n\nPASSAGES\n{render_passages(chunks)}")


def explain_prompt(question: str, chunks: Sequence[Chunk], language: str) -> str:
    schema = '{"answer": "...", "sources": ["<passage id>"]}'
    return (f"Task: answer the student's question using only the passages. If the passages do not answer it, reply with "
            f"{{\"answer\": \"NOT_COVERED\", \"sources\": []}}. {_language_line(language)}\nQuestion: {question}\n"
            f"Answer as JSON: {schema}\n\nPASSAGES\n{render_passages(chunks)}")


def evaluate_prompt(question: Question, answer: str, chunks: Sequence[Chunk], language: str) -> str:
    schema = '{"correct": true, "score": 0.0, "feedback": "...", "missing": ["key point the student missed", "..."]}'
    return (f"Task: judge a student's answer against the model answer and the passages. Be fair: wording may differ, the meaning "
            f"must match. {_language_line(language)}\nQuestion: {question.question}\nModel answer: {question.answer}\n"
            f"Student's answer: {answer}\nAnswer as JSON: {schema}\n\nPASSAGES\n{render_passages(chunks)}")


def parse_json(raw: str) -> dict:
    """The first JSON object in the model's reply (models sometimes add fences or a sentence around it)."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):] if "{" in text else text
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in the reply")
    depth, in_string, escaped = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("the JSON object in the reply is incomplete")
