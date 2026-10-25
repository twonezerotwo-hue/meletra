"""Taking the study material out of the program: a Markdown study guide and an Anki-ready file of the flashcards."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Sequence

from .models import CARD_TYPE_TITLES, QUESTION_TYPE_TITLES, SUMMARY_TITLES, StudyPack, SourceRef


def _refs(refs: Sequence[SourceRef]) -> str:
    return " [" + "; ".join(r.label for r in refs) + "]" if refs else ""


def to_markdown(pack: StudyPack) -> str:
    """A readable study guide. Answers of the questions are listed at the end (an answer key), not under each question."""
    lines = [f"# {pack.title or 'Lecture'} - study guide", ""]
    lines.append(f"*Made from: {pack.source_mode}. Source-bound: {'yes' if pack.source_bound else 'no'}. "
                 f"Study method: {pack.provider.name}{' (' + pack.provider.model + ')' if pack.provider.model else ''}.*")
    lines.append("")
    for kind, items in pack.summaries.items():
        lines += [f"## {SUMMARY_TITLES.get(kind, kind)}", ""]
        topic = None
        for item in items:
            if kind in ("detailed", "high_yield", "mechanisms", "clinical") and item.topic and item.topic != topic:
                topic = item.topic
                lines += [f"### {topic}", ""]
            lead = f"**{item.term}** - " if item.term else ""
            lines.append(f"- {lead}{item.text}{_refs(item.sources)}")
        lines.append("")
    if pack.flashcards:
        lines += ["## Flashcards", ""]
        for number, card in enumerate(pack.flashcards, start=1):
            lines += [f"{number}. **{card.front}** ({CARD_TYPE_TITLES.get(card.type, card.type)}, {card.difficulty})",
                      f"   - {card.back}{_refs(card.sources)}"]
        lines.append("")
    if pack.questions:
        lines += ["## Questions", ""]
        for number, q in enumerate(pack.questions, start=1):
            lines.append(f"{number}. ({QUESTION_TYPE_TITLES.get(q.type, q.type)}, {q.difficulty}) {q.question}")
            for letter, option in zip("ABCDE", q.options if len(q.options) > 2 else []):
                lines.append(f"    - {letter}. {option}")
        lines += ["", "### Answer key", ""]
        for number, q in enumerate(pack.questions, start=1):
            lines.append(f"{number}. **{q.answer}** - {q.explanation}{_refs(q.sources)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def to_anki_tsv(pack: StudyPack) -> str:
    """Tab-separated text Anki imports directly (File > Import): front, back, tags (topic, card type, difficulty)."""
    out = io.StringIO()
    out.write("#separator:tab\n#html:false\n#tags column:3\n")
    writer = csv.writer(out, delimiter="\t", lineterminator="\n")
    for card in pack.flashcards:
        tags = " ".join(t.replace(" ", "_") for t in (card.topic, card.type, card.difficulty) if t)
        back = card.back + _refs(card.sources)
        writer.writerow([card.front.replace("\t", " ").replace("\n", " "), back.replace("\t", " ").replace("\n", " "), tags])
    return out.getvalue()


def write_exports(pack: StudyPack, folder: Path) -> list[Path]:
    """``study_guide.md`` and ``flashcards_anki.txt`` next to the study files."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    targets = [(folder / "study_guide.md", to_markdown(pack)), (folder / "flashcards_anki.txt", to_anki_tsv(pack))]
    for path, text in targets:
        path.write_text(text, encoding="utf-8", newline="\n")
    return [p for p, _t in targets]
