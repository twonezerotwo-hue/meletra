"""The study engine: the whole pack, provider failure and fallback, grounding of model output, cancel, saving."""

import json

import pytest

from src.study.engine import StudyCancelled, StudyEngine, StudyOptions
from src.study.extractive import ExtractiveProvider
from src.study.models import (CARD_TYPES, QUESTION_TYPES, SUMMARY_KINDS, Flashcard, ProviderInfo, Question, SourceRef, StudyPack,
                              SummaryItem)
from src.study.providers import NOT_IN_SOURCES, AIProvider, Evaluation, ProviderError
from src.study.store import StudyStore
from study_support import lecture_index


@pytest.fixture(scope="module")
def idx():
    return lecture_index(with_slides=True)


class ScriptedProvider(AIProvider):
    """A language-model stand-in whose answers the test writes."""

    name, kind, model = "scripted", "local", "fake"

    def __init__(self, summary=None, cards=None, questions=None, check_error=None, explain=None, error=None):
        self.summary, self.cards, self.questions = summary or [], cards or [], questions or []
        self.check_error, self.error, self._explain = check_error, error, explain
        self.calls = []

    def check(self):
        if self.check_error:
            raise ProviderError(self.check_error)

    def _maybe_fail(self, what):
        self.calls.append(what)
        if self.error:
            raise self.error

    def summarize(self, kind, chunks, *, topic, limit, ctx):
        self._maybe_fail("summarize")
        return list(self.summary)

    def generate_flashcards(self, chunks, *, topic, types, limit, ctx):
        self._maybe_fail("cards")
        return list(self.cards)

    def generate_questions(self, chunks, *, topic, types, limit, ctx):
        self._maybe_fail("questions")
        return list(self.questions)

    def explain(self, question, chunks, ctx):
        self._maybe_fail("explain")
        return self._explain or SummaryItem("Hexokinase phosphorylates glucose.", [SourceRef.of(chunks[0])])

    def evaluate_answer(self, question, answer, chunks, ctx):
        self._maybe_fail("evaluate")
        return Evaluation(True, 1.0, "scripted")


def ref(idx, chunk_id="t0000"):
    return [SourceRef.of(idx.get(chunk_id))]


def test_builtin_pack_has_everything_and_valid_sources(idx):
    steps = []
    pack = StudyEngine().generate(idx, lecture_id="lec1", title="Glycolysis and collagen", progress=lambda f, m: steps.append((f, m)))
    assert pack.source_mode == "pdf+transcript" and pack.source_bound and pack.provider.name == "builtin" and not pack.warnings
    assert set(pack.summaries) == set(SUMMARY_KINDS)
    assert pack.flashcards and pack.questions and pack.topics == idx.topic_names()
    assert {c.type for c in pack.flashcards} <= set(CARD_TYPES) and {q.type for q in pack.questions} <= set(QUESTION_TYPES)
    for item in [*[i for v in pack.summaries.values() for i in v], *pack.flashcards, *pack.questions]:
        assert item.sources and all(idx.get(s.chunk_id) for s in item.sources)
    assert pack.stats["grounding"]["dropped"] == 0 and pack.stats["flashcards"] == len(pack.flashcards)
    assert steps[0][0] == 0.0 and steps[-1] == (1.0, "Study material ready")
    assert [f for f, _ in steps] == sorted(f for f, _ in steps)


def test_transcript_only_and_pdf_only_modes_both_work():
    t_only = StudyEngine().generate(lecture_index(with_slides=False), lecture_id="a")
    assert t_only.source_mode == "transcript" and t_only.flashcards
    from study_support import SLIDES
    from src.study.index import build_index

    pdf_only = StudyEngine().generate(build_index([], SLIDES * 3), lecture_id="b")
    assert pdf_only.source_mode == "pdf" and pdf_only.summaries.get("quick")
    assert all(s.kind == "pdf" for v in pdf_only.summaries.values() for i in v for s in i.sources)


def test_pack_survives_json_and_disk(idx, tmp_path):
    pack = StudyEngine().generate(idx, lecture_id="lec1", title="T")
    again = StudyPack.from_dict(json.loads(json.dumps(pack.to_dict())))
    assert again.to_dict() == pack.to_dict()
    store = StudyStore(tmp_path)
    store.save_pack(pack)
    store.save_index(idx)
    assert store.has_pack() and store.load_pack().to_dict() == pack.to_dict()
    assert store.load_index().topic_names() == idx.topic_names()
    assert (tmp_path / "study" / "study_pack.json").is_file()


def test_damaged_pack_file_is_set_aside_not_fatal(tmp_path):
    store = StudyStore(tmp_path)
    store.folder.mkdir(parents=True)
    store.pack_path.write_text("{broken", encoding="utf-8")
    assert store.load_pack() is None and (store.folder / "study_pack.json.bad").is_file()


def test_unavailable_provider_falls_back_to_builtin_with_a_note(idx):
    provider = ScriptedProvider(check_error="The model 'x' is not installed in Ollama.")
    pack = StudyEngine(provider).generate(idx, lecture_id="x")
    assert pack.flashcards and pack.summaries and provider.calls == []
    assert pack.provider.name == "builtin" and pack.provider.fallback_from == "scripted"
    assert any("not installed" in w for w in pack.warnings)


def test_provider_that_breaks_midway_never_empties_the_tabs(idx):
    provider = ScriptedProvider(error=ProviderError("Cannot reach localhost:11434."))
    pack = StudyEngine(provider).generate(idx, lecture_id="x")
    assert provider.calls == ["summarize"]                           # one failure, then the built-in provider for everything
    assert pack.summaries and pack.flashcards and pack.questions
    assert pack.provider.fallback_from == "scripted" and any("Cannot reach" in w for w in pack.warnings)


def test_unexpected_exception_in_a_provider_is_handled_like_a_provider_error(idx):
    pack = StudyEngine(ScriptedProvider(error=KeyError("boom"))).generate(idx, lecture_id="x")
    assert pack.flashcards and pack.provider.fallback_from == "scripted"


def test_model_output_is_grounded_and_invented_facts_are_dropped(idx):
    good = SummaryItem("Hexokinase phosphorylates glucose to glucose-6-phosphate.", ref(idx))
    invented = SummaryItem("Hexokinase deficiency affects 40 percent of newborns in Europe.", ref(idx))
    provider = ScriptedProvider(summary=[good, invented])
    pack = StudyEngine(provider).generate(idx, lecture_id="x", options=StudyOptions(summary_kinds=["quick"]), only={"summary:quick"})
    texts = [i.text for i in pack.summaries["quick"]]
    assert texts == [good.text] and pack.stats["grounding"]["reasons"]["unsupported"] >= 1
    assert pack.provider.name == "scripted" and not pack.warnings


def test_empty_model_answer_for_a_section_is_filled_in_by_the_builtin_provider(idx):
    provider = ScriptedProvider(summary=[], cards=[], questions=[])
    pack = StudyEngine(provider).generate(idx, lecture_id="x")
    assert pack.flashcards and pack.summaries["quick"]
    assert any("gave nothing usable" in w for w in pack.warnings)


def test_model_cards_and_questions_pass_when_supported(idx):
    card = Flashcard("c1", "definition", "What does hexokinase do?", "It phosphorylates glucose to glucose-6-phosphate.", ref(idx))
    question = Question("q1", "mcq", "Which enzyme phosphorylates glucose?", "Hexokinase", "Hexokinase phosphorylates glucose.",
                        ["Hexokinase", "Aldolase", "Enolase", "Lactate dehydrogenase", "Glucokinase"], ref(idx))
    provider = ScriptedProvider(cards=[card], questions=[question], summary=[SummaryItem("Glycolysis converts glucose into pyruvate.", ref(idx))])
    pack = StudyEngine(provider).generate(idx, lecture_id="x")
    assert [c.front for c in pack.flashcards] == [card.front] and [q.id for q in pack.questions] == ["q1"]
    assert pack.flashcards[0].topic and pack.questions[0].topic            # topic taken from the cited passage


def test_regenerating_one_part_keeps_the_rest(idx):
    pack = StudyEngine().generate(idx, lecture_id="x")
    before = {k: [i.text for i in v] for k, v in pack.summaries.items()}
    cards = [c.id for c in pack.flashcards]
    again = StudyEngine().generate(idx, lecture_id="x", only={"questions"}, base=pack)
    assert again is pack and [c.id for c in again.flashcards] == cards
    assert {k: [i.text for i in v] for k, v in again.summaries.items()} == before


def test_cancel_stops_generation(idx):
    with pytest.raises(StudyCancelled):
        StudyEngine().generate(idx, lecture_id="x", cancelled=lambda: True)


def test_option_limits_are_respected(idx):
    pack = StudyEngine().generate(idx, lecture_id="x", options=StudyOptions(summary_kinds=["quick"], card_types=["definition"],
                                                                         question_types=["true_false"], max_cards=4, max_questions=3))
    assert list(pack.summaries) == ["quick"] and len(pack.flashcards) <= 4 and len(pack.questions) <= 3
    assert {c.type for c in pack.flashcards} == {"definition"} and {q.type for q in pack.questions} == {"true_false"}


def test_explain_is_grounded_and_says_when_the_lecture_does_not_cover_it(idx):
    engine = StudyEngine()
    assert "hexokinase" in engine.explain(idx, "What does hexokinase do?").text.casefold()
    assert engine.explain(idx, "How is hypertension treated with lisinopril?").text == NOT_IN_SOURCES
    liar = StudyEngine(ScriptedProvider(explain=SummaryItem("Hexokinase cures cancer in 90 percent of cases.", ref(idx))))
    assert liar.explain(idx, "What does hexokinase do?").text == NOT_IN_SOURCES


def test_typed_answers_are_judged_by_the_builtin_when_the_model_fails(idx):
    question = Question("q", "short_answer", "What is glycolysis?", "Glycolysis converts glucose into pyruvate.", "", sources=ref(idx))
    broken = StudyEngine(ScriptedProvider(error=ProviderError("down")))
    assert broken.evaluate(idx, question, "converts glucose into pyruvate").correct
    working = StudyEngine(ScriptedProvider())
    assert working.evaluate(idx, question, "anything").feedback == "scripted"


def test_provider_info_is_recorded_in_the_pack(idx):
    pack = StudyEngine(ScriptedProvider(summary=[SummaryItem("Glycolysis converts glucose into pyruvate.", ref(idx))])).generate(idx, lecture_id="x")
    assert pack.provider == ProviderInfo(name="scripted", kind="local", model="fake")


def test_study_engine_does_not_import_the_transcription_pipeline():
    import subprocess
    import sys

    code = ("import sys; import src.study.engine, src.study.session, src.study.store; "
            "bad = [m for m in ('src.pipeline', 'src.transcriber', 'faster_whisper', 'torch') if m in sys.modules]; print(bad)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "[]"


# ---------------------------------------------------------------------------------------------------------------------- export
def test_markdown_study_guide_and_anki_file(idx, tmp_path):
    from src.study.export import to_anki_tsv, to_markdown, write_exports

    pack = StudyEngine().generate(idx, lecture_id="x", title="Glycolysis and collagen")
    md = to_markdown(pack)
    assert md.startswith("# Glycolysis and collagen - study guide") and "## Quick Summary" in md and "## Flashcards" in md
    assert "### Answer key" in md and "[Transcript" in md                                # sources and the answers at the end
    first_question = md.split("## Questions")[1].split("### Answer key")[0]
    assert pack.questions[0].answer not in first_question.split("\n")[1] or pack.questions[0].type == "mcq"
    anki = to_anki_tsv(pack)
    lines = anki.splitlines()
    assert lines[0] == "#separator:tab" and lines[2] == "#tags column:3"
    rows = [ln.split("\t") for ln in lines[3:]]
    assert len(rows) == len(pack.flashcards) and all(len(r) == 3 and r[0] and r[1] for r in rows)
    assert all(" " not in r[2].strip() or r[2].count(" ") == 2 for r in rows)             # tags are separated by spaces, never contain them
    written = write_exports(pack, tmp_path / "study")
    assert [p.name for p in written] == ["study_guide.md", "flashcards_anki.txt"] and all(p.read_text(encoding="utf-8") for p in written)
