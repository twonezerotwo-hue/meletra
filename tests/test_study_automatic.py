"""FEATURE-06: the MELETRA Automatic Study Engine - nobody chooses a method, the best one that is available is used, cloud only on a separate consent, and whatever a model
writes passes the premium gate before the student sees it."""

from __future__ import annotations

import json
import re

import pytest

from src.services.settings_service import Settings
from src.study import curate
from src.study.automatic import ENGINE_NAME, MAX_ROUNDS, AutomaticStudyEngine, choose_engine, pick_local_model
from src.study.engine import StudyOptions
from src.study.providers import ProviderError
from tests.study_support import lecture_index


@pytest.fixture(scope="module")
def idx():
    return lecture_index()


def chunk_id(idx, text):
    return next(c.id for c in idx.chunks if text in c.text)


class FakeOllama:
    """Ollama on this computer, scripted: /api/tags lists the models, /api/chat answers what ``cards`` / ``questions`` (functions of the prompt) say."""

    def __init__(self, models=("qwen3:8b", "qwen2.5vl:7b"), cards=None, questions=None, fail_chat=False):
        self.models, self.cards, self.questions, self.fail_chat = list(models), cards, questions, fail_chat
        self.urls: list[str] = []
        self.prompts: list[str] = []

    def __call__(self, url, headers, body, timeout):
        self.urls.append(url)
        if url.endswith("/api/tags"):
            return {"models": [{"name": m} for m in self.models]}
        if self.fail_chat:
            raise ProviderError("Ollama stopped answering")
        user = body["messages"][1]["content"]
        self.prompts.append(user)
        make = self.cards if "flashcards" in user[:90] else self.questions
        return {"message": {"content": json.dumps(make(user) if make else {})}}

    def chat_prompts(self, kind):
        return [p for p in self.prompts if kind in p[:90]]


def ids_in(prompt):
    return set(re.findall(r"^\[(t\d+) \|", prompt, re.M))


def the_model_writes_cards(idx, extra=()):
    good = [
        dict(anchor="Pyruvate kinase catalyzes", type="clinical", front="Why do patients with pyruvate kinase deficiency develop hemolytic anemia?",
             back="Red blood cells depend on glycolysis for ATP, and without pyruvate kinase the last step of glycolysis fails.", difficulty="medium",
             evidence="Pyruvate kinase deficiency causes hemolytic anemia because red blood cells depend on glycolysis for ATP."),
        dict(anchor="Phosphofructokinase is the key", type="mechanism", front="How does a high ATP level change the rate of glycolysis, and through which enzyme?",
             back="High ATP inhibits phosphofructokinase, which slows glycolysis when the cell has enough energy.", difficulty="medium",
             evidence="High levels of ATP inhibit phosphofructokinase, which slows glycolysis when the cell has enough energy."),
        dict(anchor="Vitamin C deficiency causes scurvy", type="cause_effect", front="What links a lack of vitamin C to the fragile blood vessels of scurvy?",
             back="Without vitamin C proline cannot be hydroxylated, so the collagen is unstable and the vessel walls are weak.", difficulty="hard",
             evidence="Vitamin C deficiency causes scurvy because collagen without hydroxyproline is unstable."),
        dict(anchor="Lysyl oxidase forms", type="exam_recall", front="Which enzyme forms the covalent cross links between collagen molecules, and which metal does it need?",
             back="Lysyl oxidase forms the cross links and needs copper.", difficulty="easy", evidence="Lysyl oxidase forms the covalent cross links between collagen molecules and needs copper."),
    ]
    bad = [
        dict(anchor="Pyruvate kinase catalyzes", type="exam_recall", front="Fill in the blank: _____ catalyzes the last step of glycolysis.", back="Pyruvate kinase catalyzes the last step.",
             evidence="Pyruvate kinase catalyzes the last step of glycolysis and produces pyruvate and ATP."),
        dict(anchor="Net glycolysis gives", type="concept", front="How many ATP and NADH does net glycolysis give per glucose molecule?", back="Net glycolysis gives two ATP and two NADH per glucose molecule.",
             evidence="Net glycolysis gives two ATP and two NADH per glucose molecule."),                                                    # a fine card: kept, then its twin below is a duplicate
        dict(anchor="Net glycolysis gives", type="concept", front="How many ATP and NADH does net glycolysis give per glucose molecule?", back="Two ATP and two NADH per glucose molecule from net glycolysis.",
             evidence="Net glycolysis gives two ATP and two NADH per glucose molecule."),
        dict(anchor="Hexokinase has a low Km", type="mechanism", front="At what blood concentration is hexokinase half saturated, and why does it matter?", back="At 0.05 mM, so it works at fasting glucose.",
             evidence="Hexokinase is half saturated at 0.05 mM glucose."),                                                                  # invented number
        dict(anchor="Collagen is the most abundant", type="concept", front="What does the lecture say about collagen?", back="It is the most abundant structural protein of the extracellular matrix.",
             evidence="Collagen is the most abundant structural protein of the extracellular matrix."),
    ]

    def make(prompt):
        here = ids_in(prompt)
        cards = []
        for item in [*good, *bad, *extra]:
            cid = chunk_id(idx, item["anchor"])
            if cid in here:
                cards.append({k: v for k, v in item.items() if k != "anchor"} | {"sources": [cid]})
        return {"cards": cards}

    return make


def the_model_writes_questions(idx):
    good = [
        dict(anchor="Phosphofructokinase is the key", type="mcq", question="Which enzyme is the key regulator of glycolysis, slowed down when ATP is plentiful?", answer="Phosphofructokinase",
             options=["Phosphofructokinase", "Hexokinase", "Enolase", "Aldolase"], explanation="Phosphofructokinase is the key regulatory enzyme of glycolysis, and high ATP inhibits it, so glycolysis slows when the cell has enough energy.",
             evidence="High levels of ATP inhibit phosphofructokinase, which slows glycolysis when the cell has enough energy.", difficulty="medium"),
        dict(anchor="Under anaerobic conditions lactate", type="mechanism", question="Why can glycolysis continue in a muscle that has no oxygen?", answer="Lactate dehydrogenase converts pyruvate to lactate and regenerates NAD+.",
             options=["Lactate dehydrogenase converts pyruvate to lactate and regenerates NAD+.", "Pyruvate kinase makes extra ATP in the cytoplasm.", "Hexokinase traps glucose inside the cell.",
                      "Enolase converts 2-phosphoglycerate to phosphoenolpyruvate."], explanation="Under anaerobic conditions lactate dehydrogenase converts pyruvate to lactate and regenerates NAD+, which glycolysis needs.",
             evidence="Under anaerobic conditions lactate dehydrogenase converts pyruvate to lactate and regenerates NAD+.", difficulty="medium"),
        dict(anchor="Lysyl oxidase forms", type="mcq", question="A diet very low in copper weakens collagen. Which enzyme step explains this?", answer="Lysyl oxidase cross linking",
             options=["Lysyl oxidase cross linking", "Proline hydroxylation", "Procollagen cleavage", "Glycosylation of hydroxylysine"],
             explanation="Lysyl oxidase forms the covalent cross links between collagen molecules and needs copper, so copper deficiency reduces the strength of collagen.",
             evidence="Copper deficiency therefore reduces the strength of collagen.", difficulty="hard"),
    ]
    bad = [
        dict(anchor="Phosphofructokinase is the key", type="mcq", question="Which term correctly completes this statement? _____ is the key regulatory enzyme of glycolysis.", answer="Phosphofructokinase",
             options=["Phosphofructokinase", "Hexokinase", "Enolase", "Aldolase"], explanation="Phosphofructokinase is the key regulatory enzyme of glycolysis.",
             evidence="Phosphofructokinase is the key regulatory enzyme of glycolysis and it is activated by AMP and fructose-2,6-bisphosphate."),
        dict(anchor="Pyruvate kinase catalyzes", type="mcq", question="Which enzyme catalyzes the last step of glycolysis?", answer="Pyruvate kinase",
             options=["Pyruvate kinase", "Hexokinase", "Enolase", "All of the above"], explanation="Pyruvate kinase catalyzes the last step of glycolysis and produces pyruvate and ATP.",
             evidence="Pyruvate kinase catalyzes the last step of glycolysis and produces pyruvate and ATP."),
    ]

    def make(prompt):
        here = ids_in(prompt)
        out = []
        for item in [*good, *bad]:
            cid = chunk_id(idx, item["anchor"])
            if cid in here:
                out.append({k: v for k, v in item.items() if k != "anchor"} | {"sources": [cid]})
        return {"questions": out}

    return make


# --------------------------------------------------------------------------------------------------------------------- choosing
def test_nobody_has_to_choose_a_method_the_best_local_model_is_found():
    fake = FakeOllama(models=("qwen2.5vl:7b", "nomic-embed-text", "qwen3:4b", "qwen3:8b"))
    choice = choose_engine(Settings(), transport=fake)
    assert choice.mode == "local" and choice.model == "qwen3:8b" and ENGINE_NAME in choice.label and "local AI" in choice.label


def test_without_a_local_model_the_builtin_method_is_used_and_nothing_is_sent_anywhere():
    def no_server(url, headers, body, timeout):
        raise ProviderError("cannot reach localhost")
    choice = choose_engine(Settings(), transport=no_server)
    assert choice.mode == "builtin" and "offline" in choice.label


def test_a_vision_or_embedding_model_is_never_chosen_for_text():
    assert pick_local_model(["qwen2.5vl:7b", "nomic-embed-text:latest", "llava:7b"]) == ""
    assert pick_local_model(["llama3.2:3b", "qwen3:8b"]) == "qwen3:8b" and pick_local_model(["mistral:7b"]) == "mistral:7b"
    assert pick_local_model(["qwen3:4b", "qwen3:8b"], wanted="qwen3:4b") == "qwen3:4b"


class Keys:
    def get_study_key(self, kind):
        return "sk-test-key"


def test_a_cloud_study_provider_is_not_used_without_a_separate_consent():
    fake = FakeOllama(models=())
    settings = Settings(study_provider="openai", study_model="gpt-4o-mini")
    choice = choose_engine(settings, Keys(), transport=fake)
    assert choice.mode == "builtin" and any("not been consented" in n for n in choice.notes)
    assert not [u for u in fake.urls if "openai.com" in u]                                         # nothing was even asked of the cloud


def test_with_the_separate_consent_and_a_key_the_cloud_provider_is_used():
    settings = Settings(study_provider="anthropic", study_cloud_consent=True)
    choice = choose_engine(settings, Keys(), transport=FakeOllama(models=()))
    assert choice.mode == "cloud" and "sent to the provider" in choice.label


def test_cloud_consent_never_lets_a_server_on_this_computer_ignore_a_running_transcription():
    settings = Settings(study_provider="openai", study_model="local-model", study_base_url="http://localhost:1234/v1", study_cloud_consent=True)
    busy = choose_engine(settings, Keys(), transport=FakeOllama(models=()), busy=True)
    assert busy.mode == "builtin" and busy.deferred                                                  # the consent is for the cloud: this server shares the graphics card
    assert choose_engine(settings, Keys(), transport=FakeOllama(models=()), busy=False).mode == "local"


def test_a_server_on_this_computer_needs_no_cloud_consent():
    settings = Settings(study_provider="openai", study_model="local-model", study_base_url="http://localhost:1234/v1")
    assert choose_engine(settings, Keys(), transport=FakeOllama(models=())).mode == "local"


def test_the_old_cloud_setting_is_switched_off_by_the_migration():
    from src.services.settings_service import migrate, sanitize
    old = sanitize(migrate({"schema_version": 5, "study_provider": "anthropic", "study_cloud_consent": True}))
    assert old.study_provider == "anthropic" and old.study_cloud_consent is False                  # configured before, never consented to in the new sense


# --------------------------------------------------------------------------------------------------------------------- the built-in method
def test_the_builtin_pack_has_no_blanks_and_every_item_carries_its_proof(idx):
    pack = AutomaticStudyEngine(choose_engine(Settings(), transport=FakeOllama(models=()))).generate(idx, lecture_id="x", title="t", options=StudyOptions())
    assert pack.flashcards and pack.questions and pack.summaries
    for c in pack.flashcards:
        assert "_____" not in c.front and not c.front.lower().startswith("fill in") and c.evidence and c.generation == curate.GENERATION_VERSION and c.sources
    for q in pack.questions:
        assert "_____" not in q.question and "correctly completes" not in q.question and q.explanation and q.evidence and q.generation == curate.GENERATION_VERSION
        if q.type == "mcq":
            assert len(q.options) in (4, 5) and q.answer in q.options and len(set(q.options)) == len(q.options)
    assert pack.generation["engine"] == ENGINE_NAME and pack.generation["mode"] == "builtin"
    assert any(q.type == "mcq" for q in pack.questions) or any(q.type == "mechanism" for q in pack.questions)           # real multiple choice, not only true/false and short answers


def test_the_builtin_pack_has_no_near_duplicates(idx):
    pack = AutomaticStudyEngine(choose_engine(Settings(), transport=FakeOllama(models=()))).generate(idx, lecture_id="x", title="t", options=StudyOptions())
    fronts = [c.front.casefold() for c in pack.flashcards]
    assert len(set(fronts)) == len(fronts)


# --------------------------------------------------------------------------------------------------------------------- the language model
def run_with_model(idx, fake):
    choice = choose_engine(Settings(), transport=fake)
    assert choice.mode == "local"
    return AutomaticStudyEngine(choice).generate(idx, lecture_id="x", title="t", options=StudyOptions(max_cards=12, max_questions=8))


def test_good_model_items_are_kept_and_bad_ones_never_reach_the_student(idx):
    fake = FakeOllama(cards=the_model_writes_cards(idx), questions=the_model_writes_questions(idx))
    pack = run_with_model(idx, fake)
    fronts = [c.front for c in pack.flashcards]
    assert "Why do patients with pyruvate kinase deficiency develop hemolytic anemia?" in fronts and "How does a high ATP level change the rate of glycolysis, and through which enzyme?" in fronts
    assert not [f for f in fronts if "Fill in the blank" in f or "the lecture say" in f]                                                    # blanks and vague prompts refused
    assert sum(1 for f in fronts if f.startswith("How many ATP and NADH")) == 1                                                            # the twin was a duplicate
    assert not [c for c in pack.flashcards if "0.05" in c.back]                                                                            # the invented number is not in the lecture
    stems = [q.question for q in pack.questions]
    assert "Which enzyme is the key regulator of glycolysis, slowed down when ATP is plentiful?" in stems
    assert not [s for s in stems if "completes this statement" in s]
    assert not [q for q in pack.questions if any("all of the above" in o.lower() for o in q.options)]
    premium = pack.stats["premium"]
    assert premium["flashcards"]["from_model"] >= 4 and premium["questions"]["from_model"] >= 2
    reasons = premium["flashcards"]["reasons"]
    assert reasons.get("fill-in-the-blank") and reasons.get("vague prompt")
    for c in pack.flashcards:
        if c.front in fronts[:2]:
            assert c.evidence in " ".join(ch.text for ch in idx.chunks)                                                                    # the quote is the lecture's own sentence
    assert pack.generation["mode"] == "local" and pack.generation["model"] == "qwen3:8b"


def test_the_model_is_told_why_its_items_were_refused_and_the_rounds_are_bounded(idx):
    fake = FakeOllama(cards=the_model_writes_cards(idx), questions=the_model_writes_questions(idx))
    run_with_model(idx, fake)
    card_prompts = fake.chat_prompts("flashcards")
    assert any("refused for these reasons" in p for p in card_prompts) and any("Already written" in p for p in card_prompts)
    topics = 2
    assert len(card_prompts) <= topics * MAX_ROUNDS and len(fake.chat_prompts("exam questions")) <= topics * MAX_ROUNDS


def test_a_model_that_only_invents_gives_the_student_the_builtin_material(idx):
    def invents(prompt):
        cid = sorted(ids_in(prompt))[0]
        return {"cards": [{"type": "concept", "front": "Which hormone does the pancreas release when blood sugar falls below 3 mmol per litre?", "back": "Glucagon, within five minutes.",
                           "evidence": "Glucagon is released within five minutes of a fall in blood sugar.", "sources": [cid], "difficulty": "easy"}],
                "questions": []}
    fake = FakeOllama(cards=invents, questions=lambda p: {"questions": []})
    pack = run_with_model(idx, fake)
    assert pack.flashcards and not [c for c in pack.flashcards if "glucagon" in c.back.lower()]
    assert pack.stats["premium"]["flashcards"].get("supplemented") and any("built-in method added" in n for n in pack.stats["premium"]["notes"])
    assert not any("built-in method" in w for w in pack.warnings)                                         # engine details are for Advanced, not for the Study screens


def test_a_model_that_stops_answering_never_leaves_the_study_tabs_empty(idx):
    pack = run_with_model(idx, FakeOllama(fail_chat=True))
    assert pack.flashcards and pack.questions and pack.provider.fallback_from == "ollama"
    assert any("could not be used" in n for n in pack.stats["premium"]["notes"]) and not any("could not be used" in w for w in pack.warnings)


def test_an_item_with_a_bad_type_or_no_options_is_repaired_or_refused_not_crashed(idx):
    def sloppy(prompt):
        cid = sorted(ids_in(prompt))[0]
        return {"questions": [{"type": "essay", "question": "Explain everything about metabolism in detail?", "answer": "All of it", "options": "none", "sources": [cid]},
                              "not a dict", {"question": "", "answer": ""}]}
    pack = run_with_model(idx, FakeOllama(cards=the_model_writes_cards(idx), questions=sloppy))
    assert pack.questions                                                                                      # the built-in method made them


def test_clinical_items_are_only_asked_for_when_the_passages_are_clinical(idx):
    fake = FakeOllama(cards=lambda prompt: {"cards": []}, questions=lambda prompt: {"questions": []})
    run_with_model(idx, fake)
    prompts_glycolysis = [p for p in fake.chat_prompts("flashcards") + fake.chat_prompts("exam questions") if "Hexokinase" in p or "hexokinase" in p]
    assert prompts_glycolysis and any("clinical_vignette" in p for p in fake.chat_prompts("exam questions"))             # the synthetic lecture is clinical: vignettes may be asked for

    class Dry:
        """A lecture of pure anatomy: no patient, disease or treatment anywhere."""
    from src.study.automatic import _clinical
    from src.study.index import build_index

    anatomy = build_index([{"start": i * 10.0, "end": i * 10.0 + 9, "text": t} for i, t in enumerate([
        "The nucleus ambiguus lies in the medulla and sends fibers to the pharynx and the larynx.",
        "The solitary nucleus receives taste information from the facial, glossopharyngeal and vagus nerves.",
        "The dorsal motor nucleus of the vagus supplies the parasympathetic fibers of the thorax and abdomen."])])
    assert not _clinical(anatomy.chunks) and _clinical(idx.chunks)


def test_the_prompts_warn_the_model_about_misheard_words_and_template_questions():
    from src.study import premium_prompts as PP

    assert "misheard" in PP.SYSTEM and "never begin more than two items with the same words" in PP.SYSTEM and "own wording" in PP.SYSTEM


def test_a_misheard_word_is_hidden_from_the_model(idx):
    from src.study import premium_prompts as PP

    chunk = next(c for c in idx.chunks if "Pyruvate kinase catalyzes" in c.text)
    shown = PP.render([chunk], ["pyruvate kinase deficiency", "Hexokinase"])
    assert "[unclear]" in shown and "Pyruvate kinase deficiency" not in shown and "[unclear] " in shown
    first = next(c for c in idx.chunks if "hexokinase" in c.text.casefold())
    assert "hexokinase" in PP.render([first], ["hexo"]).casefold()                              # a fragment is never masked inside a longer word


def test_the_engine_hides_the_doubted_words_in_every_prompt(idx):
    idx.unresolved = ["Hexokinase"]
    try:
        fake = FakeOllama(cards=lambda prompt: {"cards": []}, questions=lambda prompt: {"questions": []})
        run_with_model(idx, fake)
        prompts = fake.chat_prompts("flashcards") + fake.chat_prompts("exam questions")
        assert prompts and not [p for p in prompts if "Hexokinase" in p.split("PASSAGES", 1)[1] or "hexokinase" in p.split("PASSAGES", 1)[1].casefold()]
        assert any("[unclear]" in p for p in prompts)
    finally:
        idx.unresolved = []


def test_the_builtin_questions_are_real_sentences_with_the_adverbial_where_it_belongs():
    from src.study.index import build_index
    from tests import study_eval_lectures as L
    from tests.study_support import lecture_index

    pack = AutomaticStudyEngine(choose_engine(Settings(), transport=FakeOllama(models=()))).generate(lecture_index(), lecture_id="x", title="t", options=StudyOptions())
    stems = [q.question for q in pack.questions]
    assert "Which enzyme in the liver acts as a glucose sensor after meals?" in stems
    antibiotics = AutomaticStudyEngine(choose_engine(Settings(), transport=FakeOllama(models=()))).generate(build_index(L.segments("antibiotics")), lecture_id="x", title="t",
                                                                                                         options=StudyOptions())
    assert not [q for q in antibiotics.questions if "binding proteins are" in q.question]       # "Penicillin | binding proteins" was cut in the middle of a name


# --------------------------------------------------------------------------------------------------------------------- the graphics card belongs to a transcription
def test_while_a_transcription_runs_no_model_is_asked_for_or_even_looked_for():
    fake = FakeOllama()
    choice = choose_engine(Settings(), transport=fake, busy=True)
    assert choice.mode == "builtin" and choice.deferred and "transcription is running" in choice.reason
    assert fake.urls == []                                                                           # not even a request to the local server
    assert choose_engine(Settings(), transport=fake, busy=False).mode == "local"


def test_a_pack_made_while_transcribing_says_so_and_a_later_regenerate_uses_the_model(qapp, tmp_path):
    from tests.gui_support import make_services
    from tests.study_support import write_lecture_folder

    services = make_services(tmp_path, opened=[])
    folder = write_lecture_folder(tmp_path / "lecture")
    fake = FakeOllama(cards=lambda prompt: {"cards": []}, questions=lambda prompt: {"questions": []})
    services.study.transport = fake
    services.study.is_busy = lambda: True
    state = services.study.prepare(folder, title="L")
    services.study.generate(state)
    assert state.pack.generation["mode"] == "builtin" and state.pack.generation.get("deferred") and fake.urls == []
    reopened = services.study.open(folder, "L")
    assert reopened.ready and any("transcription was running" in w and "Regenerate" in w for w in reopened.warnings)
    services.study.is_busy = lambda: False
    services.study._choice = None
    again = services.study.prepare(folder, title="L")
    services.study.generate(again)
    assert again.pack.generation["mode"] == "local" and not again.pack.generation.get("deferred")


def test_the_model_is_released_from_memory_when_the_material_is_done(idx):
    seen = []

    class Recording(FakeOllama):
        def __call__(self, url, headers, body, timeout):
            if url.endswith("/api/generate"):
                seen.append(body)
                return {}
            return super().__call__(url, headers, body, timeout)

    run_with_model(idx, Recording(cards=lambda p: {"cards": []}, questions=lambda p: {"questions": []}))
    assert seen == [{"model": "qwen3:8b", "keep_alive": 0}]                                          # only the model this program loaded, nothing else is stopped


def test_a_release_that_fails_never_fails_the_study_material(idx):
    class Stubborn(FakeOllama):
        def __call__(self, url, headers, body, timeout):
            if url.endswith("/api/generate"):
                raise ProviderError("Ollama is gone")
            return super().__call__(url, headers, body, timeout)

    pack = run_with_model(idx, Stubborn(cards=lambda p: {"cards": []}, questions=lambda p: {"questions": []}))
    assert pack.flashcards and pack.questions


def test_starting_a_transcription_abandons_study_material_that_a_model_is_making(qapp, tmp_path):
    from src.gui.main_window import MainWindow
    from tests.gui_support import ScriptedLauncher, make_services, settle

    services = make_services(tmp_path, opened=[])
    window = MainWindow(services, ScriptedLauncher())
    stopped = []
    window.study._stop_running = lambda: stopped.append(True)
    assert services.study.is_busy() is False
    window._running_changed(True)
    assert stopped == [True]
    window._running_changed(False)
    assert stopped == [True]
    settle(window, window.home, window.study)
    window.deleteLater()


def test_a_request_to_the_local_model_has_a_cap_on_what_it_may_generate(idx):
    seen = []

    class Recording(FakeOllama):
        def __call__(self, url, headers, body, timeout):
            if url.endswith("/api/chat"):
                seen.append(body["options"])
            return super().__call__(url, headers, body, timeout)

    run_with_model(idx, Recording(cards=lambda p: {"cards": []}, questions=lambda p: {"questions": []}))
    assert seen and all(0 < o["num_predict"] <= 4096 for o in seen)                                  # a model that repeats itself forever is cut off, not waited for


def test_the_model_is_released_even_when_the_work_is_cancelled(idx):
    from src.study.engine import StudyCancelled

    seen = []

    class Recording(FakeOllama):
        def __call__(self, url, headers, body, timeout):
            if url.endswith("/api/generate"):
                seen.append(body)
                return {}
            return super().__call__(url, headers, body, timeout)

    fake = Recording(cards=the_model_writes_cards(idx), questions=the_model_writes_questions(idx))
    engine = AutomaticStudyEngine(choose_engine(Settings(), transport=fake))
    ticks = {"n": 0}

    def cancelled():
        ticks["n"] += 1
        return ticks["n"] > 4                                                                         # cancelled part-way through the work

    with pytest.raises(StudyCancelled):
        engine.generate(idx, lecture_id="x", title="t", options=StudyOptions(), cancelled=cancelled)
    assert seen == [{"model": "qwen3:8b", "keep_alive": 0}]                                          # a transcription that is starting finds the graphics card free


def test_starting_a_transcription_releases_the_study_model_in_the_background(qapp, tmp_path):
    import time

    from src.gui.main_window import MainWindow
    from tests.gui_support import ScriptedLauncher, make_services, settle

    services = make_services(tmp_path, opened=[])
    seen = []

    def transport(url, headers, body, timeout):
        if url.endswith("/api/tags"):
            return {"models": [{"name": "qwen3:8b"}]}
        seen.append((url, body))
        return {}

    services.study.transport = transport
    assert services.study.engine_choice().mode == "local"
    window = MainWindow(services, ScriptedLauncher())
    window._running_changed(True)
    end = time.monotonic() + 5
    while not seen and time.monotonic() < end:
        time.sleep(0.02)
    assert seen == [(seen[0][0], {"model": "qwen3:8b", "keep_alive": 0})] and seen[0][0].endswith("/api/generate")
    window._running_changed(False)
    settle(window, window.home, window.study)
    window.deleteLater()


def test_with_no_study_model_in_use_a_starting_transcription_releases_nothing(qapp, tmp_path):
    from tests.gui_support import make_services

    services = make_services(tmp_path, opened=[])
    services.study.release_models()                                                                  # nothing was chosen yet, nothing to release, nothing raised
    services.study.engine_choice()                                                                   # the built-in method (no AI server in the tests) has no model to release
    services.study.release_models()


def test_at_exit_the_study_model_is_released_before_the_program_ends(qapp, tmp_path):
    """``release_models(wait=True)``: done when it returns (a background thread would be killed with the program)."""
    from tests.gui_support import make_services

    services = make_services(tmp_path, opened=[])
    seen = []

    def transport(url, headers, body, timeout):
        if url.endswith("/api/tags"):
            return {"models": [{"name": "qwen3:8b"}]}
        seen.append(body)
        return {}

    services.study.transport = transport
    assert services.study.engine_choice().mode == "local"
    services.study.release_models(wait=True)
    assert seen == [{"model": "qwen3:8b", "keep_alive": 0}]                                          # no waiting loop: it was sent before the call returned
