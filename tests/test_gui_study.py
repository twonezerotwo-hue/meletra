"""The study workspace: opening a lecture, the six tabs, source links, flashcards, questions, Study Mode, PDF-only and provider failure."""

import json
from datetime import date, timedelta

import pytest
from PySide6.QtGui import QShortcut
from PySide6.QtWidgets import QApplication, QLabel, QPushButton

from src.gui.main_window import SIMPLE_RESULT, SIMPLE_STUDY, MainWindow
from src.gui.study_view import PAGE_MESSAGE, PAGE_TABS, PAGE_WORKING, TAB_NAMES, StudyWorkspace
from src.study.llm import Transport
from src.study.models import StudyPack
from src.study.providers import ProviderError
from src.study.store import LEARNING, MASTERED, WEAK, StudyStore
from tests.gui_support import ScriptedLauncher, make_services, settle, wait_for
from tests.study_support import write_lecture_folder, write_pdf


@pytest.fixture
def env(qapp, tmp_path):
    services = make_services(tmp_path)
    workspace = StudyWorkspace(services)
    folder = write_lecture_folder(tmp_path / "lecture")
    dialogs = []
    workspace.show_source_dialog = lambda title, text: dialogs.append((title, text))
    yield workspace, services, folder, dialogs, tmp_path
    settle(workspace)
    workspace.deleteLater()


def opened(env, **kw):
    workspace, _services, folder, *_ = env
    workspace.open_lecture(folder, "Glycolysis and collagen", **kw)
    wait_for(lambda: workspace.stack.currentIndex() in (PAGE_TABS, PAGE_MESSAGE), timeout=60, what="the study material")
    return workspace


def labels(widget) -> str:
    return "\n".join(w.text() for w in widget.findChildren(QLabel))


# ------------------------------------------------------------------------------------------------------------------ opening
def test_first_open_makes_the_material_and_a_second_open_loads_it_at_once(env):
    workspace, services, folder, _d, _t = env
    workspace.open_lecture(folder, "Glycolysis and collagen")
    assert workspace.stack.currentIndex() == PAGE_WORKING and workspace.working_title.text() == "Preparing your study material ..."
    wait_for(lambda: workspace.stack.currentIndex() == PAGE_TABS, timeout=60, what="the tabs")
    assert [workspace.tabs.tabText(i) for i in range(workspace.tabs.count())] == TAB_NAMES
    assert TAB_NAMES == ["Overview", "Summary", "Key Concepts", "Flashcards", "Questions", "Study Mode"]
    study = folder / "study"
    assert all((study / n).is_file() for n in ("study_pack.json", "study_index.json"))
    workspace.open_lecture(folder, "Glycolysis and collagen", tab="flashcards")
    assert workspace.stack.currentIndex() == PAGE_TABS and workspace.tabs.currentIndex() == 3        # no waiting, no regeneration


def test_overview_explains_the_sources_and_what_was_made(env):
    ws = opened(env)
    text = labels(ws.overview)
    assert "Made from the transcript" in text and "Source-bound" in text and "MELETRA Automatic Study Engine" in text and "Quality check:" in text
    assert "summary points" in ws.overview.counts.text() and "flashcards" in ws.overview.counts.text()
    assert "Topics:" in ws.overview.topics.text()
    assert not ws.overview.warnings.isVisibleTo(ws.overview)
    assert ws.provider_label.text() == "MELETRA Automatic Study Engine"                      # the model and the method are Advanced diagnostics, not part of the Study screens
    assert ws.stat_values["flashcards"].text() == str(len(ws.state.pack.flashcards))
    assert ws.stat_values["questions"].text() == str(len(ws.state.pack.questions))
    assert "Transcript" in ws.lecture_meta.text()


def test_a_lecture_with_nothing_to_study_says_so(env):
    workspace, services, _folder, _d, tmp = env
    empty = tmp / "nothing"
    empty.mkdir()
    workspace.open_lecture(empty, "Empty")
    assert workspace.stack.currentIndex() == PAGE_MESSAGE and workspace.message_title.text() == "Nothing to study yet"
    assert workspace.message_pdf_button.isVisibleTo(workspace) and not workspace.retry_button.isVisibleTo(workspace)


def test_cancelled_generation_returns_to_a_calm_message(env):
    workspace, *_ = env
    workspace.state = workspace.services.study.open(env[2], "x")
    workspace._work_failed("StudyCancelled")
    assert workspace.message_title.text() == "Cancelled" and workspace.stack.currentIndex() == PAGE_MESSAGE
    workspace._work_failed("disk full")
    assert workspace.message_title.text() == "The study material could not be made" and "disk full" in workspace.message_text.text()


# ------------------------------------------------------------------------------------------------------------------ summaries
def test_summary_tab_lists_kinds_and_links_every_point_to_its_source(env):
    ws = opened(env)
    s = ws.summary
    names = [s.kinds.item(i).text() for i in range(s.kinds.count())]
    assert any(n.startswith("Quick Summary") for n in names) and any(n.startswith("High-Yield") for n in names)
    assert not any("Key Concepts" in n or "Important Definitions" in n for n in names)              # those have their own tab
    s.select_kind("high_yield")
    html = s.browser.toHtml()
    assert "Glyc" in html or "Collagen" in html                                                   # grouped by topic
    assert "[Transcript" in s.browser.toPlainText()
    ws.show_source("t0000")
    title, text = env[3][-1]
    assert title.startswith("Transcript 00:00-") and "Glycolysis is the metabolic pathway" in text
    ws.show_source("no-such-chunk")
    assert len(env[3]) == 1


def test_the_summary_list_leads_with_the_learning_objectives_and_each_links_to_its_passage(env):
    ws = opened(env)
    s = ws.summary
    assert s.kinds.item(0).text().startswith("Learning Objectives") and s.current_kind == "quick"              # first in the list; the tab still opens on the quick summary
    s.select_kind("objectives")
    text = s.browser.toPlainText()
    assert s.kind_title.text() == "Learning Objectives" and "After studying this lecture you should be able to" in s.kind_meta.text()
    assert len(s.objectives) >= 2 and all(o.text in text for o in s.objectives)
    assert "[Transcript" in text and "Practise with" in text                                                   # the sources and the material that practises each
    assert all(o.sources for o in s.objectives)
    s.select_kind("quick")
    assert s.kind_title.text() == "Quick Summary"


def test_key_concepts_tab_has_concepts_and_definitions(env):
    ws = opened(env)
    text = ws.concepts.browser.toPlainText()
    assert "Key concepts" in text and "Important definitions" in text and "Glycolysis" in text


def test_clicking_a_source_link_in_the_text_opens_the_passage(env):
    ws = opened(env)
    from PySide6.QtCore import QUrl

    ws.summary.browser._clicked(QUrl("chunk:t0001"))
    assert env[3] and env[3][-1][0].startswith("Transcript")


# ------------------------------------------------------------------------------------------------------------------ flashcards
def test_flashcards_hide_the_answer_until_asked_and_remember_ratings(env):
    ws = opened(env)
    f = ws.flashcards
    assert f.stack.currentIndex() == 1 and f.front.text() and f.back.isHidden() and f.show_button.isVisibleTo(f)
    assert not f.know_button.isVisibleTo(f)
    first = f.session.current()
    f.rate("know")                                                    # ignored: the answer was not looked at yet
    assert f.session.current() is first
    f.reveal()
    assert not f.back.isHidden() and f.back.text() == first.back and f.know_button.isVisibleTo(f)
    f.rate("dont_know")
    progress = StudyStore(env[2]).progress()
    assert progress.card_state(first.id) == WEAK and "Card 2 of" in f.counter.text()


def expected_question(pack, card):
    """The question that belongs to a card: made from the same fact, else from the same source passage - else none (a question that only shares the topic is not about this card)."""
    same_fact = next((q for q in pack.questions if card.fact_id and q.fact_id == card.fact_id), None)
    if same_fact:
        return same_fact
    wanted = {ref.chunk_id for ref in card.sources}
    return next((q for q in pack.questions if wanted & {ref.chunk_id for ref in q.sources}), None) if wanted else None


def assert_rail_matches(f):
    related = expected_question(f.pack, f.session.current())
    assert (related.question in f.rail_question.text()) if related else ("No question about this card yet" in f.rail_question.text()), f.rail_question.text()


def test_flashcard_workspace_has_real_right_rail_and_previous_next(env):
    ws = opened(env)
    f = ws.flashcards
    assert "Glycolysis" in f.rail_concepts.text()
    assert_rail_matches(f)                                            # the question of THIS card, never an unrelated one
    assert "mastered" in f.rail_progress.text() and f.previous_button.text() == "Previous" and f.next_button.text() == "Next"
    first = f.session.current()
    f.next_button.click()
    assert f.session.position == 1
    assert_rail_matches(f)                                            # ... and it follows the card when the card changes
    f.previous_button.click()
    assert f.session.position == 0 and f.session.current() is first
    assert_rail_matches(f)
    shown = set()
    for _ in range(6):                                                # card after card: the panel never keeps the question of an earlier card
        f.next_button.click()
        assert_rail_matches(f)
        shown.add(f.rail_question.text())
    assert len(shown) >= 2, shown


def test_flashcard_filters_and_finishing_a_deck(env):
    ws = opened(env)
    f = ws.flashcards
    f.type_combo.setCurrentIndex(f.type_combo.findData("definition"))
    f.start()
    assert all(c.type == "definition" for c in f.session.deck)
    total = len(f.session.deck)
    for number in range(total):
        f.reveal()
        f.rate("know" if number % 2 == 0 else "unsure")
    assert f.stack.currentIndex() == 2 and "Deck finished" in f.done_label.text()
    f.focus_combo.setCurrentIndex(f.focus_combo.findData("weak"))
    f.start()
    assert f.stack.currentIndex() == 0 and "No cards match" in f.empty_page.text()


def test_flashcard_keys_and_weak_practice(env):
    ws = opened(env)
    f = ws.flashcards
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    QTest.keyClick(f, Qt.Key_Space)
    assert not f.back.isHidden()
    QTest.keyClick(f, Qt.Key_3)
    assert f.session.position == 1
    f.practice_weak()
    assert len(f.session.deck) == 1 and f.session.current().id == f.session.deck[0].id


def test_flashcards_show_when_each_answer_brings_the_card_back_and_the_study_plan(env):
    ws = opened(env)
    f = ws.flashcards
    assert f.schedule_hint.text() == "" and "due today" in f.rail_progress.text()
    f.reveal()
    assert "Know: tomorrow" in f.schedule_hint.text() and "Don't know: today" in f.schedule_hint.text()      # a new card: 1 day when known, today when not
    f.rate("know")
    assert "1 review today · 1 in all" in f.rail_progress.text()
    assert "1 know" in f.counter.text()                                                                     # the session tally sits beside the card number


def test_flashcards_mark_difficult_and_the_mark_is_kept_and_filterable(env):
    ws = opened(env)
    f = ws.flashcards
    card = f.session.current()
    assert f.difficult_button.text() == "Mark difficult" and not f.difficult_button.isChecked()
    f.difficult_button.click()
    assert f.difficult_button.isChecked() and f.difficult_button.text() == "Marked difficult"
    assert StudyStore(env[2]).progress().is_difficult(card.id)                                               # written at once
    f.next_button.click()
    f.previous_button.click()
    assert f.difficult_button.isChecked()                                                                    # still marked when the card comes back
    f.focus_combo.setCurrentIndex(f.focus_combo.findData("difficult"))
    f.start()
    assert [c.id for c in f.session.deck] == [card.id]
    f.difficult_button.click()                                                                               # un-mark
    assert not StudyStore(env[2]).progress().is_difficult(card.id)


def test_flashcard_arrow_keys_and_d_work_on_the_card_but_leave_the_list_boxes_alone(env):
    ws = opened(env)
    f = ws.flashcards
    shortcuts = {s.key().toString() for s in f.findChildren(QShortcut)}
    assert {"Left", "Right", "D"} <= shortcuts
    first = f.session.current()
    f._shortcut(f.next_card)
    assert f.session.position == 1
    f._shortcut(f.previous_card)
    assert f.session.current() is first
    f._shortcut(f.toggle_difficult)
    assert f.difficult_button.isChecked()
    f.topic_combo.setFocus()
    f.show()
    f.topic_combo.activateWindow()
    if QApplication.focusWidget() is f.topic_combo:                                                          # (an offscreen window does not always take the focus)
        f._shortcut(f.next_card)
        assert f.session.position == 0                                                                       # the list box keeps its keys
    f.rate("know")                                                                                           # ignored: not revealed
    assert f.session.position == 0


def test_the_cards_the_schedule_brought_back_are_what_the_deck_starts_with_after_a_restart(env):
    ws = opened(env)
    workspace, _services, folder, *_ = env
    f = ws.flashcards
    studied = []
    for _ in range(3):
        f.reveal()
        studied.append(f.session.current().id)
        f.rate("know")
    store = StudyStore(folder, "en")
    raw = json.loads(store.progress_path.read_text(encoding="utf-8"))
    assert all(raw["cards"][cid]["sr"]["interval"] == 1 and raw["cards"][cid]["reviews"] == 1 for cid in studied)
    for cid in studied[:2]:                                                                                  # two days pass: these two are due
        raw["cards"][cid]["sr"]["due"] = (date.today() - timedelta(days=2)).isoformat()
    store.progress_path.write_text(json.dumps(raw), encoding="utf-8")
    workspace.open_lecture(folder, "Glycolysis and collagen")                                                # closing and reopening the program
    wait_for(lambda: workspace.stack.currentIndex() == PAGE_TABS, timeout=60, what="the tabs")
    f = workspace.flashcards
    assert f.focus_combo.currentData() == "due" and {c.id for c in f.session.deck} == set(studied[:2])
    assert "2 due today" in f.rail_progress.text()
    for _ in range(2):
        f.reveal()
        f.rate("know")
    assert f.stack.currentIndex() == 2 and "Nothing is due today" in f.done_label.text() and "next review is tomorrow" in f.done_label.text()   # the third card is due tomorrow


@pytest.mark.parametrize("only", [None, {"flashcards"}], ids=["everything", "only the flashcards"])
def test_regenerating_the_study_material_keeps_the_progress_of_the_cards(env, only):
    ws = opened(env)
    workspace, _services, folder, *_ = env
    f = ws.flashcards
    f.reveal()
    first = f.session.current()
    f.rate("know")
    f.reveal()
    second = f.session.current()
    f.rate("dont_know")
    third = f.session.current()
    f.toggle_difficult()
    store = StudyStore(folder, "en")
    pack_data = json.loads(store.pack_path.read_text(encoding="utf-8"))
    progress_data = json.loads(store.progress_path.read_text(encoding="utf-8"))
    for item in pack_data["flashcards"]:                                                                     # the material of an earlier engine: the same questions under other ids
        old = "old-" + item["id"]
        if item["id"] in progress_data["cards"]:
            progress_data["cards"][old] = progress_data["cards"].pop(item["id"])
        item["id"] = old
    store.pack_path.write_text(json.dumps(pack_data), encoding="utf-8")
    store.progress_path.write_text(json.dumps(progress_data), encoding="utf-8")
    workspace.open_lecture(folder, "Glycolysis and collagen")
    wait_for(lambda: workspace.stack.currentIndex() == PAGE_TABS, timeout=60, what="the old material")
    assert workspace.state.pack.flashcards[0].id.startswith("old-")
    workspace.regenerate(only=only)
    wait_for(lambda: workspace.stack.currentIndex() == PAGE_TABS and not workspace.state.pack.flashcards[0].id.startswith("old-"), timeout=90, what="the new material")
    after = StudyStore(folder, "en").progress()
    assert after.card_state(first.id) == LEARNING and after.card_state(second.id) == WEAK and after.is_difficult(third.id)
    assert after.cards[first.id]["carried_from"] == "old-" + first.id and after.cards[first.id]["log"]
    assert {"old-" + first.id, "old-" + second.id, "old-" + third.id} <= set(after.cards)                    # nothing was erased
    assert workspace.state.pack.generation["progress_carried"]["cards"] >= 3


# ------------------------------------------------------------------------------------------------------------------ questions
def test_a_question_does_not_show_its_answer_until_it_is_checked(env):
    ws = opened(env)
    q = ws.questions
    card = next(c for c in q.cards if c.question.options and c.question.type == "mcq")
    before = labels(card)
    assert "Correct answer" not in before and card.question.explanation not in before and not card.done
    assert not card.check_button.isHidden()
    card.check()                                                     # nothing chosen: nothing happens
    assert not card.done
    card.buttons[card.question.correct_index].setChecked(True)
    card.check()
    after = labels(card)
    assert card.done and "Correct!" in after and f"Correct answer: {card.question.answer}" in after
    assert StudyStore(env[2]).progress().questions[card.question.id]["results"] == ["correct"]
    assert not any(b.isEnabled() for b in card.buttons)


def test_wrong_and_typed_answers_are_judged(env):
    ws = opened(env)
    q = ws.questions
    mcq = next(c for c in q.cards if c.question.type == "mcq")
    wrong = next(r for r in mcq.buttons if r.text() != mcq.question.answer)
    wrong.setChecked(True)
    mcq.check()
    assert "Not quite." in labels(mcq)
    short = next(c for c in q.cards if c.question.type == "short_answer")
    short.text_edit.setPlainText(short.question.answer)
    short.check()
    assert "Correct!" in labels(short)


def test_question_filters(env):
    ws = opened(env)
    q = ws.questions
    total = len(q.cards)
    q.type_combo.setCurrentIndex(q.type_combo.findData("true_false"))
    assert 0 < len(q.cards) < total and all(c.question.type == "true_false" for c in q.cards)
    assert q.count_label.text().endswith("questions") or q.count_label.text().endswith("question")


# ------------------------------------------------------------------------------------------------------------------ study mode
def test_study_mode_session_from_start_to_results(env):
    ws = opened(env)
    m = ws.study_mode
    assert m.stack.currentIndex() == 0 and m.start_button.isEnabled() and "questions available" in m.setup_note.text()
    m.count_combo.setCurrentIndex(m.count_combo.findData(10))
    m.start()
    n = len(ws.state.pack.questions)                                 # the premium gate decides how many questions there are; the session takes them all (up to 10)
    assert 5 <= n <= 10 and m.stack.currentIndex() == 1 and m.position_label.text().startswith(f"Question 1 of {n}")
    seen_answer_before = "Correct answer" in labels(m.feedback_box) or m.feedback_box.isVisibleTo(m)
    assert not seen_answer_before                                    # the answer is not on screen yet
    for _ in range(n):
        view = m.session.current()
        if view.options:
            m.choose(0)
        else:
            m.text_edit.setPlainText("some words about glycolysis")
            m.submit()
        assert m.feedback_box.isVisibleTo(m) and "Correct answer:" in labels(m.feedback_box)
        m.next()
    assert m.stack.currentIndex() == 2
    assert m.score_label.text().startswith("Score:") and "Correct:" in m.counts_label.text()
    assert "Weak topics:" in m.weak_label.text() and "Suggested review" in m.review_label.text()
    progress = StudyStore(env[2]).progress()
    assert len(progress.sessions) == 1 and progress.sessions[0]["total"] == n


def test_study_mode_resumes_an_unfinished_session_after_the_program_is_reopened(env):
    ws = opened(env)
    workspace, _services, folder, *_ = env
    m = ws.study_mode
    assert not m.resume_button.isVisibleTo(m) and not m.resume_note.isVisibleTo(m)                         # nothing to resume yet
    m.count_combo.setCurrentIndex(m.count_combo.findData(10))
    m.start()
    n = len(ws.state.pack.questions)
    first = m.session.current().id
    m.choose(0) if m.session.current().options else m.skip()
    assert StudyStore(folder).progress().active_session["outcomes"].keys() == {first}                      # written as soon as the answer is given
    workspace.open_lecture(folder, "Glycolysis and collagen")                                              # closing and reopening the program
    wait_for(lambda: workspace.stack.currentIndex() == PAGE_TABS, timeout=60, what="the tabs")
    m = workspace.study_mode
    assert m.resume_button.isVisibleTo(m) and f"1 of {min(n, 10)} questions answered" in m.resume_note.text()
    m.resume()
    assert m.stack.currentIndex() == 1 and m.position_label.text().startswith("Question 2 of") and m.session.outcomes.keys() == {first}
    m.end()
    assert m.stack.currentIndex() == 2 and "Score:" in m.score_label.text()
    progress = StudyStore(folder).progress()
    assert progress.active_session is None and len(progress.sessions) == 1
    m._back_to_setup()
    assert not m.resume_button.isVisibleTo(m)


def test_skip_end_and_override_in_study_mode(env):
    ws = opened(env)
    m = ws.study_mode
    m.count_combo.setCurrentIndex(m.count_combo.findData(10))
    m.start()
    n = len(ws.state.pack.questions)
    m.skip()
    assert "Skipped." in labels(m.feedback_box) and m.next_button.isVisibleTo(m) and not m.skip_button.isVisibleTo(m)
    m.choose(0)                                                      # a second answer to the same question is ignored
    m.next()
    free = None
    for _ in range(n - 1):
        if m.session.current() is None:
            break
        if m.session.current().free_text:
            free = m.session.current()
            break
        m.choose(0)
        m.next()
    if free is not None:
        m.text_edit.setPlainText("nothing relevant")
        m.submit()
        assert m.agree_button.isVisibleTo(m)
        m.override_right()
        assert "Correct!" in labels(m.feedback_box)
    m.end()
    assert m.stack.currentIndex() == 2
    r = m.session.results()
    assert r.skipped >= 1 and r.total == n and m.review_button.isEnabled() == bool(r.weak_question_ids)


def test_weak_questions_can_be_practised_again(env):
    ws = opened(env)
    m = ws.study_mode
    m.count_combo.setCurrentIndex(m.count_combo.findData(10))
    m.start()
    for _ in range(len(ws.state.pack.questions)):
        if m.session.current().options:
            wrong = [i for i, o in enumerate(m.session.current().options) if o != m.session.questions[m.session.position].answer][0]
            m.choose(wrong)
        else:
            m.skip()
        m.next()
    assert m.review_button.isEnabled()
    m.start_weak()
    assert m.stack.currentIndex() == 1 and m.session.questions
    m.end()


# ------------------------------------------------------------------------------------------------------------------ PDF + sources
def test_adding_a_pdf_links_the_study_material_to_the_slides(env):
    ws = opened(env)
    pdf = write_pdf(env[4] / "slides.pdf", [["Glycolysis", "Hexokinase phosphorylates glucose", "Pyruvate kinase produces pyruvate and ATP"],
                                            ["Collagen synthesis", "Proline hydroxylase needs vitamin C", "Lysyl oxidase needs copper"]])
    ws.choose_pdf = lambda: str(pdf)
    ws.add_pdf()
    assert ws.stack.currentIndex() == PAGE_WORKING
    wait_for(lambda: ws.stack.currentIndex() == PAGE_TABS, timeout=60, what="the tabs after adding the PDF")
    assert "slides and the transcript" in ws.overview.made_from.text() and "slides.pdf" in ws.overview.made_from.text()
    assert ws.state.pack.source_mode == "pdf+transcript"
    assert any(s.kind == "pdf" for v in ws.state.pack.summaries.values() for i in v for s in i.sources)
    ws.summary.select_kind("detailed")
    assert "[Slide" in ws.summary.browser.toPlainText()
    ws.show_source("p001")
    assert env[3][-1][0].startswith("Slide 1")


def test_study_from_a_pdf_without_any_recording(env):
    workspace, services, _folder, _d, tmp = env
    pdf = write_pdf(tmp / "only_slides.pdf", [["Glycolysis", "Hexokinase phosphorylates glucose to glucose-6-phosphate",
                                               "Glycolysis is the pathway that converts glucose into pyruvate",
                                               "Pyruvate kinase deficiency causes hemolytic anemia"],
                                              ["Collagen", "Collagen is the most abundant protein of the extracellular matrix",
                                               "Vitamin C deficiency causes scurvy because collagen is unstable"]])
    folder = tmp / "pdf_only"
    folder.mkdir()
    workspace.open_lecture(folder, "Only slides", pdf=pdf)
    wait_for(lambda: workspace.stack.currentIndex() in (PAGE_TABS, PAGE_MESSAGE), timeout=60, what="the pdf-only study")
    assert workspace.stack.currentIndex() == PAGE_TABS
    assert workspace.state.pack.source_mode == "pdf" and "slides only" in workspace.overview.made_from.text()
    assert (folder / "study" / "pdf_pages.json").is_file()
    workspace.open_lecture(folder, "Only slides")                      # later: the saved slide text is enough
    assert workspace.stack.currentIndex() == PAGE_TABS and workspace.state.index.mode == "pdf"


def test_pdf_without_text_is_explained(env):
    workspace, _s, _f, _d, tmp = env
    junk = tmp / "scan.pdf"
    junk.write_bytes(b"not really a pdf")
    folder = tmp / "pdf_only2"
    folder.mkdir()
    workspace.open_lecture(folder, "Scan", pdf=junk)
    wait_for(lambda: workspace.stack.currentIndex() == PAGE_MESSAGE, timeout=60, what="the message")
    assert "could not be read" in workspace.message_text.text()


# ------------------------------------------------------------------------------------------------------------------ providers
def test_a_local_model_that_stops_answering_still_gives_the_student_everything(env):
    workspace, services, folder, _d, _t = env

    def stops_answering(url, headers, body, timeout):
        if url.endswith("/api/tags"):
            return {"models": [{"name": "qwen3:8b"}]}
        raise ProviderError("Cannot reach localhost:11434. Is the server running?")

    services.study.transport = stops_answering
    ws = opened(env)
    assert ws.state.pack.provider.name == "builtin" and ws.state.pack.provider.fallback_from == "ollama"
    assert not ws.overview.warnings.isVisibleTo(ws.overview) or "Cannot reach" not in ws.overview.warnings.text()          # which model failed is not for the Study screens
    assert ws.provider_label.text() == "MELETRA Automatic Study Engine"
    assert ws.state.pack.flashcards and ws.state.pack.questions and ws.state.pack.summaries
    lines = "\n".join(services.study.diagnostics(folder))
    assert "Fallback: ollama could not be used" in lines and "Cannot reach" in lines                                         # ... they are in the Advanced diagnostics
    services.update_setting(show_advanced=True)
    ws2 = opened(env)
    assert "built-in offline method (the local AI was not available this time)" in ws2.provider_label.text()


def test_a_working_local_model_is_used_and_labelled(env):
    workspace, services, folder, _d, _t = env
    services.update_setting(study_model="m", show_advanced=True)

    def fake(url, headers, body, timeout):
        if url.endswith("/api/tags"):
            return {"models": [{"name": "m:latest"}]}
        prompt = body["messages"][1]["content"]
        if "flashcards" in prompt:
            reply = {"cards": [{"type": "definition", "front": "What is glycolysis?", "back": "The pathway that converts glucose into pyruvate.",
                                "sources": ["t0000"]}]}
        elif "exam questions" in prompt:
            reply = {"questions": [{"type": "true_false", "question": "Glycolysis happens in the cytoplasm.", "options": ["True", "False"],
                                    "answer": "True", "explanation": "Glycolysis happens in the cytoplasm of the cell.", "sources": ["t0000"]}]}
        else:
            reply = {"items": [{"text": "Glycolysis converts glucose into pyruvate.", "sources": ["t0000"]}]}
        return {"message": {"content": json.dumps(reply)}}

    services.study.transport = fake
    ws = opened(env)
    pack = ws.state.pack
    assert pack.provider.name == "ollama" and not pack.provider.fallback_from and pack.generation["mode"] == "local"
    assert any(c.front == "What is glycolysis?" for c in pack.flashcards)
    assert "local AI (m:latest)" in ws.provider_label.text()


# ------------------------------------------------------------------------------------------------------------------ inside the window
def test_study_lecture_from_the_result_screen_and_back(qapp, tmp_path):
    services = make_services(tmp_path)
    window = MainWindow(services, ScriptedLauncher())
    folder = write_lecture_folder(tmp_path / "lecture")
    window.study.show_source_dialog = lambda title, text: None
    window.show_finished(folder, "Glycolysis.mp4")
    result = window.simple_result
    assert "Made when you press Create Study Material" in result.tiles["summary"].text() and result.study_button.isEnabled()
    result.tiles["summary"].click()
    assert window.simple_stack.currentIndex() == SIMPLE_STUDY and window.study.stack.currentIndex() == PAGE_WORKING
    wait_for(lambda: window.study.stack.currentIndex() == PAGE_TABS, timeout=60, what="study tabs")
    assert window.study.tabs.currentIndex() == 1                         # the Summary tab, as the tile promised
    window.study.back_button.click()
    assert window.simple_stack.currentIndex() == SIMPLE_RESULT
    assert "Ready -" in result.tiles["summary"].text() and "Ready -" in result.tiles["flashcards"].text()
    assert "Ready -" in result.tiles["questions"].text()
    settle(window, window.home, window.study)
    window.deleteLater()


def test_study_from_a_pdf_in_the_window(qapp, tmp_path):
    services = make_services(tmp_path)
    window = MainWindow(services, ScriptedLauncher())
    pdf = write_pdf(tmp_path / "Slides.pdf", [["Glycolysis", "Glycolysis is the pathway that converts glucose into pyruvate",
                                               "Hexokinase phosphorylates glucose to glucose-6-phosphate"],
                                              ["Collagen", "Collagen is the most abundant protein of the extracellular matrix"]])
    window.choose_pdf_file = lambda: str(pdf)
    window.study_pdf_action.trigger()
    assert window.simple_stack.currentIndex() == SIMPLE_STUDY
    wait_for(lambda: window.study.stack.currentIndex() in (PAGE_TABS, PAGE_MESSAGE), timeout=60, what="the pdf-only study")
    assert window.study.stack.currentIndex() == PAGE_TABS
    assert (tmp_path / "Medical Transcription" / "Slides" / "study" / "study_pack.json").is_file()
    window.choose_pdf_file = lambda: ""
    window.study_from_pdf()                                               # cancelled dialog: stays where it is
    assert window.simple_stack.currentIndex() == SIMPLE_STUDY
    settle(window, window.home, window.study)
    window.deleteLater()


def test_closing_the_window_while_the_study_material_is_made_does_not_crash(qapp, tmp_path):
    services = make_services(tmp_path)
    window = MainWindow(services, ScriptedLauncher())
    folder = write_lecture_folder(tmp_path / "lecture")
    window.show_finished(folder, "Lecture")
    window.open_study("overview")
    window.close()                                                        # stop_threads() cancels and waits for the study thread
    assert not any(t.isRunning() for t in window.study._threads)
    window.deleteLater()


# ------------------------------------------------------------------------------------------------------------------ ask + export
def test_ask_the_lecture_answers_from_the_sources_or_says_it_cannot(env):
    ws = opened(env)
    o = ws.overview
    o.ask_edit.setText("What does hexokinase do?")
    o.ask_button.click()
    wait_for(lambda: o.ask_button.isEnabled(), what="the answer")
    assert o.answer.isVisibleTo(o) and "hexokinase" in o.answer.toPlainText().casefold() and "[Transcript" in o.answer.toPlainText()
    o.ask_edit.setText("What is the dose of lisinopril in renal failure?")
    o.ask_button.click()
    wait_for(lambda: o.ask_button.isEnabled(), what="the answer")
    assert "not covered in your lecture sources" in o.answer.toPlainText()
    o.ask_edit.setText("   ")
    o.ask_button.click()                                                    # an empty question does nothing
    assert o.ask_button.isEnabled()


def test_export_writes_the_study_guide_and_the_anki_file(env):
    ws = opened(env)
    opened_paths = []
    ws.services.open_path = opened_paths.append
    ws.overview.export_button.click()
    folder = env[2] / "study"
    assert (folder / "study_guide.md").is_file() and (folder / "flashcards_anki.txt").is_file()
    assert "study_guide.md" in ws.overview.export_note.text() and opened_paths == [folder]


def test_opening_another_lecture_abandons_the_one_still_being_made(env):
    workspace, services, folder, _d, tmp = env
    other = write_lecture_folder(tmp / "other")
    workspace.open_lecture(folder, "First")                                # starts making the first lecture ...
    workspace.open_lecture(other, "Second")                                # ... and is not confused when the second is opened at once
    wait_for(lambda: workspace.stack.currentIndex() in (PAGE_TABS, PAGE_MESSAGE), timeout=60, what="the second lecture")
    assert workspace.state.folder == other and workspace.title.text() == "Second" and workspace.stack.currentIndex() == PAGE_TABS
    assert (other / "study" / "study_pack.json").is_file()


def test_slides_attached_before_transcribing_are_used_by_the_study_material(qapp, tmp_path):
    services = make_services(tmp_path)
    window = MainWindow(services, ScriptedLauncher())
    folder = write_lecture_folder(tmp_path / "lecture")
    pdf = write_pdf(tmp_path / "slides.pdf", [["Glycolysis", "Hexokinase phosphorylates glucose", "Pyruvate kinase produces pyruvate and ATP"],
                                              ["Collagen synthesis", "Proline hydroxylase needs vitamin C", "Lysyl oxidase needs copper"]])
    window.study.show_source_dialog = lambda title, text: None
    window.current_output, window.source_name, window.current_pdf = folder, "lecture.mp4", pdf       # what a finished job leaves behind
    window.simple_result.show_results(folder, "lecture.mp4")
    window.open_study("overview")
    wait_for(lambda: window.study.stack.currentIndex() in (PAGE_TABS, PAGE_MESSAGE), timeout=60, what="the study material")
    assert window.study.state.pack.source_mode == "pdf+transcript" and services.study.pdf_name(folder) == "slides.pdf"
    again = window.study.state.pack.created
    window.study.back_button.click()
    window.open_study("overview")                                                  # the slides are saved now: opens at once, same pack
    assert window.study.stack.currentIndex() == PAGE_TABS and window.study.state.pack.created == again
    settle(window, window.home, window.study)
    window.deleteLater()


def test_start_job_remembers_the_attached_slides(qapp, tmp_path):
    from src.services.transcription_service import JobRequest

    services = make_services(tmp_path)
    window = MainWindow(services, ScriptedLauncher())
    pdf = write_pdf(tmp_path / "slides.pdf", [["Glycolysis"]])
    source = tmp_path / "a.wav"
    from tests.conftest import write_silent_wav

    write_silent_wav(source, seconds=5)
    window.start_job(JobRequest(source=str(source), output_dir=str(tmp_path / "out"), model_key="whisper-large-v3", language="en",
                                pdf_path=str(pdf)))
    assert window.current_pdf == pdf
    window.controller.kill()
    settle(window, window.home, window.study)
    window.deleteLater()


def test_overview_says_how_many_generated_items_the_source_check_left_out(env):
    from src.study.models import SourceRef, SummaryItem
    from src.study.providers import AIProvider
    from tests.test_study_engine import ScriptedProvider, ref

    workspace, services, folder, _d, _t = env
    ws = opened(env)
    assert "left out" not in ws.overview.made_from.text()                              # the built-in method quotes: nothing to drop
    pack = ws.state.pack
    pack.stats["grounding"] = {"kept": 10, "dropped": 3, "reasons": {"unsupported": 3}}
    ws.overview.show_pack(pack, ws.state.progress, "Test", "")
    assert "3 generated items left out because the lecture does not support them" in ws.overview.made_from.text()
    pack.stats["grounding"]["dropped"] = 1
    ws.overview.show_pack(pack, ws.state.progress, "Test", "")
    assert "1 generated item left out because the lecture does not support it" in ws.overview.made_from.text()


# ------------------------------------------------------------------------------------------------------------------ material of an earlier version
def test_material_of_an_earlier_version_is_shown_at_once_with_a_visible_note_and_regenerate_replaces_it(env):
    workspace, services, folder, _d, _t = env
    opened(env)                                                                                          # makes the material (this version's)
    store = StudyStore(folder, "en")
    data = json.loads(store.pack_path.read_text(encoding="utf-8"))
    data.pop("generation")                                                                               # what 4.1.1 left in the lecture's folder: no engine, no version
    for item in data["flashcards"] + data["questions"]:
        item.pop("generation", None)
        item.pop("evidence", None)
    store.pack_path.write_text(json.dumps(data), encoding="utf-8")
    before = [c["id"] for c in data["flashcards"]]
    workspace.open_lecture(folder, "Glycolysis and collagen")
    assert workspace.stack.currentIndex() == PAGE_TABS                                                   # restored instantly: nothing is generated, nothing is replaced
    assert workspace.overview.warnings.isVisibleTo(workspace.overview)
    note = workspace.overview.warnings.text()
    assert "earlier version of MELETRA" in note and "Regenerate" in note                                 # the student is told, and told what to press
    assert [c.id for c in workspace.state.pack.flashcards] == before
    workspace.regenerate()                                                                               # Regenerate: this version's engine makes it again, from the saved transcript
    wait_for(lambda: workspace.stack.currentIndex() == PAGE_TABS and workspace.state.pack.generation.get("engine") == "MELETRA Automatic Study Engine", timeout=90, what="the new material")
    assert workspace.state.pack.generation["version"] and all(c.evidence for c in workspace.state.pack.flashcards)
    assert not any("earlier version of MELETRA" in w for w in workspace.state.pack.warnings)
