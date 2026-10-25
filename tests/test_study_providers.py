"""Summaries, flashcards and questions of the built-in provider; the language-model providers with a fake transport and a fake
local HTTP server (no real AI, no internet)."""

import copy
import json
import select
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from src.study import grounding as G
from src.study import llm
from src.study.extractive import ExtractiveProvider, tidy
from src.study.models import (CARD_TYPES, QUESTION_TYPES, SUMMARY_KINDS, Chunk, Flashcard, Question, SourceRef, SummaryItem)
from src.study.prompts import parse_json
from src.study.providers import (NOT_IN_SOURCES, AIProvider, ProviderConfig, ProviderError, StudyContext, create_provider)
from study_support import lecture_index


@pytest.fixture(scope="module")
def idx():
    return lecture_index(with_slides=True)


@pytest.fixture()
def ctx(idx):
    return StudyContext(index=idx)


@pytest.fixture(scope="module")
def provider():
    return ExtractiveProvider()


def _all_text(idx):
    return " ".join(c.text for c in idx.chunks).casefold()


# --------------------------------------------------------------------------------------------------------------------- summaries
@pytest.mark.parametrize("kind", SUMMARY_KINDS)
def test_every_summary_kind_is_made_from_the_lecture_with_sources(provider, idx, ctx, kind):
    items = provider.summarize(kind, idx.chunks, topic="", limit=8, ctx=ctx)
    assert items, kind
    for item in items:
        assert item.sources and idx.get(item.sources[0].chunk_id)
        words = [w for w in item.text.casefold().replace(".", "").split() if len(w) > 4]
        assert sum(w in _all_text(idx) for w in words) >= 0.9 * len(words)      # nothing that the lecture did not say
    assert len(items) <= 8


def test_summary_keeps_lecture_order_and_terminology(provider, idx, ctx):
    items = provider.summarize("detailed", idx.chunks, topic="", limit=12, ctx=ctx)
    positions = [idx.get(i.sources[0].chunk_id).start for i in items if idx.get(i.sources[0].chunk_id).start is not None]
    assert positions == sorted(positions)
    text = " ".join(i.text for i in items)
    assert "hexokinase" in text.casefold() or "phosphofructokinase" in text.casefold()      # terms are copied, not paraphrased


def test_definitions_and_key_concepts_carry_the_term(provider, idx, ctx):
    definitions = provider.summarize("definitions", idx.chunks, topic="", limit=10, ctx=ctx)
    assert any(d.term.casefold() == "glycolysis" for d in definitions)
    concepts = provider.summarize("key_concepts", idx.chunks, topic="", limit=10, ctx=ctx)
    assert concepts and all(c.term for c in concepts)


def test_emphasis_summary_contains_what_the_lecturer_stressed(provider, idx, ctx):
    items = provider.summarize("emphasis", idx.chunks, topic="", limit=10, ctx=ctx)
    text = " ".join(i.text.casefold() for i in items)
    assert "hexokinase is inhibited by its product" in text or "phosphofructokinase is the key regulatory enzyme" in text         # what was stressed is in ...
    assert "remember that" not in text and "do not forget" not in text and "please note" not in text                           # ... without the lecturer's phrase around it


def test_a_slide_reference_is_added_when_the_slide_covers_the_sentence(provider, idx, ctx):
    items = provider.summarize("detailed", idx.chunks, topic="", limit=20, ctx=ctx)
    assert any(any(s.kind == "pdf" for s in i.sources) for i in items)
    assert all(i.sources[0].kind in ("transcript", "pdf") for i in items)


def test_tidy_removes_fillers_and_ends_the_sentence():
    assert tidy("So, um, the enzyme is hexokinase") == "The enzyme is hexokinase."


# --------------------------------------------------------------------------------------------------------------------- flashcards
def test_flashcards_cover_many_types_and_have_all_fields(provider, idx, ctx):
    cards = provider.generate_flashcards(idx.chunks, topic="", types=CARD_TYPES, limit=60, ctx=ctx)
    types = {c.type for c in cards}
    assert len(types) >= 7 and {"definition", "concept", "cause_effect", "exam_recall"} <= types
    for c in cards:
        assert c.front and c.back and c.front != c.back
        assert c.difficulty in ("easy", "medium", "hard") and c.topic and c.sources
        assert c.type in CARD_TYPES
    assert len({c.id for c in cards}) == len(cards)


def test_definition_card_asks_for_the_term(provider, idx, ctx):
    cards = provider.generate_flashcards(idx.chunks, topic="", types=["definition"], limit=10, ctx=ctx)
    assert any(c.front.casefold() == "what is glycolysis?" and "pathway" in c.back for c in cards)


def test_exam_recall_card_blanks_a_term_and_gives_it_back(provider, idx, ctx):
    cards = provider.generate_flashcards(idx.chunks, topic="", types=["exam_recall"], limit=10, ctx=ctx)
    assert cards and all("_____" in c.front for c in cards)
    for c in cards:
        answer = c.back.split(" - ")[0]
        assert answer.casefold() in c.back.casefold().split(" - ", 1)[1]


def test_card_ids_are_stable_so_progress_survives_regeneration(idx):
    first = ExtractiveProvider().generate_flashcards(idx.chunks, topic="", types=CARD_TYPES, limit=30, ctx=StudyContext(index=idx))
    second = ExtractiveProvider().generate_flashcards(idx.chunks, topic="", types=CARD_TYPES, limit=30, ctx=StudyContext(index=idx))
    assert [c.id for c in first] == [c.id for c in second]


# --------------------------------------------------------------------------------------------------------------------- questions
def test_question_types_and_fields(provider, idx, ctx):
    questions = provider.generate_questions(idx.chunks, topic="", types=QUESTION_TYPES, limit=60, ctx=ctx)
    types = {q.type for q in questions}
    assert {"mcq", "true_false", "short_answer", "explain_why"} <= types
    for q in questions:
        assert q.question and q.answer and q.explanation and q.sources and q.topic and q.difficulty in ("easy", "medium", "hard")
        assert q.type in QUESTION_TYPES
    assert len({q.id for q in questions}) == len(questions)


def test_multiple_choice_has_five_distinct_options_with_the_answer_among_them(provider, idx, ctx):
    mcqs = [q for q in provider.generate_questions(idx.chunks, topic="", types=["mcq"], limit=30, ctx=ctx)]
    assert mcqs
    for q in mcqs:
        assert len(q.options) == 5 and len({o.casefold() for o in q.options}) == 5
        assert q.answer in q.options and q.correct_index == q.options.index(q.answer)
        assert q.explanation.startswith("The lecture says")


def test_true_false_has_both_kinds_and_the_false_one_differs_from_the_lecture(provider, idx, ctx):
    tf = provider.generate_questions(idx.chunks, topic="", types=["true_false"], limit=30, ctx=ctx)
    assert {q.answer for q in tf} == {"True", "False"}
    for q in tf:
        assert q.options == ["True", "False"]
        statement = q.question.removeprefix("True or false: ")
        if q.answer == "True":
            assert statement.casefold() in _all_text(idx)
        else:
            assert statement.casefold() not in _all_text(idx) and q.explanation.startswith("False")


def test_clinical_and_mechanism_questions_offer_statements_from_other_parts(provider, idx, ctx):
    for qtype in ("clinical_vignette", "mechanism"):
        for q in provider.generate_questions(idx.chunks, topic="", types=[qtype], limit=10, ctx=ctx):
            assert len(q.options) == 5 and q.answer in q.options


def test_questions_are_stable_between_runs(idx):
    a = ExtractiveProvider().generate_questions(idx.chunks, topic="", types=QUESTION_TYPES, limit=30, ctx=StudyContext(index=idx))
    b = ExtractiveProvider().generate_questions(idx.chunks, topic="", types=QUESTION_TYPES, limit=30, ctx=StudyContext(index=idx))
    assert [(q.id, q.options) for q in a] == [(q.id, q.options) for q in b]


# --------------------------------------------------------------------------------------------------------- explain / evaluate
def test_explain_answers_from_the_lecture_and_says_so_when_it_cannot(provider, idx, ctx):
    answer = provider.explain("What does hexokinase do?", idx.chunks, ctx)
    assert "hexokinase" in answer.text.casefold() and answer.sources
    missing = provider.explain("What is the dose of metformin in renal failure?", idx.chunks, ctx)
    assert missing.text == NOT_IN_SOURCES and not missing.sources


def test_free_text_answers_are_judged_by_meaning_overlap(provider, ctx):
    q = Question("q1", "short_answer", "What is glycolysis?", "Glycolysis is the pathway that converts glucose into pyruvate in the cytoplasm.",
                 "The lecture says: ...")
    good = provider.evaluate_answer(q, "the pathway converting glucose to pyruvate in cytoplasm", [], ctx)
    assert good.correct and good.score >= 0.5
    bad = provider.evaluate_answer(q, "it is a protein of the extracellular matrix", [], ctx)
    assert not bad.correct and bad.missing
    assert not provider.evaluate_answer(q, "   ", [], ctx).correct


def test_choice_questions_are_judged_exactly(provider, ctx):
    q = Question("q2", "mcq", "?", "Enolase", "because", options=["Enolase", "A", "B", "C", "D"])
    assert provider.evaluate_answer(q, "enolase", [], ctx).correct
    assert not provider.evaluate_answer(q, "A", [], ctx).correct


# ------------------------------------------------------------------------------------------------------------------ grounding
def test_grounding_checks_words_and_numbers(idx):
    chunk = idx.get("t0000")
    assert G.support_score(chunk.text, [chunk]) == 1.0
    assert G.support_score("Hexokinase kills 400 patients every year", [chunk]) == 0.0         # an invented number
    assert G.support_score("Insulin is made in the kidney of birds", [chunk]) < 0.3


def test_ungrounded_items_are_dropped_in_strict_mode(idx):
    good = SummaryItem("Glycolysis converts glucose into pyruvate.", [SourceRef.of(idx.get("t0000"))])
    invented = SummaryItem("Metformin 500 mg twice daily cures diabetes.", [SourceRef.of(idx.get("t0000"))])
    external = SummaryItem("Glycolysis converts glucose into pyruvate.", [SourceRef.of(idx.get("t0000"))], external=True)
    kept, report = G.ground_items([good, invented, external], idx, strict=True)
    assert kept == [good] and report.dropped == 2
    assert report.reasons == {"unsupported": 1, "external-knowledge": 1}


def test_an_item_without_a_valid_source_gets_the_best_matching_passage(idx):
    item = SummaryItem("Hexokinase phosphorylates glucose to glucose-6-phosphate.", [SourceRef("zzz", "transcript", "Transcript 00:00-00:01")])
    kept, report = G.ground_items([item], idx, strict=True)
    assert kept and kept[0].sources[0].chunk_id == "t0000" and report.dropped == 0


def test_malformed_questions_are_dropped(idx):
    ref = [SourceRef.of(idx.get("t0000"))]
    four = Question("a", "mcq", "Q?", "Hexokinase", "Hexokinase phosphorylates glucose", ["Hexokinase", "x", "y", "z"], ref)
    wrong_answer = Question("b", "mcq", "Q?", "Aldolase", "Hexokinase phosphorylates glucose", ["Hexokinase", "w", "x", "y", "z"], ref)
    fine = Question("c", "true_false", "True or false: hexokinase phosphorylates glucose.", "true", "Hexokinase phosphorylates glucose", [], ref)
    kept, report = G.ground_items([four, wrong_answer, fine], idx, strict=True)
    assert [q.id for q in kept] == ["c"] and kept[0].answer == "True" and kept[0].options == ["True", "False"]
    assert report.reasons["malformed"] == 2


# ---------------------------------------------------------------------------------------------------------------------- LLM providers
class FakeTransport:
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def __call__(self, url, headers, body, timeout):
        self.calls.append((url, headers, copy.deepcopy(body), timeout))
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        return reply


def ollama_reply(payload):
    return {"message": {"role": "assistant", "content": payload if isinstance(payload, str) else json.dumps(payload)}}


def test_parse_json_accepts_fences_and_surrounding_text():
    assert parse_json('Sure!\n```json\n{"a": {"b": "}"}}\n```') == {"a": {"b": "}"}}
    with pytest.raises(ValueError):
        parse_json("no json here")
    with pytest.raises(ValueError):
        parse_json('{"a": ')


def test_ollama_summary_roundtrip_uses_the_lecture_text_only(idx, ctx):
    reply = ollama_reply({"items": [{"text": "Hexokinase phosphorylates glucose to glucose-6-phosphate.", "sources": ["t0000", "p001", "nope"]}]})
    transport = FakeTransport(reply)
    provider = llm.OllamaProvider(ProviderConfig(kind="ollama", model="qwen3:8b"), transport=transport)
    items = provider.summarize("quick", idx.chunks[:4], topic="", limit=3, ctx=ctx)
    assert items and [s.chunk_id for s in items[0].sources] == ["t0000", "p001"]            # the unknown id is ignored
    url, headers, body, timeout = transport.calls[0]
    assert url == "http://localhost:11434/api/chat" and body["model"] == "qwen3:8b" and body["stream"] is False
    assert body["keep_alive"] == "2m" and "num_gpu" not in body["options"]
    prompt = body["messages"][1]["content"]
    assert "[t0000 |" in prompt and "ONLY" in body["messages"][0]["content"]
    assert "t0009" not in prompt                                                              # only the chunks it was given


def test_invalid_json_is_retried_once_then_reported(idx, ctx):
    provider = llm.OllamaProvider(ProviderConfig(kind="ollama", model="m"), transport=FakeTransport(ollama_reply("not json at all")))
    with pytest.raises(ProviderError, match="expected format"):
        provider.summarize("quick", idx.chunks[:2], topic="", limit=2, ctx=ctx)
    transport = FakeTransport(ollama_reply("oops"), ollama_reply({"items": [{"text": "Hexokinase uses ATP.", "sources": ["t0000"]}]}))
    ok = llm.OllamaProvider(ProviderConfig(kind="ollama", model="m"), transport=transport)
    assert ok.summarize("quick", idx.chunks[:2], topic="", limit=2, ctx=ctx)[0].text == "Hexokinase uses ATP."
    assert len(transport.calls) == 2 and "not valid JSON" in transport.calls[1][2]["messages"][1]["content"]


def test_long_lectures_are_sent_in_batches(idx, ctx):
    transport = FakeTransport(ollama_reply({"items": []}))
    provider = llm.OllamaProvider(ProviderConfig(kind="ollama", model="m", context_words=300), transport=transport)
    provider.summarize("detailed", idx.chunks, topic="", limit=10, ctx=ctx)
    assert len(transport.calls) > 2
    seen = [line for _u, _h, body, _t in transport.calls for line in body["messages"][1]["content"].splitlines() if line.startswith("[t")]
    assert len(seen) == len({line for line in seen}) == len(idx.of_source("transcript"))   # every passage sent exactly once


def test_flashcards_and_questions_are_parsed_and_checked(idx, ctx):
    cards = {"cards": [{"type": "definition", "front": "What is glycolysis?", "back": "The pathway that converts glucose into pyruvate.",
                        "difficulty": "easy", "sources": ["t0000"]}, {"type": "bogus", "front": "", "back": "x"}]}
    provider = llm.OllamaProvider(ProviderConfig(kind="ollama", model="m"), transport=FakeTransport(ollama_reply(cards)))
    made = provider.generate_flashcards(idx.chunks[:3], topic="Glycolysis", types=CARD_TYPES, limit=5, ctx=ctx)
    assert len(made) == 1 and made[0].type == "definition" and made[0].topic == "Glycolysis"
    questions = {"questions": [{"type": "mcq", "question": "Which enzyme phosphorylates glucose?", "options": ["Hexokinase", "A", "B", "C", "D"],
                                "answer": "Hexokinase", "explanation": "Hexokinase phosphorylates glucose.", "sources": ["t0000"]}]}
    provider = llm.OllamaProvider(ProviderConfig(kind="ollama", model="m"), transport=FakeTransport(ollama_reply(questions)))
    made_q = provider.generate_questions(idx.chunks[:3], topic="", types=QUESTION_TYPES, limit=5, ctx=ctx)
    assert made_q[0].correct_index == 0 and G.ground_items(made_q, idx)[1].kept == 1


def test_explain_and_evaluate_through_a_provider(idx, ctx):
    provider = llm.OllamaProvider(ProviderConfig(kind="ollama", model="m"),
                                  transport=FakeTransport(ollama_reply({"answer": "NOT_COVERED", "sources": []})))
    assert provider.explain("dose of metformin?", idx.chunks[:2], ctx).text == NOT_IN_SOURCES
    judge = {"correct": True, "score": 0.9, "feedback": "Good.", "missing": ["pyruvate"]}
    provider = llm.OllamaProvider(ProviderConfig(kind="ollama", model="m"), transport=FakeTransport(ollama_reply(judge)))
    ev = provider.evaluate_answer(Question("q", "short_answer", "What is glycolysis?", "x", ""), "glucose breakdown", idx.chunks[:1], ctx)
    assert ev.correct and ev.score == 0.9 and ev.missing == ["pyruvate"]


def test_ollama_check_tells_what_is_missing():
    tags = {"models": [{"name": "llama3:8b"}]}
    with pytest.raises(ProviderError, match="ollama pull qwen3:8b"):
        llm.OllamaProvider(ProviderConfig(kind="ollama", model="qwen3:8b"), transport=FakeTransport(tags)).check()
    with pytest.raises(ProviderError, match="No Ollama model"):
        llm.OllamaProvider(ProviderConfig(kind="ollama"), transport=FakeTransport(tags)).check()
    llm.OllamaProvider(ProviderConfig(kind="ollama", model="llama3:8b"), transport=FakeTransport(tags)).check()
    llm.OllamaProvider(ProviderConfig(kind="ollama", model="llama3"), transport=FakeTransport(tags)).check()


def test_openai_compatible_request_shape_and_key_handling(idx, ctx):
    reply = {"choices": [{"message": {"content": json.dumps({"items": [{"text": "Hexokinase uses ATP.", "sources": ["t0000"]}]})}}]}
    transport = FakeTransport(reply)
    provider = llm.OpenAICompatibleProvider(ProviderConfig(kind="openai", model="gpt-x"), api_key="sk-secret", transport=transport)
    provider.check()
    assert provider.kind == "cloud"
    provider.summarize("quick", idx.chunks[:2], topic="", limit=2, ctx=ctx)
    url, headers, body, _t = transport.calls[0]
    assert url == "https://api.openai.com/v1/chat/completions" and headers == {"Authorization": "Bearer sk-secret"}
    assert body["response_format"] == {"type": "json_object"}
    with pytest.raises(ProviderError, match="API key"):
        llm.OpenAICompatibleProvider(ProviderConfig(kind="openai", model="gpt-x"), transport=transport).check()
    local = llm.OpenAICompatibleProvider(ProviderConfig(kind="openai", model="m", base_url="http://localhost:1234/v1"), transport=transport)
    local.check()
    assert local.kind == "local" and local._headers() == {}


def test_openai_compatible_retries_without_json_mode_on_http_400(idx, ctx):
    ok = {"choices": [{"message": {"content": json.dumps({"items": []})}}]}
    transport = FakeTransport(ProviderError("server answered with an error (HTTP 400)."), ok)
    provider = llm.OpenAICompatibleProvider(ProviderConfig(kind="openai", model="m", base_url="http://localhost:1234/v1"), transport=transport)
    provider.summarize("quick", idx.chunks[:2], topic="", limit=2, ctx=ctx)
    assert "response_format" in transport.calls[0][2] and "response_format" not in transport.calls[1][2]


def test_anthropic_request_shape(idx, ctx):
    reply = {"content": [{"type": "text", "text": json.dumps({"items": [{"text": "Hexokinase uses ATP.", "sources": ["t0000"]}]})}]}
    transport = FakeTransport(reply)
    provider = llm.AnthropicProvider(ProviderConfig(kind="anthropic"), api_key="key-1", transport=transport)
    provider.check()
    assert provider.model == llm.DEFAULT_ANTHROPIC_MODEL and provider.kind == "cloud"
    provider.summarize("quick", idx.chunks[:2], topic="", limit=2, ctx=ctx)
    url, headers, body, _t = transport.calls[0]
    assert url == "https://api.anthropic.com/v1/messages" and headers["x-api-key"] == "key-1" and headers["anthropic-version"]
    assert body["system"] and body["messages"][0]["role"] == "user" and body["max_tokens"] > 0
    with pytest.raises(ProviderError, match="API key"):
        llm.AnthropicProvider(ProviderConfig(kind="anthropic"), transport=transport).check()


def test_create_provider_picks_the_right_class():
    assert isinstance(create_provider(None), ExtractiveProvider)
    assert isinstance(create_provider(ProviderConfig(kind="builtin")), ExtractiveProvider)
    assert isinstance(create_provider(ProviderConfig(kind="ollama", model="m")), llm.OllamaProvider)
    assert isinstance(create_provider(ProviderConfig(kind="openai", model="m"), api_key="k"), llm.OpenAICompatibleProvider)
    assert isinstance(create_provider(ProviderConfig(kind="anthropic"), api_key="k"), llm.AnthropicProvider)
    with pytest.raises(ProviderError):
        create_provider(ProviderConfig(kind="mystery"))
    config = ProviderConfig.from_dict(ProviderConfig(kind="ollama", model="m", base_url="http://x").to_dict())
    assert (config.kind, config.model, config.base_url) == ("ollama", "m", "http://x")
    assert issubclass(ExtractiveProvider, AIProvider) and issubclass(llm.LLMProvider, AIProvider)


# --------------------------------------------------------------------------------------------- real HTTP against a fake local server
class _Handler(BaseHTTPRequestHandler):
    routes = {}

    def log_message(self, *args):                # silence
        pass

    def _send(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        code, body = self.routes.get(self.path, (404, {"error": "no route"}))
        _Handler.last_headers = dict(self.headers)
        self._send(code, body)

    do_GET = do_POST


@pytest.fixture()
def server():
    _Handler.routes = {}
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", _Handler
    httpd.shutdown()


def test_real_http_success_and_error_messages(server):
    base, handler = server
    handler.routes["/ok"] = (200, {"hello": "world"})
    handler.routes["/denied"] = (401, {"error": "bad key sk-secret"})
    handler.routes["/limit"] = (429, {})
    handler.routes["/boom"] = (500, {"error": "kaboom"})
    assert llm.urllib_transport(base + "/ok", {"X-Test": "1"}, {"a": 1}, 5) == {"hello": "world"}
    assert handler.last_headers["X-Test"] == "1"
    with pytest.raises(ProviderError, match="API key") as denied:
        llm.urllib_transport(base + "/denied", {}, {}, 5)
    assert "sk-secret" not in str(denied.value)                       # a key echoed by the server never reaches the screen
    with pytest.raises(ProviderError, match="rate"):
        llm.urllib_transport(base + "/limit", {}, {}, 5)
    with pytest.raises(ProviderError, match="HTTP 500"):
        llm.urllib_transport(base + "/boom", {}, {}, 5)
    with pytest.raises(ProviderError, match="does not know"):
        llm.urllib_transport(base + "/missing", {}, {}, 5)


def test_unreachable_server_is_a_clear_error():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]                                  # nothing listens here once the socket is closed
    with pytest.raises(ProviderError, match="Cannot reach|did not answer"):
        llm.urllib_transport(f"http://127.0.0.1:{port}/api/chat", {}, {}, 2)


def test_ollama_against_a_fake_server_end_to_end(server, idx, ctx):
    base, handler = server
    handler.routes["/api/tags"] = (200, {"models": [{"name": "m:latest"}]})
    handler.routes["/api/chat"] = (200, ollama_reply({"items": [{"text": "Hexokinase uses ATP.", "sources": ["t0000"]}]}))
    provider = llm.OllamaProvider(ProviderConfig(kind="ollama", model="m", base_url=base))
    provider.check()
    assert provider.summarize("quick", idx.chunks[:2], topic="", limit=2, ctx=ctx)[0].text == "Hexokinase uses ATP."


def test_ollama_can_be_kept_off_the_gpu_and_unloaded_quickly(idx, ctx):
    transport = FakeTransport(ollama_reply({"items": []}))
    config = ProviderConfig(kind="ollama", model="m", extra={"num_gpu": 0, "keep_alive": "30s"})
    llm.OllamaProvider(config, transport=transport).summarize("quick", idx.chunks[:2], topic="", limit=2, ctx=ctx)
    body = transport.calls[0][2]
    assert body["options"]["num_gpu"] == 0 and body["keep_alive"] == "30s"


# ---------------------------------------------------------------------------------------------------------------- Turkish wording
def test_turkish_lectures_get_turkish_wording_around_their_own_sentences():
    from study_support import TURKISH, segments
    from src.study.index import build_index

    index = build_index(segments(TURKISH * 3))
    ctx_tr = StudyContext(index=index, language="tr")
    provider = ExtractiveProvider()
    cards = provider.generate_flashcards(index.chunks, topic="", types=CARD_TYPES, limit=40, ctx=ctx_tr)
    fronts = " ".join(c.front for c in cards)
    assert cards and any(w in fronts for w in ("nedir?", "Boşluğu doldurun", "Açıklayın", "Şunun sonucu nedir"))
    assert "What is" not in fronts and "Fill in the blank" not in fronts
    assert any(c.front == "Glikoliz nedir?" for c in cards)                        # "... glikoliz denir." is a definition
    questions = provider.generate_questions(index.chunks, topic="", types=QUESTION_TYPES, limit=40, ctx=ctx_tr)
    tf = [q for q in questions if q.type == "true_false"]
    assert tf and {q.answer for q in tf} <= {"Doğru", "Yanlış"} and all(q.options == ["Doğru", "Yanlış"] for q in tf)
    assert all(q.explanation.startswith(("Doğru.", "Yanlış.")) for q in tf)
    assert all("Derste" in q.explanation for q in questions)
    kept, report = G.ground_items(questions, index, strict=True, check_text=False)
    assert report.dropped == 0 and len(kept) == len(questions)                      # grounding knows Doğru / Yanlış


def test_a_language_without_study_wording_falls_back_to_english():
    from src.study.i18n import canonical_true_false, strings, true_false_pair

    assert strings("ja")["what_is"] == "What is {x}?" and strings("tr-TR")["what_is"] == "{x} nedir?" and strings("de")["what_is"] == "Was ist {x}?"
    assert true_false_pair("tr") == ("Doğru", "Yanlış") and canonical_true_false("true") == ("True", ("True", "False"))
    assert canonical_true_false("yanlış") == ("Yanlış", ("Doğru", "Yanlış")) and canonical_true_false("maybe") is None


# ---------------------------------------------------------------------------------------------------------------- a cancel ends a model call that is still waiting
class _SlowOllama(BaseHTTPRequestHandler):
    seen = []

    def log_message(self, *args):
        pass

    def do_POST(self):                                                        # a model that takes five seconds to answer
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        type(self).seen.append(body)
        if body.get("keep_alive") == 0:                                       # "drop the model from memory": answered at once
            self._reply({"done": True})
            return
        try:
            for _ in range(50):
                readable, _w, _x = select.select([self.connection], [], [], 0.1)
                if readable and self.connection.recv(1, socket.MSG_PEEK) == b"":          # the client closed its end: a real server stops generating here
                    type(self).seen.append({"connection": "closed by the client"})
                    return
            self._reply({"message": {"content": "{}"}})
        except OSError:
            type(self).seen.append({"connection": "closed by the client"})

    def do_GET(self):                                                         # /api/tags: the installed models
        self._reply({"models": [{"name": "m"}]})

    def _reply(self, data):
        raw = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


@pytest.fixture
def slow_server():
    _SlowOllama.seen = []
    httpd = HTTPServer(("127.0.0.1", 0), _SlowOllama)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def test_a_cancel_ends_a_model_call_that_is_still_waiting_for_its_answer(slow_server):
    import time

    from src.study.providers import StudyCancelled

    stop = threading.Event()
    threading.Timer(0.5, stop.set).start()
    started = time.monotonic()
    with llm.cancel_scope(stop.is_set):
        with pytest.raises(StudyCancelled):
            llm.urllib_transport(slow_server + "/api/chat", {}, {"model": "m"}, 30.0)
    assert time.monotonic() - started < 3.0                                       # not the five seconds the server needs, not the 30 of the time-out


def test_without_a_cancel_the_call_waits_for_the_answer_as_before(slow_server):
    data = llm.urllib_transport(slow_server + "/api/chat", {}, {"model": "m"}, 30.0)
    assert data == {"message": {"content": "{}"}}


def test_releasing_the_model_still_works_after_a_cancel(slow_server):
    provider = llm.OllamaProvider(ProviderConfig(kind="ollama", model="m", base_url=slow_server, timeout=30.0))
    with llm.cancel_scope(lambda: True):                                           # everything is cancelled - except the tidy-up
        provider.release()
    assert {"model": "m", "keep_alive": 0} in _SlowOllama.seen


def test_the_cancel_closes_the_connection_so_that_the_server_stops_generating(slow_server):
    from src.study.providers import StudyCancelled

    stop = threading.Event()
    threading.Timer(0.4, stop.set).start()
    with llm.cancel_scope(stop.is_set):
        with pytest.raises(StudyCancelled):
            llm.urllib_transport(slow_server + "/api/chat", {}, {"model": "m"}, 30.0)
    deadline = threading.Event()
    for _ in range(40):
        if {"connection": "closed by the client"} in _SlowOllama.seen:
            break
        deadline.wait(0.1)
    assert {"connection": "closed by the client"} in _SlowOllama.seen


def test_closing_the_window_while_the_model_is_writing_ends_the_generation_quickly_and_frees_the_model(slow_server):
    """The shutdown that used to crash the program: the Study thread sat in a five-minute model call, the window waited six seconds, and the program exited with the thread alive."""
    import time

    from src.services.settings_service import Settings
    from src.study.automatic import AutomaticStudyEngine, choose_engine
    from src.study.engine import StudyOptions
    from src.study.providers import StudyCancelled

    choice = choose_engine(Settings(study_provider="ollama", study_model="m", study_base_url=slow_server), transport=llm.urllib_transport)
    assert choice.mode == "local"
    stop = threading.Event()
    outcome = {}

    def work():
        try:
            with llm.cancel_scope(stop.is_set):
                AutomaticStudyEngine(choice).generate(lecture_index(), lecture_id="x", title="t", options=StudyOptions(), cancelled=stop.is_set)
            outcome["result"] = "finished"
        except StudyCancelled:
            outcome["result"] = "cancelled"
        except BaseException as exc:                                              # noqa: BLE001
            outcome["result"] = repr(exc)

    thread = threading.Thread(target=work)
    thread.start()
    threading.Event().wait(1.0)                                                   # the model is busy with its first answer
    started = time.monotonic()
    stop.set()                                                                    # the window is closed
    thread.join(10)
    assert not thread.is_alive() and outcome["result"] == "cancelled"
    assert time.monotonic() - started < 3.0
    assert {"model": "m", "keep_alive": 0} in _SlowOllama.seen                   # and the model was asked to leave the graphics card
