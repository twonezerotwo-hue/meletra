"""Language-model providers: Ollama (local), OpenAI-compatible servers (local such as LM Studio / llama.cpp, or cloud), Anthropic (cloud).

They only use ``urllib`` (no extra package), take an injectable ``transport`` so tests never touch the network, and turn every kind of
failure (server not running, model not installed, wrong key, rate limit, time-out, answer that is not JSON) into a
:class:`ProviderError` with a message the student can act on. The API key is passed in by the caller (it lives in the OS credential
store) and is never written to a log or an error message.
"""

from __future__ import annotations

import http.client
import json
import logging
import math
import socket
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Optional, Sequence
from urllib.parse import urlparse

from . import prompts as P
from .models import CARD_TYPES, QUESTION_TYPES, Chunk, Flashcard, ProviderInfo, Question, SourceRef, SummaryItem, item_id
from .grounding import resolve_sources
from .providers import NOT_IN_SOURCES, AIProvider, Evaluation, ProviderConfig, ProviderError, StudyCancelled, StudyContext

log = logging.getLogger(__name__)

Transport = Callable[[str, dict, Optional[dict], float], dict]
OLLAMA_URL = "http://localhost:11434"
OPENAI_URL = "https://api.openai.com/v1"
ANTHROPIC_URL = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-5-5"
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]"}


def is_local_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in _LOCAL_HOSTS or host.endswith(".localhost") or host.startswith("192.168.") or host.startswith("10.")


_scope = threading.local()


@contextmanager
def cancel_scope(cancelled: Optional[Callable[[], bool]]) -> Iterator[None]:
    """While active, a model call made by THIS thread ends as soon as ``cancelled()`` is true: it raises :class:`StudyCancelled` and the connection to the server is closed (a local
    server stops generating). Without it a call blocks until the model has answered - minutes for a long lecture - and a window that is closed meanwhile cannot stop the thread."""
    previous = getattr(_scope, "cancelled", None)
    _scope.cancelled = cancelled
    try:
        yield
    finally:
        _scope.cancelled = previous


@contextmanager
def not_cancellable() -> Iterator[None]:
    """For the tidy-up that must still happen after a cancel (asking the server to drop the model from memory)."""
    with cancel_scope(None):
        yield


def _recording_opener(connections: list) -> urllib.request.OpenerDirector:
    """An opener (plain http only) that remembers its connection, so that a cancel can close the socket under a call that is waiting for the answer."""
    def connect(host: str, **kwargs: Any) -> http.client.HTTPConnection:
        connection = http.client.HTTPConnection(host, **kwargs)
        connections.append(connection)
        return connection

    class Handler(urllib.request.HTTPHandler):
        def http_open(self, req):                                          # noqa: N802 - urllib API
            return self.do_open(connect, req)

    return urllib.request.build_opener(Handler)


def urllib_transport(url: str, headers: dict, body: Optional[dict], timeout: float) -> dict:
    """POST ``body`` as JSON (GET when ``body`` is None) and return the decoded JSON reply. Raises ProviderError (StudyCancelled inside a :func:`cancel_scope` that was cancelled)."""
    cancelled = getattr(_scope, "cancelled", None)
    if cancelled is None:
        return _request(url, headers, body, timeout)
    connections: list = []
    opener = _recording_opener(connections) if url.startswith("http://") else None
    box: dict[str, Any] = {}

    def work() -> None:
        try:
            box["value"] = _request(url, headers, body, timeout, opener)
        except BaseException as exc:                                        # noqa: BLE001 - handed to the calling thread
            box["error"] = exc

    worker = threading.Thread(target=work, name="model-call", daemon=True)
    worker.start()
    while worker.is_alive():
        worker.join(0.2)
        if cancelled():
            for connection in list(connections):
                try:
                    if connection.sock is not None:
                        connection.sock.shutdown(socket.SHUT_RDWR)                  # close() alone waits for the reader that is blocked on this socket; shutdown ends it and tells the server
                    connection.close()
                except OSError:
                    pass
            raise StudyCancelled()
    if "error" in box:
        raise box["error"]
    return box["value"]


def _request(url: str, headers: dict, body: Optional[dict], timeout: float, opener: Optional[urllib.request.OpenerDirector] = None) -> dict:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers},
                                     method="GET" if body is None else "POST")
    host = urlparse(url).netloc
    try:
        with (opener.open if opener is not None else urllib.request.urlopen)(request, timeout=timeout) as response:        # noqa: S310 - http(s) URLs chosen by the student
            payload = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
        except Exception:                                                          # noqa: BLE001
            pass
        if exc.code in (401, 403):
            raise ProviderError(f"{host} rejected the API key (HTTP {exc.code}). Check the key in Settings > AI Study.") from None
        if exc.code == 404:
            raise ProviderError(f"{host} does not know this model or address (HTTP 404). {_short(detail)}") from None
        if exc.code == 429:
            raise ProviderError(f"{host} is rate limiting the requests (HTTP 429). Try again in a minute.") from None
        raise ProviderError(f"{host} answered with an error (HTTP {exc.code}). {_short(detail)}") from None
    except urllib.error.URLError as exc:
        reason = exc.reason
        if isinstance(reason, (socket.timeout, TimeoutError)):
            raise ProviderError(f"{host} did not answer within {int(timeout)} seconds.") from None
        raise ProviderError(f"Cannot reach {host}. Is the server running? ({reason.__class__.__name__})") from None
    except (socket.timeout, TimeoutError):
        raise ProviderError(f"{host} did not answer within {int(timeout)} seconds.") from None
    except OSError as exc:
        raise ProviderError(f"Cannot talk to {host} ({exc.__class__.__name__}).") from None
    try:
        return json.loads(payload)
    except ValueError:
        raise ProviderError(f"{host} sent a reply that is not JSON.") from None


def _short(text: str) -> str:
    text = " ".join(text.split())
    return text[:200]


class LLMProvider(AIProvider):
    """Shared logic: batching of the lecture text, prompts, JSON parsing, conversion to study items."""

    def __init__(self, config: ProviderConfig, *, transport: Optional[Transport] = None) -> None:
        self.config = config
        self.model = config.model
        self.transport: Transport = transport or urllib_transport

    # ---- to be provided by the concrete provider
    def _complete(self, system: str, user: str) -> str:                       # pragma: no cover - interface
        raise NotImplementedError

    # ---- plumbing
    def info(self) -> ProviderInfo:
        return ProviderInfo(name=self.name, kind=self.kind, model=self.model)

    def _ask(self, user: str, ctx: StudyContext, system: str = P.SYSTEM) -> dict:
        """One request; a reply that is not valid JSON gets one more try with a reminder."""
        last = ""
        for attempt in range(2):
            raw = self._complete(system, user if attempt == 0 else user + "\n\nYour last reply was not valid JSON. Reply with the JSON object only.")
            try:
                data = P.parse_json(raw)
            except ValueError as exc:
                last = str(exc)
                continue
            if isinstance(data, dict):
                return data
        raise ProviderError(f"{self.name} did not answer in the expected format ({last}).")

    def _batches(self, chunks: Sequence[Chunk]) -> list[list[Chunk]]:
        budget = max(self.config.context_words, 300)
        batches: list[list[Chunk]] = []
        current: list[Chunk] = []
        size = 0
        for chunk in chunks:
            n = len(chunk.text.split())
            if current and size + n > budget:
                batches.append(current)
                current, size = [], 0
            current.append(chunk)
            size += n
        if current:
            batches.append(current)
        return batches

    def _split_limit(self, limit: int, batches: list[list[Chunk]], batch: list[Chunk]) -> int:
        total = sum(len(c.text.split()) for b in batches for c in b) or 1
        mine = sum(len(c.text.split()) for c in batch)
        return max(1, math.ceil(limit * mine / total))

    @staticmethod
    def _ids(raw: Any) -> list[str]:
        if isinstance(raw, str):
            raw = [raw]
        return [str(i).strip().strip("[]") for i in raw] if isinstance(raw, list) else []

    # ---- interface
    def summarize(self, kind: str, chunks: Sequence[Chunk], *, topic: str, limit: int, ctx: StudyContext) -> list[SummaryItem]:
        out: list[SummaryItem] = []
        batches = self._batches(chunks)
        for batch in batches:
            part = self._split_limit(limit, batches, batch)
            data = self._ask(P.summary_prompt(kind, batch, topic, part, ctx.language), ctx)
            for row in data.get("items", []) if isinstance(data.get("items"), list) else []:
                if not isinstance(row, dict) or not str(row.get("text", "")).strip():
                    continue
                out.append(SummaryItem(text=str(row["text"]).strip(), sources=resolve_sources(self._ids(row.get("sources")), ctx.index),
                                       topic=topic, term=str(row.get("term", "") or "").strip()))
        return out[:max(limit, 1) * 2]

    def generate_flashcards(self, chunks: Sequence[Chunk], *, topic: str, types: Sequence[str], limit: int,
                            ctx: StudyContext) -> list[Flashcard]:
        types = [t for t in (types or CARD_TYPES) if t in CARD_TYPES]
        out: list[Flashcard] = []
        batches = self._batches(chunks)
        for batch in batches:
            part = self._split_limit(limit, batches, batch)
            data = self._ask(P.flashcard_prompt(batch, topic, types, part, ctx.language), ctx)
            for row in data.get("cards", []) if isinstance(data.get("cards"), list) else []:
                if not isinstance(row, dict):
                    continue
                front, back = str(row.get("front", "")).strip(), str(row.get("back", "")).strip()
                ctype = str(row.get("type", "")).strip()
                if not front or not back:
                    continue
                difficulty = str(row.get("difficulty", "medium")).lower()
                out.append(Flashcard(id=item_id("c", ctype, front, back), type=ctype if ctype in CARD_TYPES else "concept", front=front,
                                     back=back, sources=resolve_sources(self._ids(row.get("sources")), ctx.index),
                                     difficulty=difficulty if difficulty in ("easy", "medium", "hard") else "medium", topic=topic))
        return out

    def generate_questions(self, chunks: Sequence[Chunk], *, topic: str, types: Sequence[str], limit: int,
                           ctx: StudyContext) -> list[Question]:
        types = [t for t in (types or QUESTION_TYPES) if t in QUESTION_TYPES]
        out: list[Question] = []
        batches = self._batches(chunks)
        for batch in batches:
            part = self._split_limit(limit, batches, batch)
            data = self._ask(P.question_prompt(batch, topic, types, part, ctx.language), ctx)
            for row in data.get("questions", []) if isinstance(data.get("questions"), list) else []:
                if not isinstance(row, dict):
                    continue
                text, answer = str(row.get("question", "")).strip(), str(row.get("answer", "")).strip()
                qtype = str(row.get("type", "")).strip()
                if not text or not answer or qtype not in QUESTION_TYPES:
                    continue
                options = [str(o).strip() for o in row.get("options", []) if str(o).strip()] if isinstance(row.get("options"), list) else []
                difficulty = str(row.get("difficulty", "medium")).lower()
                out.append(Question(id=item_id("q", qtype, text, answer), type=qtype, question=text, answer=answer,
                                    explanation=str(row.get("explanation", "")).strip(), options=options,
                                    sources=resolve_sources(self._ids(row.get("sources")), ctx.index),
                                    difficulty=difficulty if difficulty in ("easy", "medium", "hard") else "medium", topic=topic))
        return out

    def explain(self, question: str, chunks: Sequence[Chunk], ctx: StudyContext) -> SummaryItem:
        data = self._ask(P.explain_prompt(question, list(chunks)[:12], ctx.language), ctx)
        answer = str(data.get("answer", "")).strip()
        if not answer or answer.upper().startswith("NOT_COVERED"):
            return SummaryItem(text=NOT_IN_SOURCES, sources=[])
        return SummaryItem(text=answer, sources=resolve_sources(self._ids(data.get("sources")), ctx.index))

    def evaluate_answer(self, question: Question, answer: str, chunks: Sequence[Chunk], ctx: StudyContext) -> Evaluation:
        data = self._ask(P.evaluate_prompt(question, answer, list(chunks)[:6], ctx.language), ctx)
        try:
            score = max(0.0, min(1.0, float(data.get("score", 1.0 if data.get("correct") else 0.0))))
        except (TypeError, ValueError):
            score = 1.0 if data.get("correct") else 0.0
        missing = [str(m) for m in data.get("missing", []) if str(m).strip()] if isinstance(data.get("missing"), list) else []
        return Evaluation(bool(data.get("correct")), round(score, 2), str(data.get("feedback", "")).strip(), missing)


# ------------------------------------------------------------------------------------------------------------------------ Ollama
class OllamaProvider(LLMProvider):
    name = "ollama"
    kind = "local"

    def __init__(self, config: ProviderConfig, *, transport: Optional[Transport] = None) -> None:
        super().__init__(config, transport=transport)
        self.base = (config.base_url or OLLAMA_URL).rstrip("/")

    def check(self) -> None:
        if not self.model:
            raise ProviderError("No Ollama model is chosen. Pick one in Settings > AI Study.")
        data = self.transport(self.base + "/api/tags", {}, None, min(self.config.timeout, 10.0))
        names = [str(m.get("name", "")) for m in data.get("models", []) if isinstance(m, dict)]
        if not any(n == self.model or n.split(":")[0] == self.model or n == self.model + ":latest" for n in names):
            have = ", ".join(names[:5]) or "none"
            raise ProviderError(f"The model '{self.model}' is not installed in Ollama (installed: {have}). Install it with: ollama pull {self.model}")

    def release(self) -> None:
        """Ask Ollama to drop this model from memory now (keep_alive 0). Only the model this program used; nothing else is stopped."""
        if self.model:
            with not_cancellable():                                         # this runs BECAUSE the work was cancelled
                self.transport(self.base + "/api/generate", {}, {"model": self.model, "keep_alive": 0}, min(self.config.timeout, 10.0))

    def _complete(self, system: str, user: str) -> str:
        options: dict[str, Any] = {"temperature": self.config.temperature, "num_ctx": 8192, "num_predict": int(self.config.extra.get("num_predict", 3072))}      # a runaway generation ends here, not at the time-out
        if "num_gpu" in self.config.extra:                        # 0 = keep this model off the GPU (it may be shared with a transcription)
            options["num_gpu"] = int(self.config.extra["num_gpu"])
        body = {"model": self.model, "stream": False, "format": "json", "think": False,
                "keep_alive": str(self.config.extra.get("keep_alive", "2m")),     # free the memory soon after the study material is made
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "options": options}
        data = self.transport(self.base + "/api/chat", {}, body, self.config.timeout)
        message = data.get("message") or {}
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise ProviderError("Ollama sent an empty reply.")
        return content


# ----------------------------------------------------------------------------------------------------------------- OpenAI-compatible
class OpenAICompatibleProvider(LLMProvider):
    name = "openai"

    def __init__(self, config: ProviderConfig, *, api_key: Optional[str] = None, transport: Optional[Transport] = None) -> None:
        super().__init__(config, transport=transport)
        self.base = (config.base_url or OPENAI_URL).rstrip("/")
        self.api_key = api_key or ""
        self.kind = "local" if is_local_url(self.base) else "cloud"

    def check(self) -> None:
        if not self.model:
            raise ProviderError("No model name is set. Enter one in Settings > AI Study.")
        if self.kind == "cloud" and not self.api_key:
            raise ProviderError("No API key is saved for this provider. Add it in Settings > AI Study.")

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def _complete(self, system: str, user: str) -> str:
        body = {"model": self.model, "temperature": self.config.temperature,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "response_format": {"type": "json_object"}}
        try:
            data = self.transport(self.base + "/chat/completions", self._headers(), body, self.config.timeout)
        except ProviderError as exc:
            if "HTTP 400" not in str(exc):
                raise
            body.pop("response_format")                                      # some local servers do not know json_object
            data = self.transport(self.base + "/chat/completions", self._headers(), body, self.config.timeout)
        try:
            return str(data["choices"][0]["message"]["content"] or "")
        except (KeyError, IndexError, TypeError):
            raise ProviderError("The server's reply has no message.") from None


# ----------------------------------------------------------------------------------------------------------------------- Anthropic
class AnthropicProvider(LLMProvider):
    name = "anthropic"
    kind = "cloud"

    def __init__(self, config: ProviderConfig, *, api_key: Optional[str] = None, transport: Optional[Transport] = None) -> None:
        super().__init__(config, transport=transport)
        self.base = (config.base_url or ANTHROPIC_URL).rstrip("/")
        self.model = config.model or DEFAULT_ANTHROPIC_MODEL
        self.api_key = api_key or ""

    def check(self) -> None:
        if not self.api_key:
            raise ProviderError("No Anthropic API key is saved. Add it in Settings > AI Study.")

    def _complete(self, system: str, user: str) -> str:
        body = {"model": self.model, "max_tokens": 4096, "temperature": self.config.temperature, "system": system,
                "messages": [{"role": "user", "content": user}]}
        headers = {"x-api-key": self.api_key, "anthropic-version": ANTHROPIC_VERSION}
        data = self.transport(self.base + "/messages", headers, body, self.config.timeout)
        try:
            return "".join(block.get("text", "") for block in data["content"] if isinstance(block, dict))
        except (KeyError, TypeError):
            raise ProviderError("Anthropic's reply has no text.") from None
