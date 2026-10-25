"""Optional Deepgram Nova-3 Medical backend (online, third-party). NEVER the default.

It is only used when the user asks for it (``--backend deepgram|compare``), or when the local transcript looks
suspicious AND the user allowed it (``--deepgram-if-suspect``). Audio then leaves this computer - the README says so.

Talks to the documented REST endpoint with the standard library (no SDK, so no SDK-version breakage):
    POST https://api.deepgram.com/v1/listen?model=nova-3-medical&...   body = the audio bytes
The API key comes from the environment or a ``.env`` file (``DEEPGRAM_API_KEY=...``); it is never logged,
never written to any output file, and never part of an error message.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from . import PROJECT_ROOT
from .schema import Segment, Word

log = logging.getLogger(__name__)

ENDPOINT = "https://api.deepgram.com/v1/listen"
DEFAULT_MODEL = "nova-3-medical"
MIME = {".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".mp4": "video/mp4", ".wav": "audio/wav", ".flac": "audio/flac",
        ".ogg": "audio/ogg", ".opus": "audio/opus", ".aac": "audio/aac", ".webm": "audio/webm", ".mov": "video/quicktime"}


class DeepgramError(RuntimeError):
    """Any Deepgram problem (messages never contain the API key)."""


class DeepgramUnavailable(DeepgramError):
    """No API key configured."""


class DeepgramAuthError(DeepgramError):
    """401 / 403."""


class DeepgramQuotaError(DeepgramError):
    """402: out of credits."""


class DeepgramRateLimitError(DeepgramError):
    """429."""


class DeepgramRequestError(DeepgramError):
    """Bad request, server or network error."""


def load_env_file(path: Path = PROJECT_ROOT / ".env") -> dict[str, str]:
    """Parse ``KEY=VALUE`` lines (comments, blank lines and surrounding quotes handled). Missing file -> {}."""
    values: dict[str, str] = {}
    if not Path(path).is_file():
        return values
    for raw in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def get_api_key(env_path: Path = PROJECT_ROOT / ".env") -> Optional[str]:
    """Environment variable first, then ``.env``. Placeholder values are treated as 'not set'."""
    key = os.environ.get("DEEPGRAM_API_KEY") or load_env_file(env_path).get("DEEPGRAM_API_KEY")
    if not key or key.lower().startswith(("your_", "paste", "xxxx", "changeme")):
        return None
    return key


@dataclass
class DeepgramResult:
    segments: list[Segment] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
    language: Optional[str] = None
    duration: Optional[float] = None
    transcript: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


def _words(raw_words: Sequence[dict]) -> list[Word]:
    return [Word(float(w.get("start", 0.0)), float(w.get("end", 0.0)),
                 str(w.get("punctuated_word") or w.get("word") or ""), w.get("confidence")) for w in raw_words or []]


def _group_words(words: Sequence[Word], model: str, gap: float = 1.2, max_words: int = 28) -> list[Segment]:
    segments: list[Segment] = []
    current: list[Word] = []

    def flush() -> None:
        if current:
            probs = [w.probability for w in current if w.probability is not None]
            segments.append(Segment(start=current[0].start, end=current[-1].end,
                                    text=" ".join(w.word.strip() for w in current), model=model, backend="deepgram",
                                    confidence=(sum(probs) / len(probs)) if probs else None, words=list(current)))

    for word in words:
        if current and (word.start - current[-1].end > gap or len(current) >= max_words):
            flush()
            current = []
        current.append(word)
    flush()
    return segments


def parse_response(raw: dict[str, Any], model: str = DEFAULT_MODEL) -> DeepgramResult:
    """Deepgram JSON -> :class:`DeepgramResult` (utterances preferred, else words grouped into sentences)."""
    results = raw.get("results") or {}
    metadata = raw.get("metadata") or {}
    channels = results.get("channels") or []
    alternative = ((channels[0].get("alternatives") or [{}])[0]) if channels else {}
    segments: list[Segment] = []
    for utt in results.get("utterances") or []:
        text = str(utt.get("transcript") or "").strip()
        if text:
            segments.append(Segment(start=float(utt.get("start", 0.0)), end=float(utt.get("end", 0.0)), text=text,
                                    model=model, backend="deepgram", confidence=utt.get("confidence"),
                                    words=_words(utt.get("words") or [])))
    if not segments and alternative.get("words"):
        segments = _group_words(_words(alternative["words"]), model)
    if not segments and alternative.get("transcript"):
        segments = [Segment(0.0, float(results.get("duration") or metadata.get("duration") or 0.0),
                            str(alternative["transcript"]).strip(), model=model, backend="deepgram")]
    return DeepgramResult(
        segments=segments, raw=raw, language=metadata.get("language") or (results.get("channels") or [{}])[0].get("detected_language"),
        duration=metadata.get("duration") or results.get("duration"), transcript=str(alternative.get("transcript") or ""),
        meta={"request_id": metadata.get("request_id"), "model": model, "word_count": len(alternative.get("words") or []),
              "utterance_count": len(results.get("utterances") or []), "confidence": alternative.get("confidence")})


def classify_http_error(code: int, body: str = "") -> DeepgramError:
    snippet = body.strip().replace("\n", " ")[:200]
    if code in (401, 403):
        return DeepgramAuthError(f"Deepgram rejected the API key (HTTP {code}).")
    if code == 402:
        return DeepgramQuotaError("Deepgram reports insufficient credits (HTTP 402).")
    if code == 429:
        return DeepgramRateLimitError("Deepgram rate limit (HTTP 429).")
    return DeepgramRequestError(f"Deepgram request failed (HTTP {code}): {snippet}")


class DeepgramTranscriber:
    def __init__(self, api_key: Optional[str], model: str = DEFAULT_MODEL, endpoint: str = ENDPOINT,
                 opener: Optional[Callable[..., Any]] = None, timeout: float = 1800.0, max_retries: int = 3,
                 backoff: float = 2.0, max_keyterms: int = 100, sleep: Callable[[float], None] = time.sleep) -> None:
        if not api_key:
            raise DeepgramUnavailable("DEEPGRAM_API_KEY is not set. Copy .env.example to .env and put your key there.")
        self._key = api_key
        self.model = model
        self.endpoint = endpoint
        self._open = opener or urllib.request.urlopen
        self.timeout, self.max_retries, self.backoff = timeout, max_retries, backoff
        self.max_keyterms = max_keyterms
        self._sleep = sleep

    def build_url(self, language: Optional[str] = "en", keyterms: Sequence[str] = ()) -> str:
        params: list[tuple[str, str]] = [
            ("model", self.model), ("smart_format", "true"), ("punctuate", "true"), ("utterances", "true"),
        ]
        if language and language != "auto":
            params.append(("language", language))
        params += [("keyterm", t) for t in list(dict.fromkeys(t for t in keyterms if t))[: self.max_keyterms]]
        return f"{self.endpoint}?{urllib.parse.urlencode(params)}"

    def transcribe_file(self, path: Path, language: Optional[str] = "en", keyterms: Sequence[str] = ()) -> DeepgramResult:
        path = Path(path)
        url = self.build_url(language, keyterms)
        body = path.read_bytes()
        headers = {"Authorization": f"Token {self._key}", "Content-Type": MIME.get(path.suffix.lower(), "application/octet-stream")}
        log.warning("DEEPGRAM: sending %s (%.1f MB) to a third-party service (model %s, %d keyterms).",
                    path.name, len(body) / 1e6, self.model, min(len(keyterms), self.max_keyterms))
        last: Optional[DeepgramError] = None
        started = time.time()
        for attempt in range(1, max(1, self.max_retries) + 1):
            try:
                request = urllib.request.Request(url, data=body, headers=headers, method="POST")
                with self._open(request, timeout=self.timeout) as response:
                    raw = json.loads(response.read().decode("utf-8"))
                result = parse_response(raw, self.model)
                result.meta["elapsed_seconds"] = round(time.time() - started, 1)
                result.meta["attempts"] = attempt
                return result
            except urllib.error.HTTPError as exc:
                error = classify_http_error(exc.code, exc.read().decode("utf-8", "replace") if hasattr(exc, "read") else "")
            except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as exc:
                error = DeepgramRequestError(f"Deepgram network/response problem: {type(exc).__name__}")
            last = error
            if isinstance(error, (DeepgramAuthError, DeepgramQuotaError)) or attempt >= self.max_retries:
                raise error
            log.warning("Deepgram attempt %d/%d failed (%s); retrying", attempt, self.max_retries, error)
            self._sleep(self.backoff * attempt)
        raise last or DeepgramRequestError("Deepgram request failed")


def suspicious_local_transcript(uncertain: int, total: int, hallucination_flags: int, repetition_flags: int,
                                thresholds: dict[str, float]) -> tuple[bool, str]:
    """Is the local transcript doubtful enough to justify an (opt-in) online second opinion?"""
    ratio = uncertain / total if total else 0.0
    if ratio >= thresholds.get("uncertain_ratio", 0.10):
        return True, f"{ratio:.0%} of segments are uncertain"
    if hallucination_flags + repetition_flags >= thresholds.get("loop_flags", 5):
        return True, f"{hallucination_flags} hallucination + {repetition_flags} repetition flags"
    return False, "local transcript looks healthy"
