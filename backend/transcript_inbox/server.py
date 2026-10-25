"""The inbox server: POST /v1/transcripts -> validate, store, notify. GET /healthz. Standard library only (put it behind HTTPS: a reverse proxy or the host's TLS)."""

from __future__ import annotations

import json
import os
import smtplib
import ssl
import threading
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from email.message import EmailMessage
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Optional

MAX_BYTES = 3 * 1024 * 1024
ALLOWED_KEYS = ("schema", "installation_id", "app_version", "created_at", "lecture_name", "languages", "transcript", "medical_corrected_transcript", "translated_transcript")
TEXT_KEYS = ("transcript", "medical_corrected_transcript", "translated_transcript")
RATE_PER_HOUR = 30


class Rejected(ValueError):
    """The payload is not acceptable; the message is safe to show."""


def validate(raw: bytes) -> dict[str, Any]:
    if len(raw) > MAX_BYTES:
        raise Rejected("payload too large")
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise Rejected("not valid JSON") from exc
    if not isinstance(data, dict):
        raise Rejected("not a JSON object")
    unknown = sorted(set(data) - set(ALLOWED_KEYS))
    if unknown:
        raise Rejected("unknown field(s): " + ", ".join(unknown))
    if data.get("schema") != 1:
        raise Rejected("unsupported schema")
    try:
        uuid.UUID(str(data.get("installation_id", "")))
    except ValueError as exc:
        raise Rejected("installation_id is not a UUID") from exc
    if not isinstance(data.get("transcript"), str) or not data["transcript"].strip():
        raise Rejected("no transcript text")
    for key in TEXT_KEYS:
        if key in data and not isinstance(data[key], str):
            raise Rejected(f"{key} must be text")
    for key in ("app_version", "created_at", "lecture_name"):
        if not isinstance(data.get(key, ""), str) or len(data.get(key, "")) > 300:
            raise Rejected(f"{key} is not acceptable")
    if not isinstance(data.get("languages", {}), dict):
        raise Rejected("languages must be an object")
    return data


class RateLimiter:
    def __init__(self, per_hour: int = RATE_PER_HOUR, clock: Callable[[], float] = time.time) -> None:
        self.per_hour, self.clock = per_hour, clock
        self.hits: dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = self.clock()
        with self.lock:
            window = self.hits[key]
            while window and now - window[0] > 3600:
                window.popleft()
            if len(window) >= self.per_hour:
                return False
            window.append(now)
            return True


class Inbox:
    def __init__(self, folder: Path) -> None:
        self.folder = Path(folder)

    def store(self, data: dict[str, Any]) -> Path:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        target = self.folder / day
        target.mkdir(parents=True, exist_ok=True)
        path = target / f"{data['installation_id'][:8]}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        return path


class SmtpNotifier:
    """Mails the owner about a received transcript. Everything it needs comes from the SERVER's environment."""

    def __init__(self, env: Optional[dict[str, str]] = None, smtp_factory: Optional[Callable[..., Any]] = None) -> None:
        self.env = dict(os.environ if env is None else env)
        self._smtp = smtp_factory or smtplib.SMTP

    def configured(self) -> bool:
        return bool(self.env.get("OWNER_NOTIFICATION_EMAIL") and self.env.get("SMTP_HOST") and self.env.get("SMTP_USER") and self.env.get("SMTP_PASSWORD"))

    def notify(self, data: dict[str, Any], stored: Path) -> bool:
        if not self.configured():
            return False
        languages = data.get("languages") or {}
        message = EmailMessage()
        message["To"] = self.env["OWNER_NOTIFICATION_EMAIL"]
        message["From"] = self.env.get("MAIL_FROM") or self.env["SMTP_USER"]
        message["Subject"] = f"MELETRA transcript: {data.get('lecture_name') or 'untitled'} ({languages.get('spoken') or '?'} -> {languages.get('output') or '?'})"
        message.set_content(f"A completed transcript was shared.\n\nLecture: {data.get('lecture_name')}\nLanguages: {languages}\nProgram version: {data.get('app_version')}\n"
                            f"Installation: {data.get('installation_id')}\nReceived: {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
                            f"Words: {len(data['transcript'].split())}\nStored as: {stored.name}\n")
        if self.env.get("ATTACH_TRANSCRIPT", "1") == "1":
            message.add_attachment(data["transcript"].encode("utf-8"), maintype="text", subtype="plain", filename="transcript.txt")
        port = int(self.env.get("SMTP_PORT", "587"))
        with self._smtp(self.env["SMTP_HOST"], port, timeout=20) as server:
            if port != 25:
                server.starttls(context=ssl.create_default_context())
            server.login(self.env["SMTP_USER"], self.env["SMTP_PASSWORD"])
            server.send_message(message)
        return True


def make_handler(inbox: Inbox, notifier: Any, limiter: RateLimiter):
    class Handler(BaseHTTPRequestHandler):
        server_version = "MELETRAInbox"

        def log_message(self, *args) -> None:             # no transcript text and no address ever goes into a log line
            pass

        def _reply(self, code: int, body: dict) -> None:
            raw = json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/healthz":
                self._reply(200, {"ok": True})
            else:
                self._reply(404, {"ok": False})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/transcripts":
                return self._reply(404, {"ok": False})
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_BYTES:
                return self._reply(413, {"ok": False, "error": "size"})
            if not limiter.allow(self.client_address[0]):
                return self._reply(429, {"ok": False, "error": "rate"})
            try:
                data = validate(self.rfile.read(length))
            except Rejected as exc:
                return self._reply(400, {"ok": False, "error": str(exc)})
            if not limiter.allow("install:" + data["installation_id"]):
                return self._reply(429, {"ok": False, "error": "rate"})
            stored = inbox.store(data)
            try:
                notifier.notify(data, stored)
            except Exception:  # noqa: BLE001 - a mail problem never loses the transcript: it is stored; the client is told it arrived
                pass
            self._reply(202, {"ok": True})

    return Handler


def serve(host: str = "0.0.0.0", port: int = 8080, env: Optional[dict[str, str]] = None) -> ThreadingHTTPServer:
    env = dict(os.environ if env is None else env)
    inbox = Inbox(Path(env.get("INBOX_DIR", "inbox")))
    return ThreadingHTTPServer((host, port), make_handler(inbox, SmtpNotifier(env), RateLimiter()))


if __name__ == "__main__":
    httpd = serve(port=int(os.environ.get("PORT", "8080")))
    print("transcript inbox listening on", httpd.server_address)
    httpd.serve_forever()
