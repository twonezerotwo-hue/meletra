"""Where the developer's backend is (``config/sharing.json``, shipped with the program) and who this installation is (a random number, no personal data)."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from .. import RESOURCE_ROOT

CONFIG_FILE = RESOURCE_ROOT / "config" / "sharing.json"
ENV_ENDPOINT = "MLT_SHARING_ENDPOINT"            # tests and the owner's own checks; a shipped build reads config/sharing.json


def valid_endpoint(url: str) -> bool:
    """HTTPS only (a loopback address may use http: the automated tests talk to a server on this computer)."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    if parsed.scheme == "https" and parsed.hostname:
        return True
    return parsed.scheme == "http" and parsed.hostname in ("127.0.0.1", "localhost", "::1")


def endpoint(config_file: Optional[Path] = None) -> str:
    """The address transcripts are posted to, or "" when this build has none (then nothing is ever sent)."""
    url = os.environ.get(ENV_ENDPOINT, "").strip()
    if not url:
        try:
            url = str(json.loads(Path(config_file or CONFIG_FILE).read_text(encoding="utf-8")).get("endpoint") or "").strip()
        except (OSError, ValueError):
            url = ""
    return url if valid_endpoint(url) else ""


def installation_id(folder: Path) -> str:
    """A random identifier made once on this computer (uuid4: no hardware, user or network information goes into it)."""
    file = Path(folder) / "installation.json"
    try:
        value = json.loads(file.read_text(encoding="utf-8")).get("installation_id", "")
        if isinstance(value, str) and len(value) == 36:
            uuid.UUID(value)
            return value
    except (OSError, ValueError, AttributeError):
        pass
    value = str(uuid.uuid4())
    try:
        Path(folder).mkdir(parents=True, exist_ok=True)
        file.write_text(json.dumps({"installation_id": value, "note": "random number made on first use; identifies no person or computer"}), encoding="utf-8")
    except OSError:
        pass
    return value
