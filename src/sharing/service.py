"""The sharing service of the program: ties the setting, the queue and the uploader together."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Optional

from .. import __version__
from . import config, payload
from .queue import UploadQueue
from .uploader import Uploader, http_post

log = logging.getLogger(__name__)


class SharingService:
    """``settings`` is the SettingsService; ``folder`` is <data folder>/sharing."""

    def __init__(self, settings: Any, folder: Path, post: Callable[[str, bytes], int] = http_post, endpoint: Callable[[], str] = config.endpoint) -> None:
        self.settings = settings
        self.folder = Path(folder)
        self.queue = UploadQueue(self.folder)
        self.uploader = Uploader(self.queue, endpoint=endpoint, enabled=self.active, post=post)
        self._endpoint = endpoint

    # ------------------------------------------------------------------ state
    def wanted(self) -> bool:
        """The student's choice (Settings -> Privacy & Data)."""
        return bool(self.settings.settings.share_transcripts)

    def active(self) -> bool:
        """Sharing really happens only when it is wanted AND the student agreed to the notice (both are set together, when sharing is switched on in Settings)."""
        s = self.settings.settings
        return bool(s.share_transcripts and s.share_notice_shown)

    def configured(self) -> bool:
        return bool(self._endpoint())

    def state_text(self) -> str:
        if not self.wanted():
            return "Off: nothing is shared."
        if not self.settings.settings.share_notice_shown:
            return "On, but you have not agreed to the notice yet: nothing is shared."
        if not self.configured():
            return "On, but this version has no upload address configured: nothing is sent."
        return "On: completed transcripts are shared with MELETRA."

    def status(self) -> dict[str, Any]:
        data = self.queue.status()
        data.update({"enabled": self.wanted(), "active": self.active(), "configured": self.configured()})
        return data

    def reconcile(self) -> None:
        """Make the stored choice and what really happens the same thing. Sharing that was switched on by editing the settings file (without the notice) is not on; and while sharing
        is not allowed nothing may be waiting to be sent. Called when the program starts."""
        s = self.settings.settings
        if s.share_transcripts and not s.share_notice_shown:
            self.settings.update(share_transcripts=False)
        if not self.active():
            self.queue.clear()

    # ------------------------------------------------------------------ the choice
    def needs_choice(self) -> bool:
        """The first-launch question has not been answered yet (until then nothing is shared)."""
        return not self.settings.settings.share_choice_made

    def choose(self, share: bool) -> None:
        """The answer to the first-launch question. Only an explicit "share" turns sharing on; "keep private" (or closing the question) leaves it off."""
        self.settings.update(share_choice_made=True)
        if share:
            self.set_enabled(True)
            self.notice_shown()
        else:
            self.set_enabled(False)

    def set_enabled(self, on: bool) -> None:
        """Switch sharing on or off. OFF takes effect at once: nothing more is sent, everything still waiting is deleted, and the agreement is withdrawn (switching it on again asks again)."""
        self.settings.update(share_transcripts=bool(on))
        if not on:
            self.settings.update(share_notice_shown=False)
            self.queue.clear()
            self.queue.reset_status()
        else:
            self.uploader.nudge()

    # ------------------------------------------------------------------ what would be shared
    def preview(self, folder: Optional[Path], lecture_name: str = "") -> dict[str, Any]:
        """What a completed lecture would send, shown BEFORE anything is shared: the field names with their meaning, a short excerpt of the text, the size, and what is never sent.
        Nothing is queued or sent by looking."""
        meaning = {"schema": "format version", "installation_id": "a random number made on this computer; identifies no person", "app_version": "the version of MELETRA",
                   "created_at": "the time", "lecture_name": "the name of the lecture", "languages": "the spoken, requested and output language",
                   "transcript": "the transcript text", "medical_corrected_transcript": "the medical corrected transcript text, when it differs",
                   "translated_transcript": "the translated text, when you translated"}
        data = payload.build(Path(folder), lecture_name, config.installation_id(self.folder), __version__) if folder is not None else None
        text = str((data or {}).get("transcript", ""))
        return {"fields": [(key, meaning[key]) for key in payload.ALLOWED_KEYS], "sample": data, "excerpt": text[:600] + (f" ... ({len(text) - 600:,} more characters)" if len(text) > 600 else ""),
                "bytes": len(json.dumps(data, ensure_ascii=False).encode("utf-8")) if data else 0,
                "never": ["audio", "video", "PDF files", "pictures", "file paths", "your user name", "your computer's name", "any hardware number"]}

    def notice_shown(self) -> None:
        self.settings.update(share_notice_shown=True)
        self.uploader.nudge()

    # ------------------------------------------------------------------ the transcript
    def transcript_saved(self, folder: Path, lecture_name: str, spoken: str = "", requested: str = "", output: str = "") -> bool:
        """Call AFTER the transcript is saved on this computer. Prepares the payload and queues it; the upload itself happens later, in the background.
        Returns True when something was queued. Any problem is logged and swallowed: the local transcript is already safe."""
        try:
            if not self.active():
                return False
            data = payload.build(folder, lecture_name, config.installation_id(self.folder), __version__, spoken=spoken, requested=requested, output=output)
            if data is None:
                return False
            self.queue.enqueue(data)
            self.uploader.nudge()
            return True
        except Exception:  # noqa: BLE001
            log.debug("sharing: could not queue the transcript", exc_info=True)
            return False

    def start(self) -> None:
        self.uploader.start()

    def stop(self) -> None:
        self.uploader.stop()
