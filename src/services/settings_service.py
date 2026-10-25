"""User settings: one JSON file, written atomically, versioned, migrated, always valid.

A corrupt or hand-edited file never stops the app: it is set aside as ``settings.json.corrupt`` and defaults are used.
Unknown values are replaced by the default of that field, never passed on to the pipeline.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

SCHEMA_VERSION = 6

from ..language_catalogue import INPUT_LANGUAGES, OUTPUT_LANGUAGES  # noqa: E402

LANGUAGES = tuple(code for code, _ in INPUT_LANGUAGES) + ("mixed",)
OUTPUT_LANGUAGE_CODES = tuple(code for code, _ in OUTPUT_LANGUAGES)
MODES = ("quality", "fast")
DEVICES = ("auto", "cuda", "cpu")
COMPUTE_TYPES = ("auto", "float16", "int8_float16", "int8")
THEMES = ("system", "light", "dark")
HOTWORD_MODES = ("auto", "off")
OUTPUT_FORMATS = ("txt", "srt", "vtt", "json")
UI_MODES = ("simple", "classic")                       # classic = the technical window of version 0.1.1
STUDY_PROVIDERS = ("builtin", "ollama", "openai", "anthropic")
OUTPUT_FORMAT_NOTE = "txt/srt/vtt/json are always written; the choice only controls which buttons the results screen shows"


@dataclass
class Settings:
    schema_version: int = SCHEMA_VERSION
    # --- General
    output_dir: str = ""                 # "" = next to the source file ("Medical Transcription" folder)
    theme: str = "dark"
    open_folder_after: bool = False      # the transcript is read inside MELETRA; opening Explorer after a transcription is the student's own opt-in
    language: str = "auto"               # the SPOKEN language (Input Language): auto = Whisper detects it
    output_language: str = "en"          # the language of the final transcript and the study material; the spoken transcript is always kept
    subject: str = "auto"                # auto | a vocabulary file | one of the 19 medical areas
    model_key: str = "whisper-large-v3"
    mode: str = "quality"
    # --- Performance
    device: str = "auto"
    compute_type: str = "auto"
    cpu_threads: int = 0                 # 0 = automatic
    vram_warnings: bool = True
    unload_ollama: bool = False          # advanced: `ollama stop` idle models before a job. OFF by default.
    beam_size: int = 0                   # 0 = the mode's default (5 for quality)
    vad: bool = True
    chunk_minutes: float = 15.0
    overlap_seconds: float = 3.0
    # --- Medical
    medical_context: bool = True
    hotwords: str = "auto"
    pdf_context: bool = True
    medical_expert: bool = True          # check terms in context + write the medical corrected transcript (raw transcript never edited)
    custom_terms: list[str] = field(default_factory=list)   # Settings > Medical vocabulary > Add terms (ranking hints, not replacements)
    medical_intelligence: str = "off"    # v4 knowledge layer: off | shadow (compare only) | on (adds its HIGH corrections where the Medical Expert changed nothing). Off: it did not pass its gates.
    output_formats: list[str] = field(default_factory=lambda: list(OUTPUT_FORMATS))
    # --- Models
    extra_model_dirs: list[str] = field(default_factory=list)
    # --- Cloud (the key itself lives in Windows Credential Manager, never here)
    cloud_enabled: bool = False
    deepgram_model: str = "nova-3-medical"
    # --- AI Study (0.2). The study AI is separate from the transcription engine and never runs during a transcription.
    ui_mode: str = "simple"              # simple | classic (the 0.1.1 technical window, for troubleshooting)
    show_advanced: bool = False          # show the Advanced section (performance, medical dictionaries, diagnostics) in Settings
    study_provider: str = "builtin"      # builtin (offline, no setup) | ollama | openai (also local servers) | anthropic
    study_model: str = ""
    study_base_url: str = ""             # "" = the provider's usual address
    study_language: str = "auto"         # auto = the language of the lecture, or a language code
    study_source_bound: bool = True      # only what the lecture sources say (the default); never switched off silently
    study_cloud_consent: bool = False    # a SEPARATE, explicit yes (Settings -> Advanced -> Study engine) before lecture text may go to a cloud study provider. OFF unless given.
    study_cards: int = 0                 # 0 = decided from the length of the lecture
    study_questions: int = 0
    # --- Privacy & Data: sharing of completed transcripts with MELETRA. OFF by default and only ever switched on by the student, in Settings -> Privacy & Data,
    # after reading what would be shared (share_notice_shown records that this was read and agreed to). Nothing is sent without both.
    share_transcripts: bool = False
    share_notice_shown: bool = False
    share_choice_made: bool = False      # the first-launch question ("Share completed transcript text with MELETRA" / "Keep transcripts private") has been answered. Answering "private" or closing it keeps sharing OFF.
    # --- App state
    first_run_done: bool = False
    last_input_dir: str = ""
    window_geometry: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_ENUMS = {"language": LANGUAGES, "output_language": OUTPUT_LANGUAGE_CODES, "mode": MODES, "device": DEVICES, "compute_type": COMPUTE_TYPES, "theme": THEMES,
          "hotwords": HOTWORD_MODES, "ui_mode": UI_MODES, "study_provider": STUDY_PROVIDERS}
_RANGES = {"cpu_threads": (0, 256), "beam_size": (0, 20), "chunk_minutes": (1.0, 60.0), "overlap_seconds": (0.0, 15.0),
           "study_cards": (0, 500), "study_questions": (0, 500)}


def migrate(data: dict[str, Any]) -> dict[str, Any]:
    """Bring a settings dict written by an older version up to :data:`SCHEMA_VERSION`."""
    data = dict(data)
    version = int(data.get("schema_version", 0) or 0)
    if version < 1:                                   # v0: early prototypes used these names
        for old, new in (("output_folder", "output_dir"), ("model", "model_key"), ("quality_mode", "mode")):
            if old in data and new not in data:
                data[new] = data.pop(old)
    if version < 2:                                   # v2: the model key became the registry key
        if data.get("model_key") in ("default", "large-v3", "Large-v3"):
            data["model_key"] = "whisper-large-v3"
        if data.get("model_key") in ("turkish_medical", "turkish"):
            data["model_key"] = "turkish-medical-large-v3"
    if version < 5:                                   # v5: sharing is opt-in. Any earlier value came from a default, not from a decision: it is reset to OFF
        data["share_transcripts"] = False
        data["share_notice_shown"] = False
    if version < 6:                                   # v6: the study engine is chosen by MELETRA, cloud study needs its own consent, and Explorer no longer opens after a transcription
        data["share_choice_made"] = bool(version >= 5 and data.get("share_transcripts") and data.get("share_notice_shown"))      # a 4.1.1 "yes" was an explicit one; everybody else is asked once
        if data.get("share_transcripts") and not data.get("share_notice_shown"):
            data["share_transcripts"] = False             # switched on without the notice (a file edit): it is not on
        data["study_cloud_consent"] = False
        data["open_folder_after"] = False             # the old True came from the default, not from a decision
    data["schema_version"] = SCHEMA_VERSION
    return data


def sanitize(data: dict[str, Any]) -> Settings:
    """Settings from untrusted data: unknown keys dropped, wrong types / values replaced by the default."""
    defaults = Settings()
    clean: dict[str, Any] = {}
    for f in fields(Settings):
        value = data.get(f.name, getattr(defaults, f.name))
        default = getattr(defaults, f.name)
        if f.name in _ENUMS and value not in _ENUMS[f.name]:
            value = default
        elif isinstance(default, bool):
            value = value if isinstance(value, bool) else default
        elif isinstance(default, int) and not isinstance(default, bool):
            value = value if isinstance(value, int) and not isinstance(value, bool) else default
        elif isinstance(default, float):
            value = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else default
        elif isinstance(default, str):
            value = value if isinstance(value, str) else default
        elif isinstance(default, list):
            value = [str(v) for v in value] if isinstance(value, list) else list(default)
        if f.name in _RANGES:
            low, high = _RANGES[f.name]
            value = type(default)(min(max(value, low), high))
        clean[f.name] = value
    clean["output_formats"] = [x for x in clean["output_formats"] if x in OUTPUT_FORMATS] or list(OUTPUT_FORMATS)
    return Settings(**clean)


class SettingsService:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._settings = Settings()
        self.load()

    @property
    def settings(self) -> Settings:
        return self._settings

    def load(self) -> Settings:
        if not self.path.is_file():
            self._settings = Settings()
            return self._settings
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("settings file is not a JSON object")
            self._settings = sanitize(migrate(raw))
        except (OSError, ValueError) as exc:
            log.warning("Settings file unreadable (%s); using defaults and keeping a copy as .corrupt", exc)
            try:
                os.replace(self.path, self.path.with_name(self.path.name + ".corrupt"))
            except OSError:
                pass
            self._settings = Settings()
        return self._settings

    def save(self) -> None:
        """Atomic: write a temp file, flush to disk, then replace - a crash can never leave half a file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + f".{os.getpid()}.tmp")
        payload = json.dumps(self._settings.to_dict(), indent=2, ensure_ascii=False)
        with tmp.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, self.path)

    def update(self, **changes: Any) -> Settings:
        merged = self._settings.to_dict()
        unknown = set(changes) - set(merged)
        if unknown:
            raise KeyError(f"unknown setting(s): {', '.join(sorted(unknown))}")
        merged.update(changes)
        self._settings = sanitize(merged)
        self.save()
        return self._settings

    def reset(self) -> Settings:
        self._settings = Settings(first_run_done=self._settings.first_run_done)
        self.save()
        return self._settings

    def backup_name(self) -> str:
        return time.strftime("settings-%Y%m%d-%H%M%S.json")
