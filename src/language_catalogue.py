"""The languages the start screen offers (kept apart from ``language.py``, whose behaviour is the frozen 0.1.1 core)."""

from __future__ import annotations

#: (Whisper code, name). The spoken (input) language may also be "auto"; the output language never is.
SPOKEN_LANGUAGES = (("en", "English"), ("tr", "Turkish"), ("de", "German"), ("hu", "Hungarian"), ("ko", "Korean"), ("zh", "Chinese"),
                    ("hi", "Hindi"), ("nl", "Dutch"), ("fr", "French"), ("es", "Spanish"))
INPUT_LANGUAGES = (("auto", "Auto Detect"),) + SPOKEN_LANGUAGES
OUTPUT_LANGUAGES = SPOKEN_LANGUAGES
DEFAULT_INPUT_LANGUAGE = "auto"
DEFAULT_OUTPUT_LANGUAGE = "en"


def language_name(code: str) -> str:
    """The name shown to the student for a language code ("auto" -> "Auto Detect"); an unknown code is shown as it is."""
    names = dict(INPUT_LANGUAGES)
    names["mixed"] = "Mixed"
    return names.get(code, code or "")
