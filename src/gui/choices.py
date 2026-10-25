"""The values offered in drop-down lists, in one place."""

from __future__ import annotations

from ..knowledge_subjects import AREAS

from ..language_catalogue import INPUT_LANGUAGES, OUTPUT_LANGUAGES

LANGUAGE_CHOICES = [(label, code) for code, label in INPUT_LANGUAGES] + [("Mixed languages (each sentence)", "mixed")]      # classic window; the start screen offers the plain list
INPUT_LANGUAGE_CHOICES = [(label, code) for code, label in INPUT_LANGUAGES]
OUTPUT_LANGUAGE_CHOICES = [(label, code) for code, label in OUTPUT_LANGUAGES]
MODE_CHOICES = [("Best quality", "quality"), ("Fast", "fast")]
THEME_CHOICES = [("Follow Windows", "system"), ("Light", "light"), ("Dark", "dark")]
DEVICE_CHOICES = [("Automatic (GPU if available)", "auto"), ("NVIDIA GPU (CUDA)", "cuda"), ("CPU only", "cpu")]
COMPUTE_CHOICES = [("Automatic", "auto"), ("float16 (best quality)", "float16"), ("int8_float16", "int8_float16"), ("int8 (smallest)", "int8")]
HOTWORD_CHOICES = [("Automatic (relevant terms only)", "auto"), ("Off", "off")]
SUBJECT_AUTO = ("Auto Detect", "auto")


def theme_choices() -> list[tuple[str, str]]:
    """"Follow Windows" / "Follow macOS": the first entry names the system the program runs on."""
    from ..platform import current

    system = {"windows": "Windows", "macos": "macOS"}.get(current().name, "the system")
    return [(f"Follow {system}", "system"), ("Light", "light"), ("Dark", "dark")]


def subject_choices() -> list[tuple[str, str]]:
    return [SUBJECT_AUTO] + [(area.replace("_", " ").title(), area) for area in sorted(AREAS)]


def index_of(choices: list[tuple[str, str]], value: str, default: int = 0) -> int:
    for i, (_label, v) in enumerate(choices):
        if v == value:
            return i
    return default
