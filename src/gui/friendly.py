"""Plain-language text for the simple screens: durations, time left, languages. No technical words."""

from __future__ import annotations

from typing import Optional

LANGUAGE_NAMES = {"en": "English", "tr": "Turkish", "de": "German", "fr": "French", "es": "Spanish", "ru": "Russian", "bg": "Bulgarian",
                  "hu": "Hungarian", "ar": "Arabic", "it": "Italian", "pt": "Portuguese", "pl": "Polish", "mixed": "Mixed languages",
                  "auto": "Detecting ..."}


def language_name(code: Optional[str]) -> str:
    if not code:
        return "-"
    return LANGUAGE_NAMES.get(str(code).lower(), str(code))


def subject_name(value: Optional[str]) -> str:
    if not value or value == "auto":
        return "Detecting ..."
    return str(value).replace("_", " ").title()


def duration(seconds: Optional[float]) -> str:
    """"1 h 12 min", "12 min", "45 sec"."""
    if seconds is None or seconds < 0:
        return "-"
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours} h {minutes:02d} min"
    if minutes:
        return f"{minutes} min"
    return f"{secs} sec"


def time_left(seconds: Optional[float]) -> str:
    """"Estimated time remaining: 12 min" - rounded the way people say it; never a second-by-second countdown."""
    if seconds is None or seconds < 0:
        return "Estimated time remaining: calculating ..."
    if seconds < 45:
        return "Estimated time remaining: less than a minute"
    minutes = int(round(seconds / 60))
    if minutes < 60:
        return f"Estimated time remaining: {minutes} min"
    hours, minutes = divmod(minutes, 60)
    return f"Estimated time remaining: {hours} h {minutes:02d} min"


def size(nbytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024 or unit == "GB":
            return f"{nbytes:.1f} GB" if unit == "GB" else f"{nbytes:.0f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} GB"


def plural(n: int, one: str, many: Optional[str] = None) -> str:
    return f"{n} {one if n == 1 else (many or one + 's')}"


#: what the job is doing before the first transcript segment: (title of the running screen, one line under it)
STAGES = {
    "probing": ("Preparing {kind}...", "Reading the file ..."),
    "extracting": ("Extracting audio...", "Taking the sound out of the {kind} ..."),
    "speech_analysis": ("Preparing {kind}...", "Finding where people speak ..."),
    "model_loading": ("Preparing {kind}...", "Loading the speech model ..."),
    "language": ("Analyzing the recording...", "Detecting the language ..."),
    "analysis": ("Analyzing the recording...", "Listening to the first minutes to recognise the subject ..."),
    "transcribing": ("Transcribing...", "Writing down the first words ..."),
    "finalizing": ("Finalizing...", "Correcting medical terms, checking the transcript and writing the files ..."),
    "translating": ("Translating...", "Writing the transcript in your Output Language ..."),
    "language_check": ("Checking the transcript...", "Checking the language of every passage ..."),
    "coverage": ("Checking the transcript...", "Looking for passages that were not written down ..."),
}


def stage_texts(stage: str, kind: str = "audio") -> tuple[str, str]:
    """``(title, line)`` of a stage; ``kind`` is "video" or "audio". Unknown stages keep the generic wording."""
    title, line = STAGES.get(stage, ("Preparing {kind}...", "Working ..."))
    return title.format(kind=kind), line.format(kind=kind)


_JARGON = ("cuda", "gpu", "vram", "hotword", "checkpoint", "chunk", "ctranslate", "compute type", "vad", "beam", "float16", "int8")


def simple_warning(text: str) -> str:
    """A warning of the transcription for the simple running screen, in plain words ("" = not worth showing there).

    The classic page shows the original text; the full wording is also in *Technical details*."""
    low = text.casefold()
    if "running on the cpu" in low or "cuda acceleration unavailable" in low:
        return ("This computer will use its processor instead of a graphics card. That works, but it is slower - "
                "a long lecture can take several hours.")
    if "earlier checkpoint" in low or "checkpoint" in low:
        return "An older, unfinished transcription of this file was found, but it used other settings. This one starts from the beginning."
    if "hotwords are off" in low:
        return ""
    if "gpu memory" in low or "vram" in low:
        return "Another program is using a lot of graphics memory, so the transcription may be slower than usual."
    if any(word in low for word in _JARGON):
        return "Something may slow the transcription down. Details are under \"Technical details\"."
    return text


_FAILURES = (
    (("out of memory", "ran out of memory", "cuda error", "oom"), "The computer ran out of memory. Close other programs and try again."),
    (("no space left", "disk full", "not enough space", "enospc"), "There is not enough free disk space. Free some space and try again."),
    (("could not be read", "invalid data", "moov atom", "no audio", "audio track", "damaged", "corrupt"), "The recording could not be read. It may be damaged or contain no sound."),
    (("no such file", "file not found", "filenotfound", "does not exist", "no longer"), "The recording could not be found. It may have been moved or deleted."),
    (("permission denied", "access is denied"), "The program is not allowed to write to the output folder. Choose another folder in Settings."),
    (("model",), "The speech files are missing or damaged. Open Settings, then Models, to check them."),
)


def plain_failure(message: str) -> str:
    """The reason a transcription stopped, in plain words (the original text stays in the log); an unknown reason is kept as it is."""
    low = (message or "").casefold()
    for needles, text in _FAILURES:
        if any(needle in low for needle in needles):
            return text
    return (message or "").strip() or "Something went wrong."
