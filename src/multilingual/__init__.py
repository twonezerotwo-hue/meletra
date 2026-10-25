"""Multilingual transcription: language timeline, language-aware decoding, speech coverage, and the canonical source transcript every translation starts from.

The rule of this package (and of the program): **what was heard is stored once, in the language it was spoken in, and never overwritten.** A translation is always a separate file
made from that stored text.

* ``scripts``    - writing-system evidence (Hangul, Devanagari, Han, Latin ...).
* ``source``     - ``source_transcript.json``: per-segment language, text, status; a revision that changes when the text changes; old lectures are read through the same API.
* ``timeline``   - Auto Detect over time: language identification of windows, smoothing into runs, bounded verification of unsure windows.
* ``coverage``   - does the transcript cover what is audibly there? (an audit that does not trust the voice-activity filter that may have cut the speech out)
"""

from __future__ import annotations

SOURCE_FILE = "source_transcript.json"
TRANSLATIONS_DIR = "translations"
ENGINE_VERSION = "multilingual-1"          # bumped when the way a transcript is segmented / labelled changes
