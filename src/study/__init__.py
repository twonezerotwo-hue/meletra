"""Study system of MELETRA (first made for version 0.2): summaries, flashcards, questions and an interactive study mode made from the
transcript of a lecture and, optionally, its slides (PDF).

Layers (each importable and testable without the GUI and without any AI model):

* ``models``     - plain dataclasses saved as one JSON file next to the transcript
* ``text`` / ``index`` / ``sources`` - sentences, topics, a small local BM25 index, transcript <-> slide links
* ``providers``  - the provider-independent ``AIProvider`` interface, the built-in extractive provider (needs nothing installed) and
                   optional Ollama / OpenAI-compatible / Anthropic providers
* ``grounding``  - every generated item must be supported by the sources it cites
* ``engine``     - orchestration: index -> provider (with fallback) -> grounding -> ``StudyPack``
* ``store`` / ``session`` - progress on this computer and the interactive study session

None of this is imported by the transcription pipeline: a study failure can never affect a transcription.
"""

from __future__ import annotations
