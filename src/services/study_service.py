"""Glue between the app (settings, credentials, output folders) and the study engine - no Qt in here, so it is tested without a window.

A *lecture* is its output folder (the one the transcriber wrote). The study files live in ``<output folder>/study``; the slide text of
an attached PDF is stored there too (``pdf_pages.json``), so studying never needs the original PDF again.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from ..study.automatic import ENGINE_NAME, AutomaticStudyEngine, EngineChoice, choose_engine
from ..study.curate import GENERATION_VERSION
from ..study.engine import StudyEngine, StudyOptions
from ..study.index import DocumentIndex
from ..study.llm import Transport, cancel_scope
from ..study.models import StudyPack
from ..study.providers import AIProvider, ProviderError
from ..study.session import Evaluator
from ..study.sources import LectureSources, SourceError, build_lecture_index, find_transcript, load_sources
from ..multilingual import source as source_mod
from ..study.store import Progress, StudyStore
from ..translation.service import TranslationService
from .credential_service import CredentialService
from .settings_service import SettingsService

log = logging.getLogger(__name__)

CHOICE_SECONDS = 30.0                            # how long the answer to "which engine is available" is reused (looking is a request to the local server)
PDF_PAGES_FILE = "pdf_pages.json"
LANGUAGE_NAMES = {"en": "English", "tr": "Turkish"}
ProgressFn = Callable[[float, str], None]


@dataclass
class StudyState:
    """Everything the Study screen needs about one lecture."""

    folder: Path
    title: str
    store: StudyStore
    index: Optional[DocumentIndex] = None
    pack: Optional[StudyPack] = None
    progress: Progress = field(default_factory=Progress)
    warnings: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return self.pack is not None and not self.pack.empty and self.index is not None

    @property
    def mode(self) -> str:
        return self.index.mode if self.index else ""


class StudyService:
    def __init__(self, settings: SettingsService, credentials: CredentialService, transport: Optional[Transport] = None) -> None:
        self.settings = settings
        self.credentials = credentials
        self.transport = transport                   # tests inject a fake; None = real HTTP
        self.is_busy: Callable[[], bool] = lambda: False       # the window says whether a transcription is running (the graphics card is then not shared with a study model)
        self._choice: Optional[tuple[tuple, float, EngineChoice]] = None

    # ------------------------------------------------------------------------------------------------------------------ the engine
    def engine_choice(self, fresh: bool = False) -> EngineChoice:
        """The MELETRA Automatic Study Engine for this moment: a local language model when one is available, else the built-in offline method; a cloud provider only on the
        student's separate consent (``study_cloud_consent``). The student does not choose a method."""
        s = self.settings.settings
        busy = bool(self.is_busy())
        key = (s.study_provider, s.study_model, s.study_base_url, s.study_cloud_consent, busy)
        cached = self._choice
        if not fresh and cached is not None and cached[0] == key and time.monotonic() - cached[1] < CHOICE_SECONDS:
            return cached[2]
        choice = choose_engine(s, self.credentials, self.transport, busy=busy)
        self._choice = (key, time.monotonic(), choice)
        return choice

    def release_models(self, wait: bool = False) -> None:
        """A transcription is starting (or the program is closing): free the graphics card of the study model THIS program loaded (and nothing else). It runs in the background - the request
        waits behind any answer the model is still writing - and never raises. ``wait``: do it now and return when it is done (at exit, when a background thread would be killed)."""
        cached = self._choice
        release = getattr(cached[2].provider, "release", None) if cached is not None else None
        if not callable(release):
            return

        def work() -> None:
            try:
                release()
            except Exception:  # noqa: BLE001 - a tidy-up must never get in the way of a transcription
                log.debug("could not release the study model", exc_info=True)

        if wait:
            work()
            return
        threading.Thread(target=work, name="meletra-release-study-model", daemon=True).start()

    def create_provider(self) -> AIProvider:
        """The provider behind the engine (used to judge typed answers and to answer "explain this")."""
        return self.engine_choice().provider

    def provider_summary(self) -> str:
        """One line for the Study screens, no technical words."""
        return self.describe(self.engine_choice())

    @staticmethod
    def describe(choice: EngineChoice) -> str:
        if choice.mode == "local":
            return f"{ENGINE_NAME} · local AI ({choice.model})"
        if choice.mode == "cloud":
            return f"{ENGINE_NAME} · cloud AI ({choice.provider.name}): lecture text is sent to the provider"
        return f"{ENGINE_NAME} · built-in offline method"

    @staticmethod
    def describe_pack(pack: StudyPack) -> str:
        """What made THIS study material (not what would make it now)."""
        info, generation = pack.provider, pack.generation or {}
        if not generation:
            return f"{ENGINE_NAME} · made by an earlier version of MELETRA"
        if info.fallback_from:
            return f"{ENGINE_NAME} · built-in offline method (the local AI was not available this time)"
        if info.kind == "local":
            return f"{ENGINE_NAME} · local AI ({info.model})"
        if info.kind == "cloud":
            return f"{ENGINE_NAME} · cloud AI ({info.name}): lecture text was sent to the provider"
        return f"{ENGINE_NAME} · built-in offline method"

    def test_provider(self) -> tuple[bool, str]:
        """Settings -> Study -> "Check the study engine": what will be used right now, and is it ready? Never raises."""
        try:
            choice = self.engine_choice(fresh=True)
            choice.provider.check()
        except ProviderError as exc:
            return False, str(exc)
        except Exception as exc:                     # noqa: BLE001
            return False, f"{exc.__class__.__name__}: {exc}"
        if choice.mode == "builtin":
            return True, "The built-in offline method is ready. No local AI model was found on this computer" + (" - install one (for example with Ollama) to get richer questions." if not choice.notes else ".")
        return True, f"{self.describe(choice)} is ready"

    def diagnostics(self, folder: Optional[Path] = None) -> list[str]:
        """Advanced: which engine and model, why, and what the last study material of a lecture was made with (including any fallback)."""
        choice = self.engine_choice(fresh=True)
        lines = choice.diagnostics()
        if folder is not None:
            pack = StudyStore(Path(folder), self.study_language(Path(folder))).load_pack()
            if pack is not None:
                premium = (pack.stats or {}).get("premium") or {}
                lines.append(f"This lecture's material: {self.describe_pack(pack)}; recipe {pack.generation.get('version', 'earlier')}")
                for area in ("flashcards", "questions"):
                    stat = premium.get(area) or {}
                    if stat:
                        refused = sum((stat.get("reasons") or {}).values())
                        lines.append(f"{area.title()}: {stat.get('kept', 0)} kept ({stat.get('from_model', 0)} written by the model, {stat.get('from_builtin', 0)} by the built-in method), {refused} refused, "
                                     f"{stat.get('rounds', 0)} repair round(s)")
                if pack.provider.fallback_from:
                    lines.append(f"Fallback: {pack.provider.fallback_from} could not be used; the built-in method took over")
                lines += [f"Note: {note}" for note in premium.get("notes", [])]
        return lines

    LANGUAGE_FILE = "study_language.json"

    def study_languages(self, folder: Path) -> list[str]:
        """The languages this lecture can be studied in: the spoken language (when the lecture has only one) and every saved translation. A lecture spoken in several languages
        is studied in one of its translations - material cannot be made from a transcript that changes language."""
        folder = Path(folder)
        out: list[str] = []
        try:
            source = source_mod.load(folder)
        except Exception:  # noqa: BLE001
            source = None
        if source is not None:
            if len(source.detected_languages) == 1:
                out.append(source.detected_languages[0])
            for code in TranslationService().stored_targets(folder, any_engine=True):
                if code not in out and (folder / f"translated_{code}_transcript.json").is_file():
                    out.append(code)
        else:
            spoken = self._lecture_language(folder)
            if spoken:
                out.append(spoken)
            target = self._output_language(folder)
            if target and target not in out:
                out.append(target)
        return out

    def spoken_languages(self, folder: Path) -> list[str]:
        """The languages that were recognised in the lecture (from its saved transcript)."""
        try:
            source = source_mod.load(Path(folder))
        except Exception:  # noqa: BLE001
            return []
        return source.detected_languages if source is not None else []

    def needs_translation(self, folder: Path) -> bool:
        """A lecture spoken in several languages with no translation yet: study material needs one language to be made in."""
        try:
            source = source_mod.load(Path(folder))
        except Exception:  # noqa: BLE001
            return False
        return bool(source is not None and len(source.detected_languages) > 1 and not self.study_languages(folder))

    def chosen_language(self, folder: Path) -> str:
        """The language the student picked for THIS lecture ("" = none, or no longer available)."""
        try:
            data = json.loads((Path(folder) / "study" / self.LANGUAGE_FILE).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return ""
        code = str(data.get("language") or "")
        return code if code and code in self.study_languages(folder) else ""

    def set_language(self, folder: Path, code: str) -> None:
        path = Path(folder) / "study" / self.LANGUAGE_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"language": code}, ensure_ascii=False), encoding="utf-8")

    def study_language(self, folder: Path) -> str:
        """The language the study material is written in: the student's choice for this lecture, else the choice in Settings, else the Output Language of the lecture (its
        translated transcript when one was made), else the language that was spoken."""
        chosen = self.chosen_language(folder)
        if chosen:
            return chosen
        wanted = self.settings.settings.study_language
        if wanted != "auto":
            return wanted
        return self._output_language(folder) or self._lecture_language(folder) or "en"

    @staticmethod
    def _output_language(folder: Path) -> str:
        """The language of the lecture's translated transcript, "" when it was not translated (or the translation is not there)."""
        try:
            record = json.loads((Path(folder) / "metadata.json").read_text(encoding="utf-8")).get("translation") or {}
        except (OSError, ValueError):
            return ""
        target = str(record.get("output_language") or "")
        return target if target and (Path(folder) / f"translated_{target}_transcript.json").is_file() else ""

    def study_transcript(self, folder: Path) -> Optional[Path]:
        """The transcript the study material is made from: the translated one when the study language has one, else the spoken-language transcript."""
        translated = Path(folder) / f"translated_{self.study_language(folder)}_transcript.json"
        return translated if translated.is_file() else find_transcript(Path(folder))

    @staticmethod
    def _lecture_language(folder: Path) -> str:
        try:
            meta = json.loads((Path(folder) / "metadata.json").read_text(encoding="utf-8"))
            lang = (meta.get("language") or {})
            return str(lang.get("detected") or lang.get("forced") or "")
        except (OSError, ValueError):
            return ""

    def glossary(self, folder: Path) -> list[str]:
        """Dictionary terms of the lecture's detected subject(s): used only to name topics and rank key terms."""
        try:
            from ..vocabulary import load_terms

            meta = json.loads((Path(folder) / "metadata.json").read_text(encoding="utf-8"))
            areas = (meta.get("subject") or {}).get("subjects") or []
            return load_terms(areas, "en") if areas else []
        except Exception:                            # noqa: BLE001 - a missing dictionary only makes topic names plainer
            return []

    # ----------------------------------------------------------------------------------------------------------------------- state
    def open(self, folder: Path, title: str = "") -> StudyState:
        """What is already saved for this lecture (cheap: no generation)."""
        folder = Path(folder)
        language = self.study_language(folder)
        store = StudyStore(folder, language)
        state = StudyState(folder=folder, title=title or folder.name, store=store, progress=store.progress())
        state.pack = store.load_pack()
        state.index = store.load_index()
        if state.pack and state.index is None:
            state.pack = None                        # a pack without its index cannot show sources: make it again
        if state.pack and state.index is not None and self._stale(folder, state.index):
            state.pack = None                        # made from an older version of the transcript (or of the translation): make it again
            state.warnings.append("The study material was made from an earlier version of the transcript, so it will be made again.")
        elif state.pack is not None and (state.pack.generation or {}).get("deferred"):
            state.warnings.append("This study material was made with the built-in method because a transcription was running. Press Regenerate when it has finished for the full set.")
        elif state.pack is not None and (state.pack.generation or {}).get("version") != GENERATION_VERSION:
            state.warnings.append("This study material was made by an earlier version of MELETRA. It is kept as it is; press Regenerate for flashcards and questions from the improved engine.")
        return state

    def has_transcript(self, folder: Path) -> bool:
        return find_transcript(Path(folder)) is not None

    def prepare(self, folder: Path, *, pdf_path: Optional[Path] = None, title: str = "") -> StudyState:
        """Read the sources (transcript JSON/text of the folder, the PDF given now or saved before), build and save the index.
        Raises :class:`SourceError` with a message for the student if there is nothing to study."""
        folder = Path(folder)
        state = self.open(folder, title)
        saved_pages = self._saved_pdf_pages(folder)
        transcript = self.study_transcript(folder)
        if pdf_path:
            sources = load_sources(transcript=transcript, pdf=Path(pdf_path), title=state.title)
            if sources.pdf_pages:
                self._save_pdf_pages(folder, sources.pdf_pages, Path(pdf_path).name)
        else:
            sources = load_sources(transcript=transcript, title=state.title) if transcript else LectureSources(title=state.title)
            if saved_pages:
                sources.pdf_pages = saved_pages
            if sources.empty:
                raise SourceError("There is nothing to study yet: add a transcript or a lecture PDF.")
        state.warnings = list(sources.warnings)
        state.index = build_lecture_index(sources, self.glossary(folder))
        state.index.meta.update(self._index_facts(folder))
        state.store.save_index(state.index)
        state.pack = None                            # a new index (for example slides were added) makes the old pack stale
        return state

    def _index_facts(self, folder: Path) -> dict[str, str]:
        """What the study material is tied to: the language it is written in and the version (revision) of the saved transcript it was made from."""
        language = self.study_language(folder)
        try:
            source = source_mod.load(Path(folder))
        except Exception:  # noqa: BLE001
            source = None
        return {"language": language, "source_revision": source.revision if source is not None else ""}

    def _stale(self, folder: Path, index: Any) -> bool:
        recorded = (getattr(index, "meta", None) or {}).get("source_revision") or ""
        if not recorded:
            return False                             # made before the transcript had a revision: kept
        return recorded != self._index_facts(folder)["source_revision"]

    @staticmethod
    def _saved_pdf_pages(folder: Path) -> list[str]:
        path = Path(folder) / "study" / PDF_PAGES_FILE
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return [str(p) for p in data.get("pages", [])]
        except (OSError, ValueError, AttributeError):
            return []

    @staticmethod
    def _save_pdf_pages(folder: Path, pages: list[str], name: str) -> None:
        path = Path(folder) / "study" / PDF_PAGES_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"name": name, "pages": pages}, ensure_ascii=False), encoding="utf-8")

    def attach_pages(self, folder: Path, pages: list[str], name: str) -> None:
        """v3: the text of a converted document (PDF pages, OCR, a text file) becomes the lecture's "slides" for the study material."""
        self._save_pdf_pages(Path(folder), pages, name)

    def pdf_name(self, folder: Path) -> str:
        try:
            return str(json.loads((Path(folder) / "study" / PDF_PAGES_FILE).read_text(encoding="utf-8")).get("name", ""))
        except (OSError, ValueError, AttributeError):
            return ""

    # ------------------------------------------------------------------------------------------------------------------ generation
    def generate(self, state: StudyState, *, progress: Optional[ProgressFn] = None, cancelled: Optional[Callable[[], bool]] = None,
                 only: Optional[set[str]] = None) -> StudyPack:
        """Make (or, with ``only``, re-make parts of) the study pack with the chosen provider and save it. Never raises for provider
        problems: the built-in method takes over and the pack says so."""
        if state.index is None:
            raise SourceError("The lecture has not been prepared yet.")
        s = self.settings.settings
        options = StudyOptions(language=self.study_language(state.folder), strict=s.study_source_bound, max_cards=s.study_cards,
                               max_questions=s.study_questions)
        # What the student has been studying, as a copy (a partial regeneration changes the pack in place; open() drops a stale pack from the state, not from the disk).
        earlier = StudyPack.from_dict(state.pack.to_dict()) if state.pack is not None else state.store.load_pack()
        engine = AutomaticStudyEngine(self.engine_choice(fresh=True))
        with cancel_scope(cancelled):                                      # a model call that is still waiting for its answer ends when the student cancels or the window closes
            pack = engine.generate(state.index, lecture_id=state.folder.name, title=state.title, options=options, progress=progress,
                                   cancelled=cancelled, only=only, base=state.pack if only else None)
        pack.generation["source_revision"] = str((state.index.meta or {}).get("source_revision") or "")
        state.pack = pack
        state.store.save_pack(pack)
        carried = state.progress.carry_over(earlier, pack)               # new wording of a studied card or question keeps its history; nothing is deleted
        if carried["cards"] or carried["questions"]:
            pack.generation["progress_carried"] = carried
            state.store.save_pack(pack)
            state.progress.save()
        return pack

    def evaluator(self, state: StudyState) -> Evaluator:
        """For the quiz: judges typed answers with the chosen provider; if it fails the built-in comparison takes over."""
        try:
            engine = StudyEngine(self.create_provider())
        except ProviderError:
            engine = StudyEngine()
        index = state.index

        def judge(question, answer):
            return engine.evaluate(index, question, answer, language=self.study_language(state.folder))

        return judge

    def ask(self, state: StudyState, question: str):
        """"Explain this" - answered from the lecture only."""
        try:
            engine = StudyEngine(self.create_provider())
        except ProviderError:
            engine = StudyEngine()
        return engine.explain(state.index, question, language=self.study_language(state.folder), strict=self.settings.settings.study_source_bound)
