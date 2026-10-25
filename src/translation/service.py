"""Text-only translation of a lecture that already has a source transcript - the only translation path that needs neither the recording nor the speech recogniser.

    source_transcript.json (what was heard, per segment, in the language it was spoken)
        -> translations/<target>.json (one file per target language, cached)
        -> translated_<target>_transcript.txt / .srt / .json (the files the Read / Listen / Study screens already know)

* **Direct**: every segment goes from ITS OWN language to the target (a Korean passage Korean -> German, a Hindi passage Hindi -> German); the engine (M2M100) translates between
  any two of the program's ten languages, so no pivot through English is needed and none is used. Segments already in the target language are copied.
* **Honest**: a segment whose language is not known, whose translation failed, or whose language the engine cannot handle keeps its source text and is marked ``untranslated`` /
  ``failed`` - it is never dropped and never replaced by a guess. A retry translates only those.
* **Cached**: a translation is reused while the source revision, the target language and the engine version are the same; a changed source makes it stale (never served).
* **Offline**: the engine is the local M2M100 model. Without it nothing is faked: ``TranslationUnavailable`` says what to install.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Protocol, Sequence

from ..multilingual import TRANSLATIONS_DIR
from ..multilingual import source as source_mod
from ..multilingual.scripts import script_language
from ..multilingual.source import UNKNOWN, SourceSegment, SourceTranscript
from .m2m100 import M2M100Engine, TranslationUnavailable, find_model
from .terms import TermGuard

log = logging.getLogger(__name__)

SCHEMA = 1
TARGET_LANGUAGES = ("en", "tr", "de", "hu", "ko", "zh", "hi", "nl", "fr", "es")
#: the languages the program can translate FROM (the same ten; Japanese etc. are not offered as spoken languages, but M2M100 reads them)
SOURCE_LANGUAGES = TARGET_LANGUAGES + ("ja", "ru", "uk", "ar", "fa", "el", "bg", "sr", "ur")
STATUS_TRANSLATED, STATUS_SAME, STATUS_UNTRANSLATED, STATUS_FAILED = "translated", "same_language", "untranslated", "failed"
MAX_CHUNK_CHARS = 280
BATCH = 16


class TranslationCancelled(RuntimeError):
    """The student cancelled; nothing was cached and the source transcript is untouched."""


class Engine(Protocol):
    id: str

    def translate(self, texts: Sequence[str], source: str, target: str) -> list[str]: ...


# ====================================================================================================================== text helpers
_SENTENCE_END = re.compile(r"(?<=[.!?。！？।॥])\s+|(?<=[。！？])")


def split_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Break a long segment into sentence-sized pieces (a translation model does best on sentences and has a length limit); short text stays whole."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    pieces = [p.strip() for p in _SENTENCE_END.split(text) if p and p.strip()]
    out: list[str] = []
    current = ""
    for piece in pieces:
        if len(piece) > max_chars:                         # no sentence mark: cut at the last space / comma before the limit
            if current:
                out.append(current)
                current = ""
            out.extend(_hard_cut(piece, max_chars))
            continue
        if current and len(current) + 1 + len(piece) > max_chars:
            out.append(current)
            current = piece
        else:
            current = f"{current} {piece}".strip() if current else piece
    if current:
        out.append(current)
    return out


def _hard_cut(text: str, limit: int) -> list[str]:
    out = []
    while len(text) > limit:
        cut = max(text.rfind(" ", 0, limit), text.rfind(",", 0, limit), text.rfind("、", 0, limit))
        cut = cut if cut > limit // 3 else limit
        out.append(text[:cut].strip())
        text = text[cut:].strip(" ,、")
    if text:
        out.append(text)
    return out


def join_pieces(pieces: Sequence[str], target: str) -> str:
    glue = "" if target in ("zh", "ja") else " "
    return glue.join(p.strip() for p in pieces if p and p.strip())


def effective_source_language(segment: SourceSegment) -> Optional[str]:
    """The language to translate this segment FROM: the recorded one; for a segment whose language is not known, only a language its writing system proves (Hangul -> Korean ...)."""
    if segment.language_known:
        return segment.detected_language
    return script_language(segment.source_text)


# ====================================================================================================================== the result
@dataclass
class TranslatedSegment:
    segment_id: int
    start: float
    end: float
    source_language: str
    source_text: str
    text: str
    translation_status: str
    method: str = "direct"                          # direct | copied | none
    error: str = ""
    flags: list[str] = field(default_factory=list)  # terms_not_protected: a protected medical term could not be kept through the translation of this segment

    def to_dict(self) -> dict[str, Any]:
        data = {"segment_id": self.segment_id, "start": round(self.start, 3), "end": round(self.end, 3), "source_language": self.source_language, "source_text": self.source_text,
                "text": self.text, "translation_status": self.translation_status, "method": self.method, "error": self.error or None}
        if self.flags:
            data["flags"] = list(self.flags)
        return data


@dataclass
class TranslationDoc:
    lecture_id: str
    target: str
    source_revision: str
    engine_id: str
    segments: list[TranslatedSegment]
    created_at: str = ""
    seconds: float = 0.0
    source_origin: str = "pipeline"
    warnings: list[str] = field(default_factory=list)
    schema: int = SCHEMA

    @property
    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for s in self.segments:
            counts[s.translation_status] = counts.get(s.translation_status, 0) + 1
        return counts

    @property
    def complete(self) -> bool:
        return not any(s.translation_status in (STATUS_UNTRANSLATED, STATUS_FAILED) for s in self.segments)

    @property
    def failed(self) -> int:
        """Segments whose translation failed (a new request tries only these again). Segments in a language nobody could identify are not failed: nothing can be tried for them."""
        return sum(1 for s in self.segments if s.translation_status == STATUS_FAILED)

    @property
    def status(self) -> str:
        return "complete" if self.complete else "partial"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "meletra-translation", "schema": self.schema, "lecture_id": self.lecture_id, "target_language": self.target, "source_revision": self.source_revision,
                "engine": self.engine_id, "status": self.status, "counts": self.counts, "created_at": self.created_at, "seconds": round(self.seconds, 1),
                "source_origin": self.source_origin, "warnings": self.warnings, "segments": [s.to_dict() for s in self.segments]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranslationDoc":
        segments = [TranslatedSegment(int(s["segment_id"]), float(s["start"]), float(s["end"]), str(s.get("source_language") or UNKNOWN), str(s.get("source_text") or ""),
                                      str(s.get("text") or ""), str(s.get("translation_status") or STATUS_UNTRANSLATED), str(s.get("method") or "direct"), str(s.get("error") or ""),
                                      list(s.get("flags") or []))
                    for s in data.get("segments", [])]
        return cls(str(data.get("lecture_id") or ""), str(data.get("target_language") or ""), str(data.get("source_revision") or ""), str(data.get("engine") or ""), segments,
                   str(data.get("created_at") or ""), float(data.get("seconds") or 0.0), str(data.get("source_origin") or "pipeline"), list(data.get("warnings") or []),
                   int(data.get("schema", SCHEMA)))

    def display_lines(self) -> list[str]:
        """The text a student reads: one line per segment; a segment that could not be translated shows its source text with a plain marker."""
        lines = []
        for s in self.segments:
            if s.translation_status in (STATUS_UNTRANSLATED, STATUS_FAILED):
                label = s.source_language if s.source_language != UNKNOWN else "unknown language"
                lines.append(f"[not translated - {label}] {s.source_text}".strip())
            else:
                lines.append(s.text.strip())
        return [line for line in lines if line]


class TranslationUnavailableError(TranslationUnavailable):
    """Raised before any work: the engine or a language is not available; the message says what to do."""

    def __init__(self, message: str, reason: str = "model_missing") -> None:
        super().__init__(message)
        self.reason = reason


# ====================================================================================================================== files
def translations_dir(folder: Path) -> Path:
    return Path(folder) / TRANSLATIONS_DIR


def translation_path(folder: Path, target: str) -> Path:
    return translations_dir(folder) / f"{target}.json"


def compat_stem(target: str) -> str:
    return f"translated_{target}_transcript"


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _srt_time(seconds: float) -> str:
    ms = int(round(max(seconds, 0.0) * 1000))
    h, rest = divmod(ms, 3_600_000)
    m, rest = divmod(rest, 60_000)
    s, ms = divmod(rest, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_files(folder: Path, doc: TranslationDoc) -> list[Path]:
    """``translations/<target>.json`` (the cache) and the ``translated_<target>_transcript.*`` files the other screens read."""
    import json

    folder = Path(folder)
    written: list[Path] = []
    path = translation_path(folder, doc.target)
    _keep_previous(path, doc)
    _write_atomic(path, json.dumps(doc.to_dict(), ensure_ascii=False, indent=1))
    written.append(path)
    lines = doc.display_lines()
    stem = compat_stem(doc.target)
    txt = folder / f"{stem}.txt"
    _write_atomic(txt, "\n".join(lines) + ("\n" if lines else ""))
    srt_blocks = []
    for number, s in enumerate((s for s in doc.segments if s.text.strip() or s.source_text.strip()), start=1):
        shown = s.text.strip() if s.translation_status not in (STATUS_UNTRANSLATED, STATUS_FAILED) else f"[not translated - {s.source_language}] {s.source_text.strip()}"
        srt_blocks.append(f"{number}\n{_srt_time(s.start)} --> {_srt_time(s.end)}\n{shown}\n")
    srt = folder / f"{stem}.srt"
    _write_atomic(srt, "\n".join(srt_blocks))
    js = folder / f"{stem}.json"
    _write_atomic(js, json.dumps({"language": doc.target, "source_languages": sorted({s.source_language for s in doc.segments if s.source_language != UNKNOWN}), "method": "m2m100-text",
                                  "status": doc.status, "source_revision": doc.source_revision,
                                  "note": "Translation of the source transcript (source_transcript.json), made from the saved text; the original is unchanged.",
                                  "segments": [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip() if s.translation_status not in (STATUS_UNTRANSLATED, STATUS_FAILED) else "",
                                                "source_text": s.source_text, "source_language": s.source_language, "translation_status": s.translation_status} for s in doc.segments]},
                                 ensure_ascii=False, indent=1))
    written += [txt, srt, js]
    return written


def _keep_previous(path: Path, doc: TranslationDoc) -> None:
    """A translation that is about to be replaced by one made with another engine (or from another revision) is kept in ``translations/previous/`` - earlier translations are never lost."""
    if not path.is_file():
        return
    try:
        import json

        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("engine") == doc.engine_id and old.get("source_revision") == doc.source_revision:
            return
        stamp = time.strftime("%Y%m%d-%H%M%S")
        keep = path.parent / "previous" / f"{path.stem}.{stamp}.json"
        keep.parent.mkdir(parents=True, exist_ok=True)
        keep.write_bytes(path.read_bytes())
    except (OSError, ValueError):
        log.warning("could not keep the previous translation %s", path, exc_info=True)


def read_cached(folder: Path, target: str) -> Optional[TranslationDoc]:
    import json

    path = translation_path(folder, target)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("kind") != "meletra-translation":
            return None
        return TranslationDoc.from_dict(data)
    except (OSError, ValueError, KeyError, TypeError):
        return None


# ====================================================================================================================== the service
@dataclass
class Availability:
    state: str                    # ready | cached | model_missing | model_damaged | unsupported_target | nothing_to_translate
    message: str = ""
    target: str = ""


class TranslationService:
    """``models_dirs``: folders that may hold ``translation-m2m100``; ``engine``: an object with ``id`` and ``translate`` (tests inject a fake)."""

    def __init__(self, models_dirs: Iterable[Optional[Path]] = (), engine: Optional[Engine] = None, device: str = "auto") -> None:
        self.models_dirs = [Path(p) for p in models_dirs if p]
        self._engine = engine
        self._owned = False
        self._term_guard: Optional[TermGuard] = None
        self.device = device

    # ------------------------------------------------------------------------------------------------------------------ engine
    def model_dir(self) -> Optional[Path]:
        return find_model(*self.models_dirs)

    def engine_id(self) -> str:
        if self._engine is not None:
            return str(getattr(self._engine, "id", "engine"))
        folder = self.model_dir()
        if folder is None:
            return "m2m100-ct2"
        pack = folder / "pack.json"
        version = ""
        if pack.is_file():
            try:
                import json

                version = str(json.loads(pack.read_text(encoding="utf-8")).get("pack_version") or "")
            except (OSError, ValueError):
                version = ""
        return f"m2m100-418m-ct2-int8:{version or '1'}"

    def _get_engine(self) -> Engine:
        if self._engine is not None:
            return self._engine
        folder = self.model_dir()
        if folder is None:
            raise TranslationUnavailableError("The offline translation model is not installed. Install it in Settings -> Models; nothing was changed.", "model_missing")
        engine = M2M100Engine(folder, device=self.device)
        engine.id = self.engine_id()                                   # type: ignore[attr-defined]
        try:
            engine.load()
        except TranslationUnavailable as exc:
            raise TranslationUnavailableError(f"The offline translation model could not be loaded ({exc}). Reinstall it in Settings -> Models; nothing was changed.", "model_damaged") from exc
        self._engine = engine
        self._owned = True
        return engine

    def _guard(self) -> TermGuard:
        if self._term_guard is None:
            self._term_guard = TermGuard.from_lexicon()
        return self._term_guard

    def close(self) -> None:
        """Let go of the model (its memory) after a translation; the next one loads it again. An engine handed in by the caller (tests) is left alone."""
        if self._owned:
            self._engine = None
            self._owned = False

    # ------------------------------------------------------------------------------------------------------------------ questions
    def cached(self, folder: Path, target: str, source: Optional[SourceTranscript] = None) -> Optional[TranslationDoc]:
        """The stored translation when it is still valid for the current source revision and engine version, else ``None`` (a stale translation is never returned)."""
        doc = read_cached(folder, target)
        if doc is None:
            return None
        source = source if source is not None else source_mod.load(folder)
        if source is None or doc.source_revision != source.revision:
            return None
        current = self.engine_id() if (self._engine is not None or self.model_dir() is not None) else ""
        if current and doc.engine_id != current:
            return None                                                 # made by another engine / pack version: not reused (a stored translation is only ever read as it is)
        return doc

    def stored_targets(self, folder: Path, any_engine: bool = False) -> dict[str, TranslationDoc]:
        """Every stored translation of the lecture that belongs to its current transcript, by target language. ``any_engine``: also those made by another engine / pack version
        (they are still the lecture's translations and are shown; only reusing one as the answer to a new request needs the same engine)."""
        out: dict[str, TranslationDoc] = {}
        source = source_mod.load(folder)
        if source is None or not translations_dir(folder).is_dir():
            return out
        order = {code: i for i, code in enumerate(TARGET_LANGUAGES)}                                  # the Output Language list's order: English, Turkish, German ...
        for path in sorted(translations_dir(folder).glob("*.json"), key=lambda item: (order.get(item.stem, 99), item.stem)):
            if any_engine:
                doc = read_cached(folder, path.stem)
                if doc is not None and doc.source_revision == source.revision:
                    out[path.stem] = doc
                continue
            doc = self.cached(folder, path.stem, source)
            if doc is not None:
                out[path.stem] = doc
        return out

    def availability(self, folder: Path, target: str) -> Availability:
        if target not in TARGET_LANGUAGES:
            return Availability("unsupported_target", f"{target} is not one of the Output Languages.", target)
        source = source_mod.load(folder)
        if source is None:
            return Availability("nothing_to_translate", "This lecture has no saved transcript to translate.", target)
        if self.cached(folder, target, source) is not None:
            return Availability("cached", "", target)
        if self._engine is None:
            if self.model_dir() is None:
                return Availability("model_missing", "The offline translation model is not installed.", target)
        return Availability("ready", "", target)

    # ------------------------------------------------------------------------------------------------------------------ the work
    def translate(self, folder: Path, target: str, progress: Optional[Callable[[float], None]] = None, should_cancel: Optional[Callable[[], bool]] = None,
                  retry_failed: bool = False, force: bool = False) -> TranslationDoc:
        """Translate the saved transcript of ``folder`` into ``target`` and store the result. Never reads the recording, never runs the speech recogniser."""
        folder = Path(folder)
        if target not in TARGET_LANGUAGES:
            raise TranslationUnavailableError(f"{target} is not one of the Output Languages.", "unsupported_target")
        source = source_mod.ensure_stored(folder)
        if source is None:
            raise TranslationUnavailableError("This lecture has no saved transcript to translate.", "nothing_to_translate")
        previous = None if force else read_cached(folder, target)
        if previous is not None and previous.source_revision != source.revision:
            previous = None                                             # stale: made from an older transcript
        if previous is not None and not previous.failed and not retry_failed:
            return previous                                             # everything that can be translated is: the answer is the stored one
        engine = self._get_engine()
        engine_id = str(getattr(engine, "id", self.engine_id()))
        started = time.perf_counter()
        kept = {s.segment_id: s for s in (previous.segments if previous is not None else []) if s.translation_status in (STATUS_TRANSLATED, STATUS_SAME)}
        work: list[tuple[SourceSegment, Optional[str]]] = []
        for segment in source.segments:
            if not segment.source_text.strip():
                continue
            if segment.segment_id in kept:
                continue
            work.append((segment, effective_source_language(segment)))
        done: dict[int, TranslatedSegment] = dict(kept)
        # same language: copied. unknown / unsupported: kept as source text, marked. the rest: grouped by source language, translated in batches
        groups: dict[str, list[SourceSegment]] = {}
        for segment, language in work:
            if language == target:
                done[segment.segment_id] = TranslatedSegment(segment.segment_id, segment.start, segment.end, language, segment.source_text, segment.source_text.strip(), STATUS_SAME, "copied")
            elif language is None:
                done[segment.segment_id] = TranslatedSegment(segment.segment_id, segment.start, segment.end, UNKNOWN, segment.source_text, "", STATUS_UNTRANSLATED, "none",
                                                              "the language of this segment is not known")
            elif language not in SOURCE_LANGUAGES:
                done[segment.segment_id] = TranslatedSegment(segment.segment_id, segment.start, segment.end, language, segment.source_text, "", STATUS_UNTRANSLATED, "none",
                                                              f"the translation model does not read {language}")
            else:
                groups.setdefault(language, []).append(segment)
        total = sum(len(v) for v in groups.values()) or 1
        translated_count = 0
        guard = self._guard()
        for language, segments in groups.items():
            for start in range(0, len(segments), BATCH):
                if should_cancel is not None and should_cancel():
                    raise TranslationCancelled("Translation cancelled; nothing was stored.")
                batch = segments[start:start + BATCH]
                pieces: list[str] = []
                owner: list[int] = []
                for index, segment in enumerate(batch):
                    for piece in split_text(segment.source_text):
                        pieces.append(piece)
                        owner.append(index)
                unprotected: set[int] = set()
                try:
                    marked = [guard.protect(piece) for piece in pieces]
                    results = engine.translate([m[0] for m in marked], language, target) if pieces else []
                    if len(results) != len(pieces):
                        raise RuntimeError("the engine returned a different number of sentences")
                    restored = [TermGuard.restore(text, marked[k][1]) for k, text in enumerate(results)]
                    results = [text for text, _ok in restored]
                    again = [k for k, (_t, ok) in enumerate(restored) if not ok]
                    if again:                                           # a marker did not come back: those sentences are translated again as they are
                        redo = engine.translate([pieces[k] for k in again], language, target)
                        if len(redo) != len(again):
                            raise RuntimeError("the engine returned a different number of sentences")
                        for k, text in zip(again, redo):
                            results[k] = text
                            unprotected.add(owner[k])
                    error = ""
                except TranslationCancelled:
                    raise
                except Exception as exc:  # noqa: BLE001 - one failed batch must not lose the others; the segments are marked, not dropped
                    log.warning("translation batch %s -> %s failed: %s", language, target, exc, exc_info=True)
                    results, error = [], f"{type(exc).__name__}: {str(exc)[:200]}"
                for index, segment in enumerate(batch):
                    mine = [results[i] for i, o in enumerate(owner) if o == index] if results else []
                    text = join_pieces(mine, target)
                    if error or not text:
                        done[segment.segment_id] = TranslatedSegment(segment.segment_id, segment.start, segment.end, language, segment.source_text, "", STATUS_FAILED, "direct",
                                                                      error or "the engine returned no text")
                    else:
                        done[segment.segment_id] = TranslatedSegment(segment.segment_id, segment.start, segment.end, language, segment.source_text, text, STATUS_TRANSLATED, "direct",
                                                                      flags=["terms_not_protected"] if index in unprotected else [])
                translated_count += len(batch)
                if progress is not None:
                    progress(min(translated_count / total, 1.0))
        ordered = [done[s.segment_id] for s in source.segments if s.segment_id in done]
        warnings = []
        if source.origin == source_mod.ORIGIN_LEGACY:
            warnings.append("This lecture was made by an earlier version: its saved transcript is in one language (" + ", ".join(source.detected_languages or [UNKNOWN]) +
                            "). Speech in other languages that was not recognised then is not in it; use Reprocess Original Audio to recover it.")
        if source.language_counts and set(source.language_counts) == {"en"} and target != "en" and source.origin == source_mod.ORIGIN_LEGACY:
            warnings.append("This is a translation of an existing English transcript, not of the original speech.")
        doc = TranslationDoc(lecture_id=source.lecture_id, target=target, source_revision=source.revision, engine_id=engine_id, segments=ordered,
                             created_at=time.strftime("%Y-%m-%dT%H:%M:%S"), seconds=time.perf_counter() - started, source_origin=source.origin, warnings=warnings)
        write_files(folder, doc)
        source_mod.record_translation(folder, target, engine_id, {s.segment_id: s.translation_status for s in ordered})
        return doc
