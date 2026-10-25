"""End-to-end processing of media files.

extract audio -> independent VAD pass -> probe (language + early text) -> medical context (19-area subject scores,
term priority, acronyms, PDF) -> chunked decode with per-chunk checkpoints and rolling context -> stitch ->
uncertainty / hallucination flags -> clean transcript -> medical analysis (terms, possible errors, entities,
acronyms, drug/gene/protein names) -> optional Deepgram second opinion -> export.

The raw ASR text is never edited by any step after the decode; every medical analysis is a separate report.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from . import __version__
from . import audio as audio_mod
from . import language as lang_mod
from . import quality as quality_mod
from . import second_opinion as second_mod
from . import whisperx_extra
from .acronyms import AcronymEngine, acronyms_report_text, disambiguate
from .budget import TokenBudgets, compute_token_budgets
from .checkpoint import SegmentCheckpoint, file_sha256, make_fingerprint
from .chunked_decode import ChunkedDecoder, ChunkFailedError, DecodeOutcome, PauseRequested
from .cleaner import clean_segments
from .context_engine import MedicalContextEngine
from .diagnostics import GpuMemoryMonitor, environment_summary, query_nvidia_smi, resolve_device, vram_warnings
from .events import Emitter
from .runstats import SessionStats, load_sessions, now_iso, record_session, timing_report
from .entities import build_entities, entities_summary
from .exporters import (
    OutputPaths, export_all, make_output_paths, possible_errors_text, terms_report_text, timestamped_text, write_json,
    write_text,
)
from .guard import HOTWORD_GUARD_NOTE, apply_hotword_guard  # noqa: F401 - re-exported (scripts/benchmark.py, tests)
from .long_audio import Chunk, ChunkStore, plan_chunks, stitch
from .medical import integration as medical_integration
from .translation import hook as translation_hook
from .multilingual import finalize as ml_finalize
from .multilingual import history as ml_history
from .multilingual import leveler as ml_leveler
from .multilingual import pipeline_hook as ml_hook
from .multilingual import recheck as ml_recheck
from .multilingual import source as ml_source
from .multilingual import timeline as ml_timeline
from .medical_report import build_report, term_subject_map
from .model_manager import DecodeRequest, ModelSpec, get_model_spec, load_model_specs
from .name_report import name_report_text, scan_name_variants
from .pdf_context import LectureContext, build_context, find_companion_pdf
from .schema import Segment, TranscriptionResult
from .terminology_tr import build_language_index
from .topic import detect_topic
from .transcriber import ProgressReporter, Transcriber, format_clock
from .uncertainty import CONFIDENCE_METHOD, UncertaintyConfig, annotate_segments
from .classifier import SubjectClassifier
from .knowledge_subjects import AREA_VOCAB_FILES, AREAS
from .vocabulary import (
    VocabularyIndex, available_subjects, build_hotwords, build_index, load_terms, parse_subjects,
)

log = logging.getLogger(__name__)

PRIVACY_NOTE = (
    "All processing ran locally. Audio, video and PDF files were not sent anywhere. The completed transcript text is shared with MELETRA only when "
    "'Share completed transcripts' is switched on in Settings -> Privacy & Data (off by default); that happens after this transcript is saved and is not part of the pipeline."
)
BACKENDS = ("local", "deepgram", "compare")
ENGINES = ("expert", "legacy")


@dataclass
class RunOptions:
    mode: str = "quality"
    language: Optional[str] = None            # None / "auto" => auto-detect
    subjects: list[str] = field(default_factory=lambda: ["general_medical"])
    model_key: str = "default"
    device: str = "auto"                      # auto (CUDA if it works, else CPU) | cuda | cpu
    device_index: int = 0
    preprocessing: str = "none"
    multilingual: bool = False
    use_pdf: bool = True
    pdf_path: Optional[Path] = None
    keep_audio: bool = False
    reuse_audio: Optional[Path] = None        # an already extracted 16 kHz mono WAV of the same media (skips FFmpeg)
    offline: bool = False
    outputs_root: Optional[Path] = None
    use_hotwords: bool = True                 # --no-hotwords: A/B comparison without term biasing
    align: bool = False                       # optional WhisperX forced alignment (separate venv)
    diarize: bool = False                     # optional speaker diarization (needs HF_TOKEN), OFF by default
    whisperx_python: Optional[Path] = None
    auto_topic: Optional[bool] = None         # probe the first minutes (subject detection); None -> config topic.auto
    probe_minutes: Optional[float] = None     # None -> config topic.probe_minutes
    probe_only: bool = False                  # run the probe, save the plan, stop (the `probe` command)
    resume: bool = True                       # continue from the checkpoint of an interrupted run
    keep_checkpoint: bool = False
    quality_audit: bool = True                # independent VAD pass + transcript audit (quality_report.json)
    context_engine: Optional[str] = None      # expert | legacy; None -> config context_engine.engine
    chunk_minutes: Optional[float] = None     # None -> config long_audio.chunk_minutes
    backend: str = "local"                    # local (default) | deepgram | compare
    deepgram_if_suspect: bool = False         # ask Deepgram only if the local transcript looks doubtful
    # --- desktop-app hooks (all optional; the command line leaves them alone)
    events: Optional[Callable[[dict[str, Any]], None]] = None      # receives every structured job event (src/events.py)
    should_stop: Optional[Callable[[], bool]] = None               # True = "pause safely"; polled after every saved segment
    model_spec: Optional[ModelSpec] = None                         # a model chosen by the app (local folder) instead of models.yaml
    checkpoint_root: Optional[Path] = None                         # keep checkpoints below this folder (<root>/<job key>)
    temp_root: Optional[Path] = None                               # where the extracted 16 kHz WAV goes
    deepgram_api_key: Optional[str] = None                         # from Windows Credential Manager; never logged or written
    cpu_threads: int = 0                                           # 0 = library default
    output_name: Optional[str] = None                              # sub-folder name for a single file (the app avoids clashes)
    medical_expert: bool = True                                    # medical expert pass: soft language lock + medical corrected transcript
    custom_medical_terms: list[str] = field(default_factory=list)  # Settings > Medical Vocabulary > Add Terms
    medical_intelligence: Optional[str] = None                     # v4 layer: off | shadow | on (None = MLT_V4_MODE, then off)
    output_language: Optional[str] = None                          # the language of the final transcript (None = the spoken language, no translation step)
    translation_model_dirs: list[Path] = field(default_factory=list)   # folders that may hold translation-m2m100 (the optional offline translation model)


@dataclass
class ProcessingSummary:
    media: Path
    output_dir: Optional[Path] = None
    ok: bool = False
    error: Optional[str] = None
    segments: int = 0
    uncertain: int = 0
    audio_seconds: float = 0.0
    wall_seconds: float = 0.0
    real_time_factor: Optional[float] = None
    language: Optional[str] = None
    compute_type: Optional[str] = None
    fallback_used: bool = False
    peak_vram_mib: Optional[int] = None
    peak_dedicated_mib: Optional[int] = None
    peak_shared_mib: Optional[int] = None
    words: int = 0
    terms_detected: int = 0
    possible_errors: int = 0
    hallucination_flags: int = 0
    repetition_flags: int = 0
    timestamp_issues: int = 0
    resumed_segments: int = 0
    chunks: int = 1
    acronyms: int = 0
    entities: int = 0
    deepgram: str = "not requested"
    paused: bool = False
    state: str = ""                       # completed | paused | failed
    recovered_gaps: int = 0               # stretches of real speech that came back without text and were re-decoded
    recovered_gap_seconds: float = 0.0
    loop_repairs: int = 0
    device: str = ""
    timing: dict[str, Any] = field(default_factory=dict)
    translation: dict[str, Any] = field(default_factory=dict)      # what was done for the Output Language (src/translation): method, languages, files
    languages: list[str] = field(default_factory=list)             # every language recognised in the recording (from its source transcript)
    translations: list[str] = field(default_factory=list)          # the Output Languages that exist as saved translations


@dataclass
class RunPlan:
    """Decisions taken before the main decode; stored in the checkpoint so a resume reuses them."""

    language: Optional[str]                   # code forced for the main decode (None = Whisper auto-detects)
    language_source: str                      # user | model_default | probe | auto
    language_probability: Optional[float] = None
    subjects: list[str] = field(default_factory=list)             # selected areas (expert) or subject files (legacy)
    hotword_candidates: list[str] = field(default_factory=list)   # priority-ordered, before the token budget
    pdf_terms: list[str] = field(default_factory=list)
    heard_terms: list[str] = field(default_factory=list)          # dictionary terms heard in the probe
    topic: Optional[dict[str, Any]] = None
    language_mix: Optional[dict[str, Any]] = None
    probe_seconds: float = 0.0
    probe_start: float = 0.0                  # where the probe began (first sustained speech, not 0:00)
    probe_excerpt: str = ""
    pdf_info: Optional[dict[str, Any]] = None
    engine: str = "expert"
    context: Optional[dict[str, Any]] = None  # MedicalContext.to_dict() right after the build
    budgets: Optional[dict[str, Any]] = None
    chunks: list[dict[str, float]] = field(default_factory=list)
    multilingual: bool = False                # decode every 30 s window in its own language (mixed-language recording)
    language_dominant: Optional[str] = None   # the language most of a mixed recording is in (vocabulary, reports)
    language_samples: Optional[dict[str, Any]] = None   # detection on windows spread over the whole file
    hotwords_off_reason: str = ""             # why no hotwords are sent ("" = they are allowed)
    language_plan: Optional[dict[str, Any]] = None          # Auto Detect over time: the language runs (src/multilingual/timeline.LanguagePlan.to_dict) - decoded run by run
    language_scan: Optional[dict[str, Any]] = None          # what the scan / verification did and why (for the metadata)

    @property
    def term_language(self) -> Optional[str]:
        """Language used for vocabularies and term analysis: the forced one, else the dominant one of a mixed recording."""
        return self.language or self.language_dominant

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunPlan":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@contextmanager
def file_logging(path: Path) -> Iterator[None]:
    """Mirror all log records into ``processing.log`` for the duration of one file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    handler.setLevel(logging.DEBUG)
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        yield
    finally:
        root.removeHandler(handler)
        handler.close()


OUTPUT_MARKER = ".job_id"


def mark_output_owner(folder: Path, media: Path) -> None:
    """Remember which recording a results folder belongs to, so another file with the same name never overwrites it."""
    marker = Path(folder) / OUTPUT_MARKER
    try:
        if not marker.exists():
            marker.write_text(job_key(media), encoding="utf-8")
    except OSError:
        log.debug("could not write %s", marker, exc_info=True)


def output_name_for(media: Path, batch: list[Path]) -> str:
    """Folder name for ``media``; disambiguate ``a.mp3`` / ``a.mp4`` in the same batch."""
    clashes = [m for m in batch if m.stem.lower() == media.stem.lower()]
    return media.stem if len(clashes) <= 1 else f"{media.stem}_{media.suffix.lstrip('.').lower()}"


def _effective_profile(profile: dict[str, Any], adapter_caps: Any) -> tuple[dict[str, Any], list[str]]:
    """Gracefully disable profile features the backend cannot do."""
    effective, notes = dict(profile), []
    if effective.get("vad_filter") and not adapter_caps.vad:
        effective["vad_filter"] = False
        notes.append("VAD disabled: backend has no built-in VAD")
    if effective.get("word_timestamps") and not adapter_caps.word_timestamps:
        effective["word_timestamps"] = False
        notes.append("word timestamps disabled: not supported by this backend")
    if effective.get("use_hotwords") and not adapter_caps.hotwords:
        effective["use_hotwords"] = False
        notes.append("hotwords disabled: not enabled/supported by this backend (see models.yaml options.prompt_hotwords)")
    return effective, notes


# ---------------------------------------------------------------------------------------------- settings
def _auto_topic(options: RunOptions, cfg: dict[str, Any]) -> bool:
    if options.auto_topic is not None:
        return bool(options.auto_topic)
    return bool(cfg.get("topic", {}).get("auto", False))


def _engine_name(options: RunOptions, cfg: dict[str, Any]) -> str:
    name = str(options.context_engine or cfg.get("context_engine", {}).get("engine", "expert")).lower()
    if name not in ENGINES:
        raise ValueError(f"unknown context engine '{name}' (use one of: {', '.join(ENGINES)})")
    return name


def _user_subjects(options: RunOptions) -> list[str]:
    """Subjects the user actually asked for (the default 'general_medical' is a baseline, not a request)."""
    return [s for s in options.subjects if s != "general_medical"]


def _chunk_settings(options: RunOptions, cfg: dict[str, Any]) -> dict[str, float]:
    long_cfg = cfg.get("long_audio", {})
    enabled = bool(long_cfg.get("enabled", True))
    minutes = float(options.chunk_minutes or long_cfg.get("chunk_minutes", 15.0))
    return {"enabled": enabled, "chunk_minutes": minutes if enabled else 1e6,
            "overlap_seconds": float(long_cfg.get("overlap_seconds", 3.0)),
            "snap_window_seconds": float(long_cfg.get("snap_window_seconds", 60.0)),
            "min_pause_seconds": float(long_cfg.get("min_pause_seconds", 0.6)),
            "retries": int(long_cfg.get("chunk_retries", 2))}


def _jsonable(value: Any) -> Any:
    """Round-trip through JSON so the fingerprint compares equal after being read back from disk."""
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def _fingerprint(source_sha: str, options: RunOptions, cfg: dict[str, Any], spec: ModelSpec, pdf_sha: Optional[str] = None
                 ) -> dict[str, Any]:
    profile = cfg[options.mode]
    engine = _engine_name(options, cfg)
    budgets = compute_token_budgets(cfg["context"])
    chunking = _chunk_settings(options, cfg)
    return _jsonable(make_fingerprint(
        source_sha, model=str(spec.model), mode=options.mode, language=options.language, subjects=options.subjects,
        auto_topic=_auto_topic(options, cfg), use_hotwords=options.use_hotwords, preprocessing=options.preprocessing,
        beam_size=profile["beam_size"], temperature=profile["temperature"], vad=cfg["vad"], engine=engine,
        token_budgets=[budgets.prompt, budgets.hotwords],
        context_engine=cfg.get("context_engine") if engine == "expert" else None,
        chunking={k: chunking[k] for k in ("chunk_minutes", "overlap_seconds", "snap_window_seconds", "min_pause_seconds")},
        pdf=pdf_sha, pdf_enabled=bool(options.use_pdf)))


# ---------------------------------------------------------------------------------------------- context
def load_pdf_context(media: Path, options: RunOptions, cfg: dict[str, Any], vocab: VocabularyIndex) -> LectureContext:
    """Terms from the lecture PDF next to the media file (empty context if none / disabled / unreadable)."""
    ctx_cfg = cfg["context"]
    if not (options.use_pdf and ctx_cfg.get("pdf_enabled", True)):
        return LectureContext()
    pdf = options.pdf_path or find_companion_pdf(media)
    if pdf is None:
        return LectureContext()
    log.info("Lecture PDF found: %s", pdf.name)
    context = build_context(pdf, vocab, ctx_cfg.get("pdf_min_frequency", 2), ctx_cfg.get("pdf_max_terms", 25))
    if context.error:
        log.warning("PDF context: %s", context.error)
    return context


def build_hotword_candidates(
    subjects: list[str], language: Optional[str], pdf_terms: list[str], heard_terms: list[str], cfg: dict[str, Any],
) -> list[str]:
    """LEGACY hotword candidates (``context_engine.engine: legacy``): PDF terms, terms heard in the probe, then the
    chosen subjects' terms with multi-word terms first. The expert engine scores terms instead (src/priority.py)."""
    ctx = cfg["context"]
    terms_dir = Path(cfg["paths"]["medical_terms_abs"])
    subject_terms = load_terms(_vocab_files(subjects, terms_dir), language, terms_dir)
    return build_hotwords(
        subject_terms, priority_terms=[*pdf_terms, *heard_terms],
        max_terms=int(ctx.get("max_hotwords", 40)), max_chars=int(ctx.get("max_hotword_chars", 700)),
        multiword_first=bool(ctx.get("multiword_priority", True)),
    )


def _run_whisperx(options: RunOptions, result: Any, wav: Path) -> dict[str, Any]:
    """Optional alignment / diarization. Failures are recorded, never raised."""
    info: dict[str, Any] = {"align_requested": options.align or options.diarize,
                            "diarize_requested": options.diarize, "status": "not requested", "notes": []}
    if not (options.align or options.diarize):
        return info
    try:
        info["notes"] = whisperx_extra.refine_segments(
            result.segments, wav, result.language, diarize=options.diarize, device=options.device,
            python=options.whisperx_python,
        )
        info["status"] = "ok"
    except whisperx_extra.WhisperXUnavailableError as exc:
        info.update(status="failed", error=str(exc))
        log.warning("WhisperX step skipped: %s", exc)
    return info


def make_plan(
    media: Path, wav: Path, duration: float, options: RunOptions, cfg: dict[str, Any], transcriber: Transcriber,
    spec: ModelSpec, base_profile: dict[str, Any], vocab_all: VocabularyIndex,
    speech: Optional[list[tuple[float, float]]] = None, pdf: Optional[LectureContext] = None, emit: Optional[Emitter] = None, activity: Optional[Any] = None,
) -> tuple[RunPlan, Optional[MedicalContextEngine], AcronymEngine]:
    """Decide language, subject and hotword candidates (probing the first minutes when asked).

    Returns ``(plan, context_engine_or_None, acronym_engine)``; the engine is ``None`` for ``engine: legacy``.
    """
    decision = lang_mod.decide(options.language, spec.default_language)
    language, source = decision.forced, ("user" if options.language not in (None, "auto") else
                                         ("model_default" if decision.forced else "auto"))
    pdf = pdf if pdf is not None else load_pdf_context(media, options, cfg, vocab_all)
    engine_name = _engine_name(options, cfg)
    plan = RunPlan(
        language=language, language_source=source, subjects=list(options.subjects), pdf_terms=list(pdf.terms),
        pdf_info={"source": pdf.source.name if pdf.source else None, "terms": pdf.terms, "pages": pdf.pages,
                  "error": pdf.error, "acronyms": pdf.acronyms, "acronym_tokens": pdf.acronym_tokens,
                  "rare_terms": pdf.rare_terms, "categories": pdf.categories},
        engine=engine_name,
    )
    terms_dir = Path(cfg["paths"]["medical_terms_abs"])
    topic_cfg = cfg.get("topic", {})
    probe_text = ""
    ml_plan: Optional[Any] = None                                  # src/multilingual LanguagePlan when the recording changes language / is sung (None = one language)
    sampled: Optional[dict[str, Any]] = None
    if language is None and getattr(spec, "key", "") != "turkish_medical":
        starts = lang_mod.sampling_starts(speech, duration)

        def sampled_so_far(done: int, total: int) -> None:                 # real progress of the silent language analysis
            if emit is not None:
                emit("stage", stage="language", percent=round(100.0 * done / max(total, 1), 1), done=done, total=total)

        coarse = None
        if hasattr(getattr(transcriber, "adapter", None), "detect_language_probs"):
            coarse = transcriber.detect_language_windows(wav, starts, lang_mod.SAMPLE_SECONDS, sampled_so_far, options.should_stop)
            samples = [(s, max(p, key=p.get), p[max(p, key=p.get)]) for s, p in coarse if p]        # the detector's best guess per window, as before
        else:
            samples = transcriber.detect_languages(wav, starts, lang_mod.SAMPLE_SECONDS, sampled_so_far, options.should_stop)
        if samples and coarse is not None and spec.default_language is None:
            ml_plan, ml_report = ml_hook.plan_recording(transcriber=transcriber, wav=wav, duration=duration, speech=speech or [], coarse=coarse, cfg=cfg, emit=emit,
                                                        should_stop=options.should_stop, activity=activity)
            plan.language_scan = ml_report
        if samples:
            first_speech = transcriber.detect_languages(wav, [quality_mod.find_speech_onset(speech) if speech else 0.0],
                                                        lang_mod.SAMPLE_SECONDS)
            sampled = lang_mod.summarize_language_samples(
                samples, probe=(first_speech[0][1], first_speech[0][2]) if first_speech else None)
            if options.multilingual:                          # "Mixed languages" chosen by the user
                sampled["mixed"], sampled["reason"] = True, "requested: mixed languages"
            elif options.medical_expert:                      # medical expert: soft language lock (accented speech is not "mixed")
                medical_decode_window, _ = medical_integration.make_decoders(transcriber, wav, base_profile, cfg, duration, DecodeRequest)
                medical_integration.soft_lock(sampled, medical_decode_window, on_cpu=transcriber.adapter.device == "cpu",
                                              should_stop=options.should_stop,
                                              probe=(first_speech[0][1], first_speech[0][2]) if first_speech else None)
            plan.language_samples = sampled
            log.info("LANGUAGE SAMPLING: %d windows across the recording -> %s (dominant %s, %.0f%%)%s", sampled["windows"],
                     sampled["counts"], sampled["dominant"], 100 * sampled["dominant_share"],
                     " | MIXED: " + sampled["reason"] if sampled["mixed"] else "")
    if sampled and sampled["dominant"]:
        dominant = sampled["dominant"]
        dominant_probs = [p for _s, lang, p in sampled["samples"] if lang == dominant]
        plan.language_probability = round(sum(dominant_probs) / max(len(dominant_probs), 1), 4)
        if sampled["mixed"]:
            plan.multilingual, plan.language_dominant, plan.language, plan.language_source = True, dominant, None, "mixed"
        else:
            language, plan.language, plan.language_source = dominant, dominant, "sampled"
        if options.medical_expert and sampled.get("soft_lock"):      # medical expert: confidence of the soft-locked language
            plan.language_probability = round(float(sampled["soft_lock"]["confidence"]), 4)
    if ml_plan is not None and ml_plan.mode != "single":           # Auto Detect over time: more than one language (or singing): decode run by run, each in its own language
        plan.multilingual, plan.language_dominant, plan.language, plan.language_source = True, ml_plan.languages[0], None, "timeline"
        plan.language_probability = None
        plan.language_plan = ml_plan.to_dict()
        language = ml_plan.languages[0]                              # the subject probe below reads the dominant language
        log.info("LANGUAGE PLAN: %s -> decoded run by run (%s)", ml_plan.mode, ", ".join(f"{r.language} {r.start:.0f}-{r.end:.0f}s" for r in ml_plan.runs[:12]) + (" ..." if len(ml_plan.runs) > 12 else ""))
    if _auto_topic(options, cfg) and not (ml_plan is not None and ml_plan.mode == "music"):          # the subject of a song is not a lecture subject: no probe
        wanted = (options.probe_minutes or float(topic_cfg.get("probe_minutes", 4.0))) * 60.0
        onset = quality_mod.find_speech_onset(speech) if speech else 0.0
        if ml_plan is not None and ml_plan.mode != "single":                     # the subject is judged where the dominant language is spoken
            first_main = next((r.start for r in ml_plan.runs if r.language == ml_plan.languages[0] and r.vad == "on"), None)
            onset = max(onset, first_main) if first_main is not None else onset
        elif sampled and sampled["mixed"]:                  # the subject is judged where the main language is spoken
            first_main = next((s for s, lang, p in sampled["samples"] if lang == sampled["dominant"] and p >= 0.9), None)
            onset = first_main if first_main is not None else onset
            language = sampled["dominant"]
        probe_seconds = min(wanted, max(30.0, (duration - onset) * 0.4))      # never decode most of a short file twice
        # conditioning off: measured on the real recording, conditioning made the probe loop near the end of its window
        # (a looping probe is slow and over-counts the repeated words in the subject scores)
        probe_profile = {**base_profile, "word_timestamps": False, "use_hotwords": False,
                         "condition_on_previous_text": False}
        log.info("PROBE: transcribing %.1f min starting at %s (first sustained speech; no hotwords) to detect "
                 "language and subject ...", probe_seconds / 60.0, format_clock(onset))
        last_report = [0.0]

        def probe_progress(processed: float, _total: float) -> None:        # real progress of the first-minutes analysis, and a prompt pause
            if options.should_stop is not None and options.should_stop():
                raise PauseRequested()
            if emit is not None and time.monotonic() - last_report[0] >= 0.5:
                last_report[0] = time.monotonic()
                emit("stage", stage="analysis", percent=round(min(100.0, 100.0 * max(processed - onset, 0.0) / probe_seconds), 1))

        if emit is not None:
            emit("stage", stage="analysis", percent=0.0)
        probe = transcriber.transcribe(DecodeRequest(
            audio_path=wav, language=language, profile=probe_profile, decoding=cfg["decoding"],
            vad_parameters=cfg["vad"], start_offset=onset, max_seconds=probe_seconds, total_duration=duration,
            progress=probe_progress,
        ))
        probe_text = " ".join(seg.text.strip() for seg in probe.segments)
        plan.probe_seconds, plan.probe_start, plan.probe_excerpt = probe_seconds, onset, probe_text[:1500]
        plan.language_mix = lang_mod.estimate_language_mix(probe_text)
        if language is None and probe.language and (probe.language_probability or 0.0) >= 0.8:
            plan.language, plan.language_source = probe.language, "probe"
            plan.language_probability = probe.language_probability
        elif language is None:
            plan.language_probability = probe.language_probability
    if options.multilingual and not plan.multilingual:
        plan.multilingual, plan.language_dominant, plan.language_source = True, plan.language or plan.language_dominant, "mixed"
        plan.language = None
    plan.hotwords_off_reason = ""

    acronyms = AcronymEngine()
    if pdf.text:
        acronyms.learn(pdf.text, "defined in slides")
    if probe_text:
        acronyms.learn(probe_text, "defined in speech")

    if engine_name == "legacy":
        if probe_text:
            topic = detect_topic(
                probe_text, plan.language, terms_dir, max_subjects=int(topic_cfg.get("max_subjects", 2)),
                relative_cutoff=float(topic_cfg.get("relative_cutoff", 0.5)),
                min_score=float(topic_cfg.get("min_score", 4.0)))
            plan.topic, plan.subjects, plan.heard_terms = topic.to_dict(), topic.selected, topic.matched_terms
        plan.hotword_candidates = build_hotword_candidates(plan.subjects, plan.term_language, plan.pdf_terms, plan.heard_terms, cfg)
        _gate_hotwords(plan, options)
        log.info("PLAN (legacy engine): language=%s (%s) | subjects=%s | %d hotword candidates", plan.language,
                 plan.language_source, plan.subjects, len(plan.hotword_candidates))
        return plan, None, acronyms

    engine = MedicalContextEngine(
        cfg, terms_dir, plan.term_language, user_subjects=_user_subjects(options), filename=media.stem,
        pdf_terms=pdf.terms, pdf_text=pdf.text)
    context = engine.build(probe_text)
    engine.add_acronyms([{"acronym": a, "expansion": e, "occurrences": 1} for a, (e, _) in acronyms.learned.items()])
    limit = int(cfg.get("context_engine", {}).get("candidate_limit", 60))
    plan.subjects = context.selected_areas or _user_subjects(options) or ["general_medical"]
    plan.heard_terms = engine.heard_terms()
    plan.hotword_candidates = context.hotword_candidates(limit)
    plan.context = context.to_dict()
    plan.topic = {"selected": context.selected_areas, "confident": context.confident,
                  "scores": {a.area: round(a.score, 3) for a in context.area_scores[:6] if a.score > 0},
                  "evidence": {a.area: a.evidence[:6] for a in context.selected}}
    _gate_hotwords(plan, options)
    log.info("PLAN (expert engine): language=%s (%s) | areas=%s confident=%s | %d candidates, top: %s",
             plan.language or f"mixed ({plan.language_dominant})", plan.language_source, plan.subjects, context.confident,
             len(plan.hotword_candidates), ", ".join(plan.hotword_candidates[:6]))
    return plan, engine, acronyms


def _language_plan_tuples(plan: RunPlan, transcriber: Any) -> Optional[list[tuple[float, float, str, str]]]:
    """The language runs of the plan as decode-request tuples; the runs' details (confidence, status, alternatives) travel to the segments through the transcriber."""
    if not plan.language_plan or not plan.language_plan.get("runs"):
        return None
    runs = plan.language_plan["runs"]
    transcriber.language_runs_meta = runs
    return [(float(r["start"]), float(r["end"]), str(r["language"]), str(r.get("vad", "on"))) for r in runs]


def _gate_hotwords(plan: RunPlan, options: RunOptions) -> None:
    """Hotwords are sent only when the context really supports them.

    Off for: a recording that mixes languages (English terms pull the other-language speech toward the wrong words), a language
    without a medical vocabulary, and a subject that could not be recognised with confidence (no --subject, no slides). Measured:
    generic hotwords were WORSE than none in both A/B benchmark sets (WER 6.7 % vs 5.3 %, 6.6 % vs 4.6 %).
    """
    reason = ""
    language = plan.term_language
    topic = plan.topic or {}
    if plan.multilingual:
        reason = ("the recording mixes languages (every window is decoded in its own language); hotwords would pull speech in the "
                  "other language toward the wrong words")
    elif language and language not in lang_mod.HOTWORD_LANGUAGES:
        reason = f"there is no medical vocabulary for language '{language}'"
    elif not (topic.get("confident") or topic.get("selected") or _user_subjects(options) or plan.pdf_terms):
        reason = "the subject could not be recognised with confidence (no subject chosen, no lecture slides)"
    plan.hotwords_off_reason = reason
    if reason:
        plan.hotword_candidates = []
        log.warning("HOTWORDS OFF: %s.", reason)


def _log_header(media: Path, duration: float, plan: RunPlan, spec: ModelSpec, transcriber: Transcriber,
                start_offset: float, chunks: int) -> None:
    bar = "=" * 78
    log.info(bar)
    log.info(" FILE         : %s", media.name)
    log.info(" DURATION     : %s", format_clock(duration))
    mix = plan.language_mix or {}
    shown = f"mixed - each window in its own language (mostly {plan.language_dominant})" if plan.multilingual \
        else (plan.language or "auto-detect")
    log.info(" LANGUAGE     : %s (%s)%s", shown, plan.language_source, "  [MIXED Turkish/English]" if mix.get("mixed") else "")
    if plan.hotwords_off_reason:
        log.info(" HOTWORDS     : off - %s", plan.hotwords_off_reason)
    log.info(" MODEL        : %s via %s", spec.model, transcriber.adapter.backend_name)
    log.info(" COMPUTE TYPE : %s", transcriber.current_compute_type)
    log.info(" SUBJECTS     : %s", ", ".join(plan.subjects) or "none")
    log.info(" CHUNKS       : %d", chunks)
    if start_offset:
        log.info(" RESUMING FROM: %s (earlier work is kept)", format_clock(start_offset))
    log.info(bar)


def job_key(media: Path) -> str:
    """Stable id of a source file for its checkpoint folder (path + size; the checkpoint itself also carries the SHA-256)."""
    resolved = Path(media).resolve()
    return hashlib.sha1(f"{str(resolved).lower()}|{resolved.stat().st_size}".encode("utf-8")).hexdigest()[:16]


def checkpoint_dir_for(options: RunOptions, media: Path) -> Optional[Path]:
    return Path(options.checkpoint_root) / job_key(media) if options.checkpoint_root else None


def _wav_seconds(wav: Path) -> float:
    import soundfile as sf

    info = sf.info(str(wav))
    return info.frames / float(info.samplerate)


def _empty_result(spec: ModelSpec, transcriber: Transcriber, plan: RunPlan, duration: float) -> TranscriptionResult:
    return TranscriptionResult(
        segments=[], language=plan.language, language_probability=plan.language_probability, duration=duration,
        model=str(spec.model), backend=transcriber.adapter.backend_name, compute_type=transcriber.current_compute_type)


def _save_partial(paths: OutputPaths, segments: list[Segment], checkpoint: Optional[SegmentCheckpoint],
                  exc: BaseException) -> None:
    """A run died: keep everything finished so far and say how to resume."""
    if checkpoint is not None:
        checkpoint.close()
    if segments:
        write_text(paths.partial, timestamped_text(segments))
        log.error("INTERRUPTED (%s). %d segments up to %s are saved in %s and in the checkpoint. "
                  "Run the SAME command again to resume.", type(exc).__name__, len(segments),
                  format_clock(segments[-1].end), paths.partial.name)
    else:
        log.error("INTERRUPTED (%s) before any segment was finished; nothing to resume.", type(exc).__name__)


def _finish_probe(summary: ProcessingSummary, paths: OutputPaths, checkpoint: Optional[SegmentCheckpoint],
                  plan: RunPlan, duration: float, started: float) -> ProcessingSummary:
    """`probe` command: persist the plan so the real run reuses it, write a human-readable summary."""
    if checkpoint is not None:
        checkpoint.save_plan(plan.to_dict())
    write_json(paths.directory / "probe_plan.json", plan.to_dict())
    write_text(paths.directory / "probe_transcript.txt", plan.probe_excerpt + "\n")
    summary.ok = True
    summary.audio_seconds = duration
    summary.language = plan.language
    summary.wall_seconds = time.perf_counter() - started
    log.info("Probe saved. The next `transcribe` run reuses this plan.")
    return summary


def _stop_point(options: RunOptions) -> None:
    """A place where a pause / cancel request is honoured before the first segment exists (nothing to save yet)."""
    if options.should_stop is not None and options.should_stop():
        raise PauseRequested()


def _recheck_languages(transcriber: Transcriber, wav: Path, segments: list[Segment], plan: RunPlan, options: RunOptions, profile: dict[str, Any], cfg: dict[str, Any],
                       duration: float, emit: Optional[Emitter]) -> tuple[list[Segment], dict[str, Any]]:
    """Each segment's language is checked against its own audio (a passage in another language is invisible to the window scan). Not for a language the student chose. A failure
    here never fails the recording."""
    if options.language not in (None, "", "auto") or str(ml_hook.settings_from(cfg).get("recheck", "auto")) == "off":
        return segments, {"checked": 0, "skipped": "the language was chosen by the student" if options.language not in (None, "", "auto") else "turned off in the settings"}
    deviating = [tuple(item) for item in ((plan.language_samples or {}).get("deviating") or [])]
    try:
        merged, report = ml_recheck.recheck_languages(transcriber=transcriber, wav=wav, segments=segments, duration=duration, profile=profile, cfg=cfg,
                                                      on_gpu=transcriber.adapter.device == "cuda", deviating=deviating, emit=emit, should_stop=options.should_stop)
    except PauseRequested:
        raise
    except Exception as exc:  # noqa: BLE001 - an aid on top of a finished transcript
        log.warning("Language check failed (%s: %s); the transcript is kept as decoded", type(exc).__name__, exc)
        return segments, {"checked": 0, "skipped": f"{type(exc).__name__}: {str(exc)[:200]}"}
    if report.get("changed"):
        log.info("LANGUAGE CHECK: %d of %d segment(s) were in another language and were read again (%s)", report["changed"], report["checked"],
                 ", ".join(f"{c['start']:.0f}s {c['from']}->{c['to']}" for c in report["changes"][:6]))
    _stop_point(options)
    return merged, report


def _recover_coverage(transcriber: Transcriber, wav: Path, segments: list[Segment], activity: Optional[Any], plan: RunPlan, options: RunOptions, profile: dict[str, Any],
                      cfg: dict[str, Any], duration: float, emit: Optional[Emitter]) -> tuple[list[Segment], dict[str, Any]]:
    """Audible stretches that no word covers are decoded again (bounded); what cannot be read is reported. A failure here never fails the recording."""
    timeline = ml_timeline.LanguagePlan.from_dict(plan.language_plan) if plan.language_plan else None
    forced = options.language if options.language not in (None, "", "auto") else plan.language
    try:
        merged, record = ml_hook.recover_coverage(transcriber=transcriber, wav=wav, segments=segments, activity=activity, plan=timeline, language=forced, profile=profile, cfg=cfg,
                                                  duration=duration, emit=emit, should_stop=options.should_stop)
    except PauseRequested:
        raise
    except Exception as exc:  # noqa: BLE001 - an aid on top of a finished transcript
        log.warning("Coverage check failed (%s: %s); the transcript is kept as decoded", type(exc).__name__, exc)
        return segments, {"checked": False, "reason": f"{type(exc).__name__}: {str(exc)[:200]}"}
    _stop_point(options)
    return merged, record


def _obtain_wav(media: Path, tmp: Path, options: RunOptions, audio_cfg: dict[str, Any], ffmpeg: Optional[Path],
                emit: Optional[Emitter] = None, duration: float = 0.0) -> Path:
    if options.reuse_audio is not None:
        wav = Path(options.reuse_audio)
        if not wav.is_file():
            raise FileNotFoundError(f"--reuse-audio file not found: {wav}")
        log.info("Using the pre-extracted audio %s (FFmpeg step skipped)", wav.name)
        return wav
    last = [0.0]

    def extracted(seconds: float) -> None:                  # real progress: media time written / media duration
        if emit is not None and duration > 0 and time.monotonic() - last[0] >= 0.3:
            last[0] = time.monotonic()
            emit("stage", stage="extracting", percent=round(min(100.0, 100.0 * seconds / duration), 1))

    if emit is not None:
        emit("stage", stage="extracting", percent=0.0 if duration > 0 else None)
    try:
        wav = audio_mod.extract_audio(media, tmp / "audio_16k.wav", audio_cfg, options.preprocessing, ffmpeg,
                                      allow_fallback=True, progress=extracted, should_stop=options.should_stop)
        if emit is not None and duration > 0:
            emit("stage", stage="extracting", percent=100.0)      # progress: the bar reaches 100 % when the audio is ready (the last report may have been throttled)
        return wav
    except audio_mod.ExtractionCancelled as exc:
        raise PauseRequested() from exc


def _summary_event(summary: ProcessingSummary) -> dict[str, Any]:
    return {"output_dir": str(summary.output_dir) if summary.output_dir else None, "segments": summary.segments,
            "words": summary.words, "uncertain": summary.uncertain, "terms": summary.terms_detected,
            "possible_errors": summary.possible_errors, "recovered_gaps": summary.recovered_gaps,
            "audio_seconds": summary.audio_seconds, "language": summary.language, "device": summary.device,
            "compute_type": summary.compute_type, "timing": summary.timing, "state": summary.state, "translation": summary.translation,
            "languages": summary.languages, "translations": summary.translations}


def process_media(
    media: Path, options: RunOptions, cfg: dict[str, Any], transcriber: Transcriber,
    spec: ModelSpec, output_name: Optional[str] = None,
) -> ProcessingSummary:
    """Run the whole pipeline for one media file. Raises on failure (callers decide policy).

    Emits structured events (:mod:`src.events`) to ``options.events``; a user-requested pause returns a summary with
    ``paused=True`` (checkpoint and partial transcript kept) instead of raising.
    """
    emit = Emitter(options.events)
    transcriber.events = emit
    emit("job_started", source=str(media), name=media.name, mode=options.mode, device=transcriber.adapter.device,
         model=str(spec.model), backend=options.backend, resume=options.resume)
    try:
        summary = _process_media_impl(media, options, cfg, transcriber, spec, output_name, emit)
    except BaseException as exc:
        emit("failed", error=str(exc)[:600], kind=type(exc).__name__)
        raise
    if summary.ok:
        emit("completed", **_summary_event(summary))
    return summary


def _process_media_impl(
    media: Path, options: RunOptions, cfg: dict[str, Any], transcriber: Transcriber,
    spec: ModelSpec, output_name: Optional[str], emit: Emitter,
) -> ProcessingSummary:
    started = time.perf_counter()
    outputs_root = options.outputs_root or Path(cfg["paths"]["outputs_abs"])
    paths: OutputPaths = make_output_paths(outputs_root, output_name or media.stem,
                                           checkpoint_dir=checkpoint_dir_for(options, media))
    summary = ProcessingSummary(media=media, output_dir=paths.directory, device=transcriber.adapter.device)
    ml_history.archive_previous(paths.directory)         # transcribing the same recording again keeps the earlier finished results (previous_versions/<date-time>/)
    mark_output_owner(paths.directory, media)
    if options.backend not in BACKENDS:
        raise ValueError(f"unknown backend '{options.backend}' (use one of: {', '.join(BACKENDS)})")

    with file_logging(paths.log):
        log.info("=== %s v%s | %s ===", "medical-lecture-transcriber", __version__, media.name)
        audio_cfg = cfg["audio"]
        audio_mod.validate_media_file(media, audio_cfg)
        emit("stage", stage="probing", percent=None)
        terms_dir = Path(cfg["paths"]["medical_terms_abs"])
        if options.reuse_audio is not None:
            ffmpeg, probe_info = None, {"duration_s": _wav_seconds(Path(options.reuse_audio)), "note": "pre-extracted WAV"}
        else:
            ffmpeg, ffprobe = audio_mod.locate_ffmpeg_optional(audio_cfg)      # no ffmpeg.exe -> PyAV reads the file
            probe_info = audio_mod.probe_media(media, ffprobe)
        log.info("Source: %s (%s)", media.name, probe_info or "no probe info")
        emit("source_validated", name=media.name, size_bytes=media.stat().st_size,
             duration_seconds=probe_info.get("duration_s"), container=probe_info.get("container"),
             codec=probe_info.get("codec"))
        if transcriber.adapter.device == "cpu":
            emit("warning", code="cpu_mode", message="Running on the CPU: Large-v3 can take several times the length of "
                 "the recording. An NVIDIA GPU is much faster.")

        # --- optional online backend FIRST when it is the chosen backend (no local decode then)
        opinion = second_mod.SecondOpinion()
        if options.backend == "deepgram":
            opinion = _deepgram_only(media, options, cfg, paths, terms_dir)
            if opinion.ok:
                return _finish_deepgram_only(summary, paths, opinion, media, probe_info, started)
            log.warning("Deepgram was requested but is unavailable (%s); running the local model instead.", opinion.status)

        profile, capability_notes = _effective_profile(cfg[options.mode], transcriber.adapter.capabilities)
        base_profile = dict(profile)
        if not options.use_hotwords:
            profile["use_hotwords"] = False
            base_profile["use_hotwords"] = False
            log.info("Hotwords disabled by --no-hotwords")
        for note in capability_notes:
            log.warning("Capability check: %s", note)
        multilingual = options.multilingual and transcriber.adapter.capabilities.multilingual
        if options.multilingual and not multilingual:
            log.warning("Capability check: --multilingual ignored (backend cannot re-detect language per window)")
        for warning in vram_warnings():
            log.warning(warning)

        source_sha = file_sha256(media)
        pdf_path = options.pdf_path or (find_companion_pdf(media) if options.use_pdf else None)
        pdf_sha = file_sha256(pdf_path) if pdf_path is not None and Path(pdf_path).is_file() else None
        checkpoint = SegmentCheckpoint(paths.checkpoint_dir, _fingerprint(source_sha, options, cfg, spec, pdf_sha)) \
            if cfg.get("checkpoint", {}).get("enabled", True) else None
        saved_plan: Optional[dict[str, Any]] = None
        if checkpoint is not None and options.resume:
            _, saved_plan = checkpoint.load()
            if saved_plan is None and (paths.checkpoint_dir / "checkpoint.json").is_file():
                emit("warning", code="checkpoint_incompatible", message="An earlier checkpoint for this file belongs to a "
                     "different file version or different settings and is ignored; the job starts from the beginning.")

        if options.temp_root is not None:
            Path(options.temp_root).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="mlt_", dir=options.temp_root) as tmp:
            emit("audio_preparing", name=media.name)
            wav = _obtain_wav(media, Path(tmp), options, audio_cfg, ffmpeg, emit, float(probe_info.get("duration_s") or 0.0))
            _stop_point(options)
            if options.keep_audio and options.reuse_audio is None:
                shutil.copy2(wav, paths.directory / "audio_16k.wav")
            duration = float(probe_info.get("duration_s") or 0.0) or _wav_seconds(wav)
            emit("audio_ready", duration_seconds=duration)
            leveling = ml_leveler.Leveling(path=wav, reason="turned off in the settings")
            if str(ml_hook.settings_from(cfg).get("level_quiet_speech", "auto")) != "off":      # speech far quieter than normal is lifted in the working copy before the voice filter looks at it
                leveling = ml_leveler.level_quiet_speech(wav, Path(tmp), in_place=options.reuse_audio is None)
                wav = leveling.path

            vocab_all = build_index("all", None, terms_dir)
            chunk_cfg = _chunk_settings(options, cfg)
            # One independent VAD pass: where speech really starts (probe), where the pauses are (chunk cuts),
            # and the reference for the final audit.
            speech: list[tuple[float, float]] = []
            need_vad = options.quality_audit or _auto_topic(options, cfg) or duration > chunk_cfg["chunk_minutes"] * 75
            if need_vad:
                emit("stage", stage="speech_analysis", percent=None)
                try:
                    speech = quality_mod.compute_speech_ranges(wav, cfg["vad"])
                    log.info("VAD pass: %s of speech in %s (%.1f%%)", format_clock(sum(e - s for s, e in speech)),
                             format_clock(duration), 100 * sum(e - s for s, e in speech) / max(duration, 1e-9))
                except Exception as exc:  # noqa: BLE001 - advisory only
                    log.warning("Independent VAD pass failed (%s); probe starts at 0:00, chunks use fixed cuts, no audit", exc)

            activity = ml_hook.measure_activity(wav)            # loudness per second: the language scan, the music check and the coverage check read it
            pdf = load_pdf_context(media, options, cfg, vocab_all)
            engine: Optional[MedicalContextEngine]
            if saved_plan is not None and saved_plan.get("hotword_candidates") is not None and saved_plan.get("engine"):
                plan = RunPlan.from_dict(saved_plan)
                engine, acronyms = _rebuild_context(plan, media, options, cfg, pdf)
                log.info("Reusing the saved plan (language=%s, areas=%s, %d hotword candidates, %d chunks)",
                         plan.language, plan.subjects, len(plan.hotword_candidates), len(plan.chunks))
            else:
                emit("language_detecting", probe=_auto_topic(options, cfg))
                _stop_point(options)
                plan, engine, acronyms = make_plan(media, wav, duration, options, cfg, transcriber, spec, base_profile,
                                                   vocab_all, speech, pdf, emit, activity)
            emit("language_detected", language=plan.language or f"mixed ({plan.language_dominant})", source=plan.language_source,
                 probability=plan.language_probability, mixed=plan.multilingual)
            if plan.hotwords_off_reason:
                emit("warning", code="hotwords_off", message="Hotwords are off: " + plan.hotwords_off_reason + ".")
            if plan.multilingual:
                multilingual = bool(transcriber.adapter.capabilities.multilingual)
                if not multilingual:
                    log.warning("Capability check: this backend cannot decode a mixed-language recording window by window")
            emit("subject_detected", areas=plan.subjects, confident=(plan.topic or {}).get("confident"),
                 scores=(plan.topic or {}).get("scores"), pdf=(plan.pdf_info or {}).get("source"))
            if not options.use_hotwords:
                plan.hotword_candidates = []
            budgets = compute_token_budgets(cfg["context"])
            plan.budgets = budgets.to_dict()
            for note in budgets.notes:
                log.warning("Token budget: %s", note)
                capability_notes.append(f"token budget: {note}")
            if not plan.chunks:
                chunk_list = plan_chunks(duration, chunk_cfg["chunk_minutes"], chunk_cfg["overlap_seconds"], speech,
                                         chunk_cfg["snap_window_seconds"], min_pause_s=chunk_cfg["min_pause_seconds"])
                plan.chunks = [c.to_dict() for c in chunk_list]
            chunks = [Chunk(int(c["index"]), c["own_start"], c["own_end"], c["start"], c["end"]) for c in plan.chunks]

            language_index, pairs = build_language_index(plan.term_language, terms_dir, extra_terms=plan.pdf_terms)
            language_terms_index = language_index

            if options.probe_only:
                return _finish_probe(summary, paths, checkpoint, plan, duration, started)

            hotwords_enabled = bool(profile.get("use_hotwords", True)) and options.use_hotwords and not plan.hotwords_off_reason
            static = list(plan.hotword_candidates)

            def hotword_source() -> list[str]:
                if not hotwords_enabled:
                    return []
                return engine.context.hotword_candidates(int(cfg.get("context_engine", {}).get("candidate_limit", 60))) \
                    if engine is not None else static

            log.info("Subjects: %s | engine: %s | hotword candidates: %d%s | token budget: prompt %d / hotwords %d",
                     ", ".join(plan.subjects) or "none", plan.engine, len(hotword_source()),
                     "" if hotword_source() else " (none)", budgets.prompt, budgets.hotwords)

            store = ChunkStore(paths.checkpoint_dir / "chunks", checkpoint.fingerprint if checkpoint else {"run": "no-checkpoint"})
            monitor = GpuMemoryMonitor(index=options.device_index) if transcriber.adapter.device == "cuda" else None
            if monitor:
                monitor.start()
            progress = ProgressReporter(
                step=float(cfg.get("progress", {}).get("step_percent", 5.0)),
                interval_s=float(cfg.get("progress", {}).get("interval_seconds", 30.0)),
                gpu_stats=monitor.latest if monitor else None, sticky=True, on_update=emit,
            )
            decoder = ChunkedDecoder(
                transcriber=transcriber, wav=wav, duration=duration, chunks=chunks, store=store,
                plan_dict=plan.to_dict, cfg=cfg, base_profile=profile, budgets=budgets, language=plan.language,
                hotword_source=hotword_source, progress=progress, multilingual=multilingual,
                language_plan=_language_plan_tuples(plan, transcriber),
                resume=options.resume and checkpoint is not None, retries=int(chunk_cfg["retries"]), engine=engine,
                acronyms=acronyms, pairs=pairs, events=emit, should_stop=options.should_stop,
            )
            start_offset = decoder.pending_start()
            progress.start_offset = start_offset
            resume_info = decoder.resume_summary()
            if resume_info["completed"] or resume_info["partial"]:
                emit("resume_started", resume_from_seconds=round(max(start_offset, 0.0), 2), **resume_info)
            _log_header(media, duration, plan, spec, transcriber, start_offset if start_offset > 0 else 0.0, len(chunks))
            if checkpoint is not None:
                checkpoint.mark(plan.to_dict(), "running")
            decode_started = time.perf_counter()
            coverage_record: dict[str, Any] = {}
            recheck_record: dict[str, Any] = {}
            try:
                outcome = decoder.run()
                segments, dropped = stitch(outcome.parts)
                if dropped:
                    log.info("Stitching: %d duplicate segment(s) at chunk boundaries dropped", dropped)
                segments = decoder.repair_gaps(segments, speech)
                segments, recheck_record = _recheck_languages(transcriber, wav, segments, plan, options, profile, cfg, duration, emit)
                segments, coverage_record = _recover_coverage(transcriber, wav, segments, activity, plan, options, profile, cfg, duration, emit)
            except PauseRequested:
                return _finish_paused(summary, paths, checkpoint, plan, decoder, started, emit, len(chunks))
            except BaseException as exc:
                _save_partial(paths, decoder.partial_segments(), checkpoint, exc)
                record_session(paths.checkpoint_dir, _session_from(decoder, started, "failed"))
                if isinstance(exc, ChunkFailedError):
                    log.error("Chunk failed for good: %s", exc)
                raise
            finally:
                gpu_memory = monitor.stop() if monitor else {}
            decode_wall = time.perf_counter() - decode_started
            result = outcome.result or _empty_result(spec, transcriber, plan, duration)
            result.segments = segments
            result.duration = max(result.duration, duration)
            result.hotwords_applied, result.hotword_tokens = outcome.hotwords_applied, outcome.hotword_tokens
            result.notes = outcome.notes
            result.capabilities_used = outcome.capabilities_used or result.capabilities_used
            result.fallback_used = outcome.fallback_used
            result.load_seconds = outcome.load_seconds
            whisperx_info = _run_whisperx(options, result, wav)
            # Output Language: the spoken-language transcript above is final; this only ADDS translated_<code>_transcript.* (needs the audio for English)
            translation = translation_hook.run_for_pipeline(
                output_language=options.output_language, source_language=result.language or plan.language, mixed=bool(plan.multilingual),
                segments=result.segments, transcriber=transcriber, wav=wav, profile=profile, cfg=cfg, duration=duration,
                model_parent_dirs=list(options.translation_model_dirs), emit=emit, stop_point=lambda: _stop_point(options))

        audio_seconds = result.duration or probe_info.get("duration_s", 0.0)
        session = _session_from(decoder, started, "completed")
        if session.decoded_audio_seconds + session.repair_audio_seconds > 0:
            log.info("Decoded %s of audio in %s this run (model load %.1fs)",
                     format_clock(session.decoded_audio_seconds + session.repair_audio_seconds),
                     format_clock(session.decode_seconds + session.repair_seconds), session.model_load_seconds)
        else:
            log.info("Nothing needed decoding this run: every chunk came from the checkpoints")

        # --- one uncertainty / hallucination engine: decode statistics + timeline checks, one list of flags
        unc_cfg = UncertaintyConfig.from_config(cfg)
        annotate_segments(result.segments, unc_cfg, language_terms_index, result.hotwords_applied)
        quality_mod.add_engine_flags(result.segments, audio_seconds, speech, cfg.get("audit"))
        uncertain = sum(1 for s in result.segments if s.uncertain)
        log.info("Uncertain segments: %d / %d", uncertain, len(result.segments))
        clean = clean_segments(result.segments, language_terms_index, cfg["cleaner"])
        log.info("Clean transcript: %d dictionary-backed edits", len(clean.changes))

        # --- medical analysis (reports only; the transcript is not changed)
        rep_cfg = cfg.get("medical_report", {})
        report = build_report(
            result.segments, language_terms_index, cutoff=float(rep_cfg.get("cutoff", 0.86)),
            min_word_length=int(cfg["uncertainty"].get("near_miss_min_length", 7)),
            candidate_min_count=int(rep_cfg.get("candidate_min_count", 2)),
            term_subjects=term_subject_map(terms_dir), pdf_terms=plan.pdf_terms, pdf_text=pdf.text,
            active_subjects=_active_vocab_files(plan, engine, terms_dir),
            min_score=float(rep_cfg.get("min_suggestion_score", 0.45)),
        )
        full_text = " ".join(s.text.strip() for s in result.segments)
        acronyms.learn(full_text, "defined in speech")
        acronym_hits = acronyms.scan(result.segments)
        abbreviation_findings = disambiguate(result.segments, acronyms.known)
        entities = build_entities(report.terms, terms_dir, pairs)
        name_findings = scan_name_variants(result.segments)
        final_mix = lang_mod.estimate_language_mix(full_text)
        log.info("Medical terms: %d distinct heard, %d candidates outside the dictionary, %d possible ASR errors, "
                 "%d acronyms, %d entities", len(report.terms), len(report.candidates), len(report.errors),
                 len(acronym_hits), len(entities))
        quality = None
        if options.quality_audit and speech:
            emit("quality_audit_started")
            quality = quality_mod.analyze_quality(result.segments, audio_seconds, speech, cfg.get("audit"))
            emit("quality_audit_finished", **{k: quality["summary"][k] for k in (
                "speech_coverage", "timestamp_issues", "hallucination_flags", "repetition_flags", "missing_speech_gaps")})
        if quality:
            q = quality["summary"]
            log.info("Quality audit: coverage %.1f%% of VAD speech | timestamp issues %d | hallucination flags %d | "
                     "repetition flags %d | missing-speech gaps %d", 100 * q["speech_coverage"], q["timestamp_issues"],
                     q["hallucination_flags"], q["repetition_flags"], q["missing_speech_gaps"])
            if coverage_record.get("checked"):              # the voice-activity filter does not hear singing or speech over music: the same question answered on the audio itself
                q["audible_coverage"] = coverage_record.get("covered_share_after")
                q["audible_seconds"] = coverage_record.get("audible_seconds")
                q["unrecovered_audible_gaps"] = len(coverage_record.get("unrecovered") or [])
                q["coverage_basis"] = "speech_coverage is relative to what the voice-activity filter calls speech; audible_coverage is relative to the loudness of the audio"
                log.info("Quality audit: %.1f%% of the AUDIBLE time (%.0f s, measured on the audio) carries words | %d audible stretch(es) without words",
                         100 * float(q["audible_coverage"] or 0.0), float(q["audible_seconds"] or 0.0), q["unrecovered_audible_gaps"])

        # --- medical expert pass: separate outputs, the raw / clean transcripts are never touched
        medical_expert_result = None
        if options.medical_expert:                       # (the app already shows "Finalizing ..." here; a stage event after the first words is not allowed)
            _, medical_redecode = medical_integration.make_decoders(transcriber, wav, profile, cfg, audio_seconds, DecodeRequest)
            medical_expert_result = medical_integration.run_for_pipeline(
                result.segments, plan=plan, options=options, duration=audio_seconds, asr_seconds=time.perf_counter() - started,
                redecode=medical_redecode)

        # --- the canonical source transcript: what was heard, language by language, stored once; every translation is made from it (no audio, no recogniser)
        source_transcript = ml_finalize.build_source(
            paths.directory.name, result.segments, medical_expert_result, input_language=options.language or "auto", output_language=options.output_language or "",
            engine={"asr": result.model, "backend": result.backend, "device": transcriber.adapter.device, "language_plan": plan.language_plan or None,
                    "language_scan": plan.language_scan or None}, coverage=coverage_record)
        paths.directory.mkdir(parents=True, exist_ok=True)
        ml_finalize.store_source(paths.directory, source_transcript)
        if translation.deferred:
            translation = ml_finalize.run_text_translation(paths.directory, translation, models_dirs=[*options.translation_model_dirs], device=transcriber.adapter.device,
                                                           emit=emit, stop_point=lambda: _stop_point(options),
                                                           release=transcriber.unload if transcriber.adapter.device == "cuda" else None)
        elif translation.method == translation_hook.METHOD_WHISPER and translation.written:
            ml_finalize.store_whisper_translation(paths.directory, translation, source_transcript, "whisper-translate:" + str(result.model or ""))
        source_transcript = ml_source.read_file(paths.directory) or source_transcript

        # --- optional second opinion (only if the user asked for it)
        if options.backend == "compare" or options.deepgram_if_suspect:
            opinion = _second_opinion(media, options, cfg, plan, result, uncertain, quality, language_terms_index, pdf, engine)
        elif opinion.status == "not requested":
            opinion = second_mod.SecondOpinion(status="not requested", reason="local only (default)")
        second_mod.write_second_opinion(paths, opinion)

        smi = query_nvidia_smi(options.device_index) or {}
        language_note = (f"Language mix (stop-word heuristic): English {final_mix['english_share']:.0%} / "
                         f"Turkish {final_mix['turkish_share']:.0%}" + (" -> MIXED" if final_mix["mixed"] else ""))
        reported_profile = dict(profile)
        if outcome.conditioning_used is not None:        # what the chunks that carried hotwords really ran with
            reported_profile["condition_on_previous_text"] = outcome.conditioning_used
        context_report = _context_report(plan, engine, budgets, outcome, entities, acronym_hits, pdf)
        session.wall_seconds = time.perf_counter() - started
        timing = timing_report(audio_total=audio_seconds, this_run=session, earlier=load_sessions(paths.checkpoint_dir),
                               chunk_cumulative=decoder.cumulative_stats())
        recovered = [g for g in outcome.gap_repairs if g.get("inserted")]
        metadata = {
            "tool": {"name": "medical-lecture-transcriber", "version": __version__, "edition": "medical-expert"},
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": {"file": media.name, "size_bytes": media.stat().st_size, "sha256": source_sha,
                       "probe": probe_info, "preprocessing": options.preprocessing,
                       "audio": "pre-extracted WAV" if options.reuse_audio else "FFmpeg 16 kHz mono PCM"},
            "mode": options.mode,
            "profile": {k: reported_profile.get(k) for k in (
                "beam_size", "temperature", "compute_type", "fallback_compute_type", "vad_filter",
                "condition_on_previous_text", "word_timestamps", "use_hotwords")},
            "vad": cfg["vad"],
            "model": {"key": spec.key, "engine": spec.engine, "name": result.model, "backend": result.backend,
                      "compute_type_used": result.compute_type, "fallback_used": result.fallback_used},
            "language": {"requested": options.language or "auto", "forced": plan.language,
                         "source": plan.language_source, "detected": result.language,
                         "probability": plan.language_probability or result.language_probability,
                         "multilingual": multilingual, "dominant": plan.language_dominant, "sampling": plan.language_samples,
                         "mix_probe": plan.language_mix, "mix_full": final_mix,
                         "mixed": bool(final_mix["mixed"] or (plan.language_mix or {}).get("mixed")),
                         "note": lang_mod.MIXED_LANGUAGE_NOTE},
            "subject": {"engine": plan.engine, "subjects": plan.subjects, "topic_detection": plan.topic,
                        "heard_terms_in_probe": plan.heard_terms, "hotwords": plan.hotword_candidates,
                        "hotwords_off_reason": plan.hotwords_off_reason or None,
                        "hotwords_applied": result.hotwords_applied, "hotword_tokens": result.hotword_tokens,
                        "token_budget": budgets.to_dict(),
                        "hotwords_used": result.capabilities_used.get("hotwords", False)},
            "pdf_context": plan.pdf_info or {"source": None, "terms": [], "pages": 0, "error": None},
            "long_audio": {"chunks": len(chunks), "chunk_minutes": chunk_cfg["chunk_minutes"],
                           "overlap_seconds": chunk_cfg["overlap_seconds"], "stitch_duplicates_dropped": dropped,
                           "loop_repairs": outcome.loop_repairs, "gap_repairs": outcome.gap_repairs,
                           "per_chunk": outcome.chunk_log},
            "capabilities_used": result.capabilities_used,
            "capability_notes": capability_notes + result.notes,
            "whisperx": whisperx_info,
            "resume": {"resumed_segments": outcome.resumed_segments, "resume_from_seconds": round(max(start_offset, 0.0), 2)},
            "uncertainty": {"thresholds": cfg["uncertainty"], "uncertain_segments": uncertain,
                            "total_segments": len(result.segments),
                            "flag_codes": _flag_counts(result.segments)},
            "confidence": {"is_model_confidence": False, "method": CONFIDENCE_METHOD},
            "medical_terms": {"distinct_detected": len(report.terms), "occurrences": sum(t.count for t in report.terms),
                              "candidates_outside_dictionary": len(report.candidates),
                              "possible_asr_errors": len(report.errors), "acronyms": len(acronym_hits),
                              "entities": entities_summary(entities)},
            "quality": quality["summary"] if quality else None,
            "medical_expert": medical_expert_result.quality_report if medical_expert_result is not None else None,
            "translation": translation.record(),
            "multilingual": ml_finalize.lecture_record(source_transcript, plan_report=plan.language_scan or None, translation=translation.record(), audio=leveling.record(), language_check=recheck_record),
            "second_opinion": opinion.to_dict(),
            "timing": timing,
            "gpu": {"name": smi.get("name"), "driver": smi.get("driver"), "vram_total_mib": smi.get("total_mib"),
                    **gpu_memory,
                    "note": "process_peak_dedicated_mib = this process's real VRAM; process_peak_shared_mib = memory the "
                            "driver spilled to system RAM; peak_delta_mib = rise of whole-GPU usage (other programs included)"},
            "environment": environment_summary(),
            "clean_transcript": {"edits": clean.changes,
                                 "policy": "local_raw_transcript.txt is never modified; clean text only applies "
                                           "dictionary-backed spacing/eponym-case fixes and marks doubtful text"},
            "privacy": second_mod.DEEPGRAM_PRIVACY_NOTE if opinion.ok else PRIVACY_NOTE,
        }
        export_all(
            paths, result.segments, clean_text=clean.text, metadata=metadata, language=result.language,
            language_probability=plan.language_probability or result.language_probability, model=result.model,
            backend=result.backend, duration=audio_seconds, source=media.name,
            terms_text=terms_report_text(report.terms, report.candidates, language_note),
            errors_text=possible_errors_text(report.errors), quality=quality,
            entities={"summary": entities_summary(entities), "categories": sorted({c for e in entities for c in e.categories}),
                      "entities": [e.to_dict() for e in entities]},
            acronyms_text=acronyms_report_text(acronym_hits, abbreviation_findings),
            context=context_report, names_text=name_report_text(name_findings, entities, report.errors),
        )
        if medical_expert_result is not None:
            medical_expert_result.write(paths.directory)
        translation.write(paths.directory)
        if checkpoint is not None:
            if options.keep_checkpoint:
                record_session(paths.checkpoint_dir, session)
                checkpoint.mark(plan.to_dict(), "completed")
                checkpoint.close()
            else:
                checkpoint.discard()
        else:
            shutil.rmtree(paths.checkpoint_dir, ignore_errors=True)
        summary.ok = True
        summary.translation = translation.record()
        summary.state = "completed"
        summary.timing = timing
        summary.recovered_gaps = len(recovered)
        summary.recovered_gap_seconds = round(sum(float(g.get("seconds", 0.0)) for g in recovered), 1)
        summary.loop_repairs = sum(1 for r in outcome.loop_repairs if r.get("repaired"))
        summary.segments, summary.uncertain = len(result.segments), uncertain
        summary.audio_seconds, summary.real_time_factor = audio_seconds, timing["real_time_factor_effective"]
        summary.language, summary.compute_type = result.language, result.compute_type
        summary.languages, summary.translations = list(source_transcript.detected_languages), list(source_transcript.output_languages_available)
        summary.fallback_used, summary.peak_vram_mib = result.fallback_used, gpu_memory.get("peak_mib")
        summary.peak_dedicated_mib = gpu_memory.get("process_peak_dedicated_mib")
        summary.peak_shared_mib = gpu_memory.get("process_peak_shared_mib")
        summary.words = quality["summary"]["words"] if quality else len(full_text.split())
        summary.terms_detected = sum(t.count for t in report.terms)
        summary.possible_errors = len(report.errors)
        summary.resumed_segments = outcome.resumed_segments
        summary.chunks, summary.acronyms, summary.entities = len(chunks), len(acronym_hits), len(entities)
        summary.deepgram = opinion.status
        if quality:
            q = quality["summary"]
            summary.hallucination_flags, summary.repetition_flags = q["hallucination_flags"], q["repetition_flags"]
            summary.timestamp_issues = q["timestamp_issues"]
        summary.wall_seconds = time.perf_counter() - started
        log.info("Done in %s -> %s", format_clock(summary.wall_seconds), paths.directory)
    return summary


# ---------------------------------------------------------------------------------------------- helpers
def _active_vocab_files(plan: RunPlan, engine: Optional[MedicalContextEngine], terms_dir: Path) -> list[str]:
    """Vocabulary files that belong to this lecture's subject(s): the evidence for 'subject relevance' of a suggestion."""
    if engine is not None:
        files = [f for f, weight in engine.context.vocab_files.items() if weight > 0.3 and f != "general_medical"]
        if files:
            return files
    if plan.subjects and plan.subjects != ["general_medical"]:
        return _vocab_files(plan.subjects, terms_dir)
    return []


def _flag_counts(segments: list[Segment]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for seg in segments:
        for flag in seg.flags:
            counts[flag.code] = counts.get(flag.code, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def _rebuild_context(plan: RunPlan, media: Path, options: RunOptions, cfg: dict[str, Any], pdf: LectureContext
                     ) -> tuple[Optional[MedicalContextEngine], AcronymEngine]:
    """Recreate the (deterministic) context engine for a resumed run from the saved plan."""
    acronyms = AcronymEngine()
    if pdf.text:
        acronyms.learn(pdf.text, "defined in slides")
    if plan.probe_excerpt:
        acronyms.learn(plan.probe_excerpt, "defined in speech")
    if plan.engine == "legacy":
        return None, acronyms
    engine = MedicalContextEngine(
        cfg, Path(cfg["paths"]["medical_terms_abs"]), plan.term_language, user_subjects=_user_subjects(options),
        filename=media.stem, pdf_terms=pdf.terms, pdf_text=pdf.text)
    engine.build(plan.probe_excerpt)
    engine.add_acronyms([{"acronym": a, "expansion": e, "occurrences": 1} for a, (e, _) in acronyms.learned.items()])
    return engine, acronyms


def _context_report(plan: RunPlan, engine: Optional[MedicalContextEngine], budgets: TokenBudgets,
                    outcome: DecodeOutcome, entities: list[Any], acronym_hits: list[Any], pdf: LectureContext
                    ) -> dict[str, Any]:
    final = engine.context.to_dict() if engine is not None else None
    return {
        "engine": plan.engine, "language": plan.language, "language_source": plan.language_source,
        "areas": plan.topic, "initial_context": plan.context, "final_context": final,
        "token_budget": budgets.to_dict(), "hotwords_applied": outcome.hotwords_applied,
        "hotword_tokens": outcome.hotword_tokens,
        "pdf": plan.pdf_info, "chunks": outcome.chunk_log, "rolling_updates": final["rolling_updates"] if final else [],
        "acronyms": [h.to_dict() for h in acronym_hits],
        "note": "Context only chooses which vocabulary is offered to the decoder; it never edits the transcript.",
    }


def _deepgram_only(media: Path, options: RunOptions, cfg: dict[str, Any], paths: OutputPaths, terms_dir: Path
                   ) -> second_mod.SecondOpinion:
    """--backend deepgram: no local decode. Key terms come from the file name + --subject (+ PDF)."""
    vocab = build_index("all", None, terms_dir)
    pdf = load_pdf_context(media, options, cfg, vocab)
    engine = MedicalContextEngine(cfg, terms_dir, options.language if options.language != "auto" else None,
                                  user_subjects=_user_subjects(options), filename=media.stem, pdf_terms=pdf.terms,
                                  pdf_text=pdf.text)
    context = engine.build("")
    keyterms = context.hotword_candidates(int(cfg.get("deepgram", {}).get("max_keyterms", 100)))
    result, status = second_mod.run_deepgram(media, None if options.language in (None, "auto") else options.language,
                                             keyterms, cfg, api_key=options.deepgram_api_key)
    return second_mod.SecondOpinion(status=status, requested_by="backend", result=result, reason="--backend deepgram")


def _finish_deepgram_only(summary: ProcessingSummary, paths: OutputPaths, opinion: second_mod.SecondOpinion,
                          media: Path, probe_info: dict[str, Any], started: float) -> ProcessingSummary:
    second_mod.write_second_opinion(paths, opinion)
    assert opinion.result is not None
    write_json(paths.metadata, {
        "tool": {"name": "medical-lecture-transcriber", "version": __version__, "edition": "medical-expert"},
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {"file": media.name, "probe": probe_info}, "backend": "deepgram", "second_opinion": opinion.to_dict(),
        "privacy": second_mod.DEEPGRAM_PRIVACY_NOTE,
        "note": "Deepgram-only run: no local transcript and therefore no local_* / medical analysis files. "
                "Use --backend compare for the local transcript plus a medical disagreement report."})
    summary.ok = True
    summary.segments = len(opinion.result.segments)
    summary.words = sum(len(s.text.split()) for s in opinion.result.segments)
    summary.audio_seconds = float(opinion.result.duration or 0.0)
    summary.language, summary.deepgram = opinion.result.language, opinion.status
    summary.wall_seconds = time.perf_counter() - started
    log.info("Deepgram transcript written to %s", paths.directory)
    return summary


def _second_opinion(
    media: Path, options: RunOptions, cfg: dict[str, Any], plan: RunPlan, result: TranscriptionResult, uncertain: int,
    quality: Optional[dict[str, Any]], vocab: VocabularyIndex, pdf: LectureContext, engine: Optional[MedicalContextEngine],
) -> second_mod.SecondOpinion:
    q = (quality or {}).get("summary", {})
    ask, requested_by, reason = second_mod.should_ask_deepgram(
        "compare" if options.backend == "compare" else "local", options.deepgram_if_suspect, uncertain,
        len(result.segments), int(q.get("hallucination_flags", 0)), int(q.get("repetition_flags", 0)),
        cfg.get("deepgram", {}).get("suspect_thresholds", {}))
    if not ask:
        log.info("Deepgram not asked: %s", reason)
        return second_mod.SecondOpinion(status="not requested", reason=reason)
    keyterms = engine.context.hotword_candidates(int(cfg.get("deepgram", {}).get("max_keyterms", 100))) \
        if engine is not None else plan.hotword_candidates
    dg, status = second_mod.run_deepgram(media, result.language or plan.language, keyterms, cfg,
                                         api_key=options.deepgram_api_key)
    opinion = second_mod.SecondOpinion(status=status, requested_by=requested_by, result=dg, reason=reason)
    if dg is not None:
        opinion.comparison = second_mod.compare_with_local(result.segments, dg, vocab, plan.pdf_terms, pdf.text)
    else:
        log.warning("Deepgram second opinion: %s", status)
    return opinion


def _session_from(decoder: ChunkedDecoder, started: float, outcome_name: str) -> SessionStats:
    out = decoder.outcome
    return SessionStats(
        started=now_iso(), outcome=outcome_name, wall_seconds=time.perf_counter() - started,
        model_load_seconds=out.load_seconds, decode_seconds=out.decode_seconds,
        decoded_audio_seconds=out.decoded_audio_seconds, repair_seconds=out.repair_seconds,
        repair_audio_seconds=out.repair_audio_seconds)


def _finish_paused(summary: ProcessingSummary, paths: OutputPaths, checkpoint: Optional[SegmentCheckpoint], plan: RunPlan,
                   decoder: ChunkedDecoder, started: float, emit: Emitter, chunk_count: int) -> ProcessingSummary:
    """The user asked to stop: keep everything finished so far, mark the job paused, return normally."""
    partial = decoder.partial_segments()
    if checkpoint is not None:
        checkpoint.mark(plan.to_dict(), "paused")
        checkpoint.close()
    if partial:
        write_text(paths.partial, timestamped_text(partial))
    record_session(paths.checkpoint_dir, _session_from(decoder, started, "paused"))
    position = partial[-1].end if partial else 0.0
    log.warning("PAUSED SAFELY at %s (%d segments saved). Select the same file again to resume.", format_clock(position),
                len(partial))
    emit("paused", segments=len(partial), position_seconds=round(position, 1), chunks=chunk_count,
         checkpoint=str(paths.checkpoint_dir), partial=str(paths.partial) if partial else None)
    summary.ok, summary.paused, summary.state = False, True, "paused"
    summary.segments, summary.chunks = len(partial), chunk_count
    summary.wall_seconds = time.perf_counter() - started
    return summary


def process_files(
    media_files: list[Path], options: RunOptions, cfg: dict[str, Any],
) -> list[ProcessingSummary]:
    """Process many files with ONE loaded model. A failure in one file does not stop the batch."""
    spec = options.model_spec or get_model_spec(options.model_key, load_model_specs())
    if options.cpu_threads:
        spec = replace(spec, options={**spec.options, "cpu_threads": int(options.cpu_threads)})
    if spec.engine == "mlx_whisper":                # Apple Silicon: the Mac's GPU through Metal, no CUDA question
        device, why_cpu = "metal", None
    else:
        device, why_cpu = resolve_device(options.device, options.device_index)
    if why_cpu:
        log.warning("CUDA acceleration unavailable: %s CPU fallback enabled.", why_cpu)
        if options.events is not None:
            Emitter(options.events)("warning", code="cuda_unavailable", message=f"CUDA acceleration unavailable. "
                                    f"Reason: {why_cpu} CPU fallback enabled.", reason=why_cpu)
    transcriber = Transcriber(spec, cfg[options.mode], device, options.device_index, options.offline)
    summaries: list[ProcessingSummary] = []
    try:
        for media in media_files:
            name = options.output_name if (options.output_name and len(media_files) == 1) else output_name_for(media, media_files)
            try:
                summaries.append(process_media(media, options, cfg, transcriber, spec, name))
            except PauseRequested:                          # paused / cancelled while the file was still being prepared
                Emitter(options.events)("paused", segments=0, position_seconds=0.0, chunks=0, checkpoint=None, partial=None)
                log.warning("PAUSED SAFELY before the first segment (%s). Select the same file again to start over.", media.name)
                summaries.append(ProcessingSummary(media=media, paused=True, state="paused"))
            except Exception as exc:  # noqa: BLE001 - report per file, continue the batch
                log.error("FAILED %s: %s: %s", media.name, type(exc).__name__, exc)
                if len(media_files) == 1:
                    raise
                summaries.append(ProcessingSummary(media=media, error=f"{type(exc).__name__}: {exc}"))
    finally:
        transcriber.close()
    return summaries


def resolve_subjects(value: Optional[str], cfg: dict[str, Any]) -> list[str]:
    """``--subject`` values: vocabulary files (``neuroscience``), ``all``, or any of the 19 medical areas
    (``cardiology``, ``neuroanatomy`` ...). Areas steer the context engine; files name a vocabulary directly."""
    terms_dir = Path(cfg["paths"]["medical_terms_abs"])
    raw = value or cfg["defaults"]["subject"]
    items = [i.strip() for i in (raw.split(",") if isinstance(raw, str) else list(raw)) if i and i.strip()]
    files = set(available_subjects(terms_dir))
    names: list[str] = []
    for item in items:
        key = item.lower().replace(" ", "_")
        if key == "all" or key in files:
            names += parse_subjects(key, terms_dir)
        else:
            area = SubjectClassifier._canonical_area(item)
            if area not in AREAS:
                parse_subjects(key, terms_dir)          # raises UnknownSubjectError with the list of valid names
            names.append(area)
    return list(dict.fromkeys(names))


def _vocab_files(subjects: list[str], terms_dir: Path) -> list[str]:
    """Vocabulary files for a list of subject names (areas are mapped to their files)."""
    files = set(available_subjects(terms_dir))
    out: list[str] = []
    for name in subjects:
        out += [name] if name in files else [f for f in AREA_VOCAB_FILES.get(name, ()) if f in files]
    return list(dict.fromkeys(out)) or ["general_medical"]
