"""``--selftest``: a no-window check that an installation really works (used by the packaged-build smoke test and BUILD_ALL).

Each check reports ``ok`` / ``skipped`` / failure with a reason; none can hang the run (every one that touches an
external program or the GPU driver has its own timeout). The report is machine readable JSON.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import struct
import tempfile
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .. import APP_DIR, APP_NAME, FROZEN, RESOURCE_ROOT, __version__

REQUIRED_MODULES = ["faster_whisper", "ctranslate2", "av", "numpy", "yaml", "rapidfuzz", "pypdf", "onnxruntime", "PySide6.QtWidgets", "PIL"]
OPTIONAL_MODULES = ["soundfile"]
RESOURCE_FILES = ["config/transcription.yaml", "config/app_models.json", "config/common_english_words.txt", "medical_terms"]


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    skipped: bool = False
    seconds: float = 0.0


@dataclass
class SelfTestReport:
    checks: list[Check] = field(default_factory=list)
    app: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return all(c.ok or c.skipped for c in self.checks)

    def to_json(self) -> str:
        return json.dumps({"ok": self.ok, "app": self.app, "checks": [c.__dict__ for c in self.checks]}, indent=2) + "\n"


def _run(report: SelfTestReport, name: str, fn: Callable[[], str], skip_if: Optional[Callable[[], str]] = None) -> None:
    started = time.perf_counter()
    try:
        reason = skip_if() if skip_if else ""
        if reason:
            report.checks.append(Check(name, True, reason, skipped=True, seconds=time.perf_counter() - started))
            return
        detail = fn()
        report.checks.append(Check(name, True, detail, seconds=round(time.perf_counter() - started, 2)))
    except _Skip as exc:
        report.checks.append(Check(name, True, str(exc), skipped=True, seconds=round(time.perf_counter() - started, 2)))
    except Exception as exc:  # noqa: BLE001 - a failed check is a result, not a crash
        report.checks.append(Check(name, False, f"{type(exc).__name__}: {exc}", seconds=round(time.perf_counter() - started, 2)))


def _modules() -> str:
    missing = []
    for name in REQUIRED_MODULES:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001
            missing.append(f"{name} ({type(exc).__name__}: {exc})")
    if missing:
        raise RuntimeError("cannot import: " + "; ".join(missing))
    return f"{len(REQUIRED_MODULES)} required libraries import"


def _resources() -> str:
    missing = [rel for rel in RESOURCE_FILES if not (RESOURCE_ROOT / rel).exists()]
    if missing:
        raise FileNotFoundError("missing resources: " + ", ".join(missing))
    from ..vocabulary import DEFAULT_TERMS_DIR, load_terms

    terms = load_terms("all", None, DEFAULT_TERMS_DIR)
    if len(terms) < 1000:
        raise RuntimeError(f"vocabulary looks incomplete ({len(terms)} terms)")
    from ..medical.lexicon import available_domains, shared_lexicon

    domains = available_domains()
    if len(domains) < 20 or not shared_lexicon().domain("neuroscience") or not shared_lexicon().domain("neuroscience").terms:
        raise RuntimeError(f"the Medical Expert vocabulary (medical_vocab) is missing or incomplete ({len(domains)} domains)")
    return f"resources found; {len(terms):,} medical terms load; Medical Expert vocabulary: {len(domains)} domains" + _offline_release_detail()


OFFLINE_MARKER = "offline_release.json"


def offline_marker() -> Optional[Path]:
    """``offline_release.json`` of the macOS offline release (``<App>.app/Contents/Resources``); None everywhere else."""
    candidates = [APP_DIR.parent / "Resources" / OFFLINE_MARKER] if APP_DIR.name == "MacOS" else []
    candidates += [RESOURCE_ROOT / OFFLINE_MARKER, APP_DIR / OFFLINE_MARKER]
    return next((c for c in candidates if c.is_file()), None)


def _offline_release_detail() -> str:
    """Only in the offline release: everything it promises to carry is really inside the app - the speech model (sizes; every checksum
    when ``MLT_SELFTEST_DEEP=1``), the OCR models, the licence files. A missing piece fails the self-test loudly: the release has no
    network fallback."""
    marker = offline_marker()
    if marker is None:
        return ""
    import importlib.util
    import platform

    info = json.loads(marker.read_text(encoding="utf-8"))
    problems: list[str] = []
    machine = platform.machine().lower()
    if info.get("architecture") and machine not in (info["architecture"], "amd64" if info["architecture"] == "x86_64" else info["architecture"]):
        problems.append(f"built for {info['architecture']} but this computer is {machine}")
    from .model_service import ModelService
    from .paths import get_paths

    models = ModelService(get_paths())
    key = info.get("model", "whisper-large-v3")
    bad = models.bundled_problems(key)
    if bad or not models.is_installed(key):
        problems.append(f"speech model {key}: " + ("; ".join(bad) if bad else "not found in the app"))
    elif os.environ.get("MLT_SELFTEST_DEEP") == "1":
        verified = models.verify(key)
        if not verified.ok:
            problems.append(f"speech model {key} fails its checksum: " + "; ".join(verified.problems[:3]))
    spec = importlib.util.find_spec("rapidocr_onnxruntime")
    folders = list(spec.submodule_search_locations or []) if spec else []
    ocr_models = sum(len(list((Path(f) / "models").glob("*.onnx"))) for f in folders)
    if ocr_models < 3:
        problems.append(f"text recognition models: {ocr_models} found, 3 needed")
    for name in info.get("license_files", []):
        if not (marker.parent / name).is_file():
            problems.append(f"{name} is missing")
    if problems:
        raise RuntimeError("offline release is incomplete: " + "; ".join(problems))
    return f"; offline release {info.get('release', '?')}: speech model, text recognition models and licence files present"


MEDICAL_FIXTURE = ["In neuroanatomy, the NISLO bodies are found in the neuronal soma and in the dendrites,",
                   "and Schwann cells form the myelin sheath of the peripheral nervous system.",
                   "Tanny sites line the floor of the third ventricle,",
                   "the node of Ranvier allows saltatory conduction along a myelinated axon,",
                   "and RIMAC bundles contain several unmyelinated axons wrapped by one Schwann cell,",
                   "an axodendritic synapse connects an axon to a dendrite and an axosomatic synapse ends on the cell body."]


def _medical() -> str:
    """The Medical Expert layer, from the vocabulary files INSIDE the app: known ASR errors become the right terms, correct terms stay as
    spoken, and the raw text it was given is not changed."""
    from ..medical.corrector import MedicalContextCorrector, Options, SegmentView
    from ..medical.lexicon import shared_lexicon

    views = [SegmentView(i, i * 4.0, i * 4.0 + 3.5, text, "en", [(w, 0.97) for w in text.split()]) for i, text in enumerate(MEDICAL_FIXTURE)]
    before = [v.text for v in views]
    result = MedicalContextCorrector(shared_lexicon(), Options(subjects=("neuroanatomy",))).correct(views)
    done = " ".join(result.texts)
    want = ("Nissl bodies", "Tanycytes", "Remak bundles")
    keep = ("Schwann cells", "Ranvier", "saltatory conduction", "axodendritic", "axosomatic")
    missing = [w for w in want if w not in done] + [w for w in keep if w not in done]
    if missing:
        raise RuntimeError("medical expert fixture: expected " + ", ".join(missing) + "; got: " + done[:200])
    if [v.text for v in views] != before:
        raise RuntimeError("medical expert changed its input text")
    return f"{len(want)} known ASR errors corrected from the packaged vocabulary; correct terms left alone"


def _knowledge() -> str:
    """The Medical Knowledge database of v4 (shadow engine): when one is installed - or the build requires one - it must open, match the schema and answer
    exact / full-text / phonetic / relationship queries; a damaged or missing REQUIRED database fails loudly."""
    from ..medical_knowledge import cli

    try:
        return cli.selftest_detail()
    except cli.NotInstalled as exc:
        raise _Skip(str(exc)) from exc


def _tone(path: Path, seconds: float = 2.0, rate: int = 22050) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / rate))) for i in range(int(seconds * rate)))
        w.writeframes(frames)


def _pyav_decode() -> str:
    from ..audio import extract_audio_pyav, probe_media_pyav

    with tempfile.TemporaryDirectory(prefix="mlt_selftest_") as tmp:
        src, dst = Path(tmp) / "tone.wav", Path(tmp) / "out.wav"
        _tone(src)
        info = probe_media_pyav(src)
        if abs(float(info.get("duration_s", 0)) - 2.0) > 0.2:
            raise RuntimeError(f"PyAV probe returned {info}")
        extract_audio_pyav(src, dst, {"sample_rate": 16000})
        with wave.open(str(dst), "rb") as w:
            if w.getframerate() != 16000 or w.getnchannels() != 1 or abs(w.getnframes() / 16000 - 2.0) > 0.2:
                raise RuntimeError("decoded WAV has the wrong format")
    return "decoded a 2 s tone to 16 kHz mono with PyAV"


def _ffmpeg() -> str:
    from ..audio import locate_ffmpeg_optional
    from .diagnostics_service import tool_version

    ffmpeg, _probe = locate_ffmpeg_optional()
    if ffmpeg is None:
        return "no ffmpeg.exe found - PyAV decodes instead (fully supported)"
    version = tool_version(ffmpeg, timeout=8.0)
    if version.startswith("did not answer") or "not" in version.lower()[:12]:
        return f"{ffmpeg} does not answer ({version}) - PyAV decodes instead"
    return f"{ffmpeg}: {version}"


def _gpu() -> str:
    from .gpu_service import GpuService

    status = GpuService().detect()
    if status.cuda_available:
        return f"{status.name}: CUDA available, {status.total_mib} MiB"
    return f"CUDA unavailable ({status.cuda_reason or 'no NVIDIA GPU'}); CPU fallback enabled"


def _cpu_fallback() -> str:
    import ctranslate2

    types = ctranslate2.get_supported_compute_types("cpu")
    if not types:
        raise RuntimeError("CTranslate2 reports no CPU compute types")
    return "CPU compute types: " + ", ".join(sorted(types))


def _models() -> str:
    from .model_service import ModelService
    from .paths import get_paths

    models = ModelService(get_paths())
    states = models.list_models()
    parts = [f"{s.title}: {'installed' if s.installed else 'not installed'}" for s in states]
    return "; ".join(parts) or "no models registered"


def _output_dir() -> str:
    from .paths import get_paths

    paths = get_paths().ensure()
    probe = paths.temp / "selftest.tmp"
    probe.write_text("ok", encoding="utf-8")
    probe.unlink()
    out = Path(tempfile.mkdtemp(prefix="mlt_out_")) / "Medical Transcription" / "selftest"
    out.mkdir(parents=True)
    out.rmdir()
    return f"user data folder is writable: {paths.root}"


def _credentials() -> str:
    from .credential_service import CredentialService

    creds = CredentialService()
    return f"{creds.store_label} available" if creds.secure_storage else "no secure credential store (cloud key kept in memory only)"


STUDY_SAMPLE = [
    "Glycolysis is the metabolic pathway that converts glucose into pyruvate in the cytoplasm of the cell.",
    "The first step of glycolysis is catalyzed by hexokinase, which phosphorylates glucose to glucose-6-phosphate.",
    "Hexokinase uses ATP as the phosphate donor, and this reaction traps glucose inside the cell.",
    "Remember that hexokinase is inhibited by its product glucose-6-phosphate.",
    "Phosphofructokinase is the key regulatory enzyme of glycolysis and it is activated by AMP.",
    "Pyruvate kinase deficiency causes hemolytic anemia because red blood cells depend on glycolysis for ATP.",
    "Under anaerobic conditions lactate dehydrogenase converts pyruvate to lactate and regenerates NAD+.",
    "Insulin stimulates the translocation of GLUT4 to the plasma membrane, which increases glucose uptake.",
    "This is important for the exam: red blood cells have no mitochondria and therefore rely only on glycolysis.",
    "Glucokinase is not inhibited by glucose-6-phosphate, unlike hexokinase.",
]


def _study() -> str:
    """The MELETRA Automatic Study Engine is part of the build and works without any model or network (its offline method and the premium gate): summaries, cards and questions from a sample."""
    from ..study.automatic import ENGINE_NAME, AutomaticStudyEngine, EngineChoice
    from ..study.curate import GENERATION_VERSION
    from ..study.extractive import ExtractiveProvider
    from ..study.index import build_index

    segments = [{"start": i * 20.0, "end": i * 20.0 + 19.0, "text": text} for i, text in enumerate(STUDY_SAMPLE * 2)]
    pack = AutomaticStudyEngine(EngineChoice(ExtractiveProvider(), "builtin", "", "self-test")).generate(build_index(segments), lecture_id="selftest", title="selftest")
    summaries = sum(len(v) for v in pack.summaries.values())
    if not (summaries and pack.flashcards and pack.questions):
        raise RuntimeError(f"the study engine made {summaries} summary points, {len(pack.flashcards)} cards, {len(pack.questions)} questions")
    if pack.stats["grounding"]["dropped"]:
        raise RuntimeError("source grounding dropped items of the built-in method")
    if pack.generation.get("engine") != ENGINE_NAME or pack.generation.get("version") != GENERATION_VERSION:
        raise RuntimeError(f"the pack does not name its engine: {pack.generation}")
    unproven = [c.front for c in pack.flashcards if not c.evidence or c.generation != GENERATION_VERSION] + [q.question for q in pack.questions if not q.evidence or "_____" in q.question]
    if unproven:
        raise RuntimeError(f"{len(unproven)} items without evidence or with a blank: {unproven[0]}")
    return f"{summaries} summary points, {len(pack.flashcards)} flashcards, {len(pack.questions)} questions from a sample lecture, every item with its evidence"


def _archive() -> str:
    """The lecture archive code is part of the build: a lecture folder is found from a record that names the chosen folder, its transcript is read (also from files with an older name),
    and its status is the one every screen shows."""
    import tempfile

    from ..archive.reconcile import reconcile_archive
    from ..archive.resolver import lecture_facts, lecture_folder, quick_status
    from .job_service import JobRecord, JobStore

    with tempfile.TemporaryDirectory(prefix="mlt_archive_selftest_") as tmp:
        root = Path(tmp) / "Medical Transcription"
        folder = root / "Lecture"
        folder.mkdir(parents=True)
        (folder / ".job_id").write_text("selftest00000001", encoding="utf-8")
        (folder / "Lecture_clean.txt").write_text(" ".join(STUDY_SAMPLE), encoding="utf-8")                  # the name an earlier version used
        store = JobStore(Path(tmp) / "jobs")
        store.save(JobRecord(job_id="selftest00000001", source="Lecture.mp4", source_name="Lecture.mp4", output_dir=str(root), state="completed", summary={"output_dir": str(folder)}))
        record = store.get("selftest00000001")
        if lecture_folder(record) != folder:
            raise RuntimeError("a record that names the chosen folder did not lead to the lecture's folder")
        facts = lecture_facts(record)
        if not (facts.transcript_available and facts.words):
            raise RuntimeError("the saved transcript was not read")
        report = reconcile_archive(store)
        if report.recovered or report.duplicates_prevented or len(store.list()) != 1:
            raise RuntimeError("the archive repair changed an archive that was in order")
        return f"found the lecture from its record, read {facts.words} words, status '{quick_status(record)[1]}', repair changed nothing"


class _EchoTranslator:
    """A stand-in translator: this check is about the saved-transcript machinery that ships in the build; the real model is the next check, when it is installed."""

    id = "selftest-echo:1"

    def translate(self, texts, source: str, target: str) -> list[str]:
        return [f"[{source}>{target}] {text}" for text in texts]


def _languages() -> str:
    """A recording that changes language: the language runs from window evidence, the canonical source transcript written and read back, a translation made FROM IT (no recording, no
    speech model, no job) and the same request answered from the saved translation. All of it is code of the build, so a missing module or file shows here."""
    from ..multilingual import source as source_mod
    from ..multilingual.timeline import WindowEvidence, build_plan
    from ..schema import Segment
    from ..translation.service import TranslationService

    spec = ["ko"] * 6 + ["hi"] * 6 + ["en"] * 6
    windows = [WindowEvidence(i * 10.0, i * 10.0 + 30.0, {lang: 0.95, "tr": 0.02}, speech_seconds=25.0, rms=0.05) for i, lang in enumerate(spec)]
    plan = build_plan(windows, [], 10.0, windows[-1].end)
    order = [run.language for run in plan.runs]
    if order != ["ko", "hi", "en"]:
        raise RuntimeError(f"the language plan of a Korean -> Hindi -> English recording was {order}")
    texts = {"ko": "안녕하세요 여러분", "hi": "नमस्ते दोस्तों", "en": "Hello everyone"}
    segments = [Segment(i * 5.0, i * 5.0 + 4.0, texts[lang], lang) for i, lang in enumerate(["ko", "hi", "en"])]
    with tempfile.TemporaryDirectory(prefix="mlt_lang_") as folder:
        lecture = Path(folder)
        written = source_mod.from_segments("selftest", segments)
        source_mod.write(lecture, written)
        read_back = source_mod.read_file(lecture)
        if read_back is None or [s.source_text for s in read_back.segments] != list(texts.values()) or read_back.detected_languages != ["ko", "hi", "en"]:
            raise RuntimeError("the source transcript did not come back as it was written (languages / text)")
        service = TranslationService([], engine=_EchoTranslator())
        first = service.translate(lecture, "tr")
        again = service.cached(lecture, "tr")
        if again is None or again.created_at != first.created_at:
            raise RuntimeError("the second request for the same translation was not answered from the saved one")
        if sorted(s.source_language for s in first.segments) != ["en", "hi", "ko"]:
            raise RuntimeError("the translation did not use each segment's own language")
        if source_mod.read_file(lecture).revision != written.revision:
            raise RuntimeError("translating changed the saved source transcript")
    return "3 language runs found, the source transcript saved and read back, a translation made from it and reopened from the cache"


def _translation_pack() -> str:
    """The optional offline translation model: found where the program looks for it; when it is installed one sentence is translated for real (no network)."""
    from ..translation.service import TranslationService
    from .model_service import ModelService, TRANSLATION_PACK_KEY
    from .paths import get_paths

    paths = get_paths()
    states = {s.key: s for s in ModelService(paths).list_models(include_hidden=True, kind="translation")}
    state = states.get(TRANSLATION_PACK_KEY)
    if state is None:
        raise RuntimeError("the translation pack is not in the model registry of this build")
    service = TranslationService([paths.models])
    folder = service.model_dir()
    if folder is None:
        raise _Skip("the optional translation pack is not installed on this computer (it is offered in Settings -> Models and by the installer)")
    engine = service._get_engine()
    started = time.perf_counter()
    out = engine.translate(["The patient has a severe headache and fever."], "en", "tr")
    service.close()
    if not out or not out[0].strip() or out[0].strip() == "The patient has a severe headache and fever.":
        raise RuntimeError(f"the translation model returned {out!r}")
    return f"{folder.name} translated a sentence to Turkish in {time.perf_counter() - started:.1f} s"


def _engines() -> str:
    from ..asr import describe_engines

    rows = describe_engines()
    ready = [r["engine"] for r in rows if r["ready"] == "yes"]
    if "faster_whisper" not in ready and "mlx_whisper" not in ready:
        raise RuntimeError("no speech engine can run here: " + "; ".join(f"{r['engine']}: {r['note']}" for r in rows))
    return "speech engines ready: " + ", ".join(ready)


def _router() -> str:
    """The input router recognises a recording and a text file by their content (no model, no GUI)."""
    from ..ingest.router import detect

    with tempfile.TemporaryDirectory(prefix="mlt_router_") as folder:
        wav, note = Path(folder) / "sample.wav", Path(folder) / "note.txt"
        with wave.open(str(wav), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(16000)
            out.writeframes(b"\x00\x00" * 16000)
        note.write_text("Lecture notes", encoding="utf-8")
        found = [detect(wav), detect(note)]
    if [d.kind for d in found] != ["audio", "text"]:
        raise RuntimeError("the file router did not recognise a WAV recording and a text file: " + ", ".join(d.kind for d in found))
    return "recordings and text files are recognised by their content"


def _ocr() -> str:
    """A rendered line of text is read back through the text recognition of this computer (skipped when it has no language for it)."""
    from PIL import Image, ImageDraw, ImageFont

    from ..ingest.ocr import OcrUnavailable, best_orientation, create_ocr

    try:
        engine = create_ocr("auto")
    except OcrUnavailable as exc:
        raise _Skip(str(exc)) from exc
    image = Image.new("RGB", (1200, 220), "white")
    ImageDraw.Draw(image).text((30, 80), "The afferent arteriole regulates glomerular filtration", fill="black", font=ImageFont.load_default(size=48))
    text = best_orientation(engine, image).text.lower()
    if "afferent" not in text and "arteriole" not in text:
        raise RuntimeError(f"{engine.name} read {text!r} instead of the sample line")
    return f"{engine.name} read a sample line"


def _voices() -> str:
    """Read-aloud voices that are already on this computer; one is asked for a short sentence (skipped when there is none)."""
    from ..tts import TtsService

    service = TtsService()
    voices = service.list_voices()
    if not voices:
        raise _Skip("no speech voice is installed on this computer")
    voice = next((v for v in voices if v.lang == "en"), voices[0])
    with tempfile.TemporaryDirectory(prefix="mlt_tts_") as folder:
        wav = service.synthesize("Medical lecture." if voice.lang == "en" else "Merhaba.", Path(folder) / "voice.wav", voice=voice)
        with wave.open(str(wav)) as handle:
            seconds = handle.getnframes() / float(handle.getframerate())
    if seconds < 0.3:
        raise RuntimeError(f"the voice {voice.name} produced only {seconds:.2f} s of audio")
    return f"{len(voices)} voice(s); {voice.name} spoke {seconds:.1f} s"


class _Skip(Exception):
    """A check that cannot run on this computer for a reason that is not a defect of the build."""


def run_selftest() -> SelfTestReport:
    report = SelfTestReport(app={"name": APP_NAME, "version": __version__, "frozen": FROZEN})
    for name, fn in (("Libraries", _modules), ("Resources", _resources), ("Medical Expert", _medical), ("Medical knowledge", _knowledge), ("PyAV decode", _pyav_decode), ("FFmpeg", _ffmpeg),
                     ("GPU / CUDA", _gpu), ("CPU fallback", _cpu_fallback), ("Model discovery", _models), ("Languages", _languages), ("Translation pack", _translation_pack),
                     ("User data folder", _output_dir), ("Credential store", _credentials), ("Study engine", _study), ("Lecture archive", _archive),
                     ("Speech engines", _engines), ("File router", _router), ("Text recognition", _ocr), ("Read aloud", _voices)):
        _run(report, name, fn)
    return report
