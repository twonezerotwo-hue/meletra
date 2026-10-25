"""The bridge between the GUI and the transcription pipeline.

The GUI describes WHAT it wants in a :class:`JobRequest` (plain data, saved as JSON). A separate worker process
(:func:`run_worker`) turns it into pipeline options, runs :func:`src.pipeline.process_files` and reports through two small
files in the app's ``jobs`` folder - no console, no pipes, so it behaves the same from source and from the packaged .exe:

* ``<job>.events.jsonl``  one JSON event per line (see :mod:`src.events`), written by the worker, tailed by the GUI;
* ``<job>.stop``          created by the GUI to ask for a safe pause (checked after every saved segment).

If the GUI disappears the worker pauses safely by itself. The pipeline is the same code the command line uses.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Optional

from .. import FROZEN
from .credential_service import CredentialService
from .gpu_service import GpuService
from .job_service import JobStore, pid_alive
from .model_service import ModelService
from .paths import AppPaths, default_output_dir

log = logging.getLogger(__name__)

EXIT_COMPLETED, EXIT_FAILED, EXIT_PAUSED = 0, 1, 10
STOP_POLL_SECONDS = 0.4                  # how often the worker looks for the stop file (tests set 0)


@dataclass
class JobRequest:
    source: str
    output_dir: str = ""                  # root folder; the pipeline adds one sub-folder per recording
    model_key: str = "whisper-large-v3"
    language: str = "auto"
    subject: str = "auto"
    mode: str = "quality"
    device: str = "auto"
    compute_type: str = "auto"
    beam_size: int = 0
    vad: bool = True
    chunk_minutes: float = 15.0
    overlap_seconds: float = 3.0
    medical_context: bool = True
    hotwords: str = "auto"
    pdf_path: str = ""
    use_pdf: bool = True
    cpu_threads: int = 0
    cloud_compare: bool = False           # upload to Deepgram and compare - only when the user enabled it
    resume: bool = True
    unload_ollama: bool = False
    job_id: str = ""
    data_root: str = ""
    extra_model_dirs: list[str] = field(default_factory=list)
    parent_pid: int = 0
    medical_expert: bool = True
    custom_terms: list[str] = field(default_factory=list)
    medical_intelligence: str = "off"     # v4 layer: off | shadow | on
    output_language: str = "en"           # the language of the final transcript / study material; "" = the spoken language (no translation)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=1, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "JobRequest":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


# ---------------------------------------------------------------------------------------------- IPC files
def events_path(paths: AppPaths, job_id: str) -> Path:
    return paths.jobs / f"{job_id}.events.jsonl"


def stop_path(paths: AppPaths, job_id: str) -> Path:
    return paths.jobs / f"{job_id}.stop"


def request_path(paths: AppPaths, job_id: str) -> Path:
    return paths.jobs / f"{job_id}.request.json"


def request_stop(paths: AppPaths, job_id: str) -> None:
    stop_path(paths, job_id).write_text(str(time.time()), encoding="utf-8")


class FileEventSink:
    """Worker side: append every event as one JSON line (flushed at once so the GUI sees it within a quarter second)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")
        self._lock = threading.Lock()

    def __call__(self, event: dict[str, Any]) -> None:
        line = json.dumps(event, ensure_ascii=False)
        with self._lock, self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")


class EventTail:
    """GUI side: new events since the last call. An unfinished last line is left for the next call."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._offset = 0

    def poll(self) -> list[dict[str, Any]]:
        try:
            with self.path.open("rb") as handle:
                handle.seek(self._offset)
                data = handle.read()
        except OSError:
            return []
        end = data.rfind(b"\n")
        if end < 0:
            return []
        self._offset += end + 1
        events = []
        for raw in data[:end].splitlines():
            try:
                events.append(json.loads(raw.decode("utf-8")))
            except ValueError:
                continue
        return events


# ---------------------------------------------------------------------------------------------- request -> pipeline options
def resolve_output_name(output_root: Path, source: Path) -> str:
    """Sub-folder for the results of ``source``: its own name, or ``name (2)``, ``name (3)`` ... when a folder of that name already
    holds the results of a DIFFERENT recording (two lectures called "lecture.mp4" must not overwrite each other). The same recording
    always gets the same folder back, so a resumed or repeated job stays where it was."""
    from ..exporters import safe_name
    from ..pipeline import OUTPUT_MARKER, job_key

    stem = safe_name(source.stem)
    try:
        mine = job_key(source)
    except OSError:                                   # the file is not there (yet): no folder can belong to it
        return stem
    candidate, number = stem, 1
    while True:
        folder = Path(output_root) / candidate
        if not folder.exists() or not any(folder.iterdir()):
            return candidate
        marker = folder / OUTPUT_MARKER
        try:
            owner = marker.read_text(encoding="utf-8").strip() if marker.is_file() else ""
        except OSError:
            owner = ""
        if owner == mine:
            return candidate
        number += 1
        candidate = f"{stem} ({number})"


def build_run_options(request: JobRequest, cfg: dict[str, Any], models: ModelService, paths: AppPaths,
                      deepgram_key: Optional[str] = None) -> tuple[Any, dict[str, Any]]:
    """``(RunOptions, cfg)`` for the pipeline. ``cfg`` is a private copy; the shipped configuration is never modified."""
    from ..pipeline import RunOptions, resolve_subjects

    cfg = copy.deepcopy(cfg)
    mode = request.mode if request.mode in ("quality", "fast") else "quality"
    profile = cfg[mode]
    if request.compute_type != "auto":
        profile["compute_type"] = request.compute_type
    if request.beam_size > 0:
        profile["beam_size"] = request.beam_size
    profile["vad_filter"] = bool(request.vad)
    cfg["long_audio"]["overlap_seconds"] = float(request.overlap_seconds)

    context_on = bool(request.medical_context)
    subjects = ["general_medical"] if request.subject in ("", "auto") else resolve_subjects(request.subject, cfg)
    backend = "compare" if (request.cloud_compare and deepgram_key) else "local"
    language = None if request.language in ("", "auto", "mixed") else request.language
    output_root = Path(request.output_dir) if request.output_dir else default_output_dir(Path(request.source))
    options = RunOptions(
        mode=mode, language=language, subjects=subjects, model_spec=models.model_spec(request.model_key),
        device=request.device, preprocessing="none", use_pdf=bool(request.use_pdf and context_on),
        pdf_path=Path(request.pdf_path) if request.pdf_path else None, outputs_root=output_root,
        use_hotwords=bool(context_on and request.hotwords != "off"), auto_topic=True if context_on else False,
        resume=bool(request.resume), keep_checkpoint=False, quality_audit=True,
        context_engine="expert" if context_on else "legacy", chunk_minutes=float(request.chunk_minutes),
        backend=backend, checkpoint_root=paths.checkpoints, temp_root=paths.temp, deepgram_api_key=deepgram_key,
        cpu_threads=int(request.cpu_threads), output_name=resolve_output_name(output_root, Path(request.source)),
        multilingual=(request.language == "mixed"),
        medical_expert=bool(request.medical_expert), custom_medical_terms=list(request.custom_terms),
        medical_intelligence=request.medical_intelligence if request.medical_intelligence in ("off", "shadow", "on") else "off",
        output_language=request.output_language or None,
        translation_model_dirs=[paths.models, *[Path(d) for d in request.extra_model_dirs]],
    )
    return options, cfg


def load_config() -> dict[str, Any]:
    from ..config import load_transcription_config

    return load_transcription_config()


# ---------------------------------------------------------------------------------------------- the worker process
class _EventLogHandler(logging.Handler):
    """Forward pipeline log records to the GUI's live log panel (INFO and above)."""

    def __init__(self, sink: Callable[[dict[str, Any]], None]) -> None:
        super().__init__(level=logging.INFO)
        self.sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.sink({"type": "log", "level": record.levelname, "text": record.getMessage()[:600], "time": round(time.time(), 3)})
        except Exception:  # noqa: BLE001 - logging must never raise
            pass


def run_worker(request_file: Path) -> int:
    """Entry point of the worker process. Returns the process exit code (0 done, 10 paused, 1 failed)."""
    request = JobRequest.load(request_file)
    paths = AppPaths(Path(request.data_root)).ensure()
    job_id = request.job_id
    sink = FileEventSink(events_path(paths, job_id))
    store = JobStore(paths.jobs)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    file_handler = logging.FileHandler(paths.job_log(job_id), mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    root.addHandler(file_handler)
    root.addHandler(_EventLogHandler(sink))
    for noisy in ("httpx", "httpcore", "urllib3", "filelock", "numba"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    stop_file = stop_path(paths, job_id)
    stop_file.unlink(missing_ok=True)
    last_check = [0.0, False]

    def should_stop() -> bool:
        now = time.monotonic()
        if now - last_check[0] >= STOP_POLL_SECONDS:         # cheap: a file check at most 2-3 times per second
            last_check[0] = now
            gui_gone = request.parent_pid > 0 and not pid_alive(request.parent_pid)
            last_check[1] = stop_file.exists() or gui_gone
        return bool(last_check[1])

    def on_event(event: dict[str, Any]) -> None:
        sink(event)
        try:
            store.apply_event(job_id, event)
        except OSError:
            pass

    store.mark_running(job_id, os.getpid())
    try:
        if request.unload_ollama:
            stopped = GpuService().unload_ollama()
            if stopped:
                log.info("Unloaded idle Ollama model(s): %s", ", ".join(stopped))
        models = ModelService(paths, extra_dirs=request.extra_model_dirs, discover=True)
        deepgram_key = CredentialService().get_deepgram_key() if request.cloud_compare else None
        if request.cloud_compare and not deepgram_key:
            on_event({"type": "warning", "code": "no_api_key", "time": round(time.time(), 3),
                      "message": "Cloud comparison is on but no Deepgram API key is stored; running locally only."})
        cfg = load_config()
        options, cfg = build_run_options(request, cfg, models, paths, deepgram_key)
        options.events, options.should_stop = on_event, should_stop
        from ..pipeline import process_files

        [summary] = process_files([Path(request.source)], options, cfg)
        if summary.paused:
            return EXIT_PAUSED
        return EXIT_COMPLETED if summary.ok else EXIT_FAILED
    except BaseException as exc:  # noqa: BLE001 - the GUI must always learn why a job ended
        log.exception("Job %s failed", job_id)
        failed = {"type": "failed", "error": _friendly_error(exc), "kind": type(exc).__name__, "time": round(time.time(), 3)}
        if not _already_reported_failure(events_path(paths, job_id)):
            sink(failed)
        store.mark_failed(job_id, failed["error"])
        return EXIT_FAILED
    finally:
        sink({"type": "worker_exit", "time": round(time.time(), 3)})
        root.removeHandler(file_handler)
        file_handler.close()


def _already_reported_failure(path: Path) -> bool:
    try:
        return '"type": "failed"' in Path(path).read_text(encoding="utf-8")
    except OSError:
        return False


def _friendly_error(exc: BaseException) -> str:
    """A message a user can act on (the full traceback is in the job log)."""
    text = str(exc).strip() or type(exc).__name__
    lowered = text.lower()
    if "cublas" in lowered or "cudnn" in lowered or "cudart" in lowered:
        return ("A CUDA runtime library is missing or incompatible. Switch Settings -> Performance -> Device to CPU, "
                "or update the NVIDIA driver. Details: " + text.splitlines()[0][:200])
    if "out of memory" in lowered:
        return "The GPU ran out of memory. Close other GPU programs or choose Fast mode / CPU. Details: " + text.splitlines()[0][:200]
    return text.splitlines()[0][:400] if "\n" in text else text[:400]


def worker_command(request_file: Path) -> tuple[str, list[str]]:
    """``(program, args)`` that start a worker: the packaged .exe itself, or ``python -m`` from a source checkout."""
    if FROZEN:
        return sys.executable, ["--worker", str(request_file)]
    return sys.executable, ["-m", "src.services.worker", str(request_file)]
