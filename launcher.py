"""Entry point of the desktop application (and of the packaged MELETRA.exe).

    MELETRA.exe                      open the window
    MELETRA.exe --worker REQUEST     run one transcription job (started by the window itself)
    MELETRA.exe --selftest [OUT]     check the installation without opening a window; writes JSON to OUT
    MELETRA.exe --diagnostics [OUT]  write the diagnostics report as text
    MELETRA.exe --medical-knowledge-info [OUT] [--knowledge-db FILE]
                                                       version, size, counts, sources + licences, languages, health checks of the medical knowledge
                                                       database (OUT ending in .json: JSON). Exit 0 healthy, 1 damaged / missing but required, 2 none installed
    MELETRA.exe --download-model KEY [--result OUT]
                                                       download one model without a window (resumable, SHA-256 checked)
    MELETRA.exe --autorun-queue "A|B|C" --result OUT [--timeout S] [--language en]
                                                       drive the real window for several files / documents / pictures (batch queue)
    MELETRA.exe --autorun FILE --result OUT [--timeout S] [--language en] [--pause-at PERCENT]
                                                       drive the real window for one file (used to test the packaged build);
                                                       --pause-at also presses PAUSE SAFELY at that progress, then RESUME
    MELETRA.exe --autorun-translate FOLDER --to CODE --result OUT [--timeout S]
                                                       open a finished lecture, press the real "Translate to another language" button, translate its SAVED transcript, open the saved
                                                       translation a second time (no file, no job, no speech model); the result lists what happened
    MELETRA.exe --translate-lecture FOLDER --to CODE [--result OUT]
                                                       translate the saved transcript of a lecture without a window (offline; the same code as the window)
    MELETRA.exe --autorun-study FOLDER --result OUT [--timeout S]
                                                       open a finished lecture's folder in the Study workspace (with MLT_UI_TRACE=1 the
                                                       result lists the tabs, stat cards and buttons the window shows)
    MELETRA.exe --autorun-archive RECORDING_NAME --result OUT [--study [--exercise]] [--timeout S]
                                                       open ONE lecture from the archive of the real window (looked up by the file name of its recording): the Files list, the click on
                                                       the card, Technical details, Read Transcript (inside the window, no outside program) and, with --study, its Study material;
                                                       the result holds lengths and fingerprints, never the lecture's text
    MELETRA.exe --ui-check OUT_DIR [--size 1366x768] [--ui-pages a,b] [--lecture FOLDER] [--translated FOLDER]
                                                       look at the real window like assistive technology and a person do: accessibility tree, Tab order,
                                                       a drag session against the real Windows drop target, and (for an outside script that photographs the
                                                       window) the pages one by one. Writes OUT_DIR/ui_check.json. Nothing is transcribed or sent.
"""

from __future__ import annotations

import multiprocessing
import os
import sys
from pathlib import Path


def _option(argv: list[str], name: str, default: str = "") -> str:
    if name in argv:
        i = argv.index(name)
        if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            return argv[i + 1]
    return default


def _write(text: str, target: str) -> None:
    if target:
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        Path(target).write_text(text, encoding="utf-8")
    elif sys.stdout is not None:
        sys.stdout.write(text)


def _ensure_streams() -> None:
    """A windowed .exe has no console: stdout/stderr are None and any library that prints would crash."""
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))


def main(argv: list[str]) -> int:
    _ensure_streams()
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("DO_NOT_TRACK", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    args = argv[1:]
    if args[:1] == ["--worker"]:
        if len(args) < 2:
            return 2
        from src.services.transcription_service import run_worker

        return run_worker(Path(args[1]))
    if "--selftest" in args:
        from src.services.selftest import run_selftest

        report = run_selftest()
        _write(report.to_json(), _option(args, "--selftest"))
        return 0 if report.ok else 1
    if "--medical-knowledge-info" in args:
        from src.medical_knowledge.cli import run_command

        return run_command(_option(args, "--medical-knowledge-info"), _option(args, "--knowledge-db") or None)
    if "--diagnostics" in args:
        from src.services import diagnostics_service
        from src.services.gpu_service import GpuService
        from src.services.model_service import ModelService
        from src.services.paths import get_paths

        paths = get_paths().ensure()
        text = diagnostics_service.as_text(diagnostics_service.collect(paths, ModelService(paths), GpuService()))
        _write(text, _option(args, "--diagnostics"))
        return 0
    if "--download-model" in args:
        from src.services.model_cli import download_command

        return download_command(_option(args, "--download-model"), _option(args, "--result"))
    if "--translate-lecture" in args:
        from src.translation.cli import translate_lecture_command

        return translate_lecture_command(_option(args, "--translate-lecture"), _option(args, "--to"), _option(args, "--result"))
    autorun_translate = None
    if "--autorun-translate" in args:
        autorun_translate = {"folder": _option(args, "--autorun-translate"), "target": _option(args, "--to", "en"), "result": _option(args, "--result", "autorun_result.json"),
                             "timeout": float(_option(args, "--timeout", "900"))}
    autorun = None
    if "--autorun" in args:
        autorun = {"source": _option(args, "--autorun"), "result": _option(args, "--result", "autorun_result.json"),
                   "timeout": float(_option(args, "--timeout", "900")), "language": _option(args, "--language", "auto"),
                   "pause_at": float(_option(args, "--pause-at", "0"))}
    autorun_queue = None
    if "--autorun-queue" in args:
        autorun_queue = {"paths": [p for p in _option(args, "--autorun-queue").split("|") if p],
                         "result": _option(args, "--result", "autorun_result.json"), "timeout": float(_option(args, "--timeout", "900")),
                         "language": _option(args, "--language", "auto")}
    autorun_study = None
    if "--autorun-study" in args:
        autorun_study = {"folder": _option(args, "--autorun-study"), "result": _option(args, "--result", "autorun_result.json"),
                         "timeout": float(_option(args, "--timeout", "300"))}
    autorun_archive = None
    if "--autorun-archive" in args:
        autorun_archive = {"recording": _option(args, "--autorun-archive"), "result": _option(args, "--result", "autorun_result.json"), "study": "--study" in args, "exercise": "--exercise" in args,
                           "timeout": float(_option(args, "--timeout", "300"))}
    ui_check = None
    if "--ui-check" in args:
        ui_check = {"out": _option(args, "--ui-check"), "size": _option(args, "--size"), "pages": _option(args, "--ui-pages"),
                    "lecture": _option(args, "--lecture"), "translated": _option(args, "--translated")}
    from src.gui.app import run

    return run([sys.argv[0]], autorun=autorun, autorun_queue=autorun_queue, autorun_study=autorun_study, ui_check=ui_check, autorun_translate=autorun_translate,
               autorun_archive=autorun_archive)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main(sys.argv))
