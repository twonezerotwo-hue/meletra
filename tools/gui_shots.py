"""Visual QA: render the real MELETRA window (offscreen) with sample lectures and save one PNG per page and size.

    python tools/gui_shots.py OUT_DIR [--sizes 1366x768,1440x900,1920x1080] [--theme dark|light] [--pages home,files,...]

Only the program's own widgets are rendered (``QWidget.grab``) - never the screen. The sample data is made in a temporary folder; nothing of the student's data is read.
Pages: home_empty home home_file transcription_active transcription_queue transcription_result transcript study files files_empty settings_<tab> drag
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if sys.platform == "win32":
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")                       # the offscreen platform has no fonts of its own
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QMimeData, QPoint, QUrl, Qt  # noqa: E402
from PySide6.QtGui import QDragEnterEvent, QDragLeaveEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

DEFAULT_SIZES = ("1366x768", "1440x900", "1920x1080")
SETTINGS_TABS = ("General", "Transcription", "Medical", "Study", "Privacy & Data", "Models", "Advanced")


def pump(app, times: int = 4) -> None:
    for _ in range(times):
        app.processEvents()


def sample_services(tmp: Path, populated: bool = True):
    from tests.gui_support import make_services
    from tests.study_support import write_lecture_folder

    services = make_services(tmp, opened=[])
    if not populated:
        return services

    def lecture(stem, **fields):
        folder = write_lecture_folder(tmp / "lectures" / stem)
        record = services.jobs.create_or_get("job-" + stem, tmp / f"{stem}.mp4", output_dir=str(folder), state="completed", progress=100, **fields)
        services.jobs.update(record.job_id, state="completed")
        return record

    lecture("FIXTURE_0004", display_name="Neuroanatomy — Ventricular System", language="auto", detected_language="en", output_language="tr",
            translation="m2m100", duration_seconds=2700)
    lecture("FIXTURE_0003", language="auto", detected_language="en", output_language="en", translation="none", duration_seconds=1860)
    lecture("Pharma_lecture3", display_name="Pharmacology: autonomic drugs", language="tr", detected_language="tr", output_language="en",
            translation="whisper", duration_seconds=3300)
    lecture("Histology_week2", language="de", detected_language="de", output_language="en", translation="whisper", duration_seconds=4020)
    services.jobs.create_or_get("job-run", tmp / "Biochemistry_glycolysis.mp4", state="running", progress=37.0, language="auto", output_language="en",
                                duration_seconds=3000)
    services.jobs.create_or_get("job-queued", tmp / "Physiology_renal.mp4", state="queued", language="auto", output_language="en", duration_seconds=2400)
    services.jobs.create_or_get("job-failed", tmp / "Anatomy_thorax.mp4", state="failed", language="auto", output_language="en", duration_seconds=1500)
    services.jobs.create_or_get("job-paused", tmp / "Immunology_B_cells.mp4", state="paused", progress=62.0, language="auto", output_language="en",
                                duration_seconds=3600)
    return services


def translated_lecture(folder: Path) -> Path:
    from tests.study_support import write_lecture_folder

    folder = write_lecture_folder(folder)
    meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    meta["translation"] = {"method": "whisper", "source_language": "tr", "output_language": "en", "reason": "x", "segments": 2, "seconds": 3.0, "error": None,
                           "files": ["translated_en_transcript.txt"]}
    (folder / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    lines = ["The heart pumps blood.", "The lungs exchange gas.", "Acetylcholine is the main parasympathetic transmitter."]
    (folder / "translated_en_transcript.txt").write_text(chr(10).join(lines) + chr(10), encoding="utf-8")
    return folder


def make_window(tmp: Path, theme: str, populated: bool = True):
    from src.gui import theme as theme_mod
    from src.gui.main_window import MainWindow
    from tests.gui_support import ScriptedLauncher

    app = QApplication.instance() or QApplication([])
    theme_mod.apply_theme(app, theme)
    app.setWindowIcon(theme_mod.app_icon())
    services = sample_services(tmp, populated)
    window = MainWindow(services, ScriptedLauncher())
    return app, window, services


def save(window, app, out: Path, name: str, size: str) -> Path:
    window.toast.hide()                                                                    # the start-up notice is not part of the page being inspected
    pump(app)
    path = out / f"{name}_{size}.png"
    window.grab().save(str(path))
    return path


def render(out: Path, sizes, theme: str, pages) -> list[Path]:
    from tests.conftest import write_silent_wav
    from tests.study_support import write_lecture_folder

    out.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for populated in (False, True):
        tmp = Path(tempfile.mkdtemp(prefix="wm_shots_"))
        app, window, services = make_window(tmp, theme, populated)
        for size in sizes:
            width, height = (int(v) for v in size.split("x"))
            window.resize(width, height)
            window.show()
            pump(app)

            def want(name: str) -> bool:
                return not pages or name in pages or any(p == name.split("_")[0] for p in pages)

            if not populated:
                if want("home_empty"):
                    window.show_home()
                    saved.append(save(window, app, out, "home_empty", size))
                if want("files_empty"):
                    window._goto("files")
                    window.files_page.refresh()
                    saved.append(save(window, app, out, "files_empty", size))
                continue
            window.show_home()
            window.simple_home.refresh()
            if want("home"):
                saved.append(save(window, app, out, "home", size))
            if want("home_file"):
                media = tmp / "FIXTURE_0004.wav"
                write_silent_wav(media, seconds=30)
                window.add_dropped_files([str(media)])
                for _ in range(60):
                    pump(app, 2)
                    time.sleep(0.02)
                saved.append(save(window, app, out, "home_file", size))
                window.simple_home.engine.reset_file()
                window.simple_home.refresh()
            if want("drag"):
                mime = QMimeData()
                mime.setUrls([QUrl.fromLocalFile(str(tmp / "FIXTURE_0004.mp4"))])
                event = QDragEnterEvent(QPoint(300, 300), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
                window.dragEnterEvent(event)
                for _ in range(20):
                    pump(app, 2)
                    time.sleep(0.02)
                saved.append(save(window, app, out, "drag", size))
                window.dragLeaveEvent(QDragLeaveEvent())
            if want("transcription_active"):
                window._goto("active")
                window.simple_active.begin("FIXTURE_0004.MP4", "auto", "auto", "video")
                window.simple_active.set_progress({"percent": 68.0, "rtf": 6.0, "eta": 240})
                saved.append(save(window, app, out, "transcription_active", size))
            if want("transcription_queue"):
                from src.ingest import queue as q
                from src.ingest.router import classify

                files = []
                for name in ("Neuroanatomy", "Histology", "Physiology", "Pharmacology"):
                    path = tmp / f"{name}.wav"
                    write_silent_wav(path, seconds=3)
                    files.append(str(path))
                queue = q.BatchQueue(classify(files).items, start=lambda item: None, pause=lambda: None, cancel=lambda: None)
                queue.items[0].state, queue.items[1].state = q.COMPLETED, q.PROCESSING
                queue.state = q.RUNNING
                window._goto("batch")
                window.simple_batch.begin(queue)
                window.simple_batch.progress_bar.setValue(420)
                window.simple_batch.percent_label.setText("42%")
                window.simple_batch.stage.setText("Transcribing...")
                saved.append(save(window, app, out, "transcription_queue", size))
            if want("transcription_result"):
                folder = write_lecture_folder(tmp / "result_lecture")
                window.show_finished(folder, "FIXTURE_0004.MP4")
                saved.append(save(window, app, out, "transcription_result", size))
                folder = translated_lecture(tmp / "translated_lecture")
                window.show_finished(folder, "Pharmacology: autonomic drugs")
                saved.append(save(window, app, out, "transcription_result_translated", size))
            if want("transcript"):
                folder = write_lecture_folder(tmp / "transcript_lecture")
                window.transcript_view.show_transcript(folder, "Neuroanatomy — Ventricular System")
                window._goto("transcript")
                saved.append(save(window, app, out, "transcript", size))
                folder = translated_lecture(tmp / "translated_lecture2")
                window.transcript_view.show_transcript(folder, "Pharmacology: autonomic drugs", prefer_translated=True)
                window._goto("transcript")
                saved.append(save(window, app, out, "transcript_translated", size))
            if want("study"):
                folder = write_lecture_folder(tmp / "study_lecture")
                window.show_finished(folder, "Neuroanatomy.mp4")
                window.open_study("overview")
                from src.gui import diagnostics_probe as probe
                for _ in range(400):
                    pump(app, 2)
                    time.sleep(0.03)
                    if probe.study_ready(window):
                        break
                saved.append(save(window, app, out, "study", size))
            if want("study_tabs"):
                folder = write_lecture_folder(tmp / "study_tabs_lecture")
                window.show_finished(folder, "Neuroanatomy.mp4")
                window.open_study("overview")
                from src.gui import diagnostics_probe as probe
                for _ in range(400):
                    pump(app, 2)
                    time.sleep(0.03)
                    if probe.study_ready(window):
                        break
                for tab in ("summary", "concepts", "flashcards", "questions", "study"):
                    window.study.select_tab(tab)
                    for _ in range(10):
                        pump(app, 2)
                    saved.append(save(window, app, out, "study_tab_" + tab, size))
                # states inside the tabs: a revealed flashcard, a Study Mode question, its feedback, the results
                window.study.select_tab("flashcards")
                window.study.flashcards.show_button.click()
                for _ in range(6):
                    pump(app, 2)
                saved.append(save(window, app, out, "study_flashcard_answer", size))
                window.study.select_tab("study")
                mode = window.study.study_mode
                mode.start()
                for _ in range(6):
                    pump(app, 2)
                saved.append(save(window, app, out, "study_mode_question", size))
                for _ in range(10):                                         # the next multiple-choice question, for the picture
                    if mode.option_buttons:
                        break
                    mode.skip()
                for _ in range(6):
                    pump(app, 2)
                saved.append(save(window, app, out, "study_mode_choices", size))
                if mode.option_buttons:
                    mode.choose(0)
                else:
                    mode.text_edit.setPlainText("some words about glycolysis")
                    mode.submit()
                for _ in range(6):
                    pump(app, 2)
                saved.append(save(window, app, out, "study_mode_feedback", size))
                mode.end()
                for _ in range(6):
                    pump(app, 2)
                saved.append(save(window, app, out, "study_mode_results", size))
            if want("files"):
                window._goto("files")
                window.files_page.refresh()
                saved.append(save(window, app, out, "files", size))
            if want("dialogs"):
                from src.gui.about_dialog import AboutDialog
                from src.gui.first_run import FirstRunDialog
                from src.gui.sharing_notice import SharingNoticeDialog

                from src.gui.components import ConfirmDialog

                confirm = ConfirmDialog(window, "Remove from archive", "Remove 'Neuroanatomy' from the lecture list?" + chr(10) + chr(10) + "Its transcript, study material and recording stay on your computer.",
                                        accept="Remove from archive")
                for name, dialog in (("about", AboutDialog(window)), ("first_run", FirstRunDialog(services, window)), ("sharing_notice", SharingNoticeDialog(services, window)),
                                     ("confirm", confirm)):
                    dialog.show()
                    pump(app)
                    path = out / f"dialog_{name}_{size}.png"
                    dialog.grab().save(str(path))
                    saved.append(path)
                    dialog.close()
            for tab in SETTINGS_TABS:
                if not want("settings_" + tab):
                    continue
                dialog = window.make_settings_dialog(tab)
                dialog.resize(min(width - 80, 980), min(height - 80, 660))
                dialog.show()
                pump(app)
                path = out / f"settings_{tab.split()[0].lower()}_{size}.png"
                dialog.grab().save(str(path))
                saved.append(path)
                dialog.close()
        window.close()
    return saved


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--sizes", default=",".join(DEFAULT_SIZES))
    ap.add_argument("--theme", default="dark")
    ap.add_argument("--pages", default="")
    args = ap.parse_args(argv)
    pages = {p for p in args.pages.split(",") if p}
    saved = render(args.out, args.sizes.split(","), args.theme, pages)
    for path in saved:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
