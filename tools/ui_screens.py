"""Render the screens that 4.1.2 added or changed, offscreen, at the three target window sizes, from SYNTHETIC lectures (nothing real, nothing sent anywhere).

    python tools/ui_screens.py --out logs/ui_screens [--sizes 1366x768,1440x900,1920x1080]

Writes one PNG per screen and size. It is a look-at-it tool for people (and for the release checklist), not a pass/fail test: the geometry rules are in tests/test_gui_final_polish.py.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, ROOT.as_posix())
if sys.platform == "win32" and Path(r"C:\Windows\Fonts").is_dir():
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

from PySide6.QtWidgets import QApplication  # noqa: E402

from src.gui import diagnostics_probe as probe  # noqa: E402
from src.gui import theme  # noqa: E402
from src.gui.main_window import MainWindow  # noqa: E402
from src.gui.sharing_notice import PrivacyChoiceDialog, SharingPreviewDialog  # noqa: E402
from src.study.providers import ProviderError  # noqa: E402
from tests.archive_support import finished_record, write_lecture  # noqa: E402
from tests.gui_support import ScriptedLauncher, make_services, settle, wait_for  # noqa: E402
from tests.study_support import write_lecture_folder  # noqa: E402


def pump(times: int = 12) -> None:
    for _ in range(times):
        QApplication.processEvents()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("logs/ui_screens"))
    parser.add_argument("--sizes", default="1366x768,1440x900,1920x1080")
    parser.add_argument("--theme", default="dark")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    theme.apply_theme(app, args.theme)
    tmp = Path(tempfile.mkdtemp(prefix="mlt_screens_"))
    services = make_services(tmp, opened=[])

    def no_server(url, headers, body, timeout):                    # the pictures show the built-in engine; they must not depend on (or start) a model on this computer
        raise ProviderError("no AI server (screens are rendered offline)")

    services.study.transport = no_server
    root = tmp / "Medical Transcription"
    complete = write_lecture_folder(root / "FIXTURE_0005")
    finished_record(services.jobs, root, complete, source="FIXTURE_0005.MP4", job_id="job0000000000001")
    legacy = write_lecture(root, "IMG_0002", source="IMG_0002.MP4", job_id="job0000000000002", files="legacy")
    finished_record(services.jobs, root, legacy, source="IMG_0002.MP4", job_id="job0000000000002")
    gone = write_lecture(root, "IMG_0003", source="IMG_0003.MP4", job_id="job0000000000003", files="empty")
    finished_record(services.jobs, root, gone, source="IMG_0003.MP4", job_id="job0000000000003")
    services.settings.update(first_run_done=True, share_choice_made=True)
    window = MainWindow(services, ScriptedLauncher())
    window.show()
    written: list[Path] = []

    def shot(name: str, widget, size: tuple[int, int]) -> None:
        pump()
        path = args.out / f"{name}_{size[0]}x{size[1]}.png"
        widget.grab().save(str(path))
        written.append(path)

    for text in args.sizes.split(","):
        w, h = (int(v) for v in text.lower().split("x"))
        size = (w, h)
        window.resize(w, h)
        window.show_home()
        window.simple_home.refresh()
        shot("home", window, size)
        window.files_page.refresh()
        window._goto("files")
        shot("files", window, size)
        window.open_lecture_record(services.jobs.get("job0000000000001"))
        shot("result", window, size)
        window.open_transcript()
        shot("reader", window, size)
        window.open_study("overview")
        wait_for(lambda: probe.study_ready(window), timeout=120, what="the study material")
        pump(20)
        for tab in ("overview", "flashcards", "questions"):
            window.study.select_tab(tab)
            shot(f"study_{tab}", window, size)
        for page in ("Study", "Privacy & Data"):
            dialog = window.make_settings_dialog(page)
            dialog.resize(min(w - 80, 1100), min(h - 60, 760))
            if page == "Study":
                dialog.study_advanced_check.setChecked(True)
                dialog.study_service_combo.setCurrentIndex(dialog.study_service_combo.findData("anthropic"))
                wait_for(lambda: dialog.study_details.text() and "Looking" not in dialog.study_details.text(), timeout=20, what="the engine details")
            dialog.show()
            shot("settings_" + page.split()[0].lower(), dialog, size)
            settle(dialog)
            dialog.close()
            dialog.deleteLater()
        choice = PrivacyChoiceDialog(services)
        choice.show()
        shot("first_launch_question", choice, size)
        choice.close()
        choice.deleteLater()
        preview = SharingPreviewDialog(services.sharing.preview(complete, "FIXTURE_0005.MP4"))
        preview.show()
        shot("what_would_be_shared", preview, size)
        preview.close()
        preview.deleteLater()
    settle(window, window.home, window.study)
    window.close()
    print("\n".join(str(p) for p in written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
