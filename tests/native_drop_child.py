"""Run by tests/test_native_drop.py in its own process with the real Windows platform plugin (the main test process uses the offscreen one, which has no OLE).

    python tests/native_drop_child.py RESULT.json

Builds the real window with a private data folder, puts it far outside the visible desktop, runs the ``--ui-check`` steps "native_drop" and writes the report."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.pop("QT_QPA_PLATFORM", None)                    # the real platform: windows

from PySide6.QtWidgets import QApplication  # noqa: E402

from src.gui import theme  # noqa: E402
from src.gui.main_window import MainWindow  # noqa: E402
from src.gui.ui_check import UiCheck  # noqa: E402
from tests.gui_support import ScriptedLauncher, make_services  # noqa: E402


def main() -> int:
    result = Path(sys.argv[1])
    tmp = Path(tempfile.mkdtemp(prefix="wm_native_child_"))
    app = QApplication([])
    theme.apply_theme(app, "dark")
    services = make_services(tmp, opened=[])
    window = MainWindow(services, ScriptedLauncher())
    window.resize(1366, 768)
    window.move(-6000, 100)                                # nobody has to look at it
    check = UiCheck(window, tmp / "out", pages="native_drop", exit_when_done=True)
    code = app.exec()
    report = json.loads((tmp / "out" / "ui_check.json").read_text(encoding="utf-8")) if (tmp / "out" / "ui_check.json").exists() else {"errors": ["no report"]}
    result.write_text(json.dumps(report, indent=1), encoding="utf-8")
    del check
    return code


if __name__ == "__main__":
    raise SystemExit(main())
