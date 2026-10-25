"""The drop path of the real window against the REAL Windows drop target (OLE) with a REAL Explorer data object,.

What this proves: Qt's own OLE drop target of the window receives a genuine shell ``IDataObject`` (``CF_HDROP``), converts it, the application shows the overlay while the files
are over any part of the window (sidebar included), hides it when they leave or are dropped, adds a dropped file exactly once, and gives a file it cannot use no overlay.

What it does NOT prove: the mouse and the operating system's drag loop (hit-testing the window under the cursor, cursor feedback). Windows ignores synthetic mouse input for
that loop, so there is NO automated real-OS drag; a person does the 30-second acceptance test in docs/v4_1/MANUAL_DRAG_TEST.md on the packaged app.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.windows_only
def test_the_real_windows_drop_target_gets_the_overlay_and_adds_the_file_once(tmp_path):
    result = tmp_path / "native_drop.json"
    done = subprocess.run([sys.executable, str(ROOT / "tests" / "native_drop_child.py"), str(result)], cwd=str(ROOT), capture_output=True, text=True, timeout=180)
    assert result.exists(), (done.stdout[-2000:], done.stderr[-2000:])
    report = json.loads(result.read_text(encoding="utf-8"))
    drop = report.get("native_drop", {})
    assert not report.get("errors"), report.get("errors")
    assert drop.get("available") and "error" not in drop, drop.get("error") or drop
    assert drop["has_cf_hdrop"], drop["formats"]                                            # the data object is the Explorer one: files as CF_HDROP
    assert drop["overlay_visible_after_enter"] and drop["overlay_visible_over_sidebar"] and drop["overlay_visible_over_page"]
    assert drop["overlay_text"] == ["Drop lecture files", "Release to add to transcription"] and drop["overlay_geometry_ok"]
    assert drop["overlay_hidden_after_leave"] and drop["nothing_added_by_leaving"]
    assert drop["overlay_hidden_after_drop"] and drop["added_once"] and drop["on_the_start_screen"], drop
    assert drop["unsupported_has_no_overlay"] and drop["unsupported_toast"]
    assert drop["ok"]
