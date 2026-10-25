"""Diagnostic run of the real window (``--ui-check OUT_DIR``), for the packaged build and for the tests.

It looks at the window the way assistive technology and a person do and writes ``ui_check.json`` (and pictures) into OUT_DIR:

* ``accessibility``: the Qt accessibility tree of the start screen, Files, Settings and About (role, name, description, value, focusable) and a list of checks on it;
* ``tab_order``: where the Tab key goes on the start screen and on Files;
* ``native_drop``: a drag session against the window's REAL Windows drop target with a REAL Explorer data object (see ``native_drop``): does the overlay appear,
  does the drop add the file exactly once, does a file that cannot be used get no overlay;
* ``pages`` (when the window is asked to show them): each page is prepared, then ``<name>.ready`` is written (its text is the window handle to photograph) and the run
  waits until ``<name>.captured`` exists (an outside script takes the picture with the operating system) or 20 s have passed.

Nothing is transcribed and nothing is sent anywhere. The run changes only what a person would change by clicking (the page, a selected file) in the data folder it is given.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
import traceback
import wave
from pathlib import Path
from typing import Any, Iterator, Optional

from PySide6.QtCore import QObject, Qt, QTimer
from PySide6.QtWidgets import QApplication

from . import accessibility

CAPTURE_WAIT_S = 20.0


def _pump(times: int = 6) -> None:
    for _ in range(times):
        QApplication.processEvents()


def _silent_wav(path: Path, seconds: float = 1.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * int(16000 * seconds))
    return path


def _label_of(widget) -> str:
    for getter in ("accessibleName", "text", "windowTitle", "placeholderText"):
        value = getattr(widget, getter, None)
        if callable(value):
            try:
                text = value()
            except RuntimeError:
                continue
            if text:
                return str(text).replace("\n", " ")
    return ""


# ---------------------------------------------------------------------------------------------------------------------------------- accessibility
def tab_chain(window, count: int = 60, start=None) -> list[dict]:
    """The controls reached by pressing Tab again and again from where the focus rests (the anchor, or ``start``): class, object name, name.

    ``window.focusWidget()`` is the focus of that window even when the platform has not activated it (the offscreen platform of the tests)."""
    first = start if start is not None else window.focus_anchor
    first.setFocus(Qt.TabFocusReason)
    _pump(2)
    seen: list[Any] = []
    chain: list[dict] = []
    for _ in range(count):
        window.focusNextPrevChild(True)
        _pump(2)
        widget = window.focusWidget()
        if widget is None or widget in seen or widget is first:
            break
        seen.append(widget)
        chain.append({"class": type(widget).__name__, "object": widget.objectName(), "name": _label_of(widget), "widget": widget})
    return chain


def accessibility_report(window) -> dict:
    """The tree of each page and the checks on it (see ``CHECKS`` in the tests: names, roles, values, focus)."""
    from .about_dialog import AboutDialog

    report: dict[str, Any] = {}
    window.show_home()
    _pump()
    home = accessibility.tree(window)
    report["home"] = home
    window._goto("files")
    window.files_page.refresh()
    _pump()
    report["files"] = accessibility.tree(window)
    dialog = window.make_settings_dialog("General")
    dialog.show()
    _pump()
    report["settings"] = accessibility.tree(dialog)
    dialog.close()
    about = AboutDialog(window)
    about.show()
    _pump()
    report["about"] = accessibility.tree(about)
    about.close()
    window.show_home()
    _pump()
    checks: dict[str, Any] = {}
    browse = accessibility.find(home, "Browse lecture files")
    checks["browse_is_a_named_button"] = bool(browse and browse["role"] == "Button" and browse["focusable"])
    pdf = accessibility.find(home, "Add lecture PDF")
    checks["pdf_is_a_named_button"] = bool(pdf and pdf["role"] == "Button")
    combo_in = accessibility.find(home, "Input language")
    combo_out = accessibility.find(home, "Output language")
    checks["input_language_has_name_and_value"] = bool(combo_in and combo_in["role"] == "ComboBox" and combo_in["value"])
    checks["output_language_has_name_and_value"] = bool(combo_out and combo_out["role"] == "ComboBox" and combo_out["value"])
    transcribe = accessibility.find(home, "TRANSCRIBE")
    checks["transcribe_is_a_button"] = bool(transcribe and transcribe["role"] == "Button")
    lectures = [e for e in home + report["files"] if e["class"] == "LectureCard"]
    checks["lecture_cards_are_named_buttons"] = bool(lectures) and all(e["role"] == "Button" and e["focusable"] and ", " in e["name"] for e in lectures)
    checks["lecture_card_examples"] = [e["name"] for e in lectures[:3]]
    nav = [e for e in home if e["class"] == "QPushButton" and e["name"] in ("Home", "Transcription", "Study", "Files", "Settings")]
    checks["sidebar_has_five_named_buttons"] = len({e["name"] for e in nav}) == 5
    for page, entries in report.items():
        checks[f"{page}_every_focusable_control_has_a_name"] = accessibility.unnamed_focusable(entries) == []
    report["checks"] = checks
    return report


# ---------------------------------------------------------------------------------------------------------------------------------- the native drop
def native_drop_report(window, work: Path, hold_overlay=None) -> Iterator[int]:
    """A generator (it yields a pause in ms between the steps); the report is its return value (``StopIteration.value``).

    ``hold_overlay`` (optional): a generator function called while the overlay is on screen (the outside script photographs the window then)."""
    from . import native_drop as nd

    report: dict[str, Any] = {"available": False}
    if sys.platform != "win32":
        report["reason"] = "Windows only"
        return report
    try:
        good = _silent_wav(work / "Neuroanatomy lecture.wav")
        bad = work / "notes.xyz"
        bad.write_text("not a lecture", encoding="utf-8")
        window.show_home()
        window.simple_home._new_transcription()
        _pump()
        page = window.simple_stack.currentWidget()
        x, y = nd.screen_point_in(window, 90, 420)                                    # over the sidebar, not over the drop area
        x2, y2 = nd.screen_point_in(window, window.width() // 2 + 60, window.height() // 2)
        # 1. a supported file: enter, move over the sidebar, then over the page
        drag = nd.NativeDrag(window, [good])
        report["available"] = True
        report["formats"] = drag.data.formats()
        report["has_cf_hdrop"] = 15 in report["formats"]
        report["enter_effect"] = drag.enter(x, y)
        _pump()
        report["overlay_visible_after_enter"] = window.drop_overlay.isVisible()
        report["overlay_text"] = [window.drop_overlay.TEXT, window.drop_overlay.SUB]
        drag.over(x + 8, y + 4)
        _pump()
        report["overlay_visible_over_sidebar"] = window.drop_overlay.isVisible()
        drag.over(x2, y2)
        _pump()
        report["overlay_visible_over_page"] = window.drop_overlay.isVisible()
        report["drop_area_lit"] = bool(page is window.simple_home and window.simple_home.drop_zone.drag_active())
        report["overlay_geometry_ok"] = window.drop_overlay.width() >= window.width() - 2 and window.drop_overlay.height() >= window.height() - 2
        if hold_overlay is not None:
            yield from hold_overlay()
        # 2. leaving without dropping hides the overlay and adds nothing
        drag.leave()
        _pump()
        report["overlay_hidden_after_leave"] = not window.drop_overlay.isVisible()
        report["nothing_added_by_leaving"] = not window.simple_home.selection and getattr(window.home, "source", None) is None
        drag.close()
        # 3. enter again and drop: the file arrives once
        drag = nd.NativeDrag(window, [good])
        drag.enter(x, y)
        _pump()
        drag.over(x2, y2)
        report["drop_effect"] = drag.drop(x2, y2)
        _pump(10)
        yield 400
        _pump(10)
        report["overlay_hidden_after_drop"] = not window.drop_overlay.isVisible()
        home = window.simple_home
        source = getattr(window.home, "source", None)
        names = [Path(d.path).name for d in home.selection] if home.selection else ([Path(source).name] if source is not None else [])
        report["files_after_drop"] = names
        report["added_once"] = len(names) == 1 and names[0].lower().startswith("neuroanatomy lecture")
        report["on_the_start_screen"] = window.simple_stack.currentWidget() is home
        drag.close()
        # 4. a file that cannot be used: no overlay, a small explanation
        drag = nd.NativeDrag(window, [bad])
        drag.enter(x, y)
        _pump()
        report["unsupported_has_no_overlay"] = not window.drop_overlay.isVisible()
        report["unsupported_toast"] = window.toast.isVisible() and "not supported" in window.toast.text()
        drag.leave()
        drag.close()
        _pump()
        window.toast.hide()
        window.simple_home._new_transcription()
    except Exception as exc:  # noqa: BLE001 - the report says what failed
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
    keys = ("has_cf_hdrop", "overlay_visible_after_enter", "overlay_visible_over_sidebar", "overlay_visible_over_page", "overlay_hidden_after_leave", "nothing_added_by_leaving", "overlay_hidden_after_drop",
            "added_once", "on_the_start_screen", "unsupported_has_no_overlay", "unsupported_toast")
    report["ok"] = bool(report.get("available")) and "error" not in report and all(report.get(k) for k in keys)
    return report


# ---------------------------------------------------------------------------------------------------------------------------------- the run
class UiCheck(QObject):
    """Runs the steps one after the other from the event loop (so the window keeps painting and the outside script can photograph it)."""

    def __init__(self, window, out_dir: Path, lecture: Optional[Path] = None, translated: Optional[Path] = None, size: str = "", pages: str = "",
                 exit_when_done: bool = True) -> None:
        super().__init__(window)
        self.window, self.out_dir = window, Path(out_dir)
        self.lecture, self.translated = lecture, translated
        self.size, self.pages = size, [p for p in pages.split(",") if p]
        self.exit_when_done = exit_when_done
        self.report: dict[str, Any] = {"size": size, "errors": []}
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.steps = self._run()
        QTimer.singleShot(1200, self._advance)

    def _advance(self) -> None:
        try:
            delay = next(self.steps)
        except StopIteration:
            self._finish()
            return
        except Exception as exc:  # noqa: BLE001
            self.report["errors"].append(f"{type(exc).__name__}: {exc}")
            self.report["traceback"] = traceback.format_exc()
            self._finish()
            return
        QTimer.singleShot(max(int(delay), 1), self._advance)

    def _finish(self) -> None:
        self.report["done"] = True
        (self.out_dir / "ui_check.json").write_text(json.dumps(self.report, indent=1, default=lambda value: f"<{type(value).__name__}>"), encoding="utf-8")
        if self.exit_when_done:
            QTimer.singleShot(300, lambda: QApplication.exit(0 if not self.report["errors"] else 1))

    def shot(self, name: str, target=None) -> Iterator[int]:
        """Announce a page and wait until the outside script has photographed ``target`` (default: the main window)."""
        widget = target if target is not None else self.window
        _pump(8)
        self.window.toast.hide()
        (self.out_dir / f"{name}.ready").write_text(str(int(widget.winId())), encoding="utf-8")
        deadline = time.monotonic() + CAPTURE_WAIT_S
        while not (self.out_dir / f"{name}.captured").exists() and time.monotonic() < deadline:
            yield 100
        _pump(2)

    def _run(self) -> Iterator[int]:
        window = self.window
        if self.size:
            width, height = (int(v) for v in self.size.split("x"))
            window.resize(width, height)
        window.show()
        window.raise_()
        window.activateWindow()
        yield 800
        steps = self.pages or ["accessibility", "native_drop", "home", "progress", "queue", "result", "transcript", "study", "files", "settings", "dropdown", "about"]
        for name in steps:
            try:
                handler = getattr(self, f"_step_{name}")
                result = handler()
                if result is not None:
                    yield from result
            except Exception as exc:  # noqa: BLE001
                self.report["errors"].append(f"{name}: {type(exc).__name__}: {exc}")
                self.report.setdefault("tracebacks", {})[name] = traceback.format_exc()
            yield 100

    # ---- steps
    def _step_accessibility(self) -> None:
        def plain(chain):                                            # the report holds text only (the widgets are gone by the time it is written)
            return [{key: value for key, value in item.items() if key != "widget"} for item in chain]

        self.report["accessibility"] = accessibility_report(self.window)
        self.window.show_home()
        _pump()
        self.report["tab_order"] = {"home": plain(tab_chain(self.window))}
        self.window._goto("files")
        self.window.files_page.refresh()
        _pump()
        self.report["tab_order"]["files"] = plain(tab_chain(self.window))
        self.window.show_home()

    def _step_native_drop(self) -> Iterator[int]:
        work = Path(tempfile.mkdtemp(prefix="wm_native_drop_"))
        generator = native_drop_report(self.window, work, hold_overlay=lambda: self.shot("drag_overlay"))
        try:
            while True:
                yield next(generator)
        except StopIteration as stop:
            self.report["native_drop"] = stop.value

    def _step_home(self) -> Iterator[int]:
        self.window.show_home()
        yield from self.shot("home")

    def _step_progress(self) -> Iterator[int]:
        window = self.window
        window._goto("active")
        window.simple_active.begin("FIXTURE_0004.MP4", "auto", "auto", "video")
        window.simple_active.set_stage({"stage": "finalizing", "percent": 68.0})
        window.simple_active.set_progress({"percent": 68.0, "rtf": 6.0, "eta": 240})
        yield from self.shot("progress")

    def _step_queue(self) -> Iterator[int]:
        from ..ingest import queue as q
        from ..ingest.router import classify

        window = self.window
        files = [str(_silent_wav(self.out_dir / "_queue" / f"{name}.wav", 2)) for name in ("Neuroanatomy", "Histology", "Physiology", "Pharmacology")]
        queue = q.BatchQueue(classify(files).items, start=lambda item: None, pause=lambda: None, cancel=lambda: None)
        queue.items[0].state, queue.items[1].state = q.COMPLETED, q.PROCESSING
        queue.state = q.RUNNING
        window._goto("batch")
        window.simple_batch.begin(queue)
        window.simple_batch.progress_bar.setValue(420)
        window.simple_batch.percent_label.setText("42%")
        window.simple_batch.stage.setText("Transcribing...")
        yield from self.shot("queue")

    def _step_result(self) -> Iterator[int]:
        if self.lecture is None:
            return
        self.window.show_finished(self.lecture, "FIXTURE_0004.MP4")
        yield from self.shot("result")

    def _step_transcript(self) -> Iterator[int]:
        folder = self.translated or self.lecture
        if folder is None:
            return
        self.window.transcript_view.show_transcript(folder, "Pharmacology: autonomic drugs", prefer_translated=self.translated is not None)
        self.window._goto("transcript")
        yield from self.shot("transcript_translated")

    def _step_study(self) -> Iterator[int]:
        if self.lecture is None:
            return
        from . import diagnostics_probe as probe

        window = self.window
        window.show_finished(self.lecture, "Neuroanatomy.mp4")
        window.open_study("overview")
        for _ in range(400):
            if probe.study_ready(window):
                break
            yield 50
        for tab in ("summary", "flashcards", "questions"):
            window.study.select_tab(tab)
            _pump(10)
            yield from self.shot(f"study_{tab}")

    def _step_files(self) -> Iterator[int]:
        self.window._goto("files")
        self.window.files_page.refresh()
        yield from self.shot("files")

    def _step_settings(self) -> Iterator[int]:
        dialog = self.window.make_settings_dialog("General")
        dialog.show()
        _pump(10)
        yield 300
        yield from self.shot("settings", dialog)
        dialog.close()

    def _step_dropdown(self) -> Iterator[int]:
        self.window.show_home()
        combo = self.window.simple_home.input_combo
        combo.showPopup()
        _pump(10)
        yield 300
        popup = combo.view().window()
        yield from self.shot("dropdown", popup)
        combo.hidePopup()

    def _step_about(self) -> Iterator[int]:
        from .about_dialog import AboutDialog

        dialog = AboutDialog(self.window)
        dialog.show()
        _pump(10)
        yield 300
        yield from self.shot("about", dialog)
        dialog.close()
