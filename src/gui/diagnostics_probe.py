"""Opt-in evidence for diagnostic runs (``--autorun`` / ``--autorun-queue`` with the environment variable ``MLT_UI_TRACE=1``).

The packaged-app tests on a build machine cannot look at the screen, so the window reports what it SHOWS: every progress text the person would
read (once per change, with the time), the navigation, what the first-launch model screen says and whether it offers a download or a folder
picker, and - after a finished transcription - the Study workspace (tabs, header, stat cards, buttons of every tab, a revealed flashcard).
Nothing here changes behaviour; without the variable none of it runs.
"""

from __future__ import annotations

import os
import time
from typing import Any

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QLabel, QPushButton

ENABLED_BY = "MLT_UI_TRACE"


def enabled() -> bool:
    return os.environ.get(ENABLED_BY) == "1"


def visible_texts(root, limit: int = 60) -> dict[str, list[str]]:
    buttons = sorted({b.text() for b in root.findChildren(QPushButton) if b.isVisibleTo(root) and b.text().strip()})
    labels = []
    for label in root.findChildren(QLabel):
        text = label.text().strip().replace("\n", " / ")
        if label.isVisibleTo(root) and text and text not in labels and len(text) < 240:
            labels.append(text)
    return {"buttons": buttons, "labels": labels[:limit]}


class UiProbe:
    """What the running screens show: ``[seconds, area, text]`` once per change. Every text change of the watched labels is recorded the moment it is set (a stage that lasts
    a few milliseconds - the audio of a short clip is extracted in less than the sampling interval - is still seen); a timer samples every 100 ms as well, for texts set elsewhere."""

    def __init__(self, window, interval_ms: int = 100) -> None:
        self.window = window
        self.started = time.monotonic()
        self.rows: list[list[Any]] = []
        self._last: dict[str, str] = {}
        for area, widget in self._watched().items():
            self._hook(area, widget)
        self.timer = QTimer(window)
        self.timer.setInterval(interval_ms)
        self.timer.timeout.connect(self.sample)
        self.timer.start()

    def _watched(self) -> dict:
        w = self.window
        return {"active.title": w.simple_active.title, "active.stage": w.simple_active.stage_label, "active.percent": w.simple_active.percent_label,
                "batch.title": w.simple_batch.title, "batch.stage": w.simple_batch.stage, "batch.percent": w.simple_batch.percent_label,
                "result.title": w.simple_result.title}

    def _hook(self, area: str, widget) -> None:
        original = widget.setText

        def set_text(text, _original=original, _area=area, _widget=widget):
            _original(text)
            self._note(_area, _widget)

        widget.setText = set_text

    def _note(self, area: str, widget) -> None:
        text = widget.text().strip().replace("\n", " / ")
        if widget.isVisible() and text and self._last.get(area) != text:
            self._last[area] = text
            self.rows.append([round(time.monotonic() - self.started, 2), area, text])

    def sample(self) -> None:
        for area, widget in self._watched().items():
            self._note(area, widget)

    def stop(self) -> list[list[Any]]:
        self.timer.stop()
        self.sample()
        return self.rows


def home_facts(window) -> dict[str, Any]:
    shell = window.simple_shell
    return {"navigation": [b.text() for b in shell.buttons.values()], "privacy": window.simple_home.privacy.text(),
            "home": visible_texts(window.simple_home, 30)}


def first_launch_facts(window) -> dict[str, Any]:
    """The model screen a student sees on first launch, built (not shown): what it says and which buttons it offers."""
    from .first_run import FirstRunDialog

    dialog = FirstRunDialog(window.services, window)
    dialog.refresh_model()
    thread = getattr(dialog, "check_thread", None)
    if thread is not None:
        thread.wait(30000)
    facts = {"model_text": dialog.model_label.text().replace("\n", " / "), "download_button_hidden": dialog.download_button.isHidden(),
             "import_folder_button_hidden": dialog.import_button.isHidden(), "continue_button_shown": not dialog.done_button.isHidden(),
             "check": dialog.check_label.text()[:900]}
    dialog.deleteLater()
    return facts


def message_page() -> int:
    from .study_view import PAGE_MESSAGE

    return PAGE_MESSAGE


def study_ready(window) -> bool:
    from .study_view import PAGE_TABS

    return window.study.stack.currentIndex() == PAGE_TABS


NEW_ENGINE = "NEW QUALITY ENGINE (validated study facts)"
OLD_ENGINE = "OLD ENGINE (sentence based)"


def pack_engine(pack) -> dict[str, Any]:
    """Which engine made the study pack: the validated-facts engine leaves its overview, its reject report, its quality gate and its audit in the pack, and tags every flashcard / question
    with the fact it was made from. The sentence-based method (non-English lectures) leaves none of them."""
    quality = pack.stats.get("quality") or {}
    facts = (pack.overview or {}).get("facts")
    new = bool(quality and facts and pack.stats.get("quality_gate"))
    summaries = {kind: len(items) for kind, items in pack.summaries.items()}

    def tagged(items) -> dict[str, int]:
        return {"items": len(items), "from_validated_facts": sum(1 for i in items if getattr(i, "fact_id", ""))}

    scores = quality.get("scores", {})
    return {"engine": NEW_ENGINE if new else OLD_ENGINE, "facts": facts, "gate": pack.stats.get("quality_gate"), "scores": scores,
            "summary": {"items": sum(n for k, n in summaries.items() if k in ("quick", "detailed", "high_yield")), "kinds": summaries, "engine": NEW_ENGINE if new else OLD_ENGINE},
            "key_concepts": {"items": summaries.get("key_concepts", 0), "engine": NEW_ENGINE if new else OLD_ENGINE},
            "flashcards": {**tagged(pack.flashcards), "engine": NEW_ENGINE if new else OLD_ENGINE},
            "questions": {**tagged(pack.questions), "engine": NEW_ENGINE if new else OLD_ENGINE},
            "study_mode": {"questions_available": len(pack.questions), "built_from": NEW_ENGINE if new else OLD_ENGINE, "simulated_session": quality.get("study_mode")}}


def study_facts(window) -> dict[str, Any]:
    from .study_view import TAB_NAMES

    study = window.study
    out: dict[str, Any] = {"tabs": [study.tabs.tabText(i) for i in range(study.tabs.count())], "title": study.title.text(),
                           "meta": study.lecture_meta.text(), "provider": study.provider_label.text(),
                           "stats": {key: label.text() for key, label in study.stat_values.items()},
                           "actions": sorted({b.text() for b in study.header.findChildren(QPushButton) if b.text().strip()}), "tab_content": {}}
    pack = getattr(getattr(study, "state", None), "pack", None)
    if pack is not None:
        out["engine"] = pack_engine(pack)
    for index, name in enumerate(TAB_NAMES):
        study.tabs.setCurrentIndex(index)
        page = study.tabs.widget(index)
        if name == "Flashcards":
            for button in page.findChildren(QPushButton):
                if button.text() == "Show answer" and button.isVisibleTo(page):
                    button.click()
                    for _ in range(5):
                        QApplication.processEvents()          # the source chips of the card are shown by queued layout events
                    break
        out["tab_content"][name] = visible_texts(page, 40)
    return out


def payload(probe: UiProbe | None, window, study: bool = False) -> dict[str, Any]:
    if probe is None:
        return {}
    data: dict[str, Any] = {"ui_trace": probe.stop()}
    for key, fn in (("ui_home", home_facts), ("ui_first_launch", first_launch_facts)) + ((("ui_study", study_facts),) if study else ()):
        try:
            data[key] = fn(window)
        except Exception as exc:  # noqa: BLE001 - the probe must never break the run it observes
            data[key] = {"error": f"{type(exc).__name__}: {exc}"}
    return data
