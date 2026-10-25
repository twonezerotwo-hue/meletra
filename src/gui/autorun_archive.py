"""Diagnostic hook (``--autorun-archive RECORDING_NAME --result OUT [--study] [--timeout S]``): what a person sees when they open ONE lecture from the archive of the real window.

It looks the lecture up by the file name of its recording in the job records the program started with, and reports - without writing a word of the lecture into the report (lengths and
fingerprints only):

* the Files list: how many cards name this lecture (must be one), its status chip, and how many "Queued" ghosts are in the list;
* the click on the card: the folder the lecture opens from, whether Read Transcript is enabled, what the result screen says (words, duration, language);
* Technical details: the status, duration, language, words and processing time the screen shows;
* Read Transcript: that the reader shows the text INSIDE the window, and that no outside program (Explorer, a text editor) was started;
* with ``--study``: the Study material (made with the real engine when there is none) - counts, the engine that made it, evidence on every item;
* the privacy state: the Home chip and what the uploader would do.

Nothing is transcribed, translated or uploaded by this hook. It changes only what opening a lecture and, with ``--study``, making its study material change.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from . import diagnostics_probe as probe

log = logging.getLogger(__name__)


def _fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:12]


class AutoRunArchive:
    def __init__(self, window, recording: str, result_file: Path, timeout_s: float, study: bool = False, exercise: bool = False) -> None:
        self.window, self.recording, self.result_file, self.study, self.exercise = window, recording, result_file, study, exercise
        self.deadline = time.monotonic() + timeout_s
        self.finished = False
        self.report: dict[str, Any] = {"recording": recording}
        self.external: list[str] = []
        window.services.open_path = lambda path: self.external.append(str(path))          # an outside program would be started through this: it is recorded, not run
        self.timer = QTimer(window)
        self.timer.setInterval(250)
        self.timer.timeout.connect(self._tick)
        QTimer.singleShot(0, self._look)

    # ------------------------------------------------------------------------------------------------------------------ what a person does
    def _look(self) -> None:
        w = self.window
        services = w.services
        records = [r for r in services.jobs.list() if (r.source_name or "").casefold() == self.recording.casefold()]
        self.report["records"] = len(records)
        if len(records) != 1:
            self._finish(1, {"state": "failed", "error": f"{len(records)} job records name the recording {self.recording!r}"})
            return
        record = records[0]
        w.files_page.refresh()
        cards = [c for c in w.files_page.cards if c.record.job_id == record.job_id]
        self.report["files_list"] = {"cards_for_this_lecture": len(cards), "status": cards[0].status.text() if cards else "", "lectures_listed": len(w.files_page.cards),
                                     "queued_ghosts": sum(1 for c in w.files_page.cards if c.record.state == "queued")}
        w.open_lecture_record(services.jobs.get(record.job_id))
        result = w.simple_result
        folder = w.current_output
        self.report["opened"] = {"folder_name": folder.name if folder else "", "read_enabled": result.read_button.isEnabled(), "study_enabled": result.study_button.isEnabled(),
                                 "translate_enabled": result.translate_button.isEnabled(), "info": result.info_label.text()}
        self.report["technical_details"] = {name: label.text() for name, label in w.results.values.items() if name in ("Status", "Duration", "Language", "Words", "Processing time")}
        if folder is not None and result.read_button.isEnabled():
            result.read_button.click()
            text = w.transcript_view.text.toPlainText()
            self.report["reader"] = {"shown_inside_the_window": bool(text.strip()), "characters": len(text), "fingerprint": _fingerprint(text), "view": w.transcript_view.view_combo.currentText()}
        self.report["external_programs_started"] = list(self.external)
        self.report["privacy"] = {"home_chip": w.simple_home.privacy.text(), "sharing_active": services.sharing.active(), "needs_choice": services.sharing.needs_choice()}
        if self.study and folder is not None:
            self.report["study"] = {"started": time.monotonic()}
            w.open_study("overview")
            self.timer.start()
        else:
            self._finish(0, {"state": "completed"})

    def _tick(self) -> None:
        w = self.window
        if probe.study_ready(w):
            pack = w.study.state.pack
            self.report["study"] = {
                "seconds": round(time.monotonic() - self.report["study"]["started"], 1), "flashcards": len(pack.flashcards), "questions": len(pack.questions),
                "summary_points": sum(len(v) for v in pack.summaries.values()), "engine": pack.generation, "label": w.study.provider_label.text(),
                "items_with_evidence": sum(1 for i in [*pack.flashcards, *pack.questions] if i.evidence), "items_total": len(pack.flashcards) + len(pack.questions),
                "blanks": sum(1 for i in pack.flashcards if "_____" in i.front) + sum(1 for q in pack.questions if "_____" in q.question),
                "multiple_choice": sum(1 for q in pack.questions if q.options and q.type != "true_false"), "difficulty": sorted({i.difficulty for i in [*pack.flashcards, *pack.questions]}),
                "premium": {k: v for k, v in (pack.stats.get("premium") or {}).items() if k in ("mode", "model", "flashcards", "questions", "notes")}}
            self.report["study"]["learning"] = self._learning()
            self._finish(0, {"state": "completed"})
        elif w.study.stack.currentIndex() == probe.message_page():
            self._finish(1, {"state": "failed", "error": w.study.message_text.text()})
        elif time.monotonic() >= self.deadline:
            self._finish(124, {"state": "timeout", "error": "no study material before the time limit"})

    def _learning(self) -> dict[str, Any]:
        """The learning side of the Study material: what the flashcards and Study Mode offer from the saved progress and - with ``--exercise`` - what a student does first (two flashcard
        reviews, one card marked difficult, one question answered in a Study Mode session that is left unfinished). A second run WITHOUT ``--exercise`` shows what the program kept."""
        w = self.window
        out: dict[str, Any] = {}
        try:
            flash, mode = w.study.flashcards, w.study.study_mode
            if self.exercise:
                flash.start()
                first = flash.session.current().id
                flash.reveal()
                flash.rate("know")
                second = flash.session.current().id
                flash.reveal()
                flash.rate("dont_know")
                third = flash.session.current().id
                flash.toggle_difficult()
                mode.count_combo.setCurrentIndex(mode.count_combo.findData(10))
                mode.start()
                question = mode.session.current()
                mode.choose(0) if question.options else mode.skip()
                out["exercised"] = {"cards": [first, second, third], "question": question.id}
            progress, pack = w.study.state.progress, w.study.state.pack
            from ..study.session import QuizSession

            out["plan"] = progress.study_plan(pack)
            out["resumable"] = QuizSession.resumable(pack, progress)
            out["resume_button_visible"] = mode.resume_button.isVisibleTo(mode)
            out["deck_opens_on"] = flash.focus_combo.currentData()
            out["deck_size"] = len(flash.session.deck) if flash.session is not None else 0
            out["card_states"] = {cid: progress.card_state(cid) for cid in out.get("exercised", {}).get("cards", [])}
        except Exception as exc:                                             # noqa: BLE001 - a diagnostic reports its own failure, it never hides the rest of the report
            log.exception("learning check failed")
            out["error"] = f"{type(exc).__name__}: {exc}"
        return out

    def _finish(self, code: int, payload: dict) -> None:
        if self.finished:
            return
        self.finished = True
        self.timer.stop()
        self.report["external_programs_started"] = list(self.external)
        try:
            self.result_file.parent.mkdir(parents=True, exist_ok=True)
            self.result_file.write_text(json.dumps({"exit_code": code, **payload, **self.report}, indent=2, default=str), encoding="utf-8")
        except OSError:
            log.exception("Could not write the autorun result")
        QTimer.singleShot(300, lambda: QApplication.exit(code))
