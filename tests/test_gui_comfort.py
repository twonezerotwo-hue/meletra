"""Comfort for long jobs: keep Windows awake, progress in the title bar, taskbar flash, disk-space check."""

from collections import namedtuple

import pytest

from src.gui import home_page as hp
from src.gui import power
from tests.conftest import write_silent_wav
from tests.gui_support import ScriptedLauncher, make_services, progress_event, settle, wait_for
from tests.test_gui_job import request_for, window_env  # noqa: F401  (fixture: a window with the real worker and a fake model)
from src.services.transcription_service import EXIT_PAUSED


def test_keep_awake_requests_and_releases_the_execution_state():
    calls = []
    assert power.keep_awake(True, setter=lambda f: calls.append(f) or 1) is True
    assert power.keep_awake(False, setter=lambda f: calls.append(f) or 1) is True
    assert calls == [power.ES_CONTINUOUS | power.ES_SYSTEM_REQUIRED, power.ES_CONTINUOUS]


def test_keep_awake_never_raises():
    assert power.keep_awake(True, setter=lambda f: (_ for _ in ()).throw(OSError("no"))) is False


def test_keep_awake_reports_what_windows_answered():
    assert power.keep_awake(True, setter=lambda f: 0) is False


def test_the_real_windows_call_is_accepted_and_released():
    assert power.keep_awake(True) is True
    assert power.keep_awake(False) is True


def test_the_window_keeps_windows_awake_only_while_a_job_runs(window_env, monkeypatch):  # noqa: F811
    env = window_env
    calls = []
    monkeypatch.setattr("src.gui.main_window.power.keep_awake", lambda on, setter=None: calls.append(on))
    launcher = ScriptedLauncher()
    env.window.controller.launcher = launcher
    env.window.controller.start(request_for(env.source))
    launcher.finish(EXIT_PAUSED)
    wait_for(lambda: calls == [True, False], what="keep-awake on/off")


def test_the_title_shows_progress_and_resets(window_env):  # noqa: F811
    w = window_env.window
    w.source_name = "lecture.wav"
    w._show_progress_in_title(progress_event(42.0))
    assert w.windowTitle().startswith("42%") and "lecture.wav" in w.windowTitle()
    w._running_changed(False)
    assert w.windowTitle() == "MELETRA"


def test_a_finished_or_failed_job_flashes_the_taskbar(window_env, monkeypatch):  # noqa: F811
    env = window_env
    alerts = []
    monkeypatch.setattr("src.gui.main_window.QApplication.alert", staticmethod(lambda widget, ms=0: alerts.append(widget)))
    out = env.tmp / "res"
    out.mkdir()
    env.window.on_completed({"type": "completed", "output_dir": str(out)})
    assert "Finished" in env.window.statusBar().currentMessage()
    env.window.on_failed("boom")
    assert len(alerts) == 2 and "Stopped" in env.window.statusBar().currentMessage()


def test_not_enough_disk_space_stops_the_start_with_a_clear_message(qapp, tmp_path, monkeypatch):
    services = make_services(tmp_path)
    page = hp.HomePage(services)
    source = write_silent_wav(tmp_path / "long.wav", seconds=30)
    page.set_file(str(source))
    wait_for(lambda: page.duration is not None, what="probe")
    got = []
    page.start_requested.connect(got.append)
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(hp.shutil, "disk_usage", lambda p: usage(10 ** 12, 10 ** 12 - 50 * 1024 ** 2, 50 * 1024 ** 2))
    page.start()
    assert not got and "Not enough free disk space" in page.problem_label.text() and "GB needed" in page.problem_label.text()
    monkeypatch.setattr(hp.shutil, "disk_usage", lambda p: usage(10 ** 12, 0, 10 ** 12))
    page.start()
    assert len(got) == 1
    settle(page)


def test_the_estimate_grows_with_the_recording(qapp, tmp_path, monkeypatch):
    services = make_services(tmp_path)
    page = hp.HomePage(services)
    source = write_silent_wav(tmp_path / "x.wav", seconds=5)
    page.set_file(str(source))
    wait_for(lambda: page.duration is not None, what="probe")
    usage = namedtuple("usage", "total used free")
    needs = []

    def fake(path):
        needs.append(path)
        return usage(10 ** 12, 0, 10 ** 12)

    monkeypatch.setattr(hp.shutil, "disk_usage", fake)
    page.duration = 2 * 3600.0                                  # a two-hour lecture: ~230 MB of temporary audio
    assert page.disk_problem() == "" and len(needs) == 2        # the temp folder and the output folder are both checked
    monkeypatch.setattr(hp.shutil, "disk_usage", lambda p: usage(10 ** 12, 0, 500 * 1024 ** 2))
    assert "temporary audio" in page.disk_problem()             # 500 MB free is not enough for 230 MB x 1.5 + 300 MB
    settle(page)
