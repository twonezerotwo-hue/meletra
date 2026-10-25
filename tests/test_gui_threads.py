"""Closing a window or dialog while a background task runs must never destroy a running QThread (that aborts the program)."""

import time

import pytest

from src.gui.first_run import FirstRunDialog
from src.gui.main_window import MainWindow
from src.gui.settings_dialog import SettingsDialog
from src.gui.tasks import TaskThread, stop_threads
from tests.gui_support import MODEL_KEY, install_fake_model, make_services, settle, wait_for


def slow_download(started):
    def download(key, progress=None, cancel=None):
        started.append(1)
        for _ in range(500):                                  # up to ~10 s, but it honours the cancel flag like the real download
            if cancel and cancel():
                raise RuntimeError("Download cancelled - it will resume from here next time.")
            time.sleep(0.02)
        raise AssertionError("the download was never cancelled")
    return download


def test_stop_threads_cancels_and_waits(qapp):
    from PySide6.QtWidgets import QWidget

    owner = QWidget()
    flag = []

    def work(progress, cancelled):
        while not cancelled():
            time.sleep(0.01)
        flag.append("stopped")

    thread = TaskThread(work, owner)
    thread.start()
    wait_for(lambda: thread.isRunning(), what="the thread to start")
    assert stop_threads(owner) == 0 and not thread.isRunning() and flag == ["stopped"]


def test_stop_threads_on_a_widget_without_threads_is_a_no_op(qapp):
    from PySide6.QtWidgets import QWidget

    assert stop_threads(QWidget()) == 0


def test_closing_the_settings_dialog_cancels_a_running_download(qapp, tmp_path, monkeypatch):
    services = make_services(tmp_path, installed=False)
    started = []
    monkeypatch.setattr(services.models, "download", slow_download(started))
    dialog = SettingsDialog(services)
    dialog.download_button.click()
    wait_for(lambda: started, what="the download to start")
    t0 = time.monotonic()
    dialog.reject()
    assert time.monotonic() - t0 < 5 and not any(t.isRunning() for t in dialog.findChildren(TaskThread))


def test_saving_the_settings_dialog_also_waits_for_its_tasks(qapp, tmp_path, monkeypatch):
    services = make_services(tmp_path, installed=False)
    started = []
    monkeypatch.setattr(services.models, "download", slow_download(started))
    dialog = SettingsDialog(services)
    dialog.download_button.click()
    wait_for(lambda: started, what="the download to start")
    dialog.save()
    assert not any(t.isRunning() for t in dialog.findChildren(TaskThread)) and dialog.result() == 1


def test_closing_the_first_run_wizard_cancels_a_running_download(qapp, tmp_path, monkeypatch):
    services = make_services(tmp_path, installed=False)
    started = []
    monkeypatch.setattr(services.models, "download", slow_download(started))
    wizard = FirstRunDialog(services)
    wait_for(lambda: wizard.rows, timeout=30, what="the system check")
    wizard.download_button.click()
    wait_for(lambda: started, what="the download to start")
    wizard.reject()
    assert not any(t.isRunning() for t in wizard.findChildren(TaskThread)) and services.settings.settings.first_run_done


def test_closing_the_main_window_cancels_a_running_download(qapp, tmp_path, monkeypatch):
    services = make_services(tmp_path, installed=False)
    started = []
    monkeypatch.setattr(services.models, "download", slow_download(started))
    window = MainWindow(services, None)
    window.home.start_model_download()
    wait_for(lambda: started, what="the download to start")
    t0 = time.monotonic()
    window.close()
    assert time.monotonic() - t0 < 5 and not any(t.isRunning() for t in window.findChildren(TaskThread))
    settle(window, window.home)
