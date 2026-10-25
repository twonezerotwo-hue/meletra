"""FEATURE-07: transcript sharing needs an explicit, separate, affirmative yes. The first launch asks once (Share / Keep private), the stored choice, the settings checkbox, the
header chip and the uploader always say the same thing, withdrawing stops uploads and empties the queue, and while it is OFF nothing touches the network."""

from __future__ import annotations

import json
import urllib.request

import pytest

from src.gui.main_window import MainWindow
from src.gui.sharing_notice import CHOICE_PRIVATE, CHOICE_SHARE, PrivacyChoiceDialog
from src.services.settings_service import SettingsService, migrate, sanitize
from src.sharing import payload
from src.sharing.service import SharingService
from tests.gui_support import ScriptedLauncher, make_services, settle
from tests.study_support import write_lecture_folder

ENDPOINT = "https://inbox.example.org/v1/transcripts"


def lecture(tmp_path, raw="Glycolysis converts glucose to pyruvate in the cytoplasm."):
    folder = write_lecture_folder(tmp_path / "lecture")
    (folder / "local_raw_transcript.txt").write_text(raw, encoding="utf-8")
    return folder


def configured(services, calls):
    """The services with an upload address and a recorder instead of the network."""
    services.sharing = SharingService(services.settings, services.paths.root / "sharing", post=lambda url, body: calls.append((url, json.loads(body))) or 202, endpoint=lambda: ENDPOINT)
    services.sharing.reconcile()
    return services


# --------------------------------------------------------------------------------------------------------------------- defaults and migration
def test_a_new_installation_shares_nothing_and_has_not_been_asked_yet(tmp_path):
    settings = SettingsService(tmp_path / "settings.json")
    sharing = SharingService(settings, tmp_path / "sharing", endpoint=lambda: ENDPOINT)
    assert settings.settings.share_transcripts is False and settings.settings.share_notice_shown is False and sharing.needs_choice() and not sharing.active()


@pytest.mark.parametrize("old,asked", [
    ({"schema_version": 5, "share_transcripts": True, "share_notice_shown": True}, True),          # a 4.1.1 "yes" was explicit: not asked again, still on
    ({"schema_version": 5, "share_transcripts": False, "share_notice_shown": False}, False),       # private: asked once
    ({"schema_version": 5, "share_transcripts": True, "share_notice_shown": False}, False),        # a file edit is not a consent
    ({"schema_version": 4, "share_transcripts": True, "share_notice_shown": True}, False),         # earlier values came from a default
])
def test_what_an_older_settings_file_said_about_sharing(old, asked):
    s = sanitize(migrate(old))
    assert s.share_choice_made is asked
    assert s.share_transcripts is (asked and bool(old["share_transcripts"])) and s.share_notice_shown is (asked and bool(old["share_notice_shown"]))


# --------------------------------------------------------------------------------------------------------------------- the first-launch question
def test_the_question_has_two_equal_answers_and_private_is_the_safe_one(qapp, tmp_path):
    services = make_services(tmp_path, opened=[])
    dialog = PrivacyChoiceDialog(services)
    assert [dialog.private_button.text(), dialog.share_button.text()] == [CHOICE_PRIVATE, CHOICE_SHARE]
    assert dialog.private_button.isDefault() and not dialog.share_button.isDefault()                   # Enter means private
    assert dialog.private_button.minimumHeight() == dialog.share_button.minimumHeight()                 # neither answer is pushed
    assert "audio and video are never uploaded" in dialog.details.text() and "Show what would be shared" in dialog.details.text()
    assert services.sharing.needs_choice()                                                              # looking at the question answers nothing
    dialog.deleteLater()


def test_keeping_transcripts_private_or_closing_the_question_leaves_everything_off(qapp, tmp_path):
    for how in ("button", "escape"):
        services = make_services(tmp_path / how, opened=[])
        dialog = PrivacyChoiceDialog(services)
        dialog.private_button.click() if how == "button" else dialog.reject()
        s = services.settings.settings
        assert (s.share_transcripts, s.share_notice_shown, s.share_choice_made) == (False, False, True) and not services.sharing.active() and not services.sharing.needs_choice()
        dialog.deleteLater()


def test_only_the_explicit_share_button_turns_sharing_on(qapp, tmp_path):
    services = make_services(tmp_path, opened=[])
    dialog = PrivacyChoiceDialog(services)
    dialog.share_button.click()
    s = services.settings.settings
    assert (s.share_transcripts, s.share_notice_shown, s.share_choice_made) == (True, True, True) and services.sharing.active()
    dialog.deleteLater()


# --------------------------------------------------------------------------------------------------------------------- one truth
@pytest.mark.parametrize("wanted,agreed", [(False, False), (True, False), (False, True), (True, True)])
def test_the_checkbox_the_label_the_header_chip_and_the_uploader_always_agree(qapp, tmp_path, wanted, agreed):
    services = make_services(tmp_path, opened=[])
    services.settings.update(share_transcripts=wanted, share_notice_shown=agreed, share_choice_made=True)
    calls: list = []
    configured(services, calls)                                      # (reconcile ran: a wish without the agreement is not kept)
    window = MainWindow(services, ScriptedLauncher())
    dialog = window.make_settings_dialog("Privacy & Data")
    really_on = wanted and agreed
    assert services.sharing.active() is really_on
    assert dialog.share_check.isChecked() is really_on and dialog.share_state.text() == ("ON" if really_on else "OFF")
    assert window.simple_home.privacy.text() == ("Sharing enabled" if really_on else "Local processing")
    folder = lecture(tmp_path)
    queued = services.sharing.transcript_saved(folder, "Lecture")
    sent = services.sharing.uploader.run_once()
    assert queued is really_on and (sent == 1) is really_on and (len(calls) == 1) is really_on             # what the backend does is what the screen says
    dialog.deleteLater()
    settle(window, window.home, window.study)
    window.deleteLater()


def test_a_wish_that_was_never_agreed_to_is_not_kept(tmp_path):
    settings = SettingsService(tmp_path / "settings.json")
    settings.update(share_transcripts=True, share_notice_shown=False)                              # switched on by editing the file
    sharing = SharingService(settings, tmp_path / "sharing", endpoint=lambda: ENDPOINT)
    sharing.queue.enqueue({"transcript": "x"})
    sharing.reconcile()
    assert settings.settings.share_transcripts is False and sharing.queue.count() == 0 and not sharing.active()


# --------------------------------------------------------------------------------------------------------------------- withdrawal
def test_withdrawing_stops_uploads_empties_the_queue_and_asks_again_next_time(tmp_path):
    settings = SettingsService(tmp_path / "settings.json")
    calls: list = []
    sharing = SharingService(settings, tmp_path / "sharing", post=lambda url, body: calls.append(body) or 202, endpoint=lambda: ENDPOINT)
    sharing.choose(True)
    folder = lecture(tmp_path)
    assert sharing.transcript_saved(folder, "Lecture") is True and sharing.queue.count() == 1
    sharing.set_enabled(False)                                                                      # withdrawn
    assert sharing.queue.count() == 0 and sharing.uploader.run_once() == 0 and calls == []
    s = settings.settings
    assert (s.share_transcripts, s.share_notice_shown, s.share_choice_made) == (False, False, True)
    settings.update(share_transcripts=True)                                                         # switched on again WITHOUT the notice (a file edit): still nothing
    assert not sharing.active() and sharing.transcript_saved(folder, "Lecture") is False
    sharing.notice_shown()                                                                          # only the notice makes it count again
    assert sharing.active()


# --------------------------------------------------------------------------------------------------------------------- what would be shared
def test_the_preview_lists_only_the_allowed_fields_and_shares_nothing(tmp_path):
    settings = SettingsService(tmp_path / "settings.json")
    sharing = SharingService(settings, tmp_path / "sharing", endpoint=lambda: ENDPOINT)
    folder = lecture(tmp_path, raw="Glycolysis converts glucose to pyruvate. " * 40)
    preview = sharing.preview(folder, "My lecture")
    assert [key for key, _meaning in preview["fields"]] == list(payload.ALLOWED_KEYS)
    assert set(preview["sample"]) <= set(payload.ALLOWED_KEYS) and preview["bytes"] > 0 and "more characters" in preview["excerpt"] and len(preview["excerpt"]) < 700
    assert {"audio", "video", "PDF files", "file paths"} <= set(preview["never"])
    assert sharing.queue.count() == 0 and not sharing.active()                                      # looking is not sharing
    assert sharing.preview(None)["bytes"] == 0


def test_the_preview_button_shows_what_would_be_shared(qapp, tmp_path, monkeypatch):
    from src.gui import sharing_notice

    services = make_services(tmp_path, opened=[])
    window = MainWindow(services, ScriptedLauncher())
    dialog = window.make_settings_dialog("Privacy & Data")
    shown = []
    monkeypatch.setattr(sharing_notice.SharingPreviewDialog, "exec", lambda self: shown.append((self.fields_label.text(), self.never_label.text(), self.sample_label.text())))
    dialog.share_preview.click()
    assert shown and "installation_id" in shown[0][0] and "Never sent: audio, video, PDF files" in shown[0][1] and "Nothing has been shared" in shown[0][2]
    dialog.deleteLater()
    settle(window, window.home, window.study)
    window.deleteLater()


# --------------------------------------------------------------------------------------------------------------------- nothing leaves while it is OFF
def test_while_sharing_is_off_a_finished_lecture_never_touches_the_network(qapp, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("the network was used while transcript sharing is OFF")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setenv("MLT_SHARING_ENDPOINT", ENDPOINT)                                            # an address IS configured: only the choice keeps it quiet
    services = make_services(tmp_path, opened=[])
    window = MainWindow(services, ScriptedLauncher())
    folder = lecture(tmp_path)
    window._share_transcript(folder, {"language": "en"})                                            # the completion path of a real lecture
    services.sharing.queue.enqueue({"transcript": "left over"})                                     # even something that was queued earlier
    assert services.sharing.uploader.run_once() == 0
    services.sharing.start()
    services.sharing.uploader.nudge()
    services.sharing.stop()
    assert not services.sharing.active() and services.sharing.queue.count() == 1                     # (the leftover stays inert; reconcile/withdrawal deletes it)
    services.sharing.reconcile()
    assert services.sharing.queue.count() == 0
    settle(window, window.home, window.study)
    window.deleteLater()


def test_the_shipped_configuration_holds_no_secret():
    from src.sharing import config

    from urllib.parse import urlparse

    data = json.loads(config.CONFIG_FILE.read_text(encoding="utf-8"))
    assert set(data) <= {"endpoint", "note"}                                                        # an address and a note: no key, token, password or mail address
    url = urlparse(data["endpoint"])
    assert data["endpoint"] == "" or (url.scheme == "https" and not url.username and not url.password and not url.query and "@" not in data["endpoint"])


# --------------------------------------------------------------------------------------------------------------------- the first launch of the program
def test_the_first_launch_asks_the_sharing_question_once_after_the_welcome(qapp, tmp_path, monkeypatch):
    from src.gui import app as gui_app
    from src.gui.first_run import FirstRunDialog

    services = make_services(tmp_path, opened=[])
    window = MainWindow(services, ScriptedLauncher())
    order = []
    monkeypatch.setattr(FirstRunDialog, "exec", lambda self: order.append("welcome") or 0)
    monkeypatch.setattr(PrivacyChoiceDialog, "exec", lambda self: order.append("question") or self.share_button.click())     # the student presses the sharing button
    assert gui_app.show_startup_dialogs(services, window) == ["welcome", "sharing"] and order == ["welcome", "question"]
    assert services.settings.settings.share_choice_made and services.settings.settings.share_transcripts and services.sharing.active()
    assert window.simple_home.privacy.text() in ("Sharing enabled", "Sharing on - not connected")                            # the chip shows the answer at once
    services.settings.update(first_run_done=True)
    assert gui_app.show_startup_dialogs(services, window) == [] and order == ["welcome", "question"]                         # the second start asks nothing
    settle(window, window.home, window.study)
    window.deleteLater()


def test_an_unanswered_question_leaves_everything_private(qapp, tmp_path, monkeypatch):
    from src.gui import app as gui_app

    services = make_services(tmp_path, opened=[])
    services.settings.update(first_run_done=True)
    window = MainWindow(services, ScriptedLauncher())
    monkeypatch.setattr(PrivacyChoiceDialog, "exec", lambda self: self.reject())                                              # closed with the X button
    assert gui_app.show_startup_dialogs(services, window) == ["sharing"]
    s = services.settings.settings
    assert (s.share_transcripts, s.share_notice_shown, s.share_choice_made) == (False, False, True) and window.simple_home.privacy.text() == "Local processing"
    settle(window, window.home, window.study)
    window.deleteLater()
