"""Sharing of completed transcripts (Settings -> Privacy & Data): OFF by default and opt-in (the student says yes after reading what is shared), switch-off clears the
queue, local save first, silent failure, nothing secret in the program, and the reference backend end to end."""

import json
import os
import re
import threading
import time
import uuid
from pathlib import Path

import pytest
from PySide6.QtWidgets import QCheckBox

from backend.transcript_inbox import server as inbox_server
from src import __version__
from src.gui.main_window import MainWindow
from src.gui.sharing_notice import NOTICE_HEADLINE, SharingNoticeDialog
from src.services.settings_service import Settings, SettingsService
from src.sharing import config, payload
from src.sharing.queue import BACKOFF_SECONDS, MAX_AGE_DAYS, UploadQueue
from src.sharing.service import SharingService
from src.sharing.uploader import Uploader, http_post
from tests.gui_support import ScriptedLauncher, make_services, settle
from tests.study_support import write_lecture_folder

ROOT = Path(__file__).resolve().parent.parent
REQUIRED_SENTENCE = ("MELETRA shares nothing unless you switch it on.")


def lecture(tmp: Path, raw="Glycolysis converts glucose to pyruvate.", corrected="Glycolysis converts glucose to pyruvate in the cytosol."):
    folder = write_lecture_folder(tmp / "lecture")
    (folder / "local_raw_transcript.txt").write_text(raw, encoding="utf-8")
    (folder / "medical_corrected_transcript.txt").write_text(corrected, encoding="utf-8")
    return folder


def make_sharing(tmp: Path, posts=None, endpoint="https://inbox.example.org/v1/transcripts", shown=True, post=None):
    settings = SettingsService(tmp / "settings.json")
    settings.update(share_transcripts=True)                                       # (the default is OFF: these tests are about what happens once the student switched it on)
    if shown:
        settings.update(share_notice_shown=True)
    calls = posts if posts is not None else []

    def default_post(url, body):
        calls.append((url, json.loads(body)))
        return 202

    return SharingService(settings, tmp / "sharing", post=post or default_post, endpoint=lambda: endpoint), calls


# ------------------------------------------------------------------------------------------------------------------ default, switch, restart
def test_sharing_is_off_by_default_and_does_nothing_until_the_student_agrees(tmp_path):
    assert Settings().share_transcripts is False and Settings().share_notice_shown is False
    fresh = SettingsService(tmp_path / "fresh.json")
    service = SharingService(fresh, tmp_path / "sharing_fresh", endpoint=lambda: "https://inbox.example.org/v1/transcripts")
    assert service.wanted() is False and not service.active() and "Off" in service.state_text()
    assert service.transcript_saved(lecture(tmp_path), "Biochemistry 1") is False and service.queue.count() == 0          # nothing is even queued
    on_without_agreement, _ = make_sharing(tmp_path / "x", shown=False)
    assert on_without_agreement.wanted() is True and not on_without_agreement.active()                                  # switched on by a file edit, not by the notice: still nothing
    assert "not agreed" in on_without_agreement.state_text()


def test_an_older_settings_file_never_carries_a_sharing_default_over(tmp_path):
    old = tmp_path / "settings.json"
    old.write_text(json.dumps({"schema_version": 4, "share_transcripts": True, "share_notice_shown": True, "language": "tr"}), encoding="utf-8")
    loaded = SettingsService(old)
    assert loaded.settings.share_transcripts is False and loaded.settings.share_notice_shown is False and loaded.settings.language == "tr"       # sharing was never decided, only defaulted


def test_switching_it_off_in_settings_survives_a_restart(tmp_path):
    service, _ = make_sharing(tmp_path)
    service.set_enabled(False)
    reloaded = SettingsService(tmp_path / "settings.json")                       # the next program start
    assert reloaded.settings.share_transcripts is False
    assert SharingService(reloaded, tmp_path / "sharing").wanted() is False and "Off" in SharingService(reloaded, tmp_path / "sharing").state_text()


def test_the_privacy_and_data_page_has_the_toggle_the_status_and_works(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("MLT_SHARING_ENDPOINT", "https://inbox.example.org/v1/transcripts")      # (the shipped build has no upload address yet: the chip then says "not connected")
    services = make_services(tmp_path, opened=[])
    window = MainWindow(services, ScriptedLauncher())
    dialog = window.make_settings_dialog("Privacy & Data")
    assert dialog.tabs.tabText(dialog.tabs.currentIndex()) == "Privacy & Data"
    box = dialog.share_check
    assert isinstance(box, QCheckBox) and not box.isChecked() and box.text() == "Share completed transcript text with MELETRA   (default: OFF)"
    assert dialog.share_state.text() == "OFF" and window.simple_home.privacy.text() == "Local processing"
    assert "Upload status" in [label.text() for label in dialog.findChildren(type(dialog.share_status))]
    answers = []

    def answer(self):
        answers.append((self.headline.text(), self.details.text()))
        (self.ok_button if len(answers) % 2 == 0 else self.off_button).click()                  # first "Not now", then "Share transcripts" (no nested event loop)
        return self.result()

    monkeypatch.setattr(SharingNoticeDialog, "exec", answer)
    box.setChecked(True)                                                         # a decision: the notice is shown first ... and "Not now" leaves it OFF
    assert len(answers) == 1 and not box.isChecked() and services.settings.settings.share_transcripts is False
    assert answers[0][0] == NOTICE_HEADLINE and "never sent" in answers[0][1] and "audio" in answers[0][1]
    box.setChecked(True)                                                         # ... "Share transcripts" turns it ON and records that the notice was agreed to
    assert services.settings.settings.share_transcripts is True and services.settings.settings.share_notice_shown is True
    assert dialog.share_state.text() == "ON" and window.simple_home.privacy.text() == "Sharing enabled"
    services.sharing.queue.enqueue({"transcript": "x"})
    box.setChecked(False)                                                        # off takes effect at once, without pressing Save, and without a question
    assert services.settings.settings.share_transcripts is False and services.sharing.queue.count() == 0 and len(answers) == 2
    assert "Off" in dialog.share_status.text()
    assert window.simple_home.privacy.text() == "Local processing"                 # the Home status chip follows at once
    assert dialog.share_state.text() == "OFF" and dialog.share_state.property("kind") == "off"       # and the page says so unmistakably
    dialog.deleteLater()
    settle(window, window.home, window.study)
    window.deleteLater()


# ------------------------------------------------------------------------------------------------------------------ the notice
def test_nothing_about_sharing_is_asked_when_the_program_starts(qapp, tmp_path):
    services = make_services(tmp_path, opened=[])
    window = MainWindow(services, ScriptedLauncher())
    assert not hasattr(window, "maybe_show_sharing_notice")                      # no start-up dialog: sharing is only ever asked about when the student switches it on
    assert services.settings.settings.share_transcripts is False and services.settings.settings.share_notice_shown is False
    settle(window, window.home, window.study)
    window.deleteLater()


def test_the_notice_is_a_plain_question_whose_safe_answer_is_not_now(qapp, tmp_path):
    services = make_services(tmp_path, opened=[])
    dialog = SharingNoticeDialog(services)
    assert [b.text() for b in (dialog.off_button, dialog.ok_button)] == ["Not now", "Share transcripts"]
    assert dialog.off_button.isDefault() and not dialog.ok_button.isDefault()    # Enter answers "Not now"
    assert "MELETRA" in dialog.headline.text() and "Nothing is shared unless you say yes" in dialog.headline.text()
    dialog.off_button.click()
    assert services.settings.settings.share_transcripts is False and services.settings.settings.share_notice_shown is False
    other = SharingNoticeDialog(make_services(tmp_path / "second", opened=[]))
    other.reject()                                                               # the X button / Esc: the same as "Not now"
    assert other.services.settings.settings.share_transcripts is False and other.services.settings.settings.share_notice_shown is False
    third = SharingNoticeDialog(make_services(tmp_path / "third", opened=[]))
    third.ok_button.click()
    assert third.services.settings.settings.share_transcripts is True and third.services.settings.settings.share_notice_shown is True


def test_the_installer_notice_says_the_same_thing():
    text = (ROOT / "build_tools" / "installer_notice.txt").read_text(encoding="utf-8")
    assert REQUIRED_SENTENCE in text and "off by default" in text and "What is never sent" in text and "MELETRA" in text
    iss = (ROOT / "build_tools" / "installer.iss").read_text(encoding="utf-8")
    assert "InfoBeforeFile={#NoticeFile}" in iss                                  # a wizard page BEFORE anything is installed
    assert "NoticeFile" in (ROOT / "scripts" / "build_installer.py").read_text(encoding="utf-8")


# ------------------------------------------------------------------------------------------------------------------ what is sent
def test_the_payload_contains_only_the_allowed_fields_and_nothing_personal(tmp_path):
    folder = lecture(tmp_path)
    (folder / "translated_tr_transcript.txt").write_text("Glikoliz glikozu piruvata cevirir.", encoding="utf-8")
    installation = config.installation_id(tmp_path / "sharing")
    data = payload.build(folder, "Biochemistry 1", installation, __version__, spoken="en", requested="auto", output="tr")
    assert set(data) <= set(payload.ALLOWED_KEYS) == set(inbox_server.ALLOWED_KEYS)         # the client's allow-list IS the server's
    assert data["transcript"] == "Glycolysis converts glucose to pyruvate." and "cytosol" in data["medical_corrected_transcript"]
    assert data["translated_transcript"].startswith("Glikoliz") and data["lecture_name"] == "Biochemistry 1"
    assert data["languages"] == {"spoken": "en", "requested": "auto", "output": "tr"} and data["app_version"] == __version__
    uuid.UUID(installation)                                                       # a random uuid, no hardware or user information in it
    text = json.dumps(data)
    for private in (str(tmp_path), str(folder), os.environ.get("USERNAME", "zzzzzz"), os.environ.get("COMPUTERNAME", "zzzzzz"), ".wav", ".mp4", "C:\\"):
        assert private not in text, private
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\+00:00", data["created_at"])
    assert config.installation_id(tmp_path / "sharing") == installation           # stable on this computer


def test_there_is_nothing_to_share_without_a_transcript(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert payload.build(empty, "x", str(uuid.uuid4()), "4.1.0") is None


# ------------------------------------------------------------------------------------------------------------------ local save first, silent failure, queue
def test_the_transcript_is_on_disk_before_anything_is_uploaded(tmp_path):
    folder = lecture(tmp_path)
    seen = []

    def post(url, body):
        seen.append({"file_exists": (folder / "local_raw_transcript.txt").is_file(), "same_text": (folder / "local_raw_transcript.txt").read_text(encoding="utf-8")
                     == json.loads(body)["transcript"]})
        return 202

    service, _ = make_sharing(tmp_path, post=post)
    assert service.transcript_saved(folder, "Lecture", spoken="en") is True
    assert seen == [] and service.queue.count() == 1                              # saving + queueing did not touch the network
    assert service.uploader.run_once() == 1
    assert seen == [{"file_exists": True, "same_text": True}] and service.queue.count() == 0 and service.queue.status()["sent"] == 1


def test_an_upload_failure_is_silent_keeps_the_transcript_and_the_item_for_later(tmp_path):
    folder = lecture(tmp_path)

    def broken(url, body):
        raise OSError("network is unreachable")

    service, _ = make_sharing(tmp_path, post=broken)
    before = time.perf_counter()
    assert service.transcript_saved(folder, "Lecture") is True                   # returns at once, no exception, no dialog
    assert time.perf_counter() - before < 1.0
    assert service.uploader.run_once() == 0
    [item] = list(service.queue.all())
    assert item.attempts == 1 and item.next_try > time.time() + BACKOFF_SECONDS[0] - 5 and "unreachable" in item.last_error
    assert (folder / "local_raw_transcript.txt").read_text(encoding="utf-8").startswith("Glycolysis")          # the local transcript is untouched
    assert "unreachable" in service.status()["last_error"]                        # visible only in Settings -> Privacy & Data -> Upload status
    assert service.uploader.run_once() == 0 and [i.attempts for i in service.queue.all()] == [1]                # not due yet: it waits, it does not hammer
    service.queue.failed(item, "again", now=time.time() + 400)
    assert [i.attempts for i in service.queue.all()] == [2] and BACKOFF_SECONDS[1] > BACKOFF_SECONDS[0]


def test_a_server_error_is_also_just_a_later_retry(tmp_path):
    folder = lecture(tmp_path)
    service, _ = make_sharing(tmp_path, post=lambda url, body: 503)
    service.transcript_saved(folder, "Lecture")
    assert service.uploader.run_once() == 0 and [i.last_error for i in service.queue.all()] == ["HTTP 503"]


def test_switching_sharing_off_clears_the_queue_and_stops_uploads_at_once(tmp_path):
    folder = lecture(tmp_path)
    service, posts = make_sharing(tmp_path, post=lambda url, body: 503)
    for i in range(3):
        service.transcript_saved(folder, f"Lecture {i}")
    service.uploader.run_once()
    assert service.queue.count() == 3
    service.set_enabled(False)
    assert service.queue.count() == 0 and service.queue.status()["failed_attempts"] == 0          # the queue and its status are gone
    assert service.transcript_saved(folder, "Lecture 4") is False and service.queue.count() == 0   # nothing new is queued while it is off
    assert service.uploader.run_once() == 0
    service.set_enabled(True)
    assert service.queue.count() == 0                                             # switching on again does not bring the old copies back


def test_a_switch_off_during_a_running_pass_stops_the_remaining_items(tmp_path):
    folder = lecture(tmp_path)
    holder = {}

    def post(url, body):
        holder["service"].set_enabled(False)                                       # the student switches it off while the first upload is running
        return 202

    service, _ = make_sharing(tmp_path, post=post)
    holder["service"] = service
    for i in range(3):
        service.transcript_saved(folder, f"Lecture {i}")
    service.uploader.run_once()
    assert service.queue.count() == 0 and service.queue.status()["sent"] == 0       # no item was counted or sent after the switch


def test_nothing_is_sent_before_the_notice_or_without_an_address(tmp_path):
    folder = lecture(tmp_path)
    service, posts = make_sharing(tmp_path, shown=False)
    assert service.transcript_saved(folder, "Lecture") is False and service.uploader.run_once() == 0 and posts == []
    service.notice_shown()
    assert service.transcript_saved(folder, "Lecture") is True
    unconfigured, posts2 = make_sharing(tmp_path / "other", endpoint="")
    (tmp_path / "other").mkdir(exist_ok=True)
    unconfigured.queue.enqueue({"transcript": "x"})
    assert unconfigured.uploader.run_once() == 0 and posts2 == [] and "no upload address" in unconfigured.state_text()


def test_old_items_are_dropped_and_the_log_stays_small(tmp_path):
    queue = UploadQueue(tmp_path / "q")
    queue.enqueue({"transcript": "old"}, now=time.time() - (MAX_AGE_DAYS + 1) * 86400)
    queue.enqueue({"transcript": "new"})
    assert queue.drop_old() == 1 and [i.payload["transcript"] for i in queue.all()] == ["new"]
    for i in range(400):
        queue.record(failure="x" * 150)
    assert queue.log_file.stat().st_size < 260_000


def test_the_completion_of_a_lecture_queues_it_without_ever_failing_the_lecture(qapp, tmp_path):
    services = make_services(tmp_path, opened=[])
    services.sharing.settings.update(share_transcripts=True, share_notice_shown=True)                   # the student switched it on and agreed
    window = MainWindow(services, ScriptedLauncher())
    folder = lecture(tmp_path)
    window._share_transcript(folder, {"language": "en"})
    assert services.sharing.queue.count() == 1                                    # the endpoint is not configured in the shipped config, so nothing is sent:
    assert config.endpoint() == "" and services.sharing.uploader.run_once() == 0   # sharing is on by default, the address is the owner's later step
    services.sharing.transcript_saved = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    window._share_transcript(folder, {"language": "en"})                           # even a bug in sharing cannot raise into the completion handler
    settle(window, window.home, window.study)
    window.deleteLater()


# ------------------------------------------------------------------------------------------------------------------ no secret in the program
def test_the_program_holds_no_mail_address_password_or_key():
    pattern = re.compile(r"(?i)smtp|gmail|app[ _-]?password|sendgrid|mailgun|mailchimp|postmark|@[a-z0-9-]+\.(com|org|net)\b")
    for folder in (ROOT / "src", ROOT / "config"):
        for path in folder.rglob("*"):
            if path.suffix in (".py", ".json", ".yaml") and path.is_file() and "licenses" not in path.parts:          # (licence texts of bundled data name their authors)
                text = path.read_text(encoding="utf-8", errors="ignore")
                for match in pattern.finditer(text):
                    pytest.fail(f"{path.relative_to(ROOT)} contains {match.group(0)!r}")
    assert json.loads((ROOT / "config" / "sharing.json").read_text(encoding="utf-8"))["endpoint"] == ""      # shipped without an address until the owner deploys the backend
    for binary in (ROOT / "dist").glob("*/MedicalLectureTranscriber/MedicalLectureTranscriber.exe"):             # a built program, when there is one
        data = binary.read_bytes()
        for needle in (b"owner@example.invalid", b"SMTP_PASSWORD", b"smtplib.SMTP("):
            assert needle not in data, (binary.name, needle)


def test_the_endpoint_must_be_https_and_the_server_holds_the_mail_settings(tmp_path):
    assert config.valid_endpoint("https://inbox.example.org/v1/transcripts") and config.valid_endpoint("http://127.0.0.1:8080/x")
    assert not config.valid_endpoint("http://inbox.example.org/x") and not config.valid_endpoint("ftp://x/y") and not config.valid_endpoint("")
    file = tmp_path / "sharing.json"
    file.write_text(json.dumps({"endpoint": "http://evil.example.org/"}), encoding="utf-8")
    assert config.endpoint(file) == ""                                            # a plain-http address from the config is refused
    server_text = (ROOT / "backend" / "transcript_inbox" / "server.py").read_text(encoding="utf-8")
    assert "OWNER_NOTIFICATION_EMAIL" in server_text and "SMTP_PASSWORD" in server_text


# ------------------------------------------------------------------------------------------------------------------ the reference backend
def run_inbox(tmp_path, env=None, smtp=None):
    env = {"INBOX_DIR": str(tmp_path / "inbox"), **(env or {})}
    inbox = inbox_server.Inbox(Path(env["INBOX_DIR"]))
    notifier = inbox_server.SmtpNotifier(env, smtp_factory=smtp)
    httpd = inbox_server.ThreadingHTTPServer(("127.0.0.1", 0), inbox_server.make_handler(inbox, notifier, inbox_server.RateLimiter(per_hour=5)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_port}/v1/transcripts", inbox


class FakeSmtp:
    sent = []

    def __init__(self, host, port, timeout=0):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def starttls(self, context=None):
        pass

    def login(self, user, password):
        FakeSmtp.login_used = (user, password)

    def send_message(self, message):
        FakeSmtp.sent.append(message)


def test_end_to_end_the_client_uploads_to_the_backend_which_stores_and_notifies(tmp_path):
    FakeSmtp.sent.clear()
    env = {"OWNER_NOTIFICATION_EMAIL": "owner@example.org", "SMTP_HOST": "smtp.example.org", "SMTP_USER": "sender@example.org", "SMTP_PASSWORD": "server-secret"}
    httpd, url, inbox = run_inbox(tmp_path, env, FakeSmtp)
    try:
        folder = lecture(tmp_path)
        service, _ = make_sharing(tmp_path, post=http_post, endpoint=url)                 # the real client code over a real HTTP connection
        service.transcript_saved(folder, "Biochemistry 1", spoken="en", requested="auto", output="en")
        assert service.uploader.run_once() == 1 and service.queue.count() == 0
    finally:
        httpd.shutdown()
    [stored] = list(inbox.folder.rglob("*.json"))
    data = json.loads(stored.read_text(encoding="utf-8"))
    assert data["transcript"].startswith("Glycolysis") and data["lecture_name"] == "Biochemistry 1"
    [message] = FakeSmtp.sent
    assert message["To"] == "owner@example.org" and "Biochemistry 1" in message["Subject"]               # the recipient came from the SERVER's environment
    assert message.get_body().get_content().count("Words:") == 1 and any(p.get_filename() == "transcript.txt" for p in message.iter_attachments())


def test_the_backend_refuses_unknown_fields_oversize_and_bursts_and_keeps_what_mail_loses(tmp_path):
    httpd, url, inbox = run_inbox(tmp_path)                                         # no mail configured: the transcript is stored all the same
    good = payload.build(lecture(tmp_path), "L", str(uuid.uuid4()), __version__, spoken="en")

    def post(body: dict | bytes):
        try:
            return http_post(url, body if isinstance(body, bytes) else json.dumps(body).encode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            return getattr(exc, "code", -1)

    try:
        assert post({**good, "audio_path": "C:\\Users\\x\\a.wav"}) == 400                 # a field that is not on the allow-list
        assert post({**good, "installation_id": "not-a-uuid"}) == 400
        assert post(b"{not json") == 400
        assert post({**good, "transcript": " "}) == 400
        assert post(b"x" * (inbox_server.MAX_BYTES + 10)) in (413, 400, -1)
        assert post(good) == 202 and len(list(inbox.folder.rglob("*.json"))) == 1       # stored although there is no mail account
        codes = [post(good) for _ in range(8)]
        assert 429 in codes                                                            # a burst from one installation is refused
    finally:
        httpd.shutdown()
