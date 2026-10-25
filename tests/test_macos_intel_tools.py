"""The Intel-Mac release tooling: the rules and parsers that run on the Mac, and the Windows finalizer that decides "ready". Everything here
runs on Windows with fake data - it proves the LOGIC (and that nothing is declared ready without evidence), not that a Mac build works."""

import json
import sys
from pathlib import Path

import pytest

from build_tools import macos_intel as intel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))


def passing_report(sha="a" * 64):
    return {"real_hardware": True, "machine": "x86_64", "network_disconnected": True, "dmg_sha256": sha,
            "checks": {k: "PASS" for k in intel.VALIDATION_KEYS}}


MANIFEST_OK = {"gpl_source_offer": "SET", "study_quality_included": True}


# ------------------------------------------------------------------------------------------------------------ readiness
def test_no_report_from_a_real_intel_mac_means_real_intel_mac_required():
    assert intel.readiness({}, "a" * 64, MANIFEST_OK)[0] == "REAL INTEL MAC REQUIRED"
    arm = {**passing_report(), "machine": "arm64"}
    assert intel.readiness(arm, "a" * 64, MANIFEST_OK)[0] == "REAL INTEL MAC REQUIRED"
    vm = {**passing_report(), "real_hardware": False}
    assert intel.readiness(vm, "a" * 64, MANIFEST_OK)[0] == "REAL INTEL MAC REQUIRED"


def test_everything_passing_on_the_same_file_with_the_network_off_is_ready():
    assert intel.readiness(passing_report(), "a" * 64, MANIFEST_OK) == ("READY TO UPLOAD", [])


@pytest.mark.parametrize("change, expected", [
    ({"network_disconnected": False}, "network"),
    ({"dmg_sha256": "b" * 64}, "not the DMG"),
    ({"checks": {**{k: "PASS" for k in intel.VALIDATION_KEYS}, "ocr": "FAIL"}}, "ocr: FAIL"),
    ({"checks": {k: "PASS" for k in intel.VALIDATION_KEYS if k != "queue"}}, "queue: not run"),
])
def test_any_gap_keeps_it_at_needs_work(change, expected):
    verdict, reasons = intel.readiness({**passing_report(), **change}, "a" * 64, MANIFEST_OK)
    assert verdict == "NEEDS WORK" and any(expected in r for r in reasons), reasons


def test_the_gpl_source_offer_and_study_quality_gate_the_release():
    assert "source-offer" in " ".join(intel.readiness(passing_report(), "a" * 64, {**MANIFEST_OK, "gpl_source_offer": "MISSING"})[1])
    assert "Study Quality" in " ".join(intel.readiness(passing_report(), "a" * 64, {**MANIFEST_OK, "study_quality_included": False})[1])


# ------------------------------------------------------------------------------------------------------------ parsers
OTOOL_GOOD = """/app/x.dylib:
\t@rpath/libssl.3.dylib (compatibility version 3.0.0, current version 3.0.0)
\t@loader_path/../libs/libfoo.dylib (compatibility version 1.0.0, current version 1.0.0)
\t/usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1311.0.0)
\t/System/Library/Frameworks/Cocoa.framework/Versions/A/Cocoa (compatibility version 1.0.0, current version 23.0.0)
"""
OTOOL_BAD = OTOOL_GOOD + "\t/opt/homebrew/opt/openssl@3/lib/libssl.3.dylib (compatibility version 3.0.0, current version 3.0.0)\n" \
                         "\t/Users/dev/build/libz.dylib (compatibility version 1.0.0, current version 1.0.0)\n" \
                         "\t/usr/local/Cellar/ffmpeg/lib/libavcodec.dylib (compatibility version 1.0.0, current version 1.0.0)\n"


def test_otool_audit_accepts_the_apps_own_and_system_libraries_and_flags_everything_else():
    assert intel.audit_otool("x.dylib", OTOOL_GOOD) == []
    problems = intel.audit_otool("x.dylib", OTOOL_BAD)
    assert len(problems) == 3 and all("depends on" in p for p in problems)
    assert any("homebrew" in p for p in problems) and any("/Users/dev" in p for p in problems) and any("Cellar" in p for p in problems)


def test_rpath_audit_flags_developer_machine_paths():
    out = "Load command 12\n          cmd LC_RPATH\n      cmdsize 40\n         path /Users/dev/venv/lib (offset 12)\nLoad command 13\n" \
          "          cmd LC_RPATH\n      cmdsize 32\n         path @loader_path/../Frameworks (offset 12)\n"
    assert intel.audit_rpaths("a", out) == ["a: rpath /Users/dev/venv/lib"]


def test_peak_memory_comes_from_usr_bin_time_l():
    err = "       12.34 real        40.00 user         3.00 sys\n  2147483648  maximum resident set size\n           0  average shared memory size\n"
    assert intel.parse_peak_memory(err) == 2147483648 and intel.parse_peak_memory("nothing") is None
    assert intel.realtime_factor(30, 60) == 0.5 and intel.realtime_factor(5, 0) is None


# ------------------------------------------------------------------------------------------------------------ validator helpers
def load_validator():
    import importlib
    return importlib.import_module("validate_macos_intel")


def test_video_progress_needs_every_early_step_in_order():
    v = load_validator()
    events = [{"type": "job_started", "time": 100.0}, {"type": "source_validated", "time": 100.7}, {"type": "audio_preparing", "time": 100.9},
              {"type": "audio_ready", "time": 104.0}, {"type": "language_detecting", "time": 104.1}, {"type": "model_loading", "time": 105.0},
              {"type": "chunk_progress", "time": 120.0}]
    assert v.video_progress_ok(events)[0] is True
    slow = [{**e, "time": e["time"] + (5 if e["type"] != "job_started" else 0)} for e in events]
    assert v.video_progress_ok(slow)[0] is False                                    # the first step 5.7 s after the start
    assert v.video_progress_ok(events[:2])[0] is False                              # audio steps missing
    wrong = [dict(e) for e in events]
    wrong[3]["time"] = 130.0
    assert v.video_progress_ok(wrong)[0] is False                                   # ready after the first progress


def test_documents_on_a_mac_expect_rapidocr_and_real_text(tmp_path):
    v = load_validator()
    items = []
    for name, source, label in (("embedded.pdf", "embedded_text", "PDF text"), ("scan.pdf", "ocr", "RapidOCR"), ("slide.png", "ocr", "RapidOCR"),
                                ("notes.txt", "text_file", "Text file")):
        (tmp_path / f"{name}.json").write_text(json.dumps({"text_source": source, "label": label}), encoding="utf-8")
        (tmp_path / f"{name}.txt").write_text("the nephron filters plasma", encoding="utf-8")
        items.append({"name": name, "state": "completed", "files": {"json": str(tmp_path / f"{name}.json"), "text": str(tmp_path / f"{name}.txt")}})
    assert v.check_documents_mac(items)[0] is True
    (tmp_path / "scan.pdf.json").write_text(json.dumps({"text_source": "ocr", "label": "Windows text recognition"}), encoding="utf-8")
    assert v.check_documents_mac(items)[0] is False


def test_medical_check_wants_the_files_the_expert_layer_writes_and_an_untouched_raw_transcript(tmp_path):
    v = load_validator()
    assert v.medical_check(tmp_path)[0] is False
    for name in ("local_raw_transcript.txt", "local_clean_transcript.txt"):
        (tmp_path / name).write_text("nisslo bodies", encoding="utf-8")
    (tmp_path / "medical_corrected_transcript.txt").write_text("Nissl bodies. Schwann cells. Tanycytes. Remak bundles.", encoding="utf-8")
    (tmp_path / "correction_report.json").write_text(json.dumps({"raw_transcript_modified": False}), encoding="utf-8")
    (tmp_path / "medical_quality_report.json").write_text(json.dumps({"medical_terms_detected": 18}), encoding="utf-8")
    assert v.medical_check(tmp_path)[0] is True
    (tmp_path / "medical_quality_report.json").write_text(json.dumps({"medical_terms_detected": 0}), encoding="utf-8")
    assert v.medical_check(tmp_path)[0] is False                                    # no vocabulary in the app = the bug this release fixes


def test_dev_files_opened_by_the_app_are_reported():
    v = load_validator()
    assert v.no_dev_files({"/Applications/MELETRA.app/Contents/MacOS/x", "/usr/lib/libSystem.B.dylib"})[0] is True and v.no_dev_files({"/Applications/Medical Lecture Transcriber.app/Contents/MacOS/x"})[0] is True
    assert v.no_dev_files({"/opt/homebrew/lib/libfoo.dylib"})[0] is False
    assert v.no_dev_files({"/Users/me/proj/.venv-macos-intel/lib/python3.11/site-packages/x.so"})[0] is False


def test_network_check_needs_both_the_route_and_the_name_lookup_to_fail(monkeypatch):
    v = load_validator()
    monkeypatch.setattr(v.socket, "create_connection", lambda *a, **k: (_ for _ in ()).throw(OSError("unreachable")))
    monkeypatch.setattr(v.socket, "getaddrinfo", lambda *a, **k: (_ for _ in ()).throw(OSError("no dns")))
    assert v.network_is_off()[0] is True
    monkeypatch.setattr(v.socket, "getaddrinfo", lambda *a, **k: [("x",)])
    assert v.network_is_off()[0] is False


# ------------------------------------------------------------------------------------------------------------ notices, build script
def test_notices_are_accurate_about_the_gpl_libraries_and_the_model():
    text = intel.third_party_licenses("", {"av": "18.1.0"})
    assert "libx264" in text and "libx265" in text and "NOT SET" in text
    assert "Contact for source-code requests: you@example.com" in intel.third_party_licenses("you@example.com")
    model = intel.model_licenses({"model.bin": {"size": 10, "sha256": "ab" * 32}})
    assert "MIT" in model and "ab" * 32 in model and "Nothing else is downloaded" in model


def test_the_build_script_refuses_everything_but_an_intel_mac_and_prints_its_plan(capsys, monkeypatch):
    import build_macos_intel_offline as b
    monkeypatch.setattr(b, "host_facts", lambda: {"system": "Windows", "machine": "AMD64", "macos": "", "python": "3.11.9"})
    with pytest.raises(SystemExit) as err:
        b.preflight()
    assert "REAL INTEL MAC REQUIRED" in str(err.value)
    assert b.main(["--dry-run"]) == 0 and "hdiutil create" in capsys.readouterr().out
    assert b.intel.DMG_NAME == "MELETRA-Intel-Mac-Offline.dmg" and b.intel.APP_FOLDER == "MELETRA.app" and b.intel.LEGACY_APP_FOLDER == "Medical Lecture Transcriber.app"


def test_the_shell_wrapper_checks_darwin_and_x86_64_before_building():
    text = (ROOT / "build_macos_intel.sh").read_text(encoding="utf-8")
    assert text.index("uname -s") < text.index("uname -m") < text.index("pip install -r requirements-macos-intel.txt")
    assert "REAL INTEL MAC REQUIRED" in text and "set -euo pipefail" in text


# ------------------------------------------------------------------------------------------------------------ the Windows finalizer
def finalize(tmp_path, monkeypatch, report, manifest_extra=None):
    import finalize_mac_dmg as f
    dmg = tmp_path / "in" / intel.DMG_NAME
    dmg.parent.mkdir()
    dmg.write_bytes(b"pretend this is a disk image" * 100)
    digest = f.sha256(dmg)
    manifest = {"dmg_sha256": digest, "gpl_source_offer": "SET", "study_quality_included": True, **(manifest_extra or {})}
    (tmp_path / "in" / "release_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    args = ["--dmg", str(dmg), "--manifest", str(tmp_path / "in" / "release_manifest.json"), "--desktop", str(tmp_path / "desktop")]
    if report is not None:
        (tmp_path / "in" / "validation_report.json").write_text(json.dumps({**report, "dmg_sha256": digest}), encoding="utf-8")
        args += ["--validation", str(tmp_path / "in" / "validation_report.json")]
    (tmp_path / "desktop").mkdir()
    monkeypatch.setattr(f, "TARGET", tmp_path / "release")
    return f.main(args), tmp_path


def test_the_finalizer_copies_nothing_without_a_validation_report(tmp_path, monkeypatch):
    code, base = finalize(tmp_path, monkeypatch, None)
    assert code == 1 and not list((base / "desktop").iterdir()) and not (base / "release").exists()


def test_the_finalizer_copies_byte_identical_files_only_when_everything_passed(tmp_path, monkeypatch, capsys):
    code, base = finalize(tmp_path, monkeypatch, passing_report())
    assert code == 0, capsys.readouterr().out
    for folder in (base / "desktop", base / "release"):
        assert (folder / intel.DMG_NAME).read_bytes() == (base / "in" / intel.DMG_NAME).read_bytes()
        assert (folder / f"{intel.DMG_NAME}.sha256").read_text(encoding="utf-8").startswith(base_hash(base))


def base_hash(base):
    import finalize_mac_dmg as f
    return f.sha256(base / "in" / intel.DMG_NAME)


def test_the_finalizer_refuses_a_failed_check(tmp_path, monkeypatch):
    bad = passing_report()
    bad["checks"]["ocr"] = "FAIL"
    code, base = finalize(tmp_path, monkeypatch, bad)
    assert code == 1 and not list((base / "desktop").iterdir())


# ------------------------------------------------------------------------------------------------------------ CI packaged-app checks
def ci():
    import importlib
    return importlib.import_module("ci_validate_packaged")


def video_trace():
    t = [(0.1, "active.title", "Preparing video..."), (0.1, "active.stage", "Reading the file ..."), (0.1, "active.percent", "0%"),
         (1.2, "active.title", "Extracting audio..."), (1.3, "active.percent", "100%"), (2.0, "active.stage", "Loading the speech model ..."),
         (9.0, "active.title", "Analyzing the recording..."), (9.1, "active.percent", "12%"), (9.5, "active.title", "Transcribing..."),
         (11.0, "active.percent", "8%"), (14.0, "active.title", "Finalizing..."), (15.0, "result.title", "Lecture Ready")]
    return [list(x) for x in t]


def test_video_stage_check_reads_the_windows_own_texts_in_order():
    c = ci()
    ok, note = c.stage_check({"ui_trace": video_trace()})
    assert ok, note
    frozen = [r for r in video_trace() if r[1] != "active.percent" or r[2] == "0%"]
    assert c.stage_check({"ui_trace": frozen})[0] is False                              # the percent never left 0%
    missing = [r for r in video_trace() if "Extracting" not in r[2]]
    assert c.stage_check({"ui_trace": missing})[0] is False
    shuffled = video_trace()
    shuffled[3][2], shuffled[8][2] = shuffled[8][2], shuffled[3][2]
    assert c.stage_check({"ui_trace": shuffled})[0] is False                            # out of order


def queue_trace(early_result=False):
    rows = [(0.2, "batch.title", "Converting 1 of 2"), (0.3, "batch.percent", "0%"), (6.0, "batch.percent", "55%"), (9.0, "batch.title", "Converting 2 of 2"),
            (9.5, "batch.percent", "14%"), (14.0, "batch.percent", "83%"), (18.0, "batch.title", "Done")]
    if early_result:
        rows.insert(3, (8.8, "result.title", "Lecture Ready"))
    return [list(r) for r in rows]


def test_queue_check_wants_two_of_two_visible_and_no_result_screen_in_between():
    c = ci()
    items = [{"state": "completed"}, {"state": "completed"}]
    assert c.queue_check({"ui_trace": queue_trace(), "items": items})[0] is True
    ok, note = c.queue_check({"ui_trace": queue_trace(early_result=True), "items": items})
    assert ok is False and "Lecture Ready" in note                                      # the bug fixed in 3.1.0: file 1's result hid file 2
    assert c.queue_check({"ui_trace": [r for r in queue_trace() if r[2] != "Converting 2 of 2"], "items": items})[0] is False
    no_progress = [r for r in queue_trace() if not (r[1] == "batch.percent" and r[0] > 9.0)]
    assert c.queue_check({"ui_trace": no_progress, "items": items})[0] is False


def study_report(**over):
    card = {"buttons": ["Back", "Previous", "Next", "Know", "Unsure", "Don't know", "Transcript 21:50-23:03", "Shuffle"],
            "labels": ["Topic:", "Type:", "Level:", "Show:", "Key concepts", "Question preview", "Study progress"]}
    base = {"tabs": ["Overview", "Summary", "Key Concepts", "Flashcards", "Questions", "Study Mode"], "stats": {"concepts": "24", "flashcards": "35", "questions": "22", "mastery": "0%"},
            "actions": ["Back", "Regenerate", "Export", "Start Study Mode"], "tab_content": {"Flashcards": card},
            "engine": {"engine": "NEW QUALITY ENGINE (validated study facts)", "flashcards": {"items": 35, "from_validated_facts": 35}, "questions": {"items": 22, "from_validated_facts": 22}}}
    return {"ui_study": {**base, **over}}


def test_study_check_wants_tabs_filters_rail_rating_buttons_and_a_source_link():
    c = ci()
    assert c.study_check(study_report())[0] is True
    assert c.study_check(study_report(tabs=["Overview"]))[0] is False
    ok, note = c.study_check(study_report(engine={"engine": "OLD ENGINE (sentence based)"}))
    assert ok is False and "new quality engine" in note                                                       # the old method must not pass as the quality engine
    untagged = {"engine": "NEW QUALITY ENGINE (validated study facts)", "flashcards": {"items": 35, "from_validated_facts": 0}, "questions": {"items": 22, "from_validated_facts": 22}}
    assert c.study_check(study_report(engine=untagged))[0] is False
    no_rating = study_report()
    no_rating["ui_study"]["tab_content"]["Flashcards"]["buttons"] = ["Previous", "Next"]
    ok, note = c.study_check(no_rating)
    assert ok is False and "flashcard buttons" in note and "source link" in note
    assert c.study_check({})[0] is False


def test_frontend_and_first_launch_checks():
    c = ci()
    home = {"ui_home": {"navigation": ["Home", "Transcription", "Study", "Files", "Settings"], "privacy": "Sharing enabled",
                        "home": {"labels": ["Drop lecture files here", "Audio · Video · PDF · Image"], "buttons": ["Browse files", "TRANSCRIBE"]}}}
    assert c.frontend_check(home)[0] is True
    home["ui_home"]["navigation"] = ["Home"]
    assert c.frontend_check(home)[0] is False
    ready = {"ui_first_launch": {"model_text": "Whisper Large-v3 is ready. / It is installed (...) and is used where it is", "download_button_hidden": True,
                                 "import_folder_button_hidden": True}}
    assert c.first_launch_check(ready)[0] is True
    ready["ui_first_launch"]["download_button_hidden"] = False
    assert c.first_launch_check(ready)[0] is False                                      # a downloader on screen = the release is not self-contained


def test_medical_outputs_check_needs_a_vocabulary_effect_and_an_untouched_raw_transcript(tmp_path):
    c = ci()
    for n, text in (("local_raw_transcript.txt", "the nislo bodies"), ("local_clean_transcript.txt", "the nislo bodies"),
                    ("medical_corrected_transcript.txt", "the Nissl bodies and Schwann cells")):
        (tmp_path / n).write_text(text, encoding="utf-8")
    (tmp_path / "correction_report.json").write_text(json.dumps({"raw_transcript_modified": False}), encoding="utf-8")
    (tmp_path / "medical_quality_report.json").write_text(json.dumps({"medical_terms_detected": 5, "detected_subject": "neuroanatomy"}), encoding="utf-8")
    ok, note = c.medical_outputs_check(tmp_path)
    assert ok and "nissl" in note
    (tmp_path / "medical_quality_report.json").write_text(json.dumps({"medical_terms_detected": 0, "detected_subject": ""}), encoding="utf-8")
    assert c.medical_outputs_check(tmp_path)[0] is False


# ------------------------------------------------------------------------------------------------------------ minos, CI readiness
def test_minimum_macos_comes_from_the_platform_one_build_command():
    otool = ("Load command 9\n      cmd LC_BUILD_VERSION\n  cmdsize 32\n platform 1\n    minos 11.0\n      sdk 14.0\n"
             "Load command 10\n      cmd LC_VERSION_MIN_MACOSX\n  cmdsize 16\n  version 10.9\n      sdk 10.14\n")
    assert intel.minos_of(otool) == (11, 0) and intel.minos_of("nothing") is None
    assert intel.minos_of("cmd LC_BUILD_VERSION\n  cmdsize 32\n platform 6\n    minos 13.1\n") is None        # Mac Catalyst does not count
    assert intel.split_name("a.dmg", 3) == "a.dmg.part003"


def ci_report(sha="a" * 64):
    return {"provider": "GitHub Actions", "machine": "x86_64", "dmg_sha256": sha, "checks": {k: "PASS" for k in intel.CI_KEYS}}


def ci_manifest(sha="a" * 64, **over):
    return {"dmg_sha256": sha, "dmg_check": {"mount": True, "copy": True, "eject": True, "selftest": True}, "mach_o_audit": {"problems": 0},
            "tests": {"passed": 1280, "failed": 0, "skipped": 1, "ran": True}, "study_quality_included": True, "developer_id": False, "notarized": False,
            "gpl_source_offer": "MISSING", "min_macos_effective": "12.0", **over}


def test_ci_readiness_is_about_the_tests_and_keeps_the_limits_in_the_caveats():
    verdict, reasons, caveats = intel.ci_readiness(ci_report(), "a" * 64, ci_manifest())
    assert verdict == "CI-TESTED INTEL MAC DMG READY" and reasons == []
    text = " ".join(caveats)
    assert "NOT on a physical 2019 MacBook" in text and "Gatekeeper" in text and "not notarized" in text and "source-offer" in text


@pytest.mark.parametrize("report_change, manifest_change, expected", [
    ({"checks": {**{k: "PASS" for k in intel.CI_KEYS}, "queue": "FAIL"}}, {}, "queue: FAIL"),
    ({}, {"dmg_sha256": "b" * 64}, "not the file the CI build recorded"),
    ({"dmg_sha256": "c" * 64}, {}, "different DMG"),
    ({}, {"dmg_check": {"mount": True, "copy": True, "eject": False, "selftest": True}}, "eject"),
    ({}, {"mach_o_audit": {"problems": 2}}, "audit"),
    ({}, {"tests": {"failed": 3, "ran": True}}, "3 automated tests failed"),
    ({}, {"tests": {"passed": 0, "failed": 0, "ran": False}}, "was not run"),
    ({"provider": "somebody's laptop"}, {}, "GitHub Actions runner"),
    ({}, {"study_quality_included": False}, "Study Quality"),
])
def test_ci_readiness_needs_every_check(report_change, manifest_change, expected):
    verdict, reasons, _ = intel.ci_readiness({**ci_report(), **report_change}, "a" * 64, ci_manifest(**manifest_change))
    assert verdict == "NEEDS WORK" and expected in " ".join(reasons), reasons


def test_a_higher_real_minimum_system_is_stated_not_hidden():
    _, _, caveats = intel.ci_readiness(ci_report(), "a" * 64, ci_manifest(min_macos_effective="14.0"))
    assert any("14.0" in c for c in caveats)


# ------------------------------------------------------------------------------------------------------------ the Windows fetch of the CI DMG
def test_joining_the_parts_gives_the_exact_file_and_a_damaged_part_stops_it(tmp_path):
    import hashlib

    import fetch_ci_dmg as f

    payload = bytes(range(256)) * 4000
    pieces = [payload[:400_000], payload[400_000:800_000], payload[800_000:]]
    parts = []
    for i, piece in enumerate(pieces, 1):
        name = intel.split_name("x.dmg", i)
        (tmp_path / name).write_bytes(piece)
        parts.append({"name": name, "bytes": len(piece), "sha256": hashlib.sha256(piece).hexdigest()})
    out = tmp_path / "x.dmg"
    assert f.join_parts(tmp_path, parts, out) == hashlib.sha256(payload).hexdigest() and out.read_bytes() == payload
    (tmp_path / parts[1]["name"]).write_bytes(pieces[1][:-1] + b"!")
    with pytest.raises(SystemExit) as err:
        f.join_parts(tmp_path, parts, tmp_path / "y.dmg")
    assert "damaged" in str(err.value)


def test_the_test_voice_is_a_clear_one_never_a_novelty_voice():
    import validate_macos_intel as v

    listing = ["Albert              en_US    # Hello! My name is Albert.", "Bad News            en_US    # Hello", "Alex                en_US    # Most people recognize me",
               "Samantha            en_US    # Hello! My name is Samantha.", "Yelda               tr_TR    # Merhaba"]
    assert v.pick_voice(listing) == "Samantha"
    assert v.pick_voice([listing[0], listing[1], "Kathy               en_US    # x", "Tessa               en_ZA    # Hello"]) == "Tessa"
    assert v.pick_voice([listing[0], listing[1]]) == ""                                 # only novelty voices: the system default is used


def test_a_reset_connection_continues_where_it_stopped(tmp_path):
    """The Windows link resets long downloads: the fetch resumes with Range requests and ends with the exact bytes."""
    import hashlib
    import http.server
    import threading

    import fetch_ci_dmg as f

    payload = bytes(range(251)) * 9000
    state = {"drops": 0}

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            start = int(self.headers.get("Range", "bytes=0-").split("=")[1].split("-")[0])
            body = payload[start:]
            self.send_response(206 if start else 200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if state["drops"] < 2:                                   # the first two answers are cut off half way
                state["drops"] += 1
                self.wfile.write(body[: len(body) // 3])
                self.wfile.flush()
                self.connection.close()
                return
            self.wfile.write(body)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        dest = tmp_path / "asset.bin"
        dest.write_bytes(payload[:1000])                             # a partial file from an earlier try
        notes = []
        f.download_resumable(lambda: f"http://127.0.0.1:{server.server_port}/x", dest, len(payload), pause=lambda s: None, log=notes.append)
        assert hashlib.sha256(dest.read_bytes()).hexdigest() == hashlib.sha256(payload).hexdigest()
        assert state["drops"] == 2                                    # two cut-off answers, resumed twice
    finally:
        server.shutdown()


def test_medical_intelligence_checks_of_the_packaged_validator(tmp_path):
    c = ci()
    ok = {"Medical knowledge": {"ok": True, "detail": "66,668 concepts, 263,054 terms, 11 health checks"}}
    assert c.intelligence_resources_check(ok)[0] is True
    assert c.intelligence_resources_check({})[0] is False                                                       # no such check
    assert c.intelligence_resources_check({"Medical knowledge": {"ok": True, "detail": "skipped: no medical knowledge database is installed"}})[0] is False
    assert c.intelligence_resources_check({"Medical knowledge": {"ok": False, "detail": "66,668 concepts, schema mismatch"}})[0] is False

    import json

    def folder(mode="on", status="ok", raw_modified=False, v4_status="ok"):
        (tmp_path / "correction_report.json").write_text(json.dumps({"medical_intelligence_v4": {"mode": mode, "status": status, "applied": [1], "suggested": []},
                                                                    "raw_transcript_modified": raw_modified}), encoding="utf-8")
        (tmp_path / "medical_quality_report.json").write_text(json.dumps({"v4_status": v4_status}), encoding="utf-8")
        return tmp_path

    assert c.intelligence_run_check(folder())[0] is True
    assert c.intelligence_run_check(folder(mode="off"))[0] is False                                              # the switch did not reach the packaged app
    assert c.intelligence_run_check(folder(status="unavailable: no medical knowledge database is installed"))[0] is False
    assert c.intelligence_run_check(folder(raw_modified=True))[0] is False
    assert c.intelligence_run_check(folder(v4_status="off"))[0] is False


def test_the_macos_workflow_is_valid_yaml_and_the_license_audit_step_never_fails_the_build():
    import yaml

    doc = yaml.safe_load((Path(__file__).resolve().parents[1] / ".github" / "workflows" / "build-macos-intel.yml").read_text(encoding="utf-8"))
    steps = doc["jobs"]["intel"]["steps"]
    audit = next(s for s in steps if str(s.get("name", "")).startswith("License audit of the built app"))
    assert audit.get("continue-on-error") is True and "license_audit.py" in audit["run"]
    names = [s.get("name") for s in steps]
    assert names.index(audit["name"]) < next(i for i, n in enumerate(names) if str(n).startswith("Test the packaged app"))
