"""The macOS build is prepared on Windows: names, Info.plist, entitlements, icon, DMG / signing / notarization commands, the plan, the release
folder and the benchmark helpers are all tested here without a Mac. What a Mac must still prove is listed in BUILDING.md."""

import importlib.util
import io
import json
import plistlib
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from build_tools import icns, macos as mac, release
from build_tools.layout import dist_parent, release_dir

ROOT = Path(__file__).resolve().parent.parent


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name.replace("__", "/") + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


build_macos_app = load_script("scripts__build_macos_app")
benchmark = load_script("scripts__benchmark_asr_backend")
build_py = load_script("build")


# ------------------------------------------------------------------------------------------------------------------ names + plist
def test_dmg_names_follow_the_spec():
    assert mac.dmg_name("arm64") == "MELETRA-macOS-AppleSilicon.dmg"
    assert mac.dmg_name("x86_64") == "MELETRA-macOS-Intel.dmg"
    assert mac.APP_BUNDLE == "MELETRA.app"
    assert mac.dmg_name("AMD64") == mac.DMG_INTEL and mac.dmg_name("aarch64") == mac.DMG_ARM and mac.normalize_arch("Intel") == "x86_64"
    with pytest.raises(ValueError):
        mac.dmg_name("ppc")


def test_info_plist_describes_a_modern_retina_app_that_opens_lectures():
    info = mac.info_plist("0.2.0-dev", "arm64")
    assert info["CFBundleIdentifier"] == "com.medicallecturetranscriber.app" and info["CFBundleExecutable"] == "MELETRA"
    assert info["CFBundleVersion"] == "0.2.0" and info["CFBundleShortVersionString"] == "0.2.0-dev"    # a numeric build version, the full text for people
    assert info["NSHighResolutionCapable"] is True and info["LSMinimumSystemVersion"] == "12.0" and info["CFBundleIconFile"] == "app.icns"
    types = [t for d in info["CFBundleDocumentTypes"] for t in d["LSItemContentTypes"]]
    assert {"public.audio", "public.movie", "com.adobe.pdf", "public.image", "public.plain-text"} <= set(types)
    assert info["LSArchitecturePriority"] == ["arm64"] and mac.info_plist("0.2.0", "x86_64")["LSArchitecturePriority"] == ["x86_64"]
    assert plistlib.loads(mac.info_plist_bytes("0.2.0")) == mac.info_plist("0.2.0")


def test_entitlements_are_the_minimum_the_stack_needs():
    ent = plistlib.loads(mac.entitlements_bytes())
    assert ent["com.apple.security.cs.allow-jit"] is True and ent["com.apple.security.cs.disable-library-validation"] is True
    assert not any("network.server" in k or "camera" in k or "audio-input" in k for k in ent)


# ------------------------------------------------------------------------------------------------------------------ icon
def test_icns_container_is_valid():
    fake = {size: icns.PNG_SIGNATURE + b"x" * size for size in (16, 128, 1024)}
    data = icns.icns_bytes(fake)
    assert data[:4] == b"icns" and int.from_bytes(data[4:8], "big") == len(data)
    assert icns.read_icns_sizes(data) == [16, 128, 1024]
    with pytest.raises(ValueError, match="not an icns size"):
        icns.icns_bytes({20: icns.PNG_SIGNATURE})
    with pytest.raises(ValueError, match="not a PNG"):
        icns.icns_bytes({16: b"nope"})
    with pytest.raises(ValueError):
        icns.read_icns_sizes(b"junk")


def test_the_app_icon_is_drawn_into_an_icns(qapp, tmp_path):
    path = icns.write_icns(tmp_path / "app.icns")
    data = path.read_bytes()
    assert icns.read_icns_sizes(data) == list(icns.ICNS_SIZES)
    assert data.count(icns.PNG_SIGNATURE) == len(icns.ICNS_SIZES)


# ------------------------------------------------------------------------------------------------------------------ commands
def test_codesign_commands_use_the_hardened_runtime_and_verify(tmp_path):
    commands = mac.codesign_commands(tmp_path / "A.app", "Developer ID Application: X (T)", tmp_path / "e.plist")
    sign, verify, assess = commands
    assert sign[:2] == ["codesign", "--force"] and "--options" in sign and "runtime" in sign and "--timestamp" in sign
    assert sign[sign.index("--sign") + 1] == "Developer ID Application: X (T)" and str(tmp_path / "e.plist") in sign
    assert verify[:2] == ["codesign", "--verify"] and assess[0] == "spctl"


def test_dmg_commands_make_a_drag_to_applications_image(tmp_path):
    app, dmg = tmp_path / "MELETRA.app", tmp_path / "out" / "x.dmg"
    commands = mac.dmg_commands(app, dmg, staging=tmp_path / "stage")
    flat = [" ".join(c) for c in commands]
    assert any(c.startswith("ln -s /Applications") for c in flat)
    create = next(c for c in commands if c[0] == "hdiutil")
    assert create[:2] == ["hdiutil", "create"] and "UDZO" in create and create[-1] == str(dmg) and "MELETRA" in create
    assert commands[0][:2] == ["rm", "-rf"] and commands[-1][:2] == ["rm", "-rf"]


def test_notarization_submits_waits_and_staples(tmp_path):
    submit, staple, validate = mac.notarize_commands(tmp_path / "x.dmg", "notary")
    assert submit[:3] == ["xcrun", "notarytool", "submit"] and "--wait" in submit and submit[submit.index("--keychain-profile") + 1] == "notary"
    assert staple[:3] == ["xcrun", "stapler", "staple"] and validate[2] == "validate"


def test_the_signing_report_never_claims_more_than_exists():
    none = mac.signing_state(None, None)
    assert (none["signed"], none["notarized"]) == ("NO", "NO") and "UNSIGNED" in none["note"] and "right-click" in none["note"]
    assert (mac.signing_state("ID", None)["signed"], mac.signing_state("ID", None)["notarized"]) == ("YES", "NO")
    assert mac.signing_state("ID", "profile")["notarized"] == "YES"
    assert mac.signing_state(None, "profile") == none                      # a notary profile without a certificate means nothing


# ------------------------------------------------------------------------------------------------------------------ the plan
def test_unsigned_plan_builds_app_and_dmg_in_the_v020_folders(tmp_path):
    steps = build_macos_app.build_plan("arm64", "0.2.0-dev", tmp_path, python="python3")
    names = [s.name for s in steps]
    assert names[:2] == ["assets", "pyinstaller"] and not any(n.startswith(("codesign", "notarize")) for n in names)
    pyinstaller = steps[1]
    assert pyinstaller.env == {"MLT_MAC_ARCH": "arm64"} and pyinstaller.argv[:3] == ["python3", "-m", "PyInstaller"]
    assert str(dist_parent("0.2.0-dev", tmp_path) / "macos") in pyinstaller.argv
    dmg = [s for s in steps if s.argv[0] == "hdiutil"][0].argv[-1]
    assert Path(dmg) == release_dir("0.2.0-dev", "macOS", tmp_path) / "MELETRA-macOS-AppleSilicon.dmg"
    everything = " ".join(" ".join(s.argv) for s in steps)
    assert str(tmp_path / "release" / "MELETRA") not in everything       # never next to the 0.1.1 files
    assert "dist" + chr(92) + "MELETRA" not in everything


def test_signed_and_notarized_plan_adds_the_steps_in_order(tmp_path):
    steps = build_macos_app.build_plan("x86_64", "0.2.0", tmp_path, identity="ID", notary_profile="np", python="py")
    names = [s.name for s in steps]
    assert names.index("codesign 1") < names.index("dmg 1") < names.index("codesign dmg") < names.index("notarize 1")
    assert any(s.argv[0] == "hdiutil" and s.argv[-1].endswith("Intel.dmg") for s in steps)
    only_signed = [s.name for s in build_macos_app.build_plan("arm64", "0.2.0", tmp_path, identity="ID")]
    assert "codesign dmg" in only_signed and not any(n.startswith("notarize") for n in only_signed)


def test_the_build_refuses_to_run_off_a_mac_but_can_show_its_plan(capsys, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    assert build_macos_app.main(["--arch", "arm64"]) == 2
    assert "must run on a Mac" in capsys.readouterr().err
    assert build_macos_app.main(["--arch", "arm64", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "pyinstaller:" in out and "hdiutil" in out and '"signed": "NO"' in out


def test_the_report_and_checksums_are_written_next_to_the_dmg(tmp_path):
    dmg = release_dir("0.2.0", "macOS", tmp_path) / mac.DMG_ARM
    dmg.parent.mkdir(parents=True)
    dmg.write_bytes(b"dmg bytes")
    report = json.loads(build_macos_app.write_report("arm64", "0.2.0", None, None, tmp_path).read_text(encoding="utf-8"))
    assert report["signed"] == "NO" and report["notarized"] == "NO" and report["dmg"] == str(dmg)
    assert report["dmg_sha256"] in (dmg.parent / "SHA256SUMS.txt").read_text(encoding="utf-8")


def test_the_pyinstaller_spec_makes_a_bundle_and_bundles_mlx_only_on_apple_silicon():
    spec = (ROOT / "build_tools" / "MELETRA-macos.spec").read_text(encoding="utf-8")
    assert "BUNDLE(" in spec and "info_plist=mac_build.info_plist" in spec and "app.icns" in spec and "entitlements_file" in spec
    assert 'if ARCH == "arm64"' in spec and spec.index('if ARCH == "arm64"') < spec.index("mlx_whisper")
    assert '"torch"' in spec.split("excludes")[1]                         # PyTorch stays out of the app
    compile(spec.replace("from PyInstaller", "# from PyInstaller"), "spec", "exec")   # syntax only


def test_build_script_and_requirements():
    script = (ROOT / "build_macos.sh").read_text(encoding="utf-8")
    assert script.startswith("#!/usr/bin/env bash") and "uname -s" in script and "requirements-macos.txt" in script
    assert "scripts/build_macos_app.py" in script and "pytest" in script
    bash = shutil.which("bash")
    if bash:
        assert subprocess.run([bash, "-n", str(ROOT / "build_macos.sh")], capture_output=True).returncode == 0
    reqs = (ROOT / "requirements-macos.txt").read_text(encoding="utf-8")
    assert 'mlx-whisper' in reqs and 'platform_machine == "arm64"' in reqs and "keyring" in reqs
    assert not any(line.strip().startswith(("torch", "nvidia")) for line in reqs.splitlines())


# ------------------------------------------------------------------------------------------------------------------ release folder
def make_release(tmp_path, windows=True, mac_files=False):
    win = release_dir("0.2.0", "Windows", tmp_path)
    win.mkdir(parents=True)
    if windows:
        (win / release.WINDOWS_INSTALLER).write_bytes(b"installer")
        (win / release.WINDOWS_PORTABLE).write_bytes(b"portable")
    if mac_files:
        macdir = release_dir("0.2.0", "macOS", tmp_path)
        (macdir / mac.APP_BUNDLE / "Contents").mkdir(parents=True)
        (macdir / mac.APP_BUNDLE / "Contents" / "Info.plist").write_bytes(b"plist")
        (macdir / mac.DMG_ARM).write_bytes(b"dmg")


def test_readme_says_what_is_built_and_what_is_not(tmp_path):
    make_release(tmp_path)
    text = release.readme_text("0.2.0", tmp_path)
    assert "[built] Windows installer: MELETRA-Setup-Windows-x64.exe" in text
    assert "[not built] macOS DMG (Apple Silicon)" in text and "NOT included in this release" in text and "./build_macos.sh" in text
    make_release(tmp_path / "m", mac_files=True)
    with_mac = release.readme_text("0.2.0", tmp_path / "m", mac.signing_state(None, None))
    assert "[built] macOS DMG (Apple Silicon)" in with_mac and "Signed: NO   Notarized: NO" in with_mac and "NOT included" not in with_mac


def test_all_platforms_zip_holds_exactly_what_exists(tmp_path):
    make_release(tmp_path, mac_files=False)
    archive = release.all_platforms_zip("0.2.0", tmp_path)
    assert archive == tmp_path / "release" / "v0.2.0" / "MELETRA-v0.2.0-AllPlatforms.zip"
    names = set(zipfile.ZipFile(archive).namelist())
    assert names == {"README.txt", "macOS/README.txt", "Windows/MELETRA-Setup-Windows-x64.exe",
                     "Windows/MELETRA-Portable-Windows-x64.zip"}
    note = (release_dir("0.2.0", "macOS", tmp_path) / "README.txt").read_text(encoding="utf-8")
    assert "NOT BUILT" in note and "./build_macos.sh" in note and "UNSIGNED" in note
    make_release(tmp_path / "full", mac_files=True)
    names = set(zipfile.ZipFile(release.all_platforms_zip("0.2.0", tmp_path / "full")).namelist())
    assert "macOS/README.txt" not in names                                   # a real macOS build gets no "not built" note
    assert "macOS/MELETRA-macOS-AppleSilicon.dmg" in names
    assert "macOS/MELETRA.app/Contents/Info.plist" in names


def test_building_the_package_does_not_touch_the_011_release_files(tmp_path):
    old = tmp_path / "release"
    old.mkdir()
    (old / "MELETRA-Setup-x64.exe").write_bytes(b"production 0.1.1")
    make_release(tmp_path)
    release.all_platforms_zip("0.2.0", tmp_path)
    assert (old / "MELETRA-Setup-x64.exe").read_bytes() == b"production 0.1.1"


# ------------------------------------------------------------------------------------------------------------------ build.py
def test_build_plan_per_system():
    win = build_py.plan("win32", python="py")
    assert win[0][:3] == ["py", "-m", "pytest"] and ["py", "scripts/build_installer.py"] in win and ["py", "scripts/build_portable.py"] in win
    assert [s for s in win if "build_fulloffline" in " ".join(s)] == []
    assert ["py", "scripts/build_fulloffline.py"] in build_py.plan("win32", full_offline=True, python="py")
    assert "pytest" not in " ".join(" ".join(s) for s in build_py.plan("win32", skip_tests=True, python="py"))
    assert build_py.plan("darwin") == [["./build_macos.sh"]]
    with pytest.raises(SystemExit):
        build_py.plan("linux")


def test_build_py_dry_run_changes_nothing(capsys, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    assert build_py.main(["--dry-run", "--all"]) == 0
    out = capsys.readouterr().out
    assert "scripts/build_app.py" in out and "AllPlatforms" in out and "dry run" in out


# ------------------------------------------------------------------------------------------------------------------ benchmark helpers
def test_word_error_rate_and_normalisation():
    assert benchmark.normalize("Hello, World! It's 5 mg.") == ["hello", "world", "it's", "5", "mg"]
    assert benchmark.word_error_rate("the cat sat", "the cat sat") == 0.0
    assert benchmark.word_error_rate("the cat sat", "the dog sat") == pytest.approx(1 / 3)
    assert benchmark.word_error_rate("a b c d", "a b") == 0.5 and benchmark.word_error_rate("a b", "a x b") == 0.5
    assert benchmark.word_error_rate("", "") == 0.0 and benchmark.word_error_rate("", "x") == 1.0


def test_timestamp_check_and_table():
    from types import SimpleNamespace as S

    assert benchmark.timestamps_ok([S(start=0, end=2), S(start=2, end=5)], 6.0)
    assert not benchmark.timestamps_ok([S(start=3, end=4), S(start=1, end=2)], 6.0)
    assert not benchmark.timestamps_ok([S(start=0, end=99)], 6.0)
    ok = benchmark.Result("mlx_whisper", rtf=0.08, load_seconds=2.5, decode_seconds=24.0, audio_seconds=300, segments=40, words=700, wer=0.052)
    skipped = benchmark.Result("faster_whisper", status="skipped", note="model folder not given or not found")
    table = benchmark.render_markdown([ok, skipped], Path("lecture.wav"), 300)
    assert "| mlx_whisper | ok | 0.080 |" in table and "5.2 %" in table and "| faster_whisper | skipped |" in table


def test_benchmark_skips_backends_that_are_not_available(tmp_path, capsys):
    from tests.conftest import write_silent_wav

    wav = write_silent_wav(tmp_path / "a.wav", seconds=3)
    code = benchmark.main(["--audio", str(wav), "--seconds", "0", "--out", str(tmp_path / "r.md")])
    out = capsys.readouterr().out
    assert code == 1 and out.count("skipped") == 3 and (tmp_path / "r.md").is_file()
    data = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert [d["backend"] for d in data] == ["faster_whisper", "mlx_whisper", "whispercpp"]
