"""The macOS LGPL media swap: the parts that can be checked without a Mac (names, otool parsing, the swap plan, pruning, the audit rules, the build script, the workflow, the notices).

The Mach-O editing itself (otool / install_name_tool on a real .app) has not run yet: GitHub Actions was blocked by billing when this was written.
"""

import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from build_tools import lgpl_media_macos as L  # noqa: E402
from build_tools import macos_apple_silicon as apple  # noqa: E402

MAGIC = b"\xcf\xfa\xed\xfe"


def test_library_names_are_recognised_with_and_without_the_full_version():
    assert L.family_major("libavcodec.62.28.102.dylib") == ("libavcodec", 62)
    assert L.family_major("libavcodec.62.dylib") == ("libavcodec", 62)
    assert L.family_major("libswresample.5.1.100.dylib") == ("libswresample", 5)
    assert L.family_major("libx264.165.dylib") is None and L.family_major("libavcodec.dylib") is None and L.family_major("av/frame.abi3.so") is None


def test_otool_output_is_parsed_without_the_file_itself_or_its_own_id():
    text = ("/app/x/libavformat.62.12.102.dylib:\n"
            "\t@loader_path/libavformat.62.12.102.dylib (compatibility version 62.0.0, current version 62.12.102)\n"
            "\t@loader_path/libavcodec.62.28.102.dylib (compatibility version 62.0.0, current version 62.28.102)\n"
            "\t/usr/lib/libz.1.dylib (compatibility version 1.0.0, current version 1.2.12)\n")
    assert L.parse_otool_deps(text, "libavformat.62.12.102.dylib") == ["@loader_path/libavcodec.62.28.102.dylib", "/usr/lib/libz.1.dylib"]
    assert L.is_system("/usr/lib/libz.1.dylib") and L.is_system("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation") and not L.is_system("/opt/homebrew/lib/libx.dylib")


def test_the_swap_plan_matches_by_family_and_major_and_names_what_is_missing(tmp_path):
    new8, new7 = tmp_path / "ffmpeg-8" / "lib", tmp_path / "ffmpeg-7" / "lib"
    for folder, names in ((new8, ["libavcodec.62.dylib", "libavutil.60.dylib"]), (new7, ["libavcodec.61.dylib"])):
        folder.mkdir(parents=True)
        for n in names:
            (folder / n).write_bytes(MAGIC)
    app = tmp_path / "bundled"
    app.mkdir()
    bundled = [app / "libavcodec.62.28.102.dylib", app / "libavcodec.61.3.100.dylib", app / "libswscale.9.5.102.dylib"]
    for b in bundled:
        b.write_bytes(MAGIC)
    swaps, missing = L.plan_swaps(bundled, [new8, new7])
    assert {t.name: s.name for t, s in swaps} == {"libavcodec.62.28.102.dylib": "libavcodec.62.dylib", "libavcodec.61.3.100.dylib": "libavcodec.61.dylib"}
    assert missing == ["libswscale.9.5.102.dylib"]


def test_a_copied_library_is_pointed_at_its_siblings_by_the_bundled_file_names():
    names = {("libavutil", 60): "libavutil.60.26.102.dylib", ("libswresample", 6): "libswresample.6.3.102.dylib"}
    deps = ["/ffbuild/prefix/lib/libavutil.60.dylib", "/usr/lib/libz.1.dylib", "/System/Library/Frameworks/Security.framework/Security", "@rpath/libswresample.6.dylib"]
    commands, problems = L.retarget_commands(Path("/app/libavcodec.62.28.102.dylib"), deps, names)
    assert commands[0] == ["install_name_tool", "-change", "/ffbuild/prefix/lib/libavutil.60.dylib", "@loader_path/libavutil.60.26.102.dylib", str(Path("/app/libavcodec.62.28.102.dylib"))]
    assert commands[1][3] == "@loader_path/libswresample.6.3.102.dylib" and problems == []
    _, bad = L.retarget_commands(Path("/app/libavcodec.62.28.102.dylib"), ["/opt/homebrew/opt/x264/lib/libx264.165.dylib", "/usr/lib/libz.1.dylib"], names)
    assert bad and "libx264.165.dylib" in bad[0]


def test_only_libraries_nothing_refers_to_any_more_are_pruned():
    graph = {"av_core.abi3.so": {"libavcodec.62.28.102.dylib"}, "libavcodec.62.28.102.dylib": {"libavutil.60.26.102.dylib"}, "libavutil.60.26.102.dylib": set(),
             "libx264.165.dylib": set(), "libx265.216.dylib": set(), "libdav1d.7.dylib": set(), "libpng16.16.dylib": set(), "cv2.abi3.so": {"libpng16.16.dylib"}}
    candidates = ["libavcodec.62.28.102.dylib", "libavutil.60.26.102.dylib", "libx264.165.dylib", "libx265.216.dylib", "libdav1d.7.dylib", "libpng16.16.dylib"]
    assert L.unreferenced(graph, ["av_core.abi3.so", "cv2.abi3.so"], candidates) == ["libdav1d.7.dylib", "libx264.165.dylib", "libx265.216.dylib"]
    # the old libavcodec referred to the codec libraries; after the swap it refers to none: they become unreferenced
    graph["libavcodec.62.28.102.dylib"] = {"libavutil.60.26.102.dylib", "libx264.165.dylib"}
    assert "libx264.165.dylib" not in L.unreferenced(graph, ["av_core.abi3.so"], candidates)


def test_the_audit_refuses_x264_x265_and_gpl_class_ffmpeg_options(tmp_path):
    app = tmp_path / "App.app"
    (app / "Contents" / "Frameworks").mkdir(parents=True)
    clean = app / "Contents" / "Frameworks" / "libavcodec.62.dylib"
    clean.write_bytes(MAGIC + b"configuration: --disable-gpl --disable-nonfree --disable-autodetect x264 - core %d")      # the H.264 parser knows the x264 banner: not a finding
    gpl = app / "Contents" / "Frameworks" / "libavformat.62.dylib"
    gpl.write_bytes(MAGIC + b"configuration: --enable-gpl --enable-libx264")
    x264 = app / "Contents" / "Frameworks" / "libx264.165.dylib"
    x264.write_bytes(MAGIC + b"x264")
    text = app / "Contents" / "Resources" / "notes.txt"
    text.parent.mkdir(parents=True)
    text.write_text("--enable-gpl is only text here", encoding="utf-8")
    assert L.gpl_class_files(app) == ["Contents/Frameworks/libavformat.62.dylib".replace("/", L.os.sep), "Contents/Frameworks/libx264.165.dylib".replace("/", L.os.sep)]
    gpl.unlink()
    x264.unlink()
    assert L.gpl_class_files(app) == [] and L.main(["audit", str(app)]) == 0


def test_the_pinned_ffmpeg_sources_are_stable_tarballs_with_a_sha256_and_the_same_majors_as_the_wheels():
    assert set(L.SOURCES) == {"62", "61"}
    for major, src in L.SOURCES.items():
        assert re.fullmatch(r"[0-9a-f]{64}", src["sha256"]) and src["url"] == f"https://ffmpeg.org/releases/ffmpeg-{src['version']}.tar.xz"
        assert src["version"].startswith("8.1." if major == "62" else "7.1.")
    assert "libavcodec 62" in L.SOURCES["62"]["for"] and "libavcodec 61" in L.SOURCES["61"]["for"]


def test_the_build_script_configures_an_lgpl_only_ffmpeg_and_checks_the_result():
    script = (ROOT / "scripts" / "build_macos_ffmpeg_lgpl.sh").read_text(encoding="utf-8")
    configure = script[script.index("./configure"):script.index("tee ")]
    for flag in ("--disable-gpl", "--disable-nonfree", "--disable-autodetect", "--enable-shared", "--disable-programs", "-mmacosx-version-min"):
        assert flag in configure
    for forbidden in ("--enable-gpl", "--enable-nonfree", "--enable-libx264", "--enable-libx265", "--enable-version3", "--enable-libfdk"):
        assert forbidden not in configure
    assert "shasum -a 256 -c" in script and "License: LGPL version 2.1 or later" in script and "set -euo pipefail" in script and "x26[45]" in script
    assert "\r" not in script


def test_the_public_arm_workflow_is_valid_and_records_a_license_inventory():
    wf = yaml.safe_load((ROOT / ".github" / "workflows" / "build-macos-arm64.yml").read_text(encoding="utf-8"))
    inputs = wf[True]["workflow_dispatch"]["inputs"]
    assert inputs["mode"]["default"] == "build" and "build-fast" in inputs["mode"]["options"]
    steps = {s.get("name", ""): s for s in wf["jobs"]["arm64"]["steps"]}
    informational = next(s for n, s in steps.items() if n.startswith("License inventory of the built app"))
    assert informational.get("continue-on-error") is True and "license_audit.py" in informational["run"]


def test_the_notices_say_lgpl_only_for_a_swapped_app_and_keep_the_gpl_warning_for_the_stock_wheels():
    stock = apple.third_party_licenses("", {"av": "18.1.0"})
    assert "GPL - READ THIS" in stock and "LEGAL REVIEW REQUIRED" in stock and "NO libx264" not in stock
    swapped = apple.third_party_licenses("", {"av": "18.1.0", "opencv-python-headless": "4.10.0.84"}, L.SOURCES)
    assert "LGPL-only FFmpeg libraries" in swapped and "NO libx264, NO libx265" in swapped and "GPL - READ THIS" not in swapped and "LEGAL REVIEW REQUIRED" not in swapped
    assert L.SOURCES["62"]["sha256"] in swapped and "ffmpeg-7.1.5.tar.xz" in swapped


def _ci_report(sha="a" * 64):
    return {"provider": "GitHub Actions", "machine": "arm64", "dmg_sha256": sha, "checks": {k: "PASS" for k in apple.CI_KEYS}}


def _ci_manifest(**over):
    return {"dmg_sha256": "a" * 64, "dmg_check": {"mount": True, "copy": True, "eject": True, "selftest": True}, "mach_o_audit": {"problems": 0},
            "tests": {"passed": 1, "failed": 0, "skipped": 0, "ran": True}, "study_quality_included": True, "developer_id": False, "notarized": False,
            "gpl_source_offer": "MISSING", "min_macos_effective": "14.0", **over}


def test_an_lgpl_swapped_build_needs_no_gpl_source_offer_and_the_media_check_is_required():
    assert "media_formats" in apple.CI_KEYS
    verdict, reasons, caveats = apple.ci_readiness(_ci_report(), "a" * 64, _ci_manifest(gpl_source_offer="NOT NEEDED", media_libraries={"lgpl_only": True}))
    assert verdict == "CI-TESTED APPLE SILICON DMG READY" and reasons == [] and "LGPL-only" in " ".join(caveats) and "source-offer" not in " ".join(caveats)
    report = _ci_report()
    report["checks"]["media_formats"] = "FAIL"
    assert apple.ci_readiness(report, "a" * 64, _ci_manifest())[0] == "NEEDS WORK"
