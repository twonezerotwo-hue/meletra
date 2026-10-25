"""Packaging helpers: icon, version resource, CUDA/FFmpeg assembly, dist check, zip packages, installer script, build scripts."""

import json
import os
import struct
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pytest

from build_tools import assemble, packages
from build_tools.cleanenv import windows_only_env
from build_tools.icon import ICON_SIZES, ico_bytes, read_ico_sizes, render_pngs, write_icon
from build_tools.versioninfo import version_file_text, version_tuple, write_version_file

ROOT = Path(__file__).resolve().parent.parent


# ----------------------------------------------------------------------------------------------- icon
def test_ico_roundtrip_lists_every_size():
    data = ico_bytes({16: b"a" * 10, 32: b"b" * 20, 256: b"c" * 30})
    assert read_ico_sizes(data) == [16, 32, 256]
    count, = struct.unpack_from("<H", data, 4)
    assert count == 3
    offset, = struct.unpack_from("<I", data, 6 + 12)                       # first entry's image offset
    assert data[offset:offset + 10] == b"a" * 10


def test_ico_rejects_other_files():
    with pytest.raises(ValueError):
        read_ico_sizes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)


def test_rendered_icon_is_a_real_png_in_every_size(qapp):
    pngs = render_pngs()
    assert sorted(pngs) == sorted(ICON_SIZES)
    for size, data in pngs.items():
        assert data[:8] == b"\x89PNG\r\n\x1a\n"
        width, height = struct.unpack(">II", data[16:24])
        assert (width, height) == (size, size)


def test_write_icon_creates_ico_and_png(qapp, tmp_path):
    ico = write_icon(tmp_path / "assets" / "app.ico")
    assert read_ico_sizes(ico.read_bytes()) == sorted(ICON_SIZES) and ico.with_suffix(".png").stat().st_size > 500


# ----------------------------------------------------------------------------------------------- version resource
def test_version_tuple():
    assert version_tuple("0.1.0") == (0, 1, 0, 0) and version_tuple("1.2.3.4") == (1, 2, 3, 4) and version_tuple("2") == (2, 0, 0, 0)


def test_version_file_is_valid_pyinstaller_syntax_and_carries_the_metadata():
    pytest.importorskip("pefile")                                   # a Windows build-tool dependency
    text = version_file_text("0.1.0", "Touche Holding", "Medical Lecture Transcriber", "desc", "MELETRA.exe")
    namespace: dict = {}
    from PyInstaller.utils.win32 import versioninfo

    exec("from PyInstaller.utils.win32.versioninfo import *\ninfo = " + text.split("VSVersionInfo(", 1)[0] and
         "from PyInstaller.utils.win32.versioninfo import *\ninfo = VSVersionInfo(" + text.split("VSVersionInfo(", 1)[1], namespace)
    assert isinstance(namespace["info"], versioninfo.VSVersionInfo)
    for needle in ("Touche Holding", "Medical Lecture Transcriber", "0.1.0.0", "MELETRA.exe", "(0, 1, 0, 0)"):
        assert needle in text


def test_write_version_file(tmp_path):
    path = write_version_file(tmp_path / "x" / "v.txt", "0.1.0", "Co", "Prod", "Desc", "p.exe")
    assert path.read_text(encoding="utf-8").startswith("# UTF-8")


# ----------------------------------------------------------------------------------------------- CUDA
def make_cuda_dir(folder: Path, names=None) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for name in names or (*assemble.CUDA_DLL_PATTERNS, "cudnn_ops64_9.dll", "torch_cuda.dll", "cufft64_11.dll", "c10.dll", "nvrtc64_120_0.dll"):
        (folder / name).write_bytes(b"x" * 64)
    return folder


def test_only_the_dlls_ctranslate2_needs_are_selected(tmp_path):
    chosen = assemble.select_cuda_dlls(make_cuda_dir(tmp_path / "lib"))
    names = {p.name for p in chosen}
    assert names == set(assemble.CUDA_DLL_PATTERNS)
    assert "torch_cuda.dll" not in names and "cufft64_11.dll" not in names           # PyTorch-only libraries stay out
    assert not any(n.startswith("cudnn_") for n in names)                            # cuDNN sub-libraries are never loaded


def test_missing_required_dlls_are_named(tmp_path):
    folder = make_cuda_dir(tmp_path / "lib", ["cublas64_12.dll", "cudart64_12.dll"])
    assert assemble.missing_cuda_dlls(assemble.select_cuda_dlls(folder)) == ["cublasLt64_12.dll", "cudnn64_9.dll"]


def test_copy_cuda_puts_the_dlls_in_libs_cuda(tmp_path):
    copied = assemble.copy_cuda(tmp_path / "dist", make_cuda_dir(tmp_path / "lib"))
    assert {p.parent for p in copied} == {tmp_path / "dist" / "libs" / "cuda"} and len(copied) == len(assemble.CUDA_DLL_PATTERNS)


def test_copy_cuda_removes_libraries_an_older_build_left_behind(tmp_path):
    target = tmp_path / "dist" / "libs" / "cuda"
    target.mkdir(parents=True)
    (target / "cudnn_adv64_9.dll").write_bytes(b"old")
    assemble.copy_cuda(tmp_path / "dist", make_cuda_dir(tmp_path / "lib", [*assemble.CUDA_DLL_PATTERNS, "cudnn_adv64_9.dll"]))
    assert not (target / "cudnn_adv64_9.dll").exists() and (target / "cublas64_12.dll").exists()


def test_the_shipped_runtime_is_the_measured_minimum():
    assert set(assemble.CUDA_DLL_PATTERNS) == {"cublas64_12.dll", "cublasLt64_12.dll", "cudart64_12.dll", "cudnn64_9.dll"}


def test_copy_cuda_refuses_an_incomplete_runtime(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing"):
        assemble.copy_cuda(tmp_path / "dist", make_cuda_dir(tmp_path / "lib", ["cublas64_12.dll"]))


def test_copy_cuda_without_any_runtime_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(assemble, "find_cuda_source", lambda extra=None: None)
    with pytest.raises(FileNotFoundError, match="CPU-only"):
        assemble.copy_cuda(tmp_path / "dist")


def test_find_cuda_source_prefers_the_explicit_folder(tmp_path):
    folder = make_cuda_dir(tmp_path / "mine")
    assert assemble.find_cuda_source([folder]) == folder
    assert assemble.find_cuda_source([tmp_path / "nothing"]) != tmp_path / "nothing"


# ----------------------------------------------------------------------------------------------- FFmpeg
def make_ffmpeg_zip(path: Path, with_probe=True, evil=False) -> Path:
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("ffmpeg-master-latest-win64-lgpl/bin/ffmpeg.exe", b"MZ" + b"0" * 100)
        if with_probe:
            z.writestr("ffmpeg-master-latest-win64-lgpl/bin/ffprobe.exe", b"MZ" + b"1" * 100)
        z.writestr("ffmpeg-master-latest-win64-lgpl/LICENSE.txt", "LGPL")
        z.writestr("ffmpeg-master-latest-win64-lgpl/bin/ffplay.exe", b"MZ")
        z.writestr("ffmpeg-master-latest-win64-lgpl/doc/huge.html", "x")
        if evil:
            z.writestr("../../escaped/bin/ffmpeg.exe", b"evil")
    return path


def test_extract_takes_only_the_two_tools_and_the_licence(tmp_path):
    ffmpeg, ffprobe = assemble.extract_ffmpeg_zip(make_ffmpeg_zip(tmp_path / "f.zip"), tmp_path / "x")
    assert ffmpeg.name == "ffmpeg.exe" and ffprobe.name == "ffprobe.exe"
    names = {p.name for p in (tmp_path / "x").rglob("*") if p.is_file()}
    assert names == {"ffmpeg.exe", "ffprobe.exe", "LICENSE.txt"}


def test_extract_never_writes_outside_the_target(tmp_path):
    target = tmp_path / "x"
    assemble.extract_ffmpeg_zip(make_ffmpeg_zip(tmp_path / "f.zip", evil=True), target)
    assert not (tmp_path / "escaped").exists() and not (tmp_path.parent / "escaped").exists()


def test_extract_without_ffprobe_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="ffprobe"):
        assemble.extract_ffmpeg_zip(make_ffmpeg_zip(tmp_path / "f.zip", with_probe=False), tmp_path / "x")


def test_copy_ffmpeg_into_bin(tmp_path):
    a, b = tmp_path / "ffmpeg.exe", tmp_path / "ffprobe.exe"
    a.write_bytes(b"1")
    b.write_bytes(b"2")
    out = assemble.copy_ffmpeg(tmp_path / "dist", a, b)
    assert [p.name for p in out] == ["ffmpeg.exe", "ffprobe.exe"] and (tmp_path / "dist" / "bin" / "ffprobe.exe").read_bytes() == b"2"


def test_find_ffmpeg_pair_in_a_nested_tree(tmp_path):
    (tmp_path / "a" / "b" / "bin").mkdir(parents=True)
    (tmp_path / "a" / "b" / "bin" / "ffmpeg.exe").write_bytes(b"1")
    assert assemble.find_ffmpeg_pair(tmp_path) == (tmp_path / "a" / "b" / "bin" / "ffmpeg.exe", None)


# ----------------------------------------------------------------------------------------------- notices, sizes, checks
def test_notices_mention_ffmpeg_and_cuda_only_when_bundled():
    both, neither = assemble.notices_text(True, True), assemble.notices_text(False, False)
    assert "LGPL-2.1+ build" in both and "NVIDIA CUDA runtime" in both and "OpenAI Whisper large-v3" in both
    assert "bin/ffmpeg.exe" not in neither and "NVIDIA CUDA runtime" not in neither and "PySide6" in neither


def test_human_size_and_dir_size(tmp_path):
    assert assemble.human_size(512) == "512 B" and assemble.human_size(1536) == "1.5 KB" and assemble.human_size(3 * 1024 ** 3) == "3.0 GB"
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x").write_bytes(b"1" * 100)
    (tmp_path / "y").write_bytes(b"1" * 50)
    assert assemble.dir_size(tmp_path) == 150


def test_sha256_of(tmp_path):
    (tmp_path / "f").write_bytes(b"abc")
    assert assemble.sha256_of(tmp_path / "f") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def make_dist(root: Path, cuda=True, ffmpeg=True) -> Path:
    dist = root / "dist" / "MELETRA"
    internal = dist / "_internal"
    (dist).mkdir(parents=True)
    (dist / assemble.EXE_NAME).write_bytes(b"MZ" + b"0" * 200_000)
    for rel in ("config/transcription.yaml", "config/app_models.json", "config/models.yaml", "config/common_english_words.txt",
                "medical_terms/anatomy.txt", "ctranslate2/ctranslate2.dll", "av.libs/avcodec.dll", "onnxruntime/capi/onnxruntime.dll",
                "PySide6/Qt6Widgets.dll", "faster_whisper/assets/silero_vad_v6.onnx", *[f"medical_vocab/domain{i:02d}/terms.txt" for i in range(22)]):
        path = internal / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    if cuda:
        make_cuda_dir(dist / "libs" / "cuda", assemble.CUDA_REQUIRED)
    if ffmpeg:
        (dist / "bin").mkdir()
        (dist / "bin" / "ffmpeg.exe").write_bytes(b"x")
        (dist / "bin" / "ffprobe.exe").write_bytes(b"x")
    return dist


def test_complete_dist_has_no_problems(tmp_path):
    assert assemble.verify_dist(make_dist(tmp_path), need_cuda=True, need_ffmpeg=True) == []


def test_missing_exe_is_the_only_problem_reported(tmp_path):
    assert len(assemble.verify_dist(tmp_path / "nothing")) == 1


@pytest.mark.parametrize("remove, expected", [
    ("_internal/config/app_models.json", "config/app_models.json"), ("_internal/medical_terms/anatomy.txt", "medical vocabulary"),
    ("_internal/ctranslate2/ctranslate2.dll", "ctranslate2"), ("_internal/av.libs/avcodec.dll", "av.libs"),
    ("_internal/faster_whisper/assets/silero_vad_v6.onnx", "silero_vad"), ("libs/cuda/cublas64_12.dll", "cublas64_12.dll"),
    ("bin/ffprobe.exe", "ffprobe.exe"), ("_internal/PySide6/Qt6Widgets.dll", "Qt6Widgets"),
])
def test_each_missing_piece_is_reported(tmp_path, remove, expected):
    dist = make_dist(tmp_path)
    (dist / remove).unlink()
    problems = assemble.verify_dist(dist, need_cuda=True, need_ffmpeg=True)
    assert any(expected in p for p in problems), problems


def test_cpu_only_dist_is_fine_when_cuda_is_not_required(tmp_path):
    assert assemble.verify_dist(make_dist(tmp_path, cuda=False, ffmpeg=False)) == []


# ----------------------------------------------------------------------------------------------- packages
def test_zip_has_one_root_folder_and_extras(tmp_path):
    dist = make_dist(tmp_path)
    entries = packages.portable_entries(dist)
    entries += packages.extra_text_files(tmp_path / "stage", "MELETRA", {"portable.flag": "portable\n", "models/README.txt": "x"})
    archive = packages.write_zip(tmp_path / "out" / "p.zip", entries)
    names = packages.zip_names(archive)
    assert all(n.startswith("MELETRA/") for n in names)
    assert "MELETRA/MELETRA.exe" in names and "MELETRA/portable.flag" in names
    assert "MELETRA/_internal/config/transcription.yaml" in names and "MELETRA/models/README.txt" in names
    assert zipfile.ZipFile(archive).testzip() is None and not list((tmp_path / "out").glob("*.part"))


def test_model_files_land_in_models_folder(tmp_path):
    model = tmp_path / "m"
    model.mkdir()
    (model / "model.bin").write_bytes(b"1" * 1000)
    (model / "config.json").write_text("{}")
    names = [n for _p, n in packages.model_entries(model, "faster-whisper-large-v3")]
    assert names == ["MELETRA/models/faster-whisper-large-v3/config.json",
                     "MELETRA/models/faster-whisper-large-v3/model.bin"]


def test_zip_progress_is_reported(tmp_path):
    (tmp_path / "a").write_text("1")
    (tmp_path / "b").write_text("2")
    seen = []
    packages.write_zip(tmp_path / "z.zip", [(tmp_path / "a", "a"), (tmp_path / "b", "b")], progress=lambda i, n: seen.append((i, n)))
    assert seen == [(1, 2), (2, 2)]


def test_readmes_name_the_exe_and_the_smartscreen_prompt():
    for text in (packages.PORTABLE_README, packages.OFFLINE_README):
        assert "{exe}" in text and "SmartScreen" in text and "Run anyway" in text and "Windows 10/11" in text
    assert "data" in packages.PORTABLE_README and "MIT" in packages.MODEL_LICENSE


# ----------------------------------------------------------------------------------------------- clean environment
def test_clean_env_keeps_windows_variables_but_not_the_path(tmp_path):
    source = {"SystemRoot": r"C:\Windows", "PATH": r"C:\Python311;C:\Program Files\NVIDIA\CUDA\bin", "ProgramFiles": r"C:\Program Files",
              "ProgramData": r"C:\ProgramData", "SECRET_TOKEN": "x", "CUDA_PATH": r"C:\cuda"}
    env = windows_only_env(tmp_path, {"EXTRA": "1"}, source)
    assert env["PATH"] == r"C:\Windows\System32;C:\Windows" and env["ProgramFiles"] == r"C:\Program Files" and env["EXTRA"] == "1"
    assert "SECRET_TOKEN" not in env and "CUDA_PATH" not in env and env["MLT_DATA_DIR"] == str(tmp_path / "data")
    for name in ("tmp", "local", "roaming", "profile", "data"):
        assert (tmp_path / name).is_dir()


def test_clean_env_can_run_nvidia_smi_if_present(tmp_path):
    env = windows_only_env(tmp_path)
    for key in ("SystemRoot", "ProgramFiles"):
        assert key in env or key not in os.environ


# ----------------------------------------------------------------------------------------------- build scripts and files
@pytest.mark.windows_only
def test_every_batch_script_exists_and_calls_the_right_step():
    expected = {"RUN_GUI_DEV.bat": "launcher.py", "TEST_ALL.bat": "pytest", "BUILD_APP.bat": "scripts\\build_app.py",
                "BUILD_PORTABLE.bat": "scripts\\build_portable.py", "BUILD_INSTALLER.bat": "scripts\\build_installer.py",
                "BUILD_ALL.bat": "scripts\\build_app.py"}
    for name, needle in expected.items():
        text = (ROOT / name).read_text(encoding="utf-8")
        assert needle in text and "%~dp0" in text and "\r\n" in (ROOT / name).read_bytes().decode("utf-8"), name


def test_build_all_runs_the_steps_in_order_and_stops_on_failure():
    text = (ROOT / "BUILD_ALL.bat").read_text(encoding="utf-8")
    order = [text.index(s) for s in ("TEST_ALL.bat", "build_app.py", "smoke_packaged.py", "build_portable.py", "build_installer.py")]
    assert order == sorted(order) and text.count("|| goto :fail") >= 5 and "exit /b 1" in text


def test_spec_excludes_the_unused_heavy_runtimes_and_keeps_the_resources():
    spec = (ROOT / "build_tools" / "MELETRA.spec").read_text(encoding="utf-8")
    for needle in ('"torch"', '"transformers"', "upx=False", "console=False", "app.ico", "version_info.txt", "faster_whisper",
                   '"medical_terms"', '"medical_vocab"', '"config"', 'contents_directory="_internal"'):
        assert needle in spec, needle


def test_installer_script_matches_the_requirements():
    iss = (ROOT / "build_tools" / "installer.iss").read_text(encoding="utf-8")
    for needle in ("DefaultDirName={autopf}", "DefaultGroupName", "desktopicon", "Flags: unchecked", "{uninstallexe}", "x64compatible",
                   "UninstallDisplayIcon", "AppPublisher={#Publisher}", "lzma2", "{localappdata}\\MedicalLectureTranscriber", "MB_DEFBUTTON2",
                   "PrivilegesRequiredOverridesAllowed"):
        assert needle in iss, needle
    assert "Touche Holding" in iss


def test_find_iscc_prefers_the_environment_override(tmp_path, monkeypatch):
    from scripts import build_installer

    fake = tmp_path / "ISCC.exe"
    fake.write_bytes(b"x")
    monkeypatch.setenv("ISCC_PATH", str(fake))
    assert build_installer.find_iscc() == fake
    monkeypatch.setattr(build_installer, "find_iscc", lambda: fake)
    assert build_installer.ensure_inno_setup() == fake


def test_run_streaming_kills_a_silent_process(tmp_path):
    from scripts import build_app

    started = time.monotonic()
    code = build_app.run_streaming([sys.executable, "-c", "import time; print('hi', flush=True); time.sleep(60)"], tmp_path / "l.log", 2.0)
    assert code == 124 and time.monotonic() - started < 20
    assert "no output for" in (tmp_path / "l.log").read_text(encoding="utf-8")


def test_run_streaming_returns_the_exit_code_and_logs_output(tmp_path):
    from scripts import build_app

    code = build_app.run_streaming([sys.executable, "-c", "print('line one'); raise SystemExit(3)"], tmp_path / "l.log", 30.0)
    assert code == 3 and "line one" in (tmp_path / "l.log").read_text(encoding="utf-8")


def test_pyinstaller_analysis_cannot_borrow_dlls_from_the_callers_path(tmp_path, monkeypatch):
    from scripts import build_app

    seen = {}
    monkeypatch.setattr(build_app, "DIST", tmp_path / "dist")
    monkeypatch.setattr(build_app, "BUILD", tmp_path / "build")
    monkeypatch.setattr(build_app, "SPEC", tmp_path / "app.spec")
    monkeypatch.setattr(build_app, "LOG", tmp_path / "build.log")
    monkeypatch.setattr(build_app, "run_streaming", lambda *a, **kw: seen.update(kw) or 0)
    build_app.build_pyinstaller(30)
    assert seen["env"]["PATH"].lower().split(";") == [str(Path(os.environ["SystemRoot"]) / "System32").lower(), os.environ["SystemRoot"].lower()]


def test_prepare_assets_writes_icon_and_version(qapp, tmp_path, monkeypatch):
    from scripts import build_app

    monkeypatch.setattr(build_app, "BUILD", tmp_path)
    build_app.prepare_assets()
    assert (tmp_path / "assets" / "app.ico").stat().st_size > 1000 and "ProductVersion" in (tmp_path / "assets" / "version_info.txt").read_text()


def test_installer_and_portable_names_match_the_release_names():
    from scripts import build_fulloffline, build_installer, build_portable

    assert build_installer.OUTPUT_BASE == "MELETRA-Setup-Windows-x64"
    assert build_portable.NAME == "MELETRA-Portable-Windows-x64.zip"
    assert build_fulloffline.NAME == "MELETRA-FullOffline-x64.zip"


# ----------------------------------------------------------------------------------------------- smoke helpers
def make_output_dir(tmp_path: Path, words="glycolysis hexokinase pyruvate", srt=True, vtt=True, segments=True) -> Path:
    out = tmp_path / "out"
    out.mkdir()
    for name in ("local_clean_transcript.txt", "local_transcript_with_timestamps.txt"):
        (out / name).write_text(words, encoding="utf-8")
    (out / "local_raw_transcript.txt").write_text(words, encoding="utf-8")
    (out / "local_transcript.srt").write_text("1\n00:00:00,000 --> 00:00:02,000\nhi\n" if srt else "garbage", encoding="utf-8")
    (out / "local_transcript.vtt").write_text("WEBVTT\n\n00:00.000 --> 00:02.000\nhi\n" if vtt else "nope", encoding="utf-8")
    (out / "local_transcript.json").write_text(json.dumps({"segments": [{"text": "hi"}] if segments else []}), encoding="utf-8")
    return out


def test_smoke_output_check_accepts_a_good_result(tmp_path):
    from scripts import smoke_packaged

    ok, detail = smoke_packaged.check_outputs(make_output_dir(tmp_path), ["glycolysis", "hexokinase", "pyruvate"])
    assert ok and "3 words" in detail


@pytest.mark.parametrize("kwargs, message", [({"srt": False}, "SRT"), ({"vtt": False}, "VTT"), ({"segments": False}, "no segments"),
                                              ({"words": "nothing relevant"}, "expected terms")])
def test_smoke_output_check_names_what_is_wrong(tmp_path, kwargs, message):
    from scripts import smoke_packaged

    ok, detail = smoke_packaged.check_outputs(make_output_dir(tmp_path, **kwargs), ["glycolysis", "hexokinase", "pyruvate"])
    assert not ok and message in detail


def test_smoke_output_check_reports_missing_files(tmp_path):
    from scripts import smoke_packaged

    out = make_output_dir(tmp_path)
    (out / "local_transcript.vtt").unlink()
    ok, detail = smoke_packaged.check_outputs(out, ["x"])
    assert not ok and "local_transcript.vtt" in detail


def test_mp3_made_for_the_pyav_step_is_really_decodable(tmp_path):
    from scripts import smoke_packaged
    from src.audio import probe_media_pyav
    from tests.conftest import write_silent_wav

    wav = write_silent_wav(tmp_path / "s.wav", seconds=3)
    mp3 = smoke_packaged.to_mp3(wav, tmp_path / "s.mp3")
    assert mp3.stat().st_size > 1000 and abs(probe_media_pyav(mp3)["duration_s"] - 3.0) < 0.5


def test_smoke_settings_file_skips_the_wizard(tmp_path):
    from scripts import smoke_packaged

    smoke_packaged.write_settings(tmp_path, output_dir="X")
    data = json.loads((tmp_path / "data" / "settings" / "settings.json").read_text(encoding="utf-8"))
    assert data["first_run_done"] is True and data["output_dir"] == "X"


@pytest.mark.windows_only
def test_windows_of_an_unknown_process_is_empty():
    from scripts import smoke_packaged

    assert smoke_packaged.windows_of(2_000_000_000) == []


def test_run_exe_kills_a_hung_program(tmp_path):
    from scripts import smoke_packaged

    started = time.monotonic()
    code, timed_out = smoke_packaged.run_exe(Path(sys.executable), ["-c", "import time; time.sleep(60)"], dict(os.environ), tmp_path, 2)
    assert code is None and timed_out and time.monotonic() - started < 15


def test_a_build_without_the_medical_expert_vocabulary_is_reported(tmp_path):
    import shutil
    dist = make_dist(tmp_path)
    shutil.rmtree(dist / "_internal" / "medical_vocab")
    assert any("medical_vocab" in problem for problem in assemble.verify_dist(dist))


def test_every_data_folder_the_app_loads_at_run_time_is_packaged():
    """The packaged app showed 'no vocabulary' because a resource folder was missing from the spec: keep the two lists in step."""
    import re
    spec = (ROOT / "build_tools" / "MELETRA.spec").read_text(encoding="utf-8")
    packaged = set(re.findall(r'ROOT / "(\w+)"\), "(\w+)"', spec))
    packaged = {a for a, b in packaged if a == b}
    from src.medical import lexicon
    assert lexicon.VOCAB_DIR.name in packaged and "config" in packaged and "medical_terms" in packaged


def test_the_pyinstaller_specs_are_valid_python_and_bundle_the_knowledge_pack_with_its_marker():
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    for name in ("MELETRA.spec", "MELETRA-macos.spec"):
        source = (root / "build_tools" / name).read_text(encoding="utf-8")
        ast.parse(source)
        assert "medical_knowledge_public.db" in source and "REQUIRED" in source and "NOTICE.txt" in source


def test_the_windows_notice_tells_the_truth_about_the_ffmpeg_libraries_inside_pyav():
    from build_tools import assemble

    text = assemble.notices_text(True, True)
    assert "GPL components" in text and "libx264" in text and "libx265" in text and "must be treated as GPL" in text
    assert "FFmpeg libraries (LGPL-2.1+): the decoder" not in text and "LGPL-2.1+ build" in text            # the ffmpeg.exe tools ARE an LGPL build; the PyAV libraries are not


def test_the_notices_name_sentencepiece_and_the_optional_translation_pack_and_their_licence_texts_ship(tmp_path):
    from build_tools import assemble

    text = assemble.notices_text(False, False)
    assert "SentencePiece (Apache-2.0" in text and "M2M100" in text and "MIT licence" in text and "licenses\APACHE-2.0.txt" in text and "licenses\LICENSE-M2M100.txt" in text
    assert assemble.write_extra_licenses(tmp_path) == ["APACHE-2.0.txt", "LICENSE-M2M100.txt"]
    assert "Apache License" in (tmp_path / "licenses" / "APACHE-2.0.txt").read_text(encoding="utf-8")
    assert "Copyright (c) Facebook, Inc." in (tmp_path / "licenses" / "LICENSE-M2M100.txt").read_text(encoding="utf-8")
