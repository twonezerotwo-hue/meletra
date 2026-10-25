"""Shared fixtures. The default suite never downloads a model and never needs a GPU."""

from __future__ import annotations

import os
import sys
import subprocess
import wave
from pathlib import Path

import pytest

from src.config import load_transcription_config
from src.platform_utils import find_executable
from src.schema import Segment

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")      # the GUI tests never open a real window
if sys.platform == "win32" and Path(r"C:\Windows\Fonts").is_dir():
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")   # the offscreen platform has no fonts of its own: real fonts give real text widths (layout tests)

try:                                                       # Qt must be loaded BEFORE the Windows runtime (winrt: OCR, voices): the other
    import PySide6.QtWidgets  # noqa: F401                  # order crashes the process (access violation), also in a partial test run
except ImportError:                                        # pragma: no cover - the GUI tests are skipped without PySide6
    pass


def pytest_collection_modifyitems(config, items):
    """Tests marked ``windows_only`` call real Windows APIs; on other systems they are skipped, not failed."""
    import sys

    if sys.platform == "win32":
        return
    skip = pytest.mark.skip(reason="needs Windows")
    for item in items:
        if "windows_only" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _windows_platform_baseline(request):
    """Tests marked ``windows_platform`` assert the Windows branch of the code (GPU banners, "Follow Windows", %LOCALAPPDATA% ...). On another
    host they run with the Windows platform object, so the same assertions hold; the macOS branches have their own tests with a fake Mac."""
    import sys

    if request.node.get_closest_marker("windows_platform") is None or sys.platform == "win32":
        yield
        return
    from src.platform import WindowsPlatform, use

    with use(WindowsPlatform()):
        yield


@pytest.fixture(scope="session")
def cfg() -> dict:
    return load_transcription_config()


@pytest.fixture(scope="session")
def ffmpeg_path():
    path = find_executable("ffmpeg")
    if path is None:
        pytest.skip("FFmpeg not installed")
    # A machine that is out of memory can take minutes just to START ffmpeg.exe; a hung test run is worse than a skip.
    probe = subprocess.Popen([str(path), "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        probe.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            probe.kill()                    # never wait for it again: a process stuck in the loader may not die
        except OSError:
            pass
        pytest.skip("FFmpeg did not start within 30 s (machine under heavy memory pressure)")
    return path


@pytest.fixture
def fake_ffmpeg(monkeypatch):
    """Pipeline tests feed 16 kHz mono WAV files, so extraction is a plain copy: no ffmpeg.exe is launched.
    (Real FFmpeg extraction is covered by tests/test_audio.py and tests/test_pipeline_ffmpeg.py.)"""
    import soundfile as sf

    from src import audio as audio_mod

    def extract(source, destination, audio_cfg, preprocessing="none", ffmpeg=None, allow_fallback=False, progress=None, should_stop=None):
        audio_mod.validate_media_file(source, audio_cfg)
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        info = sf.info(str(source))
        if info.format == "WAV" and info.samplerate == 16000 and info.channels == 1 and info.subtype == "PCM_16":
            import shutil

            shutil.copyfile(source, destination)                 # already in the pipeline's format: no decode, no big arrays
        else:
            data, rate = sf.read(str(source))
            sf.write(str(destination), data, rate, subtype="PCM_16")
        return Path(destination)

    def probe(path, ffprobe):
        info = sf.info(str(path))
        return {"duration_s": info.frames / float(info.samplerate)}

    monkeypatch.setattr(audio_mod, "locate_ffmpeg", lambda cfg=None: (Path("ffmpeg"), Path("ffprobe")))
    monkeypatch.setattr(audio_mod, "probe_media", probe)
    monkeypatch.setattr(audio_mod, "extract_audio", extract)


def make_segment(text: str, start: float = 0.0, end: float = 2.0, **kwargs) -> Segment:
    defaults = dict(language="en", model="large-v3", backend="faster-whisper",
                    avg_logprob=-0.2, no_speech_prob=0.01, compression_ratio=1.2)
    defaults.update(kwargs)
    return Segment(start=start, end=end, text=text, **defaults)


@pytest.fixture
def segment_factory():
    return make_segment


def write_silent_wav(path: Path, seconds: float = 1.0, rate: int = 16000) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * int(seconds * rate))
    return path


@pytest.fixture(autouse=True)
def _no_real_gpu_probe(monkeypatch):
    """Pipeline tests use fake ASR backends: do not let device="auto" import CTranslate2 / call nvidia-smi."""
    monkeypatch.setattr("src.pipeline.resolve_device", lambda requested="auto", index=0: (
        requested if requested in ("cpu", "cuda") else "cuda", None))


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(["tests"])
    yield app


@pytest.fixture(autouse=True)
def _release_finished_windows():
    """Every GUI test builds whole windows; Qt re-polishes ALL live widgets whenever the style sheet changes, so windows of finished tests are released here
    (otherwise the cost of one theme change grows with the number of tests that ran before it)."""
    yield
    if "PySide6.QtWidgets" not in sys.modules:
        return
    from PySide6.QtCore import QEvent
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        return
    import gc

    gc.collect()
    app.sendPostedEvents(None, QEvent.DeferredDelete.value if hasattr(QEvent.DeferredDelete, "value") else QEvent.DeferredDelete)


@pytest.fixture(autouse=True)
def _no_local_ai_server_in_tests(monkeypatch):
    """The Automatic Study Engine looks for Ollama on this computer. A test must not depend on what the developer's machine happens to run (and must not start a real model):
    the real probe refuses, and a test that wants a model passes its own transport."""
    from src.study import automatic
    from src.study.providers import ProviderError

    def refuse(url, headers, body, timeout):
        raise ProviderError("no AI server is reachable in tests")

    monkeypatch.setattr(automatic, "urllib_transport", refuse)
