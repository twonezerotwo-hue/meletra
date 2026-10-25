"""Speech voices that are already on the computer. No download, no model, nothing loaded until a voice is actually used.

Windows   OneCore voices (WinRT, in-process) and desktop SAPI voices (a hidden PowerShell call).
macOS     the ``say`` command (UNVERIFIED: written against its documented options, tested with a fake runner only).

Every backend writes a WAV file. A backend that cannot run here reports ``available() == False``; nothing raises at import time.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger(__name__)

Runner = Callable[..., "subprocess.CompletedProcess[str]"]


class TtsError(RuntimeError):
    """A voice problem in plain words."""


@dataclass(frozen=True)
class Voice:
    id: str
    name: str
    language: str                                    # BCP-47, e.g. "tr-TR"
    backend: str
    gender: str = ""

    @property
    def lang(self) -> str:                           # "tr"
        return self.language.split("-")[0].lower()


def _hidden() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


# ------------------------------------------------------------------------------------------------ Windows OneCore (WinRT)
class OneCoreBackend:
    name = "windows-onecore"
    max_chars = 3000

    def available(self) -> bool:
        if sys.platform != "win32":
            return False
        try:
            from winrt.windows.media.speechsynthesis import SpeechSynthesizer  # noqa: F401
            return True
        except Exception:                            # noqa: BLE001 - a missing package or Windows API
            return False

    def voices(self) -> list[Voice]:
        from winrt.windows.media.speechsynthesis import SpeechSynthesizer

        return [Voice(v.id, v.display_name, v.language, self.name, "female" if int(v.gender) == 1 else "male")
                for v in SpeechSynthesizer.all_voices]

    def synthesize(self, text: str, voice: Voice, wav_path: Path, rate: int = 0) -> None:
        wav_path.write_bytes(asyncio.run(self._render(text, voice, rate)))

    async def _render(self, text: str, voice: Voice, rate: int) -> bytes:
        from winrt.windows.media.speechsynthesis import SpeechSynthesizer
        from winrt.windows.storage.streams import DataReader

        synth = SpeechSynthesizer()
        chosen = next((v for v in SpeechSynthesizer.all_voices if v.id == voice.id), None)
        if chosen is None:
            raise TtsError(f"The voice '{voice.name}' is no longer installed.")
        synth.voice = chosen
        synth.options.speaking_rate = max(0.5, min(3.0, 1.0 + rate / 10.0))      # app scale -10..10 -> 0.5..3.0 (1.0 = normal)
        stream = await synth.synthesize_text_to_stream_async(text)
        reader = DataReader(stream.get_input_stream_at(0))
        await reader.load_async(stream.size)
        buffer = bytearray(stream.size)
        reader.read_bytes(buffer)
        return bytes(buffer)


# ------------------------------------------------------------------------------------------------ Windows desktop SAPI (PowerShell)
_LIST = ("[Console]::OutputEncoding=[Text.Encoding]::UTF8;Add-Type -AssemblyName System.Speech;"
         "(New-Object System.Speech.Synthesis.SpeechSynthesizer).GetInstalledVoices()|Where-Object{$_.Enabled}|ForEach-Object{"
         "$_.VoiceInfo.Name+'|'+$_.VoiceInfo.Culture.Name+'|'+$_.VoiceInfo.Gender}")
_SPEAK = ("Add-Type -AssemblyName System.Speech;$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
          "if($env:MLT_TTS_VOICE){$s.SelectVoice($env:MLT_TTS_VOICE)};$s.Rate=[int]$env:MLT_TTS_RATE;"
          "$s.SetOutputToWaveFile($env:MLT_TTS_WAV);"
          "$s.Speak([IO.File]::ReadAllText($env:MLT_TTS_TEXT,[Text.Encoding]::UTF8));$s.Dispose()")


def _powershell() -> str:
    """Windows PowerShell by its full path (a program started with a trimmed PATH cannot find it by name)."""
    root = os.environ.get("SystemRoot") or os.environ.get("windir") or "C:/Windows"
    candidate = Path(root) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    return str(candidate) if candidate.is_file() else "powershell"


def _encoded(script: str) -> list[str]:
    return [_powershell(), "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-EncodedCommand",
            base64.b64encode(script.encode("utf-16-le")).decode("ascii")]


class SapiBackend:
    name = "windows-sapi"
    max_chars = 8000

    def __init__(self, run: Optional[Runner] = None) -> None:
        self._run = run or subprocess.run

    def available(self) -> bool:
        return sys.platform == "win32"

    def voices(self) -> list[Voice]:
        done = self._run(_encoded(_LIST), capture_output=True, text=True, encoding="utf-8", timeout=60, creationflags=_hidden())
        out = []
        for line in (done.stdout or "").splitlines():
            parts = line.strip().split("|")
            if len(parts) == 3 and parts[0]:
                out.append(Voice(parts[0], parts[0], parts[1], self.name, parts[2].lower()))
        return out

    def synthesize(self, text: str, voice: Voice, wav_path: Path, rate: int = 0) -> None:
        with tempfile.TemporaryDirectory(prefix="mlt_tts_") as folder:
            text_file = Path(folder) / "text.txt"
            text_file.write_text(text, encoding="utf-8")
            env = {**os.environ, "MLT_TTS_VOICE": voice.id, "MLT_TTS_RATE": str(max(-10, min(10, rate))), "MLT_TTS_WAV": str(wav_path),
                   "MLT_TTS_TEXT": str(text_file)}
            done = self._run(_encoded(_SPEAK), capture_output=True, text=True, encoding="utf-8", timeout=600, env=env,
                             creationflags=_hidden())
        if done.returncode != 0 or not wav_path.is_file():
            raise TtsError(f"The voice '{voice.name}' could not speak this text. {(done.stderr or '').strip()[:200]}")


# ------------------------------------------------------------------------------------------------ macOS
class SayBackend:
    name = "macos-say"
    max_chars = 8000
    _ROW = re.compile(r"^(?P<name>.+?)\s+(?P<loc>[a-z]{2,3}[_-][A-Za-z]{2,4})\s+#")

    def __init__(self, run: Optional[Runner] = None, platform: Optional[str] = None) -> None:
        self._run = run or subprocess.run
        self._platform = platform or sys.platform

    def available(self) -> bool:
        return self._platform == "darwin"

    def voices(self) -> list[Voice]:
        done = self._run(["say", "-v", "?"], capture_output=True, text=True, encoding="utf-8", timeout=30)
        found = []
        for line in (done.stdout or "").splitlines():
            m = self._ROW.match(line)
            if m:
                found.append(Voice(m["name"].strip(), m["name"].strip(), m["loc"].replace("_", "-"), self.name))
        return found

    def synthesize(self, text: str, voice: Voice, wav_path: Path, rate: int = 0) -> None:
        with tempfile.TemporaryDirectory(prefix="mlt_tts_") as folder:
            text_file = Path(folder) / "text.txt"
            text_file.write_text(text, encoding="utf-8")
            words_per_minute = int(175 * (1.0 + max(-10, min(10, rate)) / 10.0))
            done = self._run(["say", "-v", voice.id, "-r", str(words_per_minute), "-o", str(wav_path), "--file-format=WAVE",
                              "--data-format=LEI16@22050", "-f", str(text_file)], capture_output=True, text=True, timeout=600)
        if done.returncode != 0 or not Path(wav_path).is_file():
            raise TtsError(f"The voice '{voice.name}' could not speak this text. {(done.stderr or '').strip()[:200]}")


def default_backends() -> list:
    if sys.platform == "win32":
        return [OneCoreBackend(), SapiBackend()]
    if sys.platform == "darwin":
        return [SayBackend()]
    return []


# ------------------------------------------------------------------------------------------------ playing a WAV file
class Player:
    """Plays one WAV file in the background. Windows: the system's MCI player (it can be stopped and tells when it is done);
    macOS: ``afplay``."""

    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._alias = "mlt_tts"

    def _mci(self, command: str) -> str:
        import ctypes

        buffer = ctypes.create_unicode_buffer(256)
        error = ctypes.windll.winmm.mciSendStringW(command, buffer, 255, 0)
        if error:
            raise TtsError("The sound could not be played (no audio output?).")
        return buffer.value

    def play(self, wav_path: Path) -> None:
        self.stop()
        if sys.platform == "win32":
            self._mci(f'open "{wav_path}" type waveaudio alias {self._alias}')
            self._mci(f"play {self._alias}")
        elif sys.platform == "darwin":
            self._proc = subprocess.Popen(["afplay", str(wav_path)])
        else:
            raise TtsError("Playing speech is not available on this system. Save the audio file instead.")

    def playing(self) -> bool:
        if sys.platform == "win32":
            try:
                return self._mci(f"status {self._alias} mode") == "playing"
            except TtsError:
                return False
        return self._proc is not None and self._proc.poll() is None

    def stop(self) -> None:
        if sys.platform == "win32":
            try:
                self._mci(f"close {self._alias}")                     # stops and releases the file
            except TtsError:
                pass
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
        self._proc = None
