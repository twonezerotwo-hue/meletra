"""Text to speech: read a transcript or a document aloud, or save it as an audio file.

An independent, optional subsystem. It imports nothing from the transcription or study code, loads no model, and a problem here
(no voice, no audio device) is reported to the caller as :class:`TtsError` - it can never stop the transcriber from starting.

    service = TtsService()
    service.list_voices()                       -> [Voice, ...]
    service.synthesize(text, "out.wav", language="tr")
    service.speak(text, language="en")  # blocks while reading aloud; service.stop() ends it
    service.export(text, "out.mp3")             -> WAV always, MP3 when the file name says so
"""

from __future__ import annotations

import logging
import re
import tempfile
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Optional

from .backends import Player, TtsError, Voice, default_backends

log = logging.getLogger(__name__)

FIRST_PIECE_CHARS = 1200                           # a short piece first and then the same size: playback starts after one short piece

__all__ = ["TtsService", "TtsError", "Voice", "chunk_text", "guess_language"]

_SENTENCE = re.compile(r"(?<=[.!?…])\s+|\n{2,}|\n(?=\s*[-•*\d])")
_TR_LETTERS, _HU_LETTERS, _DE_LETTERS = set("ğşıİĞŞ"), set("őűŐŰ"), set("ßäüö")
_WORDS = {"en": {"the", "and", "of", "is", "to", "in", "that", "with", "for", "are"},
          "de": {"der", "die", "das", "und", "ist", "nicht", "mit", "ein", "zu", "den"},
          "tr": {"ve", "bir", "bu", "ile", "için", "olan", "da", "de", "daha", "gibi"},
          "hu": {"a", "az", "és", "hogy", "nem", "egy", "van", "is", "meg", "mint"}}


def guess_language(text: str, default: str = "en") -> str:
    """tr / de / hu / en from the letters and the most common words of the first part of the text (voice choice only)."""
    sample = text[:4000]
    letters = {c for c in sample}
    if letters & _HU_LETTERS:
        return "hu"
    if letters & _TR_LETTERS:
        return "tr"
    words = re.findall(r"[^\W\d_]+", sample.lower())
    score = {lang: sum(1 for w in words if w in common) for lang, common in _WORDS.items()}
    if letters & _DE_LETTERS:
        score["de"] += 3
    best = max(score, key=score.get)
    return best if score[best] >= 3 else default


def chunk_text(text: str, max_chars: int) -> list[str]:
    """Pieces of at most ``max_chars``, cut at sentence ends (or at a space when one sentence is longer). No text is lost."""
    pieces: list[str] = []
    current = ""
    for sentence in (s.strip() for s in _SENTENCE.split(text.replace("\r\n", "\n"))):
        if not sentence:
            continue
        while len(sentence) > max_chars:
            cut = sentence.rfind(" ", 0, max_chars)
            cut = cut if cut > max_chars // 2 else max_chars
            head, sentence = sentence[:cut].strip(), sentence[cut:].strip()
            if current:
                pieces.append(current)
                current = ""
            pieces.append(head)
        if len(current) + len(sentence) + 1 > max_chars and current:
            pieces.append(current)
            current = ""
        current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def _join_wavs(parts: list[Path], target: Path) -> None:
    with wave.open(str(parts[0]), "rb") as first:
        params = first.getparams()
    with wave.open(str(target), "wb") as out:
        out.setparams(params)
        for part in parts:
            with wave.open(str(part), "rb") as piece:
                if piece.getparams()[:3] != params[:3]:
                    raise TtsError("The voice produced audio in different formats; the text could not be joined.")
                while frames := piece.readframes(65536):
                    out.writeframes(frames)


def _wav_to_mp3(wav_path: Path, mp3_path: Path) -> None:
    """MP3 with the encoder that is part of PyAV (the FFmpeg libraries the app already uses to read audio)."""
    import av

    with av.open(str(wav_path)) as source, av.open(str(mp3_path), "w", format="mp3") as target:
        stream_in = source.streams.audio[0]
        stream_out = target.add_stream("libmp3lame", rate=stream_in.rate)
        resampler = av.AudioResampler(format=stream_out.format.name, layout=stream_out.layout.name, rate=stream_out.rate)
        for frame in source.decode(stream_in):
            for piece in resampler.resample(frame):
                target.mux(stream_out.encode(piece))
        target.mux(stream_out.encode(None))


class TtsService:
    def __init__(self, backends: Optional[list] = None, player: Optional[Player] = None) -> None:
        self._backends = backends if backends is not None else default_backends()
        self._player = player
        self._voices: Optional[list[Voice]] = None
        self._stop = threading.Event()
        self._by_name = {b.name: b for b in self._backends}

    # ------------------------------------------------------------------ voices
    def list_voices(self, language: Optional[str] = None, refresh: bool = False) -> list[Voice]:
        if self._voices is None or refresh:
            found: list[Voice] = []
            for backend in self._backends:
                try:
                    if backend.available():
                        found += backend.voices()
                except Exception as exc:                         # noqa: BLE001 - a broken backend must not hide the others
                    log.info("Voices of %s are not available: %s", backend.name, exc)
            self._voices = found
        if language and language != "auto":
            return [v for v in self._voices if v.lang == language.split("-")[0].lower()]
        return list(self._voices)

    def pick_voice(self, language: str) -> Optional[Voice]:
        """The preferred voice for a language: the exact region first (en-US before en-GB), then any voice of the language."""
        matches = self.list_voices(language)
        want = language.lower()
        return next((v for v in matches if v.language.lower() == want), matches[0] if matches else None)

    def _resolve(self, text: str, voice: Optional[Voice], language: str) -> Voice:
        if voice is not None:
            return voice
        lang = guess_language(text) if language in ("", "auto") else language
        chosen = self.pick_voice(lang)
        if chosen is None:
            names = {"tr": "Turkish", "en": "English", "de": "German", "hu": "Hungarian"}
            raise TtsError(f"No {names.get(lang, lang)} voice is installed on this computer. Add one in the system's speech settings "
                           f"(Windows: Settings > Time & language > Speech), then try again.")
        return chosen

    # ------------------------------------------------------------------ speaking
    def synthesize(self, text: str, path: Path | str, voice: Optional[Voice] = None, language: str = "auto", rate: int = 0,
                   progress: Optional[Callable[[dict[str, Any]], None]] = None,
                   should_stop: Optional[Callable[[], bool]] = None) -> Path:
        """Speak ``text`` into a WAV file (long text in safe pieces, joined into one file)."""
        if not text.strip():
            raise TtsError("There is no text to read.")
        chosen = self._resolve(text, voice, language)
        backend = self._by_name[chosen.backend]
        pieces = chunk_text(text, backend.max_chars)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="mlt_tts_") as folder:
            parts: list[Path] = []
            for index, piece in enumerate(pieces):
                if should_stop and should_stop():
                    raise TtsError("cancelled")
                part = Path(folder) / f"{index:05d}.wav"
                try:
                    backend.synthesize(piece, chosen, part, rate)
                except TtsError:
                    raise
                except Exception as exc:                         # noqa: BLE001 - whatever the system speech API raises
                    raise TtsError(f"The voice '{chosen.name}' could not speak this text ({exc.__class__.__name__}).") from exc
                parts.append(part)
                if progress:
                    progress({"type": "tts_progress", "part": index + 1, "parts": len(pieces), "percent": 100.0 * (index + 1) / len(pieces)})
            _join_wavs(parts, target)
        return target

    def export(self, text: str, path: Path | str, **kwargs: Any) -> Path:
        """Audio file named by the extension: ``.wav`` (always possible) or ``.mp3``."""
        target = Path(path)
        if target.suffix.lower() == ".mp3":
            with tempfile.TemporaryDirectory(prefix="mlt_tts_") as folder:
                wav = self.synthesize(text, Path(folder) / "speech.wav", **kwargs)
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    _wav_to_mp3(wav, target)
                except Exception as exc:                         # noqa: BLE001 - encoder problems must not lose the speech
                    raise TtsError(f"The MP3 file could not be written ({exc.__class__.__name__}). Save as WAV instead.") from exc
            return target
        if target.suffix.lower() != ".wav":
            raise TtsError("Choose a .wav or .mp3 file name.")
        return self.synthesize(text, target, **kwargs)

    def speak(self, text: str, voice: Optional[Voice] = None, language: str = "auto", rate: int = 0,
              should_stop: Optional[Callable[[], bool]] = None, progress: Optional[Callable[[dict[str, Any]], None]] = None) -> bool:
        """Read ``text`` aloud, piece by piece: the next piece is prepared while the current one plays, so it starts quickly.
        Blocks until it has finished or :meth:`stop` was called (run it in a background thread). Returns True when it read it all."""
        if not text.strip():
            raise TtsError("There is no text to read.")
        chosen = self._resolve(text, voice, language)
        backend = self._by_name[chosen.backend]
        pieces = chunk_text(text, min(backend.max_chars, FIRST_PIECE_CHARS))
        self._player = self._player or Player()
        self._stop.clear()

        def stopped() -> bool:
            return self._stop.is_set() or bool(should_stop and should_stop())

        def prepare(index: int, folder: Path) -> Path:
            part = folder / f"{index:05d}.wav"
            try:
                backend.synthesize(pieces[index], chosen, part, rate)
            except TtsError:
                raise
            except Exception as exc:                             # noqa: BLE001 - whatever the system speech API raises
                raise TtsError(f"The voice '{chosen.name}' could not speak this text ({exc.__class__.__name__}).") from exc
            return part

        with tempfile.TemporaryDirectory(prefix="mlt_tts_play_") as name, ThreadPoolExecutor(max_workers=1) as pool:
            folder = Path(name)
            pending = pool.submit(prepare, 0, folder)
            try:
                for index in range(len(pieces)):
                    wav = pending.result()
                    if index + 1 < len(pieces) and not stopped():
                        pending = pool.submit(prepare, index + 1, folder)
                    if stopped():
                        return False
                    self._player.play(wav)
                    if progress:
                        progress({"type": "tts_progress", "part": index + 1, "parts": len(pieces), "percent": 100.0 * index / len(pieces)})
                    while self._player.playing() and not stopped():
                        time.sleep(0.05)
                    if stopped():
                        return False
                return True
            finally:
                self._player.stop()

    def stop(self) -> None:
        self._stop.set()
        if self._player is not None:
            self._player.stop()
