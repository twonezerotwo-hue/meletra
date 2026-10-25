"""Media validation and audio extraction with FFmpeg."""

from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
import wave
from pathlib import Path
from typing import Any, Callable, Optional

from .platform_utils import find_executable

log = logging.getLogger(__name__)


class UnsupportedMediaError(ValueError):
    """The file extension is not a supported audio/video format."""


class FFmpegNotFoundError(RuntimeError):
    """FFmpeg / FFprobe could not be located."""


class AudioExtractionError(RuntimeError):
    """FFmpeg ran but could not produce the WAV file."""


FFMPEG_INSTALL_HINT = (
    "FFmpeg was not found. Install it with:  winget install Gyan.FFmpeg  "
    "(then open a NEW terminal), or set audio.ffmpeg_path in config/transcription.yaml, "
    "or set the FFMPEG_PATH environment variable."
)


def supported_extensions(audio_cfg: dict[str, Any]) -> set[str]:
    """All allowed extensions (lower-case, with the leading dot) from the config."""
    groups = audio_cfg.get("supported_extensions", {})
    return {ext.lower() for group in groups.values() for ext in group}


def validate_media_file(path: Path, audio_cfg: dict[str, Any]) -> Path:
    """Check existence and extension; return the path or raise a clear error."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Media file not found: {path}")
    allowed = supported_extensions(audio_cfg)
    if path.suffix.lower() not in allowed:
        raise UnsupportedMediaError(
            f"Unsupported file type '{path.suffix}' for {path.name}. "
            f"Supported: {', '.join(sorted(allowed))}"
        )
    return path


def list_media_files(folder: Path, audio_cfg: dict[str, Any]) -> list[Path]:
    """Supported media files directly inside ``folder`` (sorted, non-recursive)."""
    allowed = supported_extensions(audio_cfg)
    return sorted(p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in allowed)


def locate_ffmpeg(audio_cfg: Optional[dict[str, Any]] = None) -> tuple[Path, Optional[Path]]:
    """Return ``(ffmpeg, ffprobe-or-None)`` or raise :class:`FFmpegNotFoundError`."""
    explicit = (audio_cfg or {}).get("ffmpeg_path")
    ffmpeg = find_executable("ffmpeg", explicit)
    if ffmpeg is None:
        raise FFmpegNotFoundError(FFMPEG_INSTALL_HINT)
    ffprobe = find_executable("ffprobe", explicit) or find_executable("ffprobe", ffmpeg.parent)
    return ffmpeg, ffprobe


def locate_ffmpeg_optional(audio_cfg: Optional[dict[str, Any]] = None) -> tuple[Optional[Path], Optional[Path]]:
    """Like :func:`locate_ffmpeg` but returns ``(None, None)`` instead of raising (the app then decodes with PyAV)."""
    try:
        return locate_ffmpeg(audio_cfg)
    except FFmpegNotFoundError:
        return None, None


class ExternalToolTimeout(RuntimeError):
    """An external program (ffmpeg / ffprobe) did not start or finish in time."""


class ExtractionCancelled(Exception):
    """The caller asked to stop (pause / cancel) while the audio was being prepared."""


STALL_SECONDS = 60.0            # an ffmpeg whose output time has not moved for this long is stuck, not slow


def _read_progress(proc: "subprocess.Popen[str]", state: dict[str, float], on_progress: Optional[Callable[[float], None]]) -> None:
    """Follow ``ffmpeg -progress pipe:1``: ``out_time_us`` is how much of the media has been written so far."""
    for line in proc.stdout or []:
        key, _, value = line.strip().partition("=")
        if key in ("out_time_us", "out_time_ms") and value.lstrip("-").isdigit() and int(value) > 0:
            seconds = int(value) / 1_000_000.0
            if seconds > state["seconds"]:
                state["seconds"], state["moved"] = seconds, time.monotonic()
                if on_progress is not None:
                    on_progress(seconds)


def _run(cmd: list[str], timeout: Optional[float] = None) -> subprocess.CompletedProcess[str]:
    """Run ``cmd`` and capture its text output. On a timeout the process is killed and NEVER waited for again: a process
    stuck in the Windows loader (seen on machines that are out of memory) may not die, and ``subprocess.run`` would then
    block forever."""
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
                            errors="replace", creationflags=flags)
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        try:
            proc.kill()
        except OSError:
            pass
        raise ExternalToolTimeout(f"{Path(cmd[0]).name} did not finish within {timeout:.0f} s") from exc
    return subprocess.CompletedProcess(cmd, proc.returncode, out, err)


def _run_extraction(cmd: list[str], destination: Path, start_deadline: float = 45.0,
                    on_progress: Optional[Callable[[float], None]] = None, should_stop: Optional[Callable[[], bool]] = None,
                    stall_seconds: Optional[float] = None) -> subprocess.CompletedProcess[str]:
    """Run an ffmpeg extraction with watchdogs: a healthy ffmpeg creates its output file within seconds (START watchdog), and - when
    ``-progress pipe:1`` was requested - keeps moving (STALL watchdog); one that does not is stuck, not slow, and is abandoned (only
    that child process is killed). ``should_stop`` ends it at once (``ExtractionCancelled``)."""
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    destination.unlink(missing_ok=True)
    watching = "-progress" in cmd
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE if watching else subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                            encoding="utf-8", errors="replace", creationflags=flags)
    started = time.monotonic()
    state = {"seconds": 0.0, "moved": started}
    if watching:
        threading.Thread(target=_read_progress, args=(proc, state, on_progress), daemon=True).start()
    limit = STALL_SECONDS if stall_seconds is None else stall_seconds
    while proc.poll() is None:
        now = time.monotonic()
        if should_stop is not None and should_stop():
            _kill(proc)
            destination.unlink(missing_ok=True)
            raise ExtractionCancelled("cancelled")
        if not destination.exists() and now - started > start_deadline:
            _kill(proc)
            raise ExternalToolTimeout(f"{Path(cmd[0]).name} produced no output within {start_deadline:.0f} s")
        if watching and now - state["moved"] > limit:
            _kill(proc)
            raise ExternalToolTimeout(f"{Path(cmd[0]).name} made no progress for {limit:.0f} s")
        time.sleep(0.25)
    return subprocess.CompletedProcess(cmd, proc.returncode, "", proc.stderr.read() if proc.stderr else "")


def probe_media(path: Path, ffprobe: Optional[Path]) -> dict[str, Any]:
    """Duration and audio stream info via ffprobe (empty dict if ffprobe is unavailable)."""
    if ffprobe is None:
        return probe_media_pyav(path)
    cmd = [str(ffprobe), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)]
    try:
        proc = _run(cmd, timeout=40)
        data = json.loads(proc.stdout or "{}")
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, ExternalToolTimeout) as exc:
        log.warning("ffprobe failed for %s: %s; reading the file with PyAV instead", path.name, exc)
        return probe_media_pyav(path)
    audio_streams = [s for s in data.get("streams", []) if s.get("codec_type") == "audio"]
    info: dict[str, Any] = {
        "duration_s": float(data.get("format", {}).get("duration", 0) or 0),
        "container": data.get("format", {}).get("format_name"),
        "audio_streams": len(audio_streams),
    }
    if audio_streams:
        first = audio_streams[0]
        info.update(codec=first.get("codec_name"), sample_rate=first.get("sample_rate"),
                    channels=first.get("channels"))
    return info


def probe_media_pyav(path: Path) -> dict[str, Any]:
    """Duration / stream info with PyAV (the FFmpeg libraries inside the app). Empty dict if the file cannot be read."""
    try:
        import av

        with av.open(str(path)) as container:
            audio = next((s for s in container.streams if s.type == "audio"), None)
            seconds = float(container.duration) / 1_000_000.0 if container.duration else 0.0
            if not seconds and audio is not None and audio.duration and audio.time_base:
                seconds = float(audio.duration * audio.time_base)
            info: dict[str, Any] = {
                "duration_s": seconds, "container": container.format.name,
                "audio_streams": sum(1 for s in container.streams if s.type == "audio"), "probed_with": "pyav",
            }
            if audio is not None:
                info.update(codec=audio.codec_context.name, sample_rate=str(audio.codec_context.sample_rate),
                            channels=audio.codec_context.channels)
            return info
    except Exception as exc:  # noqa: BLE001 - probing is advisory
        log.warning("PyAV could not read %s: %s", Path(path).name, exc)
        return {}


def _kill(proc: "subprocess.Popen[str]") -> None:
    try:
        proc.kill()
    except OSError:
        pass


def extract_audio_pyav(source: Path, destination: Path, audio_cfg: dict[str, Any],
                       progress: Optional[Callable[[float], None]] = None, should_stop: Optional[Callable[[], bool]] = None) -> Path:
    """Decode the first audio track to a 16 kHz mono 16-bit WAV with PyAV, streaming (no giant arrays). Only that audio stream is
    demuxed and decoded: the video track is never touched.

    Used when ffmpeg.exe is missing, does not start or gets stuck. Same result as the FFmpeg path for ``preprocessing: none``.
    ``progress(seconds)`` reports how much has been decoded; ``should_stop()`` True ends it (``ExtractionCancelled``).
    """
    import av

    rate = int(audio_cfg.get("sample_rate", 16000))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        container = av.open(str(source))
    except Exception as exc:  # noqa: BLE001
        raise AudioExtractionError(f"Cannot open {Path(source).name}: {exc}") from exc
    with container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise AudioExtractionError(f"{Path(source).name} contains no audio track.")
        resampler = av.AudioResampler(format="s16", layout="mono", rate=rate)
        frames_written = 0
        last_report = 0.0
        try:
            with wave.open(str(destination), "wb") as out:
                out.setnchannels(1)
                out.setsampwidth(2)
                out.setframerate(rate)
                for packet in container.demux(stream):
                    if should_stop is not None and should_stop():
                        raise ExtractionCancelled("cancelled")
                    for frame in packet.decode():
                        for converted in resampler.resample(frame):
                            data = converted.to_ndarray().tobytes()
                            out.writeframes(data)
                            frames_written += len(data) // 2
                    if progress is not None and time.monotonic() - last_report >= 0.3:
                        last_report = time.monotonic()
                        progress(frames_written / float(rate))
                for converted in resampler.resample(None) or []:
                    data = converted.to_ndarray().tobytes()
                    out.writeframes(data)
                    frames_written += len(data) // 2
                if progress is not None:
                    progress(frames_written / float(rate))                    # the last position: what was really decoded
        except ExtractionCancelled:
            destination.unlink(missing_ok=True)
            raise
    if frames_written == 0:
        destination.unlink(missing_ok=True)
        raise AudioExtractionError(f"No audio could be decoded from {Path(source).name}.")
    return destination


def extract_audio(
    source: Path,
    destination: Path,
    audio_cfg: dict[str, Any],
    preprocessing: str = "none",
    ffmpeg: Optional[Path] = None,
    allow_fallback: bool = False,
    progress: Optional[Callable[[float], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Path:
    """Decode the first audio track of ``source`` to a 16 kHz mono PCM WAV.

    The original audio track is decoded directly (no re-recording, no intermediate lossy
    encode). Denoising is never applied; ``light`` only adds a high-pass and loudness filter.

    With ``allow_fallback`` (the desktop app) a missing, unstartable or failing ffmpeg.exe is not fatal: the track is
    decoded with PyAV instead (``light`` preprocessing is then skipped with a warning). An ffmpeg that stops making progress is
    abandoned the same way. ``progress(seconds)`` / ``should_stop()`` give real progress and a prompt cancel (``ExtractionCancelled``).
    """
    source = validate_media_file(source, audio_cfg)
    filters = audio_cfg.get("preprocessing_filters", {})
    if preprocessing not in filters:
        raise ValueError(f"Unknown preprocessing profile '{preprocessing}'. Choose from: {', '.join(filters)}")
    if ffmpeg is None and not allow_fallback:
        ffmpeg, _ = locate_ffmpeg(audio_cfg)
    if ffmpeg is None:
        return _extract_with_pyav(source, Path(destination), audio_cfg, preprocessing, "FFmpeg was not found", progress, should_stop)

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
        *(["-progress", "pipe:1", "-nostats"] if (progress is not None or should_stop is not None) else []),
        "-i", str(source),
        "-map", "0:a:0", "-vn", "-sn", "-dn",
    ]
    if filters[preprocessing]:
        cmd += ["-af", filters[preprocessing]]
    cmd += [
        "-ac", str(audio_cfg.get("channels", 1)),
        "-ar", str(audio_cfg.get("sample_rate", 16000)),
        "-c:a", str(audio_cfg.get("codec", "pcm_s16le")),
        str(destination),
    ]
    log.info("Extracting audio: %s -> %s (preprocessing=%s)", source.name, destination.name, preprocessing)
    log.debug("ffmpeg command: %s", cmd)
    try:
        proc = _run_extraction(cmd, destination, on_progress=progress, should_stop=should_stop)
    except (OSError, ExternalToolTimeout) as exc:
        if allow_fallback:
            return _extract_with_pyav(source, destination, audio_cfg, preprocessing, f"ffmpeg did not run ({exc})", progress, should_stop)
        if isinstance(exc, ExternalToolTimeout):
            raise AudioExtractionError(str(exc)) from exc
        raise FFmpegNotFoundError(f"{FFMPEG_INSTALL_HINT} ({exc})") from exc
    if proc.returncode != 0 or not destination.is_file() or destination.stat().st_size == 0:
        stderr = (proc.stderr or "").strip()
        if "matches no streams" in stderr or "Stream map" in stderr:
            raise AudioExtractionError(f"{source.name} contains no audio track.\n{stderr}")
        if allow_fallback and "Invalid data" not in stderr:
            return _extract_with_pyav(source, destination, audio_cfg, preprocessing, f"ffmpeg failed: {stderr[:120]}", progress, should_stop)
        raise AudioExtractionError(f"FFmpeg failed (exit {proc.returncode}) on {source.name}:\n{stderr}")
    return destination


def _extract_with_pyav(source: Path, destination: Path, audio_cfg: dict[str, Any], preprocessing: str, why: str,
                       progress: Optional[Callable[[float], None]] = None, should_stop: Optional[Callable[[], bool]] = None) -> Path:
    log.warning("%s; decoding %s with PyAV instead%s", why, source.name,
                "" if preprocessing == "none" else f" (the '{preprocessing}' filter is not applied)")
    return extract_audio_pyav(source, destination, audio_cfg, progress, should_stop)
