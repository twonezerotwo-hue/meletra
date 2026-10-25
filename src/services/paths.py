"""Where the desktop app keeps its files.

Nothing mutable is ever written below the install folder (Program Files is read-only for normal users).

    %LOCALAPPDATA%\\MedicalLectureTranscriber\\
        models\\        downloaded / imported models
        cache\\         disposable caches
        checkpoints\\   one folder per unfinished transcription (chunk checkpoints)
        jobs\\          one small JSON per job (state shown in the "unfinished jobs" list)
        logs\\          app.log and per-job logs
        settings\\      settings.json
        temp\\          extracted 16 kHz WAV files while a job runs

The transcripts themselves go to the folder the user chose. Portable mode: a file named ``portable.flag`` next to the
executable keeps all of the above in ``<app folder>\\data`` instead. ``MLT_DATA_DIR`` overrides everything (tests, power users).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

from .. import APP_DIR, APP_ID
from ..platform import Platform, current

SUBFOLDERS = ("models", "cache", "checkpoints", "jobs", "logs", "settings", "temp")
PORTABLE_FLAG = "portable.flag"


def _override_or_portable(env: Mapping[str, str], app_dir: Optional[Path]) -> Optional[Path]:
    override = env.get("MLT_DATA_DIR")
    if override:
        return Path(override)
    app_dir = Path(app_dir) if app_dir is not None else APP_DIR
    if (app_dir / PORTABLE_FLAG).is_file():
        return app_dir / "data"
    return None


def resolve_user_data_root(env: Optional[Mapping[str, str]] = None, app_dir: Optional[Path] = None,
                          platform: Optional[Platform] = None) -> Path:
    """Windows: ``%LOCALAPPDATA%\\MedicalLectureTranscriber``; macOS: ``~/Library/Application Support/MedicalLectureTranscriber``."""
    env = os.environ if env is None else env
    fixed = _override_or_portable(env, app_dir)
    if fixed is not None:
        return fixed
    platform = platform or current()
    if env is not platform.env:                       # the caller's environment wins over the platform's own
        platform = type(platform)(env=env, home=platform.home, run=platform._run, machine=platform.machine)
    return platform.user_data_dir()


@dataclass(frozen=True)
class AppPaths:
    root: Path
    cache_root: Optional[Path] = None                 # macOS: ~/Library/Caches/...; None = <root>/cache
    logs_root: Optional[Path] = None                  # macOS: ~/Library/Logs/...;   None = <root>/logs

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def cache(self) -> Path:
        return self.cache_root if self.cache_root is not None else self.root / "cache"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def jobs(self) -> Path:
        return self.root / "jobs"

    @property
    def logs(self) -> Path:
        return self.logs_root if self.logs_root is not None else self.root / "logs"

    @property
    def settings(self) -> Path:
        return self.root / "settings"

    @property
    def temp(self) -> Path:
        return self.root / "temp"

    @property
    def settings_file(self) -> Path:
        return self.settings / "settings.json"

    @property
    def app_log(self) -> Path:
        return self.logs / "app.log"

    def job_log(self, job_id: str) -> Path:
        return self.logs / f"job-{job_id}.log"

    def ensure(self) -> "AppPaths":
        for name in SUBFOLDERS:
            (self.root / name).mkdir(parents=True, exist_ok=True)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        return self

    def model_search_dirs(self, extra: Optional[list[str]] = None, app_dir: Optional[Path] = None) -> list[Path]:
        """Folders searched for installed models, in order: user models, the app's own ``models`` folder (portable and
        full-offline packages), then any folder the user added in Settings -> Models."""
        base = Path(app_dir if app_dir is not None else APP_DIR)
        folders = [self.models, base / "models"]
        if base.name == "MacOS" and base.parent.name == "Contents":      # inside "<name>.app": the offline release keeps its model in Resources
            folders.append(base.parent / "Resources" / "models")
        folders += [Path(p) for p in (extra or []) if p]
        seen: list[Path] = []
        for folder in folders:
            if folder not in seen:
                seen.append(folder)
        return seen


def get_paths(env: Optional[Mapping[str, str]] = None, app_dir: Optional[Path] = None, create: bool = True,
              platform: Optional[Platform] = None) -> AppPaths:
    """The folders of this run. Cache and logs live in the operating system's own places where it has them (macOS); an override
    (``MLT_DATA_DIR``) or portable mode keeps everything below one folder."""
    env = os.environ if env is None else env
    fixed = _override_or_portable(env, app_dir)
    if fixed is not None:
        paths = AppPaths(fixed)
    else:
        platform = platform or current()
        if env is not platform.env:
            platform = type(platform)(env=env, home=platform.home, run=platform._run, machine=platform.machine)
        paths = AppPaths(platform.user_data_dir(), platform.cache_dir(), platform.logs_dir())
    return paths.ensure() if create else paths


def default_output_dir(source: Path) -> Path:
    """Next to the source file: ``<folder>/Medical Transcription``. The pipeline adds one sub-folder per recording."""
    return Path(source).resolve().parent / "Medical Transcription"
