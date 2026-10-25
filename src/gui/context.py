"""The services every window shares, created once. Tests build one with temporary folders and fake open/ask functions."""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from ..services.credential_service import CredentialService
from ..services.gpu_service import GpuService
from ..services.job_service import JobStore
from ..services.model_service import ModelService
from ..services.paths import AppPaths, get_paths
from ..services.settings_service import SettingsService
from ..services.lecture_service import LectureService
from ..services.study_service import StudyService
from ..sharing.service import SharingService


def default_open_path(path: Path) -> None:
    """Open a file or folder with the program the operating system associates with it."""
    from ..platform import current

    current().open_path(Path(path))


@dataclass
class AppServices:
    paths: AppPaths
    settings: SettingsService
    credentials: CredentialService
    jobs: JobStore
    models: ModelService
    gpu: GpuService
    open_path: Callable[[Path], None] = default_open_path
    log: logging.Logger = field(default_factory=lambda: logging.getLogger("gui"))
    study: Optional[StudyService] = None                      # AI Study (0.2): summaries, flashcards, questions; separate from transcription
    lectures: Optional[LectureService] = None                 # a finished lecture by its id: translate its saved transcript (text only - no recording, no job)
    sharing: Optional[SharingService] = None                  # Settings -> Privacy & Data: sharing of completed transcripts (off by default, opt-in; switch-off clears the queue)

    def __post_init__(self) -> None:
        if self.study is None:
            self.study = StudyService(self.settings, self.credentials)
        if self.lectures is None:
            self.lectures = LectureService(self.jobs, lambda: self.models.search_dirs())
        if self.sharing is None:
            self.sharing = SharingService(self.settings, self.paths.root / "sharing")
            self.sharing.reconcile()                              # what is stored and what happens are the same thing from the first moment

    @classmethod
    def create(cls, env: Optional[Mapping[str, str]] = None, app_dir: Optional[Path] = None,
               credentials: Optional[CredentialService] = None, gpu: Optional[GpuService] = None,
               models: Optional[ModelService] = None, open_path: Callable[[Path], None] = default_open_path) -> "AppServices":
        paths = get_paths(env, app_dir)
        settings = SettingsService(paths.settings_file)
        model_service = models or ModelService(paths, extra_dirs=settings.settings.extra_model_dirs, app_dir=app_dir, discover=True)
        services = cls(paths=paths, settings=settings, credentials=credentials or CredentialService(),
                       jobs=JobStore(paths.jobs), models=model_service, gpu=gpu or GpuService(), open_path=open_path)
        services.apply_platform_defaults()
        return services

    def apply_platform_defaults(self) -> None:
        """Before the first run: if the platform recommends a different model than the settings default (an Apple Silicon Mac: the MLX
        Large-v3), use it. Never changes a choice the student made (after the first run), and does nothing on Windows."""
        s = self.settings.settings
        if s.first_run_done or s.model_key != "whisper-large-v3":
            return
        preferred = self.models.preferred_key()
        if preferred != s.model_key:
            self.update_setting(model_key=preferred)

    def sync_model_dirs(self) -> None:
        """Settings -> Models can add search folders; the model service must see them."""
        self.models.extra_dirs = list(self.settings.settings.extra_model_dirs)

    def update_setting(self, **changes: Any) -> None:
        self.settings.update(**changes)
        self.sync_model_dirs()


def configure_logging(paths: AppPaths, level: int = logging.INFO) -> None:
    """App log in the user's logs folder; the previous run's log is kept as app.previous.log (crash investigation)."""
    log_file = paths.app_log
    previous = log_file.with_name("app.previous.log")
    try:
        if log_file.exists():
            if previous.exists():
                previous.unlink()
            log_file.replace(previous)
    except OSError:
        pass
    root = logging.getLogger()
    root.setLevel(level)
    handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    root.addHandler(handler)
    for noisy in ("httpx", "httpcore", "urllib3", "filelock", "numba", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
