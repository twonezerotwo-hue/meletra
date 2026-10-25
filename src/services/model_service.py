"""Model manager: which speech models exist, which are installed, download (resumable), verify, delete, import.

The registry (``config/app_models.json``) pins every model to a Hugging Face revision and lists the SHA-256 of every file, so a
download - or a folder copied from another PC - can be verified byte for byte without asking any server. The installer stays
small because no model is inside it; the first run downloads the recommended one into
``%LOCALAPPDATA%\\MedicalLectureTranscriber\\models``.
"""

from __future__ import annotations

import hashlib
import http.client
import importlib.util
import json
import logging
import os
import shutil
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Optional

from .. import RESOURCE_ROOT
from ..model_manager import ModelSpec, get_model_spec, load_model_specs
from ..platform import Platform, current
from .paths import AppPaths

log = logging.getLogger(__name__)

REGISTRY_FILE = RESOURCE_ROOT / "config" / "app_models.json"
DEFAULT_BASE_URL = "https://huggingface.co"
TRANSLATION_PACK_KEY = "translation-m2m100"        # the optional offline translation pack (registry entry kind "translation", hidden from the speech-model lists)
MANIFEST = "manifest.json"
CHUNK = 1024 * 1024
DISK_MARGIN = 300 * 1024 * 1024

ProgressCallback = Callable[[dict[str, Any]], None]
CancelCheck = Callable[[], bool]


class ModelError(RuntimeError):
    """Base class of model-manager errors; the message is meant for the user."""


class ModelNotInstalled(ModelError):
    pass


class ModelDownloadError(ModelError):
    pass


class ModelIntegrityError(ModelError):
    pass


class ModelCancelled(ModelError):
    """The user cancelled the download. Partial files are kept so the next attempt resumes."""


@dataclass(frozen=True)
class ModelFile:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class ModelInfo:
    key: str
    title: str
    repo: str
    revision: str
    engine: str
    models_yaml_key: str
    folder: str
    description: str
    runtime: str
    files: tuple[ModelFile, ...]
    required: tuple[str, ...]
    approx_bytes: int
    recommended: bool = False
    hidden: bool = False
    license: str = ""
    platforms: tuple[str, ...] = ()             # where the model can be used: windows | macos | linux; () = everywhere
    recommended_on: tuple[str, ...] = ()        # platform keys that recommend it: windows | macos-arm64 | macos-x86_64 | linux; () = ``recommended``
    kind: str = "asr"                           # asr (a speech model) | translation (the offline translation pack: never offered as a speech model)
    source_url: str = ""                        # where the files are downloaded from when it is not Hugging Face: a URL with {path} for the file name

    def file(self, path: str) -> ModelFile:
        return next(f for f in self.files if f.path == path)

    def available_on(self, platform_key: str) -> bool:
        return not self.platforms or platform_key in self.platforms or platform_key.split("-")[0] in self.platforms

    def recommended_for(self, platform_key: str) -> bool:
        return platform_key in self.recommended_on if self.recommended_on else self.recommended


@dataclass
class ModelState:
    key: str
    title: str
    installed: bool
    path: Optional[Path] = None
    size_on_disk: int = 0
    verified: Optional[bool] = None            # None = never fully verified
    partial_bytes: int = 0                     # a download that can be resumed
    runtime_ok: bool = True
    runtime_note: str = ""
    source: str = ""                           # app | portable | extra | cache (found in place)
    issues: list[str] = field(default_factory=list)


@dataclass
class VerifyResult:
    ok: bool
    problems: list[str] = field(default_factory=list)
    checked_bytes: int = 0


def load_registry(path: Path = REGISTRY_FILE) -> dict[str, ModelInfo]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    registry: dict[str, ModelInfo] = {}
    for entry in data["models"]:
        files = tuple(ModelFile(f["path"], int(f["size"]), f["sha256"]) for f in entry["files"])
        registry[entry["key"]] = ModelInfo(
            key=entry["key"], title=entry["title"], repo=entry["repo"], revision=entry["revision"], engine=entry["engine"],
            models_yaml_key=entry["models_yaml_key"], folder=entry["folder"], description=entry["description"],
            runtime=entry.get("runtime", ""), files=files, required=tuple(entry["required"]),
            approx_bytes=int(entry.get("approx_bytes", sum(f.size for f in files))),
            recommended=bool(entry.get("recommended")), hidden=bool(entry.get("hidden")), license=entry.get("license", ""),
            platforms=tuple(entry.get("platforms") or ()), recommended_on=tuple(entry.get("recommended_on") or ()),
            kind=str(entry.get("kind") or "asr"), source_url=str(entry.get("source_url") or ""))
    return registry


def platform_key(platform: Platform) -> str:
    """windows | macos-arm64 | macos-x86_64 | linux: the key the registry's ``recommended_on`` uses."""
    if platform.name == "macos":
        return "macos-arm64" if platform.is_apple_silicon else "macos-x86_64"
    return platform.name


def sha256_file(path: Path, progress: Optional[Callable[[int], None]] = None) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(8 * CHUNK):
            digest.update(block)
            if progress is not None:
                progress(len(block))
    return digest.hexdigest()


def folder_size(path: Path) -> int:
    total = 0
    for root, _dirs, names in os.walk(path):
        for name in names:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return total


def human_bytes(value: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


class ModelService:
    def __init__(self, paths: AppPaths, registry: Optional[dict[str, ModelInfo]] = None,
                 extra_dirs: Optional[list[str]] = None, app_dir: Optional[Path] = None, base_url: str = DEFAULT_BASE_URL,
                 opener: Optional[Callable[..., Any]] = None, sleep: Callable[[float], None] = time.sleep,
                 retries: int = 5, platform: Optional[Platform] = None, discover: bool = False) -> None:
        self._platform_override = platform
        self.discover = discover                    # also use a complete copy found in the Hugging Face cache, in place (no copy)
        self.paths = paths
        self.registry = registry if registry is not None else load_registry()
        self.extra_dirs = list(extra_dirs or [])
        self.app_dir = app_dir
        self.base_url = base_url.rstrip("/")
        self._open = opener or urllib.request.urlopen
        self._sleep = sleep
        self.retries = retries

    # ------------------------------------------------------------------ platform
    @property
    def platform(self) -> Platform:
        return self._platform_override or current()

    def platform_key(self) -> str:
        return platform_key(self.platform)

    def available(self, key: str) -> bool:
        """Can this model be used on this computer's operating system / processor family at all?"""
        return self.info(key).available_on(self.platform_key())

    def is_recommended(self, key: str) -> bool:
        return self.info(key).recommended_for(self.platform_key())

    # ------------------------------------------------------------------ discovery
    def info(self, key: str) -> ModelInfo:
        try:
            return self.registry[key]
        except KeyError:
            raise ModelError(f"Unknown model '{key}'. Known models: {', '.join(self.registry)}") from None

    def search_dirs(self) -> list[Path]:
        return self.paths.model_search_dirs(self.extra_dirs, self.app_dir)

    def _quick_check(self, folder: Path, info: ModelInfo) -> list[str]:
        """Missing / wrong-size required files (fast: no hashing)."""
        problems = []
        for name in info.required:
            path = folder / name
            if not path.is_file():
                problems.append(f"{name} is missing")
            elif path.stat().st_size != info.file(name).size:
                problems.append(f"{name} has the wrong size")
        return problems

    def locate(self, key: str) -> Optional[Path]:
        info = self.info(key)
        for base in self.search_dirs():
            folder = base / info.folder
            if folder.is_dir() and not self._quick_check(folder, info):
                return folder
        return self.find_hf_cache_copy(key) if self.discover else None

    def is_installed(self, key: str) -> bool:
        return self.locate(key) is not None

    @staticmethod
    def _is_bundle_models(folder: Path) -> bool:
        """``<App>.app/Contents/Resources/models``: where the macOS offline release ships its speech model."""
        return folder.name == "models" and folder.parent.name == "Resources" and folder.parent.parent.name == "Contents"

    def bundled_problems(self, key: str) -> list[str]:
        """Problems of a copy of this model that SHIPS INSIDE the app (macOS offline release); [] when none ships or it is fine.
        A damaged bundled model is reported, never "fixed" by a download: the release is meant to work without a network."""
        info = self.info(key)
        for base in self.search_dirs():
            if self._is_bundle_models(base) and (base / info.folder).is_dir():
                return self._quick_check(base / info.folder, info)
        return []

    def _partial_dir(self, info: ModelInfo) -> Path:
        return self.paths.models / ".partial" / info.folder

    def partial_bytes(self, key: str) -> int:
        folder = self._partial_dir(self.info(key))
        return folder_size(folder) if folder.is_dir() else 0

    def runtime_status(self, info: ModelInfo) -> tuple[bool, str]:
        if info.engine == "mlx_whisper":
            if not self.platform.is_apple_silicon:
                return False, "Needs a Mac with Apple Silicon (M1 or newer)."
            if importlib.util.find_spec("mlx_whisper") is None:
                return False, "The optimized macOS speech engine (mlx-whisper) is not part of this build."
        if info.engine == "transformers":
            if importlib.util.find_spec("torch") is None or importlib.util.find_spec("transformers") is None:
                return False, "Needs the optional PyTorch + Transformers runtime, which is not part of this build."
        return True, ""

    def state(self, key: str) -> ModelState:
        info = self.info(key)
        folder = self.locate(key)
        ok, note = self.runtime_status(info)
        state = ModelState(key=key, title=info.title, installed=folder is not None, runtime_ok=ok, runtime_note=note,
                           partial_bytes=0 if folder else self.partial_bytes(key))
        if folder is not None:
            state.path, state.size_on_disk = folder, folder_size(folder)
            state.source = "app" if folder.parent == self.paths.models else ("portable" if folder.parent == (
                self.app_dir or folder.parent) / "models" else ("cache" if folder.parent.name == "snapshots" else (
                    "bundled" if self._is_bundle_models(folder.parent) else "extra")))
            manifest = self._read_manifest(folder)
            state.verified = manifest.get("verified") if manifest else None
        return state

    def list_models(self, include_hidden: bool = False, all_platforms: bool = False, kind: Optional[str] = "asr") -> list[ModelState]:
        """The models offered on this computer (a model made for another operating system is not listed). ``kind``: "asr" = speech models (the default: every place that lets the
        student choose a speech model), "translation" = the offline translation pack, None = both (Settings -> Models)."""
        pkey = self.platform_key()
        return [self.state(k) for k, m in self.registry.items()
                if (include_hidden or not m.hidden) and (all_platforms or m.available_on(pkey)) and (kind is None or m.kind == kind)]

    def recommended_key(self) -> str:
        pkey = self.platform_key()
        usable = [k for k, m in self.registry.items() if m.available_on(pkey) and not m.hidden and m.kind == "asr"]
        return next((k for k in usable if self.registry[k].recommended_for(pkey)), usable[0] if usable else next(iter(self.registry)))

    def preferred_key(self) -> str:
        """The model to use when the student did not choose one: the recommended model - or, to avoid a second multi-gigabyte
        download, a model that is already installed and can run here."""
        recommended = self.recommended_key()
        if self.is_installed(recommended) and self.state(recommended).runtime_ok:
            return recommended
        for state in self.list_models():
            info = self.registry[state.key]
            if state.installed and state.runtime_ok and (info.recommended or info.recommended_on):   # a general-purpose model, not a specialist
                return state.key
        return recommended

    # ------------------------------------------------------------------ spec for the pipeline
    def model_spec(self, key: str) -> ModelSpec:
        info = self.info(key)
        folder = self.locate(key)
        if folder is None:
            raise ModelNotInstalled(f"{info.title} is not installed. Open Settings -> Models to download or import it.")
        ok, note = self.runtime_status(info)
        if not ok:
            raise ModelError(f"{info.title}: {note}")
        base = get_model_spec(info.models_yaml_key, load_model_specs())
        return replace(base, key=info.key, model=str(folder))

    # ------------------------------------------------------------------ manifest
    def _read_manifest(self, folder: Path) -> Optional[dict[str, Any]]:
        try:
            return json.loads((folder / MANIFEST).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _write_manifest(self, folder: Path, info: ModelInfo, source: str, verified: bool) -> None:
        payload = {"key": info.key, "revision": info.revision, "source": source, "verified": verified,
                   "verified_at": time.strftime("%Y-%m-%dT%H:%M:%S") if verified else None,
                   "files": {f.path: {"size": f.size, "sha256": f.sha256} for f in info.files}}
        (folder / MANIFEST).write_text(json.dumps(payload, indent=1), encoding="utf-8")

    # ------------------------------------------------------------------ download
    def download(self, key: str, progress: Optional[ProgressCallback] = None, cancel: Optional[CancelCheck] = None) -> Path:
        """Download (or resume downloading) a model into the app's models folder and verify every file."""
        info = self.info(key)
        existing = self.locate(key)
        if existing is not None and existing.parent == self.paths.models:
            return existing
        final = self.paths.models / info.folder
        partial = self._partial_dir(info)
        partial.mkdir(parents=True, exist_ok=True)
        total = sum(f.size for f in info.files)
        have = self.partial_bytes(key)
        free = shutil.disk_usage(self.paths.models).free
        if free < max(total - have, 0) + DISK_MARGIN:
            raise ModelDownloadError(
                f"Not enough free disk space for {info.title}: it needs about {human_bytes(total - have)} more, "
                f"{human_bytes(free)} are free on the drive that holds {self.paths.models}.")
        tracker = _Progress(info, total, progress)
        for index, spec in enumerate(info.files, start=1):
            tracker.start_file(spec.path, index, len(info.files))
            self._fetch_file(info, spec, partial / spec.path, tracker, cancel)
        # all files complete and verified: publish atomically
        if final.exists():
            shutil.rmtree(final, ignore_errors=True)
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(partial, final)
        self._write_manifest(final, info, "download", verified=True)
        tracker.finished()
        log.info("Model %s installed in %s", info.title, final)
        return final

    def _url(self, info: ModelInfo, path: str) -> str:
        if info.source_url:
            return info.source_url.replace("{path}", path).replace("{revision}", info.revision)
        return f"{self.base_url}/{info.repo}/resolve/{info.revision}/{path}"

    def _fetch_file(self, info: ModelInfo, spec: ModelFile, part_path: Path, tracker: "_Progress",
                    cancel: Optional[CancelCheck]) -> None:
        part_path.parent.mkdir(parents=True, exist_ok=True)
        part = part_path.with_name(part_path.name + ".part")
        if part_path.is_file() and part_path.stat().st_size == spec.size and sha256_file(part_path) == spec.sha256:
            tracker.add(spec.size)                                       # finished in an earlier attempt
            return
        attempt = 0
        while True:
            attempt += 1
            try:
                self._stream(info, spec, part, tracker, cancel)
                break
            except ModelCancelled:
                raise
            except (urllib.error.URLError, ConnectionError, TimeoutError, http.client.HTTPException, OSError) as exc:
                if isinstance(exc, urllib.error.HTTPError) and exc.code in (401, 403, 404):
                    raise ModelDownloadError(f"The download server refused {spec.path} (HTTP {exc.code}).") from exc
                if attempt >= self.retries:
                    raise ModelDownloadError(
                        f"Download interrupted ({type(exc).__name__}). Your progress is kept: press Download again to resume.") from exc
                wait = min(2.0 * attempt, 15.0)
                log.warning("Download of %s interrupted (%s); retry %d/%d in %.0fs", spec.path, exc, attempt, self.retries, wait)
                tracker.message(f"Connection problem, retrying in {wait:.0f} s ...")
                self._sleep(wait)
        if part.stat().st_size != spec.size:
            raise ModelIntegrityError(f"{spec.path}: the download ended with {part.stat().st_size} bytes, expected {spec.size}.")
        tracker.message(f"Verifying {spec.path} ...")
        if sha256_file(part) != spec.sha256:
            part.unlink(missing_ok=True)
            raise ModelIntegrityError(f"{spec.path}: checksum mismatch - the file was damaged in transit. Press Download again.")
        os.replace(part, part_path)

    def _stream(self, info: ModelInfo, spec: ModelFile, part: Path, tracker: "_Progress", cancel: Optional[CancelCheck]) -> None:
        have = part.stat().st_size if part.exists() else 0
        if have > spec.size:
            part.unlink()
            have = 0
        if have == spec.size:
            tracker.add(spec.size - tracker.file_done)
            return
        headers = {"Range": f"bytes={have}-", "User-Agent": "MedicalLectureTranscriber/0.1"} if have else {
            "User-Agent": "MedicalLectureTranscriber/0.1"}
        request = urllib.request.Request(self._url(info, spec.path), headers=headers)
        with self._open(request, timeout=30) as response:
            status = getattr(response, "status", 200)
            if have and status != 206:                                   # the server ignored the Range header: start over
                have = 0
            tracker.add(have - tracker.file_done)
            mode = "ab" if have else "wb"
            with part.open(mode) as out:
                while True:
                    if cancel is not None and cancel():
                        raise ModelCancelled("Download cancelled. Your progress is kept.")
                    block = response.read(CHUNK)
                    if not block:
                        break
                    out.write(block)
                    tracker.add(len(block))
        if part.stat().st_size < spec.size:               # the server closed the connection early: retry from here
            raise ConnectionError(f"connection closed after {part.stat().st_size} of {spec.size} bytes")

    # ------------------------------------------------------------------ verify / delete / import
    def verify(self, key: str, progress: Optional[ProgressCallback] = None) -> VerifyResult:
        info = self.info(key)
        folder = self.locate(key)
        if folder is None:
            return VerifyResult(False, [f"{info.title} is not installed."])
        problems: list[str] = []
        done = 0
        total = sum(f.size for f in info.files if (folder / f.path).is_file())
        for spec in info.files:
            path = folder / spec.path
            if not path.is_file():
                if spec.path in info.required:
                    problems.append(f"{spec.path} is missing")
                continue
            if path.stat().st_size != spec.size:
                problems.append(f"{spec.path} has the wrong size")
                continue

            def tick(n: int, name: str = spec.path) -> None:
                nonlocal done
                done += n
                if progress is not None:
                    progress({"type": "model_verify_progress", "model": key, "file": name,
                              "percent": 100.0 * done / total if total else 100.0})

            if sha256_file(path, tick) != spec.sha256:
                problems.append(f"{spec.path} does not match its checksum")
        result = VerifyResult(not problems, problems, done)
        if folder.is_dir() and not self._is_bundle_models(folder.parent):     # never write into a signed app bundle
            try:
                self._write_manifest(folder, info, (self._read_manifest(folder) or {}).get("source", "unknown"), result.ok)
            except OSError:
                pass
        return result

    def delete(self, key: str) -> bool:
        """Delete an installed model. Only models inside the app's own models folder are ever removed."""
        info = self.info(key)
        folder = self.paths.models / info.folder
        removed = False
        for target in (folder, self._partial_dir(info)):
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
                removed = True
        return removed

    def hf_cache_roots(self) -> list[Path]:
        """Folders where Hugging Face tools keep downloaded models (``models--<org>--<name>/snapshots/<revision>``)."""
        env, home = self.platform.env, Path.home()
        roots = [home / ".cache" / "huggingface" / "hub"]
        roots += [Path(env[n]) for n in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE") if env.get(n)]
        if env.get("HF_HOME"):
            roots += [Path(env["HF_HOME"]) / "hub", Path(env["HF_HOME"])]
        if self.platform.name == "macos":
            roots += [home / "Library" / "Caches" / "huggingface" / "hub", home / "Library" / "Caches"]
        unique: list[Path] = []
        for root in roots:
            if root not in unique and root.is_dir():
                unique.append(root)
        return unique

    def find_hf_cache_copy(self, key: str) -> Optional[Path]:
        """A complete, readable copy of this model in a Hugging Face cache (the pinned revision first), or None. Used in place."""
        info = self.info(key)
        for root in self.hf_cache_roots():
            snapshots = root / ("models--" + info.repo.replace("/", "--")) / "snapshots"
            if not snapshots.is_dir():
                continue
            found = [p for p in snapshots.iterdir() if p.is_dir()]
            for snapshot in sorted(found, key=lambda p: (p.name != info.revision, p.name)):
                if not self._quick_check(snapshot, info) and self._readable(snapshot / "config.json"):
                    return snapshot
        return None

    @staticmethod
    def _readable(config: Path) -> bool:
        try:
            json.loads(config.read_text(encoding="utf-8"))
            return True
        except (OSError, ValueError):
            return False

    def import_folder(self, key: str, source: Path, progress: Optional[ProgressCallback] = None,
                      verify: bool = True) -> tuple[Path, VerifyResult]:
        """Copy a model folder (from a USB stick, another PC, the hub cache ...) into the app's models folder."""
        info = self.info(key)
        source = Path(source)
        if not source.is_dir():
            raise ModelError(f"'{source}' is not a folder.")
        candidates = [source, *[p for p in source.iterdir() if p.is_dir()]]
        folder = next((c for c in candidates if not self._quick_check(c, info)), None)
        if folder is None:
            missing = self._quick_check(source, info)
            raise ModelError(f"This folder does not contain {info.title}: {'; '.join(missing[:3])}.")
        final = self.paths.models / info.folder
        staging = self.paths.models / ".partial" / f"import-{info.folder}"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True, exist_ok=True)
        total = sum(p.stat().st_size for p in folder.iterdir() if p.is_file())
        copied = 0
        for item in folder.iterdir():
            if item.is_file() and item.name != MANIFEST:
                shutil.copy2(item, staging / item.name)
                copied += item.stat().st_size
                if progress is not None:
                    progress({"type": "model_import_progress", "model": key, "file": item.name,
                              "percent": 100.0 * copied / total if total else 100.0})
        shutil.rmtree(final, ignore_errors=True)
        os.replace(staging, final)
        self._write_manifest(final, info, "import", verified=False)
        result = self.verify(key) if verify else VerifyResult(True, [])
        return final, result

    # ------------------------------------------------------------------ disk
    def disk_usage(self) -> dict[str, Any]:
        per_model = {k: folder_size(self.paths.models / m.folder) for k, m in self.registry.items()
                     if (self.paths.models / m.folder).is_dir()}
        return {"models_bytes": sum(per_model.values()), "per_model": per_model,
                "partial_bytes": folder_size(self.paths.models / ".partial") if (self.paths.models / ".partial").is_dir() else 0,
                "free_bytes": shutil.disk_usage(self.paths.models).free}


class _Progress:
    """Aggregates per-chunk byte counts into throttled progress events."""

    def __init__(self, info: ModelInfo, total: int, callback: Optional[ProgressCallback]) -> None:
        self.info, self.total, self.callback = info, total, callback
        self.done = 0
        self.file_done = 0
        self.file = ""
        self.index = self.count = 0
        self.started = time.monotonic()
        self._last = 0.0

    def start_file(self, name: str, index: int, count: int) -> None:
        self.file, self.index, self.count, self.file_done = name, index, count, 0
        self._emit(force=True)

    def add(self, nbytes: int) -> None:
        self.done += nbytes
        self.file_done += nbytes
        self._emit()

    def message(self, text: str) -> None:
        if self.callback is not None:
            self.callback({"type": "model_download_progress", "model": self.info.key, "file": self.file, "message": text,
                           "bytes_done": self.done, "bytes_total": self.total,
                           "percent": 100.0 * self.done / self.total if self.total else 0.0})

    def finished(self) -> None:
        self._emit(force=True)

    def _emit(self, force: bool = False) -> None:
        now = time.monotonic()
        if self.callback is None or (not force and now - self._last < 0.25):
            return
        self._last = now
        elapsed = max(now - self.started, 1e-6)
        speed = self.done / elapsed
        remaining = max(self.total - self.done, 0)
        self.callback({"type": "model_download_progress", "model": self.info.key, "file": self.file,
                       "file_index": self.index, "files": self.count, "bytes_done": self.done, "bytes_total": self.total,
                       "percent": 100.0 * self.done / self.total if self.total else 0.0, "speed_bps": speed,
                       "eta_seconds": remaining / speed if speed > 0 else None})
