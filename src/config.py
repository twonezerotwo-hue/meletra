"""YAML configuration loading and validation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import yaml

from . import PROJECT_ROOT

log = logging.getLogger(__name__)

CONFIG_DIR = PROJECT_ROOT / "config"
TRANSCRIPTION_CONFIG = CONFIG_DIR / "transcription.yaml"
MODELS_CONFIG = CONFIG_DIR / "models.yaml"

REQUIRED_SECTIONS = (
    "defaults", "quality", "fast", "decoding", "vad", "audio", "context",
    "uncertainty", "confidence", "cleaner", "paths",
)
REQUIRED_PROFILE_KEYS = (
    "beam_size", "temperature", "compute_type", "vad_filter",
    "condition_on_previous_text", "word_timestamps",
)
REQUIRED_UNCERTAINTY_KEYS = (
    "avg_logprob_threshold", "no_speech_threshold", "compression_ratio_threshold",
)
VALID_ENGINES = ("faster_whisper", "transformers", "mlx_whisper", "configurable")
PROFILE_NAMES = ("quality", "fast")


class ConfigError(Exception):
    """Raised when a configuration file is missing or invalid."""


def load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML mapping from ``path`` (UTF-8)."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a YAML mapping at the top level")
    return data


def load_transcription_config(path: Optional[Path] = None) -> dict[str, Any]:
    """Load and validate ``config/transcription.yaml``."""
    cfg = load_yaml(path or TRANSCRIPTION_CONFIG)
    missing = [name for name in REQUIRED_SECTIONS if name not in cfg]
    if missing:
        raise ConfigError(f"transcription config is missing sections: {', '.join(missing)}")
    for profile in PROFILE_NAMES:
        absent = [key for key in REQUIRED_PROFILE_KEYS if key not in cfg[profile]]
        if absent:
            raise ConfigError(f"profile '{profile}' is missing keys: {', '.join(absent)}")
    absent = [key for key in REQUIRED_UNCERTAINTY_KEYS if key not in cfg["uncertainty"]]
    if absent:
        raise ConfigError(f"'uncertainty' section is missing keys: {', '.join(absent)}")
    if cfg["defaults"].get("mode") not in PROFILE_NAMES:
        raise ConfigError(f"defaults.mode must be one of {PROFILE_NAMES}")
    _reject_duplicate_top_level_keys(path or TRANSCRIPTION_CONFIG)
    for key in ("input", "outputs", "medical_terms"):
        if key not in cfg["paths"]:
            raise ConfigError(f"'paths' section is missing '{key}'")
        cfg["paths"][f"{key}_abs"] = str(resolve_path(cfg["paths"][key]))
    return cfg


def _reject_duplicate_top_level_keys(path: Path) -> None:
    """PyYAML silently lets a later duplicate key replace an earlier section; refuse that instead."""
    class _Strict(yaml.SafeLoader):
        pass

    def construct_mapping(loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        seen: set[Any] = set()
        for key_node, _ in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in seen:
                raise ConfigError(f"{path.name}: duplicate key '{key}' (the later one would silently replace the earlier)")
            seen.add(key)
        return yaml.SafeLoader.construct_mapping(loader, node, deep)

    _Strict.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping)
    with Path(path).open("r", encoding="utf-8") as handle:
        yaml.load(handle, Loader=_Strict)  # noqa: S506 - SafeLoader subclass


def load_models_config(path: Optional[Path] = None) -> dict[str, dict[str, Any]]:
    """Load and validate ``config/models.yaml``."""
    cfg = load_yaml(path or MODELS_CONFIG)
    if "default" not in cfg:
        raise ConfigError("models config must define a 'default' entry")
    for key, entry in cfg.items():
        if not isinstance(entry, dict):
            raise ConfigError(f"models entry '{key}' must be a mapping")
        engine = entry.get("engine")
        if engine not in VALID_ENGINES:
            raise ConfigError(
                f"models entry '{key}' has engine {engine!r}; expected one of {VALID_ENGINES}"
            )
        if engine != "configurable" and not entry.get("model"):
            raise ConfigError(f"models entry '{key}' needs a 'model' value")
    return cfg


def resolve_path(value: str | Path, root: Path = PROJECT_ROOT) -> Path:
    """Resolve a config path: relative paths are anchored at the project root."""
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()
