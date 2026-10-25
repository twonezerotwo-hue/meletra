import pytest
import yaml

from src.config import ConfigError, load_models_config, load_transcription_config, load_yaml
from src.model_manager import (
    FasterWhisperAdapter, ModelNotConfiguredError, TransformersAdapter, UnknownModelError, create_adapter,
    get_model_spec, load_model_specs,
)
from src.uncertainty import UncertaintyConfig


def test_transcription_config_loads(cfg):
    for section in ("quality", "fast", "uncertainty", "vad", "audio", "context", "cleaner", "paths"):
        assert section in cfg
    assert cfg["paths"]["outputs_abs"].endswith("outputs")


def test_quality_and_fast_profiles_match_spec(cfg):
    assert cfg["quality"]["beam_size"] == 5
    assert cfg["quality"]["compute_type"] == "float16"
    assert cfg["quality"]["fallback_compute_type"] == "int8_float16"
    assert cfg["quality"]["vad_filter"] is True
    assert cfg["quality"]["condition_on_previous_text"] is True
    assert cfg["quality"]["word_timestamps"] is True
    assert cfg["quality"]["temperature"] == 0
    assert cfg["fast"]["beam_size"] == 3
    assert cfg["fast"]["compute_type"] == "int8_float16"
    assert cfg["fast"]["vad_filter"] is True


def test_uncertainty_thresholds_come_from_yaml(cfg):
    assert cfg["uncertainty"]["avg_logprob_threshold"] == -1.0
    assert cfg["uncertainty"]["no_speech_threshold"] == 0.6
    assert cfg["uncertainty"]["compression_ratio_threshold"] == 2.4
    unc = UncertaintyConfig.from_config(cfg)
    assert unc.avg_logprob_threshold == cfg["uncertainty"]["avg_logprob_threshold"]


def test_default_mode_is_valid(cfg):
    assert cfg["defaults"]["mode"] in ("quality", "fast")
    assert cfg["audio"]["preprocessing_filters"]["none"] is None  # denoise OFF by default
    assert cfg["defaults"]["preprocessing"] == "none"


def test_models_config_loads():
    models = load_models_config()
    assert models["default"]["engine"] == "faster_whisper"
    assert models["default"]["model"] == "large-v3"
    assert models["turkish_medical"]["engine"] == "transformers"
    assert models["turkish_medical"]["model"] == "turkmedstt/whisper-large-v3-turkish-medical"
    assert models["english_medical"]["engine"] == "configurable"
    assert models["english_medical"]["model"] is None


def test_cpu_run_uses_int8_never_float16(cfg):
    from src.transcriber import Transcriber

    spec = get_model_spec("default")
    assert Transcriber(spec, cfg["quality"], device="cpu").compute_types == ["int8"]
    assert Transcriber(spec, cfg["quality"], device="cuda").compute_types == ["float16", "int8_float16", "int8"]


def test_adapter_selection_per_engine():
    specs = load_model_specs()
    assert isinstance(create_adapter(specs["default"]), FasterWhisperAdapter)
    assert isinstance(create_adapter(specs["turkish_medical"]), TransformersAdapter)


def test_unconfigured_model_is_refused_clearly():
    spec = get_model_spec("english_medical")
    with pytest.raises(ModelNotConfiguredError, match="config/models.yaml"):
        create_adapter(spec)


def test_unknown_model_key_lists_choices():
    with pytest.raises(UnknownModelError, match="default"):
        get_model_spec("nope")


def test_missing_section_is_reported(tmp_path, cfg):
    broken = {k: v for k, v in yaml.safe_load(open("config/transcription.yaml", encoding="utf-8")).items()
              if k != "uncertainty"}
    path = tmp_path / "broken.yaml"
    path.write_text(yaml.safe_dump(broken), encoding="utf-8")
    with pytest.raises(ConfigError, match="uncertainty"):
        load_transcription_config(path)


def test_missing_file_and_bad_yaml(tmp_path):
    with pytest.raises(ConfigError):
        load_yaml(tmp_path / "absent.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("a: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_yaml(bad)


def test_invalid_engine_rejected(tmp_path):
    path = tmp_path / "models.yaml"
    path.write_text("default:\n  engine: magic\n  model: x\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="engine"):
        load_models_config(path)
