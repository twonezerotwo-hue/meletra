"""Command-line flags of the Medical Expert Edition (parser only: no model, no GPU, no network)."""

import pytest

import app


@pytest.fixture
def parse(cfg):
    parser = app.build_parser(cfg)

    def run(*argv):
        args = parser.parse_args(list(argv))
        return app.options_from_args(args, cfg)

    return run


def test_defaults_are_local_with_context_engine_following_the_config(parse):
    options = parse("transcribe", "lecture.mp4")
    assert options.backend == "local" and options.deepgram_if_suspect is False
    assert options.auto_topic is None                       # None = follow config topic.auto (ON)
    assert options.context_engine is None and options.chunk_minutes is None and options.reuse_audio is None
    assert options.use_hotwords and options.use_pdf and options.resume


def test_shipped_config_turns_subject_detection_and_expert_engine_on(cfg):
    assert cfg["topic"]["auto"] is True and cfg["context_engine"]["engine"] == "expert"
    assert cfg["long_audio"]["chunk_minutes"] == 15 and cfg["long_audio"]["overlap_seconds"] == 3
    assert cfg["quality"]["compute_type"] == "float16" and cfg["defaults"]["device"] == "auto"
    assert cfg["context"]["hotwords_disable_conditioning"] is True


def test_probe_flags(parse):
    assert parse("transcribe", "a.mp4", "--no-auto-topic").auto_topic is False
    assert parse("transcribe", "a.mp4", "--auto-topic").auto_topic is True
    probe = parse("probe", "a.mp4")
    assert probe.auto_topic is True and probe.probe_only is True


def test_online_backends_must_be_asked_for_explicitly(parse):
    assert parse("transcribe", "a.mp4", "--backend", "compare").backend == "compare"
    assert parse("transcribe", "a.mp4", "--backend", "deepgram").backend == "deepgram"
    assert parse("transcribe", "a.mp4", "--deepgram-if-suspect").deepgram_if_suspect is True
    with pytest.raises(SystemExit):
        parse("transcribe", "a.mp4", "--backend", "whisper-cloud")


def test_subject_accepts_vocabulary_files_and_medical_areas(parse):
    assert parse("transcribe", "a.mp4", "--subject", "cardiology,neuroscience").subjects == ["cardiology", "neuroscience"]
    assert parse("transcribe", "a.mp4").subjects == ["general_medical"]
    with pytest.raises(Exception, match="basketweaving"):
        parse("transcribe", "a.mp4", "--subject", "basketweaving")


def test_engine_chunk_and_audio_flags(parse, tmp_path):
    options = parse("transcribe", "a.mp4", "--context-engine", "legacy", "--chunk-minutes", "10",
                    "--reuse-audio", str(tmp_path / "x.wav"))
    assert options.context_engine == "legacy" and options.chunk_minutes == 10 and options.reuse_audio == tmp_path / "x.wav"


def test_unknown_engine_is_rejected_at_run_time(cfg, tmp_path):
    from src.pipeline import RunOptions, _engine_name

    with pytest.raises(ValueError, match="unknown context engine"):
        _engine_name(RunOptions(context_engine="magic"), cfg)
