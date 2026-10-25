"""v3 text to speech: safe chunking, voice choice, joined WAV output, MP3 export, cancel, plain errors, no coupling to STT / study."""

import subprocess
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.tts import TtsError, TtsService, Voice, chunk_text, guess_language
from src.tts.backends import OneCoreBackend, SapiBackend, SayBackend

ROOT = Path(__file__).resolve().parent.parent


class FakeBackend:
    name = "fake"

    def __init__(self, voices, max_chars=60, fail_on=None):
        self._voices, self.max_chars, self.fail_on = voices, max_chars, fail_on
        self.spoken: list[tuple[str, str]] = []

    def available(self):
        return True

    def voices(self):
        return list(self._voices)

    def synthesize(self, text, voice, wav_path, rate=0):
        if self.fail_on and self.fail_on in text:
            raise RuntimeError("engine crashed")
        self.spoken.append((text, voice.id))
        with wave.open(str(wav_path), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(16000)
            out.writeframes(b"\x01\x00" * (160 * len(text)))              # 10 ms per character


def voice(name, language, backend="fake"):
    return Voice(name, name, language, backend)


@pytest.fixture
def service():
    backend = FakeBackend([voice("Tolga", "tr-TR"), voice("Zira", "en-US"), voice("Hazel", "en-GB"), voice("Hedda", "de-DE")])
    return TtsService([backend], player=SimpleNamespace(play=lambda p: None, stop=lambda: None)), backend


# ----------------------------------------------------------------------------------------------- text
LONG = ("The nephron filters plasma. " * 40).strip()


def test_chunks_respect_the_limit_and_lose_no_words():
    pieces = chunk_text(LONG, 120)
    assert all(len(p) <= 120 for p in pieces) and " ".join(pieces).split() == LONG.split()
    assert len(pieces) > 5


def test_one_very_long_sentence_is_cut_at_spaces():
    sentence = " ".join(f"word{i}" for i in range(200))
    pieces = chunk_text(sentence, 100)
    assert all(len(p) <= 100 for p in pieces) and " ".join(pieces).split() == sentence.split()


def test_short_text_is_one_piece_and_paragraphs_stay_separate_sentences():
    assert chunk_text("One. Two.", 100) == ["One. Two."]
    assert chunk_text("", 100) == []


@pytest.mark.parametrize("text, expected", [
    ("The kidney regulates the volume and the composition of the body fluids, and it is essential.", "en"),
    ("Böbrek, vücut sıvılarının hacmini ve bileşimini düzenler ve çok önemlidir.", "tr"),
    ("Die Niere ist ein wichtiges Organ und regelt den Wasserhaushalt mit großer Genauigkeit.", "de"),
    ("A vese a vizeletet termeli, és hogy a szervezet egyensúlya megmaradjon, őrzi a vizet.", "hu"),
    ("", "en")])
def test_language_guess(text, expected):
    assert guess_language(text) == expected


# ----------------------------------------------------------------------------------------------- service
def test_voices_are_listed_and_filtered_by_language(service):
    svc, _ = service
    assert {v.name for v in svc.list_voices()} == {"Tolga", "Zira", "Hazel", "Hedda"}
    assert {v.name for v in svc.list_voices("en")} == {"Zira", "Hazel"}
    assert svc.pick_voice("en-GB").name == "Hazel" and svc.pick_voice("en").name in ("Zira", "Hazel") and svc.pick_voice("hu") is None


def test_long_text_is_spoken_in_pieces_and_joined_into_one_wav(service, tmp_path):
    svc, backend = service
    events = []
    out = svc.synthesize(LONG, tmp_path / "deep" / "speech.wav", language="en", progress=events.append)
    with wave.open(str(out)) as joined:
        frames = joined.getnframes()
    assert len(backend.spoken) == len(chunk_text(LONG, 60)) > 5 and all(v == "Zira" or v == "Hazel" for _, v in backend.spoken)
    assert frames == sum(160 * len(t) for t, _ in backend.spoken)
    assert events[-1]["percent"] == 100.0 and events[-1]["parts"] == len(backend.spoken)


def test_auto_language_picks_the_voice_from_the_text(service, tmp_path):
    svc, backend = service
    svc.synthesize("Böbrek vücut sıvılarının hacmini düzenler ve çok önemlidir.", tmp_path / "t.wav")
    assert backend.spoken[0][1] == "Tolga"


def test_missing_voice_is_explained(service, tmp_path):
    svc, _ = service
    with pytest.raises(TtsError, match="No Hungarian voice is installed"):
        svc.synthesize("Hello", tmp_path / "x.wav", language="hu")
    with pytest.raises(TtsError, match="no text"):
        svc.synthesize("   ", tmp_path / "x.wav")


def test_an_engine_crash_becomes_a_plain_error_and_cancel_stops_early(tmp_path):
    backend = FakeBackend([voice("Zira", "en-US")], max_chars=30, fail_on="BOOM")
    svc = TtsService([backend])
    with pytest.raises(TtsError, match="could not speak"):
        svc.synthesize("Fine sentence here. BOOM goes the engine.", tmp_path / "a.wav", language="en")
    calls = []
    with pytest.raises(TtsError, match="cancelled"):
        svc.synthesize(LONG, tmp_path / "b.wav", language="en", should_stop=lambda: calls.append(1) or len(calls) > 2)
    assert len(backend.spoken) < 5


def test_a_broken_backend_does_not_hide_the_working_one():
    class Broken:
        name = "broken"
        max_chars = 10

        def available(self):
            return True

        def voices(self):
            raise OSError("no audio service")
    svc = TtsService([Broken(), FakeBackend([voice("Zira", "en-US")])])
    assert [v.name for v in svc.list_voices()] == ["Zira"]


def test_wav_and_mp3_export(service, tmp_path):
    svc, _ = service
    assert svc.export("Short English sentence for export.", tmp_path / "a.wav", language="en").is_file()
    mp3 = svc.export("Short English sentence for the MP3 export, long enough to encode.", tmp_path / "a.mp3", language="en")
    import av
    with av.open(str(mp3)) as container:
        assert container.streams.audio[0].codec_context.name.startswith("mp3") and container.duration > 0
    with pytest.raises(TtsError, match=".wav or .mp3"):
        svc.export("x", tmp_path / "a.ogg")


class FakePlayer:
    """Plays instantly unless told to keep 'playing' until stop()."""

    def __init__(self, hold=False):
        self.played, self.hold, self.active, self.stops = [], hold, False, 0

    def play(self, path):
        assert Path(path).is_file()
        self.played.append(Path(path).name)
        self.active = True

    def playing(self):
        return self.active and self.hold

    def stop(self):
        self.active = False
        self.stops += 1


def test_speak_reads_every_piece_in_order_and_reports_progress():
    player = FakePlayer()
    backend = FakeBackend([voice("Zira", "en-US")], max_chars=10_000)
    svc = TtsService([backend], player=player)
    events = []
    import src.tts as tts
    text = ("The nephron filters plasma. " * 120).strip()
    assert svc.speak(text, language="en", progress=events.append) is True
    pieces = chunk_text(text, tts.FIRST_PIECE_CHARS)
    assert len(pieces) > 2 and [t for t, _ in backend.spoken] == pieces and len(player.played) == len(pieces)
    assert events[-1]["part"] == len(pieces) and player.stops >= 1


def test_stop_ends_reading_early():
    import threading
    import time
    player = FakePlayer(hold=True)
    backend = FakeBackend([voice("Zira", "en-US")], max_chars=10_000)
    svc = TtsService([backend], player=player)
    text = ("The nephron filters plasma. " * 400).strip()
    result = []
    worker = threading.Thread(target=lambda: result.append(svc.speak(text, language="en")))
    worker.start()
    deadline = time.time() + 20
    while not player.played and time.time() < deadline:
        time.sleep(0.01)
    svc.stop()
    worker.join(20)
    assert result == [False] and len(player.played) == 1 and len(backend.spoken) <= 2


def test_speak_without_a_voice_or_text_is_a_plain_error():
    svc = TtsService([FakeBackend([voice("Zira", "en-US")])], player=FakePlayer())
    with pytest.raises(TtsError, match="No Turkish voice"):
        svc.speak("Merhaba nasılsınız", language="tr")
    with pytest.raises(TtsError, match="no text"):
        svc.speak("  ")


# ----------------------------------------------------------------------------------------------- backends
def test_macos_say_voices_and_command():
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        if argv[:3] == ["say", "-v", "?"]:
            return SimpleNamespace(returncode=0, stdout="Alex                en_US    # Most people recognize me by my voice.\n"
                                                        "Eddy (German (Germany)) de_DE    # Hallo! Ich heisse Eddy.\n"
                                                        "Yelda               tr_TR    # Merhaba, benim adim Yelda.\n", stderr="")
        Path(argv[argv.index("-o") + 1]).write_bytes(b"RIFF")
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    backend = SayBackend(run=run, platform="darwin")
    assert backend.available() and not SayBackend(run=run, platform="win32").available()
    voices = backend.voices()
    assert [(v.name, v.language) for v in voices] == [("Alex", "en-US"), ("Eddy (German (Germany))", "de-DE"), ("Yelda", "tr-TR")]
    import tempfile
    target = Path(tempfile.mkdtemp()) / "o.wav"
    backend.synthesize("Merhaba", voices[2], target)
    argv = calls[-1]
    assert argv[:3] == ["say", "-v", "Yelda"] and "--file-format=WAVE" in argv and target.is_file()


def test_windows_sapi_voice_list_parsing_and_failure():
    out = "Microsoft Zira Desktop|en-US|Female\nMicrosoft David Desktop|en-US|Male\n\n"
    backend = SapiBackend(run=lambda *a, **k: SimpleNamespace(returncode=0, stdout=out, stderr=""))
    assert [(v.name, v.gender) for v in backend.voices()] == [("Microsoft Zira Desktop", "female"), ("Microsoft David Desktop", "male")]
    failing = SapiBackend(run=lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="Voice not found"))
    with pytest.raises(TtsError, match="could not speak"):
        failing.synthesize("hello", voice("X", "en-US"), Path("nowhere.wav"))


def test_importing_tts_loads_no_transcription_or_study_code():
    code = ("import sys; import src.tts; bad = [m for m in sys.modules if m.startswith(('src.study', 'src.pipeline', 'faster_whisper', "
            "'ctranslate2', 'src.transcriber', 'torch'))]; sys.exit(1 if bad else 0)")
    assert subprocess.run([sys.executable, "-c", code], cwd=ROOT, timeout=120).returncode == 0


# ----------------------------------------------------------------------------------------------- real voices (Windows)
@pytest.mark.windows_only
@pytest.mark.skipif(sys.platform != "win32", reason="Windows sound player")
def test_real_windows_player_starts_reports_and_stops(tmp_path):
    from src.tts.backends import Player
    wav = tmp_path / "tone.wav"
    with wave.open(str(wav), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes(b"\x00\x00" * 16000 * 3)                      # three seconds of silence
    player = Player()
    try:
        player.play(wav)
    except TtsError:
        pytest.skip("no audio output device")
    assert player.playing()
    player.stop()
    assert not player.playing()


@pytest.mark.windows_only
@pytest.mark.skipif(sys.platform != "win32", reason="Windows voices")
def test_real_windows_voices_speak_to_a_valid_wav(tmp_path):
    svc = TtsService()
    english = svc.pick_voice("en")
    if english is None:
        pytest.skip("no English voice installed")
    out = svc.synthesize("The afferent arteriole regulates glomerular filtration.", tmp_path / "en.wav", voice=english)
    with wave.open(str(out)) as w:
        assert w.getnframes() / w.getframerate() > 1.0
    turkish = svc.pick_voice("tr")
    if turkish is not None:
        with wave.open(str(svc.synthesize("Böbrekte glomerüler filtrasyon.", tmp_path / "tr.wav", voice=turkish))) as w:
            assert w.getnframes() > 1000


@pytest.mark.windows_only
def test_sapi_finds_powershell_even_without_it_on_the_path(monkeypatch):
    from src.tts import backends
    monkeypatch.setenv("PATH", "")
    argv = backends._encoded("Write-Output 1")
    assert argv[0].lower().endswith("powershell.exe") and ("system32" in argv[0].lower() or argv[0] == "powershell")
