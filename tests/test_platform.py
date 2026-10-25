"""The platform abstraction: Windows keeps its exact behaviour, macOS is verified on this Windows machine with a fake home folder
and a fake command runner (the first real run happens on a Mac)."""

import subprocess
import types
from pathlib import Path

import pytest

from src import platform as plat
from src.platform import MacPlatform, PosixPlatform, WindowsPlatform, current, use
from src.services import paths as paths_mod
from src.services.credential_service import (
    CredentialService, KeyringBackend, MacKeychainBackend, MemoryCredentialBackend, WindowsCredentialBackend,
)
from src.services.paths import AppPaths, get_paths, resolve_user_data_root


class Recorder:
    """A subprocess.run stand-in: remembers the commands, answers from a table."""

    def __init__(self, answers=None):
        self.calls, self.answers = [], answers or {}

    def __call__(self, args, **kwargs):
        self.calls.append(list(args))
        out, code = self.answers.get(args[0], ("", 0)) if not isinstance(self.answers.get(tuple(args)), tuple) else self.answers[tuple(args)]
        return types.SimpleNamespace(stdout=out, stderr="", returncode=code)


def mac(tmp_path, answers=None, machine="arm64", env=None):
    run = Recorder(answers)
    return MacPlatform(env=env or {}, home=tmp_path / "home", run=run, machine=machine), run


# ----------------------------------------------------------------------------------------------- Windows keeps its behaviour
def test_windows_data_folder_is_localappdata():
    w = WindowsPlatform(env={"LOCALAPPDATA": r"C:\Users\x\AppData\Local"}, home=Path("C:/Users/x"))
    assert w.user_data_dir() == Path(r"C:\Users\x\AppData\Local") / "MedicalLectureTranscriber"
    assert w.cache_dir() is None and w.logs_dir() is None                   # everything below one folder, as in 0.1.1


def test_windows_without_localappdata_falls_back_to_the_profile():
    w = WindowsPlatform(env={}, home=Path("C:/Users/x"))
    assert w.user_data_dir() == Path("C:/Users/x/AppData/Local/MedicalLectureTranscriber")


@pytest.mark.windows_only
def test_windows_facts():
    w = WindowsPlatform()
    assert w.name == "windows" and w.exe_suffix == ".exe" and w.secure_store_label == "Windows Credential Manager"
    assert w.memory_status()["total_gb"] > 1 and w.cpu_name() and "Windows" in w.os_description()
    assert w.process_flags() == 0x08000000


@pytest.mark.windows_only
def test_the_real_platform_here_is_windows():
    assert current().name == "windows" and not current().is_apple_silicon
    assert isinstance(current().credential_backend(), WindowsCredentialBackend)


@pytest.mark.windows_platform
def test_resolve_user_data_root_still_honours_the_environment(tmp_path):
    env = {"LOCALAPPDATA": str(tmp_path / "L")}
    assert resolve_user_data_root(env, tmp_path / "app") == tmp_path / "L" / "MedicalLectureTranscriber"
    assert resolve_user_data_root({**env, "MLT_DATA_DIR": str(tmp_path / "X")}, tmp_path / "app") == tmp_path / "X"


# ----------------------------------------------------------------------------------------------- macOS folders
def test_mac_folders_follow_apple_conventions(tmp_path):
    m, _ = mac(tmp_path)
    home = tmp_path / "home"
    assert m.user_data_dir() == home / "Library" / "Application Support" / "MedicalLectureTranscriber"
    assert m.cache_dir() == home / "Library" / "Caches" / "MedicalLectureTranscriber"
    assert m.logs_dir() == home / "Library" / "Logs" / "MedicalLectureTranscriber"


def test_mac_app_paths_split_cache_and_logs_from_the_data_folder(tmp_path):
    m, _ = mac(tmp_path)
    paths = get_paths(env={}, app_dir=tmp_path / "app", platform=m)
    home = tmp_path / "home" / "Library"
    assert paths.root == home / "Application Support" / "MedicalLectureTranscriber"
    assert paths.cache == home / "Caches" / "MedicalLectureTranscriber" and paths.logs == home / "Logs" / "MedicalLectureTranscriber"
    assert paths.models == paths.root / "models" and paths.settings_file == paths.root / "settings" / "settings.json"
    for folder in (paths.models, paths.jobs, paths.checkpoints, paths.settings, paths.temp, paths.cache, paths.logs):
        assert folder.is_dir()                                              # created
    assert paths.app_log == paths.logs / "app.log" and paths.job_log("abc") == paths.logs / "job-abc.log"


def test_the_override_keeps_everything_below_one_folder_even_on_a_mac(tmp_path):
    m, _ = mac(tmp_path)
    paths = get_paths(env={"MLT_DATA_DIR": str(tmp_path / "one")}, app_dir=tmp_path / "app", platform=m)
    assert paths.cache == tmp_path / "one" / "cache" and paths.logs == tmp_path / "one" / "logs"


def test_portable_mode_beats_the_platform_folders(tmp_path):
    app = tmp_path / "app"
    app.mkdir()
    (app / "portable.flag").write_text("x")
    m, _ = mac(tmp_path)
    assert get_paths(env={}, app_dir=app, platform=m).root == app / "data"


def test_windows_app_paths_are_unchanged(tmp_path):
    w = WindowsPlatform(env={"LOCALAPPDATA": str(tmp_path / "L")}, home=tmp_path)
    paths = get_paths(env={"LOCALAPPDATA": str(tmp_path / "L")}, app_dir=tmp_path / "app", platform=w)
    base = tmp_path / "L" / "MedicalLectureTranscriber"
    assert (paths.root, paths.cache, paths.logs) == (base, base / "cache", base / "logs")
    assert paths.cache_root is None and paths.logs_root is None


def test_plain_app_paths_object_still_works():
    assert AppPaths(Path("X")).cache == Path("X") / "cache" and AppPaths(Path("X")).logs == Path("X") / "logs"


def test_linux_fallback_uses_xdg(tmp_path):
    p = PosixPlatform(env={"XDG_DATA_HOME": str(tmp_path / "share")}, home=tmp_path)
    assert p.user_data_dir() == tmp_path / "share" / "MedicalLectureTranscriber"
    assert PosixPlatform(env={}, home=tmp_path).user_data_dir() == tmp_path / ".local" / "share" / "MedicalLectureTranscriber"


# ----------------------------------------------------------------------------------------------- macOS shell, power, facts
def test_mac_opens_and_reveals_with_the_open_command(tmp_path):
    m, run = mac(tmp_path)
    f = tmp_path / "a.txt"
    f.write_text("x")
    m.open_path(f)
    m.reveal_path(f)
    m.reveal_path(tmp_path)
    assert run.calls == [["open", str(f)], ["open", "-R", str(f)], ["open", str(tmp_path)]]


def test_mac_keeps_awake_with_caffeinate_and_releases_it(tmp_path, monkeypatch):
    started = []

    class FakeProc:
        def terminate(self):
            started.append("terminated")

    monkeypatch.setattr(subprocess, "Popen", lambda args, **k: started.append(args) or FakeProc())
    m, _ = mac(tmp_path)
    assert m.keep_awake(True) is True and m.keep_awake(True) is True          # a second request does not start a second helper
    assert started == [["caffeinate", "-i"]]
    assert m.keep_awake(False) is True and started[-1] == "terminated"
    assert m.keep_awake(False) is True                                       # releasing twice is harmless


def test_mac_keep_awake_survives_a_missing_helper(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(OSError("no caffeinate")))
    m, _ = mac(tmp_path)
    assert m.keep_awake(True) is False


VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               10000.
Pages active:                             50000.
Pages inactive:                           20000.
Pages speculative:                         5000.
"""


def test_mac_memory_cpu_and_os_come_from_system_tools(tmp_path):
    answers = {("sysctl", "-n", "hw.memsize"): (str(16 * 1024 ** 3), 0), ("vm_stat",): (VM_STAT, 0),
               ("sysctl", "-n", "machdep.cpu.brand_string"): ("Apple M2 Pro\n", 0), ("sw_vers", "-productVersion"): ("14.5\n", 0)}
    m, run = mac(tmp_path, answers)
    assert m.memory_status() == {"total_gb": 16.0, "free_gb": round(35000 * 16384 / 1024 ** 3, 1)}
    assert m.cpu_name() == "Apple M2 Pro" and m.os_description() == "macOS 14.5 (arm64)"
    assert m.is_apple_silicon and m.accelerator().kind == "apple" and "M2" in m.accelerator().name


def test_mac_facts_survive_missing_tools(tmp_path):
    m, _ = mac(tmp_path, machine="x86_64")
    assert m.memory_status() == {"total_gb": 0.0, "free_gb": 0.0} and m.cpu_name() == "Intel" and m.os_description() == "macOS (x86_64)"
    assert not m.is_apple_silicon and m.accelerator().kind == "none" and "Intel" in m.accelerator().detail


def test_a_failing_helper_command_is_not_fatal(tmp_path):
    def broken(args, **k):
        raise subprocess.TimeoutExpired(args, 1)

    m = MacPlatform(env={}, home=tmp_path, run=broken)
    assert m.cpu_name() in ("Apple Silicon", "Intel") and m.memory_status()["total_gb"] == 0.0


def test_describe(tmp_path):
    m, _ = mac(tmp_path)
    d = m.describe()
    assert d["name"] == "macos" and d["apple_silicon"] is True and d["machine"] == "arm64"


# ----------------------------------------------------------------------------------------------- credentials
def test_mac_keychain_backend_talks_to_the_security_tool():
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        if args[1] == "find-generic-password":
            return types.SimpleNamespace(returncode=0, stdout="s3cret\n", stderr="")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    b = MacKeychainBackend(run=run)
    b.set("target", "s3cret")
    assert b.get("target") == "s3cret" and b.delete("target") is True
    assert calls[0][:2] == ["security", "add-generic-password"] and "-U" in calls[0] and calls[0][-1] == "s3cret"
    assert calls[1][:2] == ["security", "find-generic-password"] and calls[2][:2] == ["security", "delete-generic-password"]


def test_mac_keychain_missing_or_refused():
    fail = lambda args, **k: types.SimpleNamespace(returncode=44, stdout="", stderr="could not be found")   # noqa: E731
    b = MacKeychainBackend(run=fail)
    assert b.get("t") is None and b.delete("t") is False
    with pytest.raises(OSError, match="refused"):
        b.set("t", "x")


def test_keyring_backend_uses_the_package_when_present():
    store = {}
    fake = types.SimpleNamespace(
        set_password=lambda s, t, p: store.__setitem__((s, t), p), get_password=lambda s, t: store.get((s, t)),
        delete_password=lambda s, t: store.pop((s, t)))
    b = KeyringBackend(fake)
    b.set("t", "k")
    assert b.get("t") == "k" and b.delete("t") is True and b.get("t") is None and b.delete("t") is False


def test_mac_prefers_keyring_then_the_security_tool(tmp_path, monkeypatch):
    m, _ = mac(tmp_path)
    monkeypatch.setattr(KeyringBackend, "try_create", classmethod(lambda cls: None))
    assert isinstance(m.credential_backend(), MacKeychainBackend)
    sentinel = KeyringBackend(object())
    monkeypatch.setattr(KeyringBackend, "try_create", classmethod(lambda cls: sentinel))
    assert m.credential_backend() is sentinel


def test_credential_service_names_where_the_key_lives():
    assert CredentialService(MemoryCredentialBackend()).secure_storage is False
    assert "memory" in CredentialService(MemoryCredentialBackend()).store_label
    assert CredentialService(MacKeychainBackend(run=lambda *a, **k: None)).secure_storage is True
    assert CredentialService(MacKeychainBackend(run=lambda *a, **k: None)).store_label == "macOS Keychain"


def test_the_generic_platform_keeps_keys_in_memory():
    assert isinstance(PosixPlatform().credential_backend(), MemoryCredentialBackend)


# ----------------------------------------------------------------------------------------------- the current platform
def test_use_swaps_the_platform_temporarily(tmp_path):
    m, _ = mac(tmp_path)
    before = current()
    with use(m):
        assert current() is m and resolve_user_data_root({}, tmp_path / "app") == m.user_data_dir()
    assert current() is before


def test_services_follow_the_platform(tmp_path):
    from src.services import diagnostics_service as dsvc

    answers = {("sysctl", "-n", "machdep.cpu.brand_string"): ("Apple M1", 0), ("sw_vers", "-productVersion"): ("13.6", 0),
               ("sysctl", "-n", "hw.memsize"): (str(8 * 1024 ** 3), 0)}
    m, _ = mac(tmp_path, answers)
    with use(m):
        assert dsvc.cpu_name() == "Apple M1" and dsvc.windows_version() == "macOS 13.6 (arm64)"
        assert dsvc.memory_status()["total_gb"] == 8.0


def test_executables_use_the_platform_suffix(tmp_path, monkeypatch):
    from src import platform_utils

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "ffmpeg").write_text("x")
    monkeypatch.setattr(platform_utils, "bundled_bin_dir", lambda: bin_dir)
    m, _ = mac(tmp_path)
    with use(m):
        assert platform_utils.find_executable("ffmpeg") == bin_dir / "ffmpeg"
    (bin_dir / "ffmpeg.exe").write_text("x")
    with use(WindowsPlatform()):
        assert platform_utils.find_executable("ffmpeg") == bin_dir / "ffmpeg.exe"


def test_detect_picks_by_operating_system(monkeypatch):
    for name, cls in (("win32", WindowsPlatform), ("darwin", MacPlatform), ("linux", PosixPlatform)):
        monkeypatch.setattr(plat.sys, "platform", name)
        assert type(plat.detect()) is cls
