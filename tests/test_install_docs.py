"""The install instructions of the private repository: one set of tested commands (the public repository's), a separate developer setup, no placeholders, and the
security gate that guards every push."""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import secret_gate  # noqa: E402

BASE = "https://twonezerotwo-hue.github.io/whispermedical-downloads"
WINDOWS_COMMAND = ("$f = \"$env:TEMP\\meletra-install.ps1\"; iwr -UseBasicParsing '" + BASE + "/install/install-windows.ps1' -OutFile $f; "
                   "powershell -NoProfile -ExecutionPolicy Bypass -File $f")
MAC_COMMAND = "curl -fsSL --proto '=https' '" + BASE + "/install/install-macos.sh' -o /tmp/meletra-install.sh && bash /tmp/meletra-install.sh"


def read(path) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_the_readme_starts_with_the_installation_not_with_development():
    readme = read("README.md")
    first = readme[:readme.index("## Developer setup")]
    assert WINDOWS_COMMAND in first and MAC_COMMAND in first
    assert readme.index("## Install MELETRA") < readme.index("## Developer setup") < readme.index("# Geliştirici notları")
    assert "private" in first.lower() and "whispermedical-downloads" in first and "release-manifest.json" in first
    for needle in ("one click", "One command", "Intel"):
        assert needle.lower() in first.lower()
    assert not re.search(r"<[A-Z][A-Z _-]{3,}>|(?i:\bTODO\b|\bPLACEHOLDER\b|example\.com)", first)


def test_the_commands_are_the_ones_the_public_repository_documents_and_tests():
    public = ROOT.parent / "whispermedical-downloads"
    if not (public / "README.md").is_file():
        pytest.skip("the public distribution checkout is not next to this one")
    text = (public / "README.md").read_text(encoding="utf-8") + (public / ".github" / "workflows" / "test-install-scripts.yml").read_text(encoding="utf-8")
    assert WINDOWS_COMMAND in text and MAC_COMMAND in text


def test_developer_setup_is_separate_from_the_installation_and_says_so():
    for name in ("scripts/dev-setup-windows.ps1", "scripts/dev-setup-macos.sh"):
        assert "DEVELOPER SETUP" in read(name) and "NOT the way to install" in read(name)
    readme = read("README.md")
    assert "scripts\\dev-setup-windows.ps1" in readme and "scripts/dev-setup-macos.sh" in readme
    assert "authorized collaborators" in readme and "install/README.md" in readme
    assert "COLLABORATORS" in read("install/README.md") and "do not use this folder" in read("install/README.md")


def test_the_developer_setup_scripts_parse():
    bash = shutil.which("bash")
    if bash:
        assert subprocess.run([bash, "-n", str(ROOT / "scripts" / "dev-setup-macos.sh")], capture_output=True).returncode == 0
    if sys.platform == "win32":
        parse = ("$e=$null;$t=$null;[void][System.Management.Automation.Language.Parser]::ParseFile('" + str(ROOT / "scripts" / "dev-setup-windows.ps1") +
                 "',[ref]$t,[ref]$e);if($e.Count){$e|%{$_.Message};exit 1}")
        assert subprocess.run(["powershell.exe", "-NoProfile", "-Command", parse], capture_output=True, text=True).returncode == 0


def test_the_workflows_of_the_private_repository_are_valid_and_cheap_ones_do_not_run_on_every_push():
    for path in sorted((ROOT / ".github" / "workflows").glob("*.yml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert data and "jobs" in data, path.name
    checks = yaml.safe_load(read(".github/workflows/repo-checks.yml"))
    assert set(checks["jobs"]) == {"secret-scan", "knowledge-gate", "install-docs"}
    assert "push" not in checks[True] and "workflow_dispatch" in checks[True]


def test_the_secret_gate_blocks_credentials_and_private_files_and_tolerates_the_known_harmless_findings(tmp_path):
    repo = tmp_path / "r"
    repo.mkdir()
    run = lambda *a: subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True, check=True)  # noqa: E731
    run("init", "-q")
    run("config", "user.email", "t@example.org")
    run("config", "user.name", "t")
    (repo / "README.md").write_text("see C:\\Users\\someone\\project and mail me@example.org\n", encoding="utf-8")
    (repo / ".env.example").write_text("KEY=\n", encoding="utf-8")
    run("add", "-A")
    run("commit", "-q", "-m", "ok")
    assert secret_gate.gate(repo)["result"] == "PASS"
    (repo / "token.txt").write_text("t = ghp_" + "a" * 36 + "\n", encoding="utf-8")
    (repo / "x.pfx").write_bytes(b"\x00")
    (repo / ".env").write_text("A=1\n", encoding="utf-8")
    (repo / "lecture.mp3").write_bytes(b"\x00")
    run("add", "-A")
    run("commit", "-q", "-m", "bad")
    out = secret_gate.gate(repo)
    assert out["result"] == "FAIL" and {"github token", "certificate or key", "environment file", "recording"} <= set(out["blocking"])
    assert "ghp_" not in json.dumps(out)                                                       # the secret itself is never printed
