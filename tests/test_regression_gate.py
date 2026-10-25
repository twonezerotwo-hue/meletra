"""A permanent guard for version 0.1.1: the same checks as scripts/regression_gate.py (without the long ones) run with every test run, so
a change to the production behaviour or to the production artefacts can never go unnoticed."""

import importlib.util
import shutil
import sys
from pathlib import Path

import subprocess

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("regression_gate", ROOT / "scripts" / "regression_gate.py")
gate = importlib.util.module_from_spec(spec)
sys.modules["regression_gate"] = gate
spec.loader.exec_module(gate)

needs_git = pytest.mark.skipif(shutil.which("git") is None or not (ROOT / ".git").exists(), reason="needs the git repository")


@needs_git
def test_the_production_tag_still_points_at_the_production_commit():
    if subprocess.run(["git", "rev-parse", "-q", "--verify", "refs/tags/desktop-app-v0.1.1"], cwd=ROOT, capture_output=True).returncode != 0:
        pytest.skip("the production tag is not in this checkout (a CI clone without tags)")
    check = gate.check_tag()
    assert check.ok, check.line()


@needs_git
def test_the_core_of_the_transcriber_is_the_core_of_0_1_1():
    check = gate.check_core_sources()
    assert check.ok, check.line()


@pytest.mark.skipif(not (ROOT / "release" / "MedicalLectureTranscriber-Setup-x64.exe").exists(), reason="the production artefacts are not on this machine")
def test_the_production_release_files_are_untouched():
    check = gate.check_release_files()
    assert check.ok, check.line()


@pytest.mark.skipif(not (ROOT / "dist" / "MedicalLectureTranscriber" / "MedicalLectureTranscriber.exe").exists(), reason="no production dist here")
def test_the_production_unpacked_app_is_untouched():
    check = gate.check_dist_tree()
    assert check.ok, check.line()


def test_the_gate_reports_plainly():
    assert gate.Check("x", True, "fine").line() == "[PASS] x: fine"
    assert gate.Check("y", False, "broken").line() == "[FAIL] y: broken"
    assert gate.MIN_TESTS == 712
