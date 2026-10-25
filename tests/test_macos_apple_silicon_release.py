"""Pure checks for the Apple Silicon offline release recipe; no Mac build is claimed here."""

from build_tools import macos_apple_silicon as apple
from scripts import build_macos_apple_silicon_offline as builder


def test_release_identity_and_offline_marker_are_arm64_mlx():
    files = {"config.json": {"size": 269, "sha256": "a" * 64},
             "weights.npz": {"size": 3_083_520_416, "sha256": "b" * 64}}
    marker = apple.marker("4.1.2", files, "deadbeef")
    assert marker["architecture"] == "arm64"
    assert marker["target"] == "macOS Apple Silicon"
    assert marker["model"] == "whisper-large-v3-mlx"
    assert marker["model_files"] == files
    assert apple.DMG_NAME == "MELETRA-Apple-Silicon-Mac-Offline.dmg"


def test_release_recipe_refuses_cross_building_and_names_every_evidence_step():
    plan = builder.plan_text()
    for expected in ("Darwin + arm64", "verify every SHA-256", "PyInstaller arm64",
                     "audit every Mach-O file", "codesign inside-out", "mount the DMG",
                     "run --selftest", "release_manifest.json"):
        assert expected in plan


def test_ci_readiness_requires_arm64_and_every_packaged_check():
    manifest = {"dmg_sha256": "abc", "dmg_check": {"mount": True, "copy": True, "eject": True, "selftest": True},
                "mach_o_audit": {"problems": 0}, "study_quality_included": True,
                "tests": {"ran": True, "failed": 0}, "developer_id": False, "notarized": False,
                "gpl_source_offer": "SET"}
    report = {"provider": "GitHub Actions", "machine": "arm64", "dmg_sha256": "abc",
              "checks": {key: "PASS" for key in apple.CI_KEYS}}
    verdict, reasons, caveats = apple.ci_readiness(report, "abc", manifest)
    assert verdict == "CI-TESTED APPLE SILICON DMG READY" and not reasons
    assert any("ad-hoc" in note for note in caveats)

    wrong = dict(report, machine="x86_64")
    verdict, reasons, _ = apple.ci_readiness(wrong, "abc", manifest)
    assert verdict == "NEEDS WORK" and reasons
