"""Build layout and version parsing: a 0.2.x build must never land on the production 0.1.1 files."""

from pathlib import Path

import pytest

from build_tools.layout import base_version, dist_dir, dist_parent, is_legacy, release_dir, split_version
from build_tools.versioninfo import version_file_text, version_tuple

ROOT = Path("C:/proj")


def test_versions_with_and_without_a_suffix():
    assert split_version("0.1.1") == ((0, 1, 1, 0), "")
    assert split_version("0.2.0-dev") == ((0, 2, 0, 0), "dev")
    assert split_version("0.2.0-rc1") == ((0, 2, 0, 0), "rc1")
    assert split_version("1.2.3.4") == ((1, 2, 3, 4), "")
    assert base_version("0.2.0-alpha") == "0.2.0" and base_version("0.1.1") == "0.1.1"


@pytest.mark.parametrize("junk", ["", "abc", "1", "1.2", "0.2.0--"])
def test_junk_versions_are_rejected(junk):
    with pytest.raises(ValueError):
        split_version(junk)


def test_the_production_baseline_keeps_its_old_paths():
    assert is_legacy("0.1.1") and is_legacy("0.1.0")
    assert dist_dir("0.1.1", ROOT) == ROOT / "dist" / "MedicalLectureTranscriber"
    assert release_dir("0.1.1", "Windows", ROOT) == ROOT / "release"


def test_new_versions_get_their_own_folders():
    assert not is_legacy("0.2.0-dev")
    assert dist_dir("0.2.0-dev", ROOT) == ROOT / "dist" / "v0.2.0" / "MedicalLectureTranscriber"
    assert dist_parent("0.2.0", ROOT) == ROOT / "dist" / "v0.2.0"
    assert release_dir("0.2.0-rc1", "Windows", ROOT) == ROOT / "release" / "v0.2.0" / "Windows"
    assert release_dir("0.2.0", "macOS", ROOT) == ROOT / "release" / "v0.2.0" / "macOS"


def test_the_two_never_share_a_folder():
    for version in ("0.2.0-dev", "0.2.0", "0.3.1"):
        assert dist_dir(version, ROOT) != dist_dir("0.1.1", ROOT)
        assert release_dir(version, "Windows", ROOT) != release_dir("0.1.1", "Windows", ROOT)
        assert not str(dist_dir(version, ROOT)).startswith(str(dist_dir("0.1.1", ROOT)))


def test_the_exe_resource_takes_the_numeric_part_only():
    assert version_tuple("0.2.0-dev") == (0, 2, 0, 0)
    text = version_file_text("0.2.0-dev", "Co", "Prod", "Desc", "p.exe")
    assert "(0, 2, 0, 0)" in text and "0.2.0.0" in text and "0.2.0-dev" in text


def test_the_source_tree_declares_a_valid_version():
    from src import __version__

    split_version(__version__)
