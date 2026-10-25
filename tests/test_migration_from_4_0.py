"""A user of 4.0 installs MELETRA 4.1: the lecture archive, the transcripts, the Study packs, the settings and the models are all still there.

The profile is written by the CODE OF 4.0.0 (the tag's source tree, extracted with ``git archive``), not by a fixture of 4.1. MELETRA keeps the data folder of every earlier version
(``%LOCALAPPDATA%\\MedicalLectureTranscriber``: ``src.APP_ID`` is unchanged), so nothing has to be moved; what has to hold is that the 4.0 files are read, and that the one setting that
changed meaning (transcript sharing: now OFF and opt-in) is not carried over.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import io
from pathlib import Path

import pytest

import src
from src.gui.main_window import MainWindow
from src.services.credential_service import CredentialService, MemoryCredentialBackend
from src.services.job_service import lecture_title
from src.services.model_service import ModelService
from src.services.paths import AppPaths, get_paths
from tests.gui_support import MODEL_KEY, ScriptedLauncher, fake_info, make_gpu, settle
from src.gui.context import AppServices

ROOT = Path(__file__).resolve().parent.parent
TAG = "desktop-app-v4.0.0"


def have_tag() -> bool:
    done = subprocess.run(["git", "rev-parse", "--verify", "--quiet", TAG + "^{commit}"], cwd=str(ROOT), capture_output=True, text=True)
    return done.returncode == 0


@pytest.fixture(scope="module")
def profile(tmp_path_factory):
    if not have_tag():
        pytest.skip(f"the tag {TAG} is not in this clone")
    work = tmp_path_factory.mktemp("migration")
    tree = work / "v40"
    tree.mkdir()
    archive = subprocess.run(["git", "archive", TAG, "src", "config", "medical_terms", "medical_vocab", "tests/study_support.py"], cwd=str(ROOT), capture_output=True, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(tree)
    data = work / "LocalAppData" / "MedicalLectureTranscriber"
    env = {k: v for k, v in os.environ.items() if k not in ("QT_QPA_PLATFORM",)}
    env.update({"PYTHONPATH": str(tree), "MLT_DATA_DIR": str(data)})
    done = subprocess.run([sys.executable, str(ROOT / "tests" / "migration_profile_40.py"), str(data)], cwd=str(tree), env=env, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    return data, json.loads((data / "expected.json").read_text(encoding="utf-8"))


def open_services(data: Path, tmp: Path) -> AppServices:
    paths = AppPaths(data).ensure()
    models = ModelService(paths, registry={MODEL_KEY: fake_info()}, app_dir=tmp / "appdir", sleep=lambda s: None, retries=1)
    return AppServices.create(env={"MLT_DATA_DIR": str(data)}, app_dir=tmp / "appdir", credentials=CredentialService(MemoryCredentialBackend()), gpu=make_gpu(), models=models)


def test_the_data_folder_of_every_earlier_version_is_the_one_meletra_uses():
    assert src.APP_NAME == "MELETRA" and src.APP_ID == "MedicalLectureTranscriber"
    env = {"LOCALAPPDATA": r"C:\Users\x\AppData\Local"}
    assert Path(get_paths(env, create=False).root).name == "MedicalLectureTranscriber"                       # no new, empty folder: the lectures of 4.0 are right there


def test_a_profile_written_by_4_0_is_read_by_meletra(profile, tmp_path):
    data, expected = profile
    assert expected["version"].startswith("4.0")
    before = {p.relative_to(data).as_posix(): p.stat().st_size for p in data.rglob("*") if p.is_file() and p.name != "expected.json"}
    services = open_services(data, tmp_path)
    # settings: the 4.0 choices stay, sharing is not carried over, the file is upgraded to the new schema
    s = services.settings.settings
    assert s.language == "tr" and s.theme == "light" and s.first_run_done is True and Path(s.output_dir).name == "Transcripts"
    assert s.share_transcripts is False and s.share_notice_shown is False
    # the archive: every completed lecture of 4.0 is listed, with the name 4.0 knew (the file name or the renamed folder), the failed one too
    archive = {r.job_id: r for r in services.jobs.archive()}
    assert set(archive) == {"job-biochem", "job-collagen", "job-failed"}
    assert archive["job-biochem"].state == "completed" and archive["job-failed"].state == "failed"
    assert Path(archive["job-collagen"].output_dir).name == "My own name for the collagen lecture"          # the folder the student renamed
    assert lecture_title(archive["job-biochem"]) == "Biochem_1.mp4"
    # a name chosen in 4.1 is kept, the recording is never renamed, and the 4.0 files are untouched by reading them
    services.jobs.rename("job-collagen", "Collagen and elastin")
    assert lecture_title(open_services(data, tmp_path).jobs.get("job-collagen")) == "Collagen and elastin"
    after = {p.relative_to(data).as_posix(): p.stat().st_size for p in data.rglob("*") if p.is_file() and p.name != "expected.json"}
    assert all(name in after for name in before)                                                       # nothing of 4.0 was deleted
    # transcripts and Study: the saved transcript opens, the saved pack is loaded (not rebuilt)
    lecture = Path(expected["lectures"]["job-biochem"])
    assert (lecture / "local_clean_transcript.txt").read_text(encoding="utf-8").splitlines()[0] == expected["transcript_first_line"]
    state = services.study.open(lecture, "Biochemistry 1")
    assert state.pack is not None and len(state.pack.flashcards) == expected["flashcards"] and len(state.pack.questions) == expected["questions"]
    # the model of the user is still found
    assert services.models.is_installed(MODEL_KEY)


def test_the_window_shows_the_old_archive_and_opens_an_old_lecture(profile, qapp, tmp_path):
    from src.gui import theme
    from tests.gui_support import wait_for

    data, expected = profile
    copy = tmp_path / "profile_copy"
    shutil.copytree(data, copy)
    theme.apply_theme(qapp, "light")
    services = open_services(copy, tmp_path)
    window = MainWindow(services, ScriptedLauncher())
    window.show()
    window.simple_home.refresh()
    window.files_page.refresh()
    names = [card.accessibleName() for card in window.files_page.cards]
    assert any(n.startswith("Lecture Biochem_1.mp4, completed") for n in names) and any("My own name" in n or "Collagen" in n for n in names), names
    assert window.windowTitle() == "MELETRA"
    biochem = next(card for card in window.files_page.cards if card.record.job_id == "job-biochem")
    biochem.clicked.emit(biochem.record)                                                                  # open it from the archive, as a click does
    wait_for(lambda: window.current_output is not None, what="the lecture to open")
    assert Path(window.current_output) == Path(expected["lectures"]["job-biochem"])
    window.open_transcript()
    assert expected["transcript_first_line"] in window.transcript_view.text.toPlainText()
    window.hide()
    settle(window, window.home, window.study)
    window.deleteLater()
