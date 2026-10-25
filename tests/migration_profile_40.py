"""Writes the data folder of a 4.0 user, with the CODE OF 4.0.0 (run by tests/test_migration_from_4_0.py with the tag's source tree first on PYTHONPATH):

    python tests/migration_profile_40.py DATA_DIR

settings.json (schema 3, the 4.0 fields), the job records of the archive (completed, failed, a lecture whose folder the user renamed), a lecture folder with its transcripts and
a saved Study pack, and a model folder. Nothing of 4.1 is imported here: the files are exactly what 4.0 writes.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from src import __version__
from src.services.credential_service import CredentialService, MemoryCredentialBackend
from src.services.job_service import JobStore
from src.services.paths import AppPaths
from src.services.settings_service import SettingsService
from src.services.study_service import StudyService
from tests.study_support import write_lecture_folder


def main(data: Path) -> None:
    assert __version__.startswith("4.0"), f"this script must run with the 4.0 source tree (found {__version__})"
    paths = AppPaths(data).ensure()
    settings = SettingsService(paths.settings_file)
    settings.update(language="tr", theme="light", output_dir=str(data / "Transcripts"), first_run_done=True, hotwords="auto", mode="quality")
    store = JobStore(paths.jobs)

    lectures = data / "Transcripts"
    biochem = write_lecture_folder(lectures / "Biochemistry 1 - glycolysis")
    renamed = write_lecture_folder(lectures / "My own name for the collagen lecture")                  # the student renamed this folder in Explorer
    store.create_or_get("job-biochem", data / "Biochem_1.mp4", output_dir=str(biochem), state="completed", progress=100, language="en", duration_seconds=2700)
    store.update("job-biochem", state="completed")
    store.create_or_get("job-collagen", data / "Collagen_2.mp4", output_dir=str(renamed), state="completed", progress=100, language="tr", duration_seconds=1800)
    store.update("job-collagen", state="completed")
    store.create_or_get("job-failed", data / "Anatomy_thorax.mp4", state="failed", language="auto", duration_seconds=1500)

    study = StudyService(settings, CredentialService(MemoryCredentialBackend()))
    state = study.prepare(biochem, title="Biochemistry 1")
    pack = study.generate(state)
    models = paths.models / "whisper-large-v3-folder"
    models.mkdir(parents=True, exist_ok=True)
    rng = random.Random(3)                                                                                    # the same small stand-in model files the GUI tests use
    (models / "config.json").write_bytes(b'{"model": "tiny"}')
    (models / "model.bin").write_bytes(bytes(rng.getrandbits(8) for _ in range(50_000)))
    (models / "tokenizer.json").write_bytes(b'{"vocab": [1]}')
    (data / "expected.json").write_text(json.dumps({"version": __version__, "flashcards": len(pack.flashcards), "questions": len(pack.questions),
                                                    "lectures": {"job-biochem": str(biochem), "job-collagen": str(renamed)},
                                                    "transcript_first_line": (biochem / "local_clean_transcript.txt").read_text(encoding="utf-8").splitlines()[0]}), encoding="utf-8")
    print("profile written:", data)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
