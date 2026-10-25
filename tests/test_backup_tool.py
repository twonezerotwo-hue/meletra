"""Phase 1: the backup taken before the repair is complete, verified byte for byte, and never changes what it copies."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("backup_user_data", ROOT / "tools" / "backup_user_data.py")
backup = importlib.util.module_from_spec(spec)
sys.modules["backup_user_data"] = backup
spec.loader.exec_module(backup)


def digest(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


def profile(tmp_path):
    appdata = tmp_path / "MedicalLectureTranscriber"
    (appdata / "jobs").mkdir(parents=True)
    (appdata / "jobs" / "job1.json").write_text(json.dumps({"job_id": "job1", "state": "completed"}), encoding="utf-8")
    (appdata / "settings").mkdir()
    (appdata / "settings" / "settings.json").write_text('{"language": "tr"}', encoding="utf-8")
    root = tmp_path / "Fixture Library" / "Medical Transcription"
    lecture = root / "FIXTURE_0005"
    lecture.mkdir(parents=True)
    (lecture / "local_clean_transcript.txt").write_text("Glycolysis converts glucose to pyruvate. " * 20, encoding="utf-8")
    (lecture / "metadata.json").write_text("{}", encoding="utf-8")
    models = tmp_path / "models"
    models.mkdir()
    (models / "model.bin").write_bytes(b"\x00\x01" * 5000)
    return appdata, root, models


def test_the_backup_copies_every_file_verifies_it_and_leaves_the_source_alone(tmp_path):
    appdata, root, models = profile(tmp_path)
    before = (digest(appdata), digest(root))
    dest = tmp_path / "backup"
    assert backup.main(["--dest", str(dest), "--appdata", str(appdata), "--root", str(root), "--reference-models", str(models)]) == 0
    inventory = json.loads((dest / "BACKUP_INVENTORY.json").read_text(encoding="utf-8"))
    assert inventory["all_verified"] and inventory["files"] == 4 and all(e["verified"] for e in inventory["entries"])
    assert inventory["referenced_models"][0]["sha256"] == hashlib.sha256((models / "model.bin").read_bytes()).hexdigest()         # models are recorded by hash, not copied
    assert not list(dest.rglob("model.bin"))
    copied = sorted(dest.rglob("local_clean_transcript.txt"))
    assert len(copied) == 1 and copied[0].read_bytes() == (root / "FIXTURE_0005" / "local_clean_transcript.txt").read_bytes()
    assert (digest(appdata), digest(root)) == before                                                                                  # nothing was changed by taking the backup


def test_a_second_backup_to_another_folder_does_not_touch_the_first(tmp_path):
    appdata, root, models = profile(tmp_path)
    first = tmp_path / "backup1"
    assert backup.main(["--dest", str(first), "--appdata", str(appdata), "--root", str(root)]) == 0
    kept = digest(first)
    (appdata / "jobs" / "job2.json").write_text("{}", encoding="utf-8")
    assert backup.main(["--dest", str(tmp_path / "backup2"), "--appdata", str(appdata), "--root", str(root)]) == 0
    assert digest(first) == kept                                                                                                      # the old backup is exactly as it was


def test_a_very_long_path_is_copied_too(tmp_path):
    appdata, root, models = profile(tmp_path)
    deep = root / ("a_very_long_folder_name_" * 9) / ("another_long_folder_name_" * 5)
    deep_path = Path("\\\\?\\" + str(deep)) if sys.platform == "win32" else deep
    deep_path.mkdir(parents=True)
    (deep_path / "transcript.txt").write_text("long path content", encoding="utf-8")
    dest = tmp_path / "backup"
    assert backup.main(["--dest", str(dest), "--appdata", str(appdata), "--root", str(root)]) == 0
    inventory = json.loads((dest / "BACKUP_INVENTORY.json").read_text(encoding="utf-8"))
    assert inventory["all_verified"] and inventory["files"] == 5                                         # the deep file is IN the backup, not silently left out
    assert any(e["original"].endswith("transcript.txt") and e["verified"] for e in inventory["entries"])
