"""Anonymous, resumable, verified download of the offline translation pack through the program's own model service, then a real translation from the downloaded bytes.

    python tools/check_pack_download.py [--cancel-after-mib 40] [--out report.json]

No token is used (the request is an ordinary anonymous HTTPS download). The download is cancelled once part-way, resumed, and every file is checked against the SHA-256 in
config/app_models.json before it is moved into place. Then the pack is used for a real translation (Korean / Hindi -> English, English -> Turkish) offline.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cancel-after-mib", type=float, default=40.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    from src.services.model_service import TRANSLATION_PACK_KEY, ModelCancelled, ModelService
    from src.services.paths import AppPaths

    work = Path(tempfile.mkdtemp(prefix="mlt_pack_dl_"))
    paths = AppPaths(work / "data").ensure()
    service = ModelService(paths, app_dir=work / "app")
    report: dict = {"problems": [], "steps": []}
    info = service.info(TRANSLATION_PACK_KEY)
    report["url_of_model_bin"] = service._url(info, "model.bin")
    progress_seen = []

    def progress(event):
        progress_seen.append(event.get("bytes_done", 0))

    started = time.perf_counter()
    limit = args.cancel_after_mib * 2 ** 20
    try:
        service.download(TRANSLATION_PACK_KEY, progress, lambda: bool(progress_seen) and progress_seen[-1] >= limit)
        report["problems"].append("the first download was not interrupted")
    except ModelCancelled:
        report["steps"].append({"step": "interrupted", "bytes_kept": service.partial_bytes(TRANSLATION_PACK_KEY), "seconds": round(time.perf_counter() - started, 1)})
    resumed_from = service.partial_bytes(TRANSLATION_PACK_KEY)
    if resumed_from <= 0:
        report["problems"].append("no partial download was kept")
    progress_seen.clear()
    t1 = time.perf_counter()
    folder = service.download(TRANSLATION_PACK_KEY, progress, None)
    report["steps"].append({"step": "resumed and finished", "resumed_from_bytes": resumed_from, "seconds": round(time.perf_counter() - t1, 1), "folder": "<models>/" + folder.name})
    state = service.state(TRANSLATION_PACK_KEY)
    verify = service.verify(TRANSLATION_PACK_KEY)
    report["installed"] = state.installed
    report["verify_ok"] = verify.ok
    report["verify_problems"] = verify.problems
    if not (state.installed and verify.ok):
        report["problems"].append("the downloaded pack is not installed / does not verify")

    # damage one byte of a small file: verification must notice
    victim = folder / "config.json"
    original = victim.read_bytes()
    victim.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    report["damaged_file_detected"] = not service.verify(TRANSLATION_PACK_KEY).ok
    victim.write_bytes(original)
    if not report["damaged_file_detected"]:
        report["problems"].append("a damaged file was not detected")

    # real translation from the downloaded bytes, offline
    from src.multilingual import source as source_mod
    from src.multilingual.source import SourceSegment, SourceTranscript
    from src.translation.service import TranslationService

    lecture = work / "lecture"
    rows = [SourceSegment(1, 0, 5, "오늘은 신경계의 구조에 대해 이야기하겠습니다.", detected_language="ko", language_source="detector"),
            SourceSegment(2, 6, 11, "आज हम तंत्रिका तंत्र की संरचना के बारे में बात करेंगे।", detected_language="hi", language_source="detector"),
            SourceSegment(3, 12, 17, "The cerebellum coordinates movement.", detected_language="en", language_source="detector")]
    source_mod.write(lecture, SourceTranscript(lecture_id="download-check", segments=rows))
    real = socket.socket.connect

    def blocked(self, address, *a, **k):
        raise OSError("no network while translating")

    socket.socket.connect = blocked  # type: ignore[assignment]
    try:
        english = TranslationService([paths.models], device="cpu").translate(lecture, "en")
        turkish = TranslationService([paths.models], device="cpu").translate(lecture, "tr")
    finally:
        socket.socket.connect = real  # type: ignore[assignment]
    report["translation_en"] = [s.text for s in english.segments]
    report["translation_tr"] = [s.text for s in turkish.segments]
    if not (english.complete and turkish.complete):
        report["problems"].append("the translation from the downloaded pack was incomplete")
    report["verdict"] = "PASS" if not report["problems"] else "FAIL"
    text = json.dumps(report, ensure_ascii=False, indent=1)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
