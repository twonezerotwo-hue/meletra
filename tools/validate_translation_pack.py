"""Validate an offline translation pack (translation-m2m100) the way the program uses it: files and checksums, license record, loading, REAL translation through the same service
the desktop program uses (direct from each segment's language, medical terms protected), memory, speed, and that nothing touches the network.

    python tools/validate_translation_pack.py PACK_PARENT_DIR [--manifest translation-m2m100.sha256.json] [--out report.json]

``PACK_PARENT_DIR`` holds the folder ``translation-m2m100``. The sentences are written for this check (no recordings, no copyrighted text). The report states what was measured; quality is
shown as the sentences themselves, not as a score (there is no human reference for them).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SENTENCES = {
    "ko": ["오늘은 신경계의 구조에 대해 이야기하겠습니다.", "환자는 심한 두통과 발열을 호소했으며 MRI에서 소뇌의 병변이 관찰되었습니다.", "베바시주맙은 높은 mTORC1 활성 때문에 시작되었습니다."],
    "hi": ["आज हम तंत्रिका तंत्र की संरचना के बारे में बात करेंगे।", "मरीज को तेज सिरदर्द और बुखार की शिकायत थी और MRI में अनुमस्तिष्क में घाव दिखाई दिया।", "ATP का उत्पादन माइटोकॉन्ड्रिया में होता है।"],
    "en": ["Today we will talk about the structure of the nervous system.", "The patient presented with severe headache and fever, and the MRI showed a lesion in the cerebellum.",
           "Bevacizumab was started because of high mTORC1 activity, and ATP is produced in the mitochondria."],
    "tr": ["Bugün sinir sisteminin yapısından bahsedeceğiz.", "Hasta şiddetli baş ağrısı ve ateş şikayetiyle başvurdu; MRI'da beyincikte lezyon görüldü."],
    "de": ["Heute sprechen wir über den Aufbau des Nervensystems.", "Der Patient klagte über starke Kopfschmerzen und Fieber; die MRI zeigte eine Läsion im Kleinhirn."],
}
PAIRS = [("ko", "en"), ("hi", "en"), ("en", "tr"), ("ko", "tr"), ("hi", "tr"), ("en", "de"), ("tr", "en"), ("de", "en"), ("en", "ko"), ("en", "hi"), ("en", "zh"), ("en", "fr"), ("en", "es"),
         ("en", "hu"), ("en", "nl")]
PROTECTED = ["MRI", "mTORC1", "ATP"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def peak_memory_mib() -> float:
    sys.path.insert(0, str(ROOT / "tools"))
    from eval_multilingual import peak_memory_mib as peak  # noqa: PLC0415

    return peak() or 0.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parent", type=Path)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    from src.multilingual import source as source_mod
    from src.multilingual.source import SourceSegment, SourceTranscript
    from src.translation.m2m100 import MODEL_FOLDER, REQUIRED_FILES, find_model
    from src.translation.service import TranslationService

    report: dict = {"problems": []}
    folder = args.parent / MODEL_FOLDER
    manifest_path = args.manifest or args.parent / f"{MODEL_FOLDER}.sha256.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    report["files"] = {}
    for name in sorted({p.name for p in folder.iterdir() if p.is_file()}):
        path = folder / name
        want = (manifest.get("files") or {}).get(name, {})
        actual = sha256(path)
        ok = bool(want) and want.get("sha256") == actual and want.get("bytes") == path.stat().st_size
        report["files"][name] = {"bytes": path.stat().st_size, "sha256": actual, "matches_manifest": ok}
        if want and not ok:
            report["problems"].append(f"{name}: does not match the manifest")
    missing = [n for n in REQUIRED_FILES if not (folder / n).is_file()]
    if missing:
        report["problems"].append("missing files: " + ", ".join(missing))
    pack = json.loads((folder / "pack.json").read_text(encoding="utf-8")) if (folder / "pack.json").is_file() else {}
    report["pack"] = pack
    report["license"] = pack.get("licence")
    report["find_model"] = str(find_model(args.parent)) if find_model(args.parent) else None

    # --- nothing may touch the network while translating
    attempts: list[str] = []
    real_connect = socket.socket.connect

    def blocked(self, address, *a, **k):
        attempts.append(str(address))
        raise OSError("network access is not allowed during the translation check")

    socket.socket.connect = blocked  # type: ignore[assignment]
    try:
        lecture = Path(tempfile.mkdtemp(prefix="mlt_pack_"))
        rows, number = [], 0
        for code, texts in SENTENCES.items():
            for text in texts:
                number += 1
                rows.append(SourceSegment(number, number * 6.0, number * 6.0 + 5.0, text, detected_language=code, language_source="detector"))
        source_mod.write(lecture, SourceTranscript(lecture_id="pack-check", segments=rows))
        before = peak_memory_mib()
        started = time.perf_counter()
        service = TranslationService([args.parent], device="cpu")
        service.translate(lecture, "en")                                  # loads the model
        load_and_first = time.perf_counter() - started
        report["memory_mib"] = {"before": before, "peak_after_first_translation": peak_memory_mib()}
        report["translations"] = {}
        timings = {}
        for target in sorted({t for _s, t in PAIRS}):
            t0 = time.perf_counter()
            doc = service.translate(lecture, target)
            timings[target] = round(time.perf_counter() - t0, 1)
            report["translations"][target] = [{"from": s.source_language, "source": s.source_text, "translation": s.text, "status": s.translation_status, "flags": s.flags} for s in doc.segments]
        report["seconds"] = {"load_and_first_language": round(load_and_first, 1), "per_target_language": timings}
        report["memory_mib"]["peak_after_all"] = peak_memory_mib()
        again = time.perf_counter()
        service.translate(lecture, "tr")
        report["cache_reopen_seconds"] = round(time.perf_counter() - again, 3)
    finally:
        socket.socket.connect = real_connect  # type: ignore[assignment]
    report["network_attempts"] = attempts
    if attempts:
        report["problems"].append("the translation tried to use the network: " + ", ".join(attempts[:3]))

    # --- medical terms that must come through unchanged (Latin-script terms in the target text)
    kept, lost = [], []
    for target, items in report["translations"].items():
        for item in items:
            for term in PROTECTED:
                if term in item["source"]:
                    (kept if term in item["translation"] else lost).append(f"{item['from']}->{target}: {term}")
    report["protected_terms"] = {"kept": len(kept), "lost": lost}
    report["required_pairs"] = {f"{s}->{t}": any(i["from"] == s and i["status"] in ("translated", "same_language") for i in report["translations"].get(t, [])) for s, t in PAIRS}
    for pair, ok in report["required_pairs"].items():
        if not ok:
            report["problems"].append(f"no translation for {pair}")
    report["verdict"] = "PASS" if not report["problems"] else "FAIL"
    text = json.dumps(report, ensure_ascii=False, indent=1)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text[:3000])
    print("VERDICT", report["verdict"], "| problems:", report["problems"])
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
