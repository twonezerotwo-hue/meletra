"""Measure the multilingual pipeline on recordings whose truth is known (``tools/make_multilingual_testset.py``): which language each part was recognised as, how much of what was
spoken became words, and how close the words are to the human-checked reference.

    python tools/eval_multilingual.py NAME [NAME ...] [--device cpu|cuda|auto] [--recordings DIR] [--out DIR] [--language auto]

It drives the real worker (the same code path as the desktop program) on each recording and reads the files it wrote. Nothing is uploaded; the recordings and the results stay in the
git-ignored ``benchmarks/multilingual`` folder. The reference texts come from the dataset (human-checked), so the error rates are measured against an independent reference; where a
recording has no reference (a real video) only language and coverage statistics are written, never an error rate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading
import time
import unicodedata
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CJK_CHARACTER_LANGUAGES = ("ko", "zh", "ja")


# ---------------------------------------------------------------------------------------------------------------- memory
def peak_memory_mib() -> Optional[float]:
    """Peak working set of this process in MiB (Windows); None elsewhere."""
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        return None
    return round(counters.PeakWorkingSetSize / 2 ** 20, 0)


# ---------------------------------------------------------------------------------------------------------------- scoring
def normalize(text: str) -> str:
    """Lower-case, no punctuation or symbols, single spaces (the dataset's own 'normalized' text is used the same way)."""
    kept = []
    for ch in unicodedata.normalize("NFKC", text).casefold():
        category = unicodedata.category(ch)
        kept.append(" " if (category[0] in "PSZC" or ch in "।॥") else ch)
    return " ".join("".join(kept).split())


def edit_distance(a: list, b: list) -> int:
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        current = [i]
        for j, y in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def error_rate(reference: str, hypothesis: str, characters: bool) -> Optional[float]:
    ref, hyp = normalize(reference), normalize(hypothesis)
    ref_units = list(ref.replace(" ", "")) if characters else ref.split()
    hyp_units = list(hyp.replace(" ", "")) if characters else hyp.split()
    if not ref_units:
        return None
    return edit_distance(ref_units, hyp_units) / len(ref_units)


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(min(a1, b1) - max(a0, b0), 0.0)


def score(truth: dict[str, Any], segments: list[dict[str, Any]]) -> dict[str, Any]:
    """Per truth clip: the language the recogniser gave the words that fall into it, whether anything was written, and the error rate against the reference."""
    clips = []
    for clip in truth["timeline"]:
        start, end = clip["start"], clip["end"]
        inside = [s for s in segments if overlap(s["start"], s["end"], start, end) > 0.35 * max(min(s["end"] - s["start"], end - start), 0.1) and (s.get("text") or "").strip()]
        by_language: dict[str, float] = {}
        for s in inside:
            by_language[s.get("language") or "und"] = by_language.get(s.get("language") or "und", 0.0) + overlap(s["start"], s["end"], start, end)
        predicted = max(by_language, key=by_language.get) if by_language else None
        text = " ".join(s["text"].strip() for s in inside)
        characters = clip["language"] in CJK_CHARACTER_LANGUAGES
        rate = error_rate(clip.get("normalized") or clip.get("text") or "", text, characters) if text else 1.0
        clips.append({"id": clip["id"], "start": start, "end": end, "truth": clip["language"], "predicted": predicted, "language_ok": predicted == clip["language"], "written": bool(text),
                      "error_rate": None if rate is None else round(rate, 3), "unit": "CER" if characters else "WER", "languages_seen": {k: round(v, 1) for k, v in by_language.items()}})
    n = len(clips) or 1
    by_lang: dict[str, list[float]] = {}
    for c in clips:
        if c["error_rate"] is not None:
            by_lang.setdefault(c["truth"], []).append(c["error_rate"])
    return {"clips": clips, "language_accuracy": round(sum(c["language_ok"] for c in clips) / n, 3), "clips_written": round(sum(c["written"] for c in clips) / n, 3),
            "mean_error_rate_by_language": {k: round(sum(v) / len(v), 3) for k, v in by_lang.items()}}


# ---------------------------------------------------------------------------------------------------------------- the real worker
def run_pipeline(source: Path, device: str, language: str, output_language: str, work: Path) -> Path:
    os.environ["MLT_DATA_DIR"] = str(work / "data")
    from src.pipeline import job_key
    from src.services.job_service import JobStore
    from src.services.paths import get_paths
    from src.services.transcription_service import JobRequest, events_path, request_path, run_worker

    paths = get_paths()
    job_id = job_key(source)
    model_dir = Path(os.environ["LOCALAPPDATA"]) / "MedicalLectureTranscriber" / "models"
    request = JobRequest(source=str(source), output_dir=str(work / "out"), model_key="whisper-large-v3", language=language, device=device, medical_intelligence="on",
                         output_language=output_language, chunk_minutes=15.0, job_id=job_id, data_root=str(paths.root), extra_model_dirs=[str(model_dir)], parent_pid=os.getpid())
    JobStore(paths.jobs).create_or_get(job_id, source, output_dir=request.output_dir, checkpoint_dir=str(paths.checkpoints / job_id), model_key=request.model_key,
                                       language=request.language, subject="auto", mode="quality", pdf="", state="queued", last_error="")
    req = request_path(paths, job_id)
    request.save(req)
    events_path(paths, job_id).write_text("", encoding="utf-8")
    result: dict[str, Any] = {}
    thread = threading.Thread(target=lambda: result.update(code=run_worker(req)), daemon=True)
    thread.start()
    thread.join()
    if result.get("code") != 0:
        raise RuntimeError(f"the worker ended with code {result.get('code')} for {source.name}")
    return next((work / "out").iterdir())


def read_json(path: Path) -> Any:
    return json.loads(path.read_bytes().decode("utf-8"))


def evaluate(name: str, recordings: Path, out: Path, device: str, language: str, output_language: str) -> dict[str, Any]:
    wav = recordings / f"{name}.wav"
    truth_file = recordings / f"{name}.truth.json"
    truth = read_json(truth_file) if truth_file.is_file() else None
    work = Path(tempfile.mkdtemp(prefix="mlt_eval_"))
    started = time.perf_counter()
    folder = run_pipeline(wav, device, language, output_language, work)
    wall = time.perf_counter() - started
    transcript = read_json(folder / "local_transcript.json")
    metadata = read_json(folder / "metadata.json")
    segments = transcript.get("segments", [])
    import subprocess

    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""

    report: dict[str, Any] = {"code_version": git("rev-parse", "--short", "HEAD"), "code_modified": bool(git("status", "--porcelain", "--", "src")), "name": name, "device": device, "wall_seconds": round(wall, 1), "audio_seconds": truth["seconds"] if truth else None,
                              "peak_working_set_mib_so_far": peak_memory_mib(),
                              "segments": len(segments), "languages_in_transcript": sorted({s.get("language") or "und" for s in segments}),
                              "metadata_multilingual": (metadata.get("multilingual") or {}), "coverage": ((metadata.get("multilingual") or {}).get("coverage") or {}),
                              "plan": ((metadata.get("multilingual") or {}).get("plan") or {}), "translation": metadata.get("translation"),
                              "language_check": ((metadata.get("multilingual") or {}).get("language_check") or {}), "audio_leveling": ((metadata.get("multilingual") or {}).get("audio_leveling") or {}),
                              "output_folder": str(folder)}
    report.update(extra_statistics(segments, metadata))
    if truth:
        report["truth_languages"] = truth["languages"]
        report["score"] = score(truth, segments)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.{device}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report


def extra_statistics(segments: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    """What the quality audit and the text itself say about hallucinations: flagged segments, stock subtitle credits / sign-offs, the most repeated line, segments of unknown language."""
    from collections import Counter

    from src.multilingual.artifacts import is_artifact
    from src.multilingual.decode import line_key

    texts = [(s.get("text") or "").strip() for s in segments]
    lines = Counter(line_key(t) for t in texts if line_key(t))
    quality = metadata.get("quality") or {}
    return {"quality": {k: quality.get(k) for k in ("speech_coverage", "audible_coverage", "unrecovered_audible_gaps", "hallucination_flags", "repetition_flags", "timestamp_issues")},
            "uncertain_segments": sum(1 for s in segments if s.get("uncertain")), "artifact_segments": sum(1 for t in texts if t and is_artifact(t)),
            "most_repeated_line": max(lines.values()) if lines else 0, "unknown_language_segments": sum(1 for s in segments if (s.get("language") or "und") == "und")}


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--language", default="auto")
    ap.add_argument("--output-language", default="")
    ap.add_argument("--recordings", type=Path, default=ROOT / "benchmarks" / "multilingual" / "recordings")
    ap.add_argument("--out", type=Path, default=ROOT / "benchmarks" / "multilingual" / "results")
    args = ap.parse_args(argv)
    for name in args.names:
        report = evaluate(name, args.recordings, args.out, args.device, args.language, args.output_language)
        s = report.get("score") or {}
        print(f"{name:26s} {report['wall_seconds']:7.1f}s  languages {','.join(report['languages_in_transcript'])}  language accuracy {s.get('language_accuracy')}  written {s.get('clips_written')}  "
              f"error rates {s.get('mean_error_rate_by_language')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
