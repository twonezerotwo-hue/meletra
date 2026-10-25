"""Before / after measurement of the medical expert layer on one real recording.

    python scripts/medical_benchmark.py --before DIR_OLD --after DIR_NEW [--json out.json]

``DIR_OLD`` / ``DIR_NEW`` are output folders of the transcriber (local_transcript.json, optionally medical_corrected_segments.json). Nothing
here is a human judgement: every number is a count taken from the transcripts, and the /10 scores are a stated formula over those counts.
The recording itself is never read and never added to the repository.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.medical import language as lang_mod  # noqa: E402
from src.medical import repair as repair_mod  # noqa: E402
from src.medical.lexicon import shared_lexicon  # noqa: E402

#: terms the neuroanatomy lecture is known to contain (checked by recall in each transcript)
EXPECTED = ["Cajal", "Golgi", "Nissl", "cresyl violet", "perikaryon", "axon hillock", "dendritic spine", "GABA", "axodendritic", "axosomatic",
            "axoaxonic", "anterograde", "retrograde", "Schwann", "oligodendrocyte", "Remak", "node of Ranvier", "saltatory conduction", "radial glia",
            "Müller glia", "tanycyte", "GFAP", "astrocyte", "microglia", "blood-brain barrier", "glial scar"]
#: other correct forms of an expected term (plurals, Latin plurals, spelling variants)
VARIANTS = {"perikaryon": ["perikarya"], "node of Ranvier": ["nodes of Ranvier", "node of ranvier"], "dendritic spine": ["dendritic spines"],
            "oligodendrocyte": ["oligodendrocytes"], "tanycyte": ["tanycytes"], "astrocyte": ["astrocytes"], "Müller glia": ["Muller glia", "Müller glial"],
            "GABA": ["gamma-aminobutyric acid", "gamma aminobutyric acid"], "microglia": ["microglial"], "radial glia": ["radial glial"],
            "glial scar": ["glial scars"], "axon hillock": ["axon hillocks"], "Cajal": ["Ramón y Cajal", "Ramon y Cajal"]}
#: mis-hearings of the real failure list: (family, regular expression of the WRONG forms)
KNOWN_ERRORS = [
    ("crazy violet", r"crazy\s+violet"), ("nisal bodies", r"\bnisal\b"), ("gamma-amino-butyric ester", r"butyric\s+ester"),
    ("exodendritic", r"\bexodendritic"), ("exosomatic", r"\bexosomatic"), ("regulatory conduction", r"regulatory\s+conduction"),
    ("RIMAC / REM fiber", r"\brimac\b|\bremac\b"), ("tannicides / tummy sites", r"\btannicides?\b|\btummy\s+sites?\b"),
    ("mitral glial", r"mitral\s+glial"), ("monosaccharide origin", r"monosaccharides?\b(?=[^.]{0,40}(?:origin|enter|population|blood))|monosaccharide,"),
]


def fold(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower().replace("-", " ")).strip()


def load(folder: Path) -> dict:
    data = json.loads((folder / "local_transcript.json").read_text(encoding="utf-8"))
    segs = data["segments"]
    medical = None
    path = folder / "medical_corrected_segments.json"
    if path.is_file():
        medical = json.loads(path.read_text(encoding="utf-8"))
    return {"segments": segs, "medical": medical, "duration": data.get("duration"),
            "metadata": json.loads((folder / "metadata.json").read_text(encoding="utf-8")) if (folder / "metadata.json").is_file() else {},
            "quality": json.loads((folder / "medical_quality_report.json").read_text(encoding="utf-8")) if (folder / "medical_quality_report.json").is_file() else None}


def measure(texts: list[str], times: list[tuple[float, float]], label: str) -> dict:
    index = shared_lexicon().subset(None)
    medical = lambda w: bool(index.exact(w)) or w in index.known                     # noqa: E731
    joined = fold(" ".join(texts))
    errors = {name: len(re.findall(pattern, " ".join(texts), re.I)) for name, pattern in KNOWN_ERRORS}
    found = [t for t in EXPECTED if any(fold(v) in joined or fold(v).replace(" ", "") in joined.replace(" ", "") for v in [t, *VARIANTS.get(t, [])])]
    drift = lang_mod.drifted_segments(texts, "en", medical)
    items = [repair_mod.Item(i, s, e, t) for i, ((s, e), t) in enumerate(zip(times, texts)) if t.strip()]
    _, dups = repair_mod.remove_duplicates(items)
    stutters = sum(repair_mod.collapse_repeats(t)[1] for t in texts)
    return {"label": label, "segments": len(texts), "words": sum(len(t.split()) for t in texts), "known_error_instances": sum(errors.values()),
            "known_error_families_present": sum(1 for v in errors.values() if v), "errors_by_family": errors,
            "expected_terms_found": len(found), "expected_terms_total": len(EXPECTED), "expected_terms_missing": [t for t in EXPECTED if t not in found],
            "language_drift_segments": len(drift), "duplicate_segments": len(dups), "repeated_phrases": stutters,
            "timestamp_overlaps": repair_mod.count_overlaps(times), "reversed_durations": repair_mod.count_reversed(times)}


def score(m: dict, first: dict) -> dict:
    """Transparent /10 scores from the measured counts (not a human grading)."""
    total_err = max(first["known_error_instances"], 1)
    terminology = 10.0 * (0.5 * m["expected_terms_found"] / m["expected_terms_total"] + 0.5 * (1.0 - min(1.0, m["known_error_instances"] / total_err)))
    words = max(m["words"], 1)
    language = 10.0 * (1.0 - min(1.0, 4.0 * m["language_drift_segments"] / max(m["segments"], 1)))
    repetition = 10.0 * (1.0 - min(1.0, (m["duplicate_segments"] + m["repeated_phrases"]) / max(0.02 * m["segments"], 1)))
    timing = 10.0 * (1.0 - min(1.0, (m["timestamp_overlaps"] + m["reversed_durations"]) / max(0.02 * m["segments"], 1)))
    overall = 0.35 * terminology + 0.30 * language + 0.175 * repetition + 0.175 * timing
    return {"medical_terminology": round(terminology, 1), "language_stability": round(language, 1), "duplicate_repair": round(repetition, 1),
            "timestamp_consistency": round(timing, 1), "overall_transcript": round(overall, 1), "words": words}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True, type=Path)
    ap.add_argument("--after", required=True, type=Path)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    before, after = load(args.before), load(args.after)
    rows = {}
    rows["before (raw, old run)"] = measure([s["text"] for s in before["segments"]], [(s["start"], s["end"]) for s in before["segments"]], "before raw")
    rows["after (raw, new run)"] = measure([s["text"] for s in after["segments"]], [(s["start"], s["end"]) for s in after["segments"]], "after raw")
    if after["medical"]:
        rows["after (medical corrected)"] = measure([s["text"] for s in after["medical"]], [(s["start"], s["end"]) for s in after["medical"]], "after medical")
    first = rows["before (raw, old run)"]
    out = {name: {**m, "scores": score(m, first)} for name, m in rows.items()}
    out["quality_report"] = after["quality"]
    text = json.dumps(out, indent=2, ensure_ascii=False)
    print(text)
    if args.json:
        args.json.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
