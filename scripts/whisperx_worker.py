"""Runs INSIDE the optional ``.venv-whisperx`` environment (never in the main one).

    python scripts/whisperx_worker.py <request.json> <result.json>

WhisperX pins torch~=2.8 and huggingface-hub<1.0, which conflict with the main environment,
so the main app talks to this script through JSON files and a subprocess. The transcript text
is never altered: only word timings (forced alignment) and speaker labels are produced.

Diarization needs a Hugging Face token in the HF_TOKEN environment variable and accepting the
terms of the pyannote model on huggingface.co. It is OFF unless the request asks for it.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


def _word(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "word": entry.get("word", ""),
        "start": entry.get("start"),
        "end": entry.get("end"),
        "score": entry.get("score"),
        "speaker": entry.get("speaker"),
    }


def run(request: dict[str, Any]) -> dict[str, Any]:
    import whisperx  # imported here so a missing install gives a clear error to the caller

    device = request["device"]
    audio = whisperx.load_audio(request["audio_path"])
    segments = [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in request["segments"]]
    result: dict[str, Any] = {"aligned": False, "diarized": False, "notes": [], "segments": []}

    model_a, metadata = whisperx.load_align_model(language_code=request["language"], device=device)
    aligned = whisperx.align(segments, model_a, metadata, audio, device, return_char_alignments=False)
    result["aligned"] = True

    if request.get("diarize"):
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("diarization needs a Hugging Face token in the HF_TOKEN environment variable")
        from whisperx.diarize import DiarizationPipeline

        pipeline = DiarizationPipeline(token=token, device=device)
        diarization = pipeline(audio, min_speakers=request.get("min_speakers"), max_speakers=request.get("max_speakers"))
        aligned = whisperx.assign_word_speakers(diarization, aligned)
        result["diarized"] = True

    for seg in aligned["segments"]:
        result["segments"].append({
            "start": seg.get("start"), "end": seg.get("end"), "text": seg.get("text", ""),
            "speaker": seg.get("speaker"), "words": [_word(w) for w in seg.get("words", [])],
        })
    return result


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    request_path, result_path = Path(argv[1]), Path(argv[2])
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        result = run(request)
    except Exception as exc:  # noqa: BLE001 - the parent decides how to report it
        result = {"error": f"{type(exc).__name__}: {exc}"}
        Path(result_path).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        return 1
    Path(result_path).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
