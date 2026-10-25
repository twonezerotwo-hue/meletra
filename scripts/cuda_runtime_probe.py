"""Which CUDA DLLs does CTranslate2 really need?  Used to keep ``libs/cuda`` as small as the measurements allow.

    python scripts/cuda_runtime_probe.py --cuda-dir DIR --model-dir MODEL [--drop cudnn_adv64_9.dll,...]

Runs a real Whisper encoder + decoder pass on the GPU in a child process whose PATH holds only Windows plus ``--cuda-dir``
(minus the ``--drop`` files, copied to a temp folder), so nothing else on the PC can supply a missing library.
Prints one JSON line: ``{"ok": true/false, "dropped": [...], "error": "...", "seconds": n}``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from build_tools.cleanenv import windows_only_env       # noqa: E402

CHILD = r"""
import json, os, sys, time
dll_dir, model_dir, audio_path = sys.argv[1], sys.argv[2], sys.argv[3]
os.add_dll_directory(dll_dir)
import numpy as np
from faster_whisper import WhisperModel, decode_audio
t = time.time()
model = WhisperModel(model_dir, device="cuda", compute_type="float16")
load = time.time() - t
if audio_path:
    audio = decode_audio(audio_path, sampling_rate=16000)
else:
    audio = (np.random.default_rng(0).standard_normal(16000 * 6) * 0.02).astype("float32")
t = time.time()
segments, info = model.transcribe(audio, beam_size=5, vad_filter=False, language="en", temperature=0.0)
text = " ".join(s.text.strip() for s in segments)
print(json.dumps({"ok": True, "load_s": round(load, 1), "infer_s": round(time.time() - t, 2), "text": text}))
"""


def probe(cuda_dir: Path, model_dir: Path, drop: list[str], audio: str = "", timeout: float = 240.0) -> dict:
    work = Path(tempfile.mkdtemp(prefix="mlt_cudaprobe_"))
    try:
        subset = work / "cuda"
        shutil.copytree(cuda_dir, subset)
        for name in drop:
            (subset / name).unlink(missing_ok=True)
        env = windows_only_env(work)
        python_home = str(Path(sys.executable).parent)
        env["PATH"] = os.pathsep.join([str(subset), env["PATH"], python_home])   # CTranslate2 loads cuBLAS by plain name: PATH decides
        env["HF_HUB_OFFLINE"] = "1"
        started = time.monotonic()
        proc = subprocess.Popen([sys.executable, "-c", CHILD, str(subset), str(model_dir), audio], env=env, cwd=str(work),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            return {"ok": False, "dropped": drop, "error": f"no answer within {timeout:.0f} s"}
        last = out.strip().splitlines()[-1] if out.strip() else ""
        if proc.returncode == 0 and last.startswith("{"):
            return {**json.loads(last), "dropped": drop}
        return {"ok": False, "dropped": drop, "error": (err.strip().splitlines() or [f"exit {proc.returncode}"])[-1][:300],
                "seconds": round(time.monotonic() - started, 1)}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cuda-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--drop", default="", help="comma separated DLL names to remove before the test")
    parser.add_argument("--audio", default="", help="real audio to decode (default: 6 s of noise)")
    args = parser.parse_args()
    result = probe(args.cuda_dir, args.model_dir, [d for d in args.drop.split(",") if d], args.audio)
    print(json.dumps(result))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
