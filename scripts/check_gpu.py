"""CHECK_GPU: report whether the NVIDIA GPU is usable for transcription.

Usage:  python scripts/check_gpu.py [--no-inference]
Exit code 0 when READY, 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import diagnostics as diag  # noqa: E402
from src.platform_utils import configure_utf8_console  # noqa: E402


def main() -> int:
    configure_utf8_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-inference", action="store_true",
                        help="skip the tiny Whisper inference test (no model download)")
    args = parser.parse_args()

    print("GPU STATUS")
    print("-----------")
    results = diag.run_all_checks(with_inference=not args.no_inference)
    by_name = {r.name: r for r in results}

    smi = by_name["nvidia-smi"].data
    if smi:
        print(f"GPU: {smi['name']}")
        print(f"VRAM: ~{smi['total_mib'] / 1024:.0f} GB total, {smi['free_mib'] / 1024:.1f} GB free right now")
        print(f"Driver: {smi['driver']}")
    else:
        print("GPU: not detected")

    print()
    for res in results:
        print(f"{res.name + ':':<24} {res.status:<5} {res.detail}")
    ffmpeg = diag.check_ffmpeg()
    print(f"{'FFmpeg:':<24} {ffmpeg.status:<5} {ffmpeg.detail}")

    warnings = diag.vram_warnings()
    if warnings:
        print("\nWARNING:")
        for line in warnings:
            print(f"  - {line}")

    gpu_checks = [by_name[n].status for n in ("nvidia-smi", "PyTorch CUDA", "CTranslate2 CUDA")]
    inference = by_name["Whisper GPU inference"].status
    ready = all(s == diag.PASS for s in gpu_checks) and inference in (diag.PASS, diag.SKIP)
    if not args.no_inference:
        ready = ready and inference == diag.PASS
    print()
    print(f"OVERALL: {'READY' if ready else 'NOT READY'}")
    if not ready:
        print()
        print(diag.CUDA_HELP)
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
