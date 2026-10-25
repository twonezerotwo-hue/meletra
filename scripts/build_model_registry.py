"""Write config/app_models.json: the models the desktop app can download, with PINNED revisions and SHA-256 checksums.

The checksums come from files that were already downloaded from the Hugging Face repositories (the hub cache), so a download
by the app can be verified byte for byte, offline, without asking the server. For LFS files the cache blob name is the
SHA-256 of the file, which this script cross-checks.

Usage:  python scripts/build_model_registry.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "config" / "app_models.json"
HUB = Path.home() / ".cache" / "huggingface" / "hub"

MODELS = [
    {
        "key": "whisper-large-v3", "title": "Whisper Large-v3", "repo": "Systran/faster-whisper-large-v3",
        "engine": "faster_whisper", "models_yaml_key": "default", "folder": "whisper-large-v3", "recommended": True,
        "platforms": ["windows", "macos", "linux"], "recommended_on": ["windows", "linux", "macos-x86_64"],
        "runtime": "CTranslate2 (included in the app)",
        "description": "Recommended. OpenAI Whisper large-v3 converted for CTranslate2: best accuracy for English medical lectures.",
        "files": ["config.json", "model.bin", "preprocessor_config.json", "tokenizer.json", "vocabulary.json"],
        "required": ["config.json", "model.bin", "tokenizer.json"], "license": "MIT",
    },
    {
        "key": "turkish-medical-large-v3", "title": "Turkish Medical Large-v3", "repo": "turkmedstt/whisper-large-v3-turkish-medical",
        "engine": "transformers", "models_yaml_key": "turkish_medical", "folder": "turkish-medical-large-v3", "recommended": False,
        "platforms": ["windows", "linux"],
        "runtime": "PyTorch + Transformers (optional runtime pack, not part of the standard build)",
        "description": "Advanced. Whisper large-v3 fine-tuned on Turkish medical speech. Needs the optional Transformers runtime.",
        "files": ["added_tokens.json", "config.json", "generation_config.json", "merges.txt", "model.safetensors",
                  "normalizer.json", "preprocessor_config.json", "special_tokens_map.json", "tokenizer_config.json", "vocab.json"],
        "required": ["config.json", "model.safetensors", "vocab.json"], "license": "see model card",
    },
    {
        "key": "whisper-tiny-test", "title": "Whisper Tiny (self-test)", "repo": "Systran/faster-whisper-tiny",
        "engine": "faster_whisper", "models_yaml_key": "tiny_test", "folder": "whisper-tiny-test", "recommended": False,
        "hidden": True, "runtime": "CTranslate2 (included in the app)",
        "description": "75 MB model used by the self-test and the automated download tests.",
        "files": ["config.json", "model.bin", "tokenizer.json", "vocabulary.txt"],
        "required": ["config.json", "model.bin", "tokenizer.json"], "license": "MIT",
    },
]


# Models that are not in this PC's Hugging Face cache (the macOS one): revision, sizes and SHA-256 were read from the Hugging Face API
# (the LFS oid of weights.npz is its SHA-256; config.json was downloaded once and hashed).
STATIC_MODELS = [
    {
        "key": "whisper-large-v3-mlx", "title": "Whisper Large-v3 (Apple Silicon)", "repo": "mlx-community/whisper-large-v3-mlx",
        "engine": "mlx_whisper", "models_yaml_key": "mlx_large_v3", "folder": "whisper-large-v3-mlx", "recommended": False,
        "platforms": ["macos-arm64"], "recommended_on": ["macos-arm64"], "runtime": "MLX (Apple Silicon, included in the macOS app)",
        "description": "Recommended on a Mac with Apple Silicon. OpenAI Whisper large-v3 for Apple's MLX engine: the same model as on "
                       "Windows, running on the Mac's GPU.",
        "required": ["config.json", "weights.npz"], "license": "MIT", "revision": "49e6aa286ad60c14352c404340ded53710378a11",
        "files": [
            {"path": "config.json", "size": 269, "sha256": "34982ce6ae286095000f82ae9583b3431639e8b092bf60c961f203745e6500e3"},
            {"path": "weights.npz", "size": 3083520416, "sha256": "05ff791ce3630fae47e7c51004e9666204d786246ec07cac6110af768099b40d"},
        ],
    },
]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    registry = []
    for model in MODELS:
        cache = HUB / ("models--" + model["repo"].replace("/", "--"))
        revision = (cache / "refs" / "main").read_text(encoding="utf-8").strip()
        snapshot = cache / "snapshots" / revision
        files = []
        for name in model["files"]:
            path = snapshot / name
            digest = sha256_of(path)
            blob = path.resolve()
            if blob.name != path.name and len(blob.name) == 64 and blob.name != digest:      # LFS blob name == sha256
                raise SystemExit(f"checksum mismatch for {name}: cache blob {blob.name} != {digest}")
            files.append({"path": name, "size": path.stat().st_size, "sha256": digest})
            print(f"{model['key']:28} {name:28} {files[-1]['size']:>14,}  {digest[:16]}...")
        registry.append({**{k: v for k, v in model.items() if k != "files"}, "revision": revision, "files": files,
                         "approx_bytes": sum(f["size"] for f in files)})
    for model in STATIC_MODELS:
        registry.insert(1, {**model, "approx_bytes": sum(f["size"] for f in model["files"])})      # right after Large-v3
    OUT.write_text(json.dumps({"schema": 1, "models": registry}, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
