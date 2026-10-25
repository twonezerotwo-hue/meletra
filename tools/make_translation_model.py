"""Make the offline translation model (M2M100 418M, MIT licence) in the folder layout the program expects, and write its SHA-256 list.

    python tools/make_translation_model.py OUT_DIR            OUT_DIR/translation-m2m100/ + translation-m2m100.sha256.json

Needs, on the machine that runs this (NOT part of the program): torch, transformers, sentencepiece, ctranslate2 (its converter) and an internet connection for the one-time
download of facebook/m2m100_418M from Hugging Face (about 1.9 GB). The result is the INT8 CTranslate2 model (about 0.5 GB) plus the model's SentencePiece file:

    translation-m2m100/model.bin  config.json  shared_vocabulary.json  sentencepiece.bpe.model

Copy that folder into the program's ``models`` folder (Settings -> Models shows where) to switch every Output Language on. Hosting it for a one-click download is the owner's
next step (docs/v4_1/RELEASE_NOTES.md): upload the folder as a release asset and add its address and the SHA-256 list to the model registry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.translation.m2m100 import MODEL_FOLDER, REQUIRED_FILES  # noqa: E402

REPO = "facebook/m2m100_418M"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


PACK_VERSION = 1
OFFERED = ("en", "tr", "de", "hu", "ko", "zh", "hi", "nl", "fr", "es")      # the Output Languages the program offers (the model reads about 100 languages)


def source_revision(source: Path | None) -> str:
    """The Hugging Face commit the weights came from (read from the download metadata when the model was fetched into a folder)."""
    if source is not None:
        meta = Path(source) / ".cache" / "huggingface" / "download" / "config.json.metadata"
        try:
            return meta.read_text(encoding="utf-8").splitlines()[0].strip()
        except OSError:
            return ""
    return ""


def write_pack_json(target: Path, revision: str = "") -> Path:
    """``pack.json``: what this folder is, so the program (and anyone reading it) knows the pack version, the source model and its licence. The translation cache uses ``pack_version``."""
    import ctranslate2

    data = {"name": "translation-m2m100", "pack_version": PACK_VERSION, "model": REPO, "source_revision": revision, "licence": "MIT (Facebook AI Research / Meta)",
            "format": "CTranslate2 int8", "converted_with": f"ctranslate2 {ctranslate2.__version__}", "languages": list(OFFERED),
            "note": "Neural machine translation, one direct step between any two of the listed languages (no pivot through English). Quality is that of a 418M-parameter model: "
                    "usually understandable, sometimes wrong on rare medical terms - the original transcript is always kept."}
    path = Path(target) / "pack.json"
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    return path


def describe(out: Path, source: Path | None = None) -> dict:
    """The file list (sizes and SHA-256) of a finished pack folder."""
    target = Path(out) / MODEL_FOLDER
    if not (target / "pack.json").is_file():
        write_pack_json(target, source_revision(source))
    files = {p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in sorted(target.iterdir()) if p.is_file()}
    listing = {"model": REPO, "licence": "MIT (Facebook AI Research)", "format": "CTranslate2 int8", "pack_version": PACK_VERSION, "files": files}
    (Path(out) / f"{MODEL_FOLDER}.sha256.json").write_text(json.dumps(listing, indent=1), encoding="utf-8")
    return listing


def make(out: Path, source: Path | None = None) -> dict:
    """``source``: a folder that already holds facebook/m2m100_418M (config, pytorch_model.bin, sentencepiece.bpe.model, tokenizer files); without it the model is downloaded."""
    from ctranslate2.converters import TransformersConverter

    target = Path(out) / MODEL_FOLDER
    if target.exists():
        shutil.rmtree(target)
    if source is not None:
        TransformersConverter(str(source)).convert(str(target), quantization="int8")
        shutil.copyfile(Path(source) / "sentencepiece.bpe.model", target / "sentencepiece.bpe.model")
    else:
        from huggingface_hub import hf_hub_download

        TransformersConverter(REPO).convert(str(target), quantization="int8")
        shutil.copyfile(hf_hub_download(REPO, "sentencepiece.bpe.model"), target / "sentencepiece.bpe.model")
    missing = [name for name in REQUIRED_FILES if not (target / name).is_file()]
    if missing:
        raise SystemExit("the converted model is incomplete: " + ", ".join(missing))
    return describe(out, source)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--source", type=Path, default=None, help="a folder that already holds facebook/m2m100_418M (no download)")
    ap.add_argument("--describe-only", action="store_true", help="only (re)write pack.json and the SHA-256 list of an already converted folder")
    a = ap.parse_args(argv)
    listing = describe(a.out, a.source) if a.describe_only else make(a.out, a.source)
    print(json.dumps({k: v["bytes"] for k, v in listing["files"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
