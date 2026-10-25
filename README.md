<p align="center"><img src="assets/brand/meletra_logo.png" alt="MELETRA — Medical Learning &amp; Transcription" width="220"></p>

# MELETRA

**Medical Learning & Transcription.**

MELETRA turns lecture audio, video and documents into a local transcript, medical-term review, summary, key concepts, flashcards and questions. Audio, video, PDFs and images are processed on the user's computer. Sharing is off unless the user explicitly enables it.

## Public build mirror

This is a history-free, privacy-reviewed source snapshot of MELETRA 4.1.2. It exists so the Apple Silicon release can be reproduced and tested on GitHub's public `macos-15` arm64 runner without exposing the private development repository or any lecture data.

- Private Git history, branches, benchmark labels and real-lecture review tools are not included.
- The repository contains no user profile, lecture recording, transcript, model weight or signing credential.
- The pinned Whisper Large-v3 MLX weights are downloaded by CI, verified against the SHA-256 values in `config/app_models.json`, and placed inside the offline app.
- The packaged app is tested after being copied out of the DMG, with network access blocked during application runs.
- A build without Apple credentials is ad-hoc signed and not notarized. Its release notes state that limitation explicitly.

The canonical installer page is [MELETRA Downloads](https://twonezerotwo-hue.github.io/whispermedical-downloads/). A Mac download should appear there only after the build, architecture audit, checksum verification and packaged-app tests pass.

## Reproduce the Apple Silicon build

Open **Actions → build-macos-arm64 → Run workflow** and choose:

- `recon`: verify the runner and native dependency set.
- `tests`: run the complete test set included in this sanitized snapshot.
- `build`: require green tests; build the public medical knowledge pack; download and verify the pinned MLX model; build, audit and sign the app; create and mount-test the DMG; run packaged offline transcription/UI/OCR/Study checks; publish verified split DMG parts as a prerelease.
- `build-fast`: diagnose packaging only. It is never published as the verified release.

The exact release evidence is emitted in `release_manifest.json`, `packaged_validation.json`, `test_counts.json`, the whole-DMG SHA-256 file and the workflow logs.

## Local development

Python 3.11 is required. On Apple Silicon:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-macos.txt
python -m pip install --no-deps rapidocr-onnxruntime==1.4.4
python -m pytest -q tests
```

Use `python scripts/build_macos_apple_silicon_offline.py --dry-run` on any platform to inspect the release plan. The actual build deliberately refuses to run anywhere except native macOS arm64.

## Downloads and provenance

- [MELETRA download page](https://twonezerotwo-hue.github.io/whispermedical-downloads/)
- [Public distribution repository](https://github.com/twonezerotwo-hue/whispermedical-downloads)
- Snapshot provenance: private source commit `8156a1a7f639a687b8880e6ecb269eff434552e1` (the released 4.1.2 source state; its private history is not copied here)

No license grant is added by this mirror. Third-party components and model notices are collected inside the built app.
