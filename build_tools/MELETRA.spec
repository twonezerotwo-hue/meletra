# -*- mode: python ; coding: utf-8 -*-
# PyInstaller one-folder build:  dist/MELETRA/MELETRA.exe
# Run through scripts/build_app.py (it prepares the icon and version resource first and assembles CUDA/FFmpeg afterwards).
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

ROOT = Path(SPECPATH).resolve().parent
ASSETS = ROOT / "build" / "assets"

datas = [(str(ROOT / "config"), "config"), (str(ROOT / "medical_terms"), "medical_terms"), (str(ROOT / "medical_vocab"), "medical_vocab")]   # medical_vocab: Medical Expert layer
datas += collect_data_files("faster_whisper")
# MELETRA brand: the logo and the icon sizes the window needs at run time (assets/brand, made by tools/make_brand_assets.py)
for _file in ("meletra_logo.png", "meletra_symbol.png", "meletra_icon_16.png", "meletra_icon_24.png", "meletra_icon_32.png", "meletra_icon_48.png", "meletra_icon_64.png",
              "meletra_icon_128.png", "meletra_icon_256.png", "meletra_icon_512.png", "meletra_icon_1024.png"):
    datas.append((str(ROOT / "assets" / "brand" / _file), (Path("assets/brand") / Path(_file).parent).as_posix()))                    # silero_vad_v6.onnx
# v4: the PUBLIC knowledge pack (redistributable sources only) with its notices; the app also runs without it (the Medical Expert works alone)
for _name in ("medical_knowledge_public.db", "medical_knowledge_public.db.NOTICE.txt", "medical_knowledge_public.db.manifest.json"):
    if (ROOT / "data" / _name).is_file():
        datas.append((str(ROOT / "data" / _name), "data"))
if (ROOT / "data" / "medical_knowledge_public.db").is_file():                       # a release that carries the pack must find it intact: the self-test fails loudly otherwise
    _marker = ROOT / "build" / "knowledge_required" / "REQUIRED"
    _marker.parent.mkdir(parents=True, exist_ok=True)
    _marker.write_text("This release ships the public medical knowledge pack: its absence or damage is a defect of the build.\n", encoding="utf-8")
    datas.append((str(_marker), "data"))
binaries = collect_dynamic_libs("ctranslate2") + collect_dynamic_libs("onnxruntime") + collect_dynamic_libs("av")

# 3.0: pictures / scanned pages (Pillow) and the speech / text recognition built into Windows (winrt projections, loaded on demand)
hiddenimports = collect_submodules("src") + collect_submodules("winrt") + [
    "yaml", "rapidfuzz", "rapidfuzz.fuzz", "pypdf", "soundfile", "onnxruntime", "tokenizers", "numpy", "av",
    "faster_whisper", "ctranslate2", "sentencepiece", "huggingface_hub", "PIL", "PIL.Image", "PIL.ImageOps", "PIL.ImageSequence", "PIL.ImageDraw", "PIL.ImageFont",
]

# Optional heavy runtimes that are NOT part of the app (the Turkish medical Transformers model is a source-install extra).
excludes = ["torch", "torchaudio", "torchvision", "transformers", "tkinter", "matplotlib", "pandas", "scipy", "IPython", "notebook",
            "pytest", "numba", "llvmlite", "sympy", "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtQml",
            "PySide6.QtQuick", "PySide6.Qt3DCore", "PySide6.QtMultimedia", "PySide6.QtCharts", "PySide6.QtDataVisualization",
            # the cross-platform text recognition (macOS only): importing it after the Windows runtime crashes the process
            "rapidocr_onnxruntime", "cv2", "shapely", "pyclipper"]

a = Analysis(
    [str(ROOT / "launcher.py")], pathex=[str(ROOT)], binaries=binaries, datas=datas, hiddenimports=hiddenimports,
    hookspath=[], runtime_hooks=[], excludes=excludes, noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True, name="MELETRA", debug=False, strip=False, upx=False,
    console=False, icon=str(ASSETS / "app.ico"), version=str(ASSETS / "version_info.txt"), contents_directory="_internal",
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="MELETRA")
