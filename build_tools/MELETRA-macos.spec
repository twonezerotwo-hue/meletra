# -*- mode: python ; coding: utf-8 -*-
# PyInstaller build for macOS:  dist/v<version>-macos/MELETRA.app
# Run through scripts/build_macos_app.py (it writes the icon, the Info.plist values and the entitlements first). Not run on Windows.
import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

ROOT = Path(SPECPATH).resolve().parent
ASSETS = ROOT / "build" / "assets"
sys.path.insert(0, str(ROOT))
from build_tools import macos as mac_build                                    # noqa: E402
from src import __version__                                                    # noqa: E402

ARCH = os.environ.get("MLT_MAC_ARCH", "arm64")
datas = [(str(ROOT / "config"), "config"), (str(ROOT / "medical_terms"), "medical_terms"),
         (str(ROOT / "medical_vocab"), "medical_vocab")]                         # medical_vocab: the Medical Expert layer (was missing in 0.2/3.0)
datas += collect_data_files("faster_whisper")
# MELETRA brand: the logo and the icon sizes the window needs at run time (assets/brand, made by tools/make_brand_assets.py)
for _file in ("meletra_logo.png", "meletra_symbol.png", "meletra_icon_16.png", "meletra_icon_24.png", "meletra_icon_32.png", "meletra_icon_48.png", "meletra_icon_64.png",
              "meletra_icon_128.png", "meletra_icon_256.png", "meletra_icon_512.png", "meletra_icon_1024.png"):
    datas.append((str(ROOT / "assets" / "brand" / _file), (Path("assets/brand") / Path(_file).parent).as_posix()))                                  # silero_vad_v6.onnx
# v4: the PUBLIC knowledge pack with its notices (CI provides it as data/ before the build; without it the Medical Expert works alone)
for _name in ("medical_knowledge_public.db", "medical_knowledge_public.db.NOTICE.txt", "medical_knowledge_public.db.manifest.json"):
    if (ROOT / "data" / _name).is_file():
        datas.append((str(ROOT / "data" / _name), "data"))
if (ROOT / "data" / "medical_knowledge_public.db").is_file():                       # a release that carries the pack must find it intact: the self-test fails loudly otherwise
    _marker = ROOT / "build" / "knowledge_required" / "REQUIRED"
    _marker.parent.mkdir(parents=True, exist_ok=True)
    _marker.write_text("This release ships the public medical knowledge pack: its absence or damage is a defect of the build.\n", encoding="utf-8")
    datas.append((str(_marker), "data"))
binaries = collect_dynamic_libs("ctranslate2") + collect_dynamic_libs("onnxruntime") + collect_dynamic_libs("av")
hiddenimports = collect_submodules("src") + [
    "yaml", "rapidfuzz", "rapidfuzz.fuzz", "rapidfuzz.distance", "rapidfuzz.distance.Levenshtein", "pypdf", "soundfile", "onnxruntime", "tokenizers", "numpy", "av",
    "faster_whisper", "ctranslate2", "sentencepiece", "huggingface_hub", "keyring", "keyring.backends.macOS", "PIL", "PIL.Image", "PIL.ImageOps",
    "PIL.ImageSequence",
]
# 3.0 text recognition on the Mac (RapidOCR with its bundled models; the Windows runtime is not part of this build). UNVERIFIED on a Mac.
datas += collect_data_files("rapidocr_onnxruntime")
hiddenimports += collect_submodules("rapidocr_onnxruntime") + ["cv2", "shapely", "pyclipper"]
if ARCH == "arm64":                                                            # the optimized engine: Apple Silicon only
    datas += collect_data_files("mlx_whisper") + collect_data_files("mlx")
    binaries += collect_dynamic_libs("mlx")
    hiddenimports += collect_submodules("mlx") + collect_submodules("mlx_whisper")

excludes = ["torch", "torchaudio", "torchvision", "transformers", "tkinter", "matplotlib", "pandas", "scipy", "IPython", "notebook",
            "pytest", "numba", "llvmlite", "sympy", "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtQml",
            "PySide6.QtQuick", "PySide6.Qt3DCore", "PySide6.QtMultimedia", "PySide6.QtCharts", "PySide6.QtDataVisualization", "winrt"]

a = Analysis([str(ROOT / "launcher.py")], pathex=[str(ROOT)], binaries=binaries, datas=datas, hiddenimports=hiddenimports,
             hookspath=[], runtime_hooks=[], excludes=excludes, noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name=mac_build.PRODUCT_ID, debug=False, strip=False, upx=False, console=False,
          target_arch=("arm64" if ARCH == "arm64" else "x86_64"), codesign_identity=None, entitlements_file=str(ASSETS / "entitlements.plist"))
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name=mac_build.PRODUCT_ID)
app = BUNDLE(coll, name=mac_build.APP_BUNDLE, icon=str(ASSETS / "app.icns"), bundle_identifier=mac_build.BUNDLE_ID,
             version=__version__, info_plist=mac_build.info_plist(__version__, ARCH))
