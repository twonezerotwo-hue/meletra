"""MELETRA - Medical Learning & Transcription (formerly WhisperMedical, before that Medical Lecture Transcriber): medical lecture transcription and study (CLI + desktop app)."""

import sys
from pathlib import Path

__version__ = "4.1.2"

APP_NAME = "MELETRA"                             # the product name shown everywhere (window title, About, shortcuts, installer)
APP_TAGLINE = "Medical Learning & Transcription"
PRODUCT_ID = "MELETRA"                           # the file identity: MELETRA.exe, MELETRA-Setup-Windows-x64.exe, MELETRA.app
APP_ID = "MedicalLectureTranscriber"             # internal, never shown: data folders (%LOCALAPPDATA%), credential keys, registry id - UNCHANGED so upgrades keep every lecture, setting and model
COMPANY = "Touche Holding"

#: True inside the PyInstaller build (no Python installation, no source tree).
FROZEN: bool = bool(getattr(sys, "frozen", False))

if FROZEN:
    #: folder holding MELETRA.exe (bin/, libs/ and, in the portable/offline packages, models/)
    APP_DIR: Path = Path(sys.executable).resolve().parent
    #: read-only resources shipped with the app (config/, medical_terms/); PyInstaller's _internal folder
    RESOURCE_ROOT: Path = Path(getattr(sys, "_MEIPASS", APP_DIR))
else:
    RESOURCE_ROOT = Path(__file__).resolve().parent.parent
    APP_DIR = RESOURCE_ROOT

#: kept for the code that predates the desktop app: read-only resources (config/, medical_terms/), never user data
PROJECT_ROOT: Path = RESOURCE_ROOT
