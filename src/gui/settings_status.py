"""The read-only "Status" block of Settings > General: what the program will use for the next lecture, in plain words.

    Transcription model            Whisper Large-v3 (installed)
    Medical correction             Medical Expert v3 (on)
    Experimental Medical Intelligence  v4: Off
    Medical knowledge              MedicalKnowledge 2026.10 (public), 66,668 concepts - ready
    Study Quality                  enabled (English lectures: validated study facts; other languages: standard method)
    Privacy                        transcription runs on this computer; transcript sharing is off
"""

from __future__ import annotations

from typing import Callable, Optional

MODEL_TITLES = {"whisper-large-v3": "Whisper Large-v3", "turkish-medical-large-v3": "Turkish medical Large-v3"}
V4_TITLES = {"off": "Off", "on": "On - Experimental", "shadow": "Shadow - compare only (developer)"}


def knowledge_line(info: dict) -> str:
    if not info.get("installed"):
        return "not installed - the Medical Expert works alone"
    counts = info.get("counts", {})
    name = f"{info.get('name', 'MedicalKnowledge')} {info.get('build_version', '')}".strip()
    profile = f" ({info['profile']})" if info.get("profile") else ""
    state = "ready" if info.get("ok") else (info.get("status") or "not usable - the Medical Expert works alone")
    return f"{name}{profile}, {counts.get('concepts', 0):,} concepts - {state}"


def status_lines(settings, model_installed: Callable[[str], bool], knowledge: Optional[dict] = None) -> list[tuple[str, str]]:
    """(label, value) rows. ``knowledge`` is the medical knowledge report (None while it is being read)."""
    key = getattr(settings, "model_key", "whisper-large-v3")
    model = MODEL_TITLES.get(key, key)
    try:
        installed = model_installed(key)
    except Exception:                                                     # noqa: BLE001 - a status line never breaks the dialog
        installed = False
    expert = getattr(settings, "medical_expert", True)
    v4 = getattr(settings, "medical_intelligence", "off")
    cloud = getattr(settings, "cloud_enabled", False)
    cloud_study = bool(getattr(settings, "study_cloud_consent", False)) and getattr(settings, "study_provider", "builtin") in ("openai", "anthropic")
    study = ("MELETRA Automatic Study Engine: a local AI model when one is available, otherwise the offline method; every flashcard and question is quality-checked"
             + ("; cloud study is ON (your separate consent): lecture text is sent to the provider" if cloud_study else "; lecture text stays on this computer"))
    sharing = getattr(settings, "share_transcripts", False)
    local = ("transcription runs on this computer; the audio never leaves it" if not cloud
             else "transcription runs on this computer, but cloud comparison is ON: the audio is also uploaded to Deepgram")
    local += ("; completed transcript text is shared with MELETRA (you switched this on in Privacy & Data)" if sharing else "; transcript sharing is off")
    return [("Transcription model:", f"{model} ({'installed' if installed else 'not installed yet'})"),
            ("Medical correction:", f"Medical Expert v3 ({'on' if expert else 'off'})"),
            ("Experimental Medical Intelligence:", f"v4: {V4_TITLES.get(v4, v4)}"),
            ("Medical knowledge:", "checking ..." if knowledge is None else knowledge_line(knowledge)),
            ("Study Quality:", study),
            ("Privacy:", local)]
