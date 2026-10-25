"""The hotword safeguard, applied on EVERY decode path (probe, every chunk, retries, resumes).

Measured on large-v3 (README "Hotword experiment"): hotwords together with ``condition_on_previous_text=true``
fill Whisper's 448-token decoder window with prompt, so words get cut off and the output loops (WER 2.8% -> 46.7%).
With conditioning off the same hotwords help (WER 2.2%, term recall 1.00). Therefore:

1. whenever hotwords are sent, conditioning is switched off (``context.hotwords_disable_conditioning``);
2. if that switch is turned off in the config and conditioning stays on, the hotword budget shrinks so that
   hotwords + a full previous-text allowance still fit the prompt budget.
"""

from __future__ import annotations

from typing import Any, Optional

from .budget import TokenBudgets, effective_hotword_budget

HOTWORD_GUARD_NOTE = (
    "condition_on_previous_text switched OFF because hotwords are active (measured: hotwords + conditioning "
    "makes large-v3 truncate/loop; see README)"
)


def apply_hotword_guard(profile: dict[str, Any], hotwords: list[str], cfg: dict[str, Any]) -> Optional[str]:
    """Turn off ``condition_on_previous_text`` in ``profile`` (in place) when hotwords are used.

    Returns the note to log/record, or ``None`` if nothing changed. Controlled by
    ``context.hotwords_disable_conditioning`` in config/transcription.yaml.
    """
    if hotwords and profile.get("condition_on_previous_text") and cfg["context"].get("hotwords_disable_conditioning", True):
        profile["condition_on_previous_text"] = False
        return HOTWORD_GUARD_NOTE
    return None


def prepare_decode_profile(
    base_profile: dict[str, Any], hotwords: list[str], cfg: dict[str, Any], budgets: TokenBudgets,
) -> tuple[dict[str, Any], Optional[int], list[str]]:
    """``(profile, hotword_token_budget, notes)`` for one decode call. ``base_profile`` is not modified."""
    profile = dict(base_profile)
    notes: list[str] = []
    if not hotwords:
        return profile, None, notes
    guard = apply_hotword_guard(profile, hotwords, cfg)
    if guard:
        notes.append(guard)
    budget, budget_note = effective_hotword_budget(budgets, bool(profile.get("condition_on_previous_text")))
    if budget_note:
        notes.append(budget_note)
    return profile, budget, notes
