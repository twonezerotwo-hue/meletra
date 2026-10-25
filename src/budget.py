"""Decoder token budgets for hotwords / prompt.

Whisper's decoder has a fixed context of 448 positions that the prompt (hotwords, initial prompt, and -
with ``condition_on_previous_text`` - the previous text) SHARES with the text it is about to write.
faster-whisper additionally caps each prompt part at 223 tokens. A prompt that fills the window leaves no
room for the new sentence, which is what produced the cut-off words / loops measured on large-v3.

``prompt_token_budget``   total prompt tokens we allow ourselves (hotwords + initial prompt)
``hotwords_token_budget`` tokens for the hotword list alone
Both are clamped to what the decoder can really afford, and the clamping is reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

DECODER_CONTEXT = 448          # Whisper n_text_ctx
SOT_SEQUENCE = 4               # <|startofprev|>/<|startoftranscript|>, language, task, timestamps markers
LIBRARY_PART_CAP = 223         # faster-whisper: max_length // 2 - 1 per prompt part
DEFAULT_GENERATION_RESERVE = 160


@dataclass(frozen=True)
class TokenBudgets:
    prompt: int
    hotwords: int
    generation_reserve: int
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {"prompt_token_budget": self.prompt, "hotwords_token_budget": self.hotwords,
                "generation_reserve_tokens": self.generation_reserve, "notes": list(self.notes)}


def compute_token_budgets(context_cfg: Mapping[str, Any]) -> TokenBudgets:
    """Effective budgets from config/transcription.yaml -> context (legacy key ``hotword_token_budget`` accepted)."""
    reserve = int(context_cfg.get("generation_reserve_tokens", DEFAULT_GENERATION_RESERVE))
    ceiling = DECODER_CONTEXT - SOT_SEQUENCE - reserve
    requested_prompt = int(context_cfg.get("prompt_token_budget", ceiling))
    requested_hot = int(context_cfg.get("hotwords_token_budget", context_cfg.get("hotword_token_budget", 128)))
    notes: list[str] = []

    prompt = min(requested_prompt, ceiling)
    if requested_prompt > ceiling:
        notes.append(f"prompt_token_budget {requested_prompt} lowered to {prompt}: the decoder window is {DECODER_CONTEXT} "
                     f"tokens and {reserve} must stay free for the text being written")
    hotwords = min(requested_hot, LIBRARY_PART_CAP, prompt)
    if requested_hot > hotwords:
        why = (f"faster-whisper caps a prompt part at {LIBRARY_PART_CAP}" if requested_hot > LIBRARY_PART_CAP
               else f"it cannot exceed the prompt budget {prompt}")
        notes.append(f"hotwords_token_budget {requested_hot} lowered to {hotwords}: {why}")
    return TokenBudgets(prompt=prompt, hotwords=max(hotwords, 0), generation_reserve=reserve, notes=tuple(notes))


PREVIOUS_TEXT_MAX = LIBRARY_PART_CAP     # with condition_on_previous_text faster-whisper may also feed back up to 223 tokens


def effective_hotword_budget(budgets: TokenBudgets, conditioning: bool) -> tuple[int, str | None]:
    """Hotword tokens that are really safe to send.

    Without conditioning the hotword budget stands. If conditioning is on as well (the guard was disabled in the
    config), the previous text shares the prompt window, so the hotwords may only use what is left of the prompt
    budget after a full previous-text allowance - possibly nothing.
    """
    if not conditioning:
        return budgets.hotwords, None
    room = max(0, budgets.prompt - PREVIOUS_TEXT_MAX)
    allowed = min(budgets.hotwords, room)
    if allowed < budgets.hotwords:
        return allowed, (f"hotword budget {budgets.hotwords} -> {allowed} tokens because condition_on_previous_text is on "
                         f"and may add up to {PREVIOUS_TEXT_MAX} tokens of previous text to the same decoder window")
    return allowed, None
