"""Optional online second opinion: Deepgram Nova-3 Medical, compare mode, consensus.

Nothing here runs unless the user chose it (``--backend deepgram|compare``) or allowed it for doubtful local
results (``--deepgram-if-suspect``). A missing key, a rejected key, no quota or a network problem never stops the
local result: the outcome is recorded as a status string and the local files are still written.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from .consensus import Comparison, compare_transcripts, consensus_text, disagreement_report_text
from .deepgram_backend import (
    DeepgramError, DeepgramResult, DeepgramTranscriber, DeepgramUnavailable, get_api_key, suspicious_local_transcript,
)
from .exporters import OutputPaths, write_deepgram_outputs, write_json, write_text
from .schema import Segment
from .vocabulary import VocabularyIndex

log = logging.getLogger(__name__)

DEEPGRAM_PRIVACY_NOTE = (
    "DEEPGRAM WAS USED: the audio file was uploaded to Deepgram (api.deepgram.com), a third-party service. "
    "This only happens when you ask for --backend deepgram|compare (or allow --deepgram-if-suspect)."
)


@dataclass
class SecondOpinion:
    status: str = "not requested"            # not requested | skipped: <why> | ok | failed: <why>
    requested_by: str = "none"               # none | backend | compare | if-suspect
    result: Optional[DeepgramResult] = None
    comparison: Optional[Comparison] = None
    reason: str = ""
    files: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"status": self.status, "requested_by": self.requested_by, "reason": self.reason,
                               "files": self.files}
        if self.result is not None:
            out["model"] = self.result.meta.get("model")
            out["elapsed_seconds"] = self.result.meta.get("elapsed_seconds")
            out["words"] = sum(len(s.text.split()) for s in self.result.segments)
        if self.comparison is not None:
            out["statistics"] = self.comparison.statistics
        out["privacy"] = DEEPGRAM_PRIVACY_NOTE if self.ok else "Deepgram was not used; no audio left this computer."
        return out


def should_ask_deepgram(
    backend: str, if_suspect: bool, uncertain: int, total: int, hallucination_flags: int, repetition_flags: int,
    thresholds: dict[str, float],
) -> tuple[bool, str, str]:
    """(ask?, requested_by, reason). Default (backend=local, no flag) is always: no."""
    if backend in ("deepgram", "compare"):
        return True, "backend" if backend == "deepgram" else "compare", f"--backend {backend}"
    if if_suspect:
        suspicious, why = suspicious_local_transcript(uncertain, total, hallucination_flags, repetition_flags, thresholds)
        return suspicious, "if-suspect", why
    return False, "none", "local only (default)"


def run_deepgram(
    media: Path, language: Optional[str], keyterms: Sequence[str], cfg: dict[str, Any],
    transcriber_factory: Optional[Callable[..., DeepgramTranscriber]] = None,
    api_key: Optional[str] = None,
) -> tuple[Optional[DeepgramResult], str]:
    """Send ``media`` to Deepgram. Returns ``(result, status)``; never raises for service problems."""
    dg_cfg = cfg.get("deepgram", {})
    key = api_key if api_key is not None else get_api_key()
    if not key:
        return None, "skipped: DEEPGRAM_API_KEY not found (environment or .env); local result only"
    try:
        client = (transcriber_factory or DeepgramTranscriber)(
            key, model=str(dg_cfg.get("model", "nova-3-medical")), max_retries=int(dg_cfg.get("max_retries", 3)),
            max_keyterms=int(dg_cfg.get("max_keyterms", 100)))
        result = client.transcribe_file(media, language or dg_cfg.get("language", "en"), keyterms)
    except DeepgramUnavailable as exc:
        return None, f"skipped: {exc}"
    except DeepgramError as exc:
        return None, f"failed: {type(exc).__name__}: {exc}"
    except OSError as exc:
        return None, f"failed: cannot read {media.name}: {exc}"
    return result, "ok"


def compare_with_local(
    local: Sequence[Segment], result: DeepgramResult, vocab: VocabularyIndex, pdf_terms: Sequence[str] = (),
    pdf_text: str = "",
) -> Comparison:
    return compare_transcripts(local, result.segments, vocab, pdf_terms, pdf_text)


def write_second_opinion(paths: OutputPaths, opinion: SecondOpinion) -> None:
    """deepgram_* files, and (when both transcripts exist) the comparison / disagreement / consensus files."""
    if opinion.result is not None:
        opinion.files.update({k: v.name for k, v in write_deepgram_outputs(paths, opinion.result).items()})
    if opinion.comparison is not None:
        write_text(paths.disagreement, disagreement_report_text(opinion.comparison))
        write_json(paths.comparison, opinion.comparison.to_dict())
        write_text(paths.consensus, consensus_text(opinion.comparison))
        opinion.files.update(disagreement=paths.disagreement.name, comparison=paths.comparison.name,
                             consensus=paths.consensus.name)
