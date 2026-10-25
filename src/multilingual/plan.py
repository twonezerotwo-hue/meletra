"""Planning the languages of one recording: how loud and how 'speechy' it is, whether a dense language scan is worth it, the scan itself, and bounded verification of unsure runs.

Everything that needs the model (language detection, language likelihoods) is passed in as a function, so the logic is testable without audio and the pipeline decides what to inject.
"""

from __future__ import annotations

import logging
import math
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from . import timeline as tl
from .artifacts import is_artifact
from .scripts import compatible, information_units
from .timeline import SUPPORTED_SPOKEN, LanguagePlan, LanguageRun, WindowEvidence

log = logging.getLogger(__name__)

LOUD_RMS = 0.02
MUSIC_LOUD_SHARE = 0.5               # at least this share of the recording is loud ...
MUSIC_SPEECH_RATIO = 0.35            # ... while the voice-activity filter keeps less than this share of the loud time: singing or speech over music
WEAK_MEDIAN = 0.5
OTHER_LANGUAGE_P = 0.6               # one window this likely in another offered language is enough to scan the whole recording (a brief passage fills only part of a 30 s window)
VERIFY_SAMPLE_SECONDS = 20.0
VERIFY_MARGIN = 0.05              # log-probability units: the best language must beat the next by this much to count as verified
MIN_LETTERS = 6                  # fewer letters than this is not a reading of speech
VERIFY_MAX_SAMPLES = 12


# ====================================================================================================================== loudness
@dataclass
class Activity:
    """Loudness of the recording per frame, read from the audio itself (independent of any voice-activity filter)."""

    frame: float = 1.0
    rms: list[float] = field(default_factory=list)

    @property
    def seconds(self) -> float:
        return len(self.rms) * self.frame

    @property
    def loud_share(self) -> float:
        return sum(1 for r in self.rms if r >= LOUD_RMS) / len(self.rms) if self.rms else 0.0

    def rms_between(self, start: float, end: float) -> float:
        a, b = max(int(start / self.frame), 0), min(int(end / self.frame) + 1, len(self.rms))
        part = self.rms[a:b]
        return sum(part) / len(part) if part else 0.0

    def loud_regions(self, min_seconds: float = 3.0, bridge: float = 2.0) -> list[tuple[float, float]]:
        regions: list[list[float]] = []
        for i, r in enumerate(self.rms):
            if r < LOUD_RMS:
                continue
            t0, t1 = i * self.frame, (i + 1) * self.frame
            if regions and t0 - regions[-1][1] <= bridge:
                regions[-1][1] = t1
            else:
                regions.append([t0, t1])
        return [(a, b) for a, b in regions if b - a >= min_seconds]


def activity_profile(wav: Path, frame: float = 1.0) -> Activity:
    """RMS per ``frame`` seconds of a 16 kHz WAV (read block by block: no whole-file copy)."""
    import numpy as np
    import soundfile as sf

    out: list[float] = []
    with sf.SoundFile(str(wav)) as handle:
        size = int(handle.samplerate * frame)
        while True:
            block = handle.read(size * 60, dtype="float32", always_2d=True)
            if not len(block):
                break
            mono = block.mean(axis=1)
            usable = len(mono) // size * size
            if usable:
                out.extend(np.sqrt((mono[:usable].reshape(-1, size) ** 2).mean(axis=1)).tolist())
            if len(mono) - usable:
                out.append(float(np.sqrt((mono[usable:] ** 2).mean())))
    return Activity(frame=frame, rms=out)


def speech_share(speech: Sequence[tuple[float, float]], duration: float) -> float:
    return sum(max(b - a, 0.0) for a, b in speech) / duration if duration > 0 else 0.0


def overlap_seconds(speech: Sequence[tuple[float, float]], start: float, end: float) -> float:
    return sum(max(min(b, end) - max(a, start), 0.0) for a, b in speech)


def music_suspected(activity: Optional[Activity], speech: Sequence[tuple[float, float]], duration: float) -> bool:
    """Singing or speech over music: much of the recording is loud, but the voice-activity filter keeps only a small part of that loud time."""
    if activity is None or not activity.rms:
        return False
    loud = activity.loud_share
    return loud >= MUSIC_LOUD_SHARE and speech_share(speech, duration) <= MUSIC_SPEECH_RATIO * loud


# ====================================================================================================================== whether to scan
def escalation_reason(samples: Sequence[tuple[float, str, float]], *, activity: Optional[Activity], speech: Sequence[tuple[float, float]], duration: float, setting: str = "auto") -> Optional[str]:
    """Why a dense language scan is needed (``None`` = the cheap sampling already says one language, clearly). ``setting``: auto | always | off."""
    if setting == "off":
        return None
    if setting == "always":
        return "requested: scan the whole recording for language changes"
    if music_suspected(activity, speech, duration):
        return "much of the recording is loud but not speech to the voice filter (singing or music)"
    rows = [(float(s), str(lang), float(p)) for s, lang, p in samples]
    if not rows:
        return None
    strong = [(s, lang, p) for s, lang, p in rows if p >= tl.STRONG]
    counts: dict[str, int] = {}
    for _s, lang, _p in strong:
        counts[lang] = counts.get(lang, 0) + 1
    if len(counts) >= 2:
        return "windows clearly in " + " and ".join(sorted(counts))
    dominant = max(counts, key=counts.get) if counts else None
    if dominant is not None:
        sure_other = [(s, lang, p) for s, lang, p in rows if lang != dominant and lang in tl.SUPPORTED_SPOKEN and p >= OTHER_LANGUAGE_P]
        if sure_other:
            return f"a window in {sure_other[0][1]} within a recording that is mostly {dominant}"
    if statistics.median(p for _s, _l, p in rows) < WEAK_MEDIAN:
        return "the language evidence is weak"
    return None


# ====================================================================================================================== scanning
Detector = Callable[[Sequence[float], float, Optional[Callable[[int, int], None]]], list[tuple[float, dict[str, float]]]]


def evidence_windows(found: Sequence[tuple[float, dict[str, float]]], duration: float, speech: Sequence[tuple[float, float]], activity: Optional[Activity], size: float = 30.0) -> list[WindowEvidence]:
    """Window evidence from the detector's output ``[(start, {language: probability}), ...]``, joined with how much of each window is speech and how loud it is."""
    out = []
    for start, probs in found:
        end = min(start + size, duration)
        out.append(WindowEvidence(start, end, tl.fold_aliases(probs), speech_seconds=overlap_seconds(speech, start, end), rms=activity.rms_between(start, end) if activity else 0.0))
    return sorted(out, key=lambda w: w.start)


def scan_windows(detect: Detector, starts: Sequence[float], duration: float, speech: Sequence[tuple[float, float]], activity: Optional[Activity], size: float = 30.0,
                 progress: Optional[Callable[[int, int], None]] = None) -> list[WindowEvidence]:
    """Run the detector on every window start and return the evidence."""
    return evidence_windows(detect(list(starts), size, progress), duration, speech, activity, size)


# ====================================================================================================================== verification of unsure runs
Scorer = Callable[[float, float, Sequence[str]], dict[str, dict[str, Any]]]


SCRIPT_REPRESENTATIVES = ("en", "ko", "hi", "zh")          # one language per writing system the program offers; a Latin-script run is refined among the Latin languages afterwards
MAX_CANDIDATES = 5


def language_score(language: str, evidence: dict[str, Any]) -> float:
    """How well forcing ``language`` explains one excerpt of audio: the decoder's own average log-probability of what it wrote (higher = better explained; 0 is the best possible).

    Measured on 19 windows of authorized human-read recordings in six languages: the true language had the best log-probability in every one (a forced WRONG language scores 0.1-1.5
    lower: Whisper can translate into any language, so the text alone proves nothing - an English "translation" of Korean speech is long and fluent). The text is therefore only a
    gate against outputs that are not a reading of speech at all: nothing / a few letters, a phrase Whisper writes on music, a loop, or a writing system the language cannot have."""
    text = str(evidence.get("text") or "").strip()
    logprob = evidence.get("avg_logprob")
    score = max(float(logprob if logprob is not None else -3.0), -3.0)
    if information_units(text) < MIN_LETTERS:
        score -= 1.0
    elif is_artifact(text):
        score -= 1.0
    words = text.split()
    if len(words) >= 6:
        score -= 1.0 * (1.0 - len(set(words)) / len(words))               # a loop of the same few words
    if text and not compatible(text, language):
        score -= 2.0
    return score


def candidate_pool(run: LanguageRun, plan_languages: Sequence[str], supported: Sequence[str] = SUPPORTED_SPOKEN, limit: int = MAX_CANDIDATES) -> list[str]:
    """Languages to compare for an unsure run, all of them languages the program offers: the detector's supported guesses for the run first, then the languages already decided
    elsewhere in the recording, then one language per writing system (the detector is unreliable exactly where verification is needed - it never named Korean for a sung Korean
    window - so the script representatives are always tried). A language the program does not offer is never chosen by guessing: a forced decode in it only writes plausible-looking text."""
    pool: list[str] = []
    for language in [run.language, *run.candidates, *plan_languages, *SCRIPT_REPRESENTATIVES]:
        if language in supported and language not in pool:
            pool.append(language)
    return pool[:limit]


SWITCH_PENALTY = 0.10                # cost (in score units) of changing language between two neighbouring verified windows
VERIFY_SPACING = 30.0                # one verified window per this many seconds of an unsure run, as far as the budget allows


def smooth_scores(rows: Sequence[dict[str, float]], pool: Sequence[str], penalty: float = SWITCH_PENALTY) -> list[str]:
    """The language of each verified window, chosen so that the scores add up and a change costs ``penalty`` (a lone window with a slightly better score does not cut a passage in two)."""
    if not rows:
        return []
    best = {l: rows[0].get(l, -9.0) for l in pool}
    back: list[dict[str, str]] = []
    for row in rows[1:]:
        step: dict[str, str] = {}
        nxt: dict[str, float] = {}
        for l in pool:
            others = [(best[o] - penalty, o) for o in pool if o != l]
            move, origin = max(others) if others else (-9.0, l)
            if best[l] >= move:
                nxt[l], step[l] = best[l] + row.get(l, -9.0), l
            else:
                nxt[l], step[l] = move + row.get(l, -9.0), origin
        best = nxt
        back.append(step)
    state = max(best, key=best.get)
    path = [state]
    for step in reversed(back):
        state = step[state]
        path.append(state)
    return list(reversed(path))


def verify_runs(plan: LanguagePlan, score: Scorer, *, duration: float, max_samples: int = VERIFY_MAX_SAMPLES, sample_seconds: float = VERIFY_SAMPLE_SECONDS,
                spacing: float = VERIFY_SPACING, should_stop: Optional[Callable[[], bool]] = None, progress: Optional[Callable[[int, int], None]] = None) -> dict[str, Any]:
    """Decide the language of every unsure run by comparing the model's likelihood of each candidate language on windows spread along that run (one per ``spacing`` seconds as far as
    the budget allows: at most ``max_samples`` windows in total, shared between the unsure runs in proportion to their length). Each window gets its own language (smoothed, so a
    change must be worth its cost); an unsure run is split where the verified language changes. A stretch whose best language does not beat the next by ``VERIFY_MARGIN`` keeps
    status ``unsure`` and lists the alternatives."""
    unsure = [r for r in plan.runs if r.status == "unsure"]
    report: dict[str, Any] = {"runs": [], "samples": 0, "seconds": 0.0}
    if not unsure or max_samples <= 0:
        return report
    started = time.perf_counter()
    total_len = sum(r.seconds for r in unsure) or 1.0
    allotted = {id(r): max(1, min(math.ceil(r.seconds / spacing), round(max_samples * r.seconds / total_len))) for r in unsure}
    budget_left = max_samples
    done = 0
    replacement: dict[int, list[LanguageRun]] = {}
    for run in unsure:
        n = min(allotted[id(run)], budget_left)
        if n <= 0:
            break
        pool = candidate_pool(run, plan.languages)
        centres = [run.start + run.seconds * (i + 0.5) / n for i in range(n)]
        starts = [max(0.0, min(c - sample_seconds / 2.0, max(duration - sample_seconds, 0.0))) for c in centres]
        rows: list[dict[str, float]] = []
        kept_centres: list[float] = []
        for centre, start in zip(centres, starts):
            if should_stop is not None and should_stop():
                break
            try:
                evidence = score(start, sample_seconds, pool)
            except Exception as exc:  # noqa: BLE001 - verification is an aid: a failed window leaves the run unsure, never fails the recording
                log.warning("language verification failed at %.0f s: %s", start, exc)
                continue
            rows.append({language: language_score(language, ev) for language, ev in evidence.items()})
            kept_centres.append(centre)
            done += 1
            budget_left -= 1
            if progress is not None:
                progress(done, max_samples)
        entry: dict[str, Any] = {"start": round(run.start, 1), "end": round(run.end, 1), "detector_guess": run.language, "windows": len(rows), "pool": pool}
        if rows:
            labels = smooth_scores(rows, pool)
            margins = []
            for row in rows:
                ranked = sorted(row.values(), reverse=True)
                margins.append(ranked[0] - ranked[1] if len(ranked) > 1 else ranked[0] + 9.0)
            pieces: list[LanguageRun] = []
            index = 0
            while index < len(labels):
                stop = index
                while stop + 1 < len(labels) and labels[stop + 1] == labels[index]:
                    stop += 1
                begin = run.start if index == 0 else (kept_centres[index - 1] + kept_centres[index]) / 2.0
                finish = run.end if stop == len(labels) - 1 else (kept_centres[stop] + kept_centres[stop + 1]) / 2.0
                language = labels[index]
                own = [rows[i].get(language, -9.0) for i in range(index, stop + 1)]
                mean_margin = sum(margins[index:stop + 1]) / (stop + 1 - index)
                others: dict[str, float] = {}
                for i in range(index, stop + 1):
                    for other, value in rows[i].items():
                        if other != language:
                            others[other] = others.get(other, 0.0) + value
                pieces.append(LanguageRun(start=begin, end=finish, language=language, confidence=None, status="verified" if mean_margin >= VERIFY_MARGIN else "unsure",
                                          candidates=[o for o, _v in sorted(others.items(), key=lambda item: -item[1])][:2], windows=run.windows, vad=run.vad, source="verified"))
                entry.setdefault("pieces", []).append({"start": round(begin, 1), "end": round(finish, 1), "language": language, "margin": round(mean_margin, 3),
                                                       "score": round(sum(own) / len(own), 3), "status": pieces[-1].status})
                index = stop + 1
            replacement[id(run)] = pieces
        report["runs"].append(entry)
    report["samples"] = done
    report["seconds"] = round(time.perf_counter() - started, 1)
    runs: list[LanguageRun] = []
    for run in plan.runs:
        runs.extend(replacement.get(id(run), [run]))
    # adjacent runs that now share a language are one run again
    merged: list[LanguageRun] = []
    for run in runs:
        if merged and merged[-1].language == run.language and merged[-1].vad == run.vad:
            merged[-1].end, merged[-1].windows = run.end, merged[-1].windows + run.windows
            if run.status == "unsure":
                merged[-1].status = "unsure"
        else:
            merged.append(run)
    plan.runs = merged
    seconds: dict[str, float] = {}
    for run in plan.runs:
        seconds[run.language] = seconds.get(run.language, 0.0) + run.seconds
    plan.languages = [l for l, _s in sorted(seconds.items(), key=lambda item: -item[1])]
    plan.mode = "music" if any(r.vad == "off" for r in plan.runs) else ("multilingual" if len(plan.languages) > 1 else "single")
    return report
