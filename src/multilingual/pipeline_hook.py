"""The multilingual planning step of the pipeline (kept here so ``pipeline.py`` only gains a few call lines)."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from . import coverage as coverage_mod
from . import plan as plan_mod
from . import timeline as tl
from .decode import MAX_SAME_LINE, label_segments, line_key, reject_reason
from .timeline import LanguagePlan, WindowEvidence

log = logging.getLogger(__name__)

DEFAULT_SETTINGS = {"scan": "auto", "coverage": "auto", "level_quiet_speech": "auto", "recheck": "auto", "coverage_share": coverage_mod.MAX_RECOVERY_SHARE, "verify_samples_gpu": 40, "verify_samples_cpu": 12, "min_run_seconds": tl.MIN_RUN_SECONDS}


def settings_from(cfg: dict[str, Any]) -> dict[str, Any]:
    return {**DEFAULT_SETTINGS, **(cfg.get("multilingual") or {})}


def measure_activity(wav: Path) -> Optional[plan_mod.Activity]:
    """Loudness per second of the prepared audio (None when it cannot be read: loudness is evidence, never a requirement)."""
    try:
        return plan_mod.activity_profile(wav)
    except Exception as exc:  # noqa: BLE001
        log.warning("loudness analysis skipped (%s: %s)", type(exc).__name__, exc)
        return None


def plan_recording(*, transcriber: Any, wav: Path, duration: float, speech: Sequence[tuple[float, float]], coarse: Sequence[tuple[float, dict[str, float]]],
                   cfg: dict[str, Any], emit: Optional[Callable[..., None]] = None, should_stop: Optional[Callable[[], bool]] = None,
                   activity: Optional[plan_mod.Activity] = None) -> tuple[Optional[LanguagePlan], dict[str, Any]]:
    """Decide whether the recording needs language-by-language decoding and, if so, build the plan (scan, smooth, verify). ``coarse`` is the sampling the pipeline already did
    (``[(start, {language: probability}), ...]``, reused: its windows are not scanned again). Returns ``(plan or None, report)``; ``None`` = one language, decode as before."""
    settings = settings_from(cfg)
    report: dict[str, Any] = {"setting": settings["scan"], "escalated": False}
    started = time.perf_counter()
    if activity is None:
        activity = measure_activity(wav)
    coarse = [(s, tl.fold_aliases(p)) for s, p in coarse]
    top1 = [(s, max(p, key=p.get), max(p.values())) for s, p in coarse if p]
    reason = plan_mod.escalation_reason(top1, activity=activity, speech=speech, duration=duration, setting=str(settings["scan"]))
    report["music_suspected"] = plan_mod.music_suspected(activity, speech, duration)
    if activity is not None:
        report["loud_share"] = round(activity.loud_share, 3)
    report["vad_share"] = round(plan_mod.speech_share(speech, duration), 3)
    if reason is None:
        report["reason"] = "one language is clear from the sampled windows"
        return None, report
    report.update(escalated=True, reason=reason)
    on_gpu = getattr(getattr(transcriber, "adapter", None), "device", "cpu") == "cuda"
    speech_seconds = sum(b - a for a, b in speech) or duration
    hop = tl.adaptive_hop(max(speech_seconds, 1.0), on_gpu)
    loud = activity.loud_regions() if (activity is not None and report["music_suspected"]) else None
    wanted = tl.window_starts(duration, speech, hop, 30.0, loud)
    have = [s for s, _p in coarse]
    extra = [s for s in wanted if all(abs(s - h) >= hop * 0.5 for h in have)]
    log.info("LANGUAGE SCAN (%s): hop %.0f s, %d windows, %d of them already sampled", reason, hop, len(wanted), len(wanted) - len(extra))

    def progress(done: int, total: int) -> None:
        if emit is not None:
            emit("stage", stage="language", percent=round(100.0 * done / max(total, 1), 1), done=done, total=total)

    scanned = transcriber.detect_language_windows(wav, extra, 30.0, progress, should_stop) if extra else []
    windows = plan_mod.evidence_windows([*coarse, *scanned], duration, speech, activity)
    plan = tl.build_plan(windows, speech, hop, duration)
    report.update(windows=len(windows), extra_windows=len(extra), hop=hop, mode=plan.mode, languages=plan.languages)
    unsure = [r for r in plan.runs if r.status == "unsure"]
    if unsure and hasattr(getattr(transcriber, "adapter", None), "score_languages"):
        budget = int(settings["verify_samples_gpu" if on_gpu else "verify_samples_cpu"])

        def verify_progress(done: int, total: int) -> None:
            if emit is not None:
                emit("stage", stage="language", percent=round(100.0 * done / max(total, 1), 1))

        report["verification"] = plan_mod.verify_runs(plan, lambda s, sec, langs: transcriber.score_languages(wav, s, sec, list(langs), 24), duration=duration, max_samples=budget,
                                                        should_stop=should_stop, progress=verify_progress)
        report["mode"], report["languages"] = plan.mode, plan.languages
    report["seconds"] = round(time.perf_counter() - started, 1)
    log.info("LANGUAGE PLAN: %s | %s | %d runs | %.0f s", plan.mode, ", ".join(plan.languages) or "none",
             len(plan.runs), report["seconds"])
    return plan, report


def decode_plan_tuples(plan: LanguagePlan) -> list[tuple[float, float, str, str]]:
    """The plan as the ``(start, end, language, vad)`` tuples a ``DecodeRequest`` carries."""
    return [(r.start, r.end, r.language, r.vad) for r in plan.runs]


def quiet_ok(segment: Any) -> bool:
    """Text from a stretch whose volume had to be raised is accepted only when the model is clearly reading speech: raised noise makes Whisper write plausible filler."""
    return (segment.avg_logprob is None or segment.avg_logprob > -0.9) and (segment.no_speech_prob is None or segment.no_speech_prob < 0.6)


def boosted_slice(wav: Path, start: float, end: float, target_rms: float = 0.08, max_gain: float = 100.0) -> Optional[Path]:
    """A temporary 16 kHz WAV of ``[start, end]`` with the volume raised (at most +40 dB, soft-limited) so that the speech is as loud as in an ordinary recording. None when it cannot be made."""
    try:
        import tempfile

        import numpy as np
        import soundfile as sf

        with sf.SoundFile(str(wav)) as handle:
            rate = handle.samplerate
            handle.seek(int(max(start, 0.0) * rate))
            data = handle.read(int(max(end - start, 0.1) * rate), dtype="float32", always_2d=True).mean(axis=1)
        if not len(data):
            return None
        frames = max(len(data) // int(0.1 * rate), 1)
        energies = np.sqrt((data[: frames * int(0.1 * rate)].reshape(frames, -1) ** 2).mean(axis=1))
        voiced = np.sort(energies)[-max(frames // 5, 1):]                      # the loudest fifth of the stretch: the speech, not the pauses between words
        level = float(voiced.mean())
        if level < 1e-5:
            return None
        gain = min(target_rms / level, max_gain)
        boosted = np.tanh(data * gain)
        handle_file = tempfile.NamedTemporaryFile(prefix="mlt_quiet_", suffix=".wav", delete=False)
        handle_file.close()
        sf.write(handle_file.name, boosted.astype("float32"), rate, subtype="PCM_16")
        return Path(handle_file.name)
    except Exception as exc:  # noqa: BLE001 - the stretch is simply reported as unreadable
        log.warning("could not raise the volume of %.0f-%.0f s: %s", start, end, exc)
        return None


def recover_coverage(*, transcriber: Any, wav: Path, segments: list, activity: Optional[plan_mod.Activity], plan: Optional[LanguagePlan], language: Optional[str],
                     profile: dict[str, Any], cfg: dict[str, Any], duration: float, emit: Optional[Callable[..., None]] = None,
                     should_stop: Optional[Callable[[], bool]] = None) -> tuple[list, dict[str, Any]]:
    """Find audible stretches that carry no words (measured on the audio, not on the voice-activity filter), decode them again within a bounded budget, and report what could not be read.
    Returns ``(segments, coverage record)``; the segments are the input plus the recovered ones, in time order."""
    settings = settings_from(cfg)
    if str(settings.get("coverage")) == "off":
        return segments, {"checked": False, "reason": "turned off in the settings"}
    if activity is None or not activity.rms:
        return segments, {"checked": False, "reason": "the audio loudness could not be measured"}
    before = coverage_mod.coverage_numbers(activity, segments)
    gaps = coverage_mod.find_gaps(activity, segments)
    if not gaps:
        record = coverage_mod.summary(before, before, [], 0, 0.0)
        record["checked"] = True
        return segments, record
    on_gpu = getattr(getattr(transcriber, "adapter", None), "device", "cpu") == "cuda"
    budget = coverage_mod.budget_seconds(duration, on_gpu, float(settings.get("coverage_share") or coverage_mod.MAX_RECOVERY_SHARE))
    runs = list(plan.runs) if plan is not None else []
    ordered = sorted(segments, key=lambda s: s.start)

    def language_at(start: float, end: float) -> Optional[str]:
        middle = (start + end) / 2.0
        for run in runs:
            if run.start <= middle < run.end and run.language:
                return run.language
        if language:
            return language
        before_seg = [s for s in ordered if s.end <= start + 1.0 and getattr(s, "language", None)]
        after_seg = [s for s in ordered if s.start >= end - 1.0 and getattr(s, "language", None)]
        near = []
        if before_seg:
            near.append((start - before_seg[-1].end, before_seg[-1].language))
        if after_seg:
            near.append((after_seg[0].start - end, after_seg[0].language))
        return min(near)[1] if near else None

    run_profile = {**profile, "use_hotwords": False, "condition_on_previous_text": False}
    lines: dict[str, int] = {}                                 # how often each line is already in the transcript: the main decode limited a decoder stuck on one line, a re-read must not bring it back
    for seg in segments:
        key = line_key(seg.text)
        if key:
            lines[key] = lines.get(key, 0) + 1
    refused = [0]

    def within_chorus(seg: Any) -> bool:
        key = line_key(seg.text)
        if not key or len(seg.text.split()) < 3:
            return True
        if lines.get(key, 0) >= MAX_SAME_LINE:
            refused[0] += 1
            return False
        lines[key] = lines.get(key, 0) + 1
        return True

    def decode(start: float, end: float, raise_gain: bool = False) -> list:
        from ..model_manager import DecodeRequest

        forced = language_at(start, end)
        source, offset, begin = wav, 0.0, start
        boosted = None
        if raise_gain:                                         # a speaker far from the microphone: the same stretch with the volume raised, decoded like any other
            boosted = boosted_slice(wav, start, end)
            if boosted is not None:
                source, offset, begin = boosted, start, 0.0
        try:
            request = DecodeRequest(audio_path=source, language=forced, profile=run_profile, decoding=cfg["decoding"], vad_parameters=cfg["vad"], hotwords=[], multilingual=False,
                                    total_duration=duration if boosted is None else (end - start), start_offset=begin, max_seconds=max(end - start, 0.5), vad_override=False)
            result = transcriber.transcribe(request)
        finally:
            if boosted is not None:
                Path(boosted).unlink(missing_ok=True)
        kept = [seg for seg in result.segments if reject_reason(seg) is None and (boosted is None or quiet_ok(seg)) and within_chorus(seg)]
        for seg in kept:
            if offset:
                seg.start, seg.end = seg.start + offset, seg.end + offset
                for word in seg.words:
                    word.start, word.end = word.start + offset, word.end + offset
        label_segments(kept, forced or result.language or "und", {"source": "forced" if forced else "model", "status": "repaired", "vad": "off"})
        return kept

    log.info("COVERAGE: %d audible stretch(es) without words (%.0f s); re-decoding within a %.0f s budget", len(gaps), sum(g.seconds for g in gaps), budget)

    def progress(done: int, total: int) -> None:
        if emit is not None:
            emit("stage", stage="coverage", percent=round(100.0 * done / max(total, 1), 1), done=done, total=total)

    new = coverage_mod.recover_gaps(gaps, decode, budget=budget, should_stop=should_stop, progress=progress)
    # a recovered segment that mostly repeats the time of an existing one is not added twice
    fresh = []
    for seg in sorted(new, key=lambda s: s.start):
        length = max(seg.end - seg.start, 1e-6)
        overlap = sum(max(min(seg.end, o.end) - max(seg.start, o.start), 0.0) for o in ordered)
        if overlap / length < 0.5:
            fresh.append(seg)
    merged = sorted([*segments, *fresh], key=lambda s: (s.start, s.end))
    after = coverage_mod.coverage_numbers(activity, merged)
    record = coverage_mod.summary(before, after, gaps, len(fresh), budget)
    record["checked"] = True
    if refused[0]:
        record["repeated_lines_refused"] = refused[0]            # re-read lines that the transcript already holds more often than a chorus repeats
    log.info("COVERAGE: %.1f%% -> %.1f%% of audible time carries words | %d recovered, %d unreadable, %d skipped", 100 * before["share"], 100 * after["share"],
             record["gaps_recovered"], record["gaps_unreadable"], record["gaps_skipped"])
    return merged, record
