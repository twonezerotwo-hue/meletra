"""Auto Detect over time: smoothing of window language evidence into runs (pure logic, no audio)."""

from __future__ import annotations

from src.multilingual import timeline as tl
from src.multilingual.timeline import WindowEvidence


def windows(spec, hop=10.0, size=30.0, rms=0.1, speech=None):
    """``spec``: one dict of detector probabilities per window; windows start every ``hop`` seconds."""
    out = []
    for i, probs in enumerate(spec):
        start = i * hop
        out.append(WindowEvidence(start, start + size, dict(probs), speech_seconds=size * 0.8 if speech is None else speech, rms=rms))
    return out


def sure(language, p=0.95):
    return {language: p, "en" if language != "en" else "de": round((1 - p) / 2, 3)}


def plan_of(spec, duration=None, **kw):
    ws = windows(spec, **{k: v for k, v in kw.items() if k in ("hop", "size", "rms", "speech")})
    duration = duration or ws[-1].end
    return tl.build_plan(ws, [], kw.get("hop", 10.0), duration)


def test_a_single_language_recording_is_one_run():
    plan = plan_of([sure("en")] * 12)
    assert [r.language for r in plan.runs] == ["en"] and plan.mode == "single" and not plan.multilingual
    assert plan.runs[0].start == 0.0 and plan.runs[0].confidence >= 0.9


def test_korean_hindi_english_korean_sections_become_four_runs_in_order():
    spec = [sure("ko")] * 8 + [sure("hi")] * 8 + [sure("en")] * 8 + [sure("ko")] * 6
    plan = plan_of(spec)
    assert [r.language for r in plan.runs] == ["ko", "hi", "en", "ko"] and plan.mode == "multilingual"
    assert plan.languages[0] in ("ko", "hi", "en") and set(plan.languages) == {"ko", "hi", "en"}                 # only what the evidence supports
    starts = [r.start for r in plan.runs]
    assert starts == sorted(starts) and plan.runs[0].start == 0.0
    assert all(abs(a.end - b.start) < 1e-6 for a, b in zip(plan.runs, plan.runs[1:]))                            # contiguous: no audio belongs to no run
    # the switch is placed near where the evidence changes (window 8 starts at 80 s; the first Hindi window owns the span around its centre)
    assert 70 <= plan.runs[1].start <= 100


def test_accented_english_with_unsure_windows_is_not_split():
    unsure = {"sl": 0.40, "en": 0.30, "hr": 0.2}
    spec = [sure("en")] * 3 + [unsure] * 2 + [sure("en")] * 3 + [unsure] + [sure("en")] * 4
    plan = plan_of(spec)
    assert [r.language for r in plan.runs] == ["en"]                                                           # an unsure window never splits a lecture


def test_a_weak_foreign_island_is_ignored_but_a_sure_one_is_kept():
    weak = [sure("en")] * 5 + [{"ko": 0.55, "en": 0.2}] + [sure("en")] * 5
    assert [r.language for r in plan_of(weak).runs] == ["en"]
    strong = [sure("en")] * 6 + [sure("ko", 0.98)] * 3 + [sure("en")] * 6                                       # about 30 s of Korean in an English lecture
    plan = plan_of(strong)
    assert [r.language for r in plan.runs] == ["en", "ko", "en"]
    korean = plan.runs[1]
    assert 15 <= korean.seconds <= 50


def test_a_run_the_detector_is_unsure_about_is_marked_for_verification_and_keeps_alternatives():
    spec = [sure("en")] * 6 + [{"ta": 0.45, "ml": 0.16, "hi": 0.15}] * 6                                       # singing: the detector names Indian languages without conviction
    plan = plan_of(spec)
    unsure = [r for r in plan.runs if r.status == "unsure"]
    assert unsure and unsure[0].confidence is not None and unsure[0].confidence < tl.UNSURE
    assert unsure[0].candidates                                                                                  # alternatives the detector itself gave (never invented)


def test_loud_stretches_that_the_voice_filter_calls_silence_are_decoded_without_it():
    music = windows([{"hi": 0.6, "ta": 0.2}] * 8, rms=0.2, speech=1.0)
    speech = windows([sure("en")] * 6, rms=0.1, speech=24.0)
    for w in speech:
        w.start += 80
        w.end += 80
    plan = tl.build_plan(music + speech, [], 10.0, 150.0)
    assert plan.mode == "music"
    assert any(r.vad == "off" and r.language == "hi" for r in plan.runs)
    assert all(r.vad == "on" for r in plan.runs if r.language == "en")                                          # ordinary speech keeps the filter


def test_window_without_detector_output_joins_its_neighbour():
    spec = [sure("en")] * 3 + [{}] + [sure("en")] * 3
    plan = plan_of(spec)
    assert [r.language for r in plan.runs] == ["en"]


def test_boundaries_move_to_the_nearest_pause():
    spec = [sure("ko")] * 6 + [sure("hi")] * 6
    ws = windows(spec)
    speech = [(0.0, 52.0), (55.0, 100.0)]                                                                          # a 3 s pause at 52-55 s
    plan = tl.build_plan(ws, speech, 10.0, ws[-1].end)
    assert [r.language for r in plan.runs] == ["ko", "hi"]
    assert 52.0 <= plan.runs[0].end <= 55.0                                                                      # the change is placed in the pause


def test_window_layout_skips_silence_and_covers_speech():
    starts = tl.window_starts(300.0, [(0, 40), (200, 260)], hop=10.0)
    assert starts[0] == 0.0 and any(180 <= s <= 230 for s in starts) and not any(80 <= s <= 160 for s in starts)
    assert all(s + 30.0 <= 300.0 + 1e-6 for s in starts)


def test_adaptive_hop_is_denser_for_short_recordings_and_caps_the_number_of_windows():
    assert tl.adaptive_hop(240, True) == 10.0 and tl.adaptive_hop(240, False) == 20.0
    assert tl.adaptive_hop(1200, True) == 10.0 and tl.adaptive_hop(1200, False) == 20.0
    assert tl.adaptive_hop(90 * 60, True) == 13.5 and tl.adaptive_hop(90 * 60, False) == 30.0                     # an 90-minute lecture: 400 windows on a GPU, 180 s hop capped at 30 s on a CPU
    assert tl.adaptive_hop(10 * 3600, True) == 30.0


def test_plan_round_trips_through_json():
    plan = plan_of([sure("ko")] * 4 + [sure("hi")] * 4)
    import json

    again = tl.LanguagePlan.from_dict(json.loads(json.dumps(plan.to_dict())))
    assert [(r.language, round(r.start, 1)) for r in again.runs] == [(r.language, round(r.start, 1)) for r in plan.runs] and again.mode == plan.mode
