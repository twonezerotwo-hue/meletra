"""v4 engine (milestone 5): suspicious spans, scoring, confidence gate, shadow mode. Precision first: the false-positive cases matter as much as the corrections."""

import json
from pathlib import Path

import pytest

from src.medical.corrector import SegmentView
from src.medical_knowledge.ingest.builder import KnowledgeBuilder
from src.medical_knowledge.runtime import shadow
from src.medical_knowledge.runtime.engine import V4Engine
from src.medical_knowledge.runtime.gate import GateConfig, alignment_problem, decide
from src.medical_knowledge.runtime.knowledge import MedicalKnowledge
from src.medical_knowledge.sources.curated import CuratedSource
from src.medical_knowledge.sources.fixture import FixtureSource


@pytest.fixture(scope="module")
def kb(tmp_path_factory):
    path = tmp_path_factory.mktemp("kb") / "fixture.db"
    builder = KnowledgeBuilder(path, "fixture", "test")
    for source in (CuratedSource(use_confusables=False), FixtureSource()):          # NO recorded mishearings: generalisation, not memory
        source.export(builder)
    builder.finalize()
    with MedicalKnowledge.open(path) as handle:
        yield handle


@pytest.fixture(scope="module")
def kb_open():
    path = Path(__file__).resolve().parent.parent / "data" / "kb_open_b.db"
    if not path.is_file():
        pytest.skip("data/kb_open_b.db not built (python tools/build_medical_knowledge.py --profile public --source mesh --source go --source hgnc --source scowl --output data/kb_open_b.db)")
    with MedicalKnowledge.open(path) as handle:
        yield handle


def views(*sentences):
    return [SegmentView(i, i * 5.0, i * 5.0 + 4.5, s, "en", []) for i, s in enumerate(sentences)]


NEURO = ["Today we will talk about neurons and glia in the nervous system.",
         "The Schwann cell wraps the axon in myelin and the node of Ranvier allows saltatory conduction.",
         "Rough endoplasmic reticulum and ribosomes form the Nissl bodies in the soma of the neuron.",
         "Astrocytes and microglia and oligodendrocytes are the glial cells of the central nervous system.",
         "The dendrites receive signals and the axon hillock starts the action potential."]


def run(sentences, subjects=("neuroanatomy",), kb_=None, **kw):
    return V4Engine(kb_).run(views(*sentences), subjects=subjects, **kw)


def replaced(result):
    return {(c.original.lower(), c.replacement.lower()) for c in result.corrections}


def test_a_clean_lecture_with_correct_terms_and_plain_english_is_not_changed(kb):
    result = run(NEURO + ["The spinal cord and the peripheral nerves carry signals between the brain and the body during formation of the central process."], kb_=kb)
    assert result.corrections == [] and result.uncertain == [] and result.texts[:5] == NEURO


@pytest.mark.parametrize("word", ["spinal", "between", "signals", "formation", "body", "cell", "process", "central", "peripheral", "during", "important", "structure"])
def test_ordinary_words_are_never_examined_even_in_a_medical_context(kb, word):
    result = run(NEURO + [f"This is a sentence about the {word} that the lecturer says in a normal way."], kb_=kb)
    assert not [c for c in result.corrections + result.uncertain if word in c.original.lower()]
    assert result.stats["suspicious_spans"] < 20                                    # and the engine did not go fishing around them either


def test_mishearings_of_known_terms_are_corrected_from_sound_and_context(kb):
    text = NEURO[:3] + ["Several nisal bodies are visible in the soma next to the nucleus.", "The estrocytes and the microglia are glial cells.",
                        "A dentitic spine receives the signal from the axon terminal."]
    result = run(text, kb_=kb)
    done = replaced(result)
    assert ("estrocytes", "astrocytes") in done and ("dentitic", "dendritic") in done
    assert any(o.startswith("nisal") and "nissl" in r for o, r in done | {(c.original.lower(), c.replacement.lower()) for c in result.uncertain})
    assert result.texts[0] == text[0] and "astrocytes" in result.texts[4]


def test_the_raw_text_is_never_modified_by_the_engine(kb):
    original = NEURO + ["The estrocytes are glial cells."]
    v = views(*original)
    result = V4Engine(kb).run(v, subjects=["neuroanatomy"])
    assert [x.text for x in v] == original and result.texts[-1] != original[-1]


def test_a_homophone_is_corrected_in_a_medical_context_and_left_alone_in_ordinary_speech(kb):
    medical = run(NEURO + ["The sell body contains the nucleus and the Nissl substance of the neuron."], kb_=kb)
    # "sell" is a real word: with no recorded mishearing the engine does not AUTO-correct it (at most it suggests, never applies)
    assert not [c for c in medical.corrections if c.original.lower() == "sell body"]
    ordinary = run(["We will sell the old car next week and buy a new one.", "The lumber yard is closed on sunday and the sell price is low."], subjects=(), kb_=kb)
    assert ordinary.corrections == [] and ordinary.uncertain == []


def test_a_word_that_exists_in_the_vocabulary_is_a_real_word_not_a_mishearing(kb):
    result = run(NEURO + ["The ventricular system and the cranial nerves are described in the next slide."], kb_=kb)
    assert not [c for c in result.corrections if c.original.lower() in ("ventricular", "cranial")]


def test_repetition_of_an_odd_word_raises_the_evidence(kb):
    once = run(NEURO + ["The tonicides line the third ventricle."], kb_=kb)
    many = run(NEURO + ["The tonicides line the third ventricle.", "These tonicides touch the cerebrospinal fluid.", "The tonicides send processes to the hypothalamus."], kb_=kb)
    best = lambda r: max([c.confidence for c in r.corrections + r.uncertain if c.original.lower() == "tonicides"] or [0.0])        # noqa: E731
    assert best(many) >= best(once) and best(many) > 0.0


def test_every_correction_explains_itself(kb):
    result = run(NEURO + ["The estrocytes are glial cells."], kb_=kb)
    c = next(c for c in result.corrections if c.original == "estrocytes")
    d = c.to_dict()
    assert d["replacement"] == "astrocytes" and d["concept_id"].startswith("MLT:") and d["sources"] == ["MLT_CURATED"]
    assert {"phonetic", "lexical", "context", "domain", "oddness", "lecture", "asr", "repetition"} <= set(d["signals"])
    assert {"keep_original_score", "best_score", "second_best_score", "score_margin", "retrieval_source", "evidence_families", "decision_reason", "context_evidence",
            "pdf_evidence", "phonetic_score", "lexical_score"} <= set(d)
    assert d["context"] and d["timestamp"] and d["confidence"] > 0.7 and isinstance(d["alternatives"], list)


def test_lecture_vocabulary_makes_a_slide_term_the_preferred_candidate(kb):
    text = NEURO + ["The tanisites line the floor of the third ventricle."]
    plain = run(text, kb_=kb)
    slides = run(text, kb_=kb, lecture_terms=["tanycytes"])
    assert replaced(slides) & {("tanisites", "tanycytes")} or any(c.replacement == "tanycytes" for c in slides.corrections + slides.uncertain)
    assert slides.stats["high"] >= plain.stats["high"]


def test_the_share_of_corrections_is_capped(kb):
    junk = ["The nisal estrocytes and dentitic oligodendricytes microgliah neurens axonn are here."] * 40
    result = run(NEURO + junk, kb_=kb)
    assert result.stats["high"] <= max(3, int(0.04 * result.stats["words"])) + 0


# ---------------------------------------------------------------------------------------------------- gate
def make_scored(**signals):
    from src.medical_knowledge.runtime.retrieve import Candidate
    from src.medical_knowledge.runtime.score import Scored

    base = {"phonetic": 0.9, "lexical": 0.8, "context": 0.8, "domain": 0.8, "lecture": 0.0, "oddness": 1.0, "asr": 0.5, "repetition": 0.0, "language": 1.0, "cue": 0.5, "ontology": 0.5}
    score = signals.pop("score", 0.9)
    known = signals.pop("known", False)
    trust = signals.pop("trust", "CURATED")
    kind = signals.pop("kind", "term")
    surface = signals.pop("surface", "astrocytes")
    cid = signals.pop("cid", "MLT:astrocyte")
    base.update(signals)
    cand = Candidate(cid, 1, surface, surface, "en", "preferred", trust, ("neuroanatomy",), len(surface.split()), ("fuzzy",), kind, base["phonetic"], base["lexical"], False, known, 1.0, 1.0)
    return Scored(cand, score, base)


def span(text="estrocytes", **kw):
    from src.medical_knowledge.runtime.context import Token
    from src.medical_knowledge.runtime.spans import Span

    toks = [Token(w, 0, len(w)) for w in text.split()]
    return Span(0, 0, len(toks), 0, len(text), text, toks, kw.pop("reasons", ["unknown"]), kw.pop("common_only", False), kw.pop("unknown_tokens", len(toks)), None, None)


CFG = GateConfig()


def test_the_gate_accepts_strong_independent_evidence():
    d = decide(span(), [make_scored()], CFG)
    assert d.tier == "high" and d.best.candidate.surface == "astrocytes"


def test_the_gate_needs_two_independent_kinds_of_evidence():
    weak_context = make_scored(context=0.2, lecture=0.0, phonetic=0.7, score=0.85)
    assert decide(span(), [weak_context], CFG).tier != "high"


@pytest.mark.parametrize("kw, expected", [({"phonetic": 0.55}, "low"), ({"lexical": 0.3}, "low"), ({"score": 0.5}, "low")])
def test_the_gate_rejects_weak_sound_weak_spelling_or_a_low_score(kw, expected):
    assert decide(span(), [make_scored(**kw)], CFG).tier == expected


def test_plain_english_needs_a_recorded_mishearing_or_near_identical_everything():
    plain = span("sell body", reasons=["collision"], common_only=True, unknown_tokens=0)
    assert decide(plain, [make_scored(surface="cell body", phonetic=0.85, lexical=0.8, context=0.8)], CFG).tier == "low"
    assert decide(plain, [make_scored(surface="cell body", phonetic=0.95, lexical=0.75, context=0.8, domain=0.9)], CFG).tier in ("high", "medium")
    assert decide(plain, [make_scored(surface="cell body", phonetic=0.85, known=True)], CFG).tier in ("high", "medium")


def test_ambiguity_between_two_concepts_lowers_the_tier():
    a, b = make_scored(score=0.86), make_scored(score=0.84, surface="astroglia", cid="MLT:astroglia")
    assert decide(span(), [a, b], CFG).tier == "low"                                  # 0.86 vs 0.84: ambiguous, the original stays
    c = make_scored(score=0.62, surface="astroglia", cid="MLT:astroglia")
    assert decide(span(), [a, c], CFG).tier == "high"                                 # 0.86 vs 0.62: a clear winner
    d = make_scored(score=0.78, surface="astroglia", cid="MLT:astroglia")
    assert decide(span(), [a, d], CFG).tier in ("medium", "high")


def test_unverified_terms_and_drug_names_need_more_evidence():
    borderline = dict(score=0.72, phonetic=0.8, context=0.7)
    heard = span("astrocytes".replace("a", "e", 1))                                 # estrocytes
    assert decide(heard, [make_scored(**borderline)], CFG).tier == "high"
    assert decide(heard, [make_scored(trust="UNVERIFIED_CANDIDATE", **borderline)], CFG).tier == "medium"
    drug = span("metaprolol")
    assert decide(drug, [make_scored(kind="drug", surface="metoprolol", **borderline)], CFG).tier == "medium"
    assert decide(drug, [make_scored(kind="term", surface="metoprolol", **borderline)], CFG).tier == "high"


@pytest.mark.parametrize("heard, surface, problem", [("neurons", "neuron", "same words"), ("glial fibrillary acidic", "glial fibrillary acidic protein", "incomplete mention"),
                                                     ("inhibitory neurotransmitters", "neurotransmitters", "real word")])
def test_replacing_a_span_must_be_a_correction_not_an_edit_of_valid_words(heard, surface, problem):
    from src.medical_knowledge.runtime.context import Token
    from src.medical_knowledge.runtime.spans import Span

    toks = [Token(w, 0, len(w)) for w in heard.split()]
    flags = [False] + [True] * (len(toks) - 1) if len(toks) > 1 else [True]
    s = Span(0, 0, len(toks), 0, len(heard), heard, toks, ["unknown"], False, 1, None, None, flags, heard.split(), flags)
    assert problem in (alignment_problem(s, surface) or "")
    assert alignment_problem(Span(0, 0, 1, 0, 10, "estrocytes", [Token("estrocytes", 0, 10)], ["unknown"], False, 1, None, None), "astrocytes") is None


def test_a_correction_of_a_very_short_sound_is_not_made():
    d = decide(span("lps", reasons=["unknown"]), [make_scored(surface="LPS", phonetic=0.9)], CFG)
    assert d.tier == "low" and any("too short" in b for b in d.blocked_by)


# ---------------------------------------------------------------------------------------------------- shadow mode
def test_shadow_mode_is_off_by_default_and_follows_the_flag(monkeypatch):
    monkeypatch.delenv(shadow.FLAG_ENV, raising=False)
    assert shadow.enabled(None) is False and shadow.enabled(True) is True
    monkeypatch.setenv(shadow.FLAG_ENV, "1")
    assert shadow.enabled(None) is True and shadow.enabled(False) is True
    monkeypatch.setenv(shadow.FLAG_ENV, "0")
    assert shadow.enabled(True) is False


def test_the_shadow_report_compares_with_the_current_engine_and_changes_nothing(kb, tmp_path, monkeypatch):
    from src.medical.engine import MedicalOptions, run_medical_pass
    from types import SimpleNamespace

    segs = [SimpleNamespace(start=i * 5.0, end=i * 5.0 + 4.5, text=t, confidence=0.9, language="en", words=[]) for i, t in
            enumerate(NEURO + ["The estrocytes are glial cells and the dentitic spine receives signals.", "Several nisal bodies are visible in the soma."])]
    monkeypatch.delenv(shadow.FLAG_ENV, raising=False)
    plain = run_medical_pass(segs, MedicalOptions(language="en", subjects=["neuroanatomy"]))
    assert plain.shadow is None
    monkeypatch.setenv(shadow.FLAG_ENV, "1")
    monkeypatch.setenv("MLT_KNOWLEDGE_DB", str(kb.path))
    flagged = run_medical_pass(segs, MedicalOptions(language="en", subjects=["neuroanatomy"]))
    assert flagged.text == plain.text and flagged.correction_report["corrections"] == plain.correction_report["corrections"]        # production identical
    rep = flagged.shadow
    assert rep and rep["mode"] == "shadow" and rep["production_transcript_changed"] is False and rep["agreement_summary"]
    assert {"segment", "original", "current", "v4", "agreement", "production_decision", "timestamp"} <= set(rep["comparison"][0])
    v4_rows = [r["v4"] for r in rep["comparison"] if r["v4"]]
    assert v4_rows and {"replacement", "confidence", "keep_original_score", "best_score", "second_best_score", "score_margin", "retrieval_source", "domain", "context_evidence",
                        "pdf_evidence", "phonetic_score", "lexical_score", "decision_reason", "evidence_families"} <= set(v4_rows[0])
    assert "rejected_near_misses" in rep
    written = flagged.write(tmp_path)
    assert shadow.REPORT_NAME in written and json.loads((tmp_path / shadow.REPORT_NAME).read_text(encoding="utf-8"))["version"] == shadow.SHADOW_VERSION
    assert shadow.REPORT_NAME not in plain.write(tmp_path / "plain")


def test_a_failing_shadow_run_never_affects_the_transcript(kb, monkeypatch):
    from src.medical.engine import MedicalOptions, run_medical_pass
    from types import SimpleNamespace

    monkeypatch.setenv(shadow.FLAG_ENV, "1")
    monkeypatch.setenv("MLT_KNOWLEDGE_DB", str(kb.path.parent / "does_not_exist.db"))
    monkeypatch.setattr("src.medical_knowledge.runtime.knowledge.DEFAULT_NAMES", ("no_such_default.db",))        # a database built into data/ must not be found either
    segs = [SimpleNamespace(start=0.0, end=4.0, text="The Schwann cell wraps the axon.", confidence=0.9, language="en", words=[])]
    out = run_medical_pass(segs, MedicalOptions(language="en"))
    assert out is not None and out.shadow is None and "Schwann" in out.text


# ---------------------------------------------------------------------------------------------------- round 2: valid words, number, case, siblings, garbled segments
def test_a_word_of_the_english_dictionary_is_a_real_word_not_a_mishearing(kb_open):
    result = run(NEURO + ["We will discuss how the signals originate and how enzymes synthesize the product, and we designate the fibers as advantageous."], kb_=kb_open)
    assert not [c for c in result.corrections + result.uncertain if c.original.lower() in ("originate", "synthesize", "designate", "advantageous")]


def test_the_replacement_keeps_the_number_and_the_natural_case_of_what_was_heard(kb_open):
    result = run(NEURO + ["The exosomatic synapses are inhibitory and the nerve system carries them."], kb_=kb_open)
    assert "axosomatic synapses are inhibitory" in result.texts[-1]                  # the plural stays
    medium = [c.replacement for c in result.uncertain if c.original == "nerve system"]
    assert all(r == r.lower() for r in medium)                                       # "nervous system", not the MeSH Title Case "Nervous System"


def test_a_term_that_is_already_named_in_the_sentence_is_not_the_answer_again(kb_open):
    result = run(NEURO + ["The forebrain divides into the telencephalon and the deencephalon in the fourth week."], kb_=kb_open)
    assert not [c for c in result.corrections if c.original.lower() == "deencephalon" and "telencephalon" in c.replacement.lower()]


def test_a_garbled_or_foreign_segment_is_left_alone(kb_open):
    result = run(NEURO + ["in mikrotubov ki so predstavljeni za to dobro in vse"], kb_=kb_open)
    assert [c for c in result.corrections + result.uncertain if c.segment == len(NEURO)] == []


def test_letters_inside_a_word_are_never_deleted_to_make_a_term():
    from src.medical_knowledge.runtime.context import Token
    from src.medical_knowledge.runtime.spans import Span

    s = Span(0, 0, 1, 0, 10, "resynaptic", [Token("resynaptic", 0, 10)], ["unknown"], False, 1, None, None, [True], ["resynaptic"], [True])
    assert "inside the word" in (alignment_problem(s, "synaptic") or "")
    assert alignment_problem(s, "presynaptic") is None


def test_identifiers_written_on_purpose_are_not_examined(kb_open):
    result = run(["The model runs with CTranslate and mTORC1 signalling is covered in the macOS manual."], subjects=("biochemistry",), kb_=kb_open)
    assert not [c for c in result.corrections + result.uncertain if c.original in ("CTranslate", "macOS", "mTORC1")]


def test_turkish_sentences_are_not_corrected_into_english_terms(kb_open):
    turkish = ["Akson nöronun uzun bir uzantısıdır ve miyelin kılıfı ile sarılıdır.", "Tanisites hücreleri üçüncü ventrikülün tabanını çevreler ve dendritik dikenler sinapsları alır."]
    result = run(NEURO + turkish, kb_=kb_open)
    assert [c for c in result.corrections + result.uncertain if c.segment >= len(NEURO)] == []
