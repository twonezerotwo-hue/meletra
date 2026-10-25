"""v4 precision hardening: identifiers, graded word validity, the keep-original / cost / margin / evidence-family policy, verified number forms, domain confidence."""

import re

import pytest

from src.medical.corrector import SegmentView
from src.medical_knowledge.ingest.builder import KnowledgeBuilder
from src.medical_knowledge.normalize.forms import adapt_number
from src.medical_knowledge.runtime import policy
from src.medical_knowledge.runtime.context import LectureContext
from src.medical_knowledge.runtime.engine import V4Engine
from src.medical_knowledge.runtime.gate import GateConfig, decide
from src.medical_knowledge.runtime.identifiers import identifier_reason
from src.medical_knowledge.runtime.knowledge import MedicalKnowledge
from src.medical_knowledge.runtime.validity import WordValidity
from src.medical_knowledge.sources.curated import CuratedSource
from src.medical_knowledge.sources.fixture import FixtureSource
from tests.test_mk_engine import CFG, NEURO, make_scored, run, span


@pytest.fixture(scope="module")
def kb(tmp_path_factory):
    path = tmp_path_factory.mktemp("policy") / "fixture.db"
    builder = KnowledgeBuilder(path, "fixture", "test")
    for source in (CuratedSource(use_confusables=False), FixtureSource()):
        source.export(builder)
    builder.finalize()
    with MedicalKnowledge.open(path) as handle:
        yield handle


def reason_of(text: str, word: str):
    m = re.search(rf"\b{re.escape(word)}", text)
    return identifier_reason(text, m.start(), m.start() + len(word))


# ------------------------------------------------------------------------------------------------------------------ identifiers
@pytest.mark.parametrize("text, word", [("The BRCA1 gene", "BRCA"), ("CTranslate2 runs", "CTranslate"), ("the mTORC1 pathway", "mTORC"), ("we used PyTorch and NumPy", "PyTorch"),
                                        ("the CNS and PNS", "CNS"), ("the CSF is clear", "CSF"), ("see tool.py now", "tool"), ("mail user@host.org", "user"), ("run model_v2 now", "model"),
                                        ("dataset GSE12345 is", "GSE"), ("the compound LY294002 blocks", "LY"), ("on GitHub today", "GitHub")])
def test_identifiers_are_recognised_by_structure(text, word):
    assert reason_of(text, word)


@pytest.mark.parametrize("text, word", [("the spinal cord", "spinal"), ("between the cells", "between"), ("Spinal Tap played", "Spinal"), ("A cell body", "cell"), ("Central signals form", "Central"),
                                        ("the formation of the body", "formation"), ("peripheral process", "peripheral")])
def test_ordinary_words_are_not_identifiers(text, word):
    assert reason_of(text, word) is None


# ------------------------------------------------------------------------------------------------------------------ validity
@pytest.fixture(scope="module")
def kb_open():
    from pathlib import Path

    for name in ("medical_knowledge_public.db", "kb_open_b.db"):
        path = Path(__file__).resolve().parent.parent / "data" / name
        if path.is_file():
            with MedicalKnowledge.open(path) as handle:
                yield handle
            return
    pytest.skip("no open-source knowledge database built in data/")


def test_word_validity_is_graded_and_structural(kb_open):
    kb = kb_open
    v = WordValidity(kb)
    common = [v.of(w).value for w in ("spinal", "between", "signals", "formation", "process", "body", "cell", "central", "peripheral")]
    assert min(common) >= 0.9
    assert v.of("glycerins").value == pytest.approx(0.45)                                  # a rare dictionary word
    assert v.of("dentitic").value == 0.0 and v.of("tanisites").value == 0.0                # recogniser garbles
    assert 0.3 <= v.of("neuralation").value <= 0.5                                         # derived from a known word: moderate, never "certainly real"
    assert v.of("cells").value >= 0.9 and v.of("BRCA1").value >= 0.9                       # inflection; digits are an identifier, not a spoken word
    assert v.of("neuroanatomical").value > v.of("dentitic").value


# ------------------------------------------------------------------------------------------------------------------ policy numbers
def test_keep_original_rises_with_realness_and_recogniser_confidence():
    low, _ = policy.keep_original({"asr_sure": 0.0}, 0.0)
    real, _ = policy.keep_original({"asr_sure": 0.0}, 0.95)
    sure, _ = policy.keep_original({"asr_sure": 1.0}, 0.95)
    proper, _ = policy.keep_original({"asr_sure": 1.0, "capitalised": 1.0}, 0.95)
    assert low < 0.2 < 0.6 < real < sure <= proper <= 1.0


def test_correction_cost_grows_with_distance_and_a_confident_recogniser():
    near, _ = policy.correction_cost({"lexical": 0.95, "phonetic": 0.95, "context": 0.8}, 1, 1.0)
    far, _ = policy.correction_cost({"lexical": 0.6, "phonetic": 0.7, "context": 0.8}, 1, 1.0)
    sure, _ = policy.correction_cost({"lexical": 0.95, "phonetic": 0.95, "context": 0.8, "asr_sure": 1.0}, 1, 1.0)
    unsupported, _ = policy.correction_cost({"lexical": 0.95, "phonetic": 0.95, "context": 0.0}, 1, 1.0)
    assert near < far and near < sure and near < unsupported


def test_one_ontology_relation_alone_is_not_context_evidence():
    base = {"phonetic": 0.9, "lexical": 0.8, "context": 0.5, "domain": 0.5, "lecture": 0.0, "cue": 0.0, "ontology": 0.9, "link_count": 1.0, "domain_confidence": 0.9}
    assert policy.CONTEXT not in policy.evidence_families(base, False, 0.0, 0.72, 0.55)
    assert policy.CONTEXT in policy.evidence_families({**base, "link_count": 2.0}, False, 0.0, 0.72, 0.55)         # two distinct related terms
    assert policy.CONTEXT in policy.evidence_families({**base, "cue": 0.4}, False, 0.0, 0.72, 0.55)                # a cue word of the candidate


def test_the_field_of_study_counts_only_when_the_topic_classification_is_confident():
    sig = {"phonetic": 0.9, "lexical": 0.8, "context": 0.0, "domain": 0.9, "lecture": 0.0, "domain_confidence": 0.9}
    assert policy.DOMAIN in policy.evidence_families(sig, False, 0.0, 0.72, 0.55)
    assert policy.DOMAIN not in policy.evidence_families({**sig, "domain_confidence": 0.2}, False, 0.0, 0.72, 0.55)


# ------------------------------------------------------------------------------------------------------------------ the gate
STRONG = dict(context=0.9, cue=0.6, ontology=0.6, link_count=2.0, domain=0.9, domain_confidence=0.9, lexical=0.9, phonetic=0.95, score=0.95, asr_sure=0.0, orig_freq=0.0, capitalised=0.0)


def test_a_non_word_with_form_context_and_domain_reaches_high():
    d = decide(span("estrocytes"), [make_scored(**STRONG)], CFG)
    assert d.tier == "high" and "context" in d.assessment.families and d.candidate_margin == 1.0 and d.assessment.keep < 0.2
    assert d.reason.startswith("passed")


def test_a_clear_margin_to_the_second_best_candidate_is_required():
    best = make_scored(**STRONG)
    near_tie = make_scored(**{**STRONG, "score": 0.93}, surface="astroglia", cid="MLT:astroglia")
    far = make_scored(**{**STRONG, "score": 0.60}, surface="astroglia", cid="MLT:astroglia")
    assert decide(span("estrocytes"), [best, near_tie], CFG).tier == "low"                    # 0.95 vs 0.93: ambiguous, do not correct
    assert decide(span("estrocytes"), [best, far], CFG).tier == "high"                        # 0.95 vs 0.60: a strong candidate
    mid = make_scored(**{**STRONG, "score": 0.88}, surface="astroglia", cid="MLT:astroglia")
    assert decide(span("estrocytes"), [best, mid], CFG).tier in ("medium", "high")


def test_a_real_heard_word_must_be_beaten_by_a_clear_margin():
    heard = span("sell body", reasons=["collision"], common_only=False, unknown_tokens=0)
    heard.word_validity = [0.95, 0.95]
    heard.words = ["sell", "body"]
    heard.word_unknown = [False, False]
    d = decide(heard, [make_scored(surface="cell body", **STRONG)], CFG)
    assert d.tier != "high" and d.assessment.keep > 0.6 and d.assessment.keep_margin < CFG.high_keep_margin


def test_a_single_family_of_evidence_is_never_enough_for_high():
    only_form = make_scored(**{**STRONG, "context": 0.0, "cue": 0.0, "ontology": 0.0, "link_count": 0.0, "domain": 0.4, "score": 0.95})
    sp = span("estrocytes")
    sp.word_validity = [0.95]                                       # a real word: no "non-word" evidence either
    assert decide(sp, [only_form], CFG).tier in ("low", "medium")


def test_ambiguous_abbreviations_cannot_be_high_without_the_lecture(kb):
    ambiguous = kb.db.execute("SELECT norm_short FROM abbreviation GROUP BY norm_short HAVING COUNT(DISTINCT concept_id) >= 2 LIMIT 1").fetchone()
    if not ambiguous:
        pytest.skip("the fixture database has no ambiguous abbreviation")
    short = kb.db.execute("SELECT short_form FROM abbreviation WHERE norm_short = ? LIMIT 1", (ambiguous[0],)).fetchone()[0]
    result = run(NEURO + [f"The {short.lower()} was seen in the tissue."], kb_=kb)
    assert not [c for c in result.corrections if c.replacement == short]


# ------------------------------------------------------------------------------------------------------------------ Latin safety / verified forms
def test_number_forms_are_verified_never_invented(kb):
    verify = kb.has_form
    assert adapt_number("macula denses", "macula densa", None, verify) == ("macula densa", "same")             # "-es" is not a plural of "densa"
    assert adapt_number("exosomatic synapses", "axosomatic synapse", None, verify) == ("axosomatic synapses", "adapted")
    text, status = adapt_number("macula densas", "macula densa", None, verify)
    assert status in ("same", "unverified") and "densas" not in text                                            # "densas" exists in no source: not written
    assert adapt_number("foramena", "foramen", None, verify)[0] == "foramen"


def test_a_stored_pseudo_latin_plural_is_not_in_the_database(kb):
    assert not kb.has_form("substantia nigras") or kb.db.execute("SELECT 1 FROM term WHERE compact_text = 'substantianigras'").fetchone() is None
    assert kb.db.execute("SELECT 1 FROM term WHERE compact_text IN ('maculadensas', 'substantianigras')").fetchone() is None


# ------------------------------------------------------------------------------------------------------------------ domain confidence
def views(*sentences):
    return [SegmentView(i, i * 5.0, i * 5.0 + 4.5, s, "en", []) for i, s in enumerate(sentences)]


def test_domain_confidence_needs_evidence_and_one_dominant_field(kb):
    rich = LectureContext(kb, views(*NEURO), subjects=("neuroanatomy",))
    poor = LectureContext(kb, views("We had lunch.", "The cell is small."), subjects=())
    assert rich.domain_confidence > poor.domain_confidence and poor.domain_confidence < policy.DOMAIN_CONFIDENCE_MIN
    assert 0.0 <= rich.domain_confidence <= 1.0


def test_a_word_built_from_a_long_known_word_is_treated_as_real(kb_open):
    v = WordValidity(kb_open)
    assert v.of("transdeamination").value >= 0.7                       # trans + deamination: a real term, not a garble
    assert v.of("transdeamination").value > v.of("autophage").value     # a short known component is weaker evidence
    assert v.of("dentitic").value == 0.0


# ------------------------------------------------------------------------------------------------------------------ defects found on the first held-out v2 run (fixed for the NEXT development set)
def test_a_correct_word_is_not_cut_back_to_a_shorter_term(kb_open):
    result = run(NEURO + ["The trachealis muscle closes the posterior gap of the airway rings."], subjects=("histology",), kb_=kb_open)
    assert not [c for c in result.corrections if c.original.lower() == "trachealis"]


def test_two_derived_words_of_one_stem_are_not_a_mishearing_of_each_other(kb_open):
    from src.medical_knowledge.runtime.gate import alignment_problem

    sp = span("embolize")
    sp.words, sp.word_unknown = ["embolize"], [True]
    assert "different ending" in (alignment_problem(sp, "embolus") or "")
    sp = span("autophage")
    sp.words, sp.word_unknown = ["autophage"], [True]
    assert alignment_problem(sp, "autophagy") is None                                   # a one-letter slip is still a mishearing
    result = run(NEURO + ["Thrombosis in the deep veins may embolize to the lungs and block the pulmonary trunk."], subjects=("pathology",), kb_=kb_open)
    assert not [c for c in result.corrections if c.original.lower() == "embolize"]


# ------------------------------------------------------------------------------------------------------------------ explicit protections
PROTECTED = [
    "The BRCA1 and TP53 genes were sequenced, and the mTORC1 pathway was blocked.",
    "The CNS includes the brain and spinal cord, while the PNS includes the cranial nerves, and the CSF circulates in the ventricles.",
    "We ran CTranslate2 and PyTorch from GitHub with NumPy on the dataset GSE12345.",
    "The foramina and the vertebrae and the bronchi were all labelled, and the gamma-aminobutyric acid pathway was shown.",
    "The band Spinal Tap played between the signals, and the formation of the body and the cell process stayed central and peripheral.",
    "RA and MS and ER are abbreviations that the speaker uses without explanation.",
]


@pytest.mark.parametrize("sentence", PROTECTED)
def test_identifiers_acronyms_latin_forms_and_ordinary_words_are_left_alone(kb_open, sentence):
    result = run(NEURO + [sentence], kb_=kb_open)
    changed = [(c.original, c.replacement) for c in result.corrections + result.uncertain if c.segment == len(NEURO)]
    assert changed == [], changed
