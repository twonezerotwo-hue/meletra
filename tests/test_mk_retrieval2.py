"""v4 retrieval hardening: word-level candidates, morphological forms, spoken numbers, expected (lecture-conditioned) terms, compounds, grammar / kind compatibility, diagnostics."""

import pytest

from src.medical.corrector import SegmentView
from src.medical_knowledge.ingest.builder import KnowledgeBuilder
from src.medical_knowledge.normalize.morphology import related_forms
from src.medical_knowledge.normalize.numbers import has_number, numeric_key
from src.medical_knowledge.normalize.text import wording_key
from src.medical_knowledge.runtime import grammar
from src.medical_knowledge.runtime.context import LectureContext
from src.medical_knowledge.runtime.expected import ExpectedTerms
from src.medical_knowledge.runtime.knowledge import MedicalKnowledge
from src.medical_knowledge.runtime.retrieve import CandidateRetriever
from src.medical_knowledge.runtime.spans import SpanDetector
from src.medical_knowledge.sources.curated import CuratedSource
from src.medical_knowledge.sources.fixture import FixtureSource
from tests.test_mk_engine import NEURO


@pytest.fixture(scope="module")
def kb(tmp_path_factory):
    path = tmp_path_factory.mktemp("retrieval2") / "fixture.db"
    builder = KnowledgeBuilder(path, "fixture", "test")
    for source in (CuratedSource(use_confusables=False), FixtureSource()):
        source.export(builder)
    builder.finalize()
    with MedicalKnowledge.open(path) as handle:
        yield handle


def views(*sentences):
    return [SegmentView(i, i * 5.0, i * 5.0 + 4.5, s, "en", []) for i, s in enumerate(sentences)]


# ------------------------------------------------------------------------------------------------------------------ numbers
@pytest.mark.parametrize("spoken, written", [("CRISPR Cas nine", "CRISPR-Cas9"), ("fructose two six bisphosphate", "Fructose-2,6-Bisphosphate"), ("DNA polymerase one", "DNA polymerase I"),
                                             ("seven methylguanosine cap", "7-methylguanosine cap"), ("twenty one", "21")])
def test_spoken_and_written_numbers_share_a_key(spoken, written):
    assert numeric_key(spoken) == numeric_key(written) and has_number(spoken)


def test_plain_words_have_no_numeric_key_difference():
    assert not has_number("tanycyte") and numeric_key("tanycyte") == "tanycyte"


def test_the_numeric_channel_finds_a_term_spoken_with_number_words(tmp_path):
    from src.medical_knowledge.sources.base import ConceptRecord, TermRecord

    builder = KnowledgeBuilder(tmp_path / "n.db", "fixture", "test")
    CuratedSource().export(builder)
    builder.add_source(__import__("src.medical_knowledge.sources.base", fromlist=["SourceInfo"]).SourceInfo("MLT_CURATED", "x", "1", "", "PROJECT_OWNED", True))
    builder.add_concept("MLT_CURATED", ConceptRecord("TEST:cas9", "CRISPR-Cas9", "", ("molecular_biology",)))
    builder.add_term("MLT_CURATED", TermRecord("TEST:cas9", "CRISPR-Cas9", "en", "preferred", True))
    builder.finalize()
    with MedicalKnowledge.open(tmp_path / "n.db") as k:
        found = CandidateRetriever(k).retrieve("CRISPR Cas nine", [], 20)
        assert any(c.surface == "CRISPR-Cas9" and "numeric" in c.channels for c in found)


# ------------------------------------------------------------------------------------------------------------------ forms of one word
def test_related_forms_cover_number_adjectival_and_verbal_endings():
    assert {"emboli", "embolus", "embolic"} <= related_forms("embolize") | related_forms("embolus") | related_forms("emboli")
    assert "trachea" in related_forms("tracheal") and "tracheal" in related_forms("trachea")
    assert related_forms("cell") == set()                                              # too short to say anything


def test_one_word_in_two_spellings_is_one_answer_but_different_words_stay_different():
    assert wording_key("glibenclamide") == wording_key("Glybenclamide") == wording_key("glibenclamides")
    assert wording_key("Ezetimib") == wording_key("ezetimibe") == wording_key("ezetimibes")
    for a, b in (("afferent", "efferent"), ("ileum", "ilium"), ("meiosis", "miosis"), ("ureter", "urethra"), ("abduction", "adduction")):
        assert wording_key(a) != wording_key(b)


# ------------------------------------------------------------------------------------------------------------------ word-level candidates
def test_a_rare_word_of_the_vocabulary_is_a_candidate_even_when_only_longer_terms_contain_it(kb):
    r = CandidateRetriever(kb)
    found = r.retrieve("purkinge", [], 30)
    assert any(c.kind == "word" and c.surface.lower() == "purkinje" and {"token-sound", "token-spelling"} & set(c.channels) for c in found) or \
        any(c.surface.lower().startswith("purkinje") for c in found)


def test_the_retriever_reports_what_each_channel_found(kb):
    trace = {}
    CandidateRetriever(kb).retrieve("tanisites", [], 30, trace=trace)
    assert trace["channels"] and "tokens" in trace and trace.get("kept") is not None


def test_a_lecture_prior_moves_a_concept_up_but_does_not_create_one(kb):
    r = CandidateRetriever(kb)
    plain = r.retrieve("astrocite", [], 30)
    assert plain
    boosted = r.retrieve("astrocite", [], 30, boost={plain[-1].concept_id})
    assert boosted and next(c for c in boosted if c.concept_id == plain[-1].concept_id).prior > plain[-1].prior


# ------------------------------------------------------------------------------------------------------------------ expected terms and compounds
def test_expected_terms_come_from_what_the_lecture_already_says(kb):
    ctx = LectureContext(kb, views(*NEURO), subjects=("neuroanatomy",))
    ex = ExpectedTerms(kb, ctx)
    assert ex.weights and ex.info
    hits = ex.matches("astrocites")
    assert hits and all(names == {"expected"} for names in hits.values())
    assert ex.matches("astrocytes") == {}                                              # a span that IS an expected term is left alone


def test_a_lecture_that_does_not_say_anything_expects_nothing_special(kb):
    ctx = LectureContext(kb, views("We had lunch and the weather was nice.", "Then we walked home."), subjects=())
    ex = ExpectedTerms(kb, ctx)
    assert len(ex.weights) < 60 and ex.core == set()                                   # no confident topic: no field-level expectation


def test_several_words_written_apart_can_read_like_one_rare_word(kb):
    ctx = LectureContext(kb, views(*NEURO), subjects=("neuroanatomy",))
    det = SpanDetector(kb, ctx)
    word = next(w for w in kb.long_tokens().get("oli", []) if len(w) >= 12)           # e.g. "oligodendrocyte"
    split = word[: len(word) // 2] + " " + word[len(word) // 2:-1] + "x"
    assert det._compound_hit(split)
    assert not det._compound_hit("the central nervous")


# ------------------------------------------------------------------------------------------------------------------ grammar and kind of meaning
def test_a_verb_is_never_replaced_by_a_noun_and_an_adjective_is_not_a_noun():
    assert grammar.assess("Thrombosis may ", "embolize", " to the lungs", "embolus", "term")["pos"] == 0.0
    assert grammar.assess("The ", "tracheal", " muscle closes", "trachea", "term")["pos"] < 0.5
    assert grammar.assess("The ", "estrocytes", " are glial cells", "astrocytes", "term")["pos"] >= 0.6


def test_the_sentence_frame_decides_the_kind_of_thing():
    assert grammar.assess("The ", "xyzzy", " muscle closes", "Fas Ligand Protein", "gene")["type_fit"] == 0.0              # "the X muscle" asks for a structure, not a gene
    assert grammar.assess("The ", "nerf", " system carries", "nervous system", "term")["type_fit"] >= 0.6
    assert grammar.assess("Patients with ", "cystic fibrosis", " syndrome", "Cystic Fibrosis", "term")["type_fit"] == 1.0


# ------------------------------------------------------------------------------------------------------------------ a valid word mentioned once with nothing around it that fits
PUBLIC = __import__("pathlib").Path(__file__).resolve().parent.parent / "data" / "medical_knowledge_public.db"


@pytest.mark.skipif(not PUBLIC.is_file(), reason="data/medical_knowledge_public.db is not built")
def test_a_valid_word_said_once_is_examined_only_when_nothing_in_the_lecture_supports_it():
    with MedicalKnowledge.open(PUBLIC) as big:
        def examined(lines, word):
            segs = views(*lines)
            ctx = LectureContext(big, segs, subjects=())
            det = SpanDetector(big, ctx, ExpectedTerms(big, ctx))
            det._prime_confusions()
            return [s for seg in ctx.segments for s in det.detect_segment(seg) if s.text.lower() == word and "expected" in s.reasons]

        gut = ["Today we review the digestive tract.", "The small intestine has three parts, the duodenum, the jejunum and the colon follows it.",
               "The ilium absorbs vitamin B twelve and bile salts in the small intestine."]
        pelvis = ["Today we review the bony pelvis.", "The hip bone consists of the ilium, the ischium and the pubis.", "The ilium has a wing and the iliac crest."]
        assert examined(gut, "ilium")                              # 'ileum' is expected here, 'ilium' is a real word but nothing supports it: it is looked at (never changed by this stage)
        assert not examined(pelvis, "ilium")                       # the same word in a lecture that makes it likely is left alone


# ------------------------------------------------------------------------------------------------------------------ round 2: what must never be HIGH (found on the diagnostic corpus)
@pytest.mark.skipif(not PUBLIC.is_file(), reason="data/medical_knowledge_public.db is not built")
def test_a_one_word_fix_next_to_a_word_that_stays_unexplained_is_not_high_and_a_latin_adjective_is_left_alone():
    from src.medical_knowledge.runtime.engine import V4Engine

    with MedicalKnowledge.open(PUBLIC) as big:
        engine = V4Engine(big)

        def run(*sentences):
            result = engine.run(views(*sentences))
            return {c.original: c.replacement for c in result.corrections}, {c.original: c.replacement for c in result.uncertain}

        high, _ = run("The vestibulocochlear nerve carries hearing and balance from the inner ear.",
                      "The cochlear nuclei lie at the pontum dullari junction, and the vestibular nuclei lie in the floor of the fourth ventricle.")
        assert "pontum" not in high                                                # 'dullari' is still no word: the garble is larger than the span
        high, _ = run("Echinocondins such as casp fungin inhibit the synthesis of beta glucan, while azoles inhibit ergosterol synthesis.",
                      "Candida albicans forms pseudohyphae and germ tubes at body temperature.")
        assert "fungin" not in high
        high, medium = run("The inguinal canal runs obliquely above the inguinal ligament.",
                           "The deep inguinal ring is an opening in the transversalis fascia, and the superficial ring is an opening in the external oblique aponeurosis.")
        assert "transversalis" not in high                                         # a real Latin adjective the vocabulary lacks: the original stays


@pytest.mark.skipif(not PUBLIC.is_file(), reason="data/medical_knowledge_public.db is not built")
def test_a_glued_article_is_split_not_dropped():
    from src.medical_knowledge.runtime.engine import V4Engine
    from src.medical_knowledge.runtime.spans import Span

    with MedicalKnowledge.open(PUBLIC) as big:
        engine = V4Engine(big)
        text = "Patients develop anormocytic anemia."
        span = Span(0, 2, 3, 17, 28, "anormocytic", [], ["unknown"])
        shown, _status = engine.render(span, "normocytic", text, False)
        assert shown == "a normocytic"


# ------------------------------------------------------------------------------------------------------------------ round 3: two general rules found on the second blind corpus (heldout_v4)
def test_word_class_reads_the_ending_and_a_dropped_fragment_is_found():
    from src.medical_knowledge.runtime.engine import dropped_fragment, word_class

    assert word_class("pseudohypertrophy") == "noun" and word_class("pseudohypertrophic") == "adjective"
    assert word_class("thrombosis") == "noun" and word_class("thrombotic") == "adjective"
    assert word_class("lipase") is None                                                      # no listed ending: no verdict
    assert word_class("cat") is None                                                         # too short to carry an ending
    assert dropped_fragment("omegaloblastic", "megaloblastic") == "o"                          # a glued word: the leading letter vanishes
    assert dropped_fragment("anormocytic", "a normocytic") == ""                               # the glued article is written out: nothing dropped
    assert dropped_fragment("enzymes", "enzyme") == ""                                         # an inflection is the number logic's business
    assert dropped_fragment("dissel", "disse") == ""                                           # a lost final letter is an ordinary recogniser error: the correction stands
    assert dropped_fragment("hypothalamus", "hypothalamus") == ""
    assert dropped_fragment("glucagonxyz", "glucagon") == ""                                   # more than two letters: another kind of error, not this rule


@pytest.mark.skipif(not PUBLIC.is_file(), reason="data/medical_knowledge_public.db is not built")
def test_a_correction_that_drops_heard_letters_or_changes_the_word_class_is_not_high():
    """heldout_v4, first run: 'omegaloblastic' -> 'megaloblastic' (said: "a megaloblastic") and 'pseudohybrotrophy' -> 'pseudohypertrophic' (said: "pseudohypertrophy")."""
    from src.medical_knowledge.runtime.engine import V4Engine

    with MedicalKnowledge.open(PUBLIC) as big:
        engine = V4Engine(big)

        def run(*sentences):
            result = engine.run(views(*sentences))
            return {c.original: c.replacement for c in result.corrections}, {c.original: c.replacement for c in result.uncertain}

        high, medium = run("Vitamin B12 requires intrinsic factor from the parietal cells for absorption in the terminal ilebum.",
                           "Deficiency causes omegaloblastic anemia and subacute combined degeneration of the spinal cord.",
                           "Methylmalonic acid is raised in vitamin B12 deficiency but not in folate deficiency.")
        assert "omegaloblastic" not in high                                                    # the dropped 'o' is unexplained: suggestion at most
        high, medium = run("Vitamin B12 requires intrinsic factor from the parietal cells for absorption in the terminal",                    # the recogniser's own segments
                           "ilebum. Deficiency causes omegaloblastic anemia and subacute combined degeneration of the spinal",
                           "cord. Methylmalonic acid is raised in vitamin B12 deficiency but not in folate deficiency.")
        assert "omegaloblastic anemia" not in high and "omegaloblastic" not in high            # ... also when the span is the two words "omegaloblastic anemia"
        high, medium = run("Duchenne muscular dystrophy is an X-linked recessive disorder caused by a frameshift deletion in the dystrophin gene.",
                           "Boys develop weakness of the proximal muscles, calf pseudohybrotrophy and the Gower's sign.",
                           "The creatine kinase is markedly elevated.")
        assert "pseudohybrotrophy" not in high                                                 # noun heard, adjective proposed
        high, _ = run("Patients develop anormocytic anemia with a low reticulocyte count.", "The hemoglobin is low and the mean corpuscular volume is normal.")
        assert high.get("anormocytic", "a normocytic") == "a normocytic"                         # the round-2 behaviour is intact: the glued article is written out
