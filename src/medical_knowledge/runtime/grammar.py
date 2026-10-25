"""Lightweight grammatical and semantic compatibility between what was heard and a candidate - no NLP pipeline, a handful of structural rules.

A replacement should keep the ROLE of the word: a verb stays a verb ("they embolize" is not "they embolus"), an adjective stays an adjective ("the tracheal muscle" is not "the trachea muscle"),
a noun stays a noun. And it should keep the KIND of thing the sentence is about: "the X muscle" expects a structure, "X syndrome" a disease, "the process of X" a process. Phonetic similarity must
never override either. Both checks return a value in 0..1 (1 = compatible, 0.6 = nothing can be said, 0 = incompatible); the gate refuses HIGH for 0 and charges a cost for low values.
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

_DETERMINERS = {"the", "a", "an", "this", "that", "these", "those", "each", "every", "its", "their", "his", "her", "our", "some", "any", "no", "all", "both"}
_MODALS = {"to", "may", "can", "will", "would", "should", "must", "might", "could", "shall", "does", "did", "do", "not", "also", "then", "often", "usually", "can't", "cannot"}
_VERB_END = re.compile(r"(?:ize|ise|ify)$")
_ADJ_END = re.compile(r"(?:ical|ular|ary|ory|ous|ive|ial|ic|al|ar|oid|ile)$")
_NOUN_END = re.compile(r"(?:tion|sion|ment|ness|ity|ism|ist|itis|osis|oma|emia|pathy|ology|ectomy|otomy|ase|gen|cyte|blast|plasia|genesis|lysis|ure)s?$")
_STRUCTURE_HEADS = {"cell", "cells", "nucleus", "nuclei", "nerve", "nerves", "artery", "arteries", "vein", "veins", "muscle", "muscles", "gland", "glands", "duct", "ducts", "tract", "tracts",
                    "ganglion", "ganglia", "bone", "bones", "membrane", "membranes", "layer", "layers", "plexus", "fiber", "fibers", "fibre", "fibres", "lobe", "gyrus", "sulcus", "ventricle",
                    "canal", "ligament", "valve", "vessel", "vessels", "tube", "plate", "tubule", "tubules", "organ", "tissue", "cortex", "zone", "junction", "pathway", "cord", "root", "horn",
                    "capsule", "septum", "sinus", "node", "lymph", "arch", "fold", "pouch", "vesicle", "body", "bodies", "disc", "discs", "lamina", "matrix", "wall", "border", "apparatus"}
_DISEASE_HEADS = {"syndrome", "disease", "disorder", "deficiency", "failure", "infarction", "carcinoma", "lymphoma", "tumor", "tumour", "cancer", "infection", "anemia", "palsy", "dystrophy",
                  "hernia", "fracture", "necrosis", "ischemia", "fever"}
_PROCESS_WORDS = {"transport", "signaling", "signalling", "metabolism", "synthesis", "cycle", "reaction", "response", "phosphorylation", "mitosis", "meiosis", "replication", "transcription",
                  "translation", "degradation", "regulation", "activation", "inhibition", "differentiation", "migration", "development", "secretion", "absorption", "excretion", "conduction"}
_PROCESS_END = re.compile(r"(?:ation|esis|lysis|ization|ing)$")
_DRUG_END = re.compile(r"(?:olol|pril|sartan|statin|mab|cillin|mycin|azole|pine|zepam|oxacin|cycline|barbital|caine|dipine|prazole|vir|tidine|setron|triptan|gliptin|glitazone|mide)$")
_PROTEIN_END = re.compile(r"(?:ase|kinase|ectin|globulin|albumin|insulin|opsin|actin|myosin|tropin|ferrin)$")
_PROTEIN_HEADS = {"receptor", "receptors", "kinase", "factor", "hormone", "antibody", "toxin", "channel", "transporter", "enzyme", "protein", "ligand", "cytokine"}


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def heard_role(before: str, span: str, after: str) -> str:
    """'verb' | 'adj' | 'noun' | 'unknown' for the last word of the heard span, from its ending and the words around it."""
    ws = _words(span)
    if not ws:
        return "unknown"
    w = ws[-1]
    prev = (_words(before) or [""])[-1] if len(ws) == 1 else (ws[-2] if len(ws) > 1 else "")
    nxt = (_words(after) or [""])[0]
    if _NOUN_END.search(w) and not _VERB_END.search(w):
        return "noun"
    if _VERB_END.search(w):
        return "unknown" if prev in _DETERMINERS else "verb"                 # "to embolize", "they embolize"; after "the" it is something else
    if prev in _MODALS and len(w) > 3 and not w.endswith("s"):
        return "verb"
    if _ADJ_END.search(w) and (nxt in _STRUCTURE_HEADS or nxt in _DISEASE_HEADS or prev in _DETERMINERS or prev in {"is", "are", "was", "were"}):
        return "adj"
    if prev in _DETERMINERS:
        return "noun"
    return "unknown"


def candidate_role(surface: str, kind: str = "term") -> str:
    ws = _words(re.sub(r"\([^)]*\)", " ", surface))
    if not ws:
        return "unknown"
    if kind in ("gene", "drug", "eponym", "latin") or len(ws) > 1:
        return "noun"
    w = ws[-1]
    if _VERB_END.search(w):
        return "verb"
    if _ADJ_END.search(w) and not _NOUN_END.search(w) and w not in {"nodal", "canal", "signal", "hormonal", "cortical", "plural", "oral", "fecal"}:
        return "adj"
    return "noun"


def role_fit(heard: str, candidate: str) -> float:
    if heard == "unknown" or candidate == "unknown":
        return 0.6
    if heard == candidate:
        return 1.0
    if "verb" in (heard, candidate):
        return 0.0                                  # "embolize" is not "embolus": a verb is never a noun
    return 0.25                                     # adjective <-> noun ("tracheal" / "trachea"): possible, but a different word


def candidate_class(surface: str, kind: str = "term", semantic_type: str = "") -> str:
    """'gene' | 'drug' | 'protein' | 'disease' | 'process' | 'structure' | 'unknown' - from the source's own type where there is one, else from the ending / head word."""
    if kind == "gene":
        return "gene"
    if kind == "drug":
        return "drug"
    if semantic_type in ("biological_process",):
        return "process"
    if semantic_type in ("cellular_component",):
        return "structure"
    if semantic_type in ("molecular_function",):
        return "protein"
    ws = _words(re.sub(r"\([^)]*\)", " ", surface))
    if not ws:
        return "unknown"
    head = ws[-1]
    if head in _DISEASE_HEADS or re.search(r"(?:itis|oma|pathy|emia|algia|osis)$", head) and head not in {"mitosis", "meiosis", "apoptosis", "phagocytosis", "exocytosis", "endocytosis", "osmosis"}:
        return "disease"
    if head in _PROCESS_WORDS or (len(ws) > 1 and ws[0] in _PROCESS_WORDS) or _PROCESS_END.search(head):
        return "process"
    if head in _STRUCTURE_HEADS:
        return "structure"
    if head in _PROTEIN_HEADS or _PROTEIN_END.search(head):
        return "protein"
    if _DRUG_END.search(head):
        return "drug"
    return "unknown"


def frame_expectation(before: str, after: str) -> Optional[str]:
    """What kind of thing the sentence frame asks for at this position, when it says so unmistakably."""
    nxt = (_words(after) or [""])[0]
    prev2 = _words(before)[-2:]
    if nxt in _STRUCTURE_HEADS:
        return "structure"                          # "the X muscle", "the X nucleus"
    if nxt in _DISEASE_HEADS:
        return "disease"                            # "X syndrome"
    if nxt in _PROTEIN_HEADS:
        return "protein"                            # "X receptor"
    if prev2 == ["process", "of"] or prev2[-1:] == ["during"] and nxt in {"the", "a"}:
        return "process"
    return None


def type_fit(candidate: str, heard: str, expected: Optional[str]) -> float:
    """1 compatible / 0.6 unknown / 0 incompatible."""
    if candidate == "unknown":
        return 0.6
    if expected is not None:
        ok = {"structure": {"structure", "unknown"}, "disease": {"disease", "structure", "unknown"}, "protein": {"protein", "gene", "unknown"}, "process": {"process", "unknown"}}[expected]
        return 1.0 if candidate in ok else 0.0
    if heard != "unknown" and heard != candidate:
        return 0.3
    return 1.0 if heard == candidate else 0.6


def assess(before: str, span: str, after: str, surface: str, kind: str, semantic_type: str = "") -> dict:
    """The two numbers the scorer stores: 'pos' (grammatical role) and 'type_fit' (kind of thing)."""
    h_role, c_role = heard_role(before, span, after), candidate_role(surface, kind)
    h_class = candidate_class(span, "term")
    c_class = candidate_class(surface, kind, semantic_type)
    return {"pos": role_fit(h_role, c_role), "type_fit": type_fit(c_class, h_class, frame_expectation(before, after)),
            "heard_role": h_role, "candidate_role": c_role, "heard_class": h_class, "candidate_class": c_class}
