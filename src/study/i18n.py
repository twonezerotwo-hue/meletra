"""The fixed wording the built-in study method puts around the lecture's own sentences (card fronts, question stems, explanations).

English and Turkish (the two languages of the lectures) and, since 4.1.1, the other Output Languages (src/study/i18n_more.py); any other language code falls back to English. The lecture's sentences are never
translated: only the frame around them changes. (A language model writes the whole item in the language it is asked for.)
"""

from __future__ import annotations

EN = {
    "what_is": "What is {x}?",
    "effect_of": "What is the effect of: {x}?",
    "fill_blank": "Fill in the blank: {x}",
    "explain": "Explain: {x}",
    "mechanism": "What is the mechanism involving {x}?",
    "structure": "Describe the structure of {x}.",
    "clinical": "Clinical connection: {x}",
    "compare": "Compare {a} and {b}.",
    "pathway": "What role does {x} play in the pathway described?",
    "enzyme": "What does {x} do?",
    "drug": "What does the lecture say about {x} (drug / treatment)?",
    "role": "What is the role of {x}?",
    "which_term": "Which term does the lecture describe as: \"{x}\"?",
    "complete": "Complete the statement from the lecture: \"{x}\"",
    "true_false": "True or false: {x}",
    "own_words": "In your own words: what is {x}?",
    "lecture_say": "What does the lecture say about {x}?",
    "explain_why": "Explain why: {x}.",
    "scenario": "Clinical scenario: a case comes up that involves {x}. Which statement from this lecture applies?",
    "which_mechanism": "Which statement correctly describes the mechanism involving {x}?",
    "lecture_says": "The lecture says: \"{x}\"",
    "true": "True",
    "false": "False",
    "true_prefix": "True.",
    "false_prefix": "False.",
}

TR = {
    "what_is": "{x} nedir?",
    "effect_of": "Şunun sonucu nedir: {x}?",
    "fill_blank": "Boşluğu doldurun: {x}",
    "explain": "Açıklayın: {x}",
    "mechanism": "{x} ile ilgili mekanizma nedir?",
    "structure": "{x} yapısını anlatın.",
    "clinical": "Klinik bağlantı: {x}",
    "compare": "{a} ile {b} arasındaki farkı karşılaştırın.",
    "pathway": "Anlatılan yolakta {x} hangi rolü oynar?",
    "enzyme": "{x} ne yapar?",
    "drug": "Derste {x} (ilaç / tedavi) hakkında ne söyleniyor?",
    "role": "{x} nin rolü nedir?",
    "which_term": "Derste şu şekilde tanımlanan terim hangisidir: \"{x}\"?",
    "complete": "Dersten alınan ifadeyi tamamlayın: \"{x}\"",
    "true_false": "Doğru mu yanlış mı: {x}",
    "own_words": "Kendi sözlerinizle: {x} nedir?",
    "lecture_say": "Derste {x} hakkında ne söyleniyor?",
    "explain_why": "Nedenini açıklayın: {x}.",
    "scenario": "Klinik durum: {x} ile ilgili bir olgu var. Bu dersteki hangi ifade geçerlidir?",
    "which_mechanism": "{x} ile ilgili mekanizmayı hangi ifade doğru anlatır?",
    "lecture_says": "Derste şöyle söyleniyor: \"{x}\"",
    "true": "Doğru",
    "false": "Yanlış",
    "true_prefix": "Doğru.",
    "false_prefix": "Yanlış.",
}

from .i18n_more import TABLES as _MORE_TABLES, TRUE_FALSE as _MORE_TRUE_FALSE  # noqa: E402 - the other Output Languages (German, Hungarian, Korean, Chinese, Hindi, Dutch, French, Spanish)

_TABLES = {"en": EN, "tr": TR, **_MORE_TABLES}
#: every spelling of True / False the study system accepts -> (index in the pair, language of the pair)
TRUE_FALSE_WORDS = {"true": (0, "en"), "false": (1, "en"), "doğru": (0, "tr"), "yanlış": (1, "tr"), **_MORE_TRUE_FALSE}


def strings(language: str) -> dict[str, str]:
    return _TABLES.get((language or "en").lower()[:2], EN)


def true_false_pair(language: str) -> tuple[str, str]:
    table = strings(language)
    return table["true"], table["false"]


def canonical_true_false(answer: str) -> tuple[str, tuple[str, str]] | None:
    """``"doğru"`` -> ``("Doğru", ("Doğru", "Yanlış"))``; ``None`` if the answer is not a True/False word."""
    entry = TRUE_FALSE_WORDS.get(answer.strip().casefold())
    if entry is None:
        return None
    index, language = entry
    pair = true_false_pair(language)
    return pair[index], pair
