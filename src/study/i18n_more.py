"""The fixed wording of the built-in study method for the other Output Languages (German, Hungarian, Korean, Chinese, Hindi, Dutch, French, Spanish).

Only the frame around the lecture's own sentences is written here (card fronts, question stems, explanations); the lecture's sentences are never changed. The wording is kept simple
and was written by the developers, not by native-speaking teachers: a review by a native speaker is welcome (docs/v4_1_1/LANGUAGES.md lists what to check).
"""

from __future__ import annotations

DE = {
    "what_is": "Was ist {x}?", "effect_of": "Welche Wirkung hat: {x}?", "fill_blank": "Füllen Sie die Lücke: {x}", "explain": "Erklären Sie: {x}",
    "mechanism": "Was ist der Mechanismus im Zusammenhang mit {x}?", "structure": "Beschreiben Sie den Aufbau von {x}.", "clinical": "Klinischer Bezug: {x}",
    "compare": "Vergleichen Sie {a} und {b}.", "pathway": "Welche Rolle spielt {x} in dem beschriebenen Stoffwechselweg?", "enzyme": "Was bewirkt {x}?",
    "drug": "Was sagt die Vorlesung über {x} (Arzneimittel / Therapie)?", "role": "Welche Rolle spielt {x}?",
    "which_term": "Welchen Begriff beschreibt die Vorlesung so: \"{x}\"?", "complete": "Vervollständigen Sie die Aussage aus der Vorlesung: \"{x}\"",
    "true_false": "Richtig oder falsch: {x}", "own_words": "In Ihren eigenen Worten: Was ist {x}?", "lecture_say": "Was sagt die Vorlesung über {x}?",
    "explain_why": "Erklären Sie, warum: {x}.", "scenario": "Klinisches Szenario: Es tritt ein Fall auf, der {x} betrifft. Welche Aussage aus dieser Vorlesung trifft zu?",
    "which_mechanism": "Welche Aussage beschreibt den Mechanismus im Zusammenhang mit {x} richtig?", "lecture_says": "Die Vorlesung sagt: \"{x}\"",
    "true": "Richtig", "false": "Falsch", "true_prefix": "Richtig.", "false_prefix": "Falsch.",
}

HU = {
    "what_is": "Mi a(z) {x}?", "effect_of": "Mi a hatása: {x}?", "fill_blank": "Töltse ki a hiányzó részt: {x}", "explain": "Magyarázza el: {x}",
    "mechanism": "Mi a mechanizmus a következővel kapcsolatban: {x}?", "structure": "Írja le a(z) {x} felépítését.", "clinical": "Klinikai kapcsolat: {x}",
    "compare": "Hasonlítsa össze: {a} és {b}.", "pathway": "Milyen szerepet játszik a(z) {x} a leírt útvonalban?", "enzyme": "Mit csinál a(z) {x}?",
    "drug": "Mit mond az előadás a következőről: {x} (gyógyszer / kezelés)?", "role": "Mi a(z) {x} szerepe?",
    "which_term": "Melyik fogalmat írja le az előadás így: \"{x}\"?", "complete": "Egészítse ki az előadásból származó állítást: \"{x}\"",
    "true_false": "Igaz vagy hamis: {x}", "own_words": "Saját szavaival: mi a(z) {x}?", "lecture_say": "Mit mond az előadás a következőről: {x}?",
    "explain_why": "Magyarázza el, miért: {x}.", "scenario": "Klinikai eset: olyan eset merül fel, amely a következőt érinti: {x}. Az előadás melyik állítása érvényes?",
    "which_mechanism": "Melyik állítás írja le helyesen a következővel kapcsolatos mechanizmust: {x}?", "lecture_says": "Az előadás szerint: \"{x}\"",
    "true": "Igaz", "false": "Hamis", "true_prefix": "Igaz.", "false_prefix": "Hamis.",
}

KO = {
    "what_is": "{x}(이)란 무엇입니까?", "effect_of": "다음의 효과는 무엇입니까: {x}?", "fill_blank": "빈칸을 채우세요: {x}", "explain": "설명하세요: {x}",
    "mechanism": "{x}와(과) 관련된 기전은 무엇입니까?", "structure": "{x}의 구조를 설명하세요.", "clinical": "임상 연관성: {x}",
    "compare": "{a}와(과) {b}을(를) 비교하세요.", "pathway": "설명된 경로에서 {x}은(는) 어떤 역할을 합니까?", "enzyme": "{x}은(는) 무엇을 합니까?",
    "drug": "강의는 {x}(약물/치료)에 대해 무엇이라고 말합니까?", "role": "{x}의 역할은 무엇입니까?",
    "which_term": "강의가 다음과 같이 설명하는 용어는 무엇입니까: \"{x}\"?", "complete": "강의의 다음 문장을 완성하세요: \"{x}\"",
    "true_false": "맞으면 참, 틀리면 거짓: {x}", "own_words": "자신의 말로 설명하세요: {x}(이)란 무엇입니까?", "lecture_say": "강의는 {x}에 대해 무엇이라고 말합니까?",
    "explain_why": "그 이유를 설명하세요: {x}.", "scenario": "임상 사례: {x}와(과) 관련된 사례가 있습니다. 이 강의의 어떤 진술이 적용됩니까?",
    "which_mechanism": "{x}와(과) 관련된 기전을 올바르게 설명한 진술은 무엇입니까?", "lecture_says": "강의에서는 다음과 같이 말합니다: \"{x}\"",
    "true": "참", "false": "거짓", "true_prefix": "참.", "false_prefix": "거짓.",
}

ZH = {
    "what_is": "什么是{x}？", "effect_of": "以下内容的作用是什么：{x}？", "fill_blank": "请填空：{x}", "explain": "请解释：{x}",
    "mechanism": "与{x}相关的机制是什么？", "structure": "请描述{x}的结构。", "clinical": "临床联系：{x}",
    "compare": "请比较{a}与{b}。", "pathway": "{x}在所描述的通路中起什么作用？", "enzyme": "{x}的作用是什么？",
    "drug": "讲座中关于{x}（药物/治疗）说了什么？", "role": "{x}的作用是什么？",
    "which_term": "讲座将哪个术语描述为：“{x}”？", "complete": "请补全讲座中的这句话：“{x}”",
    "true_false": "判断对错：{x}", "own_words": "请用自己的话说明：什么是{x}？", "lecture_say": "讲座中关于{x}说了什么？",
    "explain_why": "请解释原因：{x}。", "scenario": "临床情境：出现了一个涉及{x}的病例。本讲座中的哪一项陈述适用？",
    "which_mechanism": "下列哪一项正确描述了与{x}相关的机制？", "lecture_says": "讲座指出：“{x}”",
    "true": "正确", "false": "错误", "true_prefix": "正确。", "false_prefix": "错误。",
}

HI = {
    "what_is": "{x} क्या है?", "effect_of": "इसका प्रभाव क्या है: {x}?", "fill_blank": "रिक्त स्थान भरें: {x}", "explain": "समझाइए: {x}",
    "mechanism": "{x} से जुड़ा तंत्र क्या है?", "structure": "{x} की संरचना का वर्णन कीजिए।", "clinical": "नैदानिक संबंध: {x}",
    "compare": "{a} और {b} की तुलना कीजिए।", "pathway": "वर्णित मार्ग में {x} क्या भूमिका निभाता है?", "enzyme": "{x} क्या करता है?",
    "drug": "व्याख्यान में {x} (दवा / उपचार) के बारे में क्या कहा गया है?", "role": "{x} की क्या भूमिका है?",
    "which_term": "व्याख्यान किस शब्द का इस प्रकार वर्णन करता है: \"{x}\"?", "complete": "व्याख्यान के इस कथन को पूरा कीजिए: \"{x}\"",
    "true_false": "सही या गलत: {x}", "own_words": "अपने शब्दों में बताइए: {x} क्या है?", "lecture_say": "व्याख्यान में {x} के बारे में क्या कहा गया है?",
    "explain_why": "कारण समझाइए: {x}।", "scenario": "नैदानिक परिदृश्य: {x} से जुड़ा एक मामला सामने आता है। इस व्याख्यान का कौन-सा कथन लागू होता है?",
    "which_mechanism": "कौन-सा कथन {x} से जुड़े तंत्र का सही वर्णन करता है?", "lecture_says": "व्याख्यान कहता है: \"{x}\"",
    "true": "सही", "false": "गलत", "true_prefix": "सही।", "false_prefix": "गलत।",
}

NL = {
    "what_is": "Wat is {x}?", "effect_of": "Wat is het effect van: {x}?", "fill_blank": "Vul de ontbrekende tekst in: {x}", "explain": "Leg uit: {x}",
    "mechanism": "Wat is het mechanisme rond {x}?", "structure": "Beschrijf de structuur van {x}.", "clinical": "Klinische relevantie: {x}",
    "compare": "Vergelijk {a} en {b}.", "pathway": "Welke rol speelt {x} in de beschreven route?", "enzyme": "Wat doet {x}?",
    "drug": "Wat zegt het college over {x} (geneesmiddel / behandeling)?", "role": "Wat is de rol van {x}?",
    "which_term": "Welke term beschrijft het college als: \"{x}\"?", "complete": "Maak de uitspraak uit het college af: \"{x}\"",
    "true_false": "Juist of onjuist: {x}", "own_words": "In je eigen woorden: wat is {x}?", "lecture_say": "Wat zegt het college over {x}?",
    "explain_why": "Leg uit waarom: {x}.", "scenario": "Klinisch scenario: er doet zich een geval voor met {x}. Welke uitspraak uit dit college is van toepassing?",
    "which_mechanism": "Welke uitspraak beschrijft het mechanisme rond {x} correct?", "lecture_says": "Het college zegt: \"{x}\"",
    "true": "Juist", "false": "Onjuist", "true_prefix": "Juist.", "false_prefix": "Onjuist.",
}

FR = {
    "what_is": "Qu'est-ce que {x} ?", "effect_of": "Quel est l'effet de : {x} ?", "fill_blank": "Complétez le blanc : {x}", "explain": "Expliquez : {x}",
    "mechanism": "Quel est le mécanisme impliquant {x} ?", "structure": "Décrivez la structure de {x}.", "clinical": "Lien clinique : {x}",
    "compare": "Comparez {a} et {b}.", "pathway": "Quel rôle {x} joue-t-il dans la voie décrite ?", "enzyme": "Que fait {x} ?",
    "drug": "Que dit le cours à propos de {x} (médicament / traitement) ?", "role": "Quel est le rôle de {x} ?",
    "which_term": "Quel terme le cours décrit-il ainsi : « {x} » ?", "complete": "Complétez l'énoncé du cours : « {x} »",
    "true_false": "Vrai ou faux : {x}", "own_words": "Avec vos propres mots : qu'est-ce que {x} ?", "lecture_say": "Que dit le cours à propos de {x} ?",
    "explain_why": "Expliquez pourquoi : {x}.", "scenario": "Situation clinique : un cas implique {x}. Quelle affirmation de ce cours s'applique ?",
    "which_mechanism": "Quelle affirmation décrit correctement le mécanisme impliquant {x} ?", "lecture_says": "Le cours dit : « {x} »",
    "true": "Vrai", "false": "Faux", "true_prefix": "Vrai.", "false_prefix": "Faux.",
}

ES = {
    "what_is": "¿Qué es {x}?", "effect_of": "¿Cuál es el efecto de: {x}?", "fill_blank": "Complete el espacio en blanco: {x}", "explain": "Explique: {x}",
    "mechanism": "¿Cuál es el mecanismo relacionado con {x}?", "structure": "Describa la estructura de {x}.", "clinical": "Conexión clínica: {x}",
    "compare": "Compare {a} y {b}.", "pathway": "¿Qué papel desempeña {x} en la vía descrita?", "enzyme": "¿Qué hace {x}?",
    "drug": "¿Qué dice la clase sobre {x} (fármaco / tratamiento)?", "role": "¿Cuál es la función de {x}?",
    "which_term": "¿Qué término describe la clase como: \"{x}\"?", "complete": "Complete la afirmación de la clase: \"{x}\"",
    "true_false": "Verdadero o falso: {x}", "own_words": "Con sus propias palabras: ¿qué es {x}?", "lecture_say": "¿Qué dice la clase sobre {x}?",
    "explain_why": "Explique por qué: {x}.", "scenario": "Escenario clínico: surge un caso que involucra {x}. ¿Qué afirmación de esta clase se aplica?",
    "which_mechanism": "¿Qué afirmación describe correctamente el mecanismo relacionado con {x}?", "lecture_says": "La clase dice: \"{x}\"",
    "true": "Verdadero", "false": "Falso", "true_prefix": "Verdadero.", "false_prefix": "Falso.",
}

TABLES = {"de": DE, "hu": HU, "ko": KO, "zh": ZH, "hi": HI, "nl": NL, "fr": FR, "es": ES}

#: every spelling of True / False accepted as an answer -> (index in the pair, language)
TRUE_FALSE = {
    "wahr": (0, "de"), "richtig": (0, "de"), "falsch": (1, "de"),
    "igaz": (0, "hu"), "hamis": (1, "hu"),
    "참": (0, "ko"), "맞음": (0, "ko"), "거짓": (1, "ko"), "틀림": (1, "ko"),
    "正确": (0, "zh"), "对": (0, "zh"), "错误": (1, "zh"), "错": (1, "zh"),
    "सही": (0, "hi"), "सच": (0, "hi"), "गलत": (1, "hi"), "झूठ": (1, "hi"),
    "waar": (0, "nl"), "juist": (0, "nl"), "onwaar": (1, "nl"), "onjuist": (1, "nl"),
    "vrai": (0, "fr"), "faux": (1, "fr"),
    "verdadero": (0, "es"), "cierto": (0, "es"), "falso": (1, "es"),
}
