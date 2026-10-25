"""Text Whisper writes when it hears music, singing or silence: subtitle credits and sign-offs in the languages the program offers.

Measured on a real song compilation: every forced-Korean, Chinese or English decode of an instrumental or sung stretch was one of these ("subtitles provided by ...", "see you in the next
video", "God bless you" repeated). They are not speech, and a language chosen because such a phrase scored well would be a language chosen for the wrong reason. Unicode-aware (the
older English-centred list in ``quality`` does not see Hangul or Han characters).
"""

from __future__ import annotations

import re
import unicodedata

#: credits, sign-offs and filler. A text counts as an artifact when it is made of nothing but these (repeated, with a few words of slack).
PHRASES = (
    # English
    "thanks for watching", "thank you for watching", "please subscribe", "like and subscribe", "subtitles by", "subtitle by", "subtitles provided", "amara org", "transcribed by",
    "god bless you", "see you in the next video", "see you next time", "see you in the next one", "bye bye", "thank you",
    # Korean
    "자막 제공", "자막 제공 및 광고", "한글자막", "다음 영상에서 만나요", "시청해 주셔서 감사합니다", "시청해주셔서 감사합니다", "구독과 좋아요", "구독 좋아요 알림설정", "감사합니다",
    # Chinese
    "中文字幕", "字幕志愿者", "字幕提供", "优优独播剧场", "yoyo television", "请不吝点赞", "感谢观看", "谢谢观看", "谢谢", "订阅",
    # Hindi
    "सब्सक्राइब करें", "वीडियो देखने के लिए धन्यवाद", "धन्यवाद",
    # Turkish
    "abone ol", "altyazı", "izlediğiniz için teşekkürler", "bir sonraki videoda görüşmek üzere",
    # German / Dutch / French / Spanish / Hungarian
    "untertitel der amara", "untertitelung", "vielen dank fürs zuschauen", "ondertitels ingediend door", "ondertiteld door", "sous titres", "merci d avoir regardé",
    "gracias por ver", "suscríbete", "subtítulos", "köszönöm a megtekintést", "köszönöm hogy megnézted", "iratkozz fel",
    # the full credit lines
    "sous titres réalisés par la communauté d amara org", "subtítulos realizados por la comunidad de amara org", "untertitel der amara org community",
    "subtitles by the amara org community", "ondertitels ingediend door de amara org gemeenschap",
)
_SLACK_CHARS = 14
_NOISE = re.compile(r"[\W_]+", re.UNICODE)


def _norm(text: str) -> str:
    return " ".join(_NOISE.sub(" ", unicodedata.normalize("NFKC", text or "").casefold()).split())


_PHRASES = tuple(sorted({_norm(p) for p in PHRASES if _norm(p)}, key=len, reverse=True))


def is_artifact(text: str) -> bool:
    """True when ``text`` is (essentially only) one or more of the phrases above, or has no letters at all ("...", music notes)."""
    cleaned = _norm(text)
    if not cleaned or not any(ch.isalpha() for ch in cleaned):
        return bool(text and text.strip())
    remaining = cleaned
    hit = False
    for phrase in _PHRASES:
        if phrase in remaining:
            hit = True
            remaining = remaining.replace(phrase, " ")
    if not hit:
        return False
    return len(remaining.replace(" ", "")) <= _SLACK_CHARS


#: the part of the list above that is a CREDIT or a call to action ("subtitles by ...", "please subscribe"), as opposed to a polite phrase a speaker may really say ("thank you")
CREDIT_MARKERS = ("subtit", "amara", "subscribe", "watching", "next video", "yoyo", "abone", "altyaz", "untertitel", "ondertitel", "sous titres", "subtítulos", "suscríbete", "iratkozz",
                  "자막", "구독", "시청", "다음 영상", "字幕", "优优独播剧场", "点赞", "观看", "सब्सक्राइब")


_MARKERS = tuple(_norm(marker) for marker in CREDIT_MARKERS)               # normalised like the text (Devanagari vowel signs are not word characters to the pattern above)


def is_credit(text: str) -> bool:
    """A subtitle credit or call to action in any offered language (and nothing else): safe to flag in a transcript of speech, where "thank you" must stay unflagged."""
    cleaned = _norm(text)
    return is_artifact(text) and any(marker in cleaned for marker in _MARKERS)
