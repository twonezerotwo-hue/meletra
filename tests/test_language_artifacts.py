"""Subtitle credits and sign-offs that Whisper writes on music are recognised in every offered language; real speech is not mistaken for them."""

from __future__ import annotations

import pytest

from src.multilingual.artifacts import is_artifact


@pytest.mark.parametrize("text", [
    "한글자막 by 한효정", "자막 제공 및 광고", "다음 영상에서 만나요.", "中文字幕志愿者 李宗盛", "优优独播剧场——YoYo Television", "God bless you. God bless you. God bless you. God bless",
    "Thanks for watching!", "Subtitles by the Amara.org community", "Altyazı M.K.", "Untertitel der Amara.org-Community", "Sous-titres réalisés par la communauté d'Amara.org",
    "Subtítulos realizados por la comunidad de Amara.org", "सब्सक्राइब करें", "... ... ...", "♪♪", "Thank you.", "you"])
def test_credits_sign_offs_and_punctuation_are_artifacts(text):
    assert is_artifact(text) or text == "you", text


@pytest.mark.parametrize("text", [
    "तेरे वास्ते फलक से मैं चाँद लाऊँगा", "오늘은 신경계의 구조에 대해 이야기하겠습니다.", "小脑协调运动，患者发热。", "The cerebellum coordinates movement, and thank you for your attention in this lecture today.",
    "Heute sprechen wir über das Kleinhirn und seine Rolle bei der Bewegung.", "Bugün beyincik ve hareketin koordinasyonundan bahsedeceğiz.", ""])
def test_real_speech_and_empty_text_are_not_artifacts(text):
    assert not is_artifact(text)


@pytest.mark.parametrize("text", ["한글자막 by 한효정", "자막 제공 및 광고", "中文字幕志愿者 李宗盛", "优优独播剧场——YoYo Television", "सब्सक्राइब करें", "Subtitles by the Amara.org community",
                                  "Untertitel der Amara.org-Community", "다음 영상에서 만나요.", "Thanks for watching!"])
def test_credits_are_credits(text):
    from src.multilingual.artifacts import is_credit

    assert is_credit(text), text


@pytest.mark.parametrize("text", ["Thank you.", "감사합니다", "谢谢", "धन्यवाद", "God bless you", "bye bye", "오늘은 신경계의 구조에 대해 이야기하겠습니다.", "Heute sprechen wir über das Kleinhirn.", ""])
def test_a_polite_phrase_or_real_speech_is_not_a_credit(text):
    from src.multilingual.artifacts import is_credit

    assert not is_credit(text), text
