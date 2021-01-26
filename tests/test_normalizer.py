from soynlp.normalizer.normalizer import (
    PassCharacterNormalizer,
    HangleEmojiNormalizer,
    RepeatCharacterNormalizer,
    RemoveLongspaceNormalizer,
    PaddingSpacetoWordsNormalizer,
)


def test_pass_character_normalizer():
    s = "  이것은 abc 123 ().,!?-/ 이 포함된 문장 @@ "
    assert (
        PassCharacterNormalizer(
            alphabet=True, hangle=False, number=False, symbol=False, custom=None
        )(s)
        == "abc"
    )
    assert (
        PassCharacterNormalizer(
            alphabet=True, hangle=True, number=False, symbol=False, custom=None
        )(s)
        == "이것은 abc              이 포함된 문장"
    )
    assert (
        PassCharacterNormalizer(
            alphabet=True, hangle=True, number=True, symbol=False, custom=None
        )(s)
        == "이것은 abc 123          이 포함된 문장"
    )
    assert (
        PassCharacterNormalizer(
            alphabet=True, hangle=True, number=True, symbol=True, custom=None
        )(s)
        == "이것은 abc 123 ().,!?-/ 이 포함된 문장"
    )
    assert (
        PassCharacterNormalizer(
            alphabet=True, hangle=True, number=True, symbol=True, custom="@"
        )(s)
        == "이것은 abc 123 ().,!?-/ 이 포함된 문장 @@"
    )


def test_hangle_emoji_normalizer():
    s = "어머나 ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ쿠ㅜㅜㅜㅜㅜ이런게 있으면 어떻게 떼어내냐 ㅋㅋㅋㅋㅋ쿠ㅜㅜㅜㅜㅜ 하하"
    hangle_emoji = HangleEmojiNormalizer()
    assert hangle_emoji(s) == "어머나 ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅜㅜㅜㅜㅜㅜ이런게 있으면 어떻게 떼어내냐 ㅋㅋㅋㅋㅋㅋㅜㅜㅜㅜㅜㅜ 하하"

    repeat_character = RepeatCharacterNormalizer()
    assert repeat_character(hangle_emoji(s)) == "어머나 ㅋㅋㅜㅜ이런게 있으면 어떻게 떼어내냐 ㅋㅋㅜㅜ 하하"


def test_repeat_character_normalizer():
    assert RepeatCharacterNormalizer()("ㅇㅇㅇㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ") == "ㅇㅇㅋㅋ"
    assert RepeatCharacterNormalizer(max_repeat=3)("ㅇㅇㅇㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ") == "ㅇㅇㅇㅋㅋㅋ"


def test_longspace_normalizer():
    assert RemoveLongspaceNormalizer()("ab     cd    d  f ") == "ab  cd  d  f "


def test_padding_space_to_words():
    assert (
        PaddingSpacetoWordsNormalizer()("(주)일이삼 [[공지]]제목 이것은예시다!!")
        == "( 주 ) 일이삼  [[ 공지 ]] 제목   이것은예시다 !!"
    )
