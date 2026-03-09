import warnings

from soynlp.normalizer.normalizer import (
    HangleEmojiNormalizer,
    PaddingSpacetoWordsNormalizer,
    PassCharacterNormalizer,
    RemoveLongspaceNormalizer,
    RepeatCharacterNormalizer,
    TextNormalizer,
    emoticon_normalize,
    text_normalizer,
)


def test_pass_character_normalizer():
    s = "  이것은 abc 123 ().,!?-/ 이 포함된 문장 @@ "
    assert PassCharacterNormalizer(alphabet=True, hangle=False, number=False, symbol=False, custom=None)(s) == "abc"
    assert (
        PassCharacterNormalizer(alphabet=True, hangle=True, number=False, symbol=False, custom=None)(s)
        == "이것은 abc              이 포함된 문장"
    )
    assert (
        PassCharacterNormalizer(alphabet=True, hangle=True, number=True, symbol=False, custom=None)(s)
        == "이것은 abc 123          이 포함된 문장"
    )
    assert (
        PassCharacterNormalizer(alphabet=True, hangle=True, number=True, symbol=True, custom=None)(s)
        == "이것은 abc 123 ().,!?-/ 이 포함된 문장"
    )
    assert (
        PassCharacterNormalizer(alphabet=True, hangle=True, number=True, symbol=True, custom="@")(s)
        == "이것은 abc 123 ().,!?-/ 이 포함된 문장 @@"
    )


def test_hangle_emoji_normalizer():
    s = "어머나 ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ쿠ㅜㅜㅜㅜㅜ이런게 있으면 어떻게 떼어내냐 ㅋㅋㅋㅋㅋ쿠ㅜㅜㅜㅜㅜ 하하"
    hangle_emoji = HangleEmojiNormalizer()
    assert (
        hangle_emoji(s)
        == "어머나 ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅜㅜㅜㅜㅜㅜ이런게 있으면 어떻게 떼어내냐 ㅋㅋㅋㅋㅋㅋㅜㅜㅜㅜㅜㅜ 하하"
    )

    repeat_character = RepeatCharacterNormalizer()
    assert repeat_character(hangle_emoji(s)) == "어머나 ㅋㅋㅜㅜ이런게 있으면 어떻게 떼어내냐 ㅋㅋㅜㅜ 하하"


def test_repeat_character_normalizer():
    assert RepeatCharacterNormalizer()("ㅇㅇㅇㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ") == "ㅇㅇㅋㅋ"
    assert RepeatCharacterNormalizer(max_repeat=3)("ㅇㅇㅇㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ") == "ㅇㅇㅇㅋㅋㅋ"


def test_longspace_normalizer():
    assert RemoveLongspaceNormalizer()("ab     cd    d  f ") == "ab cd d f "
    assert RemoveLongspaceNormalizer()("a\t\tb") == "a b"
    assert RemoveLongspaceNormalizer()("a\n\nb") == "a b"
    assert RemoveLongspaceNormalizer()("a b") == "a b"


def test_padding_space_to_words():
    assert (
        PaddingSpacetoWordsNormalizer()("(주)일이삼 [[공지]]제목 이것은예시다!!")
        == "( 주 ) 일이삼  [[ 공지 ]] 제목   이것은예시다 !!"
    )


def test_normalizer_builder():
    normalizer = TextNormalizer.build_normalizer()
    assert (
        normalizer("어머나 ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ쿠ㅜㅜㅜㅜㅜ이런게 있으면 어떻게 떼어내냐 ㅋㅋㅋㅋㅋ쿠ㅜㅜㅜㅜㅜ 하하")
        == "어머나 ㅋㅋㅜㅜ이런게 있으면 어떻게 떼어내냐 ㅋㅋㅜㅜ 하하"
    )

    normalizer = TextNormalizer.build_normalizer(remove_repeatchar=3)
    assert (
        normalizer("어머나 ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ쿠ㅜㅜㅜㅜㅜ이런게 있으면 어떻게 떼어내냐 ㅋㅋㅋㅋㅋ쿠ㅜㅜㅜㅜㅜ 하하")
        == "어머나 ㅋㅋㅋㅜㅜㅜ이런게 있으면 어떻게 떼어내냐 ㅋㅋㅋㅜㅜㅜ 하하"
    )

    assert normalizer("(주)일이삼 [[공지]]제목 이것은예시다!!") == "(주)일이삼 [[공지]]제목 이것은예시다!!"

    normalizer = TextNormalizer.build_normalizer(padding_space=True)
    assert normalizer("(주)일이삼 [[공지]]제목 이것은예시다!!") == "( 주 ) 일이삼 [[ 공지 ]] 제목 이것은예시다 !!"

    normalizer = TextNormalizer.build_normalizer(padding_space=True, symbol=False)
    assert normalizer("(주)일이삼 [[공지]]제목 이것은예시다!!") == " 주 일이삼 공지 제목 이것은예시다 "

    normalizer = TextNormalizer.build_normalizer(padding_space=False, symbol=False, custom="/:@.")
    assert (
        normalizer("soynlp의 주소는 https://github.com/lovit/soynlp/ 입니다.")
        == "soynlp의 주소는 https://github.com/lovit/soynlp/ 입니다."
    )


def test_emoticon_normalize_deprecated():
    """emoticon_normalize는 deprecated이며 HangleEmojiNormalizer와 동일 결과를 반환한다."""
    s = "어머나 ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ쿠ㅜㅜㅜㅜㅜ이런게 있으면 어떻게 떼어내냐 ㅋㅋㅋㅋㅋ쿠ㅜㅜㅜㅜㅜ 하하"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = emoticon_normalize(s, num_repeats=2)
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "deprecated" in str(w[0].message).lower()

    expected = RepeatCharacterNormalizer(max_repeat=2)(HangleEmojiNormalizer()(s))
    assert result == expected

    # 기존 구현의 버그: 'ㅋ크ㅋ' → 'ㅋㅋ' (크가 묵소 삭제됨)
    # HangleEmojiNormalizer는 이를 올바르게 처리: 'ㅋ크ㅋ' → 'ㅋ크ㅋ' (변경 없음)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        assert emoticon_normalize("ㅋ크ㅋ", num_repeats=0) == "ㅋ크ㅋ"


def test_default_text_normalizer():
    assert (
        text_normalizer("어머나 ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ쿠ㅜㅜㅜㅜㅜ이런게 있으면 어떻게 떼어내냐 ㅋㅋㅋㅋㅋ쿠ㅜㅜㅜㅜㅜ 하하")
        == "어머나 ㅋㅋㅜㅜ이런게 있으면 어떻게 떼어내냐 ㅋㅋㅜㅜ 하하"
    )
