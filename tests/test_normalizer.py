from soynlp.normalizer.normalizer import (
    PassCharacterNormalizer,
    RepeatCharacterNormalize,
    RemoveLongspaceNormalizer,
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


def test_repeat_character_normalizer():
    assert RepeatCharacterNormalize()("ㅇㅇㅇㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ") == "ㅇㅇㅋㅋ"
    assert RepeatCharacterNormalize(max_repeat=3)("ㅇㅇㅇㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ") == "ㅇㅇㅇㅋㅋㅋ"


def test_longspace_normalizer():
    assert RemoveLongspaceNormalizer()("ab     cd    d  f ") == "ab  cd  d  f "
