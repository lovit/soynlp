import pytest

from soynlp.hangle import (
    ConvolutionHangleEncoder,
    character_is_complete_korean,
    character_is_english,
    character_is_jaum,
    character_is_korean,
    character_is_moum,
    character_is_number,
    character_is_punctuation,
    compose,
    cosine_distance,
    decompose,
    jaccard_distance,
    jamo_levenshtein,
    levenshtein,
    text_to_jamo,
    to_base,
)


class TestDecompose:
    def test_complete_korean(self):
        assert decompose("한") == ("ㅎ", "ㅏ", "ㄴ")
        assert decompose("글") == ("ㄱ", "ㅡ", "ㄹ")

    def test_no_jongsung(self):
        assert decompose("가") == ("ㄱ", "ㅏ", " ")

    def test_jaum(self):
        assert decompose("ㄱ") == ("ㄱ", " ", " ")

    def test_moum(self):
        assert decompose("ㅏ") == (" ", "ㅏ", " ")

    def test_non_korean(self):
        assert decompose("a") is None
        assert decompose("1") is None


class TestCompose:
    def test_with_jongsung(self):
        assert compose("ㅎ", "ㅏ", "ㄴ") == "한"

    def test_without_jongsung(self):
        assert compose("ㄱ", "ㅏ", " ") == "가"

    def test_roundtrip(self):
        for char in "소인엘피":
            result = decompose(char)
            assert result is not None
            cho, jung, jong = result
            assert compose(cho, jung, jong) == char


class TestCharacterIs:
    def test_korean(self):
        assert character_is_korean("한") is True
        assert character_is_korean("ㄱ") is True
        assert character_is_korean("ㅏ") is True
        assert character_is_korean("a") is False

    def test_complete_korean(self):
        assert character_is_complete_korean("한") is True
        assert character_is_complete_korean("ㄱ") is False

    def test_jaum(self):
        assert character_is_jaum("ㄱ") is True
        assert character_is_jaum("ㅏ") is False

    def test_moum(self):
        assert character_is_moum("ㅏ") is True
        assert character_is_moum("ㄱ") is False

    def test_number(self):
        assert character_is_number("5") is True
        assert character_is_number("a") is False

    def test_english(self):
        assert character_is_english("a") is True
        assert character_is_english("Z") is True
        assert character_is_english("1") is False

    def test_punctuation(self):
        assert character_is_punctuation("!") is True
        assert character_is_punctuation("?") is True
        assert character_is_punctuation("a") is False


class TestToBase:
    def test_str(self):
        assert to_base("가") == 44032

    def test_int(self):
        assert to_base(44032) == 44032

    def test_invalid_type(self):
        with pytest.raises(TypeError):
            to_base([1])  # type: ignore[arg-type]


class TestLevenshtein:
    def test_same(self):
        assert levenshtein("abc", "abc") == 0

    def test_insert(self):
        assert levenshtein("abc", "ab") == 1

    def test_delete(self):
        # s1이 더 짧으면 내부적으로 뒤집어서 처리됨
        assert levenshtein("ab", "abc") == 1

    def test_substitute(self):
        assert levenshtein("abc", "axc") == 1

    def test_empty(self):
        assert levenshtein("abc", "") == 3
        assert levenshtein("", "abc") == 3

    def test_completely_different(self):
        assert levenshtein("abc", "xyz") == 3

    def test_custom_cost(self):
        assert levenshtein("a", "b", cost={("a", "b"): 0.5, ("b", "a"): 0.5}) == 0.5

    def test_symmetry(self):
        assert levenshtein("kitten", "sitting") == levenshtein("sitting", "kitten")


class TestJamoLevenshtein:
    def test_same(self):
        assert jamo_levenshtein("한글", "한글") == 0

    def test_similar(self):
        # 한글 vs 한금: 받침만 다름 → 자모 거리 < 1
        dist = jamo_levenshtein("한글", "한금")
        assert 0 < dist < 1

    def test_empty(self):
        assert jamo_levenshtein("한", "") == 1
        assert jamo_levenshtein("", "한") == 1

    def test_non_korean_chars(self):
        # 비한글 문자는 1 단위로 처리됨
        assert jamo_levenshtein("abc", "abc") == 0
        assert jamo_levenshtein("abc", "abx") == 1

    def test_less_than_levenshtein(self):
        # 자모 거리 < 일반 거리 (한글이 비슷한 경우)
        normal = levenshtein("한글", "한금")
        jamo = jamo_levenshtein("한글", "한금")
        assert jamo < normal


class TestCosineDistance:
    def test_same(self):
        assert cosine_distance("aaa", "aaa") == pytest.approx(0.0)

    def test_empty_s1(self):
        assert cosine_distance("", "abc") == 2

    def test_empty_s2(self):
        assert cosine_distance("abc", "") == 2

    def test_orthogonal(self):
        # "aaa" vs "bbb": 공통 문자 없음 → cosine=0, distance=1
        assert cosine_distance("aaa", "bbb") == pytest.approx(1.0)

    def test_partial_overlap(self):
        d = cosine_distance("abc", "abx")
        assert 0 < d < 1

    def test_range(self):
        # 거리는 항상 [0, 2] 범위
        for s1, s2 in [("abc", "abc"), ("abc", "def"), ("abc", "abx")]:
            d = cosine_distance(s1, s2)
            assert 0.0 <= d <= 2.0


class TestJaccardDistance:
    def test_same(self):
        assert jaccard_distance("abc", "abc") == pytest.approx(0.0)

    def test_disjoint(self):
        assert jaccard_distance("abc", "def") == pytest.approx(1.0)

    def test_empty_s1(self):
        assert jaccard_distance("", "abc") == 1

    def test_empty_s2(self):
        assert jaccard_distance("abc", "") == 1

    def test_partial_overlap(self):
        # "ab" vs "bc": 교집합 {"b"}, 합집합 {"a","b","c"} → 1 - 1/3 = 2/3
        d = jaccard_distance("ab", "bc")
        assert d == pytest.approx(2 / 3)

    def test_custom_unitfy(self):
        # 단어 단위 분리

        d = jaccard_distance("hello world", "hello earth", unitfy=lambda s: set(s.split()))
        assert 0 < d < 1


class TestTextToJamo:
    def test_with_jongsung(self):
        assert text_to_jamo("한글") == "ㅎㅏㄴㄱㅡㄹ"

    def test_without_jongsung(self):
        assert text_to_jamo("나는") == "ㄴㅏㄴㅡㄴ"

    def test_non_korean_preserved(self):
        result = text_to_jamo("abc한")
        assert result == "abcㅎㅏㄴ"

    def test_space_preserved(self):
        result = text_to_jamo("한 글")
        assert " " in result

    def test_join_jongsung_false(self):
        # 종성 없는 음절은 공백 포함 3자모
        result = text_to_jamo("가나", join_jongsung=False)
        assert result == "ㄱㅏ ㄴㅏ "

    def test_mixed_jongsung(self):
        # '한'(종성ㄴ) + '가'(종성없음) — join_jongsung=False 시 길이 체크
        result = text_to_jamo("한가", join_jongsung=False)
        # '한' → ㅎㅏㄴ (3), '가' → ㄱㅏ  (3, 마지막 공백)
        assert len(result) == 6

    def test_empty(self):
        assert text_to_jamo("") == ""


class TestConvolutionHangleEncoder:
    def test_encode_decode_roundtrip(self):
        encoder = ConvolutionHangleEncoder()
        text = "한글 테스트"
        onehot = encoder.sent_to_onehot(text)
        decoded = encoder.onehot_to_sent(onehot)
        assert decoded == text

    def test_encode_shape(self):
        encoder = ConvolutionHangleEncoder()
        x = encoder.encode("한글")
        assert x.shape == (2, 80)
