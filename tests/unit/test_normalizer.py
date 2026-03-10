import warnings

from soynlp.normalizer.normalizer import (
    EmojiNormalizer,
    HangleEmojiNormalizer,
    JamoNormalizer,
    PaddingSpacetoWordsNormalizer,
    PassCharacterNormalizer,
    RemoveLongspaceNormalizer,
    RepeatCharacterNormalizer,
    TextNormalizer,
    emoticon_normalize,
    normalize,
    normalize_sent_for_lrgraph,
    only_hangle,
    only_hangle_number,
    only_text,
    remain_hangle_on_last,
    remove_doublespace,
    repeat_normalize,
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


def test_emoji_normalizer():
    """EmojiNormalizer는 Unicode 이모지를 제거하거나 치환한다."""
    n = EmojiNormalizer()

    # 단일 코드포인트 이모지 제거
    assert n.normalize("안녕 😀 반가워") == "안녕  반가워"
    assert n.normalize("파티 🎉") == "파티 "

    # replace 옵션으로 토큰 치환
    n_token = EmojiNormalizer(replace="[EMOJI]")
    assert n_token.normalize("안녕 😀") == "안녕 [EMOJI]"

    # ZWJ 시퀀스 (👨‍💻 = 👨 + ZWJ + 💻)
    assert n.normalize("👨‍💻 코딩") == " 코딩"

    # Skin tone modifier (👋🏽 = 👋 + U+1F3FD)
    assert n.normalize("👋🏽 안녕") == " 안녕"

    # Regional indicator 국기 이모지 (🇰🇷 = 🇰 + 🇷)
    assert n.normalize("🇰🇷 한국") == " 한국"

    # 이모지가 없으면 원문 유지
    assert n.normalize("이모지 없음") == "이모지 없음"

    # 한국어 자모 이모티콘은 처리 대상 아님 (HangleEmojiNormalizer 담당)
    assert n.normalize("ㅋㅋㅋ") == "ㅋㅋㅋ"


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


def _assert_deprecated(func, *args, **kwargs):
    """deprecated 함수가 DeprecationWarning을 정확히 1번 발생시키는지 검증한다."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        func(*args, **kwargs)
        assert len(w) == 1, f"Expected 1 warning, got {len(w)}"
        assert issubclass(w[0].category, DeprecationWarning)
        assert "deprecated" in str(w[0].message).lower()


def test_deprecated_normalize():
    _assert_deprecated(normalize, "안녕 hello 123")
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        assert normalize("안녕 hello 123") == "안녕"


def test_deprecated_remove_doublespace():
    _assert_deprecated(remove_doublespace, "a  b")
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        assert remove_doublespace("a  b") == "a b"


def test_deprecated_repeat_normalize():
    _assert_deprecated(repeat_normalize, "ㅋㅋㅋㅋ")
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        assert repeat_normalize("ㅋㅋㅋㅋ") == "ㅋㅋ"


def test_deprecated_only_hangle():
    _assert_deprecated(only_hangle, "안녕 hello 123")
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        assert only_hangle("안녕 hello 123") == "안녕"


def test_deprecated_only_hangle_number():
    _assert_deprecated(only_hangle_number, "안녕 hello 123")
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        assert only_hangle_number("안녕 hello 123") == "안녕 123"


def test_deprecated_only_text():
    _assert_deprecated(only_text, "안녕 hello @@ 123")
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        assert only_text("안녕 hello @@ 123") == "안녕 hello 123"


def test_remain_hangle_on_last():
    # 한글이 포함된 어절: 마지막 한글 이후 문자 제거
    assert remain_hangle_on_last("안녕123") == "안녕"
    assert remain_hangle_on_last("abc안녕123") == "abc안녕"
    # 한글로 끝나는 경우: 그대로 반환
    assert remain_hangle_on_last("안녕") == "안녕"
    # 한글이 없는 경우: 빈 문자열 반환
    assert remain_hangle_on_last("abc123") == ""
    assert remain_hangle_on_last("") == ""
    # 자모(ㄱ-ㅎ, ㅏ-ㅣ)도 한글로 처리
    assert remain_hangle_on_last("ㅋㅋ123") == "ㅋㅋ"


def test_normalize_sent_for_lrgraph():
    # 기본: 심볼 및 비허용 문자 제거, 각 어절에서 마지막 한글 이후 제거
    assert normalize_sent_for_lrgraph("안녕하세요. 반갑습니다!") == "안녕하세요 반갑습니다"
    # 괄호류(심볼)는 공백으로 치환
    assert normalize_sent_for_lrgraph("(주)삼성 [공지]제목") == "주 삼성 공지 제목"
    # 한글이 없는 어절은 필터링
    assert normalize_sent_for_lrgraph("hello world 안녕") == "안녕"
    # 빈 입력
    assert normalize_sent_for_lrgraph("") == ""
    # 전체가 한글 없는 문장
    assert normalize_sent_for_lrgraph("abc 123") == ""


class TestJamoNormalizer:
    def setup_method(self):
        self.n = JamoNormalizer()

    def test_basic_jamo_sequence(self):
        # ㅆㅡㄹㅐㄱㅣ → 쓰래기
        assert self.n.normalize("ㅆㅡㄹㅐㄱㅣ") == "쓰래기"

    def test_jamo_with_jongsung(self):
        # ㅆㅣㅂㅏㄹ → 씨발
        assert self.n.normalize("ㅆㅣㅂㅏㄹ") == "씨발"

    def test_full_word_decomposed(self):
        # ㅇㅏㄴㄴㅕㅇㅎㅏㅅㅔㅇㅛ → 안녕하세요
        assert self.n.normalize("ㅇㅏㄴㄴㅕㅇㅎㅏㅅㅔㅇㅛ") == "안녕하세요"

    def test_standalone_consonants_unchanged(self):
        # 모음이 없는 자음은 그대로
        assert self.n.normalize("ㅋㅋㅋ") == "ㅋㅋㅋ"

    def test_non_korean_unchanged(self):
        # 비한글 문자는 그대로
        assert self.n.normalize("hello 안녕") == "hello 안녕"

    def test_complete_syllables_unchanged(self):
        # 완성형 한글은 그대로
        assert self.n.normalize("안녕하세요") == "안녕하세요"

    def test_callable(self):
        # __call__ 동작
        assert self.n("ㅆㅡㄹㅐㄱㅣ") == "쓰래기"
