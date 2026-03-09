import re

import numpy as np

# 한글 음절 유니코드 범위: U+AC00 (가) ~ U+D7A3 (힣)
# 공식 규격: https://www.unicode.org/charts/PDF/UAC00.pdf
# 음절 코드포인트 = _kor_begin + (초성_idx * 21 + 중성_idx) * 28 + 종성_idx
_kor_begin = 44032  # U+AC00 '가'
_kor_end = 55203  # U+D7A3 '힣'
_chosung_base = 588  # 중성(21) × 종성(28) = 588
_jungsung_base = 28  # 종성 수

# 자모 유니코드 범위 (호환 자모 블록: Hangul Compatibility Jamo)
# https://www.unicode.org/charts/PDF/U3130.pdf
_jaum_begin = 12593  # U+3131 'ㄱ'
_jaum_end = 12622  # U+314E 'ㅎ'
_moum_begin = 12623  # U+314F 'ㅏ'
_moum_end = 12643  # U+3163 'ㅣ'

chosung_list = [
    "ㄱ",
    "ㄲ",
    "ㄴ",
    "ㄷ",
    "ㄸ",
    "ㄹ",
    "ㅁ",
    "ㅂ",
    "ㅃ",
    "ㅅ",
    "ㅆ",
    "ㅇ",
    "ㅈ",
    "ㅉ",
    "ㅊ",
    "ㅋ",
    "ㅌ",
    "ㅍ",
    "ㅎ",
]

jungsung_list = [
    "ㅏ",
    "ㅐ",
    "ㅑ",
    "ㅒ",
    "ㅓ",
    "ㅔ",
    "ㅕ",
    "ㅖ",
    "ㅗ",
    "ㅘ",
    "ㅙ",
    "ㅚ",
    "ㅛ",
    "ㅜ",
    "ㅝ",
    "ㅞ",
    "ㅟ",
    "ㅠ",
    "ㅡ",
    "ㅢ",
    "ㅣ",
]

jongsung_list = [
    " ",
    "ㄱ",
    "ㄲ",
    "ㄳ",
    "ㄴ",
    "ㄵ",
    "ㄶ",
    "ㄷ",
    "ㄹ",
    "ㄺ",
    "ㄻ",
    "ㄼ",
    "ㄽ",
    "ㄾ",
    "ㄿ",
    "ㅀ",
    "ㅁ",
    "ㅂ",
    "ㅄ",
    "ㅅ",
    "ㅆ",
    "ㅇ",
    "ㅈ",
    "ㅊ",
    "ㅋ",
    "ㅌ",
    "ㅍ",
    "ㅎ",
]

jaum_list = [
    "ㄱ",
    "ㄲ",
    "ㄳ",
    "ㄴ",
    "ㄵ",
    "ㄶ",
    "ㄷ",
    "ㄸ",
    "ㄹ",
    "ㄺ",
    "ㄻ",
    "ㄼ",
    "ㄽ",
    "ㄾ",
    "ㄿ",
    "ㅀ",
    "ㅁ",
    "ㅂ",
    "ㅃ",
    "ㅄ",
    "ㅅ",
    "ㅆ",
    "ㅇ",
    "ㅈ",
    "ㅉ",
    "ㅊ",
    "ㅋ",
    "ㅌ",
    "ㅍ",
    "ㅎ",
]

moum_list = [
    "ㅏ",
    "ㅐ",
    "ㅑ",
    "ㅒ",
    "ㅓ",
    "ㅔ",
    "ㅕ",
    "ㅖ",
    "ㅗ",
    "ㅘ",
    "ㅙ",
    "ㅚ",
    "ㅛ",
    "ㅜ",
    "ㅝ",
    "ㅞ",
    "ㅟ",
    "ㅠ",
    "ㅡ",
    "ㅢ",
    "ㅣ",
]

_doublespace_pattern = re.compile(r"\s+")


def compose(chosung: str, jungsung: str, jongsung: str) -> str:
    return chr(
        _kor_begin
        + _chosung_base * chosung_list.index(chosung)
        + _jungsung_base * jungsung_list.index(jungsung)
        + jongsung_list.index(jongsung)
    )


def decompose(c: str) -> tuple[str, str, str] | None:
    if not character_is_korean(c):
        return None
    i = to_base(c)
    if _jaum_begin <= i <= _jaum_end:
        return (c, " ", " ")
    if _moum_begin <= i <= _moum_end:
        return (" ", c, " ")
    i -= _kor_begin
    cho = i // _chosung_base
    jung = (i - cho * _chosung_base) // _jungsung_base
    jong = i - cho * _chosung_base - jung * _jungsung_base
    return (chosung_list[cho], jungsung_list[jung], jongsung_list[jong])


def character_is_korean(c: str) -> bool:
    i = to_base(c)
    return (_kor_begin <= i <= _kor_end) or (_jaum_begin <= i <= _jaum_end) or (_moum_begin <= i <= _moum_end)


def character_is_complete_korean(c: str) -> bool:
    return _kor_begin <= to_base(c) <= _kor_end


def character_is_jaum(c: str) -> bool:
    return _jaum_begin <= to_base(c) <= _jaum_end


def character_is_moum(c: str) -> bool:
    return _moum_begin <= to_base(c) <= _moum_end


def to_base(c: str | int) -> int:
    if isinstance(c, (str, int)):
        return ord(c) if isinstance(c, str) else c
    raise TypeError(f"Expected str or int, got {type(c)}")


def character_is_number(c: str) -> bool:
    i = to_base(c)
    return 48 <= i <= 57


def character_is_english(c: str) -> bool:
    i = to_base(c)
    return (97 <= i <= 122) or (65 <= i <= 90)


def character_is_punctuation(c: str) -> bool:
    i = to_base(c)
    return i in (33, 34, 39, 44, 46, 63, 96)


def text_to_jamo(text: str, join_jongsung: bool = True) -> str:
    """한국어 텍스트를 자모 단위로 분해한 문자열로 변환한다.

    각 완성형 한글 음절은 (초성, 중성, 종성) 세 자모로 분해된다.
    종성이 없는 음절은 초성·중성 두 자모로만 구성된다 (기본값).
    비한글 문자는 그대로 유지된다.

    Args:
        text: 변환할 텍스트.
        join_jongsung: True이면 종성이 없을 때 자모를 2개만 출력한다 (기본값).
            False이면 종성 자리에 공백(' ')을 추가하여 항상 3자모로 출력한다.

    Returns:
        자모 단위로 분해된 문자열.

    Examples::
        >>> text_to_jamo("한글")
        'ㅎㅏㄴㄱㅡㄹ'

        >>> text_to_jamo("나는")
        'ㄴㅏㄴㅡㄴ'

        >>> text_to_jamo("abc한")
        'abcㅎㅏㄴ'

        >>> text_to_jamo("가나", join_jongsung=False)
        'ㄱㅏ ㄴㅏ '
    """
    chars: list[str] = []
    for c in text:
        result = decompose(c)
        if result is None:
            chars.append(c)
        else:
            cho, jung, jong = result
            if cho != " ":
                chars.append(cho)
            if jung != " ":
                chars.append(jung)
            if jong != " ":
                chars.append(jong)
            elif not join_jongsung:
                chars.append(" ")
    return "".join(chars)


class ConvolutionHangleEncoder:
    """Encode Korean characters into cho/jung/jong one-hot vectors.

    one hot vector [ㄱ, ㄴ, ㄷ, ... ㅎ, ㅏ, ㅐ, .. ㅢ, ㅣ," ", ㄱ, ㄲ, ... ㅍ, ㅎ," ", 0, 1, 2, .. 9]
    """

    def __init__(self) -> None:
        self.jung_begin = 19  # len(chosung_list)
        self.jong_begin = 40  # self.jung_begin + len(jungsung_list)
        self.number_begin = 68  # self.jong_begin + len(jongsung_list)
        self.space = 78  # len(chosung_list) + len(jungsung_list) + len(jongsung_list) + 10
        self.unk = 79
        self.dim = 80
        num = [str(i) for i in range(10)]
        space = " "
        unk = "<unk>"
        idx_to_char = chosung_list + jungsung_list + jongsung_list + num + [space] + [unk]
        self.idx_to_char = np.asarray(idx_to_char)
        # 초성(0~18) → 중성(19~39) → 종성(40~67) 순으로 인덱스를 할당한다.
        # 초성·중성과 겹치는 자모는 초성·중성 인덱스를 우선하고,
        # jongsung_list 중 겹자음(초성에 없는 것)만 종성 인덱스로 추가한다.
        self.jamo_to_idx: dict[str, int] = {}
        for i, c in enumerate(chosung_list):
            self.jamo_to_idx[c] = i
        for i, c in enumerate(jungsung_list):
            self.jamo_to_idx[c] = self.jung_begin + i
        for i, c in enumerate(jongsung_list):
            if c not in self.jamo_to_idx:  # 초성·중성에 없는 겹자음·공백만 추가
                self.jamo_to_idx[c] = self.jong_begin + i

    def encode(self, sent: str) -> np.ndarray:
        onehot = self.sent_to_onehot(sent)
        x = np.zeros((len(onehot), self.dim))
        for i, xi in enumerate(onehot):
            for j in xi:
                x[i, j] = 1
        return x

    def sent_to_onehot(self, sent: str) -> list[tuple[int, ...]]:
        chars = self._normalize(sent)
        ords = [ord(c) for c in chars]
        onehot: list[tuple[int, ...]] = []
        for char, idx in zip(chars, ords):
            if idx == 32:
                onehot.append((self.space,))
            elif 48 <= idx <= 57:
                onehot.append((idx - 48 + self.number_begin,))
            else:
                onehot.append(self._decompose(char, idx))
        return onehot

    def onehot_to_sent(self, encoded_sent: list[tuple[int, ...]]) -> str:
        def check_cjj(c: tuple[int, ...]) -> None:
            cho, jung, jong = c
            if not (0 <= cho < self.jung_begin):
                raise ValueError(f"Chosung {cho} is out of index")
            if not (self.jung_begin <= jung < self.jong_begin):
                raise ValueError(f"Jungsung {jung} is out of index")
            if not (self.jong_begin <= jong < self.number_begin):
                raise ValueError(f"Jongsung {jong} is out of index")

        chars: list[str] = []
        for c in encoded_sent:
            if len(c) == 1:
                if not 0 <= c[0] < self.dim:
                    raise ValueError(f"character index {c[0]} is out of index [0, {self.dim}]")
                chars.append(self.idx_to_char[c[0]])
            elif len(c) == 3:
                check_cjj(c)
                cho, jung, jong = tuple(self.idx_to_char[ci] for ci in c)
                chars.append(compose(cho, jung, jong))
            else:
                chars.append(self.idx_to_char[-1])
        return "".join(chars)

    def _normalize(self, sent: str) -> str:
        regex = re.compile(r"[^ㄱ-ㅎㅏ-ㅣ가-힣 0-9]")
        sent = regex.sub(" ", sent)
        sent = _doublespace_pattern.sub(" ", sent).strip()
        return sent

    def _compose(self, cho: int, jung: int, jong: int) -> str:
        return chr(_kor_begin + _chosung_base * cho + _jungsung_base * jung + jong)

    def _decompose(self, c: str, i: int) -> tuple[int, ...]:
        if _kor_begin <= i <= _kor_end:
            i -= _kor_begin
            cho = i // _chosung_base
            jung = (i - cho * _chosung_base) // _jungsung_base
            jong = i - cho * _chosung_base - jung * _jungsung_base
            return (cho, self.jung_begin + jung, self.jong_begin + jong)
        else:
            return (self.jamo_to_idx.get(c, self.unk),)
