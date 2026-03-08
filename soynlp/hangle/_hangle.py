import re

import numpy as np

kor_begin = 44032
kor_end = 55203
chosung_base = 588
jungsung_base = 28
jaum_begin = 12593
jaum_end = 12622
moum_begin = 12623
moum_end = 12643

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

doublespace_pattern = re.compile(r"\s+")


def compose(chosung: str, jungsung: str, jongsung: str) -> str:
    return chr(
        kor_begin
        + chosung_base * chosung_list.index(chosung)
        + jungsung_base * jungsung_list.index(jungsung)
        + jongsung_list.index(jongsung)
    )


def decompose(c: str) -> tuple[str, str, str] | None:
    if not character_is_korean(c):
        return None
    i = to_base(c)
    if jaum_begin <= i <= jaum_end:
        return (c, " ", " ")
    if moum_begin <= i <= moum_end:
        return (" ", c, " ")
    i -= kor_begin
    cho = i // chosung_base
    jung = (i - cho * chosung_base) // jungsung_base
    jong = i - cho * chosung_base - jung * jungsung_base
    return (chosung_list[cho], jungsung_list[jung], jongsung_list[jong])


def character_is_korean(c: str) -> bool:
    i = to_base(c)
    return (kor_begin <= i <= kor_end) or (jaum_begin <= i <= jaum_end) or (moum_begin <= i <= moum_end)


def character_is_complete_korean(c: str) -> bool:
    return kor_begin <= to_base(c) <= kor_end


def character_is_jaum(c: str) -> bool:
    return jaum_begin <= to_base(c) <= jaum_end


def character_is_moum(c: str) -> bool:
    return moum_begin <= to_base(c) <= moum_end


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
        self.jamo_to_idx: dict[str, int] = {
            "ㄱ": 0,
            "ㄲ": 1,
            "ㄴ": 2,
            "ㄷ": 3,
            "ㄸ": 4,
            "ㄹ": 5,
            "ㅁ": 6,
            "ㅂ": 7,
            "ㅃ": 8,
            "ㅅ": 9,
            "ㅆ": 10,
            "ㅇ": 11,
            "ㅈ": 12,
            "ㅉ": 13,
            "ㅊ": 14,
            "ㅋ": 15,
            "ㅌ": 16,
            "ㅍ": 17,
            "ㅎ": 18,
            "ㅏ": 19,
            "ㅐ": 20,
            "ㅑ": 21,
            "ㅒ": 22,
            "ㅓ": 23,
            "ㅔ": 24,
            "ㅕ": 25,
            "ㅖ": 26,
            "ㅗ": 27,
            "ㅘ": 28,
            "ㅙ": 29,
            "ㅚ": 30,
            "ㅛ": 31,
            "ㅜ": 32,
            "ㅝ": 33,
            "ㅞ": 34,
            "ㅟ": 35,
            "ㅠ": 36,
            "ㅡ": 37,
            "ㅢ": 38,
            "ㅣ": 39,
            " ": 40,
            "ㄳ": 43,
            "ㄵ": 45,
            "ㄶ": 46,
            "ㄺ": 49,
            "ㄻ": 50,
            "ㄼ": 51,
            "ㄽ": 52,
            "ㄾ": 53,
            "ㄿ": 54,
            "ㅀ": 55,
            "ㅄ": 58,
        }

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
        regex = re.compile("[^ㄱ-ㅎㅏ-ㅣ가-힣 0-9]")
        sent = regex.sub(" ", sent)
        sent = doublespace_pattern.sub(" ", sent).strip()
        return sent

    def _compose(self, cho: int, jung: int, jong: int) -> str:
        return chr(kor_begin + chosung_base * cho + jungsung_base * jung + jong)

    def _decompose(self, c: str, i: int) -> tuple[int, ...]:
        if kor_begin <= i <= kor_end:
            i -= kor_begin
            cho = i // chosung_base
            jung = (i - cho * chosung_base) // jungsung_base
            jong = i - cho * chosung_base - jung * jungsung_base
            return (cho, self.jung_begin + jung, self.jong_begin + jong)
        else:
            return (self.jamo_to_idx.get(c, self.unk),)
