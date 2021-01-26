import re
import unicodedata
from typing import Union


class Normalizer:
    def __call__(self, s: str) -> str:
        return self.normalize(s)

    def normalize(self, s: str) -> str:
        raise NotImplementedError("Implement `normalize` function")


class PassCharacterNormalizer(Normalizer):
    """
    Args:
        alphabet (bool)
        hangle (bool)
        number (bool)
        symbol (bool) :
            If True, it allows "(, ), ., ,, !, ?, -, /"
        custom(str, optional) : custom characters

    Example::
        >>> s = "  이것은 abc 123 ().,!?-/ 이 포함된 문장 @@ "
        >>> PassCharacterNormalizer(
        >>>     alphabet=True, hangle=False, number=False, symbol=False, custom=None)(s)
        $ 'abc'
        >>> PassCharacterNormalizer(
        >>>     alphabet=True, hangle=True, number=False, symbol=False, custom=None)(s)
        $ '이것은 abc              이 포함된 문장'
        >>> PassCharacterNormalizer(
        >>>     alphabet=True, hangle=True, number=True, symbol=False, custom=None)(s)
        $ '이것은 abc 123          이 포함된 문장'
        >>> PassCharacterNormalizer(
        >>>     alphabet=True, hangle=True, number=True, symbol=True, custom=None)(s)
        $ '이것은 abc 123 ().,!?-/ 이 포함된 문장'
        >>> PassCharacterNormalizer(
        >>>     alphabet=True, hangle=True, number=True, symbol=True, custom="@")(s)
        $ '이것은 abc 123 ().,!?-/ 이 포함된 문장 @@'
    """

    def __init__(
        self,
        alphabet: bool = True,
        hangle: bool = True,
        number: bool = True,
        symbol: bool = True,
        custom: Union[None, str] = None,
    ):
        pattern = ""
        if alphabet:
            pattern += "a-zA-Z"
        if hangle:
            pattern += "가-힣ㄱ-ㅎㅏ-ㅣ"
        if number:
            pattern += "0-9"
        if symbol:
            pattern += "\(\)\.,?!-/"
        if isinstance(custom, str):
            pattern += custom
        self.pattern = re.compile(f"[^{pattern} ]")

    def normalize(self, s: str) -> str:
        return self.pattern.sub(" ", s).strip()


class HangleEmojiNormalizer(Normalizer):
    """
    Example:
        >>> s = "어머나 ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ쿠ㅜㅜㅜㅜㅜ이런게 있으면 어떻게 떼어내냐 ㅋㅋㅋㅋㅋ쿠ㅜㅜㅜㅜㅜ 하하"
        >>> hangle_emoji = HangleEmojiNormalizer()
        >>> hangle_emoji(s)
        $ '어머나 ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅜㅜㅜㅜㅜㅜ이런게 있으면 어떻게 떼어내냐 ㅋㅋㅋㅋㅋㅋㅜㅜㅜㅜㅜㅜ 하하'

        >>> repeat_character = RepeatCharacterNormalize()
        >>> repeat_character(hangle_emoji(s))
        $ '어머나 ㅋㅋㅜㅜ이런게 있으면 어떻게 떼어내냐 ㅋㅋㅜㅜ 하하'
    """
    def __init__(self):
        self.pattern = re.compile('[ㄱ-ㅎ]+[가-힣]{1}[ㅏ-ㅣ]+')
        self._hangle = re.compile('[가-힣]')

    def normalize(self, s: str) -> str:
        def decompose(target):
            i = list(self._hangle.finditer(target))[0].span()[0]
            hangle = unicodedata.normalize("NFKD", target[i])
            jaum, moum = target[i-1], target[i+1]
            jaum_ = unicodedata.normalize("NFKD", jaum)
            moum_ = unicodedata.normalize("NFKD", moum)
            if (jaum_ == hangle[0]) and (hangle[-1] == moum_):
                return target[:i] + jaum + moum + target[i+1:]
            return target

        s_ = []
        offset = 0
        for m in self.pattern.finditer(s):
            begin, end = m.span()
            s_.append(s[offset:begin])
            target = s[begin:end]
            s_.append(decompose(target))
            offset = end
        s_.append(s[offset:])
        return ''.join(s_)


class RepeatCharacterNormalize(Normalizer):
    """
    Args:
        max_repeat (int)
    Examples:
        >>> RepeatCharacterNormalize()("ㅇㅇㅇㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ")
        $ 'ㅇㅇㅋㅋ'
        >>> RepeatCharacterNormalize(max_repeat=3)("ㅇㅇㅇㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ")
        $ 'ㅇㅇㅇㅋㅋㅋ'
    """
    def __init__(self, max_repeat: int = 2):
        pattern = "(\w)\\1{" + str(max_repeat) + ",}"
        self.pattern = re.compile(pattern)
        self.replace_str = "\\1" * max_repeat

    def normalize(self, s: str) -> str:
        return self.pattern.sub(self.replace_str, s)


class RemoveLongspaceNormalizer(Normalizer):
    """
    Example:
        >>> RemoveLongspaceNormalizer()("ab     cd    d  f ")
        $ 'ab  cd  d  f '
    """
    def __init__(self):
        self.pattern = re.compile('[ ]{2,}')

    def normalize(self, s: str) -> str:
        return self.pattern.sub("  ", s)


class PaddingSpacetoWordsNormalizer(Normalizer):
    """
    Example:
        >>> PaddingSpacetoWordsNormalizer()("(주)일이삼 [[공지]]제목 이것은예시다!!")
        $ '( 주 ) 일이삼  [[ 공지 ]] 제목   이것은예시다 !!'
    """
    def __init__(self, custom_character: str = None):
        pattern = "a-zA-Z가-힣ㄱ-ㅎㅏ-ㅣ0-9"
        if isinstance(custom_character, str):
            pattern += custom_character
        self.pattern = re.compile(f"[{pattern}]+")

    def normalize(self, s: str) -> str:
        s_ = []
        offset = 0
        for m in self.pattern.finditer(s):
            begin, end = m.span()
            s_.append(s[offset:begin])
            s_.append(f" {s[begin:end]} ")
            offset = end
        s_.append(s[offset:])
        return ''.join(s_)
