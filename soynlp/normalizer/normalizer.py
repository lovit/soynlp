import re
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
