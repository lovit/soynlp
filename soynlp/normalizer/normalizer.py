import logging
import os
import re
import unicodedata
import warnings
from collections.abc import Callable
from glob import glob

from tqdm import tqdm

logger = logging.getLogger(__name__)

_doublespace_pattern = re.compile(r"\s+")
_repeatchars_pattern = re.compile(r"(\w)\1{2,}")
_number_pattern = re.compile(r"[0-9]")
_punctuation_pattern = re.compile(r"[,.?!]")
_symbol_pattern = re.compile(r"[()\[\]{}`]")
_hangle_pattern = re.compile(r"[ㄱ-ㅎㅏ-ㅣ가-힣]")
_alphabet_pattern = re.compile(r"[a-zA-Z]")

_hangle_filter = re.compile(r"[^ㄱ-ㅎㅏ-ㅣ가-힣]")
_hangle_number_filter = re.compile(r"[^ㄱ-ㅎㅏ-ㅣ가-힣0-9]")
_text_filter = re.compile(r"[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9,.?!\"'\-()\[\]{}]")


def normalize(
    doc: str,
    alphabet: bool = False,
    number: bool = False,
    punctuation: bool = False,
    symbol: bool = False,
    remove_repeat: int = 0,
) -> str:
    """.. deprecated:: Use ``TextNormalizer.build_normalizer()`` instead."""
    warnings.warn(
        "`normalize` is deprecated. Use `TextNormalizer.build_normalizer()` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    doc = _text_filter.sub(" ", doc)
    if not alphabet:
        doc = _alphabet_pattern.sub(" ", doc)
    if not number:
        doc = _number_pattern.sub(" ", doc)
    if not punctuation:
        doc = _punctuation_pattern.sub(" ", doc)
    if not symbol:
        doc = _symbol_pattern.sub(" ", doc)
    if remove_repeat > 0:
        doc = _repeatchars_pattern.sub("\\1" * remove_repeat, doc)
    return _doublespace_pattern.sub(" ", doc).strip()


def remove_doublespace(sent: str) -> str:
    """.. deprecated:: Use ``RemoveLongspaceNormalizer`` instead."""
    warnings.warn(
        "`remove_doublespace` is deprecated. Use `RemoveLongspaceNormalizer` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _doublespace_pattern.sub(" ", sent)


def repeat_normalize(sent: str, num_repeats: int = 2) -> str:
    """.. deprecated:: Use ``RepeatCharacterNormalizer`` instead."""
    warnings.warn(
        "`repeat_normalize` is deprecated. Use `RepeatCharacterNormalizer` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if num_repeats > 0:
        sent = _repeatchars_pattern.sub("\\1" * num_repeats, sent)
    sent = _doublespace_pattern.sub(" ", sent)
    return sent.strip()


def emoticon_normalize(sent: str, num_repeats: int = 2) -> str:
    """한국어 자모-음절 혼합 이모티콘을 정규화한다.

    .. deprecated::
        `emoticon_normalize`는 deprecated입니다.
        `HangleEmojiNormalizer`와 `RepeatCharacterNormalizer`를 조합하여 사용하세요.

        기존 구현은 완성형 음절 뒤에 자음이 오면서 종성 조건이 불일치할 때
        해당 음절을 묵소 삭제하는 버그가 있었습니다. (예: 'ㅋ크ㅋ' → 'ㅋㅋ')
        `HangleEmojiNormalizer`는 이 케이스를 올바르게 처리합니다.
    """
    warnings.warn(
        "`emoticon_normalize` is deprecated. Use `HangleEmojiNormalizer` and `RepeatCharacterNormalizer` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    normalized = HangleEmojiNormalizer().normalize(sent)
    if num_repeats > 0:
        normalized = RepeatCharacterNormalizer(max_repeat=num_repeats).normalize(normalized)
    normalized = _doublespace_pattern.sub(" ", normalized)
    return normalized.strip()


def only_hangle(sent: str) -> str:
    """.. deprecated:: Use ``PassCharacterNormalizer(alphabet=False, number=False, symbol=False)`` instead."""
    warnings.warn(
        "`only_hangle` is deprecated. Use `PassCharacterNormalizer(alphabet=False, number=False, symbol=False)` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _doublespace_pattern.sub(" ", _hangle_filter.sub(" ", sent)).strip()


def only_hangle_number(sent: str) -> str:
    """.. deprecated:: Use ``PassCharacterNormalizer(alphabet=False, symbol=False)`` instead."""
    warnings.warn(
        "`only_hangle_number` is deprecated. Use `PassCharacterNormalizer(alphabet=False, symbol=False)` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _doublespace_pattern.sub(" ", _hangle_number_filter.sub(" ", sent)).strip()


def only_text(sent: str) -> str:
    """.. deprecated:: Use ``PassCharacterNormalizer()`` instead."""
    warnings.warn(
        "`only_text` is deprecated. Use `PassCharacterNormalizer()` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _doublespace_pattern.sub(" ", _text_filter.sub(" ", sent)).strip()


def remain_hangle_on_last(eojeol: str) -> str:
    matches = list(_hangle_pattern.finditer(eojeol))
    if not matches:
        return ""
    last_index = matches[-1].span()[1]
    return eojeol[:last_index].strip()


def normalize_sent_for_lrgraph(sent: str) -> str:
    sent = _text_filter.sub(" ", sent)
    sent = _symbol_pattern.sub(" ", sent)
    sent_ = [remain_hangle_on_last(eojeol) for eojeol in sent.split()]
    sent_ = [eojeol for eojeol in sent_ if eojeol]
    if not sent_:
        return ""
    return " ".join(sent_)


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
            If True, it allows "(, ), ., ,, !, ?, -, /, [, ]"
        custom(str, optional) : custom characters
    """

    def __init__(
        self,
        alphabet: bool = True,
        hangle: bool = True,
        number: bool = True,
        symbol: bool = True,
        custom: str | None = None,
    ):
        pattern = ""
        if alphabet:
            pattern += "a-zA-Z"
        if hangle:
            pattern += "가-힣ㄱ-ㅎㅏ-ㅣ"
        if number:
            pattern += "0-9"
        if symbol:
            pattern += r"\(\)\.,?!-/\[\]"
        if isinstance(custom, str):
            pattern += custom
        self.pattern = re.compile(rf"[^{pattern} ]")

    def normalize(self, s: str) -> str:
        return self.pattern.sub(" ", s).strip()


class HangleEmojiNormalizer(Normalizer):
    """Decompose hangle emoji patterns like 'ㅋㅋㅋ쿠ㅜㅜ' into 'ㅋㅋㅋㅋㅜㅜㅜ'"""

    def __init__(self):
        self.pattern = re.compile(r"[ㄱ-ㅎ]+[가-힣]{1}[ㅏ-ㅣ]+")
        self._hangle = re.compile(r"[가-힣]")

    def normalize(self, s: str) -> str:
        def decompose(target):
            i = list(self._hangle.finditer(target))[0].span()[0]
            hangle = unicodedata.normalize("NFKD", target[i])
            jaum, moum = target[i - 1], target[i + 1]
            jaum_ = unicodedata.normalize("NFKD", jaum)
            moum_ = unicodedata.normalize("NFKD", moum)
            if (jaum_ == hangle[0]) and (hangle[-1] == moum_):
                return target[:i] + jaum + moum + target[i + 1 :]
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
        return "".join(s_)


class EmojiNormalizer(Normalizer):
    """Unicode 이모지를 제거하거나 지정 문자열로 치환한다.

    단일 코드포인트 이모지(😀)뿐 아니라 ZWJ 시퀀스(👨‍💻), Skin tone modifier(👋🏽),
    Regional indicator(🇰🇷) 등 복합 이모지 시퀀스도 올바르게 처리한다.
    내부적으로 `emoji` 패키지를 사용하며, 미설치 시 ImportError가 발생한다.

    한국어 자모 이모티콘(ㅋㅋㅋ, ㅎㅎㅎ)은 Unicode 이모지가 아니므로 이 클래스의
    처리 대상이 아니다. 해당 패턴은 HangleEmojiNormalizer와 RepeatCharacterNormalizer로 처리한다.

    Args:
        replace (str): 이모지를 치환할 문자열. 기본값 "" (제거).
            "[EMOJI]"로 설정하면 이모지 위치를 토큰으로 유지할 수 있다.

    Examples:
        >>> normalizer = EmojiNormalizer()
        >>> normalizer.normalize("안녕 😀 반가워 🎉")
        '안녕  반가워 '
        >>> EmojiNormalizer(replace="[EMOJI]").normalize("안녕 😀")
        '안녕 [EMOJI]'
        >>> EmojiNormalizer().normalize("👨‍💻 코딩 중")  # ZWJ 시퀀스
        ' 코딩 중'
        >>> EmojiNormalizer().normalize("🇰🇷 한국")  # Regional indicator
        ' 한국'

    Raises:
        ImportError: `emoji` 패키지가 설치되지 않은 경우.
            `pip install soynlp[emoji]` 또는 `pip install emoji` 로 설치.
    """

    def __init__(self, replace: str = "") -> None:
        try:
            import emoji as _emoji_module  # type: ignore[import-untyped]

            self._emoji = _emoji_module
        except ImportError as e:
            raise ImportError(
                "EmojiNormalizer requires the `emoji` package. Install it with: pip install soynlp[emoji]"
            ) from e
        self.replace = replace

    def normalize(self, s: str) -> str:
        return self._emoji.replace_emoji(s, replace=self.replace)


class RepeatCharacterNormalizer(Normalizer):
    """
    Args:
        max_repeat (int)
    """

    def __init__(self, max_repeat: int = 2):
        pattern = "(\\S)\\1{" + str(max_repeat) + ",}"
        self.pattern = re.compile(pattern)
        self.replace_str = "\\1" * max_repeat

    def normalize(self, s: str) -> str:
        return self.pattern.sub(self.replace_str, s)


class RemoveLongspaceNormalizer(Normalizer):
    """2개 이상의 공백(탭·개행 포함)을 단일 공백으로 줄인다."""

    def __init__(self):
        self.pattern = re.compile(r"\s+")

    def normalize(self, s: str) -> str:
        return self.pattern.sub(" ", s)


class PaddingSpacetoWordsNormalizer(Normalizer):
    def __init__(self, custom_character: str | None = None):
        pattern = "a-zA-Z가-힣ㄱ-ㅎㅏ-ㅣ0-9"
        if isinstance(custom_character, str):
            pattern += custom_character
        self.pattern = re.compile(rf"[{pattern}]+")

    def normalize(self, s: str) -> str:
        s_ = []
        offset = 0
        for m in self.pattern.finditer(s):
            begin, end = m.span()
            s_.append(s[offset:begin])
            s_.append(f" {s[begin:end]} ")
            offset = end
        s_.append(s[offset:])
        return "".join(s_)


class TextNormalizer(Normalizer):
    def __init__(self, normalizer_list):
        if not isinstance(normalizer_list, list):
            raise ValueError("Available only `list` as `normalizer_list`")
        for i, module in enumerate(normalizer_list):
            if not callable(module):
                raise ValueError(f"{i}th module is not callable")
        self.modules = normalizer_list

    def normalize(self, s: str) -> str:
        for module in self.modules:
            s = module(s)
        return s

    @classmethod
    def build_normalizer(
        cls,
        alphabet: bool = True,
        hangle: bool = True,
        number: bool = True,
        symbol: bool = True,
        custom: str | None = None,
        decompose_hangle_emoji: bool = True,
        remove_repeatchar: int = 2,
        remove_longspace: bool = True,
        padding_space: bool = False,
        custom_normalizers: Callable[[str], str] | list[Callable[[str], str]] | None = None,
    ) -> Callable[[str], str]:
        modules = []
        if alphabet or hangle or number or symbol or isinstance(custom, str):
            modules.append(
                PassCharacterNormalizer(
                    alphabet=alphabet,
                    hangle=hangle,
                    number=number,
                    symbol=symbol,
                    custom=custom,
                )
            )
        if padding_space:
            modules.append(PaddingSpacetoWordsNormalizer())
        if decompose_hangle_emoji:
            modules.append(HangleEmojiNormalizer())
        if callable(custom_normalizers) and not isinstance(custom_normalizers, list):
            custom_normalizers = [custom_normalizers]
        if isinstance(custom_normalizers, list):
            for module in custom_normalizers:
                if not callable(module):
                    raise ValueError("Module in `custom_normalizer` must be callable")
                s = module("test")
                if not isinstance(s, str):
                    raise ValueError("Module in `custom_normalizer` must return `str`")
            modules += custom_normalizers
        if remove_repeatchar > 0:
            modules.append(RepeatCharacterNormalizer(max_repeat=remove_repeatchar))
        if remove_longspace:
            modules.append(RemoveLongspaceNormalizer())
        if not modules:
            raise ValueError("Empty components. Check normalizer builder arguments")
        return TextNormalizer(modules)


def task_normalize(
    input: str | list[str],
    output: str | list[str],
    verbose: bool = True,
    force: bool = False,
    debug: bool = False,
    alphabet: bool = True,
    hangle: bool = True,
    number: bool = True,
    symbol: bool = True,
    custom: str | None = None,
    decompose_hangle_emoji: bool = True,
    remove_repeatchar: int = 2,
    remove_longspace: bool = True,
):
    """.. deprecated:: Use ``ReadTextTask`` + ``NormalizeTask`` + ``WriteTextTask`` pipeline 조합을 사용하세요."""
    warnings.warn(
        "`task_normalize` is deprecated. Use `ReadTextTask` + `NormalizeTask` + `WriteTextTask` pipeline instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    task_normalizer = TextNormalizer.build_normalizer(
        alphabet=alphabet,
        hangle=hangle,
        number=number,
        symbol=symbol,
        custom=custom,
        decompose_hangle_emoji=decompose_hangle_emoji,
        remove_repeatchar=remove_repeatchar,
        remove_longspace=remove_longspace,
    )

    if isinstance(input, list) and len(input) == 1:
        input = input[0]
    if isinstance(output, list) and len(output) == 1:
        output = output[0]

    if isinstance(input, list) and isinstance(output, list):
        if len(input) != len(output):
            raise ValueError("The length of `input` and `output` must be same")
    elif isinstance(input, str) and isinstance(output, str):
        if os.path.isdir(input):
            input = sorted([inp for inp in glob(f"{input}/*") if os.path.isfile(inp)])
            output = [f"{output}/{os.path.basename(inp)}" for inp in input]
        else:
            input = [input]
            output = [output]
    elif isinstance(input, list) and isinstance(output, str):
        input = [inp for inp in input if os.path.isfile(inp)]
        output = [f"{output}/{os.path.basename(inp)}" for inp in input]

    assert len(input) == len(output)

    if verbose:
        file_iterator = tqdm(zip(input, output), desc="Task normalize", total=len(input))
    else:
        file_iterator = zip(input, output)

    n_lines, n_exceptions = 0, 0
    for inp, outp in file_iterator:
        if not os.path.exists(inp):
            continue
        if os.path.exists(outp) and (not force):
            raise ValueError(f"Already exist {outp}. Set `force==True` or `--force`")
        basename = os.path.basename(inp)
        os.makedirs(os.path.dirname(os.path.abspath(outp)), exist_ok=True)
        with open(inp, encoding="utf-8") as fi:
            with open(outp, "w", encoding="utf-8") as fo:
                if verbose:
                    line_iterator = tqdm(fi, desc=f"Normalize {basename}", leave=False)
                else:
                    line_iterator = fi
                for i, line in enumerate(line_iterator):
                    n_lines += 1
                    try:
                        normed = task_normalizer(line.strip())
                        fo.write(f"{normed}\n")
                    except Exception as err:
                        logger.debug(f"Exception {err}\n@{basename} LN{i}: {line}")
                        fo.write(f"{line.strip()}\n")
                        n_exceptions += 1
                        continue
    logger.info(f"Found {n_exceptions} from {n_lines}")


text_normalizer = TextNormalizer.build_normalizer()  # default normalizer
