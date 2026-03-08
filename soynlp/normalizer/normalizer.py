import logging
import os
import re
import unicodedata
from collections.abc import Callable
from glob import glob

from tqdm import tqdm

logger = logging.getLogger(__name__)


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
        self.pattern = re.compile(f"[^{pattern} ]")

    def normalize(self, s: str) -> str:
        return self.pattern.sub(" ", s).strip()


class HangleEmojiNormalizer(Normalizer):
    """Decompose hangle emoji patterns like 'ㅋㅋㅋ쿠ㅜㅜ' into 'ㅋㅋㅋㅋㅜㅜㅜ'"""

    def __init__(self):
        self.pattern = re.compile("[ㄱ-ㅎ]+[가-힣]{1}[ㅏ-ㅣ]+")
        self._hangle = re.compile("[가-힣]")

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
    def __init__(self):
        self.pattern = re.compile("[ ]{2,}")

    def normalize(self, s: str) -> str:
        return self.pattern.sub("  ", s)


class PaddingSpacetoWordsNormalizer(Normalizer):
    def __init__(self, custom_character: str | None = None):
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
