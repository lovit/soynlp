import re

from soynlp.hangle import compose, decompose

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
    return _doublespace_pattern.sub(" ", sent)


def repeat_normalize(sent: str, num_repeats: int = 2) -> str:
    if num_repeats > 0:
        sent = _repeatchars_pattern.sub("\\1" * num_repeats, sent)
    sent = _doublespace_pattern.sub(" ", sent)
    return sent.strip()


def emoticon_normalize(sent: str, num_repeats: int = 2) -> str:
    if not sent:
        return sent

    def _char_type(idx: int) -> int:
        if 12593 <= idx <= 12622:
            return 0  # Jaum
        elif 12623 <= idx <= 12643:
            return 1  # Moum
        elif 44032 <= idx <= 55203:
            return 2  # Complete
        return -1

    idxs = [_char_type(ord(c)) for c in sent]
    sent_ = []
    last_idx = len(idxs) - 1
    for i, (idx, c) in enumerate(zip(idxs, sent)):
        if (0 < i < last_idx) and (idxs[i - 1] == 0 and idx == 2 and idxs[i + 1] == 1):
            cho, jung, jong = decompose(c)  # type: ignore[misc]
            if (cho == sent[i - 1]) and (jung == sent[i + 1]) and (jong == " "):
                sent_.append(cho)
                sent_.append(jung)
            else:
                sent_.append(c)
        elif (i < last_idx) and (idx == 2) and (idxs[i + 1] == 0):
            cho, jung, jong = decompose(c)  # type: ignore[misc]
            if jong == sent[i + 1]:
                sent_.append(compose(cho, jung, " "))
                sent_.append(jong)
        elif (i > 0) and (idx == 2 and idxs[i - 1] == 0):
            cho, jung, jong = decompose(c)  # type: ignore[misc]
            if cho == sent[i - 1]:
                sent_.append(cho)
                sent_.append(jung)
        else:
            sent_.append(c)
    return repeat_normalize("".join(sent_), num_repeats)


def only_hangle(sent: str) -> str:
    return _doublespace_pattern.sub(" ", _hangle_filter.sub(" ", sent)).strip()


def only_hangle_number(sent: str) -> str:
    return _doublespace_pattern.sub(" ", _hangle_number_filter.sub(" ", sent)).strip()


def only_text(sent: str) -> str:
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
