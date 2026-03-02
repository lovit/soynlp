"""Rule-based Korean named entity recognition (skeleton)."""


def is_date(word: str) -> bool:
    raise NotImplementedError


def is_day(word: str) -> bool:
    raise NotImplementedError


def is_number(word: str) -> bool:
    raise NotImplementedError


def is_monetary(word: str) -> bool:
    raise NotImplementedError


def is_period(word: str) -> bool:
    raise NotImplementedError


def is_time(word: str) -> bool:
    raise NotImplementedError
