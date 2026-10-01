"""Explicit legacy options return the original token layouts."""

from typing import Any, Literal, overload

from soynlp._legacy.tokenizer import LTokenizer as LegacyLTokenizer
from soynlp._legacy.tokenizer import MaxScoreTokenizer as LegacyMaxScoreTokenizer
from soynlp._legacy.tokenizer import NounMatchTokenizer as LegacyNounMatchTokenizer
from soynlp._legacy.tokenizer import RegexTokenizer as LegacyRegexTokenizer

from .tokenizer import LTokenizer as CorpusLTokenizer
from .tokenizer import MaxScoreTokenizer as CorpusMaxScoreTokenizer
from .tokenizer import NounMatchTokenizer as CorpusNounMatchTokenizer
from .tokenizer import RegexTokenizer as CorpusRegexTokenizer
from .tokenizer import Token


class RegexTokenizer(CorpusRegexTokenizer):
    @overload
    def tokenize(self, sentence: str, *, return_words: Literal[True] = ...) -> list[str]: ...

    @overload
    def tokenize(self, sentence: str, *, return_words: Literal[False]) -> list[Token]: ...

    @overload
    def tokenize(self, sentence: str, *args: Any, **kwargs: Any) -> Any: ...

    def tokenize(self, sentence, *args, return_words=True, debug=None, flatten=None) -> Any:
        if args:
            return LegacyRegexTokenizer().tokenize(sentence, *args, **({"flatten": flatten} if flatten is not None else {}))
        if debug is not None or flatten is not None:
            return LegacyRegexTokenizer().tokenize(sentence, debug=bool(debug), flatten=True if flatten is None else flatten)
        return super().tokenize(sentence, return_words)

    @overload
    def __call__(self, sentence: str, *, return_words: Literal[True] = ...) -> list[str]: ...

    @overload
    def __call__(self, sentence: str, *, return_words: Literal[False]) -> list[Token]: ...

    @overload
    def __call__(self, sentence: str, *args: Any, **kwargs: Any) -> Any: ...

    def __call__(self, sentence, *args, **kwargs) -> Any:
        return self.tokenize(sentence, *args, **kwargs)


class LTokenizer(CorpusLTokenizer):
    def __init__(self, scores=None, unknown_score=0.0, *, default_score=None):
        super().__init__(scores or {}, unknown_score if default_score is None else default_score)

    @overload
    def tokenize(
        self, sentence: str, tolerance: float = ..., *, return_words: Literal[True] = ..., remove_r: bool = ...
    ) -> list[str]: ...

    @overload
    def tokenize(
        self, sentence: str, tolerance: float = ..., *, return_words: Literal[False] = ..., remove_r: bool = ...
    ) -> list[Token]: ...

    @overload
    def tokenize(self, sentence: str, *args: Any, **kwargs: Any) -> Any: ...

    def tokenize(self, sentence, *args, **kwargs) -> Any:
        if "flatten" in kwargs or len(args) >= 2:
            return LegacyLTokenizer(self.scores, self.unknown_score).tokenize(sentence, *args, **kwargs)
        return super().tokenize(sentence, *args, **kwargs)

    @overload
    def __call__(
        self, sentence: str, tolerance: float = ..., *, return_words: Literal[True] = ..., remove_r: bool = ...
    ) -> list[str]: ...

    @overload
    def __call__(
        self, sentence: str, tolerance: float = ..., *, return_words: Literal[False] = ..., remove_r: bool = ...
    ) -> list[Token]: ...

    @overload
    def __call__(self, sentence: str, *args: Any, **kwargs: Any) -> Any: ...

    def __call__(self, sentence, *args, **kwargs) -> Any:
        return self.tokenize(sentence, *args, **kwargs)


class MaxScoreTokenizer(CorpusMaxScoreTokenizer):
    def __init__(self, scores=None, max_length=10, unknown_score=0.0, *, default_score=None):
        super().__init__(scores or {}, max_length, unknown_score if default_score is None else default_score)

    @overload
    def tokenize(self, sentence: str, *, return_words: Literal[True] = ...) -> list[str]: ...

    @overload
    def tokenize(self, sentence: str, *, return_words: Literal[False]) -> list[Token]: ...

    @overload
    def tokenize(self, sentence: str, *args: Any, **kwargs: Any) -> Any: ...

    def tokenize(self, sentence, *args, return_words=True, flatten=None) -> Any:
        if args or flatten is not None:
            return LegacyMaxScoreTokenizer(self.scores, self.max_len, self.unknown_score).tokenize(
                sentence, *args, **({"flatten": flatten} if flatten is not None else {})
            )
        return super().tokenize(sentence, return_words)

    @overload
    def __call__(self, sentence: str, *, return_words: Literal[True] = ...) -> list[str]: ...

    @overload
    def __call__(self, sentence: str, *, return_words: Literal[False]) -> list[Token]: ...

    @overload
    def __call__(self, sentence: str, *args: Any, **kwargs: Any) -> Any: ...

    def __call__(self, sentence, *args, **kwargs) -> Any:
        return self.tokenize(sentence, *args, **kwargs)


class NounMatchTokenizer(CorpusNounMatchTokenizer):
    @overload
    def tokenize(
        self, sentence: str, *, return_words: Literal[True] = ..., concat_compound: bool = ..., must_be_L: bool = ...
    ) -> list[str]: ...

    @overload
    def tokenize(
        self, sentence: str, *, return_words: Literal[False], concat_compound: bool = ..., must_be_L: bool = ...
    ) -> list[Token]: ...

    @overload
    def tokenize(self, sentence: str, *args: Any, **kwargs: Any) -> Any: ...

    def tokenize(
        self, sentence, *args, return_words=True, concat_compound=True, must_be_L=False, flatten=None, compose_compound=None
    ) -> Any:
        if args:
            return LegacyNounMatchTokenizer(self.scores).tokenize(sentence, *args)
        if flatten is not None or compose_compound is not None:
            return LegacyNounMatchTokenizer(self.scores).tokenize(
                sentence,
                flatten=True if flatten is None else flatten,
                compose_compound=concat_compound if compose_compound is None else compose_compound,
            )
        return super().tokenize(sentence, return_words, concat_compound, must_be_L)

    @overload
    def __call__(self, sentence: str, *, return_words: Literal[True] = ..., concat_compound: bool = ...) -> list[str]: ...

    @overload
    def __call__(self, sentence: str, *, return_words: Literal[False], concat_compound: bool = ...) -> list[Token]: ...

    @overload
    def __call__(self, sentence: str, *args: Any, **kwargs: Any) -> Any: ...

    def __call__(self, sentence, *args, **kwargs) -> Any:
        return self.tokenize(sentence, *args, **kwargs)
