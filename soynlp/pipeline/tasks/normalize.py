from dataclasses import dataclass

from soynlp.normalizer import (
    EmojiNormalizer,
    HangleEmojiNormalizer,
    PaddingSpacetoWordsNormalizer,
    PassCharacterNormalizer,
    RemoveLongspaceNormalizer,
    RepeatCharacterNormalizer,
    TextNormalizer,
)
from soynlp.pipeline.tasks.task import Task, TaskArgs


@dataclass(slots=True)
class NormalizeTaskArgs(TaskArgs):
    alphabet: bool = True
    hangle: bool = True
    number: bool = True
    symbol: bool = True
    custom: str | None = None
    decompose_hangle_emoji: bool = True
    remove_repeatchar: int = 2
    remove_longspace: bool = True
    in_key: str = "corpus"
    text_key: str = "text"
    out_key: str = "corpus"


class NormalizeTask(Task[NormalizeTaskArgs]):
    def __call__(self, parameters: dict) -> dict:
        if self._args.in_key not in parameters:
            raise ValueError(f"Not found `{self._args.in_key}` in `parameters`")

        normalizer = TextNormalizer.build_normalizer(
            alphabet=self._args.alphabet,
            hangle=self._args.hangle,
            number=self._args.number,
            symbol=self._args.symbol,
            custom=self._args.custom,
            decompose_hangle_emoji=self._args.decompose_hangle_emoji,
            remove_repeatchar=self._args.remove_repeatchar,
            remove_longspace=self._args.remove_longspace,
        )

        examples = parameters[self._args.in_key]
        normalized = []
        for example in examples:
            text = example[self._args.text_key]
            normed = normalizer(text)
            normalized.append({**example, self._args.text_key: normed})

        parameters[self._args.out_key] = normalized
        return parameters


def _apply_normalizer(normalizer, parameters: dict, in_key: str, text_key: str, out_key: str) -> dict:
    if in_key not in parameters:
        raise ValueError(f"Not found `{in_key}` in `parameters`")
    examples = parameters[in_key]
    normalized = [{**ex, text_key: normalizer(ex[text_key])} for ex in examples]
    parameters[out_key] = normalized
    return parameters


@dataclass(slots=True)
class PassCharacterNormalizeTaskArgs(TaskArgs):
    alphabet: bool = True
    hangle: bool = True
    number: bool = True
    symbol: bool = True
    custom: str | None = None
    in_key: str = "corpus"
    text_key: str = "text"
    out_key: str = "corpus"


class PassCharacterNormalizeTask(Task[PassCharacterNormalizeTaskArgs]):
    def __call__(self, parameters: dict) -> dict:
        normalizer = PassCharacterNormalizer(
            alphabet=self._args.alphabet,
            hangle=self._args.hangle,
            number=self._args.number,
            symbol=self._args.symbol,
            custom=self._args.custom,
        )
        return _apply_normalizer(normalizer, parameters, self._args.in_key, self._args.text_key, self._args.out_key)


@dataclass(slots=True)
class HangleEmojiNormalizeTaskArgs(TaskArgs):
    in_key: str = "corpus"
    text_key: str = "text"
    out_key: str = "corpus"


class HangleEmojiNormalizeTask(Task[HangleEmojiNormalizeTaskArgs]):
    def __call__(self, parameters: dict) -> dict:
        return _apply_normalizer(
            HangleEmojiNormalizer(), parameters, self._args.in_key, self._args.text_key, self._args.out_key
        )


@dataclass(slots=True)
class EmojiNormalizeTaskArgs(TaskArgs):
    replace: str = ""
    in_key: str = "corpus"
    text_key: str = "text"
    out_key: str = "corpus"


class EmojiNormalizeTask(Task[EmojiNormalizeTaskArgs]):
    def __call__(self, parameters: dict) -> dict:
        return _apply_normalizer(
            EmojiNormalizer(replace=self._args.replace),
            parameters,
            self._args.in_key,
            self._args.text_key,
            self._args.out_key,
        )


@dataclass(slots=True)
class RepeatCharacterNormalizeTaskArgs(TaskArgs):
    max_repeat: int = 2
    in_key: str = "corpus"
    text_key: str = "text"
    out_key: str = "corpus"


class RepeatCharacterNormalizeTask(Task[RepeatCharacterNormalizeTaskArgs]):
    def __call__(self, parameters: dict) -> dict:
        return _apply_normalizer(
            RepeatCharacterNormalizer(max_repeat=self._args.max_repeat),
            parameters,
            self._args.in_key,
            self._args.text_key,
            self._args.out_key,
        )


@dataclass(slots=True)
class RemoveLongspaceNormalizeTaskArgs(TaskArgs):
    in_key: str = "corpus"
    text_key: str = "text"
    out_key: str = "corpus"


class RemoveLongspaceNormalizeTask(Task[RemoveLongspaceNormalizeTaskArgs]):
    def __call__(self, parameters: dict) -> dict:
        return _apply_normalizer(
            RemoveLongspaceNormalizer(), parameters, self._args.in_key, self._args.text_key, self._args.out_key
        )


@dataclass(slots=True)
class PaddingSpaceNormalizeTaskArgs(TaskArgs):
    custom_character: str | None = None
    in_key: str = "corpus"
    text_key: str = "text"
    out_key: str = "corpus"


class PaddingSpaceNormalizeTask(Task[PaddingSpaceNormalizeTaskArgs]):
    def __call__(self, parameters: dict) -> dict:
        return _apply_normalizer(
            PaddingSpacetoWordsNormalizer(custom_character=self._args.custom_character),
            parameters,
            self._args.in_key,
            self._args.text_key,
            self._args.out_key,
        )
