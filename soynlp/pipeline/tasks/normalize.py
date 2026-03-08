from dataclasses import dataclass

from soynlp.normalizer import TextNormalizer
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
