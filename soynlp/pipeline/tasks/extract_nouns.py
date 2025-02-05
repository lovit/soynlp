from dataclasses import dataclass
from typing import Optional, Union

from soynlp.core.lrgraph import LRGraph, corpus_to_lrgraph
from soynlp.pipeline.tasks.task import Task, TaskArgs


@dataclass
class ExtractNounTaskArgs(TaskArgs):
    max_l_length: int = 10
    max_r_length: int = 9
    min_noun_score: float = 0.3
    min_noun_frequency: int = 1
    min_num_of_features: int = 1
    min_eojeol_frequency: int = 1
    min_eojeol_is_noun_frequency: int = 30

    extract_compounds: bool = True
    exclude_syllables: bool = False
    exclude_numbers: bool = True

    verbose: bool = True

    positive_features: Optional[Union[str, set]] = None
    negative_features: Optional[Union[str, set]] = None

    in_key: str = "corpus"
    text_key: str = "text"


class ExtractNounTask(Task[ExtractNounTaskArgs]):
    def __call__(self, parameters: dict) -> dict:
        if self._args.in_key not in parameters:
            raise ValueError(f"Not found `{self._args.in_key}` in `parameters`")
        examples = parameters[self._args.in_key]
        texts = [example[self._args.text_key] for example in examples]

        lrgraph: LRGraph = corpus_to_lrgraph(texts)  # noqa F841
        candidates = noun_candidates_from_lrgraph()  # noqa F841
        nouns = select_nouns_from_candidates()  # noqa F841
        if self._args.extract_compounds:
            # MaxScoreTokenizer function must be developed first.
            compounds = extract_compounds()  # noqa F841
        nouns = postprocessing()  # noqa F841

        return parameters


def noun_candidates_from_lrgraph(**kwagrs):
    pass


def select_nouns_from_candidates(**kwargs):
    pass


def extract_compounds(**kwargs):
    pass


def postprocessing():
    pass
