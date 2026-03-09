from dataclasses import dataclass

from soynlp.noun import LRNounExtractor
from soynlp.pipeline.tasks.task import Task, TaskArgs


@dataclass(slots=True)
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
    postprocessing_nj: bool = True
    verbose: bool = True
    positive_features: str | set | None = None
    negative_features: str | set | None = None
    in_key: str = "corpus"
    text_key: str = "text"
    out_key: str = "nouns"
    n_workers: int = 1


class ExtractNounTask(Task[ExtractNounTaskArgs]):
    def __call__(self, parameters: dict) -> dict:
        if self._args.in_key not in parameters:
            raise ValueError(f"Not found `{self._args.in_key}` in `parameters`")

        examples = parameters[self._args.in_key]
        texts = [example[self._args.text_key] for example in examples]

        extractor = LRNounExtractor(
            max_l_length=self._args.max_l_length,
            max_r_length=self._args.max_r_length,
            pos_features=self._args.positive_features,
            neg_features=self._args.negative_features,
            verbose=self._args.verbose,
        )
        nouns = extractor.extract(
            train_data=texts,
            min_noun_score=self._args.min_noun_score,
            min_noun_frequency=self._args.min_noun_frequency,
            min_num_of_features=self._args.min_num_of_features,
            min_eojeol_frequency=self._args.min_eojeol_frequency,
            min_eojeol_is_noun_frequency=self._args.min_eojeol_is_noun_frequency,
            extract_compounds=self._args.extract_compounds,
            exclude_syllables=self._args.exclude_syllables,
            exclude_numbers=self._args.exclude_numbers,
            postprocessing_nj=self._args.postprocessing_nj,
            n_workers=self._args.n_workers,
        )

        parameters[self._args.out_key] = nouns
        return parameters
