from dataclasses import dataclass

from soynlp.pipeline.tasks.task import Task, TaskArgs
from soynlp.word import WordExtractor


@dataclass
class ExtractWordTaskArgs(TaskArgs):
    max_l_length: int = 10
    max_r_length: int = 6
    verbose: bool = True
    min_frequency: int = 5
    min_cohesion_leftside: float = 0.05
    min_cohesion_rightside: float = 0.0
    min_brancingentropy_leftside: float = 0.1
    min_brancingentropy_rightside: float = 0.1
    min_accessorvariety_leftside: int = 2
    min_accessorvariety_rightside: int = 2
    extract_cohesion_only: bool = False
    in_key: str = "corpus"
    text_key: str = "text"
    out_key_cohesion: str = "word_cohesion"
    out_key_accessor_variety: str = "word_accessor_variety"
    out_key_branching_entropy: str = "word_branching_entropy"


class ExtractWordTask(Task[ExtractWordTaskArgs]):
    def __call__(self, parameters: dict) -> dict:
        if self._args.in_key not in parameters:
            raise ValueError(f"Not found `{self._args.in_key}` in `parameters`")

        examples = parameters[self._args.in_key]
        texts = [example[self._args.text_key] for example in examples]

        extractor = WordExtractor(
            max_l_length=self._args.max_l_length,
            max_r_length=self._args.max_r_length,
            verbose=self._args.verbose,
        )
        results = extractor.extract(
            train_data=texts,
            min_frequency=self._args.min_frequency,
            min_cohesion_leftside=self._args.min_cohesion_leftside,
            min_cohesion_rightside=self._args.min_cohesion_rightside,
            min_brancingentropy_leftside=self._args.min_brancingentropy_leftside,
            min_brancingentropy_rightside=self._args.min_brancingentropy_rightside,
            min_accessorvariety_leftside=self._args.min_accessorvariety_leftside,
            min_accessorvariety_rightside=self._args.min_accessorvariety_rightside,
            extract_cohesion_only=self._args.extract_cohesion_only,
        )

        parameters[self._args.out_key_cohesion] = results.get("cohesion", {})
        if not self._args.extract_cohesion_only:
            parameters[self._args.out_key_accessor_variety] = results.get("accessor_variety", {})
            parameters[self._args.out_key_branching_entropy] = results.get("branching_entropy", {})
        return parameters
