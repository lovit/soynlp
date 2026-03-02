from dataclasses import dataclass

from soynlp.pipeline.tasks.task import Task, TaskArgs
from soynlp.tokenizer import LTokenizer, MaxScoreTokenizer, NounMatchTokenizer, RegexTokenizer


@dataclass
class TokenizeTaskArgs(TaskArgs):
    tokenizer_type: str = "max_score"
    scores_key: str = "nouns"
    score_field: str = "score"
    in_key: str = "corpus"
    text_key: str = "text"
    out_key: str = "corpus"
    tokens_key: str = "tokens"


TOKENIZER_TYPES = {"noun_match", "max_score", "l_tokenizer", "regex"}


class TokenizeTask(Task[TokenizeTaskArgs]):
    def __call__(self, parameters: dict) -> dict:
        if self._args.in_key not in parameters:
            raise ValueError(f"Not found `{self._args.in_key}` in `parameters`")
        if self._args.tokenizer_type not in TOKENIZER_TYPES:
            raise ValueError(f"Unknown tokenizer_type `{self._args.tokenizer_type}`. Available: {sorted(TOKENIZER_TYPES)}")

        tokenizer = self._build_tokenizer(parameters)
        examples = parameters[self._args.in_key]
        tokenized = []
        for example in examples:
            text = example[self._args.text_key]
            tokens = tokenizer.tokenize(text)
            tokenized.append({**example, self._args.tokens_key: tokens})

        parameters[self._args.out_key] = tokenized
        return parameters

    def _build_tokenizer(self, parameters):
        if self._args.tokenizer_type == "regex":
            return RegexTokenizer()

        if self._args.scores_key not in parameters:
            raise ValueError(f"Not found `{self._args.scores_key}` in `parameters`")
        raw_scores = parameters[self._args.scores_key]
        scores = self._extract_scores(raw_scores)

        if self._args.tokenizer_type == "noun_match":
            return NounMatchTokenizer(scores)
        elif self._args.tokenizer_type == "max_score":
            return MaxScoreTokenizer(scores)
        elif self._args.tokenizer_type == "l_tokenizer":
            return LTokenizer(scores)
        raise ValueError(f"Unknown tokenizer_type: {self._args.tokenizer_type}")

    def _extract_scores(self, raw_scores: dict) -> dict:
        if not raw_scores:
            return {}
        first_value = next(iter(raw_scores.values()))
        if isinstance(first_value, (int, float)):
            return raw_scores
        field = self._args.score_field
        scores = {}
        for key, value in raw_scores.items():
            if hasattr(value, field):
                scores[key] = getattr(value, field)
            elif hasattr(value, "_asdict"):
                scores[key] = getattr(value, field)
            elif isinstance(value, dict):
                scores[key] = value[field]
            else:
                scores[key] = float(value)
        return scores
