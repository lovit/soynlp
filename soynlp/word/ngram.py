import inspect
import os
from collections.abc import Callable
from dataclasses import dataclass
from math import log
from typing import Any

from tqdm import tqdm

from soynlp.utils import CorpusLoader


def _count_bigrams_chunk(args: tuple) -> tuple[dict[str, int], dict[tuple[str, str], int]]:
    """Worker: count unigrams and bigrams for a chunk of sentences."""
    chunk, tokenizer = args
    unigrams: dict[str, int] = {}
    bigrams: dict[tuple[str, str], int] = {}
    for sent in chunk:
        if isinstance(sent, dict):
            sent = sent.get("text", "")
        words = tokenizer(sent) if tokenizer is not None else sent.split()
        for word in words:
            unigrams[word] = unigrams.get(word, 0) + 1
        for w0, w1 in zip(words, words[1:]):
            bigrams[(w0, w1)] = bigrams.get((w0, w1), 0) + 1
    return unigrams, bigrams


@dataclass(slots=True)
class NgramScore:
    ngram: str | tuple[str, str]
    frequency: int
    score: float


class BigramExtractor:
    """
    Args:
        score (str, callable)
            Scoring method. choice in ['frequency', 'pmi', 'mikolov', callable]
    """

    def __init__(
        self,
        min_frequency: int = 5,
        verbose: bool = True,
        score: str | Callable[..., dict[str, NgramScore]] = "frequency",
        filtering_checkpoint: int = 100000,
        tokenizer: Callable[[str], list[str]] | None = None,
    ) -> None:
        self._has_custom_tokenizer = tokenizer is not None
        if tokenizer is None:

            def _default_tokenizer(line: str) -> list[str]:
                return line.split()

            tokenizer = _default_tokenizer

        resolved_score: Scorer | Callable[..., dict[str, NgramScore]]
        if score == "frequency":
            resolved_score = FrequencyScorer()
        elif score == "pmi":
            resolved_score = PMIScorer()
        elif score == "mikolov":
            resolved_score = MikolovWord2VecScorer(min_frequency)
        elif callable(score):
            parameters = inspect.signature(score).parameters
            if (
                ("unigram" not in parameters)
                or ("bigram" not in parameters)
                or ("threshold" not in parameters)
                or ("topk" not in parameters)
            ):
                raise ValueError("Callable `score` must have `unigram`, `bigram`, and `threshold` as its arguments")
            resolved_score = score
        else:
            raise ValueError("`score` must be one of ['frequency', 'pmi', 'mikolov', callable]")

        self.min_frequency = min_frequency
        self.verbose = verbose
        self.score: Scorer | Callable[..., dict[str, NgramScore]] = resolved_score
        self.filtering_checkpoint = filtering_checkpoint
        self.tokenizer = tokenizer

        self.unigrams: dict[str, int] | None = None
        self.bigrams: dict[tuple[str, str], int] | None = None

    @property
    def is_trained(self) -> bool:
        return (self.bigrams is not None) and (len(self.bigrams) > 0)

    def extract(
        self,
        train_data: str | list[str] | CorpusLoader,
        threshold: float = 0,
        topk: int = -1,
        n_workers: int = 1,
    ) -> dict[str, NgramScore]:
        if isinstance(train_data, str) and os.path.exists(train_data):
            fmt = "jsonl" if train_data.endswith(".jsonl") else "text"
            train_data = CorpusLoader(train_data, format=fmt)

        if n_workers != 1:
            unigrams, bigrams = self._count_bigrams_parallel(train_data, n_workers)
        else:
            unigrams, bigrams = self._count_bigrams_sequential(train_data)

        unigrams = {u: f for u, f in unigrams.items() if f >= self.min_frequency}
        bigrams = {b: f for b, f in bigrams.items() if f >= self.min_frequency}
        self.unigrams = unigrams
        self.bigrams = bigrams
        scored: dict[str, NgramScore] = self.score(unigrams=unigrams, bigrams=bigrams, threshold=threshold, topk=topk)
        return scored

    def _count_bigrams_sequential(self, train_data: Any) -> tuple[dict[str, int], dict[tuple[str, str], int]]:
        if not self.verbose:
            train_iterator: Any = train_data
        else:
            total: int | None = len(train_data) if hasattr(train_data, "__len__") else None  # type: ignore[arg-type]
            train_iterator = tqdm(train_data, desc="[BigramExtractor] counting bigrams", total=total)

        unigrams: dict[str, int] = {}
        bigrams: dict[tuple[str, str], int] = {}
        for i_sent, sent in enumerate(train_iterator):
            if self.filtering_checkpoint > 0 and (i_sent % self.filtering_checkpoint == 0):
                bigrams = {b: f for b, f in bigrams.items() if f >= self.min_frequency}
            if isinstance(sent, dict):
                sent = sent.get("text", "")
            words = self.tokenizer(sent)
            for word in words:
                unigrams[word] = unigrams.get(word, 0) + 1
            for w0, w1 in zip(words, words[1:]):
                bigrams[(w0, w1)] = bigrams.get((w0, w1), 0) + 1
        return unigrams, bigrams

    def _count_bigrams_parallel(self, train_data: Any, n_workers: int) -> tuple[dict[str, int], dict[tuple[str, str], int]]:
        import pickle
        from multiprocessing import Pool, cpu_count

        texts = list(train_data)
        n = cpu_count() if n_workers == -1 else n_workers
        chunk_size = max(1, len(texts) // n)
        chunks = [texts[i : i + chunk_size] for i in range(0, len(texts), chunk_size)]

        try:
            pickle.dumps(self.tokenizer)
            tok = self.tokenizer if self._has_custom_tokenizer else None
        except (pickle.PicklingError, AttributeError):
            import logging

            logging.getLogger(__name__).info("BigramExtractor: tokenizer가 pickle 불가 — 단일 프로세스로 집계")
            return self._count_bigrams_sequential(train_data)

        worker_args = [(chunk, tok) for chunk in chunks]
        with Pool(processes=n) as pool:
            results = pool.map(_count_bigrams_chunk, worker_args)

        merged_uni: dict[str, int] = {}
        merged_bi: dict[tuple[str, str], int] = {}
        for pU, pB in results:
            for k, v in pU.items():
                merged_uni[k] = merged_uni.get(k, 0) + v
            for k, v in pB.items():
                merged_bi[k] = merged_bi.get(k, 0) + v
        return merged_uni, merged_bi


class Scorer:
    def __call__(
        self,
        unigrams: dict[str, int],
        bigrams: dict[tuple[str, str], int],
        threshold: float,
        topk: int = -1,
    ) -> dict[str, NgramScore]:
        raw_scored = self.score(unigrams=unigrams, bigrams=bigrams, threshold=threshold, topk=topk)
        filtered = self.filter(raw_scored, threshold=threshold, topk=topk)

        def strf(ngram: str | tuple[str, str]) -> str:
            return " - ".join(ngram)

        result = {strf(ngram): NgramScore(strf(ngram), s.frequency, s.score) for ngram, s in filtered.items()}
        return result

    def score(
        self,
        unigrams: dict[str, int],
        bigrams: dict[tuple[str, str], int],
        threshold: float,
        topk: int = -1,
    ) -> dict[tuple[str, str], NgramScore]:
        raise NotImplementedError("Implement score function")

    def filter(
        self,
        scored: dict[tuple[str, str], NgramScore],
        threshold: float,
        topk: int = -1,
    ) -> dict[tuple[str, str], NgramScore]:
        filtered = {ngram: s for ngram, s in scored.items() if s.score >= threshold}
        if topk > 0:
            top_items = sorted(filtered.items(), key=lambda x: -x[1].frequency)[:topk]
            filtered = dict(top_items)
        return filtered


class FrequencyScorer(Scorer):
    def score(
        self,
        unigrams: dict[str, int],
        bigrams: dict[tuple[str, str], int],
        threshold: float = 10,
        topk: int = -1,
    ) -> dict[tuple[str, str], NgramScore]:
        scored = {ngram: NgramScore(ngram, freq, freq) for ngram, freq in bigrams.items()}
        return scored


class PMIScorer(Scorer):
    def score(
        self,
        unigrams: dict[str, int],
        bigrams: dict[tuple[str, str], int],
        threshold: float = 0,
        topk: int = -1,
    ) -> dict[tuple[str, str], NgramScore]:
        def get_pmi(bigram: tuple[str, str], freq: int, N: int) -> float:
            base = unigrams.get(bigram[0], 0) * unigrams.get(bigram[1], 0)
            return -9999 if base == 0 else log(N * freq / base)

        N = sum(unigrams.values())
        scored: dict[tuple[str, str], NgramScore] = {}
        for bigram, freq in bigrams.items():
            pmi = get_pmi(bigram, freq, N)
            if pmi >= threshold:
                scored[bigram] = NgramScore(bigram, freq, pmi)
        return scored


class MikolovWord2VecScorer(Scorer):
    def __init__(self, min_frequency: int) -> None:
        self.min_frequency = min_frequency

    def score(
        self,
        unigrams: dict[str, int],
        bigrams: dict[tuple[str, str], int],
        threshold: float = 0,
        topk: int = -1,
    ) -> dict[tuple[str, str], NgramScore]:
        def get_pmi_like(bigram: tuple[str, str], freq: int, N: int) -> float:
            base = unigrams.get(bigram[0], 0) * unigrams.get(bigram[1], 0)
            return 0 if base == 0 else (freq - self.min_frequency) / base

        N = sum(unigrams.values())
        scored: dict[tuple[str, str], NgramScore] = {}
        for bigram, freq in bigrams.items():
            s = get_pmi_like(bigram, freq, N)
            if s >= threshold:
                scored[bigram] = NgramScore(bigram, freq, s)
        return scored
