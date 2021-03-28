import inspect
import os
from dataclasses import dataclass
from math import log
from tqdm import tqdm

from soynlp.utils import DoublespaceLineCorpus


@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class NgramScore:
    ngram: str
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
        min_frequency=5,
        verbose=True,
        score="frequency",
        filtering_checkpoint=100000,
        tokenizer=None,
    ):
        if tokenizer is None:
            def tokenizer(line):
                return line.split()

        if score == "frequency":
            score = FrequencyScorer()
        elif score == "pmi":
            score = PMIScorer()
        elif score == "mikolov":
            score = MikolovWord2VecScorer(min_frequency)
        elif callable(score):
            parameters = inspect.signature(score).parameters
            if (("unigram" not in parameters) or
                ("bigram" not in parameters) or
                ("threshold" not in parameters) or
                ("topk" not in parameters)
                ):
                raise ValueError(
                    "Callable `score` must have `unigram`, `bigram`, and `threshold` as its arguments"
                )
        else:
            raise ValueError(
                "`score` must be one of ['frequency', 'pmi', 'mikolov', callable]"
            )

        self.min_frequency = min_frequency
        self.verbose = verbose
        self.score = score
        self.filtering_checkpoint = filtering_checkpoint
        self.tokenizer = tokenizer

        self.unigrams = None
        self.bigrams = None

    @property
    def is_trained(self):
        return (self.bigrams is not None) and (len(self.bigrams) > 0)

    def extract(self, train_data, threshold=0, topk=-1):
        def to_bigram(words):
            bigrams = [(w0, w1) for w0, w1 in zip(words, words[1:])]
            return bigrams

        if isinstance(train_data, str) and os.path.exists(train_data):
            train_data = DoublespaceLineCorpus(train_data, verbose=False)

        total = len(train_data)
        if not self.verbose:
            train_iterator = train_data
        else:
            if hasattr(train_data, "__len__"):
                total = len(train_data)
            else:
                total = None
            desc = "[BigramExtractor] counting bigrams"
            train_iterator = tqdm(train_data, desc=desc, total=total)

        unigrams, bigrams = {}, {}
        for i_sent, sent in enumerate(train_iterator):
            if self.filtering_checkpoint > 0 and (
                i_sent % self.filtering_checkpoint == 0
            ):
                bigrams = {
                    bigram: freq
                    for bigram, freq in bigrams.items()
                    if freq >= self.min_frequency
                }
            words = self.tokenizer(sent)
            for word in words:
                unigrams[word] = unigrams.get(word, 0) + 1
            if len(words) <= 1:
                continue
            for bigram in to_bigram(words):
                bigrams[bigram] = bigrams.get(bigram, 0) + 1

        unigrams = {
            unigram: freq
            for unigram, freq in unigrams.items()
            if freq >= self.min_frequency
        }
        bigrams = {
            bigram: freq
            for bigram, freq in bigrams.items()
            if freq >= self.min_frequency
        }
        self.unigrams = unigrams
        self.bigrams = bigrams
        scored = self.score(
            unigrams=unigrams, bigrams=bigrams, threshold=threshold, topk=topk
        )
        return scored


class Scorer:
    def __call__(self, unigrams, bigrams, threshold, topk=-1):
        scored = self.score(
            unigrams=unigrams, bigrams=bigrams, threshold=threshold, topk=topk
        )
        scored = self.filter(scored, threshold=threshold, topk=topk)

        def strf(ngram):
            return " - ".join(ngram)

        scored = {
            strf(ngram): NgramScore(strf(ngram), score.frequency, score.score)
            for ngram, score in scored.items()
        }
        return scored

    def score(self, unigrams, bigrams, threshold, topk=-1):
        raise NotImplementedError("Implement score function")

    def filter(self, scored, threshold, topk=-1):
        scored = {
            ngram: score for ngram, score in scored.items() if score.score >= threshold
        }
        if topk > 0:
            scored = sorted(scored.items(), key=lambda x: -x[1].frequency)[:topk]
            scored = dict(scored)
        return scored


class FrequencyScorer(Scorer):
    def score(self, unigrams, bigrams, threshold=10, topk=-1):
        scored = filter(lambda x: x[1] >= threshold, bigrams.items())
        scored = {
            ngram: NgramScore(ngram, freq, freq) for ngram, freq in bigrams.items()
        }
        return scored


class PMIScorer(Scorer):
    def score(self, unigrams, bigrams, threshold=0, topk=-1):
        def get_pmi(bigram, freq, N):
            base = unigrams.get(bigram[0], 0) * unigrams.get(bigram[1], 0)
            return -9999 if base == 0 else log(N * freq / base)

        N = sum(unigrams.values())
        scored = {}
        for bigram, freq in bigrams.items():
            pmi = get_pmi(bigram, freq, N)
            if pmi >= threshold:
                scored[bigram] = NgramScore(bigram, freq, pmi)
        return scored


class MikolovWord2VecScorer(Scorer):
    def __init__(self, min_frequency):
        self.min_frequency = min_frequency

    def score(self, unigrams, bigrams, threshold=0, topk=-1):
        def get_pmi_like(bigram, freq, N):
            base = unigrams.get(bigram[0], 0) * unigrams.get(bigram[1], 0)
            return 0 if base == 0 else (freq - self.min_frequency) / base

        N = sum(unigrams.values())
        scored = {}
        for bigram, freq in bigrams.items():
            s = get_pmi_like(bigram, freq, N)
            if s >= threshold:
                scored[bigram] = NgramScore(bigram, freq, s)
        return scored
