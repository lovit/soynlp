import logging
from collections import namedtuple
from math import log


def _log(*args, **kwargs):
    logging.getLogger(__name__).info(" ".join(map(str, args)))


NgramScore = namedtuple("NgramScore", "frequency score")


class Bigram:
    def __init__(
        self,
        sentences=None,
        min_frequency=5,
        verbose=True,
        score="frequency",
        filtering_checkpoint=100000,
        tokenizer=None,
        ngram_selector=None,
    ):
        """
        Attribute:
        ----------
        score : str or functional
            Scoring method. choice in ['frequency', 'pmi', 'mikolov']
        """
        if tokenizer is None:

            def default_tokenizer(x):
                return x.split()

            tokenizer = default_tokenizer

        if ngram_selector is None:

            def default_selector(x):
                return x

            ngram_selector = default_selector

        self.min_frequency = min_frequency
        self.verbose = verbose
        self.score = score
        self.filtering_checkpoint = filtering_checkpoint
        self.tokenizer = tokenizer
        self.ngram_selector = ngram_selector
        self._counter = None

    @property
    def is_trained(self):
        return self._counter

    def train(self, sentences):

        def to_bigram(words):
            bigrams = [(w0, w1) for w0, w1 in zip(words, words[1:])]
            return bigrams

        self._counter = {}
        for i_sent, sent in enumerate(sentences):
            if self.filtering_checkpoint > 0 and i_sent % self.filtering_checkpoint == 0:
                self._counter = {bigram: freq for bigram, freq in self._counter.items() if freq >= self.min_frequency}
            if self.verbose and i_sent % 3000 == 0:
                _log(f"\r[Bigram Extractor] scanning {len(self._counter)} bigrams from {i_sent} sents", end="", flush=True)
            words = self.tokenizer(sent)
            if len(words) <= 1:
                continue
            bigrams = to_bigram(words)
            for bigram in bigrams:
                self._counter[bigram] = self._counter.get(bigram, 0) + 1
        self._counter = {bigram: freq for bigram, freq in self._counter.items() if freq >= self.min_frequency}
        self._unigram = {}
        for bigram, freq in self._counter.items():
            for unigram in bigram:
                self._unigram[unigram] = self._unigram.get(unigram, 0) + 1
        if self.verbose:
            _log(
                f"\r[Bigram Extractor] scanning {len(self._unigram)} unigrams, {len(self._counter)} bigrams from {i_sent} sents",
                flush=True,
            )

    def extract(self, topk=-1, threshold=0):
        if self._counter is None:
            raise ValueError("train() must be called before extraction")
        if self.score == "frequency":
            return self._extract_by_frequency(topk, threshold)
        elif self.score == "pmi":
            return self._extract_by_pmi(topk, threshold)
        elif self.score == "mikolov":
            return self._extract_by_mikolov(topk, threshold)
        raise NotImplementedError

    def _extract_by_pmi(self, topk=-1, threshold=0):
        assert self._counter is not None

        def score(bigram, freq, N):
            base = self._unigram[bigram[0]] * self._unigram[bigram[1]]
            return 0 if base == 0 else log(N * freq / base)

        N = 2 * sum(self._counter.values())
        pmis = {}
        for bigram, freq in self._counter.items():
            pmi = score(bigram, freq, N)
            if pmi >= threshold:
                pmis[bigram] = pmi
        bigrams = {word: NgramScore(freq, pmis[word]) for word, freq in self._counter.items() if word in pmis}
        return bigrams

    def _extract_by_frequency(self, topk=-1, threshold=10):
        assert self._counter is not None
        bigrams = filter(lambda x: x[1] >= threshold, self._counter.items())
        if topk > 0:
            bigrams = sorted(bigrams, key=lambda x: -x[1])
        bigrams = {word: NgramScore(freq, freq) for word, freq in bigrams}
        return bigrams

    def _extract_by_mikolov(self, topk=-1, threshold=0):
        assert self._counter is not None

        def score(bigram, freq, N):
            base = self._unigram[bigram[0]] * self._unigram[bigram[1]]
            return 0 if base == 0 else (freq - self.min_frequency) / base

        N = 2 * sum(self._counter.values())
        scores = {}
        for bigram, freq in self._counter.items():
            s = score(bigram, freq, N)
            if s >= threshold:
                scores[bigram] = s
        bigrams = {word: NgramScore(freq, scores[word]) for word, freq in self._counter.items() if word in scores}
        return bigrams
