class Bigram:
    def __init__(self, sentences=None, min_count=5, verbose=True, score='count',
        filtering_checkpoint=100000, tokenizer=None, ngram_selector=None):

        """
        Attribute:
        ----------
        score : str or functional
            Scoring method. choice in ['count', 'pmi', 'mikolov']
        """

        if tokenizer is None:
            tokenizer = lambda x:x.split()

        if ngram_selector is None:
            ngram_selector = lambda x:x

        self.min_count = min_count
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
                self._counter = {bigram:count for bigram, count
                    in self._counter.items() if count >= self.min_count}

            if self.verbose and i_sent % 3000 == 0:
                print('\r[Bigram Extractor] scanning {} bigrams from {} sents'.format(
                    len(self._counter), i_sent), end='', flush=True)

            words = self.tokenizer(sent)
            if len(words) <= 1:
                continue

            bigrams = to_bigram(words)
            for bigram in bigrams:
                self._counter[bigram] = self._counter.get(bigram, 0) + 1

        self._counter = {bigram:count for bigram, count
            in self._counter.items() if count >= self.min_count}

        self._unigram = {}
        for bigram, count in self._counter.items():
            for unigram in bigram:
                self._unigram[unigram] = self._unigram.get(unigram, 0) + 1

        if self.verbose:
            print('\r[Bigram Extractor] scanning {} unigrams, {} bigrams from {} sents'.format(
                len(self._unigram), len(self._counter), i_sent), flush=True)

    def extract(self, topk=-1, threshold=0):
        if self.score == 'count':
            return self._extract_by_count(topk, threshold)
        elif self.score == 'pmi':
            return self._extract_by_pmi(topk, threshold)

        raise NotImplemented

    def _extract_by_pmi(self, topk=-1, threshold=0):
        raise NotImplemented

    def _extract_by_count(self, topk=-1, threshold=10):
        raise NotImplemented