from ._tokenizer import MaxScoreTokenizer

class NounLMatchTokenizer:

    def __init__(self, nouns):
        self._nouns  = nouns

    def __call__(self, sentence, compose_compound=True):
        return self.tokenize(sentence, compose_compound)

    def tokenize(self, sentence, compose_compound=True):

        tokens = [self._max_length_l_tokenize(token)
            for token in sentence.split() if token]

        # remove eojeols which noun does not exist
        tokens = [token for token in tokens if token and token[0]]

        # remove r parts
        tokens = [token[0] for token in tokens]

        if compose_compound:
            tokens = [''.join(token) for token in tokens]
        else:
            tokens = [unit for token in tokens for unit in token]

        tokens = [token for token in tokens if token]
        return tokens

    def _max_length_l_tokenize(self, token):
        
        def nouns_to_larray_and_r(token, nouns_):
            e = sum((len(noun) for noun in nouns_))
            return nouns_, token[e:]

        nouns = []
        n = len(token)

        # string match for generating candidats
        for b in range(n):
            for e in range(b, n+1):
                subword = token[b:e]
                if subword in self._nouns:
                    # (word, begin, length)
                    nouns.append((subword, b, e - b))

        # sort. fisrt order: begin index, second order: length (desc)
        nouns = sorted(nouns, key=lambda x:(x[1], -x[2]))

        nouns_ = []
        e = 0

        while nouns:
            # pop first element
            noun, b, len_ = nouns.pop(0)
            # only concatenate nouns
            if not (b == e):
                return nouns_to_larray_and_r(token, nouns_)
            # append noun and update end index
            nouns_.append(noun)
            e = b + len_
            nouns = [noun for noun in nouns if noun[1] >= e]

        return nouns_to_larray_and_r(token, nouns_)

class NounMatchTokenizer:

    def __init__(self, noun_scores):
        self._tokenizer = MaxScoreTokenizer(scores=noun_scores)

    def __call__(self, sentence, flatten=True, compose_compound=True):
        return self.tokenize(sentence, flatten, compose_compound)

    def tokenize(self, sentence, flatten=True, compose_compound=True):

        def concatenate(eojeol, words):
            words_, b, e, score = [], 0, 0, 0
            for noun_, b_, e_, score_, _ in words:
                if e == b_:
                    e, score = e_, max(score, score_)
                else:
                    words_.append((eojeol[b:e], b, e, score, e-b))
                    b, e = b_, e_
            if e > b:
                words_.append((eojeol[b:e], b, e, score, e-b))
            return words_

        sentence_ = []
        for eojeol in sentence.split():

            eojeol = eojeol.strip()
            if not eojeol:
                continue

            words = self._tokenizer(eojeol, flatten=False)[0]
            # remove non-noun words
            words = [word for word in words if word[3] > 0]

            if compose_compound:
                words = concatenate(eojeol, words)

            sentence_.append(words)

        if flatten:
            sentence_ = [word[0] for words in sentence_ for word in words if word[0]]

        return sentence_