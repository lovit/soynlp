from ._tokenizer import MaxScoreTokenizer

class NounLMatchTokenizer:

    def __init__(self, nouns):
        self._nouns  = nouns

    def __call__(self, sentence, tolerance=0.0, compose_compound=True):
        return self.tokenize(sentence, tolerance, compose_compound)

    def tokenize(self, sentence, tolerance=0.0, compose_compound=True):

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