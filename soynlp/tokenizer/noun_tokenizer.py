from .tokenizer import MaxScoreTokenizer, Token


class NounMatchTokenizer(MaxScoreTokenizer):
    """NounMatchTokenizer recognizes nouns from input sentence.
    NounMatchTokenizer works similar to soynlp.tokenizer.MaxScoreTokenizer.
    The difference is that NounMatchTokenizer provides merging
    consecutive nouns into one compound noun.

    Args:
        noun_scores ({str: float}) : {noun: noun_score}

    Examples::
        With noun scores. Match first with higher scored noun.

            >>> noun_scores = {'아이': 0.5, '아이오': 0.7, '아이오아이': 0.8, '오이': 0.7}
            >>> noun_tokenizer = NounMatchTokenizer(noun_scores)
            >>> sentence = '아이오아이의아이들은 오이오이를 좋아하는 아이들이오'
            >>> noun_tokenizer.tokenize(sentence)
            $ ['아이오아이', '아이', '오이오이', '아이']

        With noun set or list. Match longer one first if noun scores are tied.

            >>> noun_set = {'아이', '아이오', '아이오아이', '오이'}
            >>> noun_tokenizer = NounMatchTokenizer(noun_set)
            >>> noun_tokenizer.tokenize(sentence)
            $ ['아이오아이', '아이', '오이오이', '아이']

        Without concatenating consecutive nouns

            >>> noun_tokenizer.tokenize(sentence, concat_compound=False)
            $ ['아이오아이', '아이', '오이', '오이', '아이']

        Without flattening tokens

            >>> noun_tokenizer.tokenize(sentence, concat_compound=False, flatten=False)
            $ [[Token(word='아이오아이', b=0, e=5, score=1.0, length=5),
                Token(word='아이', b=6, e=8, score=1.0, length=2)],
               [Token(word='오이', b=11, e=13, score=1.0, length=2),
                Token(word='오이', b=13, e=15, score=1.0, length=2)],
               [],
               [Token(word='아이', b=22, e=24, score=1.0, length=2)]]

        Remain only L parts

            >>> sentence = '아이오아이의아이들은 오이오이를 좋아하는 아이들이오'
            >>> noun_tokenizer.tokenize(sentence, concat_compound=True, must_be_L=True)
            $ ['오이오이', '아이']
    """
    def __init__(self, noun_scores):
        if (isinstance(noun_scores, list) or
            isinstance(noun_scores, set) or
            isinstance(noun_scores, tuple)):
            noun_scores = {noun: 1.0 for noun in noun_scores}
        super().__init__(noun_scores)

    def __call__(self, sentence, flatten=True, concat_compound=True):
        return self.tokenize(sentence, flatten, concat_compound)

    def tokenize(self, sentence, flatten=True, concat_compound=True, must_be_L=False):
        """
        Args:
            sentence (str) : input string
            flatten (Boolean) :
                If True, it returns tokens as form of list of str
                Otherwise, it returns nested list of `Token`
            concat_compound (Boolean) :
                If True, it concatenates consecutive nouns into one compound noun.
            must_be_L (Boolean) :
                If True, it remains nouns which position left-side on eojeol.

        Returns:
            tokens (list of str or nested list of Token)
        """

        def concatenate(eojeol, tokens, offset):
            concats, b, e, score = [], offset, offset, 0
            for token in tokens:
                if e == token.b:
                    e, score = token.e, max(score, token.score)
                else:
                    concats.append(Token(eojeol[b - offset: e - offset], b, e, score, e-b))
                    b, e = token.b, token.e
            if e > b:
                concats.append(Token(eojeol[b - offset: e - offset], b, e, score, e - b))
            return concats

        offset = 0
        tokens = []
        for s in sentence.split():
            nouns = self._recursive_tokenize(s, offset)
            nouns = [noun for noun in nouns if noun.score > 0]
            if concat_compound:
                nouns = concatenate(s, nouns, offset)
            if must_be_L and nouns:
                if nouns[0].b != offset:
                    nouns = []
                else:
                    nouns = nouns[:1]
            tokens.append(nouns)
            offset += (len(s) + 1)
        if flatten:
            tokens = [noun.word for nouns in tokens for noun in nouns]
        return tokens
