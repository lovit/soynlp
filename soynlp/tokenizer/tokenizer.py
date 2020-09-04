import re
from collections import namedtuple


class Token(namedtuple('Token', 'word begin end score length')):
    """collections.namedtuple class

    Args:
        word (str) : surfacial form of word
        begin (int) : begin position of `word` in the input sentence
        end (int) : end position of `word` in the input sentence
        score (float) : word score
        length (int) : word length. It must be equal with `end` - `begin`
    """
    def __repr__(self):
        return f'Token({self.word}, score={self.score}, offset=({self.begin}, {self.end}))'


class RegexTokenizer:
    """
    Split sentence based on type of characters and regex pattern.
    Or it is available to customize RegexTokenizer with my regex patterns.

    Args:
        pipelines (list of re.Pattern or None) :
            The regex patterns will be applied one by one to input string.
            If None, it uses default patterns (number -> Korean -> jaum -> moum -> Alphabet)

    Examples::
        >>> from soynlp.tokenizer import RegexTokenizer

        >>> s = 'abc123가나다 alphabet!!3.14한글 hank`s report'
        >>> regex_tokenizer = RegexTokenizer()
        >>> regex_tokenizer.tokenize(s)
        $ ['abc', '123', '가나다', 'alphabet', '!!', '3.14', '한글', 'hank`s', 'report']

        >>> regex_tokenizer(s, flatten=False)
        $ [[Token(abc, score=1, offset=(0, 3)),
            Token(123, score=1, offset=(3, 6)),
            Token(가나다, score=1, offset=(6, 9))],
           [Token(alphabet, score=1, offset=(10, 18)),
            Token(!!, score=1, offset=(18, 20)),
            Token(3.14, score=1, offset=(20, 24)),
            Token(한글, score=1, offset=(24, 26))],
           [Token(hank`s, score=1, offset=(27, 33))],
           [Token(report, score=1, offset=(34, 40))]]

    """
    def __init__(self, pipelines=None):
        if pipelines is None:
            pipelines = self._default_pipelines()
        self.pipelines = pipelines
        self.doublewhite_pattern = re.compile('\s+')

    def _default_pipelines(self):
        return [
            re.compile(u'[-+]?\d*[\.]?[\d]+|[-+]?\d+', re.UNICODE),  # number
            re.compile(u'[가-힣]+', re.UNICODE),                      # Korean
            re.compile(u'[ㄱ-ㅎ]+', re.UNICODE),                      # jaum
            re.compile(u'[ㅏ-ㅣ]+', re.UNICODE),                      # moum
            re.compile(u"[a-zA-ZÀ-ÿ]+[[`']{1,1}s]*|[a-zA-ZÀ-ÿ]+", re.UNICODE),  # Alphabet
        ]

    def __call__(self, sentence, flatten=True):
        return self.tokenize(sentence, flatten)

    def tokenize(self, sentence, flatten=True):
        """Split sentence based on type of characters and regex pattern.

        Args:
            sentence (str) : input string
            flatten (Boolean) :
                If True, it returns tokens as form of list of str
                Otherwise, it returns nested list of `Token`

        Returns:
            tokens (list of str or nested list of Token)

        Examples::
            >>> from soynlp.tokenizer import RegexTokenizer

            >>> s = 'abc123가나다 alphabet!!3.14한글 hank`s report'
            >>> regex_tokenizer = RegexTokenizer()
            >>> regex_tokenizer.tokenize(s)
            >>> regex_tokenizer(s) # same with above line.
            $ ['abc', '123', '가나다', 'alphabet', '!!', '3.14', '한글', 'hank`s', 'report']

            >>> regex_tokenizer(s, flatten=False)
            $ [[Token(abc, score=1, offset=(0, 3)),
                Token(123, score=1, offset=(3, 6)),
                Token(가나다, score=1, offset=(6, 9))],
               [Token(alphabet, score=1, offset=(10, 18)),
                Token(!!, score=1, offset=(18, 20)),
                Token(3.14, score=1, offset=(20, 24)),
                Token(한글, score=1, offset=(24, 26))],
               [Token(hank`s, score=1, offset=(27, 33))],
               [Token(report, score=1, offset=(34, 40))]]
        """
        offset = 0
        tokens = []
        for token in sentence.split():
            tokens.append(self._tokenize(token, offset))
            offset += (len(token) + 1)
        if flatten:
            tokens = [token.word for tokens_in_eojeol in tokens for token in tokens_in_eojeol if token.word]
        return tokens

    def _tokenize(self, s, offset=0):
        # TODO: handle 3.1.2.1
        for pattern in self.pipelines:
            founds = pattern.findall(s)
            if not founds:
                continue
            found = founds.pop(0)
            len_found = len(found)

            s_ = ''
            begin = 0
            for i, char in enumerate(s):
                if begin > i:
                    continue
                if s[i: i + len_found] == found:
                    s_ += ' %s ' % s[i: i + len_found]
                    begin = i + len_found
                    if not founds:
                        s_ += s[begin:]
                        break
                    else:
                        found = founds.pop(0)
                        len_found = len(found)
                    continue
                s_ += char
            s = s_
        words = self.doublewhite_pattern.sub(' ', s).strip().split()
        r = len(words[0])
        tokens = [Token(words[0], 0 + offset, r + offset, 1, r)]
        begin = tokens[0].end
        for word in words[1:]:
            r = len(word)
            tokens.append(Token(word, begin, begin + r, 1, r))
            begin += r
        return tokens


class LTokenizer:
    """It finds the most word-like substring which is positioned on the left-side
    in given Eojeol. An `Eojeol` is space-separated token in Korean.


    Args:
        scores ({str: float}) : {word: score}
        unknown_score (float) : unknown word score

    Examples::
        Without tolerance

            >>> from soynlp.tokenizer import LTokenizer

            >>> scores = {'파스': 0.65, '파스타': 0.7, '좋아': 0.3}
            >>> ltokenizer = LTokenizer(scores)
            >>> ltokenizer.tokenize('파스타가 좋아요 파스타가좋아요')
            >>> ltokenizer('파스타가 좋아요 파스타가좋아요')  # same with above line
            $ ['파스타', '가', '좋아', '요', '파스타', '가좋아요']

            >>> ltokenizer.tokenize('파스타가 좋아요 파스타가좋아요', flatten=False)
            $ [[Token(파스타, score=0.7, offset=(0, 3)),
                Token(가, score=0, offset=(3, 4))],
               [Token(좋아, score=0.3, offset=(5, 7)),
                Token(요, score=0, offset=(7, 8))],
               [Token(파스타, score=0.7, offset=(9, 12)),
                Token(가좋아요, score=0, offset=(12, 16))]]

        With tolerance

            >>> scores = {'파스': 0.75, '파스타': 0.7, '좋아': 0.3}
            >>> ltokenizer = LTokenizer(scores)
            >>> ltokenizer.tokenize('파스타가 좋아요 파스타가좋아요', tolerance=0.06)
            $ ['파스타', '가', '좋아', '요', '파스타', '가좋아요']

            >>> ltokenizer.tokenize('파스타가 좋아요 파스타가좋아요', tolerance=0.06, flatten=False)
            $ [[Token(파스타, score=0.7, offset=(0, 3)),
                Token(가, score=0, offset=(3, 4))],
               [Token(좋아, score=0.3, offset=(5, 7)),
                Token(요, score=0, offset=(7, 8))],
               [Token(파스타, score=0.7, offset=(9, 12)),
                Token(가좋아요, score=0, offset=(12, 16))]]
    """
    def __init__(self, scores, unknown_score=0.0):
        self.scores = scores
        self.unknown_score = unknown_score

    def __call__(self, sentence, tolerance=0.0, flatten=True, remove_r=False):
        return self.tokenize(sentence, tolerance, flatten, remove_r)

    def tokenize(self, sentence, tolerance=0.0, flatten=True, remove_r=False):
        """
        Args:
            sentence (str) : input string
            tolerance (float) :
                If the difference between the highest and the second highest score
                is less than `tolerance`, this tokenizer chooses longer one as word
            flatten (Boolean) :
                If True, it returns tokens as form of list of str
                Otherwise, it returns nested list of `Token`
            remove_r (Boolean) :
                If True, it returns only L parts

        Returns:
            tokens (list of str or nested list of Token)
        """

        def token_to_lr(token):
            """Returns: (score of L, L, R)"""
            n = len(token)
            if n <= 2:
                return (self.scores.get(l, self.unknown_score), token, '')

            candidates = [(token[:end], token[end:]) for end in range(2, n + 1)]
            candidates = [(self.scores.get(l, self.unknown_score), l, r) for l, r in candidates]
            if tolerance > 0:
                max_score = max([c[0] for c in candidates])
                candidates = [c for c in candidates if (max_score - c[0]) <= tolerance]
                best = sorted(candidates, key=lambda x: len(x[1]), reverse=True)[0]
            else:
                best = sorted(candidates, key=lambda x: (x[0], len(x[1])), reverse=True)[0]
            return best

        offset = 0
        tokens = []
        for s in sentence.split():
            score, l, r = token_to_lr(s)
            len_l, len_r = len(l), len(r)
            tokens.append([
                Token(l, offset, offset + len_l, score, len_l),
                Token(r, offset + len_l, offset + len_l + len_r, 0, len_r)
            ])
            offset += (len_l + len_r + 1)

        if remove_r:
            tokens = [l.word for l, r in tokens]

        if (flatten) and (not remove_r):
            tokens = [subtoken.word for token in tokens for subtoken in token if subtoken.length > 0]

        return tokens


class MaxScoreTokenizer:
    """It finds the most word-like substring in a given Eojeol regardless the position
    of sustring in eojeol. An `Eojeol` is space-separated token in Korean.

    Args:
        scores ({str: float}) : {word: word_score}
        max_length (int) : maximum length of L part word
        unknown_score (float) : score of unknown word

    Examples::
        Import class

            >>> from soynlp.tokenizer import MaxScoreTokenizer
            >>> from soynlp.utils import DoublespaceLineCorpus
            >>> from soynlp.word import WordExtractor

        With pretrained word scores

            >>> scores = {'파스': 0.65, '파스타': 0.7, '좋아': 0.3}
            >>> tokenizer = MaxScoreTokenizer(scores)
            >>> tokenizer.tokenize('파스타가좋아요')
            $ ['파스타', '가', '좋아', '요']

            >>> tokenizer.tokenize('파스타가좋아요', flatten=False)
            $ [[Token(파스타, score=0.7, offset=(0, 3)),
                Token(가, score=0.0, offset=(3, 4)),
                Token(좋아, score=0.3, offset=(4, 6)),
                Token(요, score=0.0, offset=(6, 7))]]

        With training word extractor

            >>> corpus = DoublespaceLineCorpus('path/to/corpus', iter_sent=True)
            >>> word_extractor = WordExtractor()
            >>> word_extractor.train(corpus)
            >>> cohesion_scores = word_extractor.all_cohesion_scores()
            >>> cohesion_scores = {l: l_score for l, (l_score, r_score) in cohesion_scores.items()}
            >>> tokenizer = MaxScoreTokenizer(cohesion_scores)
            >>> tokenizer.tokenize('예시문장입니다')
            >>> tokenizer.tokenize('예시문장입니다', flatten=False)
    """
    def __init__(self, scores, max_length=10, unknown_score=0.0):
        self.scores = scores
        self.max_len = max_length
        self.unknown_score = unknown_score

    def __call__(self, sentence, flatten=True):
        return self.tokenize(sentence, flatten)

    def tokenize(self, sentence, flatten=True):
        """
        Args:
            sentence (str) : input string
            flatten (Boolean) :
                If True, it returns tokens as form of list of str
                Otherwise, it returns nested list of `Token`

        Returns:
            tokens (list of str or nested list of Token)
        """
        offset = 0
        tokens = []
        for s in sentence.split():
            tokens.append(self._recursive_tokenize(s, offset))
            offset += (len(s) + 1)
        if flatten:
            tokens = [subtoken.word for token in tokens for subtoken in token]
        return tokens

    def _recursive_tokenize(self, s, offset):
        length = len(s)
        if length <= 2:
            token = Token(
                s,
                offset,
                offset + length,
                self.scores.get(s, self.unknown_score),
                length
            )
            return [token]

        scored = self._initialize(s, length, offset)
        tokens = self._find(scored)
        adds = self._add_inter_tokens(s, tokens, offset)
        if tokens[-1].end != offset + length:
            adds += self._add_last_token(s, tokens, offset)
        if tokens[0].begin != offset:
            adds += self._add_first_token(s, tokens, offset)
        return sorted(tokens + adds, key=lambda x: x.begin)

    def _initialize(self, s, length, offset=0):
        max_r = min(length, self.max_len)
        scored = []
        for begin in range(0, length - 1):
            for r in range(2, max_r + 1):
                end = begin + r
                if end > length:
                    continue
                subtoken = s[begin: end]
                if subtoken not in self.scores:
                    continue
                score = self.scores[subtoken]
                scored.append(Token(subtoken, offset + begin, offset + end, score, r))
        if not scored:
            return [Token(s, offset, offset + len(s), self.unknown_score, len(s))]
        return sorted(scored, key=lambda x: (-x.score, -x.length, x.begin))

    def _find(self, scored):
        result = []
        num_iter = 0
        while scored:
            token = scored.pop(0)
            result.append(token)
            if not scored:
                break
            removals = []
            for i, token_i in enumerate(scored):
                if ((token_i.begin < token.end and token.begin < token_i.end) or
                    (token_i.begin < token.end and token_i.end > token.begin)):
                    removals.append(i)
            for i in reversed(removals):
                del scored[i]
            num_iter += 1
            if num_iter > 100:
                break
        return sorted(result, key=lambda x: x.begin)

    def _add_inter_tokens(self, s, tokens, offset=0):
        adds = []
        for i, token in enumerate(tokens[: -1]):
            if token.end == tokens[i + 1].begin:
                continue
            begin = token.end - offset
            end = tokens[i + 1].begin - offset
            sub = s[begin: end]
            adds.append(Token(sub, offset + begin, offset + end, self.unknown_score, end - begin))
        return adds

    def _add_first_token(self, s, tokens, offset=0):
        begin = tokens[0].begin
        sub = s[0: begin - offset]
        score = self.scores.get(sub, self.unknown_score)
        return [Token(sub, offset, begin, score, begin - offset)]

    def _add_last_token(self, s, tokens, offset=0):
        end = tokens[-1].end
        sub = s[end - offset:]
        if not sub:
            return []
        score = self.scores.get(sub, self.unknown_score)
        return [Token(sub, end, end + len(sub), score, len(sub))]


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
            $ [[Token(아이오아이, score=0.8, offset=(0, 5)),
                Token(아이, score=0.5, offset=(6, 8))],
               [Token(오이, score=0.7, offset=(11, 13)),
                Token(오이, score=0.7, offset=(13, 15))],
               [],
               [Token(아이, score=0.5, offset=(22, 24))]]

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
            concats, begin, end, score = [], offset, offset, 0
            for token in tokens:
                if end == token.begin:
                    end, score = token.end, max(score, token.score)
                else:
                    concats.append(Token(eojeol[begin - offset: end - offset], begin, end, score, end - begin))
                    begin, end = token.begin, token.end
            if end > begin:
                concats.append(Token(eojeol[begin - offset: end - offset], begin, end, score, end - begin))
            return concats

        offset = 0
        tokens = []
        for s in sentence.split():
            nouns = self._recursive_tokenize(s, offset)
            nouns = [noun for noun in nouns if noun.score > 0]
            if concat_compound:
                nouns = concatenate(s, nouns, offset)
            if must_be_L and nouns:
                if nouns[0].begin != offset:
                    nouns = []
                else:
                    nouns = nouns[:1]
            tokens.append(nouns)
            offset += (len(s) + 1)
        if flatten:
            tokens = [noun.word for nouns in tokens for noun in nouns]
        return tokens
