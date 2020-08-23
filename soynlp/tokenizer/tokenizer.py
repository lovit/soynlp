import re
from collections import namedtuple
from pprint import pprint


Token = namedtuple('Token', 'word b e score length')


class RegexTokenizer:
    """
    Split sentence based on type of characters and regex pattern.
    Or it is available to customize RegexTokenizer with my regex patterns.

    Args:
        pipielines (list of re.Pattern or None) :
            The regex patterns will be applied one by one to input string.
            If None, it uses default patterns (number -> Korean -> jaum -> moum -> Alphabet)
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
            >>> s = 'abc123가나다 alphabet!!3.14한글 hank`s report'
            >>> regex_tokenizer = RegexTokenizer()
            >>> regex_tokenizer.tokenize(s)
            >>> regex_tokenizer(s) # same with above line.
            $ ['abc', '123', '가나다', 'alphabet', '!!', '3.14', '한글', 'hank`s', 'report']

            >>> regex_tokenizer.tokenize(s, flatten=False)
            >>> regex_tokenizer(s, flatten=False)  # same with above line.
            $ [[Token(word='abc', b=0, e=3, score=1, length=3),
                Token(word='123', b=3, e=6, score=1, length=3),
                Token(word='가나다', b=6, e=9, score=1, length=3)],
               [Token(word='alphabet', b=10, e=18, score=1, length=8),
                Token(word='!!', b=18, e=20, score=1, length=2),
                Token(word='3.14', b=20, e=24, score=1, length=4),
                Token(word='한글', b=24, e=26, score=1, length=2)],
               [Token(word='hank`s', b=27, e=33, score=1, length=6)],
               [Token(word='report', b=34, e=40, score=1, length=6)]]
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
            b = 0
            for i, c in enumerate(s):
                if b > i:
                    continue
                if s[i: i + len_found] == found:
                    s_ += ' %s ' % s[i: i + len_found]
                    b = i + len_found
                    if not founds:
                        s_ += s[b:]
                        break
                    else:
                        found = founds.pop(0)
                        len_found = len(found)
                    continue
                s_ += c
            s = s_
        words = self.doublewhite_pattern.sub(' ', s).strip().split()
        r = len(words[0])
        tokens = [Token(words[0], 0 + offset, r + offset, 1, r)]
        b = tokens[0].e
        for word in words[1:]:
            r = len(word)
            tokens.append(Token(word, b, b + r, 1, r))
            b += r
        return tokens


class LTokenizer:
    """
    Args:
        scores ({str: float}) : {word: score}
        unknown_score (float) : unknown word score

    Examples::
        Without tolerance

            >>> scores = {'파스': 0.65, '파스타': 0.7, '좋아': 0.3}
            >>> ltokenizer = LTokenizer(scores)
            >>> ltokenizer.tokenize('파스타가 좋아요 파스타가좋아요')
            >>> ltokenizer('파스타가 좋아요 파스타가좋아요')  # same with above line
            $ ['파스타', '가', '좋아', '요', '파스타', '가좋아요']

            >>> ltokenizer.tokenize('파스타가 좋아요 파스타가좋아요', flatten=False)
            $ [[Token(word='파스타', b=0, e=3, score=0.7, length=3),
                Token(word='가', b=3, e=4, score=0, length=1)],
               [Token(word='좋아', b=5, e=7, score=0.3, length=2),
                Token(word='요', b=7, e=8, score=0, length=1)],
               [Token(word='파스타', b=9, e=12, score=0.7, length=3),
                Token(word='가좋아요', b=12, e=16, score=0, length=4)]]

        With tolerance

            >>> scores = {'파스': 0.75, '파스타': 0.7, '좋아': 0.3}
            >>> ltokenizer = LTokenizer(scores)
            >>> ltokenizer.tokenize('파스타가 좋아요 파스타가좋아요', tolerance=0.06)
            $ ['파스타', '가', '좋아', '요', '파스타', '가좋아요']

            >>> ltokenizer.tokenize('파스타가 좋아요 파스타가좋아요', tolerance=0.06, flatten=False)
            $ [[Token(word='파스타', b=0, e=3, score=0.7, length=3),
                Token(word='가', b=3, e=4, score=0, length=1)],
               [Token(word='좋아', b=5, e=7, score=0.3, length=2),
                Token(word='요', b=7, e=8, score=0, length=1)],
               [Token(word='파스타', b=9, e=12, score=0.7, length=3),
                Token(word='가좋아요', b=12, e=16, score=0, length=4)]]
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
                is less than `tolerance`, this tokenizer choose longer one as word
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
                return (token, '', self.scores.get(l, self.unknown_score))

            candidates = [(token[:e], token[e:]) for e in range(2, n + 1)]
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
            tolerance (float) :
                If the difference between the highest and the second highest score
                is less than `tolerance`, this tokenizer choose longer one as word
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
                self.scores.get(token, self.unknown_score),
                length
            )
            return [token]

        scored = self._initialize(s, length, offset)
        tokens = self._find(scored)
        adds = self._add_inter_tokens(s, tokens, offset)
        if result[-1].e != offset + length:
            adds += self._add_last_token(s, tokens, offset)
        if result[0].b != offset:
            adds += self._add_first_token(s, tokens, offset)
        return sorted(tokens + adds, key=lambda x: x.b)

    def _initialize(self, s, length, offset=0):
        max_r = min(length, self.max_len)
        scored = []
        for b in range(0, length - 1):
            for r in range(2, max_r + 1):
                e = b + r
                if e > length:
                    continue
                subtoken = s[b: e]
                if subtoken not in self.scores:
                    continue
                score = self.scores[subtoken]
                scored.append(Token(subtoken, offset + b, offset + e, score, r))
        if not scored:
            return [Token(s, offset, offset + len(s), self.unknown_score, len(s))]
        return sorted(scored, key=lambda x:(-x.score, -x.length, x.b))

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
                if ((token_i.b < token.e and token.b < token_i.e) or
                    (token_i.b < token.e and token_i.e > token.b)):
                    removals.append(i)
            for i in reversed(removals):
                del scored[i]
            num_iter += 1
            if num_iter > 100:
                break
        return sorted(result, key=lambda x: x.b)

    def _add_inter_tokens(self, s, result, offset=0):
        adds = []
        for i, base in enumerate(result[: -1]):
            if base[2] == result[i + 1][1]:
                continue
            b = base[2] - offset
            e = result[i + 1][1] - offset
            sub = s[b: e]
            adds.append(Token(sub, b, e, self.unknown_score, e - b))
        return adds

    def _add_first_token(self, s, result, offset=0):
        b = result[0].b
        sub = s[0: b - offset]
        score = self.scores.get(sub, self.unknown_score)
        return [Token(sub, offset, b, score, b - offset)]
    
    def _add_last_token(self, s, result, offset=0):
        e = result[-1].e
        sub = s[e - offset:]
        if not sub:
            return []
        score = self.scores.get(sub, self.unknown_score)
        return [Token(sub, e, e + len(sub), score, len(sub))]
