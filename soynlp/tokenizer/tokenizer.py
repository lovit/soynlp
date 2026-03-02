import re
from collections import namedtuple


class Token(namedtuple("Token", "word begin end score length eojeol_id")):
    """collections.namedtuple class

    Args:
        word (str) : surfacial form of word
        begin (int) : begin position of `word` in the input sentence
        end (int) : end position of `word` in the input sentence
        score (float) : word score
        length (int) : word length. It must be equal with `end` - `begin`
        eojeol_id (int) : index of eojeol in a sentence
    """

    def __repr__(self):
        return f"Token({self.word}, score={self.score}, position=({self.begin}, {self.end}), eojeol_id={self.eojeol_id})"


class RegexTokenizer:
    """Split sentence based on type of characters and regex pattern."""

    def __init__(self, pipelines=None):
        if pipelines is None:
            pipelines = self._default_pipelines()
        self.pipelines = pipelines
        self.doublewhite_pattern = re.compile(r"\s+")

    def _default_pipelines(self):
        return [
            re.compile(r"[-+]?\d*[\.]?[\d]+|[-+]?\d+", re.UNICODE),  # number
            re.compile(r"[가-힣]+", re.UNICODE),  # Korean
            re.compile(r"[ㄱ-ㅎ]+", re.UNICODE),  # jaum
            re.compile(r"[ㅏ-ㅣ]+", re.UNICODE),  # moum
            re.compile(r"[a-zA-ZÀ-ÿ]+[[`']{1,1}s]*|[a-zA-ZÀ-ÿ]+", re.UNICODE),  # Alphabet
        ]

    def __call__(self, sentence, return_words=True):
        return self.tokenize(sentence, return_words)

    def tokenize(self, sentence, return_words=True):
        """Split sentence based on type of characters and regex pattern.

        Args:
            sentence (str) : input string
            return_words (Boolean) :
                If True, it returns tokens as form of list of str
                Otherwise, it returns list of `Token`

        Returns:
            tokens (list of str or list of Token)
        """
        offset = 0
        tokens = []
        for eojeol_id, token in enumerate(sentence.split()):
            tokens += self._tokenize(token, offset, eojeol_id)
            offset += len(token) + 1
        if return_words:
            tokens = [token.word for token in tokens]
        return tokens

    def _tokenize(self, s, offset=0, eojeol_id=0):
        for pattern in self.pipelines:
            founds = pattern.findall(s)
            if not founds:
                continue
            found = founds.pop(0)
            len_found = len(found)

            s_ = ""
            begin = 0
            for i, char in enumerate(s):
                if begin > i:
                    continue
                if s[i : i + len_found] == found:
                    s_ += " %s " % s[i : i + len_found]
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
        words = self.doublewhite_pattern.sub(" ", s).strip().split()
        r = len(words[0])
        tokens = [Token(words[0], 0 + offset, r + offset, 1, r, eojeol_id)]
        begin = tokens[0].end
        for word in words[1:]:
            r = len(word)
            tokens.append(Token(word, begin, begin + r, 1, r, eojeol_id))
            begin += r
        return tokens


class LTokenizer:
    """It finds the most word-like substring which is positioned on the left-side
    in given Eojeol.

    Args:
        scores ({str: float}) : {word: score}
        unknown_score (float) : unknown word score
    """

    def __init__(self, scores, unknown_score=0.0):
        self.scores = scores
        self.unknown_score = unknown_score

    def __call__(self, sentence, tolerance=0.0, return_words=True, remove_r=False):
        return self.tokenize(sentence, tolerance, return_words, remove_r)

    def tokenize(self, sentence, tolerance=0.0, return_words=True, remove_r=False):
        """
        Args:
            sentence (str) : input string
            tolerance (float) :
                If the difference between the highest and the second highest score
                is less than `tolerance`, this tokenizer chooses longer one as word
            return_words (Boolean) :
                If True, it returns tokens as form of list of str
                Otherwise, it returns list of `Token`
            remove_r (Boolean) :
                If True, it returns only L parts

        Returns:
            tokens (list of str or list of Token)
        """

        def token_to_lr(token):
            """Returns: (score of L, L, R)"""
            n = len(token)
            if n <= 2:
                return (self.scores.get(token, self.unknown_score), token, "")

            candidates = [(token[:end], token[end:]) for end in range(2, n + 1)]
            candidates = [(self.scores.get(l, self.unknown_score), l, r) for l, r in candidates]  # noqa: E741
            if tolerance > 0:
                max_score = max([c[0] for c in candidates])
                candidates = [c for c in candidates if (max_score - c[0]) <= tolerance]
                best = sorted(candidates, key=lambda x: len(x[1]), reverse=True)[0]
            else:
                best = sorted(candidates, key=lambda x: (x[0], len(x[1])), reverse=True)[0]
            return best

        offset = 0
        tokens = []
        for eojeol_id, s in enumerate(sentence.split()):
            score, l, r = token_to_lr(s)  # noqa: E741
            len_l, len_r = len(l), len(r)
            tokens.append(
                [
                    Token(l, offset, offset + len_l, score, len_l, eojeol_id),
                    Token(r, offset + len_l, offset + len_l + len_r, 0, len_r, eojeol_id),
                ]
            )
            offset += len_l + len_r + 1

        if remove_r:
            tokens = [l.word for l, r in tokens]  # noqa: E741
        else:
            tokens = [lrtoken for lr in tokens for lrtoken in lr]

        if (return_words) and (not remove_r):
            tokens = [token.word for token in tokens if token.length > 0]

        return tokens


class MaxScoreTokenizer:
    """It finds the most word-like substring in a given Eojeol regardless the position.

    Args:
        scores ({str: float}) : {word: word_score}
        max_length (int) : maximum length of L part word
        unknown_score (float) : score of unknown word
    """

    def __init__(self, scores, max_length=10, unknown_score=0.0):
        self.scores = scores
        self.max_len = max_length
        self.unknown_score = unknown_score

    def __call__(self, sentence, return_words=True):
        return self.tokenize(sentence, return_words)

    def tokenize(self, sentence, return_words=True):
        """
        Args:
            sentence (str) : input string
            return_words (Boolean) :
                If True, it returns tokens as form of list of str
                Otherwise, it returns list of `Token`

        Returns:
            tokens (list of str or list of Token)
        """
        offset = 0
        tokens = []
        for eojeol_id, s in enumerate(sentence.split()):
            tokens += self._tokenize_eojeol(s, offset, eojeol_id)
            offset += len(s) + 1
        if return_words:
            tokens = [token.word for token in tokens]
        return tokens

    def _tokenize_eojeol(self, s, offset, eojeol_id):
        length = len(s)
        if length <= 2:
            token = Token(s, offset, offset + length, self.scores.get(s, self.unknown_score), length, eojeol_id)
            return [token]

        scored = self._prepare_word_candidates(s, length, offset, eojeol_id)
        tokens = self._select_best_words(scored)
        adds = self._add_inter_tokens(s, tokens, offset, eojeol_id)
        if tokens[-1].end != offset + length:
            adds += self._add_last_token(s, tokens, offset, eojeol_id)
        if tokens[0].begin != offset:
            adds += self._add_first_token(s, tokens, offset, eojeol_id)
        return sorted(tokens + adds, key=lambda x: x.begin)

    def _prepare_word_candidates(self, s, length, offset=0, eojeol_id=0):
        max_r = min(length, self.max_len)
        scored = []
        for begin in range(0, length - 1):
            for r in range(2, max_r + 1):
                end = begin + r
                if end > length:
                    continue
                subtoken = s[begin:end]
                if subtoken not in self.scores:
                    continue
                score = self.scores[subtoken]
                scored.append(Token(subtoken, offset + begin, offset + end, score, r, eojeol_id))
        if not scored:
            return [Token(s, offset, offset + len(s), self.unknown_score, len(s), eojeol_id)]
        return sorted(scored, key=lambda x: (-x.score, -x.length, x.begin))

    def _select_best_words(self, scored):
        result = []
        num_iter = 0
        while scored:
            token = scored.pop(0)
            result.append(token)
            if not scored:
                break
            removals = []
            for i, token_i in enumerate(scored):
                if (token_i.begin < token.end and token.begin < token_i.end) or (
                    token_i.begin < token.end and token_i.end > token.begin
                ):
                    removals.append(i)
            for i in reversed(removals):
                del scored[i]
            num_iter += 1
            if num_iter > 100:
                break
        return sorted(result, key=lambda x: x.begin)

    def _add_inter_tokens(self, s, tokens, offset=0, eojeol_id=0):
        adds = []
        for i, token in enumerate(tokens[:-1]):
            if token.end == tokens[i + 1].begin:
                continue
            begin = token.end - offset
            end = tokens[i + 1].begin - offset
            sub = s[begin:end]
            adds.append(Token(sub, offset + begin, offset + end, self.unknown_score, end - begin, eojeol_id))
        return adds

    def _add_first_token(self, s, tokens, offset=0, eojeol_id=0):
        begin = tokens[0].begin
        sub = s[0 : begin - offset]
        score = self.scores.get(sub, self.unknown_score)
        return [Token(sub, offset, begin, score, begin - offset, eojeol_id)]

    def _add_last_token(self, s, tokens, offset=0, eojeol_id=0):
        end = tokens[-1].end
        sub = s[end - offset :]
        if not sub:
            return []
        score = self.scores.get(sub, self.unknown_score)
        return [Token(sub, end, end + len(sub), score, len(sub), eojeol_id)]


class NounMatchTokenizer(MaxScoreTokenizer):
    """NounMatchTokenizer recognizes nouns from input sentence.

    Args:
        noun_scores ({str: float}) : {noun: noun_score}
    """

    def __init__(self, noun_scores):
        if isinstance(noun_scores, (list, set, tuple)):
            noun_scores = {noun: 1.0 for noun in noun_scores}
        super().__init__(noun_scores)

    def __call__(self, sentence, return_words=True, concat_compound=True):
        return self.tokenize(sentence, return_words, concat_compound)

    def tokenize(self, sentence, return_words=True, concat_compound=True, must_be_L=False):
        """
        Args:
            sentence (str) : input string
            return_words (Boolean) :
                If True, it returns tokens as form of list of str
                Otherwise, it returns list of `Token`
            concat_compound (Boolean) :
                If True, it concatenates consecutive nouns into one compound noun.
            must_be_L (Boolean) :
                If True, it remains nouns which position left-side on eojeol.

        Returns:
            tokens (list of str or list of Token)
        """

        def concatenate(eojeol, tokens, offset, eojeol_id):
            concats, begin, end, score = [], offset, offset, 0
            for token in tokens:
                if end == token.begin:
                    end, score = token.end, max(score, token.score)
                else:
                    concats.append(Token(eojeol[begin - offset : end - offset], begin, end, score, end - begin, eojeol_id))
                    begin, end = token.begin, token.end
            if end > begin:
                concats.append(Token(eojeol[begin - offset : end - offset], begin, end, score, end - begin, eojeol_id))
            return concats

        offset = 0
        tokens = []
        for eojeol_id, s in enumerate(sentence.split()):
            nouns = self._tokenize_eojeol(s, offset, eojeol_id)
            nouns = [noun for noun in nouns if noun.score > 0]
            if concat_compound:
                nouns = concatenate(s, nouns, offset, eojeol_id)
            if must_be_L and nouns:
                if nouns[0].begin != offset:
                    nouns = []
                else:
                    nouns = nouns[:1]
            tokens += nouns
            offset += len(s) + 1
        if return_words:
            tokens = [token.word for token in tokens]
        return tokens
