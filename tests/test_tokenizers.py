import os
import sys
from pprint import pprint
soynlp_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, soynlp_path)

from soynlp.tokenizer import MaxScoreTokenizer, LTokenizer, RegexTokenizer


def test_regex_tokenizer():
    test_cases = [
        {
            "input": 'abc123가나다 alphabet!!3.14한글 hank`s report',
            "tokens": ['abc', '123', '가나다', 'alphabet', '!!', '3.14', '한글', 'hank`s', 'report'],
            "offsets": [0, 3, 6, 10, 18, 20, 24, 27, 34]
        }
    ]

    regex_tokenizer = RegexTokenizer()
    for test_case in test_cases:
        s = test_case['input']
        true_tokens = test_case['tokens']
        true_offsets = test_case['offsets']
        flatten_tokens = regex_tokenizer.tokenize(s, flatten=True)
        result = regex_tokenizer.tokenize(s, flatten=False)
        offsets = [token.b for tokens in result for token in tokens]
        assert true_tokens == flatten_tokens
        assert true_offsets == offsets
        print(f'\ninput : {s}')
        pprint(result)


def test_l_tokenizer():
    pass


def test_maxscore_tokenizer():
    pass
