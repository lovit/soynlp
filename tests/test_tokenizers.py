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
            "words": ['abc', '123', '가나다', 'alphabet', '!!', '3.14', '한글', 'hank`s', 'report'],
            "offsets": [0, 3, 6, 10, 18, 20, 24, 27, 34]
        }
    ]

    regex_tokenizer = RegexTokenizer()

    for test_case in test_cases:
        sentence = test_case['input']
        true_words = test_case['words']
        true_offsets = test_case['offsets']

        flatten_words = regex_tokenizer.tokenize(sentence, flatten=True)
        result = regex_tokenizer.tokenize(sentence, flatten=False)
        offsets = [token.b for tokens in result for token in tokens]

        assert true_words == flatten_words
        assert true_offsets == offsets
        print(f'\ninput : {sentence}')
        pprint(result)


def test_l_tokenizer():
    test_cases = [
        {
            'scores': {'파스': 0.65, '파스타': 0.7, '좋아': 0.3},
            'input': '파스타가 좋아요 파스타가좋아요',
            'tolerance': 0.0,
            'words': ['파스타', '가', '좋아', '요', '파스타', '가좋아요'],
            'remove_r': False
        },
        {
            'scores': {'파스': 0.65, '파스타': 0.7, '좋아': 0.3},
            'input': '파스타가 좋아요 파스타가좋아요',
            'tolerance': 0.0,
            'words': ['파스타', '좋아', '파스타'],
            'remove_r': True
        },
        {
            'scores': {'파스': 0.75, '파스타': 0.7, '좋아': 0.3},
            'input': '파스타가 좋아요 파스타가좋아요',
            'tolerance': 0.06,
            'words': ['파스타', '가', '좋아', '요', '파스타', '가좋아요'],
            'remove_r': False
        },
        {
            'scores': {'파스': 0.75, '파스타': 0.7, '좋아': 0.3},
            'input': '파스타가 좋아요 파스타가좋아요',
            'tolerance': 0.0,
            'words': ['파스', '타가', '좋아', '요', '파스', '타가좋아요'],
            'remove_r': False
        }
    ]
    for test_case in test_cases:
        scores = test_case['scores']
        ltokenizer = LTokenizer(scores)
        true_words = test_case['words']
        tolerance = test_case['tolerance']
        sentence = test_case['input']
        remove_r = test_case['remove_r']

        tokens = ltokenizer(sentence, flatten=False, tolerance=tolerance, remove_r=remove_r)
        flatten_words = ltokenizer.tokenize(sentence, tolerance=tolerance, remove_r=remove_r)

        assert flatten_words == true_words
        print(f'\ninput : {sentence}\ntolerance : {tolerance}\nremove_r : {remove_r}')
        print(f'scores : {scores}')
        print(f'flatten words : {flatten_words}')
        pprint(tokens)


def test_maxscore_tokenizer():
    test_cases = [
        {
            'scores': {'파스': 0.65, '파스타': 0.7, '좋아': 0.3, '스타': 0.65},
            'input': '파스타짱좋아 파스타짱 짱좋아요 짱짱맨 파스좋아!',
            'words': ['파스타', '짱', '좋아', '파스타', '짱', '짱', '좋아', '요', '짱짱맨', '파스', '좋아', '!'],
            'b': [0, 3, 4, 7, 10, 12, 13, 15, 17, 21, 23, 25]
        }
    ]

    for test_case in test_cases:
        scores = test_case['scores']
        sentence = test_case['input']
        true_words = test_case['words']
        true_b = test_case['b']

        tokenizer = MaxScoreTokenizer(scores)
        nested_tokens = tokenizer.tokenize(sentence, flatten=False)
        flatten_tokens = [t for tokens in nested_tokens for t in tokens]
        words = [t.word for t in flatten_tokens]
        b = [t.b for t in flatten_tokens]

        assert words == true_words
        assert b == true_b
        print(f'\ninput : {sentence}\nscores : {scores}')
        print(f'flatten words : {words}')
        pprint(flatten_tokens)
