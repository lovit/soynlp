import os
import sys
from pprint import pprint
soynlp_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, soynlp_path)

from soynlp.tokenizer import NounMatchTokenizer


def test_nounmatch_tokenizer():
    test_cases = [
        {
            'noun_scores': {'아이': 0.5, '아이오': 0.7, '아이오아이': 0.8, '오이': 0.7},
            'sentence': '아이오아이의아이들은 오이오이를 좋아하는 아이들이오',
            'flatten': True,
            'concat': True,
            'nouns': ['아이오아이', '아이', '오이오이', '아이']
        },
        {
            'noun_scores': {'아이', '아이오', '아이오아이', '오이'},
            'sentence': '아이오아이의아이들은 오이오이를 좋아하는 아이들이오',
            'flatten': True,
            'concat': True,
            'nouns': ['아이오아이', '아이', '오이오이', '아이']
        },
        {
            'noun_scores': {'아이': 1.0, '아이오': 1.0, '아이오아이': 1.0, '오이': 1.0},
            'sentence': '아이오아이의아이들은 오이오이를 좋아하는 아이들이오',
            'flatten': True,
            'concat': False,
            'nouns': ['아이오아이', '아이', '오이', '오이', '아이']
        },
        {
            'noun_scores': {'아이': 1.0, '아이오': 1.0, '아이오아이': 1.0, '오이': 1.0},
            'sentence': '헐아이오아이의아이들은 오이오이를 좋아하는 아이들이오',
            'flatten': True,
            'concat': False,
            'nouns': ['아이오아이', '아이', '오이', '오이', '아이']
        }
    ]

    for test_case in test_cases:
        noun_scores = test_case['noun_scores']
        sentence = test_case['sentence']
        expected = test_case['nouns']

        tokenizer = NounMatchTokenizer(noun_scores)
        nouns = tokenizer.tokenize(
            sentence,
            flatten=test_case['flatten'],
            concat_compound=test_case['concat']
        )

        assert nouns == test_case['nouns']

        print(f'\ninput : {sentence}\nnoun_scores : {noun_scores}')
        print(f'expected  : {expected}')
        print(f'tokenized : {nouns}')
