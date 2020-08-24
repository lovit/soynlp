import os
import sys
from pprint import pprint
soynlp_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, soynlp_path)

from soynlp.noun import LRNounExtractor
from soynlp.tokenizer import NounMatchTokenizer


def test_nounmatch_tokenizer():
    test_cases = [
        {
            'noun_scores': {'아이': 0.5, '아이오': 0.7, '아이오아이': 0.8, '오이': 0.7},
            'sentence': '아이오아이의아이들은 오이오이를 좋아하는 아이들이오',
            'flatten': True,
            'concat': True,
            'must_be_L': False,
            'nouns': ['아이오아이', '아이', '오이오이', '아이']
        },
        {
            'noun_scores': {'아이', '아이오', '아이오아이', '오이'},
            'sentence': '아이오아이의아이들은 오이오이를 좋아하는 아이들이오',
            'flatten': True,
            'concat': True,
            'must_be_L': False,
            'nouns': ['아이오아이', '아이', '오이오이', '아이']
        },
        {
            'noun_scores': {'아이': 1.0, '아이오': 1.0, '아이오아이': 1.0, '오이': 1.0},
            'sentence': '아이오아이의아이들은 오이오이를 좋아하는 아이들이오',
            'flatten': True,
            'concat': False,
            'must_be_L': False,
            'nouns': ['아이오아이', '아이', '오이', '오이', '아이']
        },
        {
            'noun_scores': {'아이': 1.0, '아이오': 1.0, '아이오아이': 1.0, '오이': 1.0},
            'sentence': '헐아이오아이의아이들은 오이오이를 좋아하는 아이들이오',
            'flatten': True,
            'concat': False,
            'must_be_L': False,
            'nouns': ['아이오아이', '아이', '오이', '오이', '아이']
        },
        {
            'noun_scores': {'아이': 1.0, '아이오': 1.0, '아이오아이': 1.0, '오이': 1.0},
            'sentence': '헐아이오아이의아이들은 오이오이를 좋아하는 아이들이오',
            'flatten': True,
            'concat': False,
            'must_be_L': True,
            'nouns': ['오이', '아이']
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
            concat_compound=test_case['concat'],
            must_be_L=test_case['must_be_L']
        )

        assert nouns == test_case['nouns']

        print(f'\ninput : {sentence}\nnoun_scores : {noun_scores}')
        print(f'expected  : {expected}')
        print(f'tokenized : {nouns}')


def test_noun_tokenizer_usage():
    train_data = f'{soynlp_path}/data/2016-10-20.txt'
    train_zip_data = f'{soynlp_path}/data/2016-10-20.zip'
    if not os.path.exists(train_data):
        assert os.path.exists(train_zip_data)
        with zipfile.ZipFile(train_zip_data, 'r') as zip_ref:
            zip_ref.extractall(f'{soynlp_path}/data/')
    assert os.path.exists(train_data)

    noun_extractor = LRNounExtractor()
    _ = noun_extractor.extract(train_data)
    noun_tokenizer = noun_extractor.get_noun_tokenizer()
    sentence = '네이버의 뉴스기사를 이용하여 학습한 모델예시입니다'
    assert noun_tokenizer.tokenize(sentence) == ['네이버', '뉴스기사', '이용', '학습', '모델예시']
    print(f'\ninput : {sentence}\nnouns : {noun_tokenizer.tokenize(sentence)}')
