import pytest
import os
import sys
soynlp_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, soynlp_path)
from soynlp.noun.lr import remove_ambiguous_features, _base_predict


def test_base_predict_helper():
    pos_features = {'로', '의', '로는', '으로', '으로는', '으로써', '이지만'}
    neg_features = {'다고', '고', '자는', '자고', '지만'}
    common_features = {'은', '는'}

    test_cases = [
        {'l': '대학생으',
         'features': [('로', 10), ('로써', 5), ('로는', 3)],
         'refined_features': [],
         'pos': 0,
         'neg': 0,
         'common': 0,
         'end': 0},
        {'l': '대학생',
         'features': [('으로', 10), ('으로써', 5), ('으로는', 3)],
         'refined_features': [('으로', 10), ('으로써', 5), ('으로는', 3)],
         'pos': 18,
         'neg': 0,
         'common': 0,
         'end': 0},
        {'l': '너로',
         'features': [('는', 5), ('', 3)],
         'refined_features': [('', 3)],
         'pos': 0,
         'neg': 0,
         'common': 0,
         'end': 3},
        {'l': '너',
         'features': [('로는', 5), ('는', 3)],
         'refined_features': [('로는', 5), ('는', 3)],
         'pos': 5,
         'neg': 0,
         'common': 3,
         'end': 0},
        {'l': '관계자',
         'features': [('는', 10), ('로', 5), ('이지만', 5)],
         'refined_features': [('로', 5), ('이지만', 5)],
         'pos': 10,
         'neg': 0,
         'common': 0,
         'end': 0},
        {'l': '관계',
         'features': [('자는', 10), ('자로', 5), ('는', 20), ('의', 25), ('자이지만', 5)],
         'refined_features': [('자는', 10), ('자로', 5), ('는', 20), ('의', 25), ('자이지만', 5)],
         'pos': 25,
         'neg': 10,
         'common': 20,
         'end': 0},
        {'l': '하자',
         'features': [('는', 10), ('고', 5)],
         'refined_features': [],
         'pos': 0,
         'neg': 0,
         'common': 0,
         'end': 0},
        {'l': '하',
         'features': [('자는', 10), ('자고', 5)],
         'refined_features': [('자는', 10), ('자고', 5)],
         'pos': 0,
         'neg': 15,
         'common': 0,
         'end': 0},
        {'l': '가고있다',
         'features': [('고', 10), ('지만', 3)],
         'refined_features': [('지만', 3)],
         'pos': 0,
         'neg': 3,
         'common': 0,
         'end': 0},
        {'l': '가고있',
         'features': [('다고', 10), ('지만', 5), ('다지만', 3)],
         'refined_features': [('다고', 10), ('지만', 5), ('다지만', 3)],
         'pos': 0,
         'neg': 15,
         'common': 0,
         'end': 0}
    ]
    for test_case in test_cases:
        word = test_case['l']
        features = test_case['features']
        refined = remove_ambiguous_features(word, features, pos_features, neg_features)
        pos, common, neg, unk, end = _base_predict(word, refined, pos_features, neg_features, common_features)
        print(f'\nword: {word}\n - before: {features}\n - after:  {refined}')
        print(f' - pos={pos}, neg={neg}, common={common}, end={end}')

        assert sorted(refined) == sorted(test_case['refined_features'])
        assert test_case['pos'] == pos
        assert test_case['neg'] == neg
        assert test_case['common'] == common
        assert test_case['end'] == end
