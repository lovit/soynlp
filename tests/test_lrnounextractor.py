import os
import sys
import zipfile
soynlp_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, soynlp_path)

from soynlp.noun.lr import remove_ambiguous_features, _base_predict, base_predict
from soynlp.noun import LRNounExtractor


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
         'refined_features': [],
         'pos': 0,
         'neg': 0,
         'common': 0,
         'end': 0},
        {'l': '너',
         'features': [('로는', 5), ('는', 3)],
         'refined_features': [('는', 3), ('로는', 5)],
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
        refined, _ = remove_ambiguous_features(word, features, pos_features, neg_features, common_features)
        pos, common, neg, unk, end = _base_predict(word, refined, pos_features, neg_features, common_features)
        print(f'\nword: {word}\n - before: {features}\n - after:  {refined}')
        print(f' - pos={pos}, neg={neg}, common={common}, end={end}')

        assert sorted(refined) == sorted(test_case['refined_features'])
        assert test_case['pos'] == pos
        assert test_case['neg'] == neg
        assert test_case['common'] == common
        assert test_case['end'] == end


def test_base_predict():
    pos_features = {'로', '에', '의', '로는', '에서', '으로', '으로는', '으로써', '이지만'}
    neg_features = {'다고', '고', '자는', '자고', '지만'}
    common_features = {'은', '는'}

    test_cases = [
        {'l': '대학생으', 'r': [('로', 10), ('로써', 5), ('로는', 3)], 'support': 0, 'score': 0},
        {'l': '대학생', 'r': [('으로', 10), ('으로써', 5), ('으로는', 3)], 'support': 18, 'score': 1.0},
        {'l': '너로', 'r': [('는', 5), ('', 3)], 'support': 0, 'score': 0},
        {'l': '너', 'r': [('로는', 5), ('는', 3)], 'support': 8, 'score': 1.0},
        {'l': '관계자', 'r': [('는', 10), ('로', 5), ('이지만', 5)], 'support': 10, 'score': 1.0},
        {'l': '관계', 'r': [('자는', 10), ('자로', 5), ('는', 20), ('의', 25), ('자이지만', 5)], 'support': 45, 'score': 0.42857142857142855},
        {'l': '하자', 'r': [('는', 10), ('고', 5)], 'support': 0, 'score': 0},
        {'l': '하', 'r': [('자는', 10), ('자고', 5)], 'support': 15, 'score': -1.0},
        {'l': '가고있다', 'r': [('고', 10), ('지만', 3)], 'support': 3, 'score': 0},
        {'l': '가고있', 'r': [('다고', 10), ('지만', 5), ('다지만', 3)], 'support': 15, 'score': -1.0},
        {'l': '경찰국', 'r': [('은', 1), ('에', 1), ('에서', 1)], 'support': 3, 'score': 1.0},
        {'l': '아이웨딩', 'r': [('', 90), ('은', 3), ('측은', 1)], 'support': 93, 'score': 0.9893617021276596},
        {'l': '아이엠텍', 'r': [('은', 2), ('', 2)], 'support': 4, 'score': 1.0}
    ]
    for test_case in test_cases:
        word = test_case['l']
        features = test_case['r']
        support, score = base_predict(word, features, pos_features, neg_features, common_features)
        print(f'{word} : {features}\n  support={support}, score = {score}\n')
        assert support == test_case['support']
        assert score == test_case['score']


def test_usage():
    test_cases = {
        '아이오아이': {'frequency': 127, 'score': 1.0},
        '아이디': {'frequency': 59, 'score': 1.0},
        '아이디어': {'frequency': 142, 'score': 1.0},
    }
    train_data = f'{soynlp_path}/data/2016-10-20.txt'
    train_zip_data = f'{soynlp_path}/data/2016-10-20.zip'
    if not os.path.exists(train_data):
        assert os.path.exists(train_zip_data)
        with zipfile.ZipFile(train_zip_data, 'r') as zip_ref:
            zip_ref.extractall(f'{soynlp_path}/data/')
    assert os.path.exists(train_data)

    noun_extractor = LRNounExtractor()
    nouns = noun_extractor.extract(
        train_data,
        exclude_numbers=True,
        exclude_syllables=True,
        extract_compounds=True
    )
    for noun in test_cases:
        assert nouns[noun].frequency == test_cases[noun]['frequency']
        assert nouns[noun].score == test_cases[noun]['score']
        print(f'noun={noun}, scores={nouns[noun]}')
