import os
from pprint import pprint

from soynlp.tokenizer import MaxScoreTokenizer, LTokenizer, RegexTokenizer
from soynlp.word import WordExtractor
from soynlp.utils import DoublespaceLineCorpus
from tqdm import tqdm


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

        words = regex_tokenizer.tokenize(sentence, return_words=True)
        result = regex_tokenizer.tokenize(sentence, return_words=False)
        offsets = [token.begin for token in result]

        assert true_words == words
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

        tokens = ltokenizer(sentence, return_words=False, tolerance=tolerance, remove_r=remove_r)
        words = ltokenizer.tokenize(sentence, tolerance=tolerance, remove_r=remove_r)

        assert words == true_words
        print(f'\ninput : {sentence}\ntolerance : {tolerance}\nremove_r : {remove_r}')
        print(f'scores : {scores}')
        print(f'words : {words}')
        pprint(tokens)


def test_maxscore_tokenizer():
    test_cases = [
        {
            'scores': {'파스': 0.65, '파스타': 0.7, '좋아': 0.3, '스타': 0.65},
            'input': '파스타짱좋아 파스타짱 짱좋아요 짱짱맨 파스좋아!',
            'words': ['파스타', '짱', '좋아', '파스타', '짱', '짱', '좋아', '요', '짱짱맨', '파스', '좋아', '!'],
            'begin': [0, 3, 4, 7, 10, 12, 13, 15, 17, 21, 23, 25]
        }
    ]

    for test_case in test_cases:
        scores = test_case['scores']
        sentence = test_case['input']
        true_words = test_case['words']
        true_begin = test_case['begin']

        tokenizer = MaxScoreTokenizer(scores)
        tokens = tokenizer.tokenize(sentence, return_words=False)
        words = [t.word for t in tokens]
        begin = [t.begin for t in tokens]

        assert words == true_words
        assert begin == true_begin
        print(f'\ninput : {sentence}\nscores : {scores}')
        print(f'words : {words}')
        pprint(tokens)


def test_maxscore_tokenizer_usage():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
    train_data = f'{root_dir}/data/2016-10-20.txt'
    train_zip_data = f'{root_dir}/data/2016-10-20.zip'
    if not os.path.exists(train_data):
        assert os.path.exists(train_zip_data)
        with zipfile.ZipFile(train_zip_data, 'r') as zip_ref:
            zip_ref.extractall(f'{root_dir}/data/')
    assert os.path.exists(train_data)

    with open(train_data, encoding='utf-8') as f:
        sents = [sent.strip() for doc in f for sent in doc.split('  ')]
    sents = [sent for sent in sents if sent][:10000]
    word_extractor = WordExtractor()
    word_extractor.train(sents)
    cohesion_scores = word_extractor.all_cohesion_scores()
    cohesion_scores = {l: cohesion for l, (cohesion, _) in cohesion_scores.items()}
    tokenizer = MaxScoreTokenizer(cohesion_scores)

    for i, sentence in enumerate(tqdm(sents, desc='MaxScoreTokenizer usage test', total=len(sents))):
        try:
            words = tokenizer.tokenize(sentence)
            assert ''.join(words) == sentence.replace(' ', '')
        except:
            raise RuntimeError(f'{i}: {sentence}')
            break
