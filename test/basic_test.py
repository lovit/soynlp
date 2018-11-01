# -*- encoding:utf8 -*-

import argparse
import sys
sys.path.append('../')
import soynlp

def hangle_test():
    from soynlp.hangle import normalize
    from soynlp.hangle import compose
    from soynlp.hangle import decompose
    from soynlp.hangle import character_is_korean
    from soynlp.hangle import character_is_jaum
    from soynlp.hangle import character_is_moum
    from soynlp.hangle import to_base
    from soynlp.hangle import levenshtein
    from soynlp.hangle import jamo_levenshtein
    
    normalized_ = normalize('123이건테스트ab테스트')
    if not (normalized_ == '이건테스트 테스트'):
        raise ValueError('{} should be 이건테스트 테스트'.format(normalized_))
    
    if not (('ㄱ', 'ㅏ', 'ㄴ') == decompose('간')):
        raise ValueError('decompose("간") -> {}'.format(decompose('간')))
    
    if not ((' ', 'ㅗ', ' ') == decompose('ㅗ')):
        raise ValueError('decompose("ㅗ") -> {}'.format(decompose('ㅗ')))
    
    if not (('ㅋ', ' ', ' ') == decompose('ㅋ')):
        raise ValueError('decompose("ㅋ") -> {}'.format(decompose('ㅋ')))
    
    if not ('감' == compose('ㄱ', 'ㅏ', 'ㅁ')):
        raise ValueError("compose('ㄱ', 'ㅏ', 'ㅁ') -> {}".format(compose('ㄱ', 'ㅏ', 'ㅁ')))
    
    if not character_is_korean('감'):
        raise ValueError('character_is_korean("감") -> {}'.format(character_is_korean('감')))
    
    if character_is_korean('a'):
        raise ValueError('character_is_korean("a") -> {}'.format(character_is_korean('a')))
    
    if not character_is_jaum('ㅋ'):
        raise ValueError('character_is_jaum("ㅋ") -> {}'.format(character_is_jaum('ㅋ')))
    
    if character_is_jaum('a'):
        raise ValueError('character_is_jaum("a") -> {}'.format(character_is_jaum('a')))

    if not character_is_moum('ㅗ'):
        raise ValueError('character_is_jaum("ㅗ") -> {}'.format(character_is_jaum('ㅗ')))
    
    if character_is_moum('a'):
        raise ValueError('character_is_jaum("a") -> {}'.format(character_is_jaum('a')))
    
    if not (to_base('ㄱ') == 12593):
        raise ValueError('to_base("ㄱ") -> {}'.format(to_base('ㄱ')))

    if 1 != levenshtein('가나', '가남'):
        raise ValueError("levenshtein('가나', '가남') -> {}".format(levenshtein('가나', '가남')))
    
    if 0.1 != levenshtein('가나', '가남', {('나', '남'):0.1}):
        raise ValueError("levenshtein('가나', '가남', {('나', '남'):0.1}) -> {}".format(levenshtein('가나', '가남', {('나', '남'):0.1})))
    
    if 1/3 != jamo_levenshtein('가나', '가남'):
        raise ValueError("jamo_levenshtein('가나', '가남') -> {}".format(jamo_levenshtein('가나', '가남')))
    
    print('all hangle tests have been successed\n')

def tokenizer_test():
    from soynlp.tokenizer import LTokenizer
    from soynlp.tokenizer import MaxScoreTokenizer
    from soynlp.tokenizer import RegexTokenizer

    regex_tokenizer = RegexTokenizer()
    if not (regex_tokenizer.tokenize('아라랄랄111이히힝ㅇㅇㅠㅠ우유우유ab!') 
            == ['아라랄랄', '111', '이히힝', 'ㅇㅇ', 'ㅠㅠ', '우유우유', 'ab', '!']):
        raise ValueError("regex_tokenizer.tokenize('아라랄랄111이히힝ㅇㅇㅠㅠ우유우유ab!') == {}".format(
            regex_tokenizer.tokenize('아라랄랄111이히힝ㅇㅇㅠㅠ우유우유ab!')))

    ltokenizer = LTokenizer({'데이터':0.4, '데이':0.35, '데이터센터':0.38})
    if not (ltokenizer.tokenize('데이터는 데이터센터의 데이데이') 
            == ['데이터', '는', '데이터', '센터의', '데이', '데이']):
        raise ValueError("ltokenizer.tokenize('데이터는 데이터센터의 데이데이') == {}".format(
            ltokenizer.tokenize('데이터는 데이터센터의 데이데이')))

    if not (ltokenizer.tokenize('데이터는 데이터센터의 데이데이', tolerance=0.05)
            == ['데이터', '는', '데이터센터', '의', '데이', '데이']):
        raise ValueError("ltokenizer.tokenize('데이터는 데이터센터의 데이데이', tolerance=0.05) == {}".format(
            ltokenizer.tokenize('데이터는 데이터센터의 데이데이', tolerance=0.05)))

    maxscore_tokenizer = MaxScoreTokenizer({'데이터':0.4, '데이':0.35, '데이터센터':0.38})
    if not (maxscore_tokenizer.tokenize('데이터는 데이터센터의 데이데이') 
            == ['데이터', '는', '데이터', '센터의', '데이', '데이']):
        raise ValueError("maxscore_tokenizer.tokenize('데이터는 데이터센터의 데이데이') == {}".format(
            maxscore_tokenizer.tokenize('데이터는 데이터센터의 데이데이')))

    print('all tokenizer tests have been successed\n')

def word_extractor_test(corpus_path):
    print('WordExtractor test')
    from soynlp import DoublespaceLineCorpus
    from soynlp.word import WordExtractor

    corpus = DoublespaceLineCorpus(corpus_path, num_doc=1000)
    word_extractor = WordExtractor()
    word_extractor.train(corpus)
    word_scores = word_extractor.extract()

    print('top 20 left frequency * forward cohesion words')
    topwords = sorted(word_scores, key=lambda x: -word_scores[x].cohesion_forward * word_scores[x].leftside_frequency)[:20]
    for word in topwords:
        print('word = {}, cohesion = {}'.format(word, word_scores[word].cohesion_forward))
    print('word extractor test has been done\n\n')

def noun_extractor_test(corpus_path):
    from soynlp import DoublespaceLineCorpus
    from soynlp.noun import LRNounExtractor
    from soynlp.noun import LRNounExtractor_v2
    from soynlp.noun import NewsNounExtractor
    corpus = DoublespaceLineCorpus(corpus_path, num_doc=1000)
    
    # LRNounExtractor
    print('LRNounExtractor test\n{}'.format('-'*40))
    noun_extractor = LRNounExtractor()
    noun_scores = noun_extractor.train_extract(corpus)

    print('{}\n{} words are extracted\ntop 20 frequency * score'.format('-'*30, len(noun_scores)))
    topwords = sorted(noun_scores, key=lambda x: -noun_scores[x].score * noun_scores[x].frequency)[:20]
    for word in topwords:
        print('word = {}, score = {}'.format(word, noun_scores[word].score))

    # NewsNounExtractor
    print('\nNewsNounExtractor test\n{}'.format('-'*40))
    newsnoun_extractor = NewsNounExtractor()
    newsnoun_scores = newsnoun_extractor.train_extract(corpus)

    print('\n{}\n{} words are extracted\ntop 20 frequency * score'.format('-'*30, len(newsnoun_scores)))
    topwords = sorted(newsnoun_scores, key=lambda x: -newsnoun_scores[x].score * newsnoun_scores[x].frequency)[:20]
    for word in topwords:
        print('word = {}, score = {}'.format(word, newsnoun_scores[word].score))
    print('noun extractor test has been done\n\n')

    # LRNounExtractor_v2
    print('\nNounExtractor_v2 test\n{}'.format('-'*40))
    noun_extractor_v2 = LRNounExtractor_v2()
    noun_scores_v2 = noun_extractor_v2.train_extract(corpus)
    noun_scores_v2 = {noun:score for noun, score in noun_scores_v2.items() if len(noun) > 1}

    print('\n{}\n{} words are extracted\ntop 20 frequency * score'.format('-'*30, len(noun_scores_v2)))
    topwords = sorted(noun_scores_v2, key=lambda x: -noun_scores_v2[x].score * noun_scores_v2[x].frequency)[:20]
    for word in topwords:
        print('word = {}, score = {}'.format(word, noun_scores_v2[word].score))
    print('noun extractor test has been done\n\n')

def pos_tagger_test():
    from soynlp.pos import Dictionary
    from soynlp.pos import LRTemplateMatcher
    from soynlp.pos import LREvaluator
    from soynlp.pos import SimpleTagger
    from soynlp.pos import UnknowLRPostprocessor

    pos_dict = {
        'Adverb': {'너무', '매우'}, 
        'Noun': {'너무너무너무', '아이오아이', '아이', '노래', '오', '이', '고양'},
        'Josa': {'는', '의', '이다', '입니다', '이', '이는', '를', '라', '라는'},
        'Verb': {'하는', '하다', '하고'},
        'Adjective': {'예쁜', '예쁘다'},
        'Exclamation': {'우와'}    
    }

    dictionary = Dictionary(pos_dict)

    if not (dictionary.get_pos('아이오아이') == ['Noun']):
        raise ValueError("dictionary.get_pos('아이오아이') = {}".format(dictionary.get_pos('아이오아이')))

    if not (sorted(dictionary.get_pos('이')) == ['Josa', 'Noun']):
        raise ValueError("dictionary.get_pos('이') = {}".format(dictionary.get_pos('이')))

    if not (dictionary.word_is_tag('아이오아이', 'Noun') == True):
        raise ValueError("dictionary.word_is_tag('아이오아이', 'Noun') = {}".format(dictionary.word_is_tag('아이오아이', 'Noun')))

    if not (dictionary.word_is_tag('아이오아이', '명사') == False):
        raise ValueError("dictionary.word_is_tag('아이오아이', '명사') = {}".format(dictionary.word_is_tag('아이오아이', '명사')))

    generator = LRTemplateMatcher(dictionary)
    evaluator = LREvaluator()
    postprocessor = UnknowLRPostprocessor()
    tagger = SimpleTagger(generator, evaluator, postprocessor)
    
    sent = '너무너무너무는아이오아이의노래입니다!!'
    if not (tagger.tag(sent) == [('너무너무너무', 'Noun'), ('는', 'Josa'), ('아이오아이', 'Noun'), ('의', 'Josa'), ('노래', 'Noun'), ('입니다', 'Josa'), ('!!', None)]):
        raise ValueError("tagger.tag(sent) = {}".format(tagger.tag(sent)))

    print('all pos tagger tests have been successed\n\n')

def pmi_test(corpus_path):
    print('PMI test\n{}'.format('-'*40))

    from soynlp import DoublespaceLineCorpus
    from soynlp.word import WordExtractor
    from soynlp.tokenizer import LTokenizer
    from soynlp.vectorizer import sent_to_word_contexts_matrix
    from soynlp.word import pmi

    corpus = DoublespaceLineCorpus(corpus_path, iter_sent=True)
    print('num sents = {}'.format(len(corpus)))

    word_extractor = WordExtractor()
    word_extractor.train(corpus)
    cohesions = word_extractor.all_cohesion_scores()

    l_cohesions = {word:score[0] for word, score in cohesions.items()}
    tokenizer = LTokenizer(l_cohesions)
    print('trained l tokenizer')

    x, idx2vocab = sent_to_word_contexts_matrix(
        corpus,
        windows=3,
        min_tf=10,
        tokenizer=tokenizer, # (default) lambda x:x.split(),
        dynamic_weight=False,
        verbose=True)

    pmi_dok = pmi(
        x,
        min_pmi=0,
        alpha=0.0001,
        verbose=True)

    for pair, pmi in sorted(pmi_dok.items(), key=lambda x:-x[1])[100:110]:
        pair_ = (idx2vocab[pair[0]], idx2vocab[pair[1]])
        print('pmi {} = {:.3f}'.format(pair_, pmi))
    print('computed PMI')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus_path', type=str,
        default='../tutorials/doublespace_line_corpus_sample.txt',
        help='DoublespaceLineCorpus text file')
    parser.add_argument('--pass_hangle', dest='pass_hangle', action='store_true')
    parser.add_argument('--pass_tokenizer', dest='pass_tokenizer', action='store_true')
    parser.add_argument('--pass_word', dest='pass_word', action='store_true')
    parser.add_argument('--pass_noun', dest='pass_noun', action='store_true')
    parser.add_argument('--pass_pos', dest='pass_pos', action='store_true')
    parser.add_argument('--pass_pmi', dest='pass_pmi', action='store_true')
    
    args = parser.parse_args()
    corpus_path = args.corpus_path
    
    if not corpus_path:
        print('You should insert corpus path\nTerminate test code\nSee argument option')
        return

    if not args.pass_hangle:
        hangle_test()
    
    if not args.pass_tokenizer:
        tokenizer_test()
    
    if not args.pass_word:
        word_extractor_test(corpus_path)
    
    if not args.pass_noun:
        noun_extractor_test(corpus_path)
    
    if not args.pass_pos:
        pos_tagger_test()

    if not args.pass_pmi:
        pmi_test(corpus_path)

if __name__ == '__main__':
    main()