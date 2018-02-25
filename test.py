# -*- encoding:utf8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from soynlp.noun import LRNounExtractor

noun_extractor = LRNounExtractor(l_max_length=10,
    r_max_length=7, predictor_fnames=None, verbose=True)

from soynlp.utils import DoublespaceLineCorpus

corpus_fname = '/Users/Knight/PycharmProjects/soynlp/data/news.dat'
sentences = DoublespaceLineCorpus(corpus_fname, iter_sent=True)
print len(sentences)
nouns = noun_extractor.train_extract(sentences, minimum_noun_score=0.3, min_count=100)
print('num nouns = %d' % len(nouns))
print nouns['뉴스']
words = ['김영철', '천안함', '이방카', '평창', '회담']
for word in words:
    print('%s is noun? %r' % (word, word in nouns))

print noun_extractor.is_noun('폐막식', minimum_noun_score=0.3)

top100 = sorted(nouns.items(),
    key=lambda x:-x[1].frequency * x[1].score)[:100]

for i, (word, score) in enumerate(top100):
    if i % 5 == 0:
        print
    print('%-15s (%.2f)' % (word, score.score)),

from soynlp.noun import NewsNounExtractor

noun_extractor = NewsNounExtractor(l_max_length=10,
    r_max_length=7, predictor_fnames=None, verbose=True)
nouns = noun_extractor.train_extract(sentences)
for word in nouns.keys():
    print word, nouns[word]