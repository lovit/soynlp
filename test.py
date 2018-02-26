# -*- encoding:utf8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import soynlp
from soynlp.word import WordExtractor
from soynlp.normalizer import *


class Sentences:
    def __init__(self, fname):
        self.fname = fname
        self.length = 0

    def __iter__(self):
        with open(self.fname) as f:
            for doc in f:
                doc = only_hangle(doc)
                doc = doc.strip()
                if not doc:
                    continue
                for sent in doc.split('  '):
                    yield sent

    def __len__(self):
        if self.length == 0:
            with open(self.fname) as f:
                for doc in f:
                    doc = doc.strip()
                    if not doc:
                        continue
                    self.length += len(doc.split('  '))
        return self.length


corpus_fname = '/Users/Knight/PycharmProjects/soynlp/data/news.dat'
sentences = Sentences(corpus_fname)
print('num sentences = %d' % len(sentences))

word_extractor = WordExtractor(min_count=100,
                               min_cohesion_forward=0.05,
                               min_right_branching_entropy=0.0)

word_extractor.train(sentences)
words = word_extractor.extract()
print len(words)
print('type: %s\n' % type(words[u'대통령']))
print(words[u'대통령'])

def word_score(score):
    import math
    return (score.cohesion_forward * math.exp(score.right_branching_entropy))

print('단어   (빈도수, cohesion, branching entropy)\n')
for word, score in sorted(words.items(), key=lambda x:word_score(x[1]), reverse=True)[:30]:
    print('%s     (%d, %.3f, %.3f, %.3f)' % (word,
                                   score.leftside_frequency,
                                   score.cohesion_forward,
                                   score.right_branching_entropy,
                                    score.cohesion_forward*
                                    score.right_branching_entropy
                                  ))