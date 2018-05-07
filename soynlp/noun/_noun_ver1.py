# -*- encoding:utf8 -*-

from collections import defaultdict, namedtuple
import sys
from soynlp.word import WordExtractor

NounScore = namedtuple('NounScore', 'frequency score known_r_ratio')

class LRNounExtractor:

    def __init__(self, l_max_length=10, r_max_length=7,
            predictor_fnames=None, verbose=True, min_num_of_features=1):
        
        self.coefficient = {}
        self.verbose = verbose
        self.l_max_length = l_max_length
        self.r_max_length = r_max_length
        self.lrgraph = None
        self.words = None
        self._wordset_l_counter = {}
        self.min_num_of_features = min_num_of_features
        
        if not predictor_fnames:
            import os
            directory = '/'.join(os.path.abspath(__file__).replace('\\', '/').split('/')[:-2])
            predictor_fnames = ['%s/trained_models/noun_predictor_sejong' % directory]
            if verbose:
                print('used default noun predictor; Sejong corpus predictor')
            
        for fname in predictor_fnames:
            if verbose:
                print('used %s' % fname.split('/')[-1])
            self._load_predictor(fname)
        if verbose:
            print('All %d r features was loaded' % len(self.coefficient))
        
    def _load_predictor(self, fname):
        try:
            if sys.version_info.major == 2:
                f = open(fname)
            else:
                f = open(fname, encoding='utf-8')
            try:
                for num_line, line in enumerate(f):
                    r, score = line.strip().split('\t')
                    score = float(score)
                    if r in self.coefficient:
                        self.coefficient[r] = max(self.coefficient[r], score)
                    else:
                        self.coefficient[r] = score
            except Exception as e:
                print('predictor parsing error line {} = {}'.format(num_line+1, line))
            finally:
                f.close()
        except Exception as e:
            print(e)
    
    def train_extract(self, sents, minimum_noun_score=0.5, min_count=5,
            noun_candidates=None):
        
        self.train(sents, min_count)
        return self.extract(minimum_noun_score, min_count, noun_candidates)

    def train(self, sents, min_count=5):
        wordset_l, wordset_r = self._scan_vocabulary(sents)
        self.lrgraph = self._build_lrgraph(sents, wordset_l, wordset_r)
        self.words = wordset_l
    
    def _scan_vocabulary(self, sents, min_count=5):
        """
        Parameters
        ----------
            sents: list-like iterable object which has string
            
        It computes subtoken frequency first. 
        After then, it builds lr-graph with sub-tokens appeared at least min count
        """
        
        _ckpt = int(len(sents) / 40)
        
        wordset_l = defaultdict(lambda: 0)
        wordset_r = defaultdict(lambda: 0)
        
        for i, sent in enumerate(sents):
            for token in sent.split(' '):
                if not token:
                    continue
                token_len = len(token)
                for i in range(1, min(self.l_max_length, token_len)+1):
                    wordset_l[token[:i]] += 1
                for i in range(1, min(self.r_max_length, token_len)):
                    wordset_r[token[-i:]] += 1
            if self.verbose and (i % _ckpt == 0):
                args = ('#' * int(i/_ckpt), '-' * (40 - int(i/_ckpt)), 100.0 * i / len(sent), '%')
                sys.stdout.write('\rscanning: %s%s (%.3f %s)' % args)
        
        self._wordset_l_counter = {w:f for w,f in wordset_l.items() if f >= min_count}
        wordset_l = set(self._wordset_l_counter.keys())
        wordset_r = {w for w,f in wordset_r.items() if f >= min_count}
        
        if self.verbose:
            print('\rscanning completed')
            print('(L,R) has (%d, %d) tokens' % (len(wordset_l), len(wordset_r)))

        return wordset_l, wordset_r
    
    def _build_lrgraph(self, sents, wordset_l, wordset_r):
        _ckpt = int(len(sents) / 40)
        lrgraph = defaultdict(lambda: defaultdict(lambda: 0))
        
        for i, sent in enumerate(sents):
            for token in sent.split():
                if not token:
                    continue
                n = len(token)
                for i in range(1, min(self.l_max_length, n)+1):
                    l = token[:i]
                    r = token[i:]
                    if not (l in wordset_l):
                        continue
                    if (len(r) > 0) and not (r in wordset_r):
                        continue
                    lrgraph[l][r] += 1

            if self.verbose and (i % _ckpt == 0):
                args = ('#' * int(i/_ckpt), '-' * (40 - int(i/_ckpt)), 100.0 * i / len(sents), '%')
                sys.stdout.write('\rbuilding lr-graph: %s%s (%.3f %s)' % args)
        if self.verbose:                       
            sys.stdout.write('\rbuilding lr-graph completed')
        lrgraph = {l:{r:f for r,f in rdict.items()} for l,rdict in lrgraph.items()}
        return lrgraph
    
    def extract(self, minimum_noun_score=0.5, min_count=5, noun_candidates=None):
        if not noun_candidates:
            noun_candidates = self.words

        nouns = {}
        for word in sorted(noun_candidates, key=lambda w:len(w)):
            if len(word) <= 1:
                continue

            score = self.predict(word, nouns)

            if score[0] < minimum_noun_score:
                continue
            nouns[word] = score
            
        nouns = self._postprocess(nouns, minimum_noun_score, min_count)
        nouns = {word:NounScore(self._wordset_l_counter.get(word, 0), score[0], score[1]) for word, score in nouns.items()}
        return nouns
    
    def _get_r_features(self, word):
        features = self.lrgraph.get(word, {})
        # remove empty str r only in features
        features = {k:v for k,v in features.items() if k}
        return features
    
    def _get_subword_score(self, word, minimum_noun_score, nouns):
        subword_scores = {}
        for e in range(1, len(word)):
            subword = word[:e]
            suffix = word[e:]
            # Add word if compound
            if (subword in nouns) and (suffix in nouns):
                score1 = nouns[subword]
                score2 = nouns[suffix]
                subword_scores[subword] = max(score1, score2)
            elif (subword in nouns) and (self.coefficient.get(suffix,0.0) > minimum_noun_score):
                subword_scores[subword] = (self.coefficient.get(suffix,0.0), 0)
        if not subword_scores:
            return (0.0, 0)
        return sorted(subword_scores.items(), key=lambda x:-x[1][0])[0][1]

    def is_noun(self, word, minimum_noun_score=0.5):
        return self.predict(word)[0] >= minimum_noun_score

    def predict(self, word, minimum_noun_score=0.5, nouns=None):
        features = self._get_r_features(word)

        # (감사합니다 + 만) 처럼 뒤에 등장하는 R 의 종류가 한가지 뿐이면 제대로 된 판단이 되지 않음
        if len(features) > self.min_num_of_features:
            score = self._predict(features, word)
        else:
            if nouns is None:
                nouns = {}
            score = self._get_subword_score(word, minimum_noun_score, nouns)

        return score

    def _predict(self, features, word):
        
        def exist_longer_r_feature(features, word):
            for e in range(len(word)-1, -1, -1):
                suffix = word[e:] + features
                if suffix in self.coefficient: 
                    return True
            return False
        
        """Parameters
        ----------
            features: dict
                예시: {을: 35, 는: 22, ...}
        """
        
        score = 0
        norm = 0
        unknown = 0
        
        for r, freq in features.items():
            if r in self.coefficient:
                if not exist_longer_r_feature(r, word):  
                    score += freq * self.coefficient[r]
                    norm += freq
            else:
                unknown += freq
        
        return (0 if norm == 0 else score / norm, 
                0 if (norm + unknown == 0) else norm / (norm + unknown))
    
    def _postprocess(self, nouns, minimum_noun_score, min_count):
        removals = set()
        for word in nouns:
            if len(word) <= 2:
                continue
            if word[-1] == '.' or word[-1] == ',':
                removals.add(word)
                continue
            for e in range(2, len(word)):
                if (word[:e] in nouns) and (self.coefficient.get(word[e:], 0.0) > minimum_noun_score):
                    removals.add(word)
                    break
        nouns_ = {word:score for word, score in nouns.items() if (word in removals) == False}
        return nouns_