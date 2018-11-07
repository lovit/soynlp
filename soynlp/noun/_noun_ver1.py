# -*- encoding:utf8 -*-

from collections import defaultdict, namedtuple
import math
import sys
from soynlp.normalizer import normalize_sent_for_lrgraph
from soynlp.word import WordExtractor
from soynlp.utils import LRGraph

NounScore_v1 = namedtuple('NounScore_v1', 'frequency score known_r_ratio')

class LRNounExtractor:

    def __init__(self, max_left_length=10, max_right_length=7,
            predictor_fnames=None, verbose=True,
            min_num_of_features=1, ensure_normalized=False):

        self.coefficient = {}
        self.verbose = verbose
        self.max_left_length = max_left_length
        self.max_right_length = max_right_length
        self.lrgraph = None
        self.words = None
        self._substring_counter = {}
        self.min_num_of_features = min_num_of_features
        self.ensure_normalized = ensure_normalized

        if not predictor_fnames:
            import os
            directory = '/'.join(os.path.abspath(__file__).replace('\\', '/').split('/')[:-2])
            predictor_fnames = ['%s/trained_models/noun_predictor_sejong' % directory]
            if verbose:
                print('[Noun Extractor] used default noun predictor; Sejong corpus predictor')

        for fname in predictor_fnames:
            if verbose:
                print('[Noun Extractor] used %s' % fname.split('/')[-1])
            self._load_predictor(fname)
        if verbose:
            print('[Noun Extractor] All %d r features was loaded' % len(self.coefficient))

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
                print('[Noun Extractor] predictor parsing error line {} = {}'.format(num_line+1, line))
            finally:
                f.close()
        except Exception as e:
            print(e)

    def train_extract(self, sents, min_noun_score=0.5, min_noun_frequency=5,
            noun_candidates=None):

        self.train(sents, min_noun_frequency)
        return self.extract(min_noun_score, min_noun_frequency, noun_candidates)

    def train(self, sents, min_noun_frequency=5):
        wordset_l, wordset_r = self._scan_vocabulary(sents, min_noun_frequency)
        lrgraph = self._build_lrgraph(sents, wordset_l, wordset_r)
        self.lrgraph = LRGraph(lrgraph)
        self.words = wordset_l

    def _scan_vocabulary(self, sents, min_frequency=5):
        """
        Parameters
        ----------
            sents: list-like iterable object which has string
            
        It computes subtoken frequency first. 
        After then, it builds lr-graph with sub-tokens appeared at least min count
        """

        wordset_l = defaultdict(lambda: 0)
        wordset_r = defaultdict(lambda: 0)

        for i, sent in enumerate(sents):
            if not self.ensure_normalized:
                sent = normalize_sent_for_lrgraph(sent)
            for token in sent.split(' '):
                if not token:
                    continue
                token_len = len(token)
                for i in range(1, min(self.max_left_length, token_len)+1):
                    wordset_l[token[:i]] += 1
                for i in range(1, min(self.max_right_length, token_len)):
                    wordset_r[token[-i:]] += 1
            if self.verbose and (i % 1000 == 999):
                message = 'scanning {} / {} sents'.format(i+1, len(sents))
                print('\r[Noun Extractor] {}'.format(message), end='')

        self._substring_counter = {w:f for w,f in wordset_l.items() if f >= min_frequency}
        wordset_l = set(self._substring_counter.keys())
        wordset_r = {w for w,f in wordset_r.items() if f >= min_frequency}

        if self.verbose:
            message = '(L,R) has (%d, %d) tokens' % (len(wordset_l), len(wordset_r))
            print('\r[Noun Extractor] scanning was done {}'.format(message))

        return wordset_l, wordset_r

    def _build_lrgraph(self, sents, wordset_l, wordset_r):
        lrgraph = defaultdict(lambda: defaultdict(lambda: 0))

        for i, sent in enumerate(sents):
            if not self.ensure_normalized:
                sent = normalize_sent_for_lrgraph(sent)
            for token in sent.split():
                if not token:
                    continue
                n = len(token)
                for i in range(1, min(self.max_left_length, n)+1):
                    l = token[:i]
                    r = token[i:]
                    if not (l in wordset_l):
                        continue
                    if (len(r) > 0) and not (r in wordset_r):
                        continue
                    lrgraph[l][r] += 1

            if self.verbose and (i % 1000 == 999):
                message = 'building L-R graph from {} / {} sents'.format(i+1, len(sents))
                print('\r[Noun Extractor] {}'.format(message), end='')

        if self.verbose:
            print('\r[Noun Extractor] building L-R graph was done'.format(' '*20))
        lrgraph = {l:{r:f for r,f in rdict.items()} for l,rdict in lrgraph.items()}
        return lrgraph

    def extract(self, min_noun_score=0.5, min_noun_frequency=5, noun_candidates=None):
        if not noun_candidates:
            noun_candidates = self.words

        # prediction
        nouns = {}
        for word in sorted(noun_candidates, key=lambda w:len(w)):
            if len(word) <= 1:
                continue

            score = self.predict(word, nouns)

            if score[0] < min_noun_score:
                continue
            nouns[word] = score

        # postprocessing
        nouns = self._postprocess(nouns, min_noun_score, min_noun_frequency)

        # summary information as NounScore
        nouns_ = self._to_NounScore(nouns)

        if self.verbose:
            print('[Noun Extractor] {} nouns are extracted'.format(len(nouns_)))

        return nouns_

    def _get_r_features(self, word):
        features = self.lrgraph.get_r(word, -1)
        # remove empty str r only in features
        features = [feature for feature in features if feature[0]]
        return features

    def _get_subword_score(self, word, min_noun_score, nouns):
        subword_scores = {}
        for e in range(1, len(word)):
            subword = word[:e]
            suffix = word[e:]
            # Add word if compound
            if (subword in nouns) and (suffix in nouns):
                score1 = nouns[subword]
                score2 = nouns[suffix]
                subword_scores[subword] = max(score1, score2)
            elif (subword in nouns) and (self.coefficient.get(suffix,0.0) > min_noun_score):
                subword_scores[subword] = (self.coefficient.get(suffix,0.0), 0)
        if not subword_scores:
            return (0.0, 0)
        return sorted(subword_scores.items(), key=lambda x:-x[1][0])[0][1]

    def is_noun(self, word, min_noun_score=0.5):
        return self.predict(word)[0] >= min_noun_score

    def predict(self, word, min_noun_score=0.5, nouns=None):
        """Returns (noun_score, known_r_ratio)
        """
        features = self._get_r_features(word)

        # (감사합니다 + 만) 처럼 뒤에 등장하는 R 의 종류가 한가지 뿐이면 제대로 된 판단이 되지 않음
        if len(features) > self.min_num_of_features:
            score = self._predict(features, word)
        else:
            if nouns is None:
                nouns = {}
            score = self._get_subword_score(word, min_noun_score, nouns)

        return score

    def _predict(self, features, word):

        def exist_longer_r_feature(word, r):
            for e in range(len(word)-1, -1, -1):
                suffix = word[e:] + r
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

        for r, freq in features:
            if r in self.coefficient:
                if not exist_longer_r_feature(word, r):
                    score += freq * self.coefficient[r]
                    norm += freq
            else:
                unknown += freq

        return (0 if norm == 0 else score / norm,
                0 if (norm + unknown == 0) else norm / (norm + unknown))

    def _postprocess(self, nouns, min_noun_score, min_noun_frequency):

        def is_Noun_Josa(l, r):
            return (l in nouns) and (self.coefficient.get(r, 0.0) > min_noun_score)

        def cohesion(word):
            base = self._substring_counter.get(word[0], 0)
            n = len(word)
            if not base or n <= 1:
                return 0
            return math.pow(self._substring_counter.get(word, 0) / base, 1/(n-1))

        def longer_has_larger_cohesion(word):
            return cohesion(word) >= cohesion(word[:-1])

        removals = set()
        for word in nouns:
            if word[-1] == '.' or word[-1] == ',':
                removals.add(word)
                continue
            n = len(word)
            if n <= 2 or longer_has_larger_cohesion(word):
                continue
            for e in range(2, len(word)):
                l = word[:e]
                r = word[e:]
                if is_Noun_Josa(l, r):
                    removals.add(word)
                    break
        nouns_ = {word:score for word, score in nouns.items() if (word in removals) == False}
        return nouns_

    def _to_NounScore(self, nouns):
        noun_frequencies = {}
        for word in sorted(nouns, key=lambda x:-len(x)):
            r_count = self.lrgraph.get_r(word, -1)
            noun_frequencies[word] = sum(c for w, c in r_count)
            for r, count in r_count:
                self.lrgraph.remove_eojeol(word+r, count)
        self.lrgraph.reset_lrgraph()

        nouns_ = {}
        for word, score in nouns.items():
            nouns_[word] = NounScore_v1(noun_frequencies[word], score[0], score[1])

        return nouns_
