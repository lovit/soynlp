from collections import defaultdict
from collections import namedtuple
import math
import numpy as np
import pickle
import sys
from soynlp.utils import get_process_memory

Scores = namedtuple('Scores', 'cohesion_forward cohesion_backward left_branching_entropy right_branching_entropy left_accessor_variety right_accessor_variety leftside_frequency rightside_frequency')

def _entropy(dic):
    if not dic: 
        return 0.0
    sum_ = sum(dic.values())
    entropy = 0
    for freq in dic.values():
        prob = freq / sum_
        entropy += prob * math.log(prob)
    return -1 * entropy


class WordExtractor:
    
    def __init__(self, left_max_length=10, right_max_length=6, 
                min_count=5, verbose_points=100000, 
                min_cohesion_forward=0.1, min_cohesion_backward=0.0, 
                max_droprate_cohesion=0.95, max_droprate_leftside_frequency=0.95,
                min_left_branching_entropy=0.0, min_right_branching_entropy=0.0,
                min_left_accessor_variety=0, min_right_accessor_variety=0,
                remove_subwords=True):
        self.left_max_length = left_max_length
        self.right_max_length = right_max_length
        self.min_count = min_count
        self.L = {}
        self.R = {}
        self._aL = {}
        self._aR = {}
        self.verbose = verbose_points

        self.min_cohesion_forward = min_cohesion_forward
        self.min_cohesion_backward = min_cohesion_backward
        self.max_droprate_cohesion = max_droprate_cohesion
        self.max_droprate_leftside_frequency = max_droprate_leftside_frequency
        self.min_left_branching_entropy = min_left_branching_entropy
        self.min_right_branching_entropy = min_right_branching_entropy
        self.min_left_accessor_variety = min_left_accessor_variety
        self.min_right_accessor_variety = min_right_accessor_variety
        self.remove_subwords = remove_subwords
        
    def train(self, sents, num_for_pruning = 0):
        def prune_extreme_case():
            self.L = defaultdict(lambda: 0, {w:f for w,f in self.L.items() if f >= self.min_count})
            self.R = defaultdict(lambda: 0, {w:f for w,f in self.R.items() if f >= self.min_count})
        def prune_extreme_case_a():
            self._aL = defaultdict(lambda: 0, {w:f for w,f in self._aL.items() if f > 1})
            self._aR = defaultdict(lambda: 0, {w:f for w,f in self._aR.items() if f > 1})
            
        self.L = defaultdict(lambda: 0)
        self.R = defaultdict(lambda: 0)
        self._aL = defaultdict(lambda: 0)
        self._aR = defaultdict(lambda: 0)
        
        for num_sent, sent in enumerate(sents):
            words = sent.strip().split()
            for word in words:
                if (not word) or (len(word) <= 1):
                    continue
                word_len = len(word)
                for i in range(1, min(self.left_max_length + 1, word_len)+1):
                    self.L[word[:i]] += 1
                for i in range(1, min(self.right_max_length + 1, word_len)):
                    self.R[word[-i:]] += 1

            if len(words) <= 1:
                continue
            for left_word, word, right_word in zip([words[-1]]+words[:-1], words, words[1:]+[words[0]]):
                self._aL['%s %s' % (word, right_word[0])] += 1
                self._aR['%s %s' % (left_word[-1], word)] += 1
                
                word_len = len(word)
                for i in range(1, min(self.right_max_length + 1, word_len)):
                    self._aL['%s %s' % (word[-i:], right_word[0])] += 1
                for i in range(1, min(self.left_max_length + 1, word_len)):
                    self._aR['%s %s' % (left_word[-1], word[:i])] += 1
                    
            if (num_for_pruning > 0) and ( num_sent % num_for_pruning == 0):
                prune_extreme_case()
            if (self.verbose > 0) and ( num_sent % self.verbose == 0):
                sys.stdout.write('\rtraining ... (%d in %d sents) use memory %.3f Gb' % (num_sent, len(sents), get_process_memory()))
                
        prune_extreme_case()
        prune_extreme_case_a()
        if (self.verbose > 0):
            print('\rtraining was done. used memory %.3f Gb' % (get_process_memory()))
        self.L = dict(self.L)
        self.R = dict(self.R)
        self._aL = dict(self._aL)
        self._aR = dict(self._aR)

    def extract(self, scores=None):
        if not scores:
            scores = self.word_scores()
        scores_ = {}
        for word, score in sorted(scores.items(), key=lambda x:len(x[0])):
            if len(word) <= 2:
                scores_[word] = score
                continue
            if (score.cohesion_forward < self.min_cohesion_forward) or \
                (score.cohesion_backward < self.min_cohesion_backward) or \
                (score.left_branching_entropy < self.min_left_branching_entropy) or \
                (score.right_branching_entropy < self.min_right_branching_entropy) or \
                (score.left_accessor_variety < self.min_left_accessor_variety) or \
                (score.right_accessor_variety < self.min_right_accessor_variety) or \
                (max(score.leftside_frequency, score.rightside_frequency) < self.min_count):
                continue
            scores_[word] = score
            if not self.remove_subwords:
                continue
            subword = word[:-1]
            droprate_cohesion_forward = 0 if not (subword in self.L) else score.cohesion_forward / self.cohesion_score(subword)[0]
            if (droprate_cohesion_forward > self.max_droprate_cohesion) and (subword in scores_):
                del scores_[subword]
            droprate_leftside_frequency = 0 if not (subword in self.L) else score.leftside_frequency / self.L[subword]
            if (droprate_leftside_frequency > self.max_droprate_leftside_frequency) and (subword in scores_):
                del scores_[subword]
        return scores_
    
    def word_scores(self):
        cps = self.all_cohesion_scores()
        bes = self.all_branching_entropy()
        avs = self.all_accessor_variety()
        scores = {}
        for word in self.words():
            cp = cps.get(word, (0, 0))
            be = bes.get(word, (0, 0))
            av = avs.get(word, (0, 0))
            scores[word] = Scores(cp[0], cp[1], be[0], be[1], av[0], av[1], self.L.get(word, 0), self.R.get(word, 0))
        return scores
    
    def all_cohesion_scores(self):
        cps = {}
        words = self.words()
        for i, word in enumerate(words):
            if (self.verbose > 0) and (i % self.verbose == 0):
                sys.stdout.write('\r cohesion probabilities ... (%d in %d)' % (i+1, len(words)))
            cp = self.cohesion_score(word)
            if (cp[0] == 0) and (cp[1] == 0):
                continue
            cps[word] = cp
        if (self.verbose > 0):
            print('\rall cohesion probabilities was computed. # words = %d' % len(cps))
        return cps

    def cohesion_score(self, word):
        word_len = len(word)        
        if (not word) or (word_len <= 1):
            return (0, 0)
        l_freq, r_freq = self.frequency(word)
        l_cohesion = 0 if l_freq == 0 else np.power( (l_freq / self.L[word[0]]), (1 / (word_len - 1)) )
        r_cohesion = 0 if r_freq == 0 else np.power( (r_freq / self.R[word[-1]]), (1 / (word_len - 1)) )        
        return (l_cohesion, r_cohesion)
    
    def frequency(self, word):
        return (self.L.get(word, 0), self.R.get(word, 0))
    
    def all_branching_entropy(self, get_score=_entropy):
        def parse_left(extension):
            return extension[:-1]
        def parse_right(extension):
            return extension[1:]
        def sort_by_length(counter):
            sorted_by_length = defaultdict(lambda: [])
            for w in counter.keys():
                sorted_by_length[len(w)].append(w)
            return sorted_by_length
        def get_entropy_table(parse, sorted_by_length, sorted_by_length_a, max_length, counter, counter_a):
            num_sum = sum((len(words) for length, words in sorted_by_length.items()))
            be = {}
            for word_len in range(2, max_length):
                words = sorted_by_length.get(word_len, [])
                extensions = defaultdict(lambda: [])
                for word in words:
                    extensions[parse(word)].append(word)
                words_ = sorted_by_length_a.get(word_len+1, [])
                for word in words_:
                    extensions[parse(word.replace(' ',''))].append(word)
                for root_word, extension_words in extensions.items():
                    extension_frequency = {ext:counter_a.get(ext) if ' ' in ext else counter.get(ext) for ext in extension_words}
                    be[root_word] = get_score(extension_frequency)
            return be
        def merge(be_l, be_r):
            be = {word:(v, be_r.get(word, 0)) for word, v in be_l.items()}
            for word, v in be_r.items():
                if word in be_l: continue
                be[word] = (0, v)
            return be

        be_l = get_entropy_table(parse_right, sort_by_length(self.R), sort_by_length(self._aR), self.right_max_length+1, self.R, self._aR)
        be_r = get_entropy_table(parse_left, sort_by_length(self.L), sort_by_length(self._aL), self.left_max_length+1, self.L, self._aL)
        be = merge(be_l, be_r)
        if self.verbose > 0:
            print_head = 'branching entropies' if get_score == _entropy else 'accessor variety'
            print('\rall %s was computed # words = %d' % (print_head, len(be)))
        return be

    def branching_entropy(self, word):
        word_len = len(word)
        lsb = { w:f for w,f in self.R.items() if ((len(w) - 1) == word_len) and (w[1:] == word) }
        lsb.update({ w:f for w,f in self._aR.items() if ((len(w) - 2) == word_len) and (w[2:] == word) })
        rsb = { w:f for w,f in self.L.items() if ((len(w) - 1) == word_len) and (w[:-1] == word) }
        rsb.update({ w:f for w,f in self._aL.items() if ((len(w) - 2) == word_len) and (w[:-2] == word) })
        be_l = 0 if not lsb else _entropy(lsb)
        be_r = 0 if not rsb else _entropy(rsb)
        return (be_l, be_r)

    def all_accessor_variety(self):
        return self.all_branching_entropy(get_score=len)

    def accessor_variety(self, word):
        word_len = len(word)
        lsb = { w:f for w,f in self.R.items() if ((len(w) - 1) == word_len) and (w[1:] == word) }
        lsb.update({ w:f for w,f in self._aR.items() if ((len(w) - 2) == word_len) and (w[2:] == word) })
        rsb = { w:f for w,f in self.L.items() if ((len(w) - 1) == word_len) and (w[:-1] == word) }
        rsb.update({ w:f for w,f in self._aL.items() if ((len(w) - 2) == word_len) and (w[:-2] == word) })
        av_l = 0 if lsb == False else len(lsb)
        av_r = 0 if rsb == False else len(rsb)
        return (av_l, av_r)

    def words(self):
        words = {word for word in self.L.keys() if len(word) <= self.left_max_length}
        words.update({word for word in self.R.keys() if len(word) <= self.right_max_length})
        return words

    def save(self, fname):
        configuration = {
            'left_max_length': self.left_max_length,
            'right_max_length': self.right_max_length,
            'min_count': self.min_count,
            'verbose_points': self.verbose,
            'min_cohesion_forward': self.min_cohesion_forward,
            'min_cohesion_backward': self.min_cohesion_backward,
            'max_droprate_cohesion': self.max_droprate_cohesion,
            'max_droprate_leftside_frequency': self.max_droprate_leftside_frequency,
            'min_left_branching_entropy': self.min_left_branching_entropy,
            'min_right_branching_entropy': self.min_right_branching_entropy,
            'min_left_accessor_variety': self.min_left_accessor_variety,
            'min_right_accessor_variety': self.min_right_accessor_variety,
            'remove_subwords': self.remove_subwords
        }
        data = {
            'L': self.L,
            'R': self.R,
            'aL': self._aL,
            'aR': self._aR
        }
        params = {
            'configuration': configuration,
            'data': data
            }
        with open(fname, 'wb') as f:
            pickle.dump(params, f)

    def load(self, fname):
        with open(fname, 'rb') as f:
            params = pickle.load(f)

        configuration = params['configuration']
        self.left_max_length = configuration['left_max_length']
        self.right_max_length = configuration['right_max_length']
        self.min_count = configuration['min_count']
        self.verbose = configuration['verbose_points']

        self.min_cohesion_forward = configuration['min_cohesion_forward']
        self.min_cohesion_backward = configuration['min_cohesion_backward']
        self.max_droprate_cohesion = configuration['max_droprate_cohesion']
        self.max_droprate_leftside_frequency = configuration['max_droprate_leftside_frequency']
        self.min_left_branching_entropy = configuration['min_left_branching_entropy']
        self.min_right_branching_entropy = configuration['min_right_branching_entropy']
        self.min_left_accessor_variety = configuration['min_left_accessor_variety']
        self.min_right_accessor_variety = configuration['min_right_accessor_variety']
        self.remove_subwords = configuration['remove_subwords']

        data = params['data']
        self.L = data['L']
        self.R = data['R']
        self._aL = data['aL']
        self._aR = data['aR']

        del params
        del configuration
        del data
