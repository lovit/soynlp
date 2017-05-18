from collections import defaultdict
from collections import namedtuple
import math
import numpy as np
import sys
from soynlp.utils import get_process_memory

Scores = namedtuple('Scores', 'cohesion_forward cohesion_backward  left_branching_entropy right_branching_entropy left_accessor_variety right_accessor_variety leftside_frequency rightside_frequency')

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
    
    def __init__(self, left_max_length=10, right_max_length=6, min_count=5, verbose_points=100000):
        self.left_max_length = left_max_length
        self.right_max_length = right_max_length
        self.min_count = min_count
        self.L = {}
        self.R = {}
        self.verbose = verbose_points
        
    def train(self, sents, num_for_pruning = 0):
        def prune_extreme_case():
            self.L = defaultdict(lambda: 0, {w:f for w,f in self.L.items() if f >= self.min_count})
            self.R = defaultdict(lambda: 0, {w:f for w,f in self.R.items() if f >= self.min_count})
            
        self.L = defaultdict(lambda: 0)
        self.R = defaultdict(lambda: 0)
        
        for num_sent, sent in enumerate(sents):            
            for word in sent.split():
                
                if (not word) or (len(word) <= 1):
                    continue
                    
                word_len = len(word)
                for i in range(1, min(self.left_max_length + 1, word_len)+1):                    
                    self.L[word[:i]] += 1                
                for i in range(1, min(self.right_max_length + 1, word_len)):
                    self.R[word[-i:]] += 1

            if (num_for_pruning > 0) and ( (num_sent + 1) % num_for_pruning == 0):
                prune_extreme_case()
            if (self.verbose > 0) and (num_sent % self.verbose == 0):
                sys.stdout.write('\rtraining ... (%d in %d sents) use memory %.3f Gb' % (num_sent + 1, len(sents), get_process_memory()))
                
        prune_extreme_case()
        if (self.verbose > 0):
            print('\rtraining was done. used memory %.3f Gb' % (get_process_memory()))
        self.L = dict(self.L)
        self.R = dict(self.R)

    def extract(self, min_cohesion_forward=0.0, min_cohesion_backward=0.0, 
                min_droprate_cohesion_forward=0.0, min_droprate_cohesion_backward=0.0,
                min_left_branching_entropy=0.0, min_right_branching_entropy=0.0,
                min_left_accessor_variety=0, min_right_accessor_variety=0,
                min_count=5, scores=None):
        if not scores:
            scores = self.word_scores()
        scores_ = {}
        for word, score in scores.items():
            if len(word) <= 2:
                scores_[word] = score
                continue
            droprate_cohesion_forward = 0 if not (word[:-1] in self.L) else score.cohesion_forward / self.cohesion_score(word[:-1])[0]
            droprate_cohesion_backward = 0 if not (word[1:] in self.R) else score.cohesion_backward / self.cohesion_score(word[1:])[1]
            if (score.cohesion_forward < min_cohesion_forward) or \
                (score.cohesion_backward < min_cohesion_backward) or \
                (score.left_branching_entropy < min_left_branching_entropy) or \
                (score.right_branching_entropy < min_right_branching_entropy) or \
                (score.left_accessor_variety < min_left_accessor_variety) or \
                (score.right_accessor_variety < min_right_accessor_variety) or \
                (droprate_cohesion_forward < min_droprate_cohesion_forward) or \
                (droprate_cohesion_backward < min_droprate_cohesion_backward) or \
                (max(score.leftside_frequency, score.rightside_frequency) < min_count):
                continue
            scores_[word] = score
        return scores_
    
    def word_scores(self):
        cps = self.all_cohesion_scores()
        bes = self.all_branching_entropy()
        avs = self.all_accessor_variety()
        scores = {}
        for word, cp in cps.items():
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
        def get_entropy_table(parse, sorted_by_length, max_length, print_head, counter):
            num_sum = sum((len(words) for length, words in sorted_by_length.items()))
            num = 0
            be = {}
            for word_len in range(2, max_length):
                words = sorted_by_length.get(word_len, [])
                extensions = defaultdict(lambda: [])
                for word in words:
                    extensions[parse(word)].append(word)
                    num += 1
                    if (self.verbose > 0) and (num % self.verbose == 0):
                        args = (print_head, word_len - 1, num, num_sum)
                        sys.stdout.write('\r%s ... (len = %d, %d in %d)' % args)
                for root_word, extension_words in extensions.items():
                    extension_frequency = {ext:counter.get(ext) for ext in extension_words}
                    be[root_word] = get_score(extension_frequency)
            return be
        def merge(be_l, be_r):
            be = {word:(v, be_r.get(word, 0)) for word, v in be_l.items()}
            for word, v in be_r.items():
                if word in be_l: continue
                be[word] = (0, v)
            return be

        be_l = get_entropy_table(parse_right, sort_by_length(self.R), self.right_max_length+1, 'right to left branching entropy', self.R)
        be_r = get_entropy_table(parse_left, sort_by_length(self.L), self.left_max_length+1, 'left to right branching entropy', self.L)
        be = merge(be_l, be_r)
        if self.verbose > 0:
            print_head = 'branching entropies' if get_score == _entropy else 'accessor variety'
            print('\rall %s was computed # words = %d' % (print_head, len(be)))
        return be

    def branching_entropy(self, word):
        # TODO: check direction of entropy
        word_len = len(word)
        lsb = { w:f for w,f in self.R.items() if ((len(w) - 1) == word_len) and (w[1:] == word) }
        rsb = { w:f for w,f in self.L.items() if ((len(w) - 1) == word_len) and (w[:-1] == word) }
        be_l = 0 if (word in self.R) == False else _entropy(lsb)
        be_r = 0 if (word in self.L) == False else _entropy(rsb)
        return (be_l, be_r)

    def all_accessor_variety(self):
        return self.all_branching_entropy(get_score=len)

    def accessor_variety(self, word):
        word_len = len(word)
        lsb = { w:f for w,f in self.R.items() if ((len(w) - 1) == word_len) and (w[1:] == word) }
        rsb = { w:f for w,f in self.L.items() if ((len(w) - 1) == word_len) and (w[:-1] == word) }
        print(lsb, rsb)
        av_l = 0 if (word in self.R) == False else len(lsb)
        av_r = 0 if (word in self.L) == False else len(rsb)
        return (av_l, av_r)

    def words(self):
        words = {word for word in self.L.keys() if len(word) <= self.left_max_length}
        words.update({word for word in self.R.keys() if len(word) <= self.right_max_length})
        return words