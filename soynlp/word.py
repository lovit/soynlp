from collections import defaultdict
import math
import numpy as np
import sys
from soynlp.utils import get_process_memory


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
    
    def get_word_scores(self):
        cps = self.get_all_cohesion_probabilities()
        bes = self.get_all_branching_entropy()
        cpbe = {}
        for word, cp in cps.items():
            be = bes.get(word, (0, 0))
            cpbe[word] = (cp[0], cp[1], be[0], be[1], self.L.get(word, 0), self.R.get(word, 0))
        return cpbe
    
    def get_all_cohesion_probabilities(self):
        cps = {}
        words = self.words()
        for i, word in enumerate(words):
            if (self.verbose > 0) and (i % self.verbose == 0):
                sys.stdout.write('\r cohesion probabilities ... (%d in %d)' % (i+1, len(words)))
            cp = self.get_cohesion_probability(word)
            if (cp[0] == 0) and (cp[1] == 0):
                continue
            cps[word] = cp
        if (self.verbose > 0):
            print('\rall cohesion probabilities was computed. # words = %d' % len(cps))
        return cps

    def get_cohesion_probability(self, word):
        word_len = len(word)        
        if (not word) or (word_len <= 1):
            return (0, 0)
        l_freq, r_freq = self.get_frequency(word)
        l_cohesion = 0 if l_freq == 0 else np.power( (l_freq / self.L[word[0]]), (1 / (word_len - 1)) )
        r_cohesion = 0 if r_freq == 0 else np.power( (r_freq / self.R[word[-1]]), (1 / (word_len - 1)) )        
        return (l_cohesion, r_cohesion)
    
    def get_frequency(self, word):
        return (self.L.get(word, 0), self.R.get(word, 0))
    
    def get_all_branching_entropy(self):
        # left -> right extension
        be_l = {}
        sorted_by_length = defaultdict(lambda: [])
        for l in self.L.keys():
            sorted_by_length[len(l)].append(l)
        num_sum = sum((len(words) for length, words in sorted_by_length.items() if length <= self.left_max_length))
        num = 0
        for word_len in range(2, self.left_max_length+1):
            words = sorted_by_length.get(word_len, [])
            if not words:
                continue
            extensions = defaultdict(lambda: [])
            for word in words:
                extensions[word[:-1]].append(word)
                num += 1
                if (self.verbose > 0) and (num % self.verbose == 0):
                    sys.stdout.write('\rleft to right branching entropy ... (len = %d, %d in %d)' % (word_len - 1, num, num_sum))
            for from_word, extension_words in extensions.items():
                be_l[from_word] = self._entropy({ext:self.L[ext] for ext in extension_words})
        
        # left <- right extension
        be_r = {}
        sorted_by_length = defaultdict(lambda: [])
        for r in self.R.keys():
            sorted_by_length[len(r)].append(r)
        num_sum = sum((len(words) for length, words in sorted_by_length.items() if length <= self.right_max_length))
        num = 0
        for word_len in range(2, self.right_max_length+1):
            words = sorted_by_length.get(word_len, [])
            if not words:
                continue
            extensions = defaultdict(lambda: [])
            for word in words:
                extensions[word[1:]].append(word)
                num += 1
                if (self.verbose > 0) and (num % self.verbose == 0):
                    sys.stdout.write('\rright to left branching entropy ... (len = %d, %d in %d)' % (word_len - 1, num, num_sum))
            for from_word, extension_words in extensions.items():
                be_r[from_word] = self._entropy({ext:self.R[ext] for ext in extension_words})
        
        # merging be_l, be_r
        be = {word:(v, be_r.get(word, 0)) for word, v in be_l.items()}
        for word, v in be_r.items():
            if word in be_l: continue
            be[word] = (0, v)
        if self.verbose > 0:
            print('\rall branching entropies was computed # words = %d' % len(be))
        return be
            
    def get_branching_entropy(self, word):
        # TODO: check direction of entropy
        word_len = len(word)
        be_l = 0 if (word in self.L) == False else self._entropy({ w:f for w,f in self.L.items() if (len(w) - 1 == word_len) and (w[:-1] == word) })
        be_r = 0 if (word in self.R) == False else self._entropy({ w:f for w,f in self.R.items() if (len(w) - 1 == word_len) and (w[1:] == word) })
        return (be_l, be_r)
        
    def _entropy(self, dic):
        if not dic: 
            return 0.0
        sum_ = sum(dic.values())
        entropy = 0
        for freq in dic.values():
            prob = freq / sum_
            entropy += prob * math.log(prob)
        return -1 * entropy

    def get_all_accessor_variety(self):
        raise NotImplemented

    def get_accessor_variety(self, word):
        word_len = len(word)
        av_l = 0 if (word in self.L) == False else len({ w:f for w,f in self.L.items() if (len(w) - 1 == word_len) and (w[:-1] == word) })
        av_r = 0 if (word in self.R) == False else len({ w:f for w,f in self.R.items() if (len(w) - 1 == word_len) and (w[1:] == word) })
        return (av_l, av_r)

    def words(self):
        words = {word for word in self.L.keys() if len(word) <= self.left_max_length}
        words.update({word for word in self.R.keys() if len(word) <= self.right_max_length})
        return words