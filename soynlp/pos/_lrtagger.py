# -*- encoding:utf8 -*-

from collections import OrderedDict
from collections import namedtuple
from math import log

from soynlp.tokenizer import MaxScoreTokenizer
from ._dictionary import Dictionary

default_profile= OrderedDict([
        ('cohesion_l', 0.5),        
        ('droprate_l', 0.5),
        ('log_count_l', 0.1),
        ('prob_l2r', 0.1),
        ('log_count_l2r', 0.1),
        ('known_LR', 1.0),
        ('R_is_syllable', -0.1),
        ('log_length', 0.5)
])

ScoreTable = namedtuple('ScoreTable', list(default_profile))
Table = namedtuple('Table', 'L R begin end length, lr_prop lr_count cohesion_l droprate_l lcount')

class LREvaluator:
    def __init__(self, profile=None):
        self.profile = profile if profile else default_profile

    def evaluate(self, candidates, preference=None):
        scores = []
        for c in candidates:
            score = self._evaluate(
                              self.make_scoretable(c.L[0],
                              c.L[1],
                              c.R[0],
                              c.R[1],
                              c.cohesion_l,
                              c.droprate_l,
                              c.lcount,
                              c.lr_prop,
                              c.lr_count,
                              c.length
                         ))
            if preference:
                if c.L[1] and c.L[1] in preference:
                    score += preference.get(c.L[1], {}).get(c.L[0], 0)
                if c.R[1] and c.R[1] in preference:
                    score += preference.get(c.R[1], {}).get(c.R[0], 0)
            scores.append((c, score))
        return sorted(scores, key=lambda x:-x[-1])

    def make_scoretable(self, l, pos_l, r, pos_r, cohesion, droprate, lcount, lr_prop, lr_count, len_LR):
        return ScoreTable(cohesion,
                          droprate,
                          log(lcount+1),
                          lr_prop,
                          log(lr_count+1),
                          1 if (pos_l and pos_r) else 0,
                          1 if len(r) == 1 else 0,
                          log(len_LR)
                         )

    def _evaluate(self, scoretable):
        return sum(score * self.profile.get(field, 0) for field, score in scoretable._asdict().items())


class LRMaxScoreTagger:
    def __init__(self, domain_dictionary_folders=None, use_base_dictionary=True,
                 dictionary_word_mincount=3,
                 evaluator=None, sents=None, lrgraph=None, 
                 lrgraph_lmax=12, lrgraph_rmax=8,
                 base_tokenizer=None, preference=None, verbose=False
                ):
        
        self.dictionary = Dictionary(domain_dictionary_folders, use_base_dictionary, dictionary_word_mincount, verbose=verbose)
        self.evaluator = evaluator if evaluator else LREvaluator()
        self.preference = preference if preference else {}
        self.lrgraph = lrgraph if lrgraph else {}
        
        if (not self.lrgraph) and (sents):
            self.lrgraph = _build_lrgraph(sents, lrgraph_lmax, lrgraph_rmax)
            
        self.lrgraph_norm, self.lcount, self.cohesion_l, self.droprate_l\
            = self._initialize_scores(self.lrgraph)

        self.base_tokenizer = base_tokenizer if base_tokenizer else lambda x:x.split()
        if not base_tokenizer:
            try:
                self.base_tokenizer = MaxScoreTokenizer(scores=self.cohesion_l)
            except Exception as e:
                print('MaxScoreTokenizer(cohesion) exception: {}'.format(e))
        
    def _build_lrgraph(self, sents, lmax=12, rmax=8):
        from collections import Counter
        from collections import defaultdict
        eojeols = Counter((eojeol for sent in sents for eojeol in sent.split() if eojeol))
        lrgraph = defaultdict(lambda: defaultdict(int))
        for eojeol, count in eojeols.items():
            n = len(eojeol)
            for i in range(1, min(n, lmax)+1):
                (l, r) = (eojeol[:i], eojeol[i:])
                if len(r) > rmax:
                    continue
                lrgraph[l][r] += count
                
        return lrgraph
    
    def _initialize_scores(self, lrgraph):
        def to_counter(dd):
            return {k:sum(d.values()) for k,d in dd.items()}
        def to_normalized_graph(dd):
            normed = {}
            for k,d in dd.items():
                sum_ = sum(d.values())
                normed[k] = {k1:c/sum_ for k1,c in d.items()}
            return normed

        lrgraph_norm = to_normalized_graph(lrgraph)
        lcount = to_counter(lrgraph)
        cohesion_l = {w:pow(c/lcount[w[0]], 1/(len(w)-1)) for w, c in lcount.items() if len(w) > 1}
        droprate_l = {w:c/lcount[w[:-1]] for w, c in lcount.items() if len(w) > 1 and w[:-1] in lcount}
        
        return lrgraph_norm, lcount, cohesion_l, droprate_l
    
    def pos(self, sent, flatten=True, debug=False):
        sent_ = [self._pos(eojeol, debug) for eojeol in sent.split() if eojeol]
        if flatten:
            sent_ = [word for words in sent_ for word in words]
        return sent_

    def _pos(self, eojeol, debug=False):
        candidates = self._initialize(eojeol)
        scores = self._scoring(candidates)
        best = self._find_best(scores)
        if best:
            post = self._postprocessing(eojeol, best)
        else:
            post = self._base_tokenizing_subword(eojeol, 0)
            
        if not debug:
            post = [w for lr in post for w in lr[:2] if w[0]]
        return post
    
    def _initialize(self, t):
        candidates = self._initialize_L(t)
        candidates = self._initialize_LR(t, candidates)
        return candidates

    def _initialize_L(self, t):
        n = len(t)
        candidates = []
        for b in range(n):
            for e in range(b+2, min(n, b+self.dictionary._lmax)+1):
                l = t[b:e]
                l_pos = self.dictionary.pos_L(l)
                if not l_pos:
                    continue

                candidates.append([l,       # 0
                                   l_pos,       # 1
                                   b,      # 2                             
                                   e,      # 3
                                   e-b,   # 4
                                  ])

        candidates = self._remove_l_subsets(candidates)
        return sorted(candidates, key=lambda x:x[2])

    def _remove_l_subsets(self, candidates):
        candidates_ = []
        for pos in ['Noun', 'Verb', 'Adjective', 'Adverb', 'Exclamation']:
            # Sort by len_L
            sorted_ = sorted(filter(lambda x:x[1] == pos, candidates), key=lambda x:-x[4])
            while sorted_:
                candidates_.append(sorted_.pop(0))
                (b, e) = (candidates_[-1][2], candidates_[-1][3])
    #             removals = [i for i, c in enumerate(sorted_) if b < c[3] and e > c[2]] # Overlap
                removals = [i for i, c in enumerate(sorted_) if b <= c[2] and e >= c[3]] # Subset (Contain)
                for idx in reversed(removals):
                    del sorted_[idx]
        return candidates_

    def _initialize_LR(self, t, candidates, threshold_prop=0.001, threshold_count=2):
        n = len(t)
        expanded = []

        for (l, pos, b, e, len_l) in candidates:        
            for len_r in range(min(self.dictionary._rmax, n-e)+1):

                r = t[e:e+len_r]
                lr_prop = self.lrgraph_norm.get(l, {}).get(r, 0)
                lr_count = self.lrgraph.get(l, {}).get(r, 0)

                if (r) and ((lr_prop <= threshold_prop) or (lr_count <= threshold_count)):
                    continue

                expanded.append([(l, pos),
                                 (r, None if not r else self.dictionary.pos_R(r)),
                                 b,
                                 e,
                                 e + len_r,
                                 len_r,
                                 len_l + len_r,
                                 lr_prop,
                                 lr_count
                                ])

        expanded = self._remove_r_subsets(expanded)
        return sorted(expanded, key=lambda x:x[2])

    def _remove_r_subsets(self, expanded):
        expanded_ = []
        for pos in ['Josa', 'Verb', 'Adjective', None]:
            # Sory by len_R
            sorted_ = sorted(filter(lambda x:x[1][1] == pos, expanded), key=lambda x:-x[5])
            while sorted_:
                expanded_.append(sorted_.pop(0))
                (b, e) = (expanded_[-1][3], expanded_[-1][4])
    #             removals = [i for i, c in enumerate(sorted_) if b < c[3] and e > c[2]] # Overlap
                removals = [i for i, c in enumerate(sorted_) if b <= c[3] and e >= c[4]] # Subset (Contain)
                for idx in reversed(removals):
                    del sorted_[idx]
        expanded_ = [[L, R, p0, p2, len_LR, prop, count] for L, R, p0, p1, p2, len_R, len_LR, prop, count in expanded_]
        return expanded_
    
    def _scoring(self, candidates):
        candidates = [self._to_table(c) for c in candidates]
        scores = self.evaluator.evaluate(candidates, self.preference if self.preference else None)
        return scores

    def _to_table(self, c):
        return Table(c[0], c[1], c[2], c[3], c[4], c[5], c[6], 
                     self.cohesion_l.get(c[0][0], 0),
                     self.droprate_l.get(c[0][0], 0),
                     self.lcount.get(c[0][0], 0)
                    )
    
    def _find_best(self, scores):
        best = []
        sorted_ = sorted(scores, key=lambda x:-x[-1])
        while sorted_:
            best.append(sorted_.pop(0)[0])
            (b, e) = (best[-1][2], best[-1][3])
            removals = [i for i, (c, _) in enumerate(sorted_) if b < c[3] and e > c[2]] # Overlap
            for idx in reversed(removals):
                del sorted_[idx]
        return sorted(best, key=lambda x:x[2])
    
    def _postprocessing(self, t, words):
        n = len(t)
        adds = []
        if words and words[0][2] > 0:
            adds += self._add_first_subword(t, words)
        if words and words[-1][3] < n:
            adds += self._add_last_subword(t, words, n)
        adds += self._add_inter_subwords(t, words)
        post = [w for w in words] + [self._to_table(a) for a in adds]
        return sorted(post, key=lambda x:x[2])

    def _infer_subword_information(self, subword):
        pos = self.dictionary.pos_L(subword)
        prop = self.lrgraph_norm.get(subword, {}).get('', 0.0)
        count = self.lrgraph.get(subword, {}).get('', 0)    
        if not pos:
            pos = self.dictionary.pos_R(subword)
        return (pos, prop, count)

    def _add_inter_subwords(self, t, words):
        adds = []        
        for i, base in enumerate(words[:-1]):
            if base[3] == words[i+1][2]:
                continue

            b = base[3]
            e = words[i+1][2]
            subword = t[b:e]
            #(pos, prop, count) = self._infer_subword_information(subword)
            #adds.append([(subword, pos), ('', None), b, e, e-b, prop, count, 0.0])
            adds += self._base_tokenizing_subword(subword, b)
        return adds

    def _add_last_subword(self, t, words, n):
        b = words[-1][3]
        subword = t[b:]
        #(pos, prop, count) = self._infer_subword_information(subword)
        #return [[(subword, pos), ('', None), b, n, n-b, prop, count, 0.0]]
        return self._base_tokenizing_subword(subword, b)

    def _add_first_subword(self, t, words):    
        e = words[0][2]
        subword = t[0:e]
        #(pos, prop, count) = self._infer_subword_information(subword)
        #return [[(subword, pos), ('', None), 0, e, e, prop, count, 0.0]]
        return self._base_tokenizing_subword(subword, 0)
    
    def _base_tokenizing_subword(self, t, b):
        subwords = []
        _subwords = self.base_tokenizer.tokenize(t, flatten=False)
        if not _subwords:
            return []
        for w in _subwords[0]:
            (pos, prop, count) = self._infer_subword_information(w[0])
            subwords.append([(w[0], pos), ('', None), b+w[1], b+w[2], w[2]-w[1], prop, count, 0.0])
        return subwords
    
    def add_words_into_dictionary(self, words, tag):
        if not (tag in self.dictionary._pos):
            raise ValueError('{} does not exist base dictionary'.format(tag))
        self.dictionary.add_words(words, tag)
        
    def remove_words_from_dictionary(self, words, tag):
        if not (tag in self.dictionary._pos):
            raise ValueError('{} does not exist base dictionary'.format(tag))
        self.dictionary.remove_words(words, tag)
    
    def save_domain_dictionary(self, folder, head=None):
        self.dictionary.save_domain_dictionary(folder, head)
    
    def set_word_preferance(self, words, tag, preference=10):
        if type(words) == str:
            words = {words}
        preference_table = self.preference.get(tag, {})
        preference_table.update({word:preference for word in words})
        self.preference[tag] = preference_table
    
    def save_tagger(self, fname):
        raise NotImplemented